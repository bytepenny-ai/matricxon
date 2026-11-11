"""TimesFM 2.5 (Google, 200M): a decoder-only patch transformer that forecasts time series. Numbers
in, numbers out: no tokenizer, no vocabulary, no KV cache. `ForecastEngine` does the global
normalization and the output post-processing around `predict`."""

import logging
from collections.abc import Callable

import torch
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.rope import RotaryEmbedding
from app.architectures.timesfm25_layers import PatchMask, ResidualBlock, TimesFmDecoderLayer
from app.architectures.timesfm25_revin import RevIN, RunningPatchStats
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class TimesFm25Architecture(ModelArchitecture):
    """The `timesfm25` GGUF layout (zsfm's converter; the only published TimesFM 2.5 GGUF).
    Always computes in float32: 200M parameters fit, and forecasts are precision-sensitive."""

    NAME = "timesfm25"
    # A forecaster: chat, generate, embed and rerank refuse it (ModelKind).
    IS_FORECASTER = True

    def __init__(self, metadata: GGUFMetadata) -> None:
        super().__init__()
        arch = metadata.arch_key
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.n_embd = metadata.get_u32(arch("embedding_length"))
        ffn_dim = metadata.get_u32(arch("feed_forward_length"))
        n_head = metadata.get_u32(arch("attention.head_count"))
        self.head_dim = head_dim = metadata.get_u32(arch("attention.head_dim"))
        self.rope_theta = metadata.get_f32(arch("rope_theta"))
        eps = metadata.get_f32(arch("rms_norm_epsilon"))
        self.patch_len = metadata.get_u32(arch("input_patch_len"))
        self.horizon_len = metadata.get_u32(arch("output_patch_len"))
        self.decode_index = metadata.get_u32(arch("decode_index"))
        self.quantiles = [float(q) for q in metadata.require(arch("quantiles"))]
        self.context_limit = metadata.get_u32(arch("context_limit"))
        self.n_outputs = len(self.quantiles) + 1

        self.tokenizer = ResidualBlock(2 * self.patch_len, self.n_embd, self.n_embd, bias=True)
        self.layers = nn.ModuleList(
            [
                TimesFmDecoderLayer(self.n_embd, ffn_dim, n_head, head_dim, eps)
                for _ in range(self.n_layer)
            ]
        )
        self.rope = RotaryEmbedding(head_dim, self.rope_theta)
        self.out_point = ResidualBlock(
            self.n_embd, self.n_embd, self.horizon_len * self.n_outputs, bias=False
        )
        # The quantile head's length: its output rows over the number of outputs (1024 for 2.5).
        self.quantile_len = metadata.get_u32(arch("output_quantile_len"), 1024)
        self.out_quantile = ResidualBlock(
            self.n_embd, self.n_embd, self.quantile_len * self.n_outputs, bias=False
        )

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls, loader: GGUFModelLoader, dtype: torch.dtype = torch.float32, **_: object
    ) -> "TimesFm25Architecture":
        del dtype  # float32 always (class docstring)
        model = cls._construct_without_init(loader.metadata)
        model._rebuild_derived_buffers()
        model._defer_materialization(loader)
        return model

    def _rebuild_derived_buffers(self) -> None:
        self.rope = RotaryEmbedding(self.head_dim, self.rope_theta)

    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        self._copy_block(loader, self.tokenizer, "tokenizer", bias=True)
        self._copy_block(loader, self.out_point, "out_point", bias=False)
        self._copy_block(loader, self.out_quantile, "out_quantile", bias=False)
        for i, layer in enumerate(self.layers):
            names = {
                "pre_attn_norm.weight": layer.pre_attn_norm.weight,
                "post_attn_norm.weight": layer.post_attn_norm.weight,
                "pre_ff_norm.weight": layer.pre_ff_norm.weight,
                "post_ff_norm.weight": layer.post_ff_norm.weight,
                "attn_qkv.weight": layer.attn.qkv.weight,
                "attn_out.weight": layer.attn.out.weight,
                "attn_q_norm.weight": layer.attn.q_norm.weight,
                "attn_k_norm.weight": layer.attn.k_norm.weight,
                "attn_q_scale.weight": layer.attn.q_scale,
                "ffn_up.weight": layer.ffn_up.weight,
                "ffn_down.weight": layer.ffn_down.weight,
            }
            for name, target in names.items():
                target.copy_(loader.load_tensor(f"blk.{i}.{name}"))
            if stop_check is not None and stop_check():
                raise GenerationCancelledError(f"stopped loading weights at layer {i + 1}")

    @staticmethod
    def _copy_block(loader: GGUFModelLoader, block: ResidualBlock, prefix: str, bias: bool) -> None:
        for part in ("hidden", "output", "skip"):
            linear: nn.Linear = getattr(block, part)
            linear.weight.copy_(loader.load_tensor(f"{prefix}.{part}.weight"))
            if bias:
                linear.bias.copy_(loader.load_tensor(f"{prefix}.{part}.bias"))

    def predict(
        self, series: torch.Tensor, padded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One decode step. `series` (B, L) already globally normalized, `padded` (B, L) bool,
        L a multiple of `patch_len` -> point (B, horizon_len, n_outputs) and quantile
        (B, quantile_len, n_outputs) from the last patch, in `series`' scale."""
        self._ensure_materialized()
        batch = series.shape[0]
        patches = series.float().view(batch, -1, self.patch_len)
        patch_pad = padded.view(batch, -1, self.patch_len)
        mu, sigma = RunningPatchStats.compute(patches, patch_pad)
        normed = RevIN.normalize(patches, mu, sigma).masked_fill(patch_pad, 0.0)
        x = self.tokenizer(torch.cat([normed, patch_pad.float()], dim=-1))

        last_padded = patch_pad[..., -1]
        positions = torch.arange(x.shape[1]) - last_padded.sum(dim=-1, keepdim=True)
        cos, sin = self.rope(positions)
        mask = PatchMask.build(last_padded)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)

        last = x[:, -1]
        mu_last, sigma_last = mu[:, -1], sigma[:, -1]
        point = RevIN.denormalize(self.out_point(last), mu_last, sigma_last)
        quantile = RevIN.denormalize(self.out_quantile(last), mu_last, sigma_last)
        return (
            point.view(batch, self.horizon_len, self.n_outputs),
            quantile.view(batch, self.quantile_len, self.n_outputs),
        )

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_ids: torch.Tensor | None = None,
        stop_check: Callable[[], bool] | None = None,
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        raise TypeError("TimesFM forecasts numbers; call predict(), not forward().")
