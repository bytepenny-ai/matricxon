import torch
import torch.nn.functional as F
from torch import nn

from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata

_ROPE_THETA = 100.0
_POOL = 3  # 3x3 patches -> one soft token (HF `pooling_kernel_size`)


def _rms(x: torch.Tensor, eps: float, weight: torch.Tensor | None = None) -> torch.Tensor:
    x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return x if weight is None else x * weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class ClampedLinear(nn.Module):
    """A bias-free linear whose input and output are clamped to ranges stored with the weight
    (`.input_min`/`.input_max`/`.output_min`/`.output_max`; HF `Gemma4ClippableLinear`)."""

    def __init__(self, n_in: int, n_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(n_out, n_in))
        self.register_buffer("bounds", torch.tensor([-torch.inf, torch.inf, -torch.inf, torch.inf]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo_in, hi_in, lo_out, hi_out = self.bounds.tolist()
        return F.linear(x.clamp(lo_in, hi_in), self.weight).clamp(lo_out, hi_out)


class Gemma4VisionLayer(nn.Module):
    """Sandwich-norm ViT block: RMSNorm -> attention (q/k RMSNorm, unscaled v RMSNorm, 2D rotary,
    no 1/sqrt(d) since q/k are normed) -> RMSNorm, residual; RMSNorm -> gated tanh-GELU MLP ->
    RMSNorm, residual. Every linear is a ClampedLinear."""

    def __init__(self, n_embd: int, n_head: int, ffn_len: int, eps: float) -> None:
        super().__init__()
        self.n_head, self.eps = n_head, eps
        head_dim = n_embd // n_head
        self.q, self.k, self.v, self.o = (ClampedLinear(n_embd, n_embd) for _ in range(4))
        self.gate, self.up = ClampedLinear(n_embd, ffn_len), ClampedLinear(n_embd, ffn_len)
        self.down = ClampedLinear(ffn_len, n_embd)
        self.q_norm = nn.Parameter(torch.ones(head_dim))
        self.k_norm = nn.Parameter(torch.ones(head_dim))
        self.norms = nn.Parameter(torch.ones(4, n_embd))  # input, post-attn, pre-ffn, post-ffn

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], self.n_head, -1)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        h = _rms(x, self.eps, self.norms[0])
        q = _rms(self._heads(self.q(h)), self.eps, self.q_norm)
        k = _rms(self._heads(self.k(h)), self.eps, self.k_norm)
        v = _rms(self._heads(self.v(h)), self.eps)
        q, k = (Gemma4VisionEncoder.rotate(t, cos, sin) for t in (q, k))
        q, k, v = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        attn = F.scaled_dot_product_attention(q, k, v, scale=1.0)[0].transpose(0, 1)
        x = x + _rms(self.o(attn.reshape(x.shape[0], -1)), self.eps, self.norms[1])
        h = _rms(x, self.eps, self.norms[2])
        h = self.down(F.gelu(self.gate(h), approximate="tanh") * self.up(h))
        return x + _rms(h, self.eps, self.norms[3])


class Gemma4VisionEncoder(nn.Module):
    """The `gemma4v` vision tower + projector of a Gemma 4 mmproj GGUF
    (`clip.vision.projector_type`; checked on `unsloth/gemma-4-E2B-it-GGUF`'s mmproj against HF
    `Gemma4VisionModel` + `Gemma4MultimodalEmbedder`, tools/oracle/validate_gemma4_vision.py).
    The file also holds an audio encoder (`a.*`), never loaded here.

    - Pixels in [0, 1] are mapped to [-1, 1] (no mean/std); 16x16 patches go through a bias-free
      linear (stored as a conv kernel, `v.patch_embd`).
    - A two-axis learned position table: the patch's column embedding plus its row embedding.
    - 16 Gemma4VisionLayers with 2D rotary attention: the first half of each head rotates by the
      column, the second half by the row (theta 100), full attention over every patch.
    - 3x3 average pooling (raster order over the pooled grid) times sqrt(width), then an unscaled
      RMSNorm and a linear to the text model's width (`mm.input_projection`).

    Output: one row per soft token, (rows/3 * cols/3, text width), in the text model's units.
    """

    def __init__(self, metadata: GGUFMetadata, out_dim: int) -> None:
        super().__init__()
        self.n_embd = metadata.get_u32("clip.vision.embedding_length")
        self.n_head = metadata.get_u32("clip.vision.attention.head_count")
        n_layer = metadata.get_u32("clip.vision.block_count")
        ffn_len = metadata.get_u32("clip.vision.feed_forward_length")
        self.eps = metadata.get_f32("clip.vision.attention.layer_norm_epsilon", 1e-6)
        self.patch_size = metadata.get_u32("clip.vision.patch_size")
        self.out_dim = out_dim
        self.patch_embd = nn.Parameter(
            torch.zeros(self.n_embd, 3, self.patch_size, self.patch_size)
        )
        self.position_embd = nn.Parameter(torch.zeros(2, 1, self.n_embd))
        self.layers = nn.ModuleList(
            [Gemma4VisionLayer(self.n_embd, self.n_head, ffn_len, self.eps) for _ in range(n_layer)]
        )
        self.projection = nn.Parameter(torch.zeros(out_dim, self.n_embd))

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == "clip" and (
            metadata.get_str("clip.vision.projector_type", "") == "gemma4v"
        )

    @classmethod
    def from_gguf(cls, loader: GGUFModelLoader) -> "Gemma4VisionEncoder":
        load = loader.load_tensor
        projection = load("mm.input_projection.weight")
        model = cls(loader.metadata, out_dim=projection.shape[0])
        with torch.no_grad():
            model.projection.copy_(projection)
            model.patch_embd.copy_(load("v.patch_embd.weight"))
            model.position_embd = nn.Parameter(load("v.position_embd.weight").clone())
            for i, layer in enumerate(model.layers):
                p = f"v.blk.{i}."
                pairs = (("q", "attn_q"), ("k", "attn_k"), ("v", "attn_v"), ("o", "attn_out"),
                         ("gate", "ffn_gate"), ("up", "ffn_up"), ("down", "ffn_down"))  # fmt: skip
                for attr, name in pairs:
                    linear: ClampedLinear = getattr(layer, attr)
                    linear.weight.copy_(load(p + name + ".weight"))
                    linear.bounds.copy_(
                        torch.stack([load(f"{p}{name}.{b}").reshape(()) for b in
                                     ("input_min", "input_max", "output_min", "output_max")])
                    )  # fmt: skip
                layer.q_norm.copy_(load(p + "attn_q_norm.weight"))
                layer.k_norm.copy_(load(p + "attn_k_norm.weight"))
                norms = ("ln1", "attn_post_norm", "ln2", "ffn_post_norm")
                layer.norms.copy_(torch.stack([load(f"{p}{n}.weight") for n in norms]))
        return model.eval()

    @staticmethod
    def rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """2D rotary on (n, heads, head_dim): each half of head_dim rotates on its own axis."""
        half = x.shape[-1] // 2
        parts = (x[..., :half], x[..., half:])
        return torch.cat(
            [p * c + _rotate_half(p) * s for p, c, s in zip(parts, cos, sin, strict=True)], dim=-1
        )

    def _rotary(
        self, gh: int, gw: int
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """cos/sin per axis, each (n_patches, 1, head_dim/2): columns first, then rows."""
        quarter = self.n_embd // self.n_head // 2
        device = self.projection.device
        steps = torch.arange(0, quarter, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (_ROPE_THETA ** (steps / quarter))
        cols = torch.arange(gw, device=device).repeat(gh).float()
        rows = torch.arange(gh, device=device).repeat_interleave(gw).float()
        cos, sin = [], []
        for axis in (cols, rows):
            freqs = torch.outer(axis, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(1)
            cos.append(emb.cos())
            sin.append(emb.sin())
        return tuple(cos), tuple(sin)

    def soft_grid(self, height: int, width: int) -> tuple[int, int]:
        unit = self.patch_size * _POOL
        return height // unit, width // unit

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values (1, 3, H, W) in [0, 1], H and W multiples of 48 -> (tokens, out_dim)."""
        x = F.conv2d(2 * (pixel_values - 0.5), self.patch_embd, stride=self.patch_size)
        gh, gw = x.shape[-2:]
        x = x[0].flatten(1).transpose(0, 1)
        cols = torch.arange(gw, device=x.device).repeat(gh)
        rows = torch.arange(gh, device=x.device).repeat_interleave(gw)
        x = x + self.position_embd[0][cols] + self.position_embd[1][rows]
        cos, sin = self._rotary(gh, gw)
        for layer in self.layers:
            x = layer(x, cos, sin)
        pooled = (
            x.view(gh // _POOL, _POOL, gw // _POOL, _POOL, -1)
            .mean(dim=(1, 3))
            .reshape(-1, self.n_embd)
        )
        pooled = pooled * self.n_embd**0.5
        return F.linear(_rms(pooled, self.eps), self.projection)
