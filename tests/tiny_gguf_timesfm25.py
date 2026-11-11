"""Builds a tiny, valid `timesfm25` GGUF (random F32 weights, the real tensor layout), so tests run
the whole forecasting path fast. tools/oracle/validate_timesfm25.py checks the math on real
weights; this exercises the wiring."""

from pathlib import Path

import numpy as np

from app.gguf.constants import GGMLQuantizationType, GGUFValueType
from tools.make_tiny_gguf import GGUFBuilder

N_EMBD = 16
N_HEAD = 2
HEAD_DIM = 8
N_LAYER = 2
PATCH = 4
HORIZON = 8
QUANTILE_LEN = 16
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_OUT = len(QUANTILES) + 1


class TinyTimesFm25:
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.RandomState(seed)
        self._builder = (
            GGUFBuilder()
            .set_str("general.architecture", "timesfm25")
            .set_u32("timesfm25.block_count", N_LAYER)
            .set_u32("timesfm25.embedding_length", N_EMBD)
            .set_u32("timesfm25.feed_forward_length", N_EMBD)
            .set_u32("timesfm25.attention.head_count", N_HEAD)
            .set_u32("timesfm25.attention.head_dim", HEAD_DIM)
            .set_u32("timesfm25.input_patch_len", PATCH)
            .set_u32("timesfm25.output_patch_len", HORIZON)
            .set_u32("timesfm25.output_quantile_len", QUANTILE_LEN)
            .set_u32("timesfm25.decode_index", 5)
            .set_u32("timesfm25.quantile_count", len(QUANTILES))
            .set_array("timesfm25.quantiles", GGUFValueType.FLOAT32, QUANTILES)
            .set_f32("timesfm25.rope_theta", 10000.0)
            .set_f32("timesfm25.rms_norm_epsilon", 1e-6)
            .set_u32("timesfm25.context_limit", 64)
        )

    def build(self, path: Path) -> Path:
        self._block("tokenizer", 2 * PATCH, N_EMBD, bias=True)
        self._block("out_point", N_EMBD, HORIZON * N_OUT, bias=False)
        self._block("out_quantile", N_EMBD, QUANTILE_LEN * N_OUT, bias=False)
        for i in range(N_LAYER):
            p = f"blk.{i}."
            for norm in ("pre_attn_norm", "post_attn_norm", "pre_ff_norm", "post_ff_norm"):
                self._add(p + norm + ".weight", 1 + 0.1 * self._rng.randn(N_EMBD))
            for norm in ("attn_q_norm", "attn_k_norm", "attn_q_scale"):
                self._add(p + norm + ".weight", 1 + 0.1 * self._rng.randn(HEAD_DIM))
            self._add(p + "attn_qkv.weight", self._weight(3 * N_EMBD, N_EMBD))
            self._add(p + "attn_out.weight", self._weight(N_EMBD, N_EMBD))
            self._add(p + "ffn_up.weight", self._weight(N_EMBD, N_EMBD))
            self._add(p + "ffn_down.weight", self._weight(N_EMBD, N_EMBD))
        self._builder.write(path)
        return path

    def _block(self, prefix: str, in_dim: int, out_dim: int, bias: bool) -> None:
        for part, rows, cols in (
            ("hidden", N_EMBD, in_dim),
            ("output", out_dim, N_EMBD),
            ("skip", out_dim, in_dim),
        ):
            self._add(f"{prefix}.{part}.weight", self._weight(rows, cols))
            if bias:
                self._add(f"{prefix}.{part}.bias", 0.1 * self._rng.randn(rows))

    def _weight(self, rows: int, cols: int) -> np.ndarray:
        return self._rng.randn(rows, cols) / np.sqrt(cols)

    def _add(self, name: str, array: np.ndarray) -> None:
        data = array.astype("<f4")
        self._builder.add_tensor(
            name, list(reversed(data.shape)), GGMLQuantizationType.F32, data.tobytes()
        )


def build_tiny_timesfm25_gguf(path: Path, seed: int = 0) -> Path:
    return TinyTimesFm25(seed).build(path)
