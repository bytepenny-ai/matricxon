"""TimesFM 2.5 building blocks (layout of the `timesfm25` GGUF, checked against HF transformers'
`TimesFm2_5ModelForPrediction`; tools/oracle/validate_timesfm25.py)."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.layers import RMSNorm
from app.architectures.rope import apply_rotary_pos_emb

# 1/ln(2), as in the reference's query scale.
_LOG2_E = 1.442695041


class ResidualBlock(nn.Module):
    """`output(silu(hidden(x))) + skip(x)`: the patch tokenizer and both output heads."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, bias: bool) -> None:
        super().__init__()
        self.hidden = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.output = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.skip = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(F.silu(self.hidden(x))) + self.skip(x)


class TimesFmAttention(nn.Module):
    """Fused QKV, rotary, then RMS-normed q/k and a learned per-dim query scale."""

    def __init__(self, dim: int, n_head: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, 3 * n_head * head_dim, bias=False)
        self.out = nn.Linear(n_head * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)
        self.q_scale = nn.Parameter(torch.ones(head_dim))

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        shape = (batch, seq, self.n_head, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q, k = self.q_norm(q), self.k_norm(k)
        scale = F.softplus(self.q_scale) * (_LOG2_E / math.sqrt(self.head_dim))
        q = q * scale
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        return self.out(attn.transpose(1, 2).reshape(batch, seq, -1))


class TimesFmDecoderLayer(nn.Module):
    """Sandwich-normed attention and a two-layer SiLU MLP (no gate, no bias)."""

    def __init__(self, dim: int, ffn_dim: int, n_head: int, head_dim: int, eps: float) -> None:
        super().__init__()
        self.pre_attn_norm = RMSNorm(dim, eps)
        self.post_attn_norm = RMSNorm(dim, eps)
        self.pre_ff_norm = RMSNorm(dim, eps)
        self.post_ff_norm = RMSNorm(dim, eps)
        self.attn = TimesFmAttention(dim, n_head, head_dim, eps)
        self.ffn_up = nn.Linear(dim, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, dim, bias=False)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.post_attn_norm(self.attn(self.pre_attn_norm(x), cos, sin, mask))
        ff = self.ffn_down(F.silu(self.ffn_up(self.pre_ff_norm(x))))
        return x + self.post_ff_norm(ff)


class PatchMask:
    """Attention mask for left-padded patch sequences: causal, padded keys hidden. Each query
    still sees itself, so an all-padding row never turns into NaN."""

    @staticmethod
    def build(patch_padding: torch.Tensor) -> torch.Tensor:
        """`patch_padding` (B, N) bool, True = padded -> (B, 1, N, N) bool, True = may attend."""
        n = patch_padding.shape[1]
        causal = torch.ones(n, n, dtype=torch.bool, device=patch_padding.device).tril()
        allowed = causal[None] & ~patch_padding[:, None, :]
        allowed |= torch.eye(n, dtype=torch.bool, device=patch_padding.device)[None]
        return allowed[:, None]
