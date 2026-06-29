"""Shared modern building blocks (Llama-style) for the transformer baselines.

RMSNorm, rotary position embedding, bias-free attention with optional QK-norm,
and a SwiGLU feed-forward. WONNText reuses RMSNorm/RoPE/no-bias for symmetry but
keeps its oscillator core (it does not use the SwiGLU FFN).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos[None, None], sin[None, None]  # (1, 1, N, D)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class Attention(nn.Module):
    """Bidirectional multi-head attention with RoPE, no bias, optional QK-norm."""

    def __init__(self, dim: int, heads: int, qk_norm: bool = False) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} not divisible by heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.wqkv = nn.Linear(dim, 3 * dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.qk_norm = RMSNorm(self.head_dim) if qk_norm else None

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.wqkv(x).reshape(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, N, D)
        if self.qk_norm is not None:
            q, k = self.qk_norm(q), self.qk_norm(k)
        cos, sin = self.rope(n, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        attn_mask = None
        if mask is not None:
            attn_mask = torch.zeros(b, 1, 1, n, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(~mask[:, None, None, :], float("-inf"))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return self.wo(o.transpose(1, 2).reshape(b, n, c))


def swiglu_hidden(dim: int, multiple: int = 16) -> int:
    """SwiGLU hidden size ~ 8/3 * dim, rounded to a multiple (matches a 4x FFN)."""
    return round(8 * dim / 3 / multiple) * multiple


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    """Pre-norm RMSNorm block: RoPE attention + SwiGLU, residual."""

    def __init__(self, dim: int, heads: int, qk_norm: bool = False) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, heads, qk_norm=qk_norm)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, swiglu_hidden(dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.ffn(self.ffn_norm(x))
