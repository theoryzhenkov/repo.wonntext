"""Transformer baselines for the WONNText study (modern Llama-style stack).

Two architectures share one block (RoPE attention, RMSNorm, SwiGLU, no bias):
  * ``TransformerLM`` - classical, ``n_layers`` distinct blocks.
  * ``UniversalTransformerLM`` - one tied block applied ``n_steps`` times.

The Universal Transformer is the param- and FLOP-matched control for WONNText
(both reuse one block's weights across depth); the classical Transformer is
FLOP-matched with untied weights (more parameters).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from wonntext.layers import RMSNorm, TransformerBlock


def _init_weights(module: nn.Module, depth: int) -> None:
    """Llama-style init: normal(0, 0.02), output projections scaled by depth."""
    for name, p in module.named_parameters():
        if p.dim() < 2:
            continue
        if name.endswith("wo.weight") or name.endswith("w_down.weight"):
            nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * max(depth, 1)))
        elif "weight" in name:
            nn.init.normal_(p, mean=0.0, std=0.02)


class _BaseLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, pad_token_id: int, mask_token_id: int) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = vocab_size - 1 if mask_token_id < 0 else int(mask_token_id)
        self.embed = nn.Embedding(self.vocab_size, self.dim)
        self.norm_f = RMSNorm(self.dim)
        self.head = nn.Linear(self.dim, self.vocab_size, bias=False)
        self.head.weight = self.embed.weight  # tied

    def _run_blocks(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask = attention_mask.bool() if attention_mask is not None else None
        if mask is not None and bool(mask.all()):
            mask = None  # all-ones -> let SDPA take the FlashAttention fast path
        x = self.embed(input_ids)
        x = self._run_blocks(x, mask)
        logits = self.head(self.norm_f(x))
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = F.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1), ignore_index=-100
            )
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TransformerLM(_BaseLM):
    """Classical transformer: ``n_layers`` distinct blocks."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        heads: int,
        n_layers: int,
        pad_token_id: int = 0,
        mask_token_id: int = -1,
        qk_norm: bool = False,
        **_: object,
    ) -> None:
        super().__init__(vocab_size, dim, pad_token_id, mask_token_id)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, qk_norm=qk_norm) for _ in range(n_layers)]
        )
        _init_weights(self, depth=n_layers)

    def _run_blocks(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class UniversalTransformerLM(_BaseLM):
    """Universal transformer: one tied block applied ``n_steps`` times."""

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        heads: int,
        n_steps: int,
        pad_token_id: int = 0,
        mask_token_id: int = -1,
        qk_norm: bool = False,
        **_: object,
    ) -> None:
        super().__init__(vocab_size, dim, pad_token_id, mask_token_id)
        self.n_steps = int(n_steps)
        self.block = TransformerBlock(dim, heads, qk_norm=qk_norm)
        _init_weights(self, depth=self.n_steps)

    def _run_blocks(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        for _ in range(self.n_steps):
            x = self.block(x, mask)
        return x
