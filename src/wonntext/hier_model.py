"""Hierarchical WONNText: a two-timescale (fast/slow) oscillator language model.

A variant of :class:`WONNText` built on :class:`HierarchicalWinfreeLayer`. The
core WONN design choices are preserved:

* token id -> embedding -> omega_init (frequency carries token content);
* 1-D bidirectional attention coupling over token positions;
* tied output projection (input embedding == output head);
* masked cross-entropy.

Added hierarchy (HRM-inspired):

* a slow oscillator population modulates the fast one (see ``winfree_hier``);
* deep supervision over ``segments`` refinement passes, with the carried state
  detached between segments and only the final macro-step of each segment
  carrying gradient (the fast/slow gradient);
* readout from the slow phase by default (HRM reads the high-level state).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from wonntext.utils import wrap_pm_pi
from wonntext.winfree import ThetaEmbedding1D
from wonntext.winfree_hier import HierarchicalWinfreeLayer


class HierarchicalWONNText(nn.Module):
    """Two-timescale oscillator LM with deep-supervision refinement."""

    def __init__(
        self,
        vocab_size: int,
        ch: int = 256,
        max_seq_len: int = 128,
        heads: int = 8,
        n_cycles: int = 2,
        tau: int = 4,
        segments: int = 4,
        slow_scale: float = 0.25,
        gamma_f: float = 0.1,
        gamma_s: float = 0.1,
        readout: str = "slow",
        theta_init_sigma: float = 0.1,
        pad_token_id: int = 0,
        mask_token_id: int = -1,
    ) -> None:
        super().__init__()
        if mask_token_id < 0:
            mask_token_id = vocab_size - 1
        if readout not in ("fast", "slow", "sum"):
            raise ValueError(f"readout must be fast|slow|sum, got {readout!r}")

        self.vocab_size = int(vocab_size)
        self.ch = int(ch)
        self.max_seq_len = int(max_seq_len)
        self.heads = int(heads)
        self.n_cycles = int(n_cycles)
        self.tau = int(tau)
        self.segments = int(segments)
        self.slow_scale = float(slow_scale)
        self.readout = str(readout)
        self.theta_init_sigma = float(theta_init_sigma)
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)

        # Fixed coupling strengths (matches baseline WONN's non-learnable gamma).
        self.gamma_f = nn.Parameter(torch.tensor([float(gamma_f)]), requires_grad=False)
        self.gamma_s = nn.Parameter(torch.tensor([float(gamma_s)]), requires_grad=False)

        # token id -> embedding; the embedding doubles as the fast frequency.
        self.token_embed = nn.Embedding(self.vocab_size, self.ch)
        self.layer = HierarchicalWinfreeLayer(ch=self.ch, heads=self.heads)

        # Readout: phase -> periodic features -> vocab logits, tied to embedding.
        self.theta_embed = ThetaEmbedding1D(self.ch, learnable=True)
        self.output_proj = nn.Linear(self.ch, self.vocab_size, bias=False)
        self.output_proj.weight = self.token_embed.weight  # tie

        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

    def _readout_phase(
        self, theta_f: torch.Tensor, theta_s: torch.Tensor
    ) -> torch.Tensor:
        if self.readout == "fast":
            return theta_f
        if self.readout == "slow":
            return theta_s
        return theta_f + theta_s  # "sum"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, N), got {tuple(input_ids.shape)}")
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len {self.max_seq_len}"
            )

        # (B, N, C) -> (B, C, N)
        omega_f = self.token_embed(input_ids).transpose(1, 2).contiguous()
        omega_s = omega_f * self.slow_scale  # slower natural frequency, same content

        theta_f = wrap_pm_pi(self.theta_init_sigma * torch.randn_like(omega_f))
        theta_s = wrap_pm_pi(self.theta_init_sigma * torch.randn_like(omega_s))

        # Collapse an all-ones mask to None so SDPA can use the flash kernel.
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None

        logits: torch.Tensor | None = None
        losses: list[torch.Tensor] = []
        for _ in range(self.segments):
            theta_f, theta_s = self.layer(
                theta_f, theta_s, omega_f, omega_s,
                n_cycles=self.n_cycles, tau=self.tau,
                gamma_f=self.gamma_f, gamma_s=self.gamma_s, mask=attention_mask,
            )
            phase = self._readout_phase(theta_f, theta_s)
            phase_repr = self.theta_embed(phase).transpose(1, 2).contiguous()
            logits = self.output_proj(phase_repr)
            if labels is not None:
                losses.append(
                    F.cross_entropy(
                        logits.view(-1, self.vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                )
            # Segment boundary: detach the carried state (fast/slow gradient).
            theta_f = theta_f.detach()
            theta_s = theta_s.detach()

        assert logits is not None  # segments >= 1 guarantees logits is set
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            out["loss"] = torch.stack(losses).mean()
        return out
