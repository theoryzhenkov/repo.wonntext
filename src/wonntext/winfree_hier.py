"""Hierarchical (two-timescale) Winfree layer for WONNText.

Inspired by the Hierarchical Reasoning Model (Wang et al. 2025, arXiv:2506.21734):
two coupled recurrent populations at different timescales --

* a **fast** population (theta_f, omega_f): one Winfree step per micro-step,
  the detailed computation (analogous to HRM's low-level / L module);
* a **slow** population (theta_s, omega_s): one Winfree step per ``tau``
  micro-steps, the abstract planning (analogous to HRM's high-level / H module).

They are bidirectionally coupled: the slow phase biases the fast coupling field
(``slow_mod``) and a summary of the fast phase biases the slow coupling field
(``fast_summ``).

The slow population runs at a lower natural frequency (``omega_s = slow_scale *
omega_f``) *and* updates ``tau`` times less often, so it is on a genuinely
slower timescale -- coupled oscillators at different frequencies is Winfree's
home territory.

Gradient (the "fast/slow gradient"): the bulk of the recurrence runs under
``torch.no_grad()``; only the **final macro-step of each segment** runs with
gradient (HRM's 1-step gradient approximation). The model detaches the carried
state between deep-supervision segments, so credit assignment is local to each
segment's final step instead of unrolled through the full N*tau trajectory --
O(1) memory in the recurrence depth and stable training of deep recurrence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from wonntext.layers import RMSNorm
from wonntext.utils import wrap_pm_pi
from wonntext.winfree import SequenceAttention


class CrossModulation(nn.Module):
    """Map the other population's phase to a per-channel additive modulation.

    cos/sin(theta_other) -> grouped 1x1 conv -> bias on the coupling field.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.ch = int(ch)
        self.proj = nn.Conv1d(
            in_channels=2 * self.ch,
            out_channels=self.ch,
            kernel_size=1,
            groups=self.ch,
            bias=True,
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        x = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
        return self.proj(x)


class HierarchicalWinfreeLayer(nn.Module):
    """Two-timescale Winfree: a slow population modulating a fast one."""

    def __init__(
        self,
        ch: int,
        heads: int = 8,
        rope: bool = True,
    ) -> None:
        super().__init__()
        self.ch = int(ch)
        self.heads = int(heads)

        # Fast population coupling (attention over token positions).
        self.f_attn = SequenceAttention(self.ch, self.heads, rope=rope, causal=False)
        self.f_norm = RMSNorm(self.ch)
        # Slow population coupling.
        self.s_attn = SequenceAttention(self.ch, self.heads, rope=rope, causal=False)
        self.s_norm = RMSNorm(self.ch)
        # Cross-population additive modulation of the coupling field.
        self.slow_mod = CrossModulation(self.ch)   # slow phase  -> fast field
        self.fast_summ = CrossModulation(self.ch)  # fast phase  -> slow field

    @staticmethod
    def _norm(norm: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        # (B, C, N) -> (B, N, C) -> norm -> (B, C, N)
        x = x.transpose(1, 2).contiguous()
        x = norm(x)
        return x.transpose(1, 2).contiguous()

    def _fast_step(
        self,
        theta_f: torch.Tensor,
        omega_f: torch.Tensor,
        theta_s: torch.Tensor,
        gamma_f: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        field = self.f_attn(torch.sin(theta_f), mask=mask)
        field = self._norm(self.f_norm, field)
        field = torch.relu(field)
        field = field + self.slow_mod(theta_s)  # slow plan biases fast field
        dtheta = omega_f + torch.cos(theta_f) * field
        return wrap_pm_pi(theta_f + gamma_f * dtheta)

    def _slow_step(
        self,
        theta_s: torch.Tensor,
        omega_s: torch.Tensor,
        theta_f: torch.Tensor,
        gamma_s: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        field = self.s_attn(torch.sin(theta_s), mask=mask)
        field = self._norm(self.s_norm, field)
        field = torch.relu(field)
        field = field + self.fast_summ(theta_f)  # fast summary biases slow field
        dtheta = omega_s + torch.cos(theta_s) * field
        return wrap_pm_pi(theta_s + gamma_s * dtheta)

    def _macro(
        self,
        theta_f: torch.Tensor,
        theta_s: torch.Tensor,
        omega_f: torch.Tensor,
        omega_s: torch.Tensor,
        tau: int,
        gamma_f: torch.Tensor,
        gamma_s: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _ in range(int(tau)):
            theta_f = self._fast_step(theta_f, omega_f, theta_s, gamma_f, mask)
        theta_s = self._slow_step(theta_s, omega_s, theta_f, gamma_s, mask)
        return theta_f, theta_s

    def forward(
        self,
        theta_f: torch.Tensor,
        theta_s: torch.Tensor,
        omega_f: torch.Tensor,
        omega_s: torch.Tensor,
        n_cycles: int,
        tau: int,
        gamma_f: torch.Tensor,
        gamma_s: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ``n_cycles`` macro-steps. All but the final macro-step run under
        ``torch.no_grad()`` (HRM 1-step gradient); the final macro-step carries
        gradient for backprop."""
        for m in range(int(n_cycles)):
            if m < int(n_cycles) - 1:
                with torch.no_grad():
                    theta_f, theta_s = self._macro(
                        theta_f, theta_s, omega_f, omega_s, tau, gamma_f, gamma_s, mask
                    )
            else:
                theta_f, theta_s = self._macro(
                    theta_f, theta_s, omega_f, omega_s, tau, gamma_f, gamma_s, mask
                )
        return theta_f, theta_s
