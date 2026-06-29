"""1-D Winfree oscillatory layer and sequence attention.

Derived from the Sudoku Winfree layer (Jiawen-Dai/WONN/sudoku/wlayer.py) and
common/modules.py. Spatial convolutions are replaced by 1-D sequence operations
so the coupling is over token positions rather than a 2-D grid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from wonntext.layers import RMSNorm
from wonntext.utils import wrap_pm_pi


class SequenceAttention(nn.Module):
    """Multi-head self-attention over 1-D token positions with RoPE.

    Supports bidirectional or causal masking.

    Unlike the original WONN ``StandardAttention``, which assumes an
    ``(B, C, H, W)`` image layout, this module operates on ``(B, C, N)``
    sequences and applies full bidirectional attention by default.
    """

    def __init__(
        self,
        ch: int,
        heads: int = 8,
        rope: bool = True,
        causal: bool = False,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        if ch % heads != 0:
            raise ValueError(f"ch={ch} must be divisible by heads={heads}.")

        self.ch = int(ch)
        self.heads = int(heads)
        self.head_dim = self.ch // self.heads
        self.rope = bool(rope)
        self.causal = bool(causal)
        self.attn_drop_p = float(attn_drop)
        self.proj_drop = nn.Dropout(float(proj_drop))
        self.rope_base = float(rope_base)

        self.W_qkv = nn.Linear(self.ch, 3 * self.ch, bias=qkv_bias)
        self.W_o = nn.Linear(self.ch, self.ch, bias=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(start_dim=-2)

    def _rope_cos_sin_1d(
        self,
        N: int,
        d: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin for 1-D RoPE over positions ``0..N-1``."""
        cos = torch.ones((N, d), device=device, dtype=dtype)
        sin = torch.zeros((N, d), device=device, dtype=dtype)

        d_rot = d - (d % 2)
        if d_rot == 0:
            return cos, sin

        exponents = torch.arange(0, d_rot, 2, device=device, dtype=dtype) / max(d_rot, 1)
        inv_freq = 1.0 / (self.rope_base**exponents)
        pos = torch.arange(N, device=device, dtype=dtype)
        freqs = torch.outer(pos, inv_freq)
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)

        cos[:, :d_rot] = freqs.cos()
        sin[:, :d_rot] = freqs.sin()
        return cos, sin

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.causal and attn_mask is not None:
            # Combine padding mask with causal mask as a float additive mask.
            N = q.shape[-2]
            device = q.device
            causal_mask = torch.triu(
                torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1
            )
            # attn_mask is (B, 1, 1, N) bool with True = keep.
            combined = attn_mask.clone()
            combined = combined.squeeze(1).squeeze(1)  # (B, N)
            bsz = combined.shape[0]
            full_mask = (
                combined[:, None, :].expand(bsz, N, N)
                & ~causal_mask[None, :, :]
            )
            additive_mask = full_mask.float().masked_fill(~full_mask, float("-inf"))
            additive_mask = additive_mask.unsqueeze(1)  # (B, 1, N, N)
            attn_mask_for_sdpa = additive_mask
        else:
            attn_mask_for_sdpa = attn_mask

        if hasattr(F, "scaled_dot_product_attention"):
            dropout_p = self.attn_drop_p if self.training else 0.0
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask_for_sdpa,
                dropout_p=dropout_p,
                is_causal=self.causal and attn_mask is None,
            )

        scale = 1.0 / math.sqrt(q.shape[-1])
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        if attn_mask_for_sdpa is not None:
            if attn_mask_for_sdpa.dtype == torch.bool:
                logits = logits.masked_fill(~attn_mask_for_sdpa, float("-inf"))
            else:
                logits = logits + attn_mask_for_sdpa
        elif self.causal:
            N = q.shape[-2]
            logits = logits.masked_fill(
                torch.triu(torch.ones(N, N, device=logits.device, dtype=torch.bool), diagonal=1),
                float("-inf"),
            )
        attn = torch.softmax(logits, dim=-1)

        if self.attn_drop_p > 0.0:
            attn = F.dropout(attn, p=self.attn_drop_p, training=self.training)

        return torch.matmul(attn, v)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x of shape (B, C, N), got {tuple(x.shape)}")

        B, C, N = x.shape
        d = self.head_dim

        x_tokens = x.transpose(1, 2).contiguous()
        qkv = self.W_qkv(x_tokens)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.reshape(B, N, self.heads, d).transpose(1, 2).contiguous()
        k = k.reshape(B, N, self.heads, d).transpose(1, 2).contiguous()
        v = v.reshape(B, N, self.heads, d).transpose(1, 2).contiguous()

        if self.rope:
            cos, sin = self._rope_cos_sin_1d(N, d, device=x.device, dtype=q.dtype)
            q, k = self._apply_rope(q, k, cos, sin)

        # ``mask`` shape (B, N). Broadcast across heads and query positions.
        attn_mask = mask.bool().unsqueeze(1).unsqueeze(2) if mask is not None else None

        out = self._sdpa(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, N, C).contiguous()
        out = self.W_o(out)
        out = self.proj_drop(out)

        return out.transpose(1, 2).contiguous()


class TokenwiseSingleIFunc(nn.Module):
    """Per-channel MLP acting on the influence signal ``sin(theta)``.

    With ``group_size=1`` the Sudoku ``PatchwiseSingleIFunc`` collapses to a
    per-token, per-channel MLP. We implement it directly in 1-D.
    """

    def __init__(
        self,
        ch: int,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()

        self.ch = int(ch)
        self.hidden_ratio = int(hidden_ratio)
        hidden = self.ch * self.hidden_ratio

        self.op = nn.Sequential(
            nn.Conv1d(self.ch, hidden, kernel_size=1, groups=self.ch, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, self.ch, kernel_size=1, groups=self.ch, bias=True),
        )

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.op(torch.sin(theta))


class ThetaEmbedding1D(nn.Module):
    """Map phase angles to periodic features in channel space.

    Mirrors WONN's ``ThetaEmbedding`` but uses a 1-D grouped convolution so it
    keeps the ``(B, C, N)`` sequence layout.
    """

    def __init__(self, channels: int, learnable: bool = True) -> None:
        super().__init__()
        self.channels = int(channels)
        self.proj = nn.Conv1d(
            in_channels=2 * self.channels,
            out_channels=self.channels,
            kernel_size=1,
            groups=self.channels,
            bias=True,
        )
        self.proj.requires_grad_(learnable)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        x = torch.cat([cos_theta, sin_theta], dim=1)
        return self.proj(x)


class WinfreeTextLayer(nn.Module):
    """A single 1-D Winfree oscillatory layer with attention coupling.

    This is the Sudoku Winfree layer with:
      * ``TokenwiseSingleIFunc`` instead of the 2-D patchwise variant;
      * ``SequenceAttention`` (1-D RoPE, full bidirectional attention);
      * channel LayerNorm over token embeddings.
    """

    def __init__(
        self,
        ch: int,
        heads: int = 8,
        rope: bool = True,
        causal: bool = False,
        hidden_ratio: int = 2,
    ) -> None:
        super().__init__()

        self.ch = int(ch)
        self.heads = int(heads)
        self.hidden_ratio = int(hidden_ratio)

        self.i_func = TokenwiseSingleIFunc(ch=self.ch, hidden_ratio=self.hidden_ratio)

        self.norm = RMSNorm(self.ch)
        self.coupling = nn.Sequential(
            SequenceAttention(
                ch=self.ch, heads=self.heads, rope=rope, causal=causal
            ),
            nn.Identity(),  # placeholder so norm is applied after attention
            nn.ReLU(inplace=True),
        )

    def _apply_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer norm on the channel-last view and restore layout."""
        # x: (B, C, N) -> (B, N, C) -> (B, C, N)
        x = x.transpose(1, 2).contiguous()
        x = self.norm(x)
        return x.transpose(1, 2).contiguous()

    def winfree_step(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        theta = wrap_pm_pi(theta)

        sensitivity = torch.cos(theta)
        influence = torch.sin(theta)

        # Coupling is attention on influence, followed by norm and ReLU.
        field = self.coupling[0](influence, mask=mask)
        field = self._apply_norm(field)
        field = self.coupling[2](field)

        dtheta = omega + sensitivity * field

        energy_int = influence * field
        return dtheta, energy_int

    def _checkpointed_step(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        mask: torch.Tensor | None,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """Single Winfree step for use with gradient checkpointing."""
        dtheta, _ = self.winfree_step(theta=theta, omega=omega, mask=mask)
        return wrap_pm_pi(theta + gamma * dtheta)

    def _forward_predictor_corrector(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        gamma: torch.Tensor,
        mask: torch.Tensor | None,
        return_thetas: bool,
        return_es: bool,
        pred_scale: float,
        t_pred: int,
        t_corr: int,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        thetas: list[torch.Tensor] | None = [] if return_thetas else None
        es: list[torch.Tensor] | None = [] if return_es else None
        if return_es and es is not None:
            es.append(torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype))

        theta = wrap_pm_pi(theta)
        gamma_pred = gamma * pred_scale

        for _ in range(t_pred):
            dtheta, energy_int = self.winfree_step(theta=theta, omega=omega, mask=mask)
            theta = wrap_pm_pi(theta + gamma_pred * dtheta)
            if return_thetas and thetas is not None:
                thetas.append(theta)
            if return_es and es is not None:
                es.append((-energy_int).reshape(theta.shape[0], -1).sum(dim=-1))

        for _ in range(t_corr):
            dtheta, energy_int = self.winfree_step(theta=theta, omega=omega, mask=mask)
            theta = wrap_pm_pi(theta + gamma * dtheta)
            if return_thetas and thetas is not None:
                thetas.append(theta)
            if return_es and es is not None:
                es.append((-energy_int).reshape(theta.shape[0], -1).sum(dim=-1))

        return theta, thetas, es

    def _forward_lazy_coupling(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        T: int,
        gamma: torch.Tensor,
        mask: torch.Tensor | None,
        lazy_k: int,
        return_thetas: bool,
        return_es: bool,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        """Multirate Winfree: recompute the expensive attention coupling field
        only every ``lazy_k`` steps and reuse the cached field in between.

        The phase still advances every step (cheap sin/cos + elementwise), but
        the slow-varying influence field ``I(sin(theta))`` is refreshed lazily.
        This cuts full attention passes from ``T`` to ``ceil(T / lazy_k)`` per
        layer. ``lazy_k = 1`` recovers the exact recurrent dynamics.
        """
        thetas: list[torch.Tensor] | None = [] if return_thetas else None
        es: list[torch.Tensor] | None = [] if return_es else None
        if return_es and es is not None:
            es.append(torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype))

        theta = wrap_pm_pi(theta)
        k = max(1, int(lazy_k))
        field: torch.Tensor | None = None

        for t in range(int(T)):
            if t % k == 0:
                # Refresh the cached coupling field (the expensive attention).
                influence = torch.sin(theta)
                field = self.coupling[0](influence, mask=mask)
                field = self._apply_norm(field)
                field = self.coupling[2](field)

            assert field is not None
            sensitivity = torch.cos(theta)
            dtheta = omega + sensitivity * field
            theta = wrap_pm_pi(theta + gamma * dtheta)

            if return_thetas and thetas is not None:
                thetas.append(theta)
            if return_es and es is not None:
                influence = torch.sin(theta)
                es.append((-(influence * field)).reshape(theta.shape[0], -1).sum(dim=-1))

        return theta, thetas, es

    def _forward_parallel_scan(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        T: int,
        gamma: torch.Tensor,
        mask: torch.Tensor | None,
        refine: bool,
    ) -> torch.Tensor:
        """Parallel-scan Winfree: evaluate coupling on a reference trajectory,
        accumulate corrections via cumsum (parallel prefix sum).

        Serial depth is O(1) on GPU: one batched attention + one cumsum.
        The coupling is evaluated on the free-running (uncoupled) trajectory,
        which is a first-order linearisation. ``refine`` does a second pass
        using the approximated trajectory as the new reference.
        """
        B, C, N = theta.shape
        device = theta.device
        dtype = theta.dtype

        steps = torch.arange(1, T + 1, device=device, dtype=dtype)
        theta_free = wrap_pm_pi(
            theta.unsqueeze(1) + gamma * steps.view(1, T, 1, 1) * omega.unsqueeze(1)
        )
        theta_ref = theta_free

        mask_expanded = (
            mask.unsqueeze(1).expand(B, T, N).reshape(B * T, N)
            if mask is not None
            else None
        )

        for _ in range(2 if refine else 1):
            flat = theta_ref.reshape(B * T, C, N)
            influence = torch.sin(flat)
            field = self.coupling[0](influence, mask=mask_expanded)
            field = self._apply_norm(field)
            field = self.coupling[2](field)

            sensitivity = torch.cos(flat)
            corrections = (gamma * sensitivity * field).reshape(B, T, C, N)
            cum_corr = torch.cumsum(corrections, dim=1)

            theta_ref = wrap_pm_pi(theta_free + cum_corr)

        return theta_ref[:, -1]

    def forward(
        self,
        theta: torch.Tensor,
        omega: torch.Tensor,
        T: int,
        gamma: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_thetas: bool = False,
        return_es: bool = False,
        grad_checkpoint: bool = False,
        mode: str = "recurrent",
        pred_scale: float = 3.0,
        t_pred: int = 2,
        t_corr: int = 3,
        lazy_k: int = 2,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        if mode in ("parallel_scan", "parallel_scan_refined"):
            result = self._forward_parallel_scan(
                theta=theta,
                omega=omega,
                T=T,
                gamma=gamma,
                mask=mask,
                refine=(mode == "parallel_scan_refined"),
            )
            return result, None, None

        if mode == "lazy_coupling":
            return self._forward_lazy_coupling(
                theta=theta,
                omega=omega,
                T=T,
                gamma=gamma,
                mask=mask,
                lazy_k=lazy_k,
                return_thetas=return_thetas,
                return_es=return_es,
            )

        if mode == "predictor_corrector":
            return self._forward_predictor_corrector(
                theta=theta,
                omega=omega,
                gamma=gamma,
                mask=mask,
                return_thetas=return_thetas,
                return_es=return_es,
                pred_scale=pred_scale,
                t_pred=t_pred,
                t_corr=t_corr,
            )

        thetas: list[torch.Tensor] | None = [] if return_thetas else None
        es: list[torch.Tensor] | None = [] if return_es else None

        if return_es and es is not None:
            es.append(torch.zeros(theta.shape[0], device=theta.device, dtype=theta.dtype))

        theta = wrap_pm_pi(theta)

        # Gradient checkpointing trades ~33% extra compute for O(1) memory
        # across the T recurrence. Disabled for diagnostic passes that need
        # intermediate thetas/energies.
        use_ckpt = grad_checkpoint and not (return_thetas or return_es)

        for _ in range(int(T)):
            if use_ckpt:
                theta = torch.utils.checkpoint.checkpoint(
                    self._checkpointed_step,
                    theta,
                    omega,
                    mask,
                    gamma,
                    use_reentrant=False,
                )
            else:
                dtheta, energy_int = self.winfree_step(
                    theta=theta, omega=omega, mask=mask
                )
                theta = wrap_pm_pi(theta + gamma * dtheta)

            if return_thetas:
                assert thetas is not None
                thetas.append(theta)

            if return_es:
                assert es is not None
                es.append((-energy_int).reshape(theta.shape[0], -1).sum(dim=-1))

        return theta, thetas, es
