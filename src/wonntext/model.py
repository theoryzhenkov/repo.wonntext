"""WONNText masked-language model.

Input: token ids -> embedding -> omega_init (frequency carries token content)
Coupling: 1-D bidirectional attention over token positions
Output head: final phase -> (sin, cos) -> linear -> vocab logits, tied to input embedding
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wonntext.utils import wrap_pm_pi
from wonntext.winfree import ThetaEmbedding1D, WinfreeTextLayer


class WONNText(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        ch: int = 256,
        max_seq_len: int = 128,
        L: int = 1,
        T: int | Sequence[int] = 16,
        heads: int = 8,
        gamma: float = 0.1,
        rope: bool = True,
        causal: bool = False,
        theta_init_sigma: float = 0.1,
        hidden_ratio: int = 2,
        omega_as_token_embed: bool = True,
        pad_token_id: int = 0,
        mask_token_id: int = -1,
        grad_checkpoint: bool = False,
        winfree_mode: str = "recurrent",
    ) -> None:
        super().__init__()

        if mask_token_id < 0:
            # Default: last vocab entry is the mask token, like a regular symbol.
            mask_token_id = vocab_size - 1

        self.vocab_size = int(vocab_size)
        self.ch = int(ch)
        self.max_seq_len = int(max_seq_len)
        self.L = int(L)
        self.heads = int(heads)
        self.gamma = nn.Parameter(torch.tensor([float(gamma)]), requires_grad=False)
        self.theta_init_sigma = float(theta_init_sigma)
        self.hidden_ratio = int(hidden_ratio)
        self.pad_token_id = int(pad_token_id)
        self.mask_token_id = int(mask_token_id)

        self.T = self._expand_T(T, self.L)
        self.omega_as_token_embed = omega_as_token_embed

        # f_init: token -> embedding -> frequency content for the oscillator.
        self.token_embed = nn.Embedding(self.vocab_size, self.ch)
        if self.omega_as_token_embed:
            self.omega_embed: nn.Embedding = self.token_embed
        else:
            # Separate random frequency embedding (still learnable).
            self.omega_embed = nn.Embedding(self.vocab_size, self.ch)

        # Single (or stacked) Winfree text layer(s).
        self.layers = nn.ModuleList(
            [
                WinfreeTextLayer(
                    ch=self.ch,
                    heads=self.heads,
                    rope=rope,
                    causal=causal,
                    hidden_ratio=self.hidden_ratio,
                )
                for _ in range(self.L)
            ]
        )

        # Readout: final phase -> periodic features -> vocab logits.
        self.theta_embed = ThetaEmbedding1D(self.ch, learnable=True)
        self.output_proj = nn.Linear(self.ch, self.vocab_size, bias=False)

        # Tie input embedding and output projection for honest parameter counts.
        self.output_proj.weight = self.token_embed.weight

        self.grad_checkpoint = bool(grad_checkpoint)
        self.winfree_mode = str(winfree_mode)

        self._init_weights()

    def _init_weights(self) -> None:
        # Embedding is left at default PyTorch init; small scale matches WONN.
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        if not self.omega_as_token_embed:
            nn.init.normal_(self.omega_embed.weight, mean=0.0, std=0.02)

    @staticmethod
    def _expand_T(T: int | Sequence[int], L: int) -> list[int]:
        if isinstance(T, (list, tuple)):
            if len(T) == L:
                return [int(t) for t in T]
            if len(T) == 1:
                return [int(T[0])] * L
            raise ValueError(
                f"T must be an int, a one-element list, or length L={L}; got len={len(T)}."
            )
        assert isinstance(T, int)
        return [T] * L

    def feature(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_thetas: bool = False,
        return_es: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[list[torch.Tensor]] | None,
        list[list[torch.Tensor]] | None,
    ]:
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be (B, N), got {tuple(input_ids.shape)}")
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len {self.max_seq_len}"
            )

        # (B, N, C) -> (B, C, N)
        if self.omega_as_token_embed:
            omega = self.token_embed(input_ids).transpose(1, 2).contiguous()
        else:
            omega = self.omega_embed(input_ids).transpose(1, 2).contiguous()

        theta = wrap_pm_pi(self.theta_init_sigma * torch.randn_like(omega))

        # An all-ones attention mask (no padding, e.g. chunked corpus data) is a
        # no-op but forces SDPA off the FlashAttention-2 fast path. Collapse it
        # to None once here so all T*L attention passes can use the flash kernel.
        # One sync per forward is negligible against the T*L attention calls.
        if attention_mask is not None and bool(attention_mask.all()):
            attention_mask = None

        thetas: list[list[torch.Tensor]] | None = [] if return_thetas else None
        es: list[list[torch.Tensor]] | None = [] if return_es else None

        for layer_idx, layer in enumerate(self.layers):
            theta, layer_thetas, layer_es = layer(
                theta=theta,
                omega=omega,
                T=self.T[layer_idx],
                gamma=self.gamma,
                mask=attention_mask,
                return_thetas=return_thetas,
                return_es=return_es,
                grad_checkpoint=self.grad_checkpoint,
                mode=self.winfree_mode,
            )

            if return_thetas and thetas is not None:
                thetas.append(layer_thetas or [])

            if return_es and es is not None:
                es.append(layer_es or [])

        return theta, thetas, es

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_thetas: bool = False,
        return_es: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        final_theta, _, es = self.feature(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_thetas=return_thetas,
            return_es=return_es,
        )

        # (B, C, N) -> (B, N, C)
        phase_repr = self.theta_embed(final_theta).transpose(1, 2).contiguous()
        logits = self.output_proj(phase_repr)

        loss: torch.Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        output: dict[str, torch.Tensor] = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        if return_thetas:
            output["thetas"] = final_theta
        if return_es:
            output["energies"] = (
                torch.stack(
                    [torch.stack(layer_es, dim=1) for layer_es in es if layer_es],
                    dim=1,
                )
                if es
                else torch.empty(0)
            )

        # Preserve the simple ``logits`` return when no auxiliary outputs are requested.
        if labels is None and not return_thetas and not return_es:
            return logits

        return output
