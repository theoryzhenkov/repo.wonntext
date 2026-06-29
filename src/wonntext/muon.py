"""Muon optimizer (Jordan et al.) with an AdamW fallback for non-matrix params.

Muon orthogonalises each 2-D weight's momentum via a Newton-Schulz iteration
before the update, which is strong and FLOP-cheap for hidden weight matrices.
Embeddings, norms, biases and other <2-D (or vocab-tied) params use AdamW.

``build_optimizer(model)`` returns a :class:`HybridOptimizer` that routes
parameters automatically and exposes the usual ``step`` / ``zero_grad``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def _zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalisation of a 2-D matrix (bf16 inner math)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    transpose = x.size(-2) > x.size(-1)
    if transpose:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        aa = x @ x.mT
        x = a * x + (b * aa + c * aa @ aa) @ x
    if transpose:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: object = None) -> float | None:  # ty: ignore[invalid-method-override]
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                g = _zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                if group["weight_decay"] != 0.0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                # Scale so the update RMS is ~consistent across shapes.
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g, alpha=-group["lr"] * scale)


class HybridOptimizer:
    """Muon for hidden 2-D weights, AdamW for everything else."""

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW) -> None:
        self.muon = muon
        self.adamw = adamw

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    @property
    def param_groups(self) -> list[dict]:
        return self.muon.param_groups + self.adamw.param_groups


def build_optimizer(
    model: torch.nn.Module,
    lr: float = 0.02,
    adamw_lr: float = 3e-4,
    momentum: float = 0.95,
    weight_decay: float = 0.0,
) -> HybridOptimizer:
    """Route 2-D hidden weights to Muon, embeddings/norms/biases to AdamW."""
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embed = "embed" in name or name.endswith("head.weight")
        if p.ndim == 2 and not is_embed:
            muon_params.append(p)
        else:
            adamw_params.append(p)
    muon = Muon(muon_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    adamw = torch.optim.AdamW(
        adamw_params, lr=adamw_lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    return HybridOptimizer(muon, adamw)
