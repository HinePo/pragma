"""Muon: Newton-Schulz-orthogonalized momentum optimizer for 2D hidden-layer weights.

A minimal, direct implementation of Keller Jordan's public Muon algorithm
(https://github.com/KellerJordan/Muon) — not a new dependency, since the
reference implementation is ~20 lines and there is no PyPI package to pin.
Only used for the 2D weight matrices `OptimizerFactory` routes to it
(ADR 0011); everything else (embeddings, LayerNorm, biases) uses AdamW.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


def _zeropower_via_newton_schulz(grad: torch.Tensor, steps: int) -> torch.Tensor:
    """Approximately orthogonalizes `grad` (a 2D matrix) via Newton-Schulz iteration —
    the core Muon operation, replacing the raw gradient with a direction of the same
    shape but ~orthonormal singular vectors, which is what lets a single learning
    rate work well across differently-scaled hidden matrices."""
    if grad.ndim != 2:
        raise ValueError(f"Muon only supports 2D parameters, got shape {tuple(grad.shape)}")
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(grad.dtype)


class Muon(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {"lr": lr, "momentum": momentum, "nesterov": nesterov, "ns_steps": ns_steps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> torch.Tensor | None:  # type: ignore[override]
        loss = closure() if closure is not None else None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(grad)
                update = grad.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = _zeropower_via_newton_schulz(update, steps=group["ns_steps"])
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(update, alpha=-group["lr"] * scale)

        return loss
