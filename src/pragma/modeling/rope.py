"""`ContinuousRoPE`: rotary position encoding for arbitrary floating-point coordinates.

Implementation plan, section 7.3. Standard rotary encoding assumes integer
token positions; PRAGMA needs it to accept continuous elapsed-time
coordinates instead (profile milestone time, event time-to-latest). The
rotation math is otherwise the standard GPT-NeoX/RoFormer "rotate-half"
formulation, base 10000.0 (ADR 0009 — the paper doesn't specify a base).
"""

from __future__ import annotations

import torch
from torch import nn


class ContinuousRoPE(nn.Module):
    inv_freq: torch.Tensor  # registered as a buffer in __init__; annotated here for mypy

    def __init__(self, head_dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("ContinuousRoPE requires an even head_dim")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`positions`: `[N]` continuous coordinates. Returns `cos`, `sin`, each `[N, head_dim]`."""
        angles = positions.unsqueeze(-1).float() * self.inv_freq.unsqueeze(0)  # [N, head_dim/2]
        angles = torch.cat([angles, angles], dim=-1)  # [N, head_dim]
        return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to `x`: `[N, num_heads, head_dim]`, given per-token `cos`/`sin`: `[N, D]`."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)
