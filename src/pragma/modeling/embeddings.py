"""`SharedKeyValueEmbedding` and `WithinFieldPositionEncoding` (section 7.2, 6.4).

Keys, values, and special tokens all live in one shared ID space (Phase 2's
`PragmaProcessor` lays out key IDs immediately after special tokens, then
value ID ranges after that — section 6.4/6.5, ADR 0008), so one
`nn.Embedding` table covers all of them. A `[USR]`/`[EVT]` special-token slot
reuses the same `E(k) + E(v) + P(0)` formula with `k = v = special_id`
(ADR 0009) rather than a separate code path.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class WithinFieldPositionEncoding(nn.Module):
    """Deterministic (non-learned) sinusoidal encoding of within-field token position."""

    table: torch.Tensor  # registered as a buffer in __init__; annotated here for mypy

    def __init__(self, hidden_size: int, max_position: int = 32) -> None:
        super().__init__()
        position = torch.arange(max_position, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_size)
        )
        table = torch.zeros(max_position, hidden_size)
        table[:, 0::2] = torch.sin(position * div_term)
        table[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("table", table, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.table[positions]


class SharedKeyValueEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, hidden_size: int, max_within_field_position: int = 32
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.within_field_position = WithinFieldPositionEncoding(
            hidden_size, max_within_field_position
        )

    def forward(
        self, key_ids: torch.Tensor, value_ids: torch.Tensor, within_field_pos: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.embedding(key_ids)
            + self.embedding(value_ids)
            + self.within_field_position(within_field_pos)
        )

    def embed_special(self, special_id: int, n: int, device: torch.device) -> torch.Tensor:
        """`[USR]`/`[EVT]`-style slot embedding: `E(id) + E(id) + P(0)`, broadcast `n` times."""
        ids = torch.full((n,), special_id, dtype=torch.long, device=device)
        zeros = torch.zeros((n,), dtype=torch.long, device=device)
        return self.forward(ids, ids, zeros)
