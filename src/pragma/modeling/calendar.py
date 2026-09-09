"""`CalendarEncoder`: two-layer MLP over the six calendar cyclical features.

Implementation plan, section 7.3. Maps `pragma.processing.temporal.calendar_features`'s
sine/cosine hour/weekday/day-of-month features to model width; the result is
added to the Event Encoder's `[EVT]` summary (not to every event token).
`6 -> hidden_size -> hidden_size` with GELU between layers is ADR 0009's
decision — the paper doesn't specify this MLP's hidden dimension.
"""

from __future__ import annotations

import torch
from torch import nn


class CalendarEncoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(6, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, calendar_features: torch.Tensor) -> torch.Tensor:
        """`calendar_features`: `[n_events, 6]` -> `[n_events, hidden_size]`."""
        return self.fc2(nn.functional.gelu(self.fc1(calendar_features)))
