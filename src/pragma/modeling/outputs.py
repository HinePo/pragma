"""`PragmaOutput`: typed `ModelOutput` for `PragmaModel`/`PragmaForMaskedModeling`."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput


@dataclass
class PragmaOutput(ModelOutput):
    loss: torch.Tensor | None = None
    mlm_logits: torch.Tensor | None = None
    record_embeddings: torch.Tensor | None = None
    """Contextual `[USR]` state per record — `[n_records, hidden_size]`."""
    event_embeddings: torch.Tensor | None = None
    """Contextual `[EVT]` state per event — `[n_events, hidden_size]`."""
