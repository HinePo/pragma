"""`PragmaForTask`: backbone plus a configurable binary classification head.

Implementation plan section 11.3's `PragmaForTask` row ("Backbone plus
configurable binary, multiclass, multilabel, or regression head") and
section 15.2's task-head decision (open per section 18 — resolved in
ADR 0013). Phase 11's MVP only needs binary classification (section 15.2:
"the first MVP task should be a well-defined binary classification
problem"), so this is deliberately the binary case only — multiclass/
multilabel/regression heads are a mechanical extension of the same pattern
(swap the loss and output width) whenever a later task needs them, not
speculative code written now for hypothetical future tasks.

The head sits on `record_embeddings` (the contextual `[USR]` state) — the
same "customer-level summary" role section 7 gives `[USR]`, and the variant
Phase 10's probe comparison (`probe_usr`) showed carries as much signal as
the richer `concat` variant on this corpus (ADR 0013).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.utils import ModelOutput

from pragma.data.batch import PragmaBatch
from pragma.modeling.config import PragmaConfig
from pragma.modeling.model import PragmaModel


@dataclass
class PragmaTaskOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    """`[n_records]` — raw (pre-sigmoid) binary logits, one per record."""
    record_embeddings: torch.Tensor | None = None


class PragmaForTask(PreTrainedModel):
    """`PragmaModel` backbone + a single `Linear(hidden_size, 1)` binary
    classification head on `record_embeddings`. Not a pretraining-objective
    model (no MLM head, no masking) — `forward` expects a plain, unmasked
    `PragmaBatch` (a `PragmaCollator()` with no `masking_planner`, same as
    `pragma.evaluation.embeddings.EmbeddingExtractor` uses)."""

    config_class = PragmaConfig
    base_model_prefix = "pragma"

    def __init__(self, config: PragmaConfig) -> None:
        super().__init__(config)
        self.pragma = PragmaModel(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        self.pragma._init_weights(module)

    def forward(
        self,
        batch: PragmaBatch,
        *,
        labels: torch.Tensor | None = None,
    ) -> PragmaTaskOutput:
        _, _, record_embeddings = self.pragma(batch)
        logits = self.classifier(record_embeddings).squeeze(-1)

        loss = None
        if labels is not None:
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels.float())

        return PragmaTaskOutput(loss=loss, logits=logits, record_embeddings=record_embeddings)
