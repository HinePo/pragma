"""`PragmaMLMHead` (implementation plan, section 7.5).

For every masked value-token position, concatenates the Event Encoder's
local token state, the History Encoder's contextual state for that event,
and the History Encoder's contextual `[USR]` state for that record —
`3 x hidden_size` — projects back to `hidden_size`, and matches against the
tied value-vocabulary slice of the shared embedding table (ADR 0009: MLM
logits are computed only over `value_vocab_start:`, never the full
vocabulary, since a masked value token can never equal a special/key ID).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pragma.modeling.config import PragmaConfig
from pragma.modeling.embeddings import SharedKeyValueEmbedding
from pragma.modeling.packing import group_index_per_token

IGNORE_INDEX = -100


class PragmaMLMHead(nn.Module):
    def __init__(self, config: PragmaConfig, embedding: SharedKeyValueEmbedding) -> None:
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.proj = nn.Linear(3 * config.hidden_size, config.hidden_size)

    def forward(
        self,
        local_token_states: torch.Tensor,
        contextual_event: torch.Tensor,
        contextual_usr: torch.Tensor,
        event_cu_seqlens: torch.Tensor,
        event_to_record: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Returns `(loss_or_None, logits)`; `logits` is `[n_masked, value_vocab_size]`.

        Only masked positions (`labels != -100`) get logits materialized —
        section 7.5's "avoid allocating a full batch x sequence x vocabulary
        tensor."
        """
        masked = labels != IGNORE_INDEX
        n_event_tokens = local_token_states.shape[0]
        if not bool(masked.any()):
            empty = local_token_states.new_zeros(
                (0, self.embedding.embedding.num_embeddings - self.config.value_vocab_start)
            )
            return None, empty

        token_to_event = group_index_per_token(event_cu_seqlens, n_event_tokens)
        event_idx = token_to_event[masked]
        record_idx = event_to_record[event_idx]

        local = local_token_states[masked]
        ctx_event = contextual_event[event_idx]
        ctx_usr = contextual_usr[record_idx]

        combined = torch.cat([local, ctx_event, ctx_usr], dim=-1)
        projected = self.proj(combined)

        value_weight = self.embedding.embedding.weight[self.config.value_vocab_start :]
        logits = projected @ value_weight.T

        shifted_labels = labels[masked] - self.config.value_vocab_start
        loss = F.cross_entropy(logits, shifted_labels, label_smoothing=self.config.label_smoothing)
        return loss, logits
