"""`PragmaModel` (the reusable backbone) and `PragmaForMaskedModeling` (section 7, 11.3).

Both are Hugging Face-compatible custom classes (`PreTrainedModel` subclass,
section 3) so `save_pretrained`/`from_pretrained` and weight tying work out
of the box, without depending on any particular attention kernel — the
`AttentionBackend` passed to `forward` is a plain constructor argument, not
baked into saved weights.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedModel

from pragma.attention.backend import AttentionBackend
from pragma.attention.padded import PaddedAttentionBackend
from pragma.attention.varlen import VarLenAttentionBackend
from pragma.data.batch import PragmaBatch
from pragma.modeling.config import PragmaConfig
from pragma.modeling.embeddings import SharedKeyValueEmbedding
from pragma.modeling.encoders import EventEncoder, HistoryEncoder, ProfileStateEncoder
from pragma.modeling.mlm_head import PragmaMLMHead
from pragma.modeling.outputs import PragmaOutput
from pragma.modeling.packing import cu_seqlens_from_group_ids


class PragmaModel(PreTrainedModel):
    """Profile State Encoder -> Event Encoder -> History Encoder backbone."""

    config_class = PragmaConfig
    base_model_prefix = "pragma"

    def __init__(self, config: PragmaConfig) -> None:
        super().__init__(config)
        self.embedding = SharedKeyValueEmbedding(
            config.vocab_size, config.hidden_size, config.max_within_field_position
        )
        self.profile_encoder = ProfileStateEncoder(config, self.embedding)
        self.event_encoder = EventEncoder(config, self.embedding)
        self.history_encoder = HistoryEncoder(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        # ADR 0009: BERT/GPT-2-style init — Normal(0, 0.02) truncated at 2 std,
        # zero biases, unit-scale LayerNorm.
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        batch: PragmaBatch,
        *,
        attention_backend: AttentionBackend | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns `(local_token_states, event_embeddings, record_embeddings)`.

        Without an explicit `attention_backend`, resolves from
        `config.attention_backend` ("padded" or "varlen") — this is the only
        place that string gets interpreted (ADR 0006: encoders never know
        which concrete backend ran).
        """
        backend = attention_backend or _resolve_backend(self.config.attention_backend)

        n_records = batch.n_records
        profile_cu_seqlens = _profile_cu_seqlens(batch, n_records)

        usr_states = self.profile_encoder(
            batch.profile_key_ids,
            batch.profile_value_ids,
            batch.profile_within_field_pos,
            batch.profile_time_coords,
            profile_cu_seqlens,
            backend,
        )

        local_token_states, event_summaries = self.event_encoder(
            batch.event_key_ids,
            batch.event_value_ids,
            batch.event_within_field_pos,
            batch.event_calendar_features,
            batch.event_cu_seqlens,
            backend,
        )

        record_embeddings, event_embeddings = self.history_encoder(
            usr_states,
            event_summaries,
            batch.event_time_to_latest,
            batch.history_cu_seqlens,
            backend,
        )

        return local_token_states, event_embeddings, record_embeddings


def _profile_cu_seqlens(batch: PragmaBatch, n_records: int) -> torch.Tensor:
    return cu_seqlens_from_group_ids(batch.profile_token_to_record, n_records)


def _resolve_backend(name: str) -> AttentionBackend:
    if name == "varlen":
        return VarLenAttentionBackend()
    return PaddedAttentionBackend()


class PragmaForMaskedModeling(PreTrainedModel):
    """Randomly initialized pretraining model: backbone + `PragmaMLMHead` + loss."""

    config_class = PragmaConfig
    base_model_prefix = "pragma"

    def __init__(self, config: PragmaConfig) -> None:
        super().__init__(config)
        self.pragma = PragmaModel(config)
        self.mlm_head = PragmaMLMHead(config, self.pragma.embedding)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        self.pragma._init_weights(module)

    def forward(
        self,
        batch: PragmaBatch,
        *,
        attention_backend: AttentionBackend | None = None,
    ) -> PragmaOutput:
        local_token_states, event_embeddings, record_embeddings = self.pragma(
            batch, attention_backend=attention_backend
        )
        loss, logits = self.mlm_head(
            local_token_states,
            event_embeddings,
            record_embeddings,
            batch.event_cu_seqlens,
            batch.event_to_record,
            batch.event_mlm_labels,
        )
        return PragmaOutput(
            loss=loss,
            mlm_logits=logits,
            record_embeddings=record_embeddings,
            event_embeddings=event_embeddings,
        )
