"""`ProfileStateEncoder`, `EventEncoder`, `HistoryEncoder` (implementation plan, section 7.4).

All three share the same packing pattern (ADR 0009): embed real tokens,
prepend one summary vector per group (`[USR]` or `[EVT]`) via
`pragma.modeling.packing`, run transformer blocks with per-group attention
isolation, then split the output back into "local token states" and "group
summary" via `unprepend_vector`/`group_starts`.
"""

from __future__ import annotations

import torch
from torch import nn

from pragma.attention.backend import AttentionBackend
from pragma.modeling.calendar import CalendarEncoder
from pragma.modeling.config import PragmaConfig
from pragma.modeling.embeddings import SharedKeyValueEmbedding
from pragma.modeling.packing import group_starts, prepend_vector, unprepend_vector
from pragma.modeling.rope import ContinuousRoPE
from pragma.modeling.transformer_block import PragmaTransformerBlock


class ProfileStateEncoder(nn.Module):
    """Encodes one `[USR]` + profile key/value tokens per record, bidirectionally.

    Applies `ContinuousRoPE` to the profile temporal coordinates (lifelong-
    milestone elapsed time; static attributes and the `[USR]` slot get `0.0`
    — ADR 0009).
    """

    def __init__(self, config: PragmaConfig, embedding: SharedKeyValueEmbedding) -> None:
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.rope = ContinuousRoPE(config.hidden_size // config.num_heads, config.rope_base)
        self.blocks = nn.ModuleList(
            [
                PragmaTransformerBlock(
                    config.hidden_size, config.num_heads, config.intermediate_size, config.dropout
                )
                for _ in range(config.profile_layers)
            ]
        )

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        within_field_pos: torch.Tensor,
        time_coords: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_backend: AttentionBackend,
    ) -> torch.Tensor:
        """Returns the contextual `[USR]` summary per record: `[n_records, hidden_size]`."""
        n_records = cu_seqlens.shape[0] - 1
        device = key_ids.device

        x = self.embedding(key_ids, value_ids, within_field_pos)
        usr_vec = self.embedding.embed_special(self.config.usr_token_id, n_records, device)
        x, new_cu = prepend_vector(x, cu_seqlens, usr_vec)

        usr_time = torch.zeros(n_records, device=device)
        positions, _ = prepend_vector(time_coords.unsqueeze(-1), cu_seqlens, usr_time.unsqueeze(-1))
        positions = positions.squeeze(-1)
        cos, sin = self.rope(positions)

        for block in self.blocks:
            x = block(x, new_cu, attention_backend, rope_cos=cos, rope_sin=sin)

        return group_starts(x, cu_seqlens)


class EventEncoder(nn.Module):
    """Encodes every event independently, isolated from every other event.

    Emits per-token local states (for the MLM head) and one calendar-augmented
    `[EVT]` summary per event. No RoPE here — within-event token order comes
    from the within-field positional encoding baked into the embedding, not
    from a temporal coordinate (section 7.4).
    """

    def __init__(self, config: PragmaConfig, embedding: SharedKeyValueEmbedding) -> None:
        super().__init__()
        self.config = config
        self.embedding = embedding
        self.calendar = CalendarEncoder(config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                PragmaTransformerBlock(
                    config.hidden_size, config.num_heads, config.intermediate_size, config.dropout
                )
                for _ in range(config.event_layers)
            ]
        )

    def forward(
        self,
        key_ids: torch.Tensor,
        value_ids: torch.Tensor,
        within_field_pos: torch.Tensor,
        calendar_features: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_backend: AttentionBackend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(local_token_states [n_event_tokens, H], event_summaries [n_events, H])`."""
        n_events = cu_seqlens.shape[0] - 1
        device = key_ids.device

        x = self.embedding(key_ids, value_ids, within_field_pos)
        evt_vec = self.embedding.embed_special(self.config.evt_token_id, n_events, device)
        x, new_cu = prepend_vector(x, cu_seqlens, evt_vec)

        for block in self.blocks:
            x = block(x, new_cu, attention_backend)

        local_token_states = unprepend_vector(x, cu_seqlens)
        event_summaries = group_starts(x, cu_seqlens) + self.calendar(calendar_features)
        return local_token_states, event_summaries


class HistoryEncoder(nn.Module):
    """Encodes `[USR]` (the profile encoder's contextual output) followed by
    ordered event summaries per record. Applies `ContinuousRoPE` using each
    event's time-to-latest coordinate (`[USR]`'s own coordinate is `0.0`,
    ADR 0009), contextualizing the complete customer history.
    """

    def __init__(self, config: PragmaConfig) -> None:
        super().__init__()
        self.config = config
        self.rope = ContinuousRoPE(config.hidden_size // config.num_heads, config.rope_base)
        self.blocks = nn.ModuleList(
            [
                PragmaTransformerBlock(
                    config.hidden_size, config.num_heads, config.intermediate_size, config.dropout
                )
                for _ in range(config.history_layers)
            ]
        )

    def forward(
        self,
        usr_states: torch.Tensor,
        event_summaries: torch.Tensor,
        event_time_to_latest: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_backend: AttentionBackend,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns `(contextual_usr [n_records, H], contextual_event [n_events, H])`."""
        n_records = cu_seqlens.shape[0] - 1
        device = event_summaries.device

        x, new_cu = prepend_vector(event_summaries, cu_seqlens, usr_states)

        usr_time = torch.zeros(n_records, device=device)
        positions, _ = prepend_vector(
            event_time_to_latest.unsqueeze(-1), cu_seqlens, usr_time.unsqueeze(-1)
        )
        positions = positions.squeeze(-1)
        cos, sin = self.rope(positions)

        for block in self.blocks:
            x = block(x, new_cu, attention_backend, rope_cos=cos, rope_sin=sin)

        contextual_usr = group_starts(x, cu_seqlens)
        contextual_event = unprepend_vector(x, cu_seqlens)
        return contextual_usr, contextual_event
