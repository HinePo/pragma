"""`MaskingPlanner`: samples token/event/key masks for one record (section 8; ADR 0005).

Operates on one `TokenizedRecord` at a time, over its *event* value tokens
only — profile-state values are never masked (section 7.5: they stay visible
as contextual signal). Masking eligibility is a schema property, not a
per-call decision: `MaskingPlanner.from_registry` precomputes which key IDs
are eligible once, from `SchemaRegistry.get_field(key).maskable`. In
practice every key that ever appears on an *event* token is already
maskable by construction — `KeyVocabulary` only tokenizes
NUMERICAL/CATEGORICAL/TEXT fields plus lifelong-milestone fields, and
milestone fields never appear on events (only in the profile) — but the
eligibility set is still computed explicitly rather than assumed, so a
future schema change that adds a non-maskable event field is handled
correctly without touching this module.

The three mask sources are sampled independently and combined by union
(ADR 0005); semantic-key masking is record-wide, not per-event.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import IntFlag

from pragma.config.masking_config import MaskingConfig
from pragma.processing.special_tokens import SpecialTokens
from pragma.processing.tokenized_record import TokenizedRecord
from pragma.processing.vocabulary import KeyVocabulary
from pragma.schema import SchemaRegistry

IGNORE_LABEL = -100


class MaskSource(IntFlag):
    """Bitflags recording which mask source(s) selected a position (ADR 0005)."""

    NONE = 0
    TOKEN = 1
    EVENT = 2
    KEY = 4


@dataclass(frozen=True)
class RecordMaskPlan:
    """Masking result for one record's event value tokens.

    Every sequence here is aligned with the record's flattened event-token
    order: `event.value_ids()` concatenated across `record.events`, in
    order — the same order `PragmaCollator` already builds its packed event
    buffers in.
    """

    input_value_ids: tuple[int, ...]
    labels: tuple[int, ...]
    origin: tuple[int, ...]
    """`MaskSource` bitflags per position; `0` (`MaskSource.NONE`) if not selected."""


def _record_rng(seed: int, entity_id: str, epoch: int) -> random.Random:
    """Deterministic per-(seed, entity, epoch) RNG — same entity+epoch always
    reproduces the same mask, but advancing `epoch` changes it (section 16.3)."""
    digest = hashlib.sha256(f"{seed}:{entity_id}:{epoch}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


class MaskingPlanner:
    def __init__(
        self,
        config: MaskingConfig,
        maskable_key_ids: frozenset[int],
        *,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        self.config = config
        self.maskable_key_ids = maskable_key_ids
        self.special_tokens = special_tokens or SpecialTokens()

    @classmethod
    def from_registry(
        cls, registry: SchemaRegistry, key_vocab: KeyVocabulary, config: MaskingConfig
    ) -> MaskingPlanner:
        # `key_vocab.keys()` lists semantic-key names (a KeyVocabulary method,
        # not a dict) — the SIM118 lint doesn't know that, hence the noqa.
        maskable_key_ids = {
            key_vocab.id_for(key)
            for key in key_vocab.keys()  # noqa: SIM118
            if key in registry.fields and registry.get_field(key).maskable
        }
        return cls(config, frozenset(maskable_key_ids))

    def plan_record(self, record: TokenizedRecord, *, epoch: int = 0) -> RecordMaskPlan:
        rng = _record_rng(self.config.seed, record.entity_id, epoch)

        flat_value_ids: list[int] = []
        flat_key_ids: list[int] = []
        event_bounds: list[tuple[int, int]] = []
        for event in record.events:
            start = len(flat_value_ids)
            flat_key_ids.extend(event.key_ids())
            flat_value_ids.extend(event.value_ids())
            event_bounds.append((start, len(flat_value_ids)))

        n_tokens = len(flat_value_ids)
        # A position is only eligible if its key allows masking *and* its current
        # value isn't already `[UNK]` — an out-of-vocabulary value has no real
        # class for the model to predict (its true value was already lost at
        # tokenization time), and `[UNK]`'s ID sits outside the value-vocabulary
        # slice `PragmaMLMHead` computes logits over, so treating it as a
        # legitimate target would be a genuine indexing bug, not just a
        # meaningless objective.
        eligible = [
            key_id in self.maskable_key_ids and value_id != self.special_tokens.UNK
            for key_id, value_id in zip(flat_key_ids, flat_value_ids, strict=True)
        ]
        origin = [MaskSource.NONE] * n_tokens

        # 1. Individual value-token masking: each eligible position independently.
        for i in range(n_tokens):
            if eligible[i] and rng.random() < self.config.token_mask_prob:
                origin[i] |= MaskSource.TOKEN

        # 2. Whole-event masking: select events, then mark every eligible token in them.
        for start, end in event_bounds:
            if end > start and rng.random() < self.config.event_mask_prob:
                for i in range(start, end):
                    if eligible[i]:
                        origin[i] |= MaskSource.EVENT

        # 3. Semantic-key masking: select keys, mask every eligible occurrence
        # of that key across the *entire* record's event history (ADR 0005).
        eligible_keys = sorted({flat_key_ids[i] for i in range(n_tokens) if eligible[i]})
        for key_id in eligible_keys:
            if rng.random() < self.config.key_mask_prob:
                for i in range(n_tokens):
                    if eligible[i] and flat_key_ids[i] == key_id:
                        origin[i] |= MaskSource.KEY

        labels = [IGNORE_LABEL] * n_tokens
        input_ids = list(flat_value_ids)
        for i in range(n_tokens):
            if origin[i] == MaskSource.NONE:
                continue
            if rng.random() < self.config.unk_dropout_frac:
                # Input dropout: perturbs the input but is excluded from the
                # loss (label stays -100) — ADR 0005's corrected corruption rule.
                input_ids[i] = self.special_tokens.UNK
            else:
                input_ids[i] = self.special_tokens.MASK
                labels[i] = flat_value_ids[i]

        return RecordMaskPlan(
            input_value_ids=tuple(input_ids),
            labels=tuple(labels),
            origin=tuple(int(o) for o in origin),
        )
