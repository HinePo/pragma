"""`PragmaBatch`: the canonical packed batch contract (implementation plan, section 9.1; ADR 0004).

Flat token/event buffers plus cumulative offsets, never padded tensors —
padding is reserved for `PaddedAttentionBackend`'s small correctness-oracle
tests (Phase 5+), not real training. The offsets and mappings here are
safety controls, not just performance metadata: they are what a future
attention backend uses to guarantee one event's tokens never attend past
that event's boundary, and one customer's history never attends into
another customer's (ADR 0004's isolation requirement, tested at the model
level in Phase 5/7 — `PragmaBatch.validate()` below only checks that the
*batch itself* is internally consistent, e.g. offsets are monotonic and
mappings are in range).

`event_mlm_labels` and `event_mask_origin` are populated by an optional
`MaskingPlanner` passed to `PragmaCollator` (Phase 4, section 8; ADR 0005).
Without one, every position is left unmasked: `event_mlm_labels` is all
`-100` and `event_mask_origin` is all `MaskSource.NONE` (0) — the same shape
either way, so downstream code never needs to branch on whether masking was
applied.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pragma.masking.planner import MaskingPlanner
from pragma.processing.tokenized_record import TokenizedRecord

IGNORE_INDEX = -100


@dataclass
class PragmaBatch:
    entity_ids: list[str]
    n_records: int

    # Profile tokens: one flat sequence per batch, `[USR]` prepended per
    # record is a model-embedding-layer concern (Phase 5), not stored here.
    profile_key_ids: torch.Tensor
    profile_value_ids: torch.Tensor
    profile_within_field_pos: torch.Tensor
    profile_time_coords: torch.Tensor
    profile_token_to_record: torch.Tensor
    """Which record (batch-local index) each profile token belongs to."""

    # Event tokens: flat across every event in the batch. `event_cu_seqlens`
    # bounds each event's own tokens (Event Encoder isolation); `event_to_record`
    # groups whole events back into their owning record (History Encoder
    # isolation).
    event_key_ids: torch.Tensor
    event_value_ids: torch.Tensor
    event_within_field_pos: torch.Tensor
    event_mlm_labels: torch.Tensor
    event_mask_origin: torch.Tensor
    """`MaskSource` bitflags per event value token (0 == never selected)."""
    event_cu_seqlens: torch.Tensor
    """Length `n_events + 1`; event `i`'s tokens are `[cu_seqlens[i], cu_seqlens[i+1])`."""
    event_to_record: torch.Tensor
    """Length `n_events`; batch-local record index each event belongs to."""
    event_time_to_latest: torch.Tensor
    event_calendar_features: torch.Tensor

    history_cu_seqlens: torch.Tensor
    """Length `n_records + 1`; record `i`'s events are `[cu_seqlens[i], cu_seqlens[i+1])`."""

    @property
    def n_events(self) -> int:
        return self.event_to_record.shape[0]

    @property
    def n_profile_tokens(self) -> int:
        return self.profile_key_ids.shape[0]

    @property
    def n_event_tokens(self) -> int:
        return self.event_key_ids.shape[0]

    def to(self, device: torch.device | str) -> PragmaBatch:
        """Moves every tensor field to `device` in place and returns `self` — used by
        `PretrainingEngine` to move a batch onto whatever device `Accelerate` selected
        (section 13.1; device selection itself is never hardcoded here or anywhere
        in `pragma.training` — ADR 0011)."""
        for name in (
            "profile_key_ids",
            "profile_value_ids",
            "profile_within_field_pos",
            "profile_time_coords",
            "profile_token_to_record",
            "event_key_ids",
            "event_value_ids",
            "event_within_field_pos",
            "event_mlm_labels",
            "event_mask_origin",
            "event_cu_seqlens",
            "event_to_record",
            "event_time_to_latest",
            "event_calendar_features",
            "history_cu_seqlens",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def validate(self) -> None:
        """Assert the batch's own boundaries/mappings are internally consistent.

        This is the Phase 3 exit-gate check ("cross-event and cross-record
        boundaries are validated") at the data-contract level — it does not
        run a model, it only checks that the offsets a future attention
        backend would rely on are well-formed.
        """
        if self.event_mlm_labels.shape[0] != self.n_event_tokens:
            raise ValueError("event_mlm_labels must have length n_event_tokens")
        if self.event_mask_origin.shape[0] != self.n_event_tokens:
            raise ValueError("event_mask_origin must have length n_event_tokens")
        if self.history_cu_seqlens.shape[0] != self.n_records + 1:
            raise ValueError("history_cu_seqlens must have length n_records + 1")
        if self.event_cu_seqlens.shape[0] != self.n_events + 1:
            raise ValueError("event_cu_seqlens must have length n_events + 1")
        if not bool(torch.all(self.history_cu_seqlens[:-1] <= self.history_cu_seqlens[1:])):
            raise ValueError("history_cu_seqlens must be non-decreasing")
        if not bool(torch.all(self.event_cu_seqlens[:-1] <= self.event_cu_seqlens[1:])):
            raise ValueError("event_cu_seqlens must be non-decreasing")
        if int(self.history_cu_seqlens[-1]) != self.n_events:
            raise ValueError("history_cu_seqlens must span exactly n_events")
        if int(self.event_cu_seqlens[-1]) != self.n_event_tokens:
            raise ValueError("event_cu_seqlens must span exactly n_event_tokens")

        if self.n_events > 0:
            expected_to_record = torch.repeat_interleave(
                torch.arange(self.n_records),
                self.history_cu_seqlens[1:] - self.history_cu_seqlens[:-1],
            )
            if not torch.equal(self.event_to_record, expected_to_record):
                raise ValueError("event_to_record does not match history_cu_seqlens grouping")

        if self.n_profile_tokens > 0:
            if int(self.profile_token_to_record.max()) >= self.n_records:
                raise ValueError("profile_token_to_record has an out-of-range record index")
            head = self.profile_token_to_record[:-1]
            tail = self.profile_token_to_record[1:]
            if not bool(torch.all(head <= tail)):
                raise ValueError("profile_token_to_record must be non-decreasing (record-major)")


class PragmaCollator:
    """Packs a list of `TokenizedRecord`s into one `PragmaBatch`.

    Without a `masking_planner`, no masking is applied: `event_mlm_labels`
    and `event_mask_origin` are all zero/`IGNORE_INDEX`. This keeps the
    packing/offset logic — the part that must be correct for attention
    isolation — independently testable from masking correctness (Phase 3's
    tests use a plain `PragmaCollator()`; Phase 4's masking tests use one
    with a `masking_planner`).

    `set_epoch` mirrors `DistributedSampler.set_epoch`: masking is
    deterministic per `(seed, entity_id, epoch)`, so advancing the epoch
    here reshuffles every record's mask on the next `__call__` (section
    16.3's "changes across epochs when expected").
    """

    def __init__(self, masking_planner: MaskingPlanner | None = None) -> None:
        self.masking_planner = masking_planner
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __call__(self, records: list[TokenizedRecord]) -> PragmaBatch:
        n_records = len(records)

        profile_key_ids: list[int] = []
        profile_value_ids: list[int] = []
        profile_within_field_pos: list[int] = []
        profile_time_coords: list[float] = []
        profile_token_to_record: list[int] = []

        event_key_ids: list[int] = []
        event_value_ids: list[int] = []
        event_within_field_pos: list[int] = []
        event_mlm_labels: list[int] = []
        event_mask_origin: list[int] = []
        event_cu_seqlens: list[int] = [0]
        event_to_record: list[int] = []
        event_time_to_latest: list[float] = []
        event_calendar_features: list[list[float]] = []
        history_cu_seqlens: list[int] = [0]

        for record_idx, record in enumerate(records):
            profile_key_ids.extend(record.profile_key_ids())
            profile_value_ids.extend(record.profile_value_ids())
            profile_within_field_pos.extend(record.profile_within_field_pos())
            profile_time_coords.extend(record.profile_time_coords)
            profile_token_to_record.extend([record_idx] * len(record.profile_key_ids()))

            if self.masking_planner is not None:
                mask_plan = self.masking_planner.plan_record(record, epoch=self._epoch)
                record_value_ids = mask_plan.input_value_ids
                record_labels = mask_plan.labels
                record_origin = mask_plan.origin
            else:
                record_value_ids = tuple(tid for e in record.events for tid in e.value_ids())
                record_labels = (IGNORE_INDEX,) * len(record_value_ids)
                record_origin = (0,) * len(record_value_ids)

            flat_idx = 0
            for event in record.events:
                n_tokens = len(event.key_ids())
                event_key_ids.extend(event.key_ids())
                event_value_ids.extend(record_value_ids[flat_idx : flat_idx + n_tokens])
                event_mlm_labels.extend(record_labels[flat_idx : flat_idx + n_tokens])
                event_mask_origin.extend(record_origin[flat_idx : flat_idx + n_tokens])
                event_within_field_pos.extend(event.within_field_pos())
                event_cu_seqlens.append(event_cu_seqlens[-1] + n_tokens)
                event_to_record.append(record_idx)
                event_time_to_latest.append(event.time_to_latest)
                event_calendar_features.append(list(event.calendar_features))
                flat_idx += n_tokens

            history_cu_seqlens.append(history_cu_seqlens[-1] + len(record.events))

        batch = PragmaBatch(
            entity_ids=[r.entity_id for r in records],
            n_records=n_records,
            profile_key_ids=torch.tensor(profile_key_ids, dtype=torch.long),
            profile_value_ids=torch.tensor(profile_value_ids, dtype=torch.long),
            profile_within_field_pos=torch.tensor(profile_within_field_pos, dtype=torch.long),
            profile_time_coords=torch.tensor(profile_time_coords, dtype=torch.float32),
            profile_token_to_record=torch.tensor(profile_token_to_record, dtype=torch.long),
            event_key_ids=torch.tensor(event_key_ids, dtype=torch.long),
            event_value_ids=torch.tensor(event_value_ids, dtype=torch.long),
            event_within_field_pos=torch.tensor(event_within_field_pos, dtype=torch.long),
            event_mlm_labels=torch.tensor(event_mlm_labels, dtype=torch.long),
            event_mask_origin=torch.tensor(event_mask_origin, dtype=torch.long),
            event_cu_seqlens=torch.tensor(event_cu_seqlens, dtype=torch.long),
            event_to_record=torch.tensor(event_to_record, dtype=torch.long),
            event_time_to_latest=torch.tensor(event_time_to_latest, dtype=torch.float32),
            event_calendar_features=(
                torch.tensor(event_calendar_features, dtype=torch.float32)
                if event_calendar_features
                else torch.zeros((0, 6), dtype=torch.float32)
            ),
            history_cu_seqlens=torch.tensor(history_cu_seqlens, dtype=torch.long),
        )
        batch.validate()
        return batch
