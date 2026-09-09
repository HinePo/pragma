"""Point-in-time-correct evaluation records (implementation plan, section 5).

`PointInTimeRecordBuilder` converts the raw `events_df`/`profile_df` tables
produced by `pragma.data.synthetic` (or any source conforming to the same
shape) into `EvaluationRecord` objects: one leakage-safe observation per
entity, with profile state reconstructed as of `evaluation_time` and only
events at or before that point included.

See ADR 0002 (point-in-time rules) and ADR 0008 (evaluation-point sampling,
train/val/test partitioning) for the decisions this module implements.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from pragma.schema import FieldType, SchemaRegistry

Split = str  # "train" | "val" | "test"


@dataclass(frozen=True)
class EventRecord:
    """One validated raw event belonging to a single entity."""

    entity_id: str
    event_id: str
    created_at: datetime
    event_type: str
    fields: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ProfileState:
    """Point-in-time profile snapshot: static attributes plus lifelong milestones.

    `milestones` maps a canonical milestone key (e.g. `first_topup_at`) to its
    timestamp, or `None` if it had not happened by `as_of`. Milestone values
    are already clipped to point-in-time correctness by the builder: a
    milestone whose true timestamp is after `as_of` is reported as `None`
    here, never as a future timestamp.
    """

    entity_id: str
    as_of: datetime
    attributes: dict[str, object] = field(default_factory=dict)
    milestones: dict[str, datetime | None] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRecord:
    """One pretraining observation (implementation plan, section 5.3)."""

    entity_id: str
    evaluation_time: datetime
    profile_state: ProfileState
    events_before_evaluation: tuple[EventRecord, ...]
    lifelong_events: tuple[tuple[str, datetime | None], ...]
    split: Split = "train"


def _entity_split(entity_id: str, *, train_frac: float, val_frac: float, seed: int) -> Split:
    """Deterministically assign an entity to train/val/test by hashing its ID.

    Hash-based (not row-order-based) so partitioning is stable across reruns
    and independent of table sort order.
    """
    digest = hashlib.sha256(f"{seed}:{entity_id}".encode()).hexdigest()
    # First 4 hex bytes -> a uniform float in [0, 1] used as the split cutoff.
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_frac:
        return "train"
    if bucket < train_frac + val_frac:
        return "val"
    return "test"


def _resolve_field_type(registry: SchemaRegistry, canonical_key: str) -> FieldType:
    return registry.get_field(canonical_key).field_type


def _profile_attributes_and_milestones(
    registry: SchemaRegistry,
    profile_row: dict[str, object],
    *,
    as_of: datetime,
) -> tuple[dict[str, object], dict[str, datetime | None]]:
    attributes: dict[str, object] = {}
    milestones: dict[str, datetime | None] = {}

    for key in registry.profile_fields:
        if key not in profile_row:
            continue
        value = profile_row[key]
        is_milestone = key in registry.lifelong_milestone_fields
        if is_milestone:
            if value is None or pd.isna(value):
                milestones[key] = None
            else:
                ts = value if isinstance(value, datetime) else pd.Timestamp(value).to_pydatetime()
                milestones[key] = ts if ts <= as_of else None
        else:
            attributes[key] = None if (value is None or pd.isna(value)) else value

    return attributes, milestones


def _events_before(
    entity_events: pd.DataFrame,
    registry: SchemaRegistry,
    *,
    evaluation_time: datetime,
) -> tuple[EventRecord, ...]:
    if entity_events.empty:
        return ()

    records: list[EventRecord] = []
    kept = entity_events[entity_events["created_at"] <= evaluation_time]
    # Deterministic tie-breaking per ADR 0002: sort by (created_at, event_id).
    kept = kept.sort_values(["created_at", "event_id"])

    value_columns = [
        c
        for c in kept.columns
        if c not in ("entity_id", "event_id", "created_at", "type")
        and registry.resolve(c) is not None
    ]
    for row in kept.itertuples(index=False):
        row_dict = row._asdict()
        fields = {
            c: row_dict[c]
            for c in value_columns
            if row_dict[c] is not None and not pd.isna(row_dict[c])
        }
        records.append(
            EventRecord(
                entity_id=row_dict["entity_id"],
                event_id=row_dict["event_id"],
                created_at=row_dict["created_at"],
                event_type=row_dict["type"],
                fields=fields,
            )
        )
    return tuple(records)


def drop_zero_event_records(
    records: list[EvaluationRecord],
) -> tuple[list[EvaluationRecord], int]:
    """Filter out zero-event records ahead of pretraining or inference.

    Per ADR 0014, the paper discards zero-event customers during pretraining
    and PRAGMA has no cold-start path for a customer with no history at all;
    `PointInTimeRecordBuilder.build()` still constructs a record for them
    (it must, to stay a faithful point-in-time snapshot of the raw tables —
    see `test_zero_event_entities_have_no_events_and_no_milestones`), but
    every consumer that feeds records into training or inference must drop
    them first, via this function.
    """
    kept = [r for r in records if r.events_before_evaluation]
    return kept, len(records) - len(kept)


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.8
    val_frac: float = 0.1
    seed: int = 20260101

    def __post_init__(self) -> None:
        if not (0 < self.train_frac < 1) or not (0 <= self.val_frac < 1):
            raise ValueError("train_frac and val_frac must be fractions in (0, 1)")
        if self.train_frac + self.val_frac >= 1:
            raise ValueError("train_frac + val_frac must leave a non-empty test partition")


class PointInTimeRecordBuilder:
    """Builds leakage-safe `EvaluationRecord`s from raw event/profile tables."""

    def __init__(self, registry: SchemaRegistry, split_config: SplitConfig | None = None) -> None:
        self._registry = registry
        self._split_config = split_config or SplitConfig()

    def build(self, events_df: pd.DataFrame, profile_df: pd.DataFrame) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        has_events = not events_df.empty and "created_at" in events_df.columns
        events_by_entity = {k: v for k, v in events_df.groupby("entity_id")} if has_events else {}
        empty_events = events_df.iloc[0:0] if has_events else pd.DataFrame(columns=["created_at"])

        for profile_row_ns in profile_df.itertuples(index=False):
            profile_row = profile_row_ns._asdict()
            entity_id = profile_row["entity_id"]
            entity_events = events_by_entity.get(entity_id, empty_events)

            signup_at = profile_row["signup_at"]
            if len(entity_events) > 0:
                evaluation_time = entity_events["created_at"].max()
            else:
                evaluation_time = signup_at

            attributes, milestones = _profile_attributes_and_milestones(
                self._registry, profile_row, as_of=evaluation_time
            )
            profile_state = ProfileState(
                entity_id=entity_id,
                as_of=evaluation_time,
                attributes=attributes,
                milestones=milestones,
            )

            events_before = _events_before(
                entity_events, self._registry, evaluation_time=evaluation_time
            )

            split = _entity_split(
                entity_id,
                train_frac=self._split_config.train_frac,
                val_frac=self._split_config.val_frac,
                seed=self._split_config.seed,
            )

            records.append(
                EvaluationRecord(
                    entity_id=entity_id,
                    evaluation_time=evaluation_time,
                    profile_state=profile_state,
                    events_before_evaluation=events_before,
                    lifelong_events=tuple(milestones.items()),
                    split=split,
                )
            )

        return records
