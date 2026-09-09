"""`TokenizedRecord`: stable serialized representation of one processed `EvaluationRecord`.

Implementation plan, section 11.2. Nested per-event structure is kept here;
flattening into packed buffers with cumulative offsets is a Phase 3 concern
(`PragmaBatch`/`PragmaCollator`), not the processor's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FieldTokens:
    """Token IDs for one key/value field, possibly multi-token (BPE text)."""

    key_ids: tuple[int, ...]
    value_ids: tuple[int, ...]
    within_field_pos: tuple[int, ...]

    def __post_init__(self) -> None:
        if not (len(self.key_ids) == len(self.value_ids) == len(self.within_field_pos)):
            raise ValueError("key_ids, value_ids, and within_field_pos must have equal length")


@dataclass(frozen=True)
class TokenizedEvent:
    """Token fields for one event, plus its history/calendar temporal features."""

    event_id: str
    fields: tuple[FieldTokens, ...]
    time_to_latest: float
    calendar_features: tuple[float, float, float, float, float, float]
    truncated: bool = False

    def key_ids(self) -> list[int]:
        return [tid for f in self.fields for tid in f.key_ids]

    def value_ids(self) -> list[int]:
        return [tid for f in self.fields for tid in f.value_ids]

    def within_field_pos(self) -> list[int]:
        return [p for f in self.fields for p in f.within_field_pos]

    def field_lengths(self) -> list[int]:
        """Token count per field, in order — recovers `FieldTokens` boundaries
        from the flattened `key_ids()`/`value_ids()` buffers (used by the Arrow
        shard format, section 10.1's nested-list columns)."""
        return [len(f.key_ids) for f in self.fields]


@dataclass(frozen=True)
class TokenizedRecord:
    """One processed `EvaluationRecord`, ready for masking/batching (Phase 3+)."""

    entity_id: str
    split: str
    profile_fields: tuple[FieldTokens, ...]
    profile_time_coords: tuple[float, ...]
    """One temporal RoPE coordinate per profile field/token, aligned with `profile_fields`."""
    events: tuple[TokenizedEvent, ...]
    n_events_total: int
    n_events_kept: int
    events_truncated: bool = False

    def profile_key_ids(self) -> list[int]:
        return [tid for f in self.profile_fields for tid in f.key_ids]

    def profile_value_ids(self) -> list[int]:
        return [tid for f in self.profile_fields for tid in f.value_ids]

    def profile_within_field_pos(self) -> list[int]:
        return [p for f in self.profile_fields for p in f.within_field_pos]

    def profile_field_lengths(self) -> list[int]:
        """Token count per profile field, in order — recovers `FieldTokens` boundaries
        from the flattened `profile_*_ids()` buffers (used by the Arrow shard format)."""
        return [len(f.key_ids) for f in self.profile_fields]

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "split": self.split,
            "profile_fields": [
                {
                    "key_ids": list(f.key_ids),
                    "value_ids": list(f.value_ids),
                    "within_field_pos": list(f.within_field_pos),
                }
                for f in self.profile_fields
            ],
            "profile_time_coords": list(self.profile_time_coords),
            "events": [
                {
                    "event_id": e.event_id,
                    "fields": [
                        {
                            "key_ids": list(f.key_ids),
                            "value_ids": list(f.value_ids),
                            "within_field_pos": list(f.within_field_pos),
                        }
                        for f in e.fields
                    ],
                    "time_to_latest": e.time_to_latest,
                    "calendar_features": list(e.calendar_features),
                    "truncated": e.truncated,
                }
                for e in self.events
            ],
            "n_events_total": self.n_events_total,
            "n_events_kept": self.n_events_kept,
            "events_truncated": self.events_truncated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenizedRecord:
        def _field_tokens(f: dict[str, Any]) -> FieldTokens:
            return FieldTokens(
                key_ids=tuple(f["key_ids"]),
                value_ids=tuple(f["value_ids"]),
                within_field_pos=tuple(f["within_field_pos"]),
            )

        return cls(
            entity_id=data["entity_id"],
            split=data["split"],
            profile_fields=tuple(_field_tokens(f) for f in data["profile_fields"]),
            profile_time_coords=tuple(data["profile_time_coords"]),
            events=tuple(
                TokenizedEvent(
                    event_id=e["event_id"],
                    fields=tuple(_field_tokens(f) for f in e["fields"]),
                    time_to_latest=e["time_to_latest"],
                    calendar_features=tuple(e["calendar_features"]),
                    truncated=e["truncated"],
                )
                for e in data["events"]
            ),
            n_events_total=data["n_events_total"],
            n_events_kept=data["n_events_kept"],
            events_truncated=data["events_truncated"],
        )
