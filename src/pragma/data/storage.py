"""Durable tokenized-record storage (implementation plan, section 10.1).

Replaces the Phase 2 JSONL placeholder (`scripts/tokenize_shards.py`'s
original format) with the nested-list Parquet/Arrow shard format the plan
actually specifies. `PragmaRecordStore` is an interface so a distributed
object-store backend can later expose the same record semantics as this
local `ParquetShardStore` without downstream code changing (section 10.1).

Per section 9.3, shards are bucketed by coarse event-count ranges rather than
one shard per exact event count — cheap to build now, and compatible with
finer-grained sharding later since the bucket boundaries are just a write-time
grouping, not part of the `TokenizedRecord` contract.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from pragma.processing.tokenized_record import FieldTokens, TokenizedEvent, TokenizedRecord

MANIFEST_FILE = "manifest.json"

# Event-count bucket edges: [0, edges[0]) is one bucket, [edges[0], edges[1])
# the next, etc., with everything >= edges[-1] as the final "long tail" bucket.
DEFAULT_BUCKET_EDGES: tuple[int, ...] = (1, 50, 500)

_ARROW_SCHEMA = pa.schema(
    [
        ("entity_id", pa.string()),
        ("split", pa.string()),
        ("profile_key_ids", pa.list_(pa.int64())),
        ("profile_value_ids", pa.list_(pa.int64())),
        ("profile_within_field_pos", pa.list_(pa.int64())),
        ("profile_field_lengths", pa.list_(pa.int64())),
        ("profile_time_coords", pa.list_(pa.float64())),
        ("n_events_total", pa.int64()),
        ("n_events_kept", pa.int64()),
        ("events_truncated", pa.bool_()),
        ("event_ids", pa.list_(pa.string())),
        ("event_key_ids", pa.list_(pa.list_(pa.int64()))),
        ("event_value_ids", pa.list_(pa.list_(pa.int64()))),
        ("event_within_field_pos", pa.list_(pa.list_(pa.int64()))),
        ("event_field_lengths", pa.list_(pa.list_(pa.int64()))),
        ("event_time_to_latest", pa.list_(pa.float64())),
        ("event_calendar_features", pa.list_(pa.list_(pa.float64()))),
        ("event_truncated", pa.list_(pa.bool_())),
    ]
)


def _lengths_to_boundaries(lengths: list[int]) -> list[int]:
    boundaries = [0]
    for n in lengths:
        boundaries.append(boundaries[-1] + n)
    return boundaries


def _split_by_lengths(flat: list[int], lengths: list[int]) -> list[list[int]]:
    boundaries = _lengths_to_boundaries(lengths)
    return [flat[boundaries[i] : boundaries[i + 1]] for i in range(len(lengths))]


def _record_to_row(record: TokenizedRecord) -> dict[str, object]:
    """Flatten one `TokenizedRecord` into the Arrow row shape.

    Field-token grouping (which tokens belong to the same `FieldTokens`) isn't
    representable as a plain nested list, so it's recovered from a parallel
    `*_field_lengths` column instead of being stored directly.
    """
    return {
        "entity_id": record.entity_id,
        "split": record.split,
        "profile_key_ids": record.profile_key_ids(),
        "profile_value_ids": record.profile_value_ids(),
        "profile_within_field_pos": record.profile_within_field_pos(),
        "profile_field_lengths": record.profile_field_lengths(),
        "profile_time_coords": list(record.profile_time_coords),
        "n_events_total": record.n_events_total,
        "n_events_kept": record.n_events_kept,
        "events_truncated": record.events_truncated,
        "event_ids": [e.event_id for e in record.events],
        "event_key_ids": [e.key_ids() for e in record.events],
        "event_value_ids": [e.value_ids() for e in record.events],
        "event_within_field_pos": [e.within_field_pos() for e in record.events],
        "event_field_lengths": [e.field_lengths() for e in record.events],
        "event_time_to_latest": [e.time_to_latest for e in record.events],
        "event_calendar_features": [list(e.calendar_features) for e in record.events],
        "event_truncated": [e.truncated for e in record.events],
    }


def _fields_from_flat(
    key_ids: list[int], value_ids: list[int], within_field_pos: list[int], lengths: list[int]
) -> tuple[FieldTokens, ...]:
    key_groups = _split_by_lengths(key_ids, lengths)
    value_groups = _split_by_lengths(value_ids, lengths)
    pos_groups = _split_by_lengths(within_field_pos, lengths)
    return tuple(
        FieldTokens(key_ids=tuple(k), value_ids=tuple(v), within_field_pos=tuple(p))
        for k, v, p in zip(key_groups, value_groups, pos_groups, strict=True)
    )


def _row_to_record(row: dict[str, object]) -> TokenizedRecord:
    profile_fields = _fields_from_flat(
        row["profile_key_ids"],  # type: ignore[arg-type]
        row["profile_value_ids"],  # type: ignore[arg-type]
        row["profile_within_field_pos"],  # type: ignore[arg-type]
        row["profile_field_lengths"],  # type: ignore[arg-type]
    )

    events = []
    event_ids = row["event_ids"]
    for i in range(len(event_ids)):  # type: ignore[arg-type]
        fields = _fields_from_flat(
            row["event_key_ids"][i],  # type: ignore[index]
            row["event_value_ids"][i],  # type: ignore[index]
            row["event_within_field_pos"][i],  # type: ignore[index]
            row["event_field_lengths"][i],  # type: ignore[index]
        )
        events.append(
            TokenizedEvent(
                event_id=event_ids[i],  # type: ignore[index]
                fields=fields,
                time_to_latest=row["event_time_to_latest"][i],  # type: ignore[index]
                calendar_features=tuple(row["event_calendar_features"][i]),  # type: ignore[index]
                truncated=row["event_truncated"][i],  # type: ignore[index]
            )
        )

    return TokenizedRecord(
        entity_id=row["entity_id"],  # type: ignore[arg-type]
        split=row["split"],  # type: ignore[arg-type]
        profile_fields=profile_fields,
        profile_time_coords=tuple(row["profile_time_coords"]),  # type: ignore[arg-type]
        events=tuple(events),
        n_events_total=row["n_events_total"],  # type: ignore[arg-type]
        n_events_kept=row["n_events_kept"],  # type: ignore[arg-type]
        events_truncated=row["events_truncated"],  # type: ignore[arg-type]
    )


class PragmaRecordStore(ABC):
    """Storage interface for tokenized records (section 10.1).

    A local Parquet backend and a future distributed object-store backend
    both implement this; downstream code (the dataset/collator) only ever
    talks to this interface, never to Parquet or a specific filesystem.
    """

    @abstractmethod
    def write_shard(self, records: Sequence[TokenizedRecord], path: Path) -> None: ...

    @abstractmethod
    def read_shard(self, path: Path) -> list[TokenizedRecord]: ...


class ParquetShardStore(PragmaRecordStore):
    """Local Parquet/Arrow implementation with nested-list columns."""

    def write_shard(self, records: Sequence[TokenizedRecord], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [_record_to_row(r) for r in records]
        columns = {name: [row[name] for row in rows] for name in _ARROW_SCHEMA.names}
        table = pa.table(columns, schema=_ARROW_SCHEMA)
        pq.write_table(table, path)

    def read_shard(self, path: Path) -> list[TokenizedRecord]:
        table = pq.read_table(path, schema=_ARROW_SCHEMA)
        rows = table.to_pylist()
        return [_row_to_record(row) for row in rows]


@dataclass(frozen=True)
class ShardInfo:
    """One entry in the shard index (section 10.1)."""

    path: str
    split: str
    length_bucket: str
    n_records: int
    n_events: int
    n_value_tokens: int
    sha256: str


@dataclass(frozen=True)
class DataManifest:
    """Dataset manifest (section 10.1): schema/processor identity, shards, lineage."""

    schema_version: int
    processor_train_fingerprint: str | None
    created_at: str
    total_records: int
    shards: list[ShardInfo]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "processor_train_fingerprint": self.processor_train_fingerprint,
            "created_at": self.created_at,
            "total_records": self.total_records,
            "shards": [asdict(s) for s in self.shards],
        }

    def save(self, out_dir: Path) -> None:
        (out_dir / MANIFEST_FILE).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, out_dir: Path) -> DataManifest:
        data = json.loads((out_dir / MANIFEST_FILE).read_text())
        return cls(
            schema_version=data["schema_version"],
            processor_train_fingerprint=data["processor_train_fingerprint"],
            created_at=data["created_at"],
            total_records=data["total_records"],
            shards=[ShardInfo(**s) for s in data["shards"]],
        )


def event_count_bucket(n_events: int, edges: Sequence[int] = DEFAULT_BUCKET_EDGES) -> str:
    """Coarse length-bucket label for one record's kept event count (section 9.3)."""
    lower = 0
    for edge in edges:
        if n_events < edge:
            return f"{lower}-{edge - 1}"
        lower = edge
    return f"{lower}+"


def write_dataset(
    records: Sequence[TokenizedRecord],
    out_dir: Path,
    store: PragmaRecordStore,
    *,
    processor_train_fingerprint: str | None = None,
    bucket_edges: Sequence[int] = DEFAULT_BUCKET_EDGES,
) -> DataManifest:
    """Write `records` as one shard per (split, event-count bucket) and a manifest."""
    groups: dict[tuple[str, str], list[TokenizedRecord]] = {}
    for record in records:
        bucket = event_count_bucket(len(record.events), bucket_edges)
        groups.setdefault((record.split, bucket), []).append(record)

    out_dir.mkdir(parents=True, exist_ok=True)
    shard_infos = []
    for (split, bucket), group_records in sorted(groups.items()):
        shard_name = f"{split}_{bucket}.parquet"
        shard_path = out_dir / shard_name
        store.write_shard(group_records, shard_path)

        n_events = sum(len(r.events) for r in group_records)
        n_value_tokens = sum(
            len(r.profile_value_ids()) + sum(len(e.value_ids()) for e in r.events)
            for r in group_records
        )
        shard_infos.append(
            ShardInfo(
                path=shard_name,
                split=split,
                length_bucket=bucket,
                n_records=len(group_records),
                n_events=n_events,
                n_value_tokens=n_value_tokens,
                sha256=hashlib.sha256(shard_path.read_bytes()).hexdigest(),
            )
        )

    manifest = DataManifest(
        schema_version=1,
        processor_train_fingerprint=processor_train_fingerprint,
        created_at=datetime.now(UTC).isoformat(),
        total_records=len(records),
        shards=shard_infos,
    )
    manifest.save(out_dir)
    return manifest


def read_dataset(
    out_dir: Path, store: PragmaRecordStore, *, split: str | None = None
) -> list[TokenizedRecord]:
    """Read every shard (optionally filtered to one split) listed in `out_dir`'s manifest."""
    manifest = DataManifest.load(out_dir)
    records: list[TokenizedRecord] = []
    for shard in manifest.shards:
        if split is not None and shard.split != split:
            continue
        shard_path = out_dir / shard.path
        actual_checksum = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        if actual_checksum != shard.sha256:
            raise ValueError(f"checksum mismatch for shard '{shard.path}' — corrupt or stale")
        records.extend(store.read_shard(shard_path))
    return records
