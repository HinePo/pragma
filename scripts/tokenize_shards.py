"""Thin CLI entry point: offline-tokenize data/raw/ into versioned Parquet shards.

Writes one Parquet shard per (split, event-count bucket) plus a data
manifest (schema/processor identity, row/event/token counts, checksums) via
`pragma.data.storage.write_dataset` and `ParquetShardStore` — the section
10.1 nested-list Arrow format that replaced Phase 2's JSONL placeholder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pragma.data.records import PointInTimeRecordBuilder, SplitConfig, drop_zero_event_records
from pragma.data.storage import ParquetShardStore, write_dataset
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--processor-dir", type=Path, default=REPO_ROOT / "data" / "processor")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "shards")
    args = parser.parse_args()

    events_df = pd.read_parquet(args.raw_dir / "events.parquet")
    profile_df = pd.read_parquet(args.raw_dir / "profile_state.parquet")

    registry = SchemaRegistry.default()
    processor = PragmaProcessor.load(args.processor_dir, registry)
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    records, n_dropped = drop_zero_event_records(records)
    print(f"Dropped {n_dropped} zero-event records before tokenizing (ADR 0014)")
    tokenized = [processor.transform(r) for r in records]

    manifest = write_dataset(
        tokenized,
        args.out_dir,
        ParquetShardStore(),
        processor_train_fingerprint=processor.train_fingerprint,
    )

    by_split: dict[str, list] = {}
    for shard in manifest.shards:
        by_split.setdefault(shard.split, []).append(shard)
    for split, shards in sorted(by_split.items()):
        n_records = sum(s.n_records for s in shards)
        n_value_tokens = sum(s.n_value_tokens for s in shards)
        print(
            f"{split}: {n_records} records across {len(shards)} shards, "
            f"{n_value_tokens} value tokens"
        )
    print(f"Manifest written to {args.out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
