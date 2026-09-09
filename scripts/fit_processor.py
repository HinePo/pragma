"""Thin CLI entry point: fit `PragmaProcessor` on the train-split of data/raw/.

Business logic lives in `pragma.processing` and `pragma.data.records`; this
script only wires config -> record building -> fitting -> bundle + report.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pandas as pd

from pragma.config import ProcessorConfig
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig, drop_zero_event_records
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processor")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "processor" / "default.json",
        help="ProcessorConfig JSON file (configs/processor/).",
    )
    parser.add_argument("--n-numeric-buckets", type=int, default=None)
    parser.add_argument("--bpe-vocab-size", type=int, default=None)
    args = parser.parse_args()

    events_df = pd.read_parquet(args.raw_dir / "events.parquet")
    profile_df = pd.read_parquet(args.raw_dir / "profile_state.parquet")

    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    records, n_dropped = drop_zero_event_records(records)
    print(f"Dropped {n_dropped} zero-event records before fitting (ADR 0014)")

    config = ProcessorConfig.load(args.config)
    overrides = {}
    if args.n_numeric_buckets is not None:
        overrides["n_numeric_buckets"] = args.n_numeric_buckets
    if args.bpe_vocab_size is not None:
        overrides["bpe_vocab_size"] = args.bpe_vocab_size
    if overrides:
        config = dataclasses.replace(config, **overrides)

    processor = PragmaProcessor(registry, config)
    report = processor.fit(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    processor.save(args.out_dir)
    (args.out_dir / "fit_report.json").write_text(
        json.dumps(dataclasses.asdict(report), indent=2, default=str)
    )

    print(f"Fitted processor on {report.n_train_records} training records")
    print(f"Total vocabulary size: {report.total_vocab_size}")
    print(f"Text OOV rate: {report.text_oov_rate:.4f}")
    print(f"Bundle written to {args.out_dir}")


if __name__ == "__main__":
    main()
