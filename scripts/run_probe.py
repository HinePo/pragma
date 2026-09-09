"""Thin CLI entry point: linear probes + conventional baselines -> one report.

Business logic lives in `pragma.evaluation`; this script only wires
`extract_embeddings.py`'s saved embeddings + the raw profile/event tables ->
`LinearProbeRunner` + `run_baselines` + `build_comparison_report` -> a CSV on
disk (section 14.2/14.3, ADR 0012).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pragma.data.records import PointInTimeRecordBuilder, SplitConfig, drop_zero_event_records
from pragma.evaluation import (
    EmbeddingBundle,
    LinearProbeRunner,
    build_aggregated_features,
    build_comparison_report,
    run_baselines,
)
from pragma.schema import SchemaRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_bundle(path: Path) -> EmbeddingBundle:
    df = pd.read_parquet(path)
    return EmbeddingBundle(
        entity_ids=df["entity_id"].tolist(),
        usr=torch.from_numpy(np.stack(df["usr"].to_numpy())).float(),
        last_event=torch.from_numpy(np.stack(df["last_event"].to_numpy())).float(),
        concat=torch.from_numpy(np.stack(df["concat"].to_numpy())).float(),
    )


def _concat_bundles(train: EmbeddingBundle, val: EmbeddingBundle) -> EmbeddingBundle:
    return EmbeddingBundle(
        entity_ids=train.entity_ids + val.entity_ids,
        usr=torch.cat([train.usr, val.usr], dim=0),
        last_event=torch.cat([train.last_event, val.last_event], dim=0),
        concat=torch.cat([train.concat, val.concat], dim=0),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        required=True,
        help="Directory with train_embeddings.parquet and val_embeddings.parquet "
        "(scripts/extract_embeddings.py --split train / --split val).",
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--label-column", default="_downstream_is_high_value")
    parser.add_argument("--checkpoint-name", default="final")
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train_bundle = _load_bundle(args.embeddings_dir / "train_embeddings.parquet")
    val_bundle = _load_bundle(args.embeddings_dir / "val_embeddings.parquet")
    bundle = _concat_bundles(train_bundle, val_bundle)
    val_entity_ids = set(val_bundle.entity_ids)

    events_df = pd.read_parquet(args.raw_dir / "events.parquet")
    profile_df = pd.read_parquet(args.raw_dir / "profile_state.parquet")
    labels = dict(zip(profile_df["entity_id"], profile_df[args.label_column], strict=True))

    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    records, _ = drop_zero_event_records(records)  # ADR 0014: no inference on zero-event customers
    records_by_entity = {r.entity_id: r for r in records if r.entity_id in set(bundle.entity_ids)}
    features_df = build_aggregated_features(list(records_by_entity.values()))

    probe_results = LinearProbeRunner(seed=args.seed).run(
        bundle, labels, val_entity_ids=val_entity_ids
    )
    baseline_results = run_baselines(
        features_df, labels, val_entity_ids=val_entity_ids, seed=args.seed
    )

    report = build_comparison_report(
        probe_results + baseline_results, checkpoint_name=args.checkpoint_name
    )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out_path, index=False)
    print(report.to_string(index=False))
    print(f"Report written to {args.out_path}")


if __name__ == "__main__":
    main()
