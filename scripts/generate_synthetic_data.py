"""Thin CLI entry point: generate the Phase 1 synthetic corpus into data/raw/.

Business logic lives in `pragma.data.synthetic`; this script only wires
config -> generation -> validation -> disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus, validate_corpus

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-entities", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Directory to write events.parquet, profile_state.parquet, and reports.",
    )
    args = parser.parse_args()

    config = SyntheticDataConfig(n_entities=args.n_entities, seed=args.seed)
    events_df, profile_df, manifest = generate_synthetic_corpus(config)
    report = validate_corpus(events_df, profile_df)

    if not report["passed"]:
        raise SystemExit(
            f"Generated corpus failed validation: "
            f"{report['n_schema_issues']} schema issues, "
            f"{report['n_leakage_issues']} leakage issues. "
            f"See report for details:\n{json.dumps(report, indent=2, default=str)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_parquet(args.out_dir / "events.parquet", index=False)
    profile_df.to_parquet(args.out_dir / "profile_state.parquet", index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (args.out_dir / "validation_report.json").write_text(json.dumps(report, indent=2, default=str))

    print(
        f"Wrote {manifest['n_events']} events and "
        f"{manifest['n_entities']} entities to {args.out_dir}"
    )
    print(f"Validation passed: {report['passed']}")


if __name__ == "__main__":
    main()
