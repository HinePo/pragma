"""Thin CLI entry point: extract frozen `[USR]`/last-`[EVT]`/concat embeddings.

Business logic lives in `pragma.evaluation.embeddings`; this script only
wires a checkpoint + a shard split -> `EmbeddingExtractor` -> Parquet on
disk, so a later `run_probe.py` call (or `mlflow`/notebook analysis) never
needs to re-run the model. Rebuilds only the model (not the optimizer or
scheduler) from the checkpoint's own `PragmaConfig`, via a fresh
`Accelerator` - the same mechanism `PretrainingEngine.resume` uses
internally, minus the optimizer/scheduler restore this doesn't need.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from accelerate import Accelerator

from pragma.artifacts.checkpoint import CheckpointManager
from pragma.data.dataset import TokenizedRecordDataset
from pragma.data.storage import ParquetShardStore
from pragma.evaluation import EmbeddingExtractor
from pragma.modeling import PragmaConfig
from pragma.modeling.model import PragmaForMaskedModeling

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, default=REPO_ROOT / "data" / "shards")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="final")
    parser.add_argument(
        "--split",
        default="val",
        help="Shard split to extract embeddings for (e.g. 'train', 'val', or 'all').",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    manager = CheckpointManager(args.checkpoint_dir)
    manifest = manager.read_manifest(args.checkpoint_name)
    pragma_config = PragmaConfig.from_dict(manifest["pragma_config"])

    accelerator = Accelerator()
    model = accelerator.prepare(PragmaForMaskedModeling(pragma_config))
    accelerator.load_state(str(args.checkpoint_dir / args.checkpoint_name))
    backbone = accelerator.unwrap_model(model).pragma

    split = None if args.split == "all" else args.split
    dataset = TokenizedRecordDataset.from_store(args.shards_dir, ParquetShardStore(), split=split)

    bundle = EmbeddingExtractor(
        backbone, batch_size=args.batch_size, device=accelerator.device
    ).extract(dataset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(
        {
            "entity_id": bundle.entity_ids,
            "usr": bundle.usr.tolist(),
            "last_event": bundle.last_event.tolist(),
            "concat": bundle.concat.tolist(),
        }
    )
    out_path = args.out_dir / f"{args.split}_embeddings.parquet"
    out_df.to_parquet(out_path, index=False)
    print(f"Wrote {len(out_df)} embeddings to {out_path}")


if __name__ == "__main__":
    main()
