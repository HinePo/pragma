"""Thin CLI entry point: LoRA fine-tune a frozen checkpoint on one downstream task.

Business logic lives in `pragma.downstream`; this script only wires a base
checkpoint + tokenized shards + a label column -> `DownstreamTrainer` ->
a saved adapter directory (section 12's repo structure names this script
explicitly; section 15.2).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pandas as pd

from pragma.artifacts.checkpoint import CheckpointManager
from pragma.config import DownstreamConfig
from pragma.data.dataset import TokenizedRecordDataset
from pragma.data.storage import ParquetShardStore
from pragma.downstream import DownstreamTrainer
from pragma.modeling import PragmaConfig

REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, default=REPO_ROOT / "data" / "shards")
    parser.add_argument("--processor-dir", type=Path, default=REPO_ROOT / "data" / "processor")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--label-column", default="_downstream_is_escalating_spender")
    parser.add_argument("--base-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--base-checkpoint-name", default="final")
    parser.add_argument("--adapter-out-dir", type=Path, required=True)
    parser.add_argument("--downstream-config", type=Path, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    args = parser.parse_args(argv)

    manager = CheckpointManager(args.base_checkpoint_dir)
    manifest = manager.read_manifest(args.base_checkpoint_name)
    pragma_config = PragmaConfig.from_dict(manifest["pragma_config"])

    downstream_config = (
        DownstreamConfig.load(args.downstream_config)
        if args.downstream_config
        else DownstreamConfig()
    )
    overrides: dict[str, object] = {}
    if args.lora_r is not None:
        overrides["lora_r"] = args.lora_r
    if args.lora_alpha is not None:
        overrides["lora_alpha"] = args.lora_alpha
    if args.n_epochs is not None:
        overrides["n_epochs"] = args.n_epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate
    if overrides:
        downstream_config = dataclasses.replace(downstream_config, **overrides)

    store = ParquetShardStore()
    train_dataset = TokenizedRecordDataset.from_store(args.shards_dir, store, split="train")
    val_dataset = TokenizedRecordDataset.from_store(args.shards_dir, store, split="val")

    profile_df = pd.read_parquet(args.raw_dir / "profile_state.parquet")
    labels = dict(zip(profile_df["entity_id"], profile_df[args.label_column], strict=True))

    trainer = DownstreamTrainer(
        train_dataset,
        labels,
        pragma_config,
        downstream_config,
        base_checkpoint_dir=args.base_checkpoint_dir,
        base_checkpoint_name=args.base_checkpoint_name,
        processor_dir=args.processor_dir,
        val_dataset=val_dataset,
    )
    history = trainer.train()
    for m in history:
        print(
            f"epoch {m.epoch}: train_loss={m.train_loss:.4f} "
            f"val_loss={m.val_loss} val_auc={m.val_auc}"
        )

    adapter_path = trainer.save_adapter(args.adapter_out_dir)
    print(f"Adapter written to {adapter_path}")


if __name__ == "__main__":
    main()
