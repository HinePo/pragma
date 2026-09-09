"""Thin CLI entry point: run `PretrainingEngine` against tokenized shards.

Business logic lives in `pragma.training.engine` and `pragma.evaluation`;
this script only wires config -> data -> `PretrainingEngine.train()`, with an
optional periodic frozen-embedding probe (section 14.2, Phase 9's exit gate)
run between training chunks. Deferred until Phase 9 needed a real pilot-corpus
run outside a notebook (see `plans/progress.md`'s Phase 8 open items) - this
is what `accelerate launch scripts/pretrain.py ...` would eventually run on
real (multi-)GPU hardware; on this dev machine it just runs single-process
on CPU, per `Accelerate`'s own auto-detection (ADR 0011).
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import mlflow
import pandas as pd
import torch

from pragma.config import MaskingConfig, TokenBudgetConfig, TrainingConfig
from pragma.data.dataset import TokenizedRecordDataset
from pragma.data.storage import ParquetShardStore
from pragma.evaluation import extract_record_embeddings, run_linear_probe
from pragma.modeling import PragmaConfig
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry
from pragma.training import PretrainingEngine

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_probe(
    engine: PretrainingEngine, val_dataset: TokenizedRecordDataset, args: argparse.Namespace
) -> None:
    if not engine.accelerator.is_main_process:
        return
    profile_df = pd.read_parquet(args.raw_dir / "profile_state.parquet")
    labels = dict(zip(profile_df["entity_id"], profile_df[args.probe_label_column], strict=True))

    backbone = engine.accelerator.unwrap_model(engine.model).pragma
    train_entity_ids, train_embeddings = extract_record_embeddings(
        backbone, engine.train_dataset, device=engine.accelerator.device
    )
    val_entity_ids, val_embeddings = extract_record_embeddings(
        backbone, val_dataset, device=engine.accelerator.device
    )

    all_ids = train_entity_ids + val_entity_ids
    all_embeddings = torch.cat([train_embeddings, val_embeddings], dim=0)
    result = run_linear_probe(all_ids, all_embeddings, labels, val_entity_ids=set(val_entity_ids))
    print(f"  probe: auc={result.auc:.4f} (n_train={result.n_train}, n_val={result.n_val})")
    if result.auc == result.auc:  # not NaN
        mlflow.log_metrics(
            {"probe_auc": result.auc, "probe_n_train": result.n_train, "probe_n_val": result.n_val},
            step=engine.counters.epoch,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, default=REPO_ROOT / "data" / "shards")
    parser.add_argument("--processor-dir", type=Path, default=REPO_ROOT / "data" / "processor")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Only read for --probe-every-n-epochs > 0 (needs profile_state.parquet's label col).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=REPO_ROOT / "data" / "checkpoints" / "pretrain"
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=None,
        help="TrainingConfig JSON (configs/pretraining/). Defaults built-in otherwise.",
    )
    parser.add_argument(
        "--masking-config",
        type=Path,
        default=None,
        help="MaskingConfig JSON (configs/pretraining/). Defaults built-in otherwise.",
    )
    parser.add_argument("--optimizer", choices=["adamw", "muon_adamw"], default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--event-layers", type=int, default=5)
    parser.add_argument("--history-layers", type=int, default=2)
    parser.add_argument("--max-event-tokens-per-batch", type=int, default=4096)
    parser.add_argument("--max-events-per-batch", type=int, default=2048)
    parser.add_argument("--max-records-per-batch", type=int, default=256)
    parser.add_argument(
        "--probe-every-n-epochs",
        type=int,
        default=0,
        help="0 disables periodic frozen-embedding probing (section 14.2).",
    )
    parser.add_argument("--probe-label-column", default="_downstream_is_high_value")
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=None,
        help=(
            "Defaults to a persistent sqlite:///data/mlflow/mlflow.db - not "
            "TrainingConfig's bare 'sqlite:///mlflow.db' default, which would "
            "otherwise write to whatever the current working directory happens to be."
        ),
    )
    args = parser.parse_args(argv)

    registry = SchemaRegistry.default()
    processor = PragmaProcessor.load(args.processor_dir, registry)
    store = ParquetShardStore()
    train_dataset = TokenizedRecordDataset.from_store(args.shards_dir, store, split="train")
    val_dataset = TokenizedRecordDataset.from_store(args.shards_dir, store, split="val")

    pragma_config = PragmaConfig.from_processor(
        processor,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        event_layers=args.event_layers,
        history_layers=args.history_layers,
    )
    masking_config = (
        MaskingConfig.load(args.masking_config) if args.masking_config else MaskingConfig()
    )
    training_config = (
        TrainingConfig.load(args.training_config) if args.training_config else TrainingConfig()
    )
    overrides: dict[str, object] = {}
    if args.optimizer is not None:
        overrides["optimizer"] = args.optimizer
    if args.n_epochs is not None:
        overrides["n_epochs"] = args.n_epochs
    mlflow_db = REPO_ROOT / "data" / "mlflow" / "mlflow.db"
    mlflow_db.parent.mkdir(parents=True, exist_ok=True)
    overrides["mlflow_tracking_uri"] = args.mlflow_tracking_uri or f"sqlite:///{mlflow_db}"
    if overrides:
        training_config = dataclasses.replace(training_config, **overrides)
    token_budget_config = TokenBudgetConfig(
        max_event_tokens_per_batch=args.max_event_tokens_per_batch,
        max_events_per_batch=args.max_events_per_batch,
        max_records_per_batch=args.max_records_per_batch,
    )

    engine = PretrainingEngine(
        train_dataset,
        processor,
        pragma_config,
        masking_config,
        training_config,
        token_budget_config,
        processor_dir=args.processor_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    chunk = args.probe_every_n_epochs if args.probe_every_n_epochs > 0 else training_config.n_epochs
    epoch = 0
    while epoch < training_config.n_epochs:
        end_epoch = min(epoch + chunk, training_config.n_epochs)
        history = engine.train(start_epoch=epoch, end_epoch=end_epoch, val_dataset=val_dataset)
        for metrics in history:
            print(
                f"epoch {metrics.epoch}: train_loss={metrics.mean_loss:.4f} "
                f"val_loss={metrics.val_loss}"
            )
        epoch = end_epoch
        if args.probe_every_n_epochs > 0:
            _run_probe(engine, val_dataset, args)

    engine.save_checkpoint("final")
    engine.close()
    print(f"Checkpoint written to {args.checkpoint_dir / 'final'}")


if __name__ == "__main__":
    main()
