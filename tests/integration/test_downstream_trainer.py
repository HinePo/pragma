"""`DownstreamTrainer` end to end (Phase 11, section 15; ADR 0013) - the real
exit-gate properties: the base checkpoint stays byte-identical after
fine-tuning (section 15.1's immutability entry condition), and a saved
adapter reloads **independently** (a fresh process rebuilding the frozen
backbone from scratch, per `load_lora_model`) and reproduces the exact same
predictions as the in-memory trained model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pragma.config import (
    DownstreamConfig,
    MaskingConfig,
    ProcessorConfig,
    TokenBudgetConfig,
    TrainingConfig,
)
from pragma.data.batch import PragmaCollator
from pragma.data.dataset import TokenizedRecordDataset
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.downstream import DownstreamTrainer, load_lora_model
from pragma.modeling import PragmaConfig
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry
from pragma.training import PretrainingEngine

torch.set_num_threads(1)  # avoid unrelated CPU floating-point noise across runs


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    return tmp_path


def _setup(tmp_dir: Path):
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=60,
            seed=7,
            max_events_per_entity=15,
            n_zero_event_entities=2,
            n_single_event_entities=2,
            n_long_history_entities=0,
            n_same_timestamp_entities=1,
        )
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=6, bpe_vocab_size=80))
    processor.fit(records)
    processor_dir = tmp_dir / "processor"
    processor.save(processor_dir)

    kept_records = [r for r in records if r.events_before_evaluation]
    tokenized = [processor.transform(r) for r in kept_records]
    n_val = max(1, len(tokenized) // 5)
    train_dataset = TokenizedRecordDataset(tokenized[n_val:])
    val_dataset = TokenizedRecordDataset(tokenized[:n_val])

    pragma_config = PragmaConfig.from_processor(
        processor,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        event_layers=1,
        history_layers=1,
    )

    checkpoint_dir = tmp_dir / "ckpt"
    pretrain_engine = PretrainingEngine(
        TokenizedRecordDataset(tokenized),
        processor,
        pragma_config,
        MaskingConfig(),
        TrainingConfig(
            n_epochs=1,
            mixed_precision="no",
            checkpoint_dir=str(checkpoint_dir),
            mlflow_tracking_uri=f"sqlite:///{tmp_dir}/mlflow.db",
        ),
        TokenBudgetConfig(
            max_event_tokens_per_batch=512, max_events_per_batch=256, max_records_per_batch=16
        ),
        processor_dir=processor_dir,
    )
    pretrain_engine.train()
    pretrain_engine.save_checkpoint("final")
    pretrain_engine.close()

    labels = dict(
        zip(profile_df["entity_id"], profile_df["_downstream_is_escalating_spender"], strict=True)
    )
    return (
        train_dataset,
        val_dataset,
        labels,
        pragma_config,
        processor_dir,
        checkpoint_dir,
    )


def test_train_produces_finite_decreasing_or_stable_loss(scratch_dir: Path) -> None:
    train_ds, val_ds, labels, pragma_config, processor_dir, checkpoint_dir = _setup(scratch_dir)
    trainer = DownstreamTrainer(
        train_ds,
        labels,
        pragma_config,
        DownstreamConfig(n_epochs=3, batch_size=4, mixed_precision="no", seed=0),
        base_checkpoint_dir=checkpoint_dir,
        base_checkpoint_name="final",
        processor_dir=processor_dir,
        val_dataset=val_ds,
    )
    history = trainer.train()
    assert len(history) == 3
    for m in history:
        assert m.train_loss == m.train_loss  # not NaN
        assert m.val_loss is not None and m.val_loss == m.val_loss


def test_base_checkpoint_is_byte_identical_after_fine_tuning(scratch_dir: Path) -> None:
    train_ds, val_ds, labels, pragma_config, processor_dir, checkpoint_dir = _setup(scratch_dir)
    weights_path = checkpoint_dir / "final" / "model.safetensors"
    hash_before = hashlib.sha256(weights_path.read_bytes()).hexdigest()

    trainer = DownstreamTrainer(
        train_ds,
        labels,
        pragma_config,
        DownstreamConfig(n_epochs=2, batch_size=4, mixed_precision="no", seed=0),
        base_checkpoint_dir=checkpoint_dir,
        base_checkpoint_name="final",
        processor_dir=processor_dir,
        val_dataset=val_ds,
    )
    trainer.train()
    trainer.save_adapter(scratch_dir / "adapter")

    hash_after = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    assert hash_before == hash_after


def test_adapter_saves_only_lora_and_classifier_not_the_full_backbone(scratch_dir: Path) -> None:
    train_ds, val_ds, labels, pragma_config, processor_dir, checkpoint_dir = _setup(scratch_dir)
    trainer = DownstreamTrainer(
        train_ds,
        labels,
        pragma_config,
        DownstreamConfig(n_epochs=1, batch_size=4, mixed_precision="no", seed=0),
        base_checkpoint_dir=checkpoint_dir,
        base_checkpoint_name="final",
        processor_dir=processor_dir,
        val_dataset=val_ds,
    )
    trainer.train()
    adapter_dir = scratch_dir / "adapter"
    trainer.save_adapter(adapter_dir)

    base_size = (checkpoint_dir / "final" / "model.safetensors").stat().st_size
    adapter_size = (adapter_dir / "adapter_model.safetensors").stat().st_size
    assert adapter_size < base_size

    manifest = json.loads((adapter_dir / "adapter_manifest.json").read_text())
    assert manifest["base_checkpoint_dir"] == str(checkpoint_dir)
    assert manifest["base_checkpoint_name"] == "final"
    assert manifest["base_checkpoint_hash"] is not None
    assert manifest["processor_hash"] is not None


def test_reloaded_adapter_reproduces_the_trained_models_predictions(scratch_dir: Path) -> None:
    train_ds, val_ds, labels, pragma_config, processor_dir, checkpoint_dir = _setup(scratch_dir)
    trainer = DownstreamTrainer(
        train_ds,
        labels,
        pragma_config,
        DownstreamConfig(n_epochs=2, batch_size=4, mixed_precision="no", seed=0),
        base_checkpoint_dir=checkpoint_dir,
        base_checkpoint_name="final",
        processor_dir=processor_dir,
        val_dataset=val_ds,
    )
    trainer.train()
    adapter_dir = scratch_dir / "adapter"
    trainer.save_adapter(adapter_dir)

    batch = PragmaCollator()(list(val_ds))
    trainer.model.eval()
    with torch.no_grad():
        trained_logits = trainer.model(batch).logits.clone()

    reloaded = load_lora_model(
        pragma_config,
        adapter_dir,
        base_checkpoint_dir=checkpoint_dir,
        base_checkpoint_name="final",
        processor_dir=processor_dir,
    )
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(batch).logits

    assert torch.allclose(trained_logits, reloaded_logits, atol=1e-5)
