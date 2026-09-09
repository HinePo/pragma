"""`PretrainingEngine.evaluate` / `train(val_dataset=...)` (section 13.1's "explicit
Accelerate-based train/evaluate loop") — added when the user asked for train/valid
loss curves in `009_distributed_pretraining.ipynb` and there was no validation loss
to plot yet."""

from pathlib import Path

import pytest

from pragma.config import MaskingConfig, ProcessorConfig, TokenBudgetConfig, TrainingConfig
from pragma.data import TokenizedRecordDataset
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.modeling import PragmaConfig
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry
from pragma.training import PretrainingEngine


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    return tmp_path


def _setup(tmp_dir: Path):
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=30,
            seed=5,
            max_events_per_entity=10,
            n_zero_event_entities=1,
            n_single_event_entities=2,
            n_long_history_entities=0,
            n_same_timestamp_entities=1,
        )
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=6, bpe_vocab_size=80))
    processor.fit(records)
    tokenized = [processor.transform(r) for r in records if r.events_before_evaluation]
    train_records, val_records = tokenized[:20], tokenized[20:26]

    processor_dir = tmp_dir / "processor"
    processor.save(processor_dir)

    pragma_config = PragmaConfig.from_processor(
        processor,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        event_layers=2,
        history_layers=1,
    )
    masking_config = MaskingConfig()
    token_budget_config = TokenBudgetConfig(
        max_event_tokens_per_batch=300, max_events_per_batch=200, max_records_per_batch=8
    )
    return (
        TokenizedRecordDataset(train_records),
        TokenizedRecordDataset(val_records),
        processor,
        processor_dir,
        pragma_config,
        masking_config,
        token_budget_config,
    )


def _training_config(scratch_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        n_epochs=2,
        mixed_precision="no",
        checkpoint_dir=str(scratch_dir / "ckpt"),
        checkpoint_every_n_steps=1000,
        log_every_n_steps=1000,
        mlflow_tracking_uri=f"sqlite:///{scratch_dir}/mlflow.db",
        seed=123,
    )


def test_evaluate_returns_a_finite_mean_loss(scratch_dir: Path) -> None:
    train_ds, val_ds, processor, processor_dir, pragma_config, masking_config, tb_config = _setup(
        scratch_dir
    )
    engine = PretrainingEngine(
        train_ds,
        processor,
        pragma_config,
        masking_config,
        _training_config(scratch_dir),
        tb_config,
        processor_dir=processor_dir,
    )
    val_loss = engine.evaluate(val_ds)
    assert val_loss == val_loss  # not NaN
    assert val_loss > 0
    engine.close()


def test_train_with_val_dataset_populates_val_loss_every_epoch(scratch_dir: Path) -> None:
    train_ds, val_ds, processor, processor_dir, pragma_config, masking_config, tb_config = _setup(
        scratch_dir
    )
    engine = PretrainingEngine(
        train_ds,
        processor,
        pragma_config,
        masking_config,
        _training_config(scratch_dir),
        tb_config,
        processor_dir=processor_dir,
    )
    history = engine.train(val_dataset=val_ds)
    assert len(history) == 2
    for metrics in history:
        assert metrics.val_loss is not None
        assert metrics.val_loss > 0
    engine.close()


def test_train_without_val_dataset_leaves_val_loss_none(scratch_dir: Path) -> None:
    train_ds, _val_ds, processor, processor_dir, pragma_config, masking_config, tb_config = _setup(
        scratch_dir
    )
    engine = PretrainingEngine(
        train_ds,
        processor,
        pragma_config,
        masking_config,
        _training_config(scratch_dir),
        tb_config,
        processor_dir=processor_dir,
    )
    history = engine.train()
    assert all(m.val_loss is None for m in history)
    engine.close()


def test_evaluate_does_not_leave_the_model_in_eval_mode(scratch_dir: Path) -> None:
    train_ds, val_ds, processor, processor_dir, pragma_config, masking_config, tb_config = _setup(
        scratch_dir
    )
    engine = PretrainingEngine(
        train_ds,
        processor,
        pragma_config,
        masking_config,
        _training_config(scratch_dir),
        tb_config,
        processor_dir=processor_dir,
    )
    engine.evaluate(val_ds)
    assert engine.model.training is True
    engine.close()


def test_mlflow_run_id_is_available_while_the_run_is_open(scratch_dir: Path) -> None:
    train_ds, _val_ds, processor, processor_dir, pragma_config, masking_config, tb_config = _setup(
        scratch_dir
    )
    engine = PretrainingEngine(
        train_ds,
        processor,
        pragma_config,
        masking_config,
        _training_config(scratch_dir),
        tb_config,
        processor_dir=processor_dir,
    )
    assert engine.mlflow_run_id is not None
    engine.close()
    assert engine.mlflow_run_id is None
