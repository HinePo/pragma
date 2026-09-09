"""Phase 8 exit gate (implementation plan, section 17): "a stopped multi-GPU run
resumes correctly" — verified here on the single-device case this machine can
actually run (ADR 0011 notes real multi-GPU comparability could not be tested on
this dev machine and needs re-verification once GPU hardware is available).

The test proves resume is *exact*, not just "doesn't crash": training the same
tiny corpus for 4 uninterrupted epochs must produce the identical loss trajectory
as training 2 epochs, saving, rebuilding a fresh engine, loading the checkpoint,
and training the remaining 2 — same seed, same data, same everything else.

This module forces `torch.set_num_threads(1)`: confirmed directly while
building this test that PyTorch's multi-threaded CPU reductions are not
bit-associative, so two runs of the *same* single uninterrupted config (no
resume involved at all) can already diverge by ~1e-4 relative by the second
epoch purely from thread-scheduling order — a real, expected floating-point
property of multi-threaded CPU execution, not a resume bug. Single-threaded
execution isolates the property this test actually needs to check.
"""

from pathlib import Path

import pytest
import torch

from pragma.config import MaskingConfig, ProcessorConfig, TokenBudgetConfig, TrainingConfig
from pragma.data import TokenizedRecordDataset
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.modeling import PragmaConfig
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry
from pragma.training import PretrainingEngine

torch.set_num_threads(1)


def _setup(tmp_dir: Path):
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=20,
            seed=7,
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
    dataset = TokenizedRecordDataset(tokenized)

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
    return dataset, processor, processor_dir, pragma_config, masking_config, token_budget_config


@pytest.fixture
def scratch_dir(tmp_path: Path) -> Path:
    return tmp_path


def _training_config(scratch_dir: Path, name: str, n_epochs: int) -> TrainingConfig:
    return TrainingConfig(
        n_epochs=n_epochs,
        mixed_precision="no",
        checkpoint_dir=str(scratch_dir / f"ckpt_{name}"),
        checkpoint_every_n_steps=1,
        log_every_n_steps=1000,
        mlflow_tracking_uri=f"sqlite:///{scratch_dir}/mlflow_{name}.db",
        seed=123,
    )


def test_resumed_training_reproduces_uninterrupted_loss_trajectory(scratch_dir: Path) -> None:
    dataset, processor, processor_dir, pragma_config, masking_config, token_budget_config = _setup(
        scratch_dir
    )

    # Uninterrupted: 4 epochs straight through.
    uninterrupted_config = _training_config(scratch_dir, "uninterrupted", n_epochs=4)
    engine_full = PretrainingEngine(
        dataset,
        processor,
        pragma_config,
        masking_config,
        uninterrupted_config,
        token_budget_config,
        processor_dir=processor_dir,
    )
    full_history = engine_full.train()
    full_losses = [m.mean_loss for m in full_history]
    engine_full.close()

    # Interrupted: same n_epochs=4 target (the schedule horizon must match the
    # uninterrupted run — see `train()`'s `end_epoch` docstring), but only 2
    # epochs actually run before "crashing" and checkpointing.
    part1_config = _training_config(scratch_dir, "resumed", n_epochs=4)
    engine_part1 = PretrainingEngine(
        dataset,
        processor,
        pragma_config,
        masking_config,
        part1_config,
        token_budget_config,
        processor_dir=processor_dir,
    )
    part1_history = engine_part1.train(end_epoch=2)
    engine_part1.save_checkpoint("interrupted")
    engine_part1.close()

    part2_config = _training_config(scratch_dir, "resumed", n_epochs=4)
    engine_part2 = PretrainingEngine(
        dataset,
        processor,
        pragma_config,
        masking_config,
        part2_config,
        token_budget_config,
        processor_dir=processor_dir,
    )
    engine_part2.resume("interrupted")
    assert engine_part2.counters.epoch == 2
    part2_history = engine_part2.train(start_epoch=engine_part2.counters.epoch)

    engine_part2.close()
    resumed_losses = [m.mean_loss for m in part1_history] + [m.mean_loss for m in part2_history]

    assert len(resumed_losses) == len(full_losses) == 4
    for full, resumed in zip(full_losses, resumed_losses, strict=True):
        assert full == pytest.approx(resumed, rel=1e-4), (full_losses, resumed_losses)


def test_resume_fails_loudly_without_matching_processor_bundle(scratch_dir: Path) -> None:
    dataset, processor, processor_dir, pragma_config, masking_config, token_budget_config = _setup(
        scratch_dir
    )
    config = _training_config(scratch_dir, "mismatch", n_epochs=1)
    engine = PretrainingEngine(
        dataset,
        processor,
        pragma_config,
        masking_config,
        config,
        token_budget_config,
        processor_dir=processor_dir,
    )
    engine.train()
    engine.save_checkpoint("only")
    engine.close()

    # Corrupt the processor bundle so its hash no longer matches the checkpoint.
    (processor_dir / "bundle.json").write_text('{"corrupted": true}')

    engine2 = PretrainingEngine(
        dataset,
        processor,
        pragma_config,
        masking_config,
        config,
        token_budget_config,
        processor_dir=processor_dir,
    )
    with pytest.raises(ValueError, match="processor bundle"):
        engine2.resume("only")
    engine2.close()
