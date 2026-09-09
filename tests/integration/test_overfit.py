"""Phase 6 exit gate (implementation plan, section 17): the model can overfit a tiny
synthetic corpus, MLM loss decreases for all three masking strategies, and predictions
structurally depend on context. Slower than `tests/unit/` (real training steps), so it
lives here per CLAUDE.md's testing conventions."""

import torch

from pragma.config import MaskingConfig, ProcessorConfig
from pragma.data.batch import PragmaCollator
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.masking import MaskingPlanner
from pragma.modeling import PragmaConfig
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry
from pragma.training import DebugTrainingConfig, context_dependency_grad, run_debug_training


def _tiny_debug_setup():
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=12,
            seed=42,
            max_events_per_entity=8,
            n_zero_event_entities=0,
            n_single_event_entities=1,
            n_long_history_entities=0,
            n_same_timestamp_entities=0,
        )
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=6, bpe_vocab_size=80))
    processor.fit(records)
    tokenized = [processor.transform(r) for r in records if r.events_before_evaluation]

    pragma_config = PragmaConfig.from_processor(
        processor,
        hidden_size=24,
        num_heads=2,
        intermediate_size=48,
        profile_layers=1,
        event_layers=2,
        history_layers=1,
    )
    masking_config = MaskingConfig(token_mask_prob=0.3, event_mask_prob=0.1, key_mask_prob=0.1)
    return tokenized, processor, pragma_config, masking_config


def test_model_overfits_a_tiny_corpus() -> None:
    tokenized, processor, pragma_config, masking_config = _tiny_debug_setup()
    debug_config = DebugTrainingConfig(n_epochs=40, batch_size=4, learning_rate=3e-3)

    result = run_debug_training(tokenized, processor, pragma_config, masking_config, debug_config)

    assert len(result.epoch_losses) == debug_config.n_epochs
    # Loss should drop substantially on a corpus this small — not just noise.
    assert result.epoch_losses[-1] < result.epoch_losses[0] * 0.7


def test_loss_decreases_for_every_masking_source() -> None:
    tokenized, processor, pragma_config, masking_config = _tiny_debug_setup()
    debug_config = DebugTrainingConfig(n_epochs=40, batch_size=4, learning_rate=3e-3)

    result = run_debug_training(tokenized, processor, pragma_config, masking_config, debug_config)

    first_seen = next(d for d in result.epoch_losses_by_origin if d)
    last = result.epoch_losses_by_origin[-1]
    for source in ("token", "event", "key"):
        assert source in first_seen and source in last, f"'{source}' never appeared in training"
        assert last[source] < first_seen[source], f"'{source}' loss did not decrease"


def test_predictions_structurally_depend_on_context() -> None:
    """Gradient-connectivity form of "destroyed context degrades predictions" — see
    `context_dependency_grad`'s docstring for why this is the robust check rather
    than a magnitude-based before/after loss comparison."""
    tokenized, processor, pragma_config, masking_config = _tiny_debug_setup()
    debug_config = DebugTrainingConfig(n_epochs=20, batch_size=4, learning_rate=3e-3)
    result = run_debug_training(tokenized, processor, pragma_config, masking_config, debug_config)

    planner = MaskingPlanner.from_registry(processor.registry, processor.key_vocab, masking_config)
    collator = PragmaCollator(masking_planner=planner)
    batch = collator(tokenized)

    assert result.model is not None
    report = context_dependency_grad(result.model, batch)
    assert report["n_context_ids_checked"] > 0
    assert report["any_nonzero"] is True


def test_debug_training_is_deterministic_for_a_fixed_seed() -> None:
    tokenized, processor, pragma_config, masking_config = _tiny_debug_setup()
    debug_config = DebugTrainingConfig(n_epochs=5, batch_size=4, learning_rate=3e-3, seed=123)

    result_a = run_debug_training(tokenized, processor, pragma_config, masking_config, debug_config)
    result_b = run_debug_training(tokenized, processor, pragma_config, masking_config, debug_config)

    assert torch.allclose(
        torch.tensor(result_a.epoch_losses), torch.tensor(result_b.epoch_losses), atol=1e-5
    )
