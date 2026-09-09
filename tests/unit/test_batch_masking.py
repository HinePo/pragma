"""Masking wired into `PragmaCollator`/`PragmaBatch` (Phase 4 integration, section 9.1 + 16.3)."""

from pragma.config import MaskingConfig, ProcessorConfig
from pragma.data.batch import IGNORE_INDEX, PragmaCollator
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.masking import MaskingPlanner
from pragma.processing import PragmaProcessor
from pragma.processing.special_tokens import SpecialTokens
from pragma.schema import SchemaRegistry


def _tokenized_records(**overrides: object):
    defaults = dict(
        n_entities=60,
        seed=555,
        max_events_per_entity=20,
        n_zero_event_entities=5,
        n_single_event_entities=5,
        n_long_history_entities=2,
        long_history_event_count=40,
        n_same_timestamp_entities=5,
    )
    defaults.update(overrides)
    events_df, profile_df, _ = generate_synthetic_corpus(SyntheticDataConfig(**defaults))
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=8, bpe_vocab_size=150))
    processor.fit(records)
    return [processor.transform(r) for r in records], registry, processor


def test_batch_without_planner_masks_nothing() -> None:
    tokenized, _, _ = _tokenized_records()
    batch = PragmaCollator()(tokenized[:10])
    assert bool((batch.event_mlm_labels == IGNORE_INDEX).all())
    assert bool((batch.event_mask_origin == 0).all())


def test_batch_with_planner_matches_per_record_plans() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner = MaskingPlanner.from_registry(
        registry,
        processor.key_vocab,
        MaskingConfig(token_mask_prob=0.3, event_mask_prob=0.2, key_mask_prob=0.2),
    )
    collator = PragmaCollator(masking_planner=planner)
    records = [r for r in tokenized if r.events][:8]
    batch = collator(records)
    batch.validate()

    flat_idx = 0
    for record in records:
        plan = planner.plan_record(record, epoch=0)
        n = len(plan.origin)
        assert batch.event_value_ids[flat_idx : flat_idx + n].tolist() == list(plan.input_value_ids)
        assert batch.event_mlm_labels[flat_idx : flat_idx + n].tolist() == list(plan.labels)
        assert batch.event_mask_origin[flat_idx : flat_idx + n].tolist() == list(plan.origin)
        flat_idx += n


def test_set_epoch_changes_the_batch_mask() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner = MaskingPlanner.from_registry(
        registry,
        processor.key_vocab,
        MaskingConfig(token_mask_prob=0.3, event_mask_prob=0.2, key_mask_prob=0.2),
    )
    collator = PragmaCollator(masking_planner=planner)
    records = [r for r in tokenized if len(r.events) > 5][:4]

    batch_epoch_0 = collator(records)
    collator.set_epoch(1)
    batch_epoch_1 = collator(records)

    assert not bool((batch_epoch_0.event_mask_origin == batch_epoch_1.event_mask_origin).all())


def test_masked_input_never_leaks_the_target_value() -> None:
    tokenized, registry, processor = _tokenized_records()
    special = SpecialTokens()
    planner = MaskingPlanner.from_registry(
        registry,
        processor.key_vocab,
        MaskingConfig(token_mask_prob=1.0, event_mask_prob=0.0, key_mask_prob=0.0),
    )
    collator = PragmaCollator(masking_planner=planner)
    records = [r for r in tokenized if r.events][:8]
    batch = collator(records)

    selected = batch.event_mask_origin != 0
    inputs_at_selected = batch.event_value_ids[selected]
    assert bool(((inputs_at_selected == special.MASK) | (inputs_at_selected == special.UNK)).all())


def test_batch_boundaries_still_validate_with_masking_enabled() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner = MaskingPlanner.from_registry(registry, processor.key_vocab, MaskingConfig())
    collator = PragmaCollator(masking_planner=planner)

    zero_event = next(r for r in tokenized if not r.events)
    nonzero = next(r for r in tokenized if r.events)
    batch = collator([zero_event, nonzero, zero_event])
    batch.validate()
