from pragma.config import MaskingConfig, ProcessorConfig
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.masking import IGNORE_LABEL, MaskingPlanner, MaskSource
from pragma.processing import PragmaProcessor
from pragma.processing.special_tokens import SpecialTokens
from pragma.schema import SchemaRegistry


def _tokenized_records(**overrides: object):
    defaults = dict(
        n_entities=60,
        seed=321,
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
    tokenized = [processor.transform(r) for r in records]
    return tokenized, registry, processor


def _planner(registry, processor, **overrides: object) -> MaskingPlanner:
    config = MaskingConfig(**overrides)
    return MaskingPlanner.from_registry(registry, processor.key_vocab, config)


def test_every_objective_token_is_corrupted_in_the_input() -> None:
    tokenized, registry, processor = _tokenized_records()
    special = SpecialTokens()
    planner = _planner(
        registry, processor, token_mask_prob=1.0, event_mask_prob=0.0, key_mask_prob=0.0
    )

    record = next(r for r in tokenized if r.events)
    plan = planner.plan_record(record)
    original = [tid for e in record.events for tid in e.value_ids()]

    for i, origin in enumerate(plan.origin):
        if origin != MaskSource.NONE:
            assert plan.input_value_ids[i] in (special.MASK, special.UNK)
            assert plan.input_value_ids[i] != original[i]


def test_every_non_objective_token_has_ignore_label() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner = _planner(
        registry, processor, token_mask_prob=0.15, event_mask_prob=0.1, key_mask_prob=0.1
    )

    for record in tokenized:
        if not record.events:
            continue
        plan = planner.plan_record(record)
        for origin, label in zip(plan.origin, plan.labels, strict=True):
            if origin == MaskSource.NONE:
                assert label == IGNORE_LABEL


def test_unk_dropout_positions_have_ignore_label() -> None:
    tokenized, registry, processor = _tokenized_records()
    special = SpecialTokens()
    planner = _planner(
        registry,
        processor,
        token_mask_prob=1.0,
        event_mask_prob=0.0,
        key_mask_prob=0.0,
        unk_dropout_frac=1.0,
    )

    record = next(r for r in tokenized if r.events)
    plan = planner.plan_record(record)
    selected = [i for i, o in enumerate(plan.origin) if o != MaskSource.NONE]
    assert selected, "expected at least one selected position with token_mask_prob=1.0"
    for i in selected:
        assert plan.input_value_ids[i] == special.UNK
        assert plan.labels[i] == IGNORE_LABEL


def test_no_unk_dropout_keeps_original_value_as_label() -> None:
    tokenized, registry, processor = _tokenized_records()
    special = SpecialTokens()
    planner = _planner(
        registry,
        processor,
        token_mask_prob=1.0,
        event_mask_prob=0.0,
        key_mask_prob=0.0,
        unk_dropout_frac=0.0,
    )

    record = next(r for r in tokenized if r.events)
    plan = planner.plan_record(record)
    original = [tid for e in record.events for tid in e.value_ids()]
    selected = [i for i, o in enumerate(plan.origin) if o != MaskSource.NONE]
    assert selected
    for i in selected:
        assert plan.input_value_ids[i] == special.MASK
        assert plan.labels[i] == original[i]


def test_whole_event_masking_covers_the_complete_event() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner = _planner(
        registry, processor, token_mask_prob=0.0, event_mask_prob=1.0, key_mask_prob=0.0
    )

    record = next(r for r in tokenized if any(len(e.value_ids()) > 0 for e in r.events))
    plan = planner.plan_record(record)

    flat_idx = 0
    for event in record.events:
        n = len(event.value_ids())
        event_origin = plan.origin[flat_idx : flat_idx + n]
        assert all(o & MaskSource.EVENT for o in event_origin)
        flat_idx += n


def test_semantic_key_masking_covers_every_occurrence_record_wide() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner = _planner(
        registry, processor, token_mask_prob=0.0, event_mask_prob=0.0, key_mask_prob=1.0
    )

    record = next(r for r in tokenized if len(r.events) >= 2)
    plan = planner.plan_record(record)

    flat_key_ids = [kid for e in record.events for kid in e.key_ids()]
    assert all(o & MaskSource.KEY for o in plan.origin), (
        "key_mask_prob=1.0 should select every eligible key"
    )
    # Every occurrence of the same key across different events must share selection.
    for key_id in set(flat_key_ids):
        positions = [i for i, k in enumerate(flat_key_ids) if k == key_id]
        assert all(plan.origin[i] & MaskSource.KEY for i in positions)


def test_masking_never_touches_profile_tokens() -> None:
    """Masking operates only on event value tokens; profile state stays visible (section 7.5)."""
    tokenized, registry, processor = _tokenized_records()
    planner = _planner(
        registry, processor, token_mask_prob=1.0, event_mask_prob=1.0, key_mask_prob=1.0
    )

    for record in tokenized[:10]:
        plan = planner.plan_record(record)
        n_event_tokens = sum(len(e.value_ids()) for e in record.events)
        assert len(plan.origin) == n_event_tokens


def test_masking_is_deterministic_for_a_fixed_seed_and_epoch() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner_a = _planner(
        registry, processor, token_mask_prob=0.3, event_mask_prob=0.1, key_mask_prob=0.1
    )
    planner_b = _planner(
        registry, processor, token_mask_prob=0.3, event_mask_prob=0.1, key_mask_prob=0.1
    )

    record = next(r for r in tokenized if r.events)
    plan_a = planner_a.plan_record(record, epoch=0)
    plan_b = planner_b.plan_record(record, epoch=0)
    assert plan_a == plan_b


def test_masking_changes_across_epochs() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner = _planner(
        registry, processor, token_mask_prob=0.3, event_mask_prob=0.2, key_mask_prob=0.2
    )

    record = next(r for r in tokenized if len(r.events) > 5)
    plan_epoch_0 = planner.plan_record(record, epoch=0)
    plan_epoch_1 = planner.plan_record(record, epoch=1)
    assert plan_epoch_0.origin != plan_epoch_1.origin


def test_different_entities_get_different_masks_under_the_same_seed() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner = _planner(
        registry, processor, token_mask_prob=0.3, event_mask_prob=0.2, key_mask_prob=0.2
    )

    candidates = [r for r in tokenized if len(r.events) > 5][:2]
    assert len(candidates) == 2
    plan_a = planner.plan_record(candidates[0])
    plan_b = planner.plan_record(candidates[1])
    assert plan_a.origin != plan_b.origin


def test_zero_probabilities_select_nothing() -> None:
    tokenized, registry, processor = _tokenized_records()
    planner = _planner(
        registry, processor, token_mask_prob=0.0, event_mask_prob=0.0, key_mask_prob=0.0
    )

    for record in tokenized[:10]:
        plan = planner.plan_record(record)
        assert all(o == MaskSource.NONE for o in plan.origin)
        assert all(label == IGNORE_LABEL for label in plan.labels)
