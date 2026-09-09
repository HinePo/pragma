import tempfile
from pathlib import Path

import pytest

from pragma.config import ProcessorConfig
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.processing import PragmaProcessor
from pragma.processing.special_tokens import SpecialTokens
from pragma.schema import SchemaRegistry


def _fitted_processor(**config_overrides: object):
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=150,
            seed=42,
            max_events_per_entity=30,
            n_zero_event_entities=8,
            n_single_event_entities=8,
            n_long_history_entities=2,
            long_history_event_count=50,
            n_same_timestamp_entities=8,
        )
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    config = ProcessorConfig(n_numeric_buckets=8, bpe_vocab_size=200, **config_overrides)
    processor = PragmaProcessor(registry, config)
    processor.fit(records)
    return processor, records, registry


def test_save_load_round_trip_produces_identical_token_ids() -> None:
    processor, records, registry = _fitted_processor()
    before = [processor.transform(r).to_dict() for r in records]

    with tempfile.TemporaryDirectory() as tmp:
        bundle_path = Path(tmp) / "bundle"
        processor.save(bundle_path)
        restored = PragmaProcessor.load(bundle_path, registry)
        after = [restored.transform(r).to_dict() for r in records]

    assert before == after


def test_fitting_uses_only_train_split_records() -> None:
    processor, records, _ = _fitted_processor()
    train_entities = {r.entity_id for r in records if r.split == "train"}
    assert processor._train_fingerprint is not None

    # Refitting on the exact same train-split subset must reproduce identical
    # fingerprints/vocabularies, proving val/test rows never influenced fitting.
    train_only = [r for r in records if r.entity_id in train_entities]
    processor2 = PragmaProcessor(processor.registry, processor.config)
    processor2.fit(train_only)
    assert processor2._train_fingerprint == processor._train_fingerprint
    assert processor2.total_vocab_size == processor.total_vocab_size


def test_zero_value_maps_to_dedicated_bucket() -> None:
    processor, records, _ = _fitted_processor()
    zero_id = processor.numeric_bucketizer.zero_id
    found_zero = False
    for record in records:
        tokenized = processor.transform(record)
        for event in tokenized.events:
            for field_tokens in event.fields:
                key = processor.key_vocab.key_for(field_tokens.key_ids[0])
                if key == "amount" and field_tokens.value_ids[0] == zero_id:
                    found_zero = True
    assert found_zero, "expected at least one zero-amount event in the synthetic corpus"


def test_unknown_category_and_bpe_fragments_map_correctly() -> None:
    processor, records, _ = _fitted_processor()
    special = SpecialTokens()
    for record in records:
        tokenized = processor.transform(record)
        for event in tokenized.events:
            for field_tokens in event.fields:
                key = processor.key_vocab.key_for(field_tokens.key_ids[0])
                if key == "currency":
                    # RARE_CURRENCY ("ISK") is out-of-vocab for the train-fitted
                    # encoder whenever it lands in val/test only; either way the
                    # value must be a valid known id or the shared UNK id.
                    assert field_tokens.value_ids[0] >= 0
    # Directly exercise the UNK fallback path.
    assert processor.categorical_encoder.transform("currency", "NOT_A_REAL_CURRENCY") == special.UNK


def test_multi_token_fields_repeat_key_id_and_increment_position() -> None:
    processor, records, _ = _fitted_processor()
    found_multi_token = False
    for record in records:
        tokenized = processor.transform(record)
        for event in tokenized.events:
            for field_tokens in event.fields:
                if len(field_tokens.key_ids) > 1:
                    found_multi_token = True
                    assert len(set(field_tokens.key_ids)) == 1
                    assert list(field_tokens.within_field_pos) == list(
                        range(len(field_tokens.within_field_pos))
                    )
    assert found_multi_token, "expected at least one multi-token BPE field in the corpus"


def test_truncation_preserves_max_event_tokens_and_marks_truncated_flag() -> None:
    processor, records, _ = _fitted_processor(max_event_tokens=2)
    saw_truncation = False
    for record in records:
        tokenized = processor.transform(record)
        for event in tokenized.events:
            n_tokens = sum(len(f.key_ids) for f in event.fields)
            assert n_tokens <= 2
            if event.truncated:
                saw_truncation = True
    assert saw_truncation


def test_truncation_keeps_only_most_recent_events() -> None:
    processor, records, _ = _fitted_processor(max_history_events=3)
    for record in records:
        tokenized = processor.transform(record)
        assert len(tokenized.events) <= 3
        if tokenized.n_events_total > 3:
            assert tokenized.events_truncated
            kept_ids = [e.event_id for e in tokenized.events]
            all_sorted = sorted(
                record.events_before_evaluation, key=lambda e: (e.created_at, e.event_id)
            )
            expected_ids = [e.event_id for e in all_sorted[-3:]]
            assert kept_ids == expected_ids


def test_lifelong_milestone_fields_use_presence_tokens_not_value_tokens() -> None:
    processor, records, registry = _fitted_processor()
    special = SpecialTokens()
    milestone_keys = registry.lifelong_milestone_fields
    for record in records:
        tokenized = processor.transform(record)
        for field_tokens, coord in zip(
            tokenized.profile_fields, tokenized.profile_time_coords, strict=True
        ):
            key = processor.key_vocab.key_for(field_tokens.key_ids[0])
            if key in milestone_keys:
                assert field_tokens.value_ids[0] in (
                    special.MILESTONE_PRESENT,
                    special.MILESTONE_ABSENT,
                )
                if field_tokens.value_ids[0] == special.MILESTONE_ABSENT:
                    assert coord == 0.0


def test_transform_before_fit_raises() -> None:
    registry = SchemaRegistry.default()
    processor = PragmaProcessor(registry, ProcessorConfig())
    events_df, profile_df, _ = generate_synthetic_corpus(SyntheticDataConfig(n_entities=5))
    records = PointInTimeRecordBuilder(registry).build(events_df, profile_df)
    with pytest.raises(RuntimeError):
        processor.transform(records[0])


def test_fit_requires_at_least_one_train_record() -> None:
    registry = SchemaRegistry.default()
    processor = PragmaProcessor(registry, ProcessorConfig())
    with pytest.raises(ValueError):
        processor.fit([])
