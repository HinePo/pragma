import pytest

from pragma.config import ProcessorConfig
from pragma.data.batch import IGNORE_INDEX, PragmaCollator
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry


def _tokenized_records(**overrides: object):
    defaults = dict(
        n_entities=60,
        seed=123,
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
    return [processor.transform(r) for r in records]


def _reconstruct_per_record_event_tokens(batch, record_idx: int) -> list[int]:
    """Slice one record's event value-token ids back out of the packed batch."""
    event_indices = (batch.event_to_record == record_idx).nonzero(as_tuple=True)[0].tolist()
    ids: list[int] = []
    for event_idx in event_indices:
        start = int(batch.event_cu_seqlens[event_idx])
        end = int(batch.event_cu_seqlens[event_idx + 1])
        ids.extend(batch.event_value_ids[start:end].tolist())
    return ids


def test_batch_reconstructs_source_records_exactly() -> None:
    records = _tokenized_records()
    batch = PragmaCollator()(records)

    for record_idx, record in enumerate(records):
        expected_event_value_ids = [tid for e in record.events for tid in e.value_ids()]
        assert _reconstruct_per_record_event_tokens(batch, record_idx) == expected_event_value_ids

        profile_mask = batch.profile_token_to_record == record_idx
        reconstructed_profile = batch.profile_value_ids[profile_mask].tolist()
        assert reconstructed_profile == record.profile_value_ids()


def test_event_tokens_never_cross_an_event_boundary() -> None:
    records = _tokenized_records()
    batch = PragmaCollator()(records)

    event_idx = 0
    for record in records:
        for event in record.events:
            start = int(batch.event_cu_seqlens[event_idx])
            end = int(batch.event_cu_seqlens[event_idx + 1])
            assert end - start == len(event.key_ids())
            assert batch.event_key_ids[start:end].tolist() == event.key_ids()
            event_idx += 1


def test_history_boundaries_never_mix_two_records_events() -> None:
    records = _tokenized_records()
    batch = PragmaCollator()(records)

    for record_idx in range(batch.n_records):
        start = int(batch.history_cu_seqlens[record_idx])
        end = int(batch.history_cu_seqlens[record_idx + 1])
        assigned = batch.event_to_record[start:end]
        assert bool((assigned == record_idx).all())


def test_zero_event_entities_contribute_no_events_but_keep_profile_tokens() -> None:
    records = _tokenized_records()
    zero_event_records = [r for r in records if len(r.events) == 0]
    assert zero_event_records, "expected at least one zero-event record in the fixture"

    batch = PragmaCollator()([zero_event_records[0]])
    assert batch.n_events == 0
    assert batch.n_profile_tokens > 0
    batch.validate()


def test_batch_with_mixed_zero_and_nonzero_event_records_validates() -> None:
    records = _tokenized_records()
    zero_event = next(r for r in records if len(r.events) == 0)
    nonzero_event = next(r for r in records if len(r.events) > 0)
    batch = PragmaCollator()([zero_event, nonzero_event, zero_event])
    batch.validate()
    assert batch.n_records == 3


def test_event_mlm_labels_are_all_ignore_index_before_masking_exists() -> None:
    records = _tokenized_records()
    batch = PragmaCollator()(records)
    assert bool((batch.event_mlm_labels == IGNORE_INDEX).all())
    assert batch.event_mlm_labels.shape[0] == batch.n_event_tokens


def test_validate_rejects_a_tampered_event_to_record_mapping() -> None:
    records = _tokenized_records()
    batch = PragmaCollator()(records)
    if batch.n_events < 2:
        pytest.skip("fixture did not produce enough events to tamper with")
    batch.event_to_record[0] = batch.n_records - 1
    with pytest.raises(ValueError, match="event_to_record"):
        batch.validate()


def test_validate_rejects_non_monotonic_cu_seqlens() -> None:
    records = _tokenized_records()
    batch = PragmaCollator()(records)
    if batch.n_events < 2:
        pytest.skip("fixture did not produce enough events to tamper with")
    batch.event_cu_seqlens[1] = batch.event_cu_seqlens[-1] + 1
    with pytest.raises(ValueError, match="non-decreasing"):
        batch.validate()


def test_single_record_batch_matches_multi_record_batch_slice() -> None:
    """Packing one record alone must produce identical per-record token content
    to packing it inside a larger batch (no batch-position leakage)."""
    records = _tokenized_records()
    target = next(r for r in records if len(r.events) > 0)

    solo_batch = PragmaCollator()([target])
    group_batch = PragmaCollator()(records[:5] + [target])

    solo_ids = _reconstruct_per_record_event_tokens(solo_batch, 0)
    group_ids = _reconstruct_per_record_event_tokens(group_batch, group_batch.n_records - 1)
    assert solo_ids == group_ids
