"""`build_aggregated_features` (Phase 10's conventional-baseline feature table,
section 14.3) - the correctness property that matters most is the same one
`pragma.data.records` is already tested for: only events at or before
`evaluation_time` may ever influence a feature."""

from __future__ import annotations

from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.evaluation.features import build_aggregated_features
from pragma.schema import SchemaRegistry


def _records():
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=30,
            seed=5,
            max_events_per_entity=10,
            n_zero_event_entities=3,
            n_single_event_entities=2,
            n_long_history_entities=0,
            n_same_timestamp_entities=1,
        )
    )
    registry = SchemaRegistry.default()
    return PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)


def test_one_row_per_record_indexed_by_entity_id() -> None:
    records = _records()
    features = build_aggregated_features(records)
    assert len(features) == len(records)
    assert set(features.index) == {r.entity_id for r in records}


def test_n_events_matches_events_before_evaluation() -> None:
    records = _records()
    features = build_aggregated_features(records)
    for record in records:
        assert features.loc[record.entity_id, "n_events"] == len(record.events_before_evaluation)


def test_zero_event_entities_get_zero_amount_features_not_nan() -> None:
    records = _records()
    zero_event_records = [r for r in records if not r.events_before_evaluation]
    assert zero_event_records  # the fixture guarantees at least 3
    features = build_aggregated_features(records)
    for record in zero_event_records:
        row = features.loc[record.entity_id]
        assert row["n_events"] == 0
        assert row["total_amount"] == 0.0
        assert row["mean_amount"] == 0.0


def test_total_amount_only_sums_events_actually_before_evaluation() -> None:
    # Leakage check: build features from a truncated event history and confirm
    # only the retained events' amounts are summed - mirrors
    # tests/unit/test_records.py's point-in-time correctness style.
    records = _records()
    record = next(r for r in records if len(r.events_before_evaluation) >= 2)
    features = build_aggregated_features([record])
    expected_total = sum(
        float(e.fields["amount"]) for e in record.events_before_evaluation if "amount" in e.fields
    )
    assert features.loc[record.entity_id, "total_amount"] == expected_total


def test_categorical_attributes_are_passed_through_raw() -> None:
    records = _records()
    features = build_aggregated_features(records)
    for record in records:
        assert (
            features.loc[record.entity_id, "country"] == record.profile_state.attributes["country"]
        )
