from pragma.data.records import PointInTimeRecordBuilder, SplitConfig, drop_zero_event_records
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.schema import SchemaRegistry


def _small_corpus(**overrides: object):
    config = SyntheticDataConfig(
        n_entities=80,
        seed=11,
        max_events_per_entity=25,
        n_zero_event_entities=5,
        n_single_event_entities=5,
        n_long_history_entities=2,
        long_history_event_count=60,
        n_same_timestamp_entities=5,
        **overrides,
    )
    return generate_synthetic_corpus(config)


def test_one_record_per_entity() -> None:
    events_df, profile_df, _ = _small_corpus()
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry).build(events_df, profile_df)
    assert len(records) == len(profile_df)
    assert {r.entity_id for r in records} == set(profile_df["entity_id"])


def test_no_input_event_occurs_after_evaluation_time() -> None:
    events_df, profile_df, _ = _small_corpus()
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry).build(events_df, profile_df)
    for record in records:
        for event in record.events_before_evaluation:
            assert event.created_at <= record.evaluation_time


def test_zero_event_entities_have_no_events_and_no_milestones() -> None:
    events_df, profile_df, _ = _small_corpus()
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry).build(events_df, profile_df)
    zero_event_ids = set(
        profile_df.loc[profile_df["_edge_case_profile"] == "zero_events", "entity_id"]
    )
    for record in records:
        if record.entity_id in zero_event_ids:
            assert record.events_before_evaluation == ()
            assert record.evaluation_time == record.profile_state.as_of


def test_drop_zero_event_records_removes_only_zero_event_entities() -> None:
    events_df, profile_df, _ = _small_corpus()
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry).build(events_df, profile_df)
    zero_event_ids = set(
        profile_df.loc[profile_df["_edge_case_profile"] == "zero_events", "entity_id"]
    )

    kept, n_dropped = drop_zero_event_records(records)

    assert n_dropped == len(zero_event_ids)
    assert {r.entity_id for r in kept}.isdisjoint(zero_event_ids)
    assert all(r.events_before_evaluation for r in kept)
    assert len(kept) + n_dropped == len(records)


def test_milestones_reconstructed_as_of_evaluation_point() -> None:
    events_df, profile_df, _ = _small_corpus()
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry).build(events_df, profile_df)
    for record in records:
        for milestone_time in record.profile_state.milestones.values():
            if milestone_time is not None:
                assert milestone_time <= record.evaluation_time


def test_deterministic_ordering_for_equal_timestamps() -> None:
    events_df, profile_df, _ = _small_corpus()
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry).build(events_df, profile_df)
    for record in records:
        timestamps_and_ids = [(e.created_at, e.event_id) for e in record.events_before_evaluation]
        assert timestamps_and_ids == sorted(timestamps_and_ids)


def test_split_assignment_is_deterministic_and_respects_fractions() -> None:
    events_df, profile_df, _ = _small_corpus()
    registry = SchemaRegistry.default()
    builder = PointInTimeRecordBuilder(registry, SplitConfig(train_frac=0.8, val_frac=0.1))
    records_a = builder.build(events_df, profile_df)
    records_b = builder.build(events_df, profile_df)

    split_a = {r.entity_id: r.split for r in records_a}
    split_b = {r.entity_id: r.split for r in records_b}
    assert split_a == split_b

    counts = {"train": 0, "val": 0, "test": 0}
    for split in split_a.values():
        counts[split] += 1
    assert counts["train"] > counts["val"]
    assert counts["train"] > counts["test"]
    assert sum(counts.values()) == len(profile_df)


def test_split_assignment_is_pairwise_disjoint_across_entities() -> None:
    """No entity may appear in more than one split (paper: strict unseen-customer
    evaluation), and this must hold even if the hash function or fractions change
    in the future — hence testing the set-intersection property directly rather
    than relying on `_entity_split`'s implementation being read carefully."""
    events_df, profile_df, _ = _small_corpus()
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)

    ids_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for record in records:
        ids_by_split[record.split].add(record.entity_id)

    assert ids_by_split["train"] & ids_by_split["val"] == set()
    assert ids_by_split["train"] & ids_by_split["test"] == set()
    assert ids_by_split["val"] & ids_by_split["test"] == set()
    # Every entity is assigned to exactly one split, not silently dropped.
    assert sum(len(ids) for ids in ids_by_split.values()) == len(profile_df)


def test_different_seed_reshuffles_split_assignment() -> None:
    events_df, profile_df, _ = _small_corpus()
    registry = SchemaRegistry.default()
    records_a = PointInTimeRecordBuilder(registry, SplitConfig(seed=1)).build(events_df, profile_df)
    records_b = PointInTimeRecordBuilder(registry, SplitConfig(seed=2)).build(events_df, profile_df)
    split_a = {r.entity_id: r.split for r in records_a}
    split_b = {r.entity_id: r.split for r in records_b}
    assert split_a != split_b
