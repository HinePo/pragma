from datetime import UTC, datetime

from pragma.data.synthetic import (
    SyntheticDataConfig,
    _is_escalating_spender,
    generate_synthetic_corpus,
    validate_corpus,
)


def _small_config(**overrides: object) -> SyntheticDataConfig:
    defaults = dict(
        n_entities=120,
        seed=7,
        max_events_per_entity=40,
        n_zero_event_entities=5,
        n_single_event_entities=5,
        n_long_history_entities=2,
        long_history_event_count=150,
        n_same_timestamp_entities=5,
    )
    defaults.update(overrides)
    return SyntheticDataConfig(**defaults)


def test_generation_is_deterministic_for_a_fixed_seed() -> None:
    config = _small_config()
    events_a, profile_a, _ = generate_synthetic_corpus(config)
    events_b, profile_b, _ = generate_synthetic_corpus(config)
    assert events_a.equals(events_b)
    assert profile_a.equals(profile_b)


def test_different_seeds_produce_different_corpora() -> None:
    events_a, _, _ = generate_synthetic_corpus(_small_config(seed=1))
    events_b, _, _ = generate_synthetic_corpus(_small_config(seed=2))
    assert not events_a.equals(events_b)


def test_generated_corpus_passes_schema_and_leakage_validation() -> None:
    events_df, profile_df, _ = generate_synthetic_corpus(_small_config())
    report = validate_corpus(events_df, profile_df)
    assert report["passed"] is True
    assert report["n_schema_issues"] == 0
    assert report["n_leakage_issues"] == 0


def test_edge_case_entities_are_present() -> None:
    events_df, profile_df, _ = generate_synthetic_corpus(_small_config())
    counts = profile_df["_edge_case_profile"].value_counts().to_dict()
    assert counts.get("zero_events", 0) == 5
    assert counts.get("single_event", 0) == 5
    assert counts.get("long_history", 0) == 2
    assert counts.get("same_timestamp", 0) >= 1

    zero_event_ids = profile_df.loc[profile_df["_edge_case_profile"] == "zero_events", "entity_id"]
    assert not events_df["entity_id"].isin(zero_event_ids).any()

    single_event_ids = profile_df.loc[
        profile_df["_edge_case_profile"] == "single_event", "entity_id"
    ]
    single_event_counts = (
        events_df[events_df["entity_id"].isin(single_event_ids)].groupby("entity_id").size()
    )
    assert (single_event_counts == 1).all()

    long_history_ids = profile_df.loc[
        profile_df["_edge_case_profile"] == "long_history", "entity_id"
    ]
    long_history_counts = (
        events_df[events_df["entity_id"].isin(long_history_ids)].groupby("entity_id").size()
    )
    assert (long_history_counts == 150).all()


def test_same_timestamp_entities_have_colliding_events() -> None:
    events_df, profile_df, _ = generate_synthetic_corpus(_small_config())
    same_ts_ids = profile_df.loc[profile_df["_edge_case_profile"] == "same_timestamp", "entity_id"]
    same_ts_events = events_df[events_df["entity_id"].isin(same_ts_ids)]
    dup_counts = same_ts_events.groupby(["entity_id", "created_at"]).size()
    assert (dup_counts > 1).any()


def test_no_event_precedes_its_entitys_signup() -> None:
    events_df, profile_df, _ = generate_synthetic_corpus(_small_config())
    merged = events_df.merge(profile_df[["entity_id", "signup_at"]], on="entity_id")
    assert (merged["created_at"] >= merged["signup_at"]).all()


def test_milestones_match_earliest_event_of_their_family() -> None:
    events_df, profile_df, _ = generate_synthetic_corpus(_small_config())
    family_for_milestone = {
        "first_topup_at": "topup",
        "first_card_payment_at": "card_payment",
        "first_p2p_at": "p2p_transfer",
    }
    for milestone_col, family in family_for_milestone.items():
        family_events = events_df[events_df["type"] == family]
        true_min = family_events.groupby("entity_id")["created_at"].min()
        recorded = profile_df.set_index("entity_id")[milestone_col].dropna()
        common_ids = recorded.index.intersection(true_min.index)
        assert (recorded.loc[common_ids] == true_min.loc[common_ids]).all()


def test_manifest_reports_consistent_counts() -> None:
    events_df, profile_df, manifest = generate_synthetic_corpus(_small_config())
    assert manifest["n_entities"] == len(profile_df)
    assert manifest["n_events"] == len(events_df)
    assert sum(manifest["edge_case_counts"].values()) == len(profile_df)


def test_downstream_high_value_label_is_present_and_not_a_schema_field() -> None:
    events_df, profile_df, manifest = generate_synthetic_corpus(_small_config())
    assert profile_df["_downstream_is_high_value"].dtype == bool
    # Underscore-prefixed like `_edge_case_profile` -> excluded from schema
    # validation and never surfaced to `PointInTimeRecordBuilder` as a model input.
    report = validate_corpus(events_df, profile_df)
    assert report["passed"] is True
    assert sum(manifest["downstream_is_high_value_counts"].values()) == len(profile_df)


def test_downstream_high_value_label_correlates_with_own_event_volume() -> None:
    # Not a model input, but must be *derived from* the entity's own history
    # (not assigned independently of events like `is_active`) for a probe on
    # frozen embeddings to have any real signal to detect.
    events_df, profile_df, _ = generate_synthetic_corpus(_small_config())
    totals = events_df.groupby("entity_id")["amount"].sum(min_count=1).fillna(0.0)
    joined = profile_df.set_index("entity_id")["_downstream_is_high_value"]
    aligned_totals = totals.reindex(joined.index).fillna(0.0)
    high = aligned_totals[joined]
    low = aligned_totals[~joined]
    assert high.mean() > low.mean()


def test_downstream_high_value_label_is_deterministic_for_a_fixed_seed() -> None:
    config = _small_config()
    _, profile_a, _ = generate_synthetic_corpus(config)
    _, profile_b, _ = generate_synthetic_corpus(config)
    assert (profile_a["_downstream_is_high_value"] == profile_b["_downstream_is_high_value"]).all()


def _event(day: int, amount: float) -> dict[str, object]:
    return {"created_at": datetime(2024, 1, day, tzinfo=UTC), "amount": amount}


def test_is_escalating_spender_false_for_fewer_than_four_events() -> None:
    assert _is_escalating_spender([]) is False
    assert _is_escalating_spender([_event(1, 100.0), _event(2, 200.0)]) is False


def test_is_escalating_spender_true_when_second_half_spends_more() -> None:
    events = [_event(1, 10.0), _event(2, 10.0), _event(3, 100.0), _event(4, 100.0)]
    assert _is_escalating_spender(events) is True


def test_is_escalating_spender_false_when_second_half_spends_less() -> None:
    events = [_event(1, 100.0), _event(2, 100.0), _event(3, 10.0), _event(4, 10.0)]
    assert _is_escalating_spender(events) is False


def test_is_escalating_spender_depends_on_chronological_order_not_list_order() -> None:
    # Same events, given out of chronological order - the function must sort
    # by created_at itself, not trust caller order.
    events_in_order = [_event(1, 10.0), _event(2, 10.0), _event(3, 100.0), _event(4, 100.0)]
    events_shuffled = [
        events_in_order[2],
        events_in_order[0],
        events_in_order[3],
        events_in_order[1],
    ]
    assert _is_escalating_spender(events_shuffled) == _is_escalating_spender(events_in_order)


def test_downstream_escalating_spender_label_is_present_and_not_a_schema_field() -> None:
    events_df, profile_df, manifest = generate_synthetic_corpus(_small_config())
    assert profile_df["_downstream_is_escalating_spender"].dtype == bool
    report = validate_corpus(events_df, profile_df)
    assert report["passed"] is True
    assert sum(manifest["downstream_is_escalating_spender_counts"].values()) == len(profile_df)


def test_downstream_escalating_spender_label_is_deterministic_for_a_fixed_seed() -> None:
    config = _small_config()
    _, profile_a, _ = generate_synthetic_corpus(config)
    _, profile_b, _ = generate_synthetic_corpus(config)
    assert (
        profile_a["_downstream_is_escalating_spender"]
        == profile_b["_downstream_is_escalating_spender"]
    ).all()


def test_is_escalating_spender_not_determined_by_total_amount_alone() -> None:
    # The property this label exists to test (ADR 0013): two entities with the
    # exact same total spend can still get opposite escalation labels, since
    # the label depends on the order events happened in, which a plain
    # `total_amount` aggregate discards.
    front_loaded = [_event(1, 100.0), _event(2, 100.0), _event(3, 10.0), _event(4, 10.0)]
    back_loaded = [_event(1, 10.0), _event(2, 10.0), _event(3, 100.0), _event(4, 100.0)]
    assert sum(e["amount"] for e in front_loaded) == sum(e["amount"] for e in back_loaded)
    assert _is_escalating_spender(front_loaded) != _is_escalating_spender(back_loaded)
