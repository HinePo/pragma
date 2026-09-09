"""`build_aggregated_features`: conventional point-in-time features for baselines.

Implementation plan section 14.3's "aggregated features plus LightGBM or
another strong GBDT" and "logistic regression on conventional aggregated
features" baselines both need *some* hand-built feature table to run
against - this is deliberately the kind of feature set a data scientist
would build without a foundation model: event counts/totals and static
profile attributes, all read from the same leakage-safe `EvaluationRecord`s
(`events_before_evaluation`, `profile_state`) the tokenizer sees, so the
comparison in `pragma.evaluation.report` is apples-to-apples with what
PRAGMA's embeddings had access to.
"""

from __future__ import annotations

import pandas as pd

from pragma.data.records import EvaluationRecord

_MILESTONE_FLAGS = ("first_topup_at", "first_card_payment_at", "first_p2p_at")
_CATEGORICAL_ATTRIBUTES = ("country", "plan", "kyc_level", "age_band", "balance_quantile")


def build_aggregated_features(records: list[EvaluationRecord]) -> pd.DataFrame:
    """One row per record, indexed by `entity_id` (matches the 1:1 entity/record
    mapping `PointInTimeRecordBuilder` produces - one evaluation record per entity).

    Columns: `n_events`, `n_distinct_event_types`, `total_amount`, `mean_amount`,
    `days_since_signup`, one `has_<milestone>` boolean per lifelong milestone,
    and the static categorical profile attributes (`country`, `plan`,
    `kyc_level`, `age_band`, `balance_quantile`) as raw strings - encoding
    (one-hot, etc.) is a model-fitting concern, left to `pragma.evaluation.baselines`.
    """
    rows: list[dict[str, object]] = []
    for record in records:
        events = record.events_before_evaluation
        amounts: list[float] = []
        for e in events:
            amount = e.fields.get("amount")
            if isinstance(amount, int | float):
                amounts.append(float(amount))
        signup_at = record.profile_state.milestones.get("signup_at")
        days_since_signup = (
            (record.evaluation_time - signup_at).total_seconds() / 86400.0
            if signup_at is not None
            else float("nan")
        )

        row: dict[str, object] = {
            "entity_id": record.entity_id,
            "n_events": len(events),
            "n_distinct_event_types": len({e.event_type for e in events}),
            "total_amount": sum(amounts) if amounts else 0.0,
            "mean_amount": (sum(amounts) / len(amounts)) if amounts else 0.0,
            "days_since_signup": days_since_signup,
        }
        for milestone_key in _MILESTONE_FLAGS:
            milestone = record.profile_state.milestones.get(milestone_key)
            row[f"has_{milestone_key}"] = milestone is not None
        for attribute_key in _CATEGORICAL_ATTRIBUTES:
            row[attribute_key] = record.profile_state.attributes.get(attribute_key)
        rows.append(row)

    return pd.DataFrame(rows).set_index("entity_id")
