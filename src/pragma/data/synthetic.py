"""Synthetic raw-event and profile-state generator for the PRAGMA Phase 1 corpus.

Produces data shaped like section 5 of the implementation plan (raw events,
profile state with lifelong milestones) and deliberately covers the edge
cases later phases must handle: zero-event entities, single-event entities,
long histories, same-timestamp events, missing optional fields, and rare
categorical values for out-of-vocabulary testing.

This module contains no notebook- or script-specific code so it can be unit
tested and reused by both `notebooks/000_synthetic_data_generation.ipynb`
and `scripts/generate_synthetic_data.py`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

import numpy as np
import pandas as pd
from faker import Faker

from pragma.schema import SchemaRegistry

CURRENCIES = ["GBP", "EUR", "USD", "PLN", "RON"]
RARE_CURRENCY = "ISK"  # deliberately rare, for OOV/tail coverage

COUNTRIES = ["GB", "PL", "RO", "FR", "DE", "IE"]
PLANS = ["standard", "plus", "premium", "metal"]
KYC_LEVELS = ["basic", "full", "enhanced"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
BALANCE_QUANTILES = ["q1", "q2", "q3", "q4", "q5"]
CHANNELS = ["bank_transfer", "card", "open_banking", "cash"]
APP_VIEWS = ["home", "cards", "p2p_amount", "analytics", "support", "crypto"]
MCC_CODES = ["5411", "5812", "6012", "4111", "5732", "7995"]
RARE_MCC = "9999"  # deliberately rare/unknown for OOV coverage

MERCHANT_NAMES = ["Tesco", "Amazon", "Uber", "Netflix", "Spotify", "Local Cafe"]
DESCRIPTION_TEMPLATES = [
    "monthly subscription",
    "grocery shopping",
    "dinner with friends",
    "metal plan",
    "rent payment",
    None,  # description is optional -> exercises null handling
]

EVENT_FAMILY_WEIGHTS = {
    "card_payment": 0.45,
    "topup": 0.15,
    "app_event": 0.20,
    "p2p_transfer": 0.10,
    "atm_withdrawal": 0.05,
    "direct_debit": 0.03,
    "fx_exchange": 0.02,
}


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Parameters controlling the size and edge-case coverage of the corpus."""

    n_entities: int = 500
    seed: int = 1337
    history_start: datetime = datetime(2024, 1, 1, tzinfo=UTC)
    history_end: datetime = datetime(2026, 6, 30, 23, 59, 59, tzinfo=UTC)
    max_events_per_entity: int = 400
    n_zero_event_entities: int = 15
    n_single_event_entities: int = 15
    n_long_history_entities: int = 5
    long_history_event_count: int = 3000
    n_same_timestamp_entities: int = 10
    rare_value_rate: float = 0.01


def _rng(seed: int) -> tuple[random.Random, np.random.Generator, Faker]:
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    faker = Faker()
    Faker.seed(seed)
    return py_rng, np_rng, faker


def _sample_event_family(py_rng: random.Random) -> str:
    names = list(EVENT_FAMILY_WEIGHTS.keys())
    weights = list(EVENT_FAMILY_WEIGHTS.values())
    return py_rng.choices(names, weights=weights, k=1)[0]


def _sample_amount(np_rng: np.random.Generator, *, allow_zero: bool) -> float:
    if allow_zero and np_rng.random() < 0.02:
        return 0.0
    # Log-normal amounts are a reasonable stand-in for real transaction
    # distributions: mostly small, with a long tail of large payments.
    return float(np.round(np_rng.lognormal(mean=2.5, sigma=1.2), 2))


def _sample_currency(py_rng: random.Random, rare_rate: float) -> str:
    if py_rng.random() < rare_rate:
        return RARE_CURRENCY
    return py_rng.choice(CURRENCIES)


def _sample_mcc(py_rng: random.Random, rare_rate: float) -> str:
    if py_rng.random() < rare_rate:
        return RARE_MCC
    return py_rng.choice(MCC_CODES)


def _build_event_fields(
    family: str,
    *,
    py_rng: random.Random,
    np_rng: np.random.Generator,
    faker: Faker,
    rare_rate: float,
) -> dict[str, object]:
    fields: dict[str, object] = {}

    if family == "topup":
        fields["amount"] = _sample_amount(np_rng, allow_zero=False)
        fields["currency"] = _sample_currency(py_rng, rare_rate)
        fields["direction"] = "in"
        fields["channel"] = py_rng.choice(CHANNELS)
        fields["description"] = py_rng.choice(DESCRIPTION_TEMPLATES)
    elif family == "card_payment":
        fields["amount"] = _sample_amount(np_rng, allow_zero=True)
        fields["currency"] = _sample_currency(py_rng, rare_rate)
        fields["direction"] = "out"
        fields["mcc"] = _sample_mcc(py_rng, rare_rate)
        fields["merchant_name"] = py_rng.choice(MERCHANT_NAMES)
        fields["description"] = py_rng.choice(DESCRIPTION_TEMPLATES)
    elif family == "p2p_transfer":
        fields["amount"] = _sample_amount(np_rng, allow_zero=False)
        fields["currency"] = _sample_currency(py_rng, rare_rate)
        fields["direction"] = py_rng.choice(["in", "out"])
        fields["counterparty_id"] = f"u{py_rng.randint(0, 999999):06d}"
        fields["description"] = py_rng.choice(DESCRIPTION_TEMPLATES)
    elif family == "atm_withdrawal":
        fields["amount"] = _sample_amount(np_rng, allow_zero=False)
        fields["currency"] = _sample_currency(py_rng, rare_rate)
        fields["direction"] = "out"
        fields["channel"] = "cash"
    elif family == "fx_exchange":
        fields["amount"] = _sample_amount(np_rng, allow_zero=False)
        fields["currency"] = _sample_currency(py_rng, rare_rate)
        fields["source_currency"] = py_rng.choice(CURRENCIES)
        fields["direction"] = "out"
        fields["description"] = py_rng.choice(DESCRIPTION_TEMPLATES)
    elif family == "direct_debit":
        fields["amount"] = _sample_amount(np_rng, allow_zero=False)
        fields["currency"] = _sample_currency(py_rng, rare_rate)
        fields["direction"] = "out"
        fields["merchant_name"] = py_rng.choice(MERCHANT_NAMES)
        fields["description"] = py_rng.choice(DESCRIPTION_TEMPLATES)
    elif family == "app_event":
        fields["view"] = py_rng.choice(APP_VIEWS)
    else:
        raise ValueError(f"unknown event family: {family}")

    return {k: v for k, v in fields.items() if v is not None}


def _random_timestamp(py_rng: random.Random, start: datetime, end: datetime) -> datetime:
    delta_seconds = int((end - start).total_seconds())
    offset = py_rng.randint(0, max(delta_seconds, 0))
    return start + timedelta(seconds=offset)


def _is_escalating_spender(entity_events: list[dict[str, object]]) -> bool:
    """Phase 11's downstream label: did this entity spend more in the second
    half of its (time-ordered) history than the first half?

    Deliberately **order-dependent**, unlike `_downstream_is_high_value`
    (Phase 9's label, a threshold of `total_amount`) — Phase 10 found that a
    label defined as a simple aggregate is trivially recoverable by a
    baseline that includes that exact aggregate as a feature
    (`pragma.evaluation.build_aggregated_features`), which made the
    probe-vs-baseline comparison uninteresting (ADR 0013's motivation for
    this new label). `total_amount`/`n_events` alone cannot recover this
    property — a conventional baseline would have to engineer a
    first-half/second-half split itself, the kind of extra feature-engineering
    cost section 14.3 asks whether PRAGMA's sequence-aware representation can
    make unnecessary. Requires >= 4 events for the two halves to be
    meaningful; fewer events (or zero) default to `False`.
    """
    if len(entity_events) < 4:
        return False
    ordered = sorted(entity_events, key=lambda e: cast(datetime, e["created_at"]))
    midpoint = len(ordered) // 2

    def _half_total(half: list[dict[str, object]]) -> float:
        total = 0.0
        for e in half:
            amount = e.get("amount")
            if isinstance(amount, int | float):
                total += float(amount)
        return total

    return _half_total(ordered[midpoint:]) > _half_total(ordered[:midpoint])


def _entity_edge_case_plan(config: SyntheticDataConfig) -> list[str]:
    """Assign each entity index a generation profile so edge cases are guaranteed."""
    plan = (
        ["zero_events"] * config.n_zero_event_entities
        + ["single_event"] * config.n_single_event_entities
        + ["long_history"] * config.n_long_history_entities
        + ["same_timestamp"] * config.n_same_timestamp_entities
    )
    n_normal = max(config.n_entities - len(plan), 0)
    plan = plan + ["normal"] * n_normal
    return plan[: config.n_entities]


def generate_synthetic_corpus(
    config: SyntheticDataConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Generate synthetic events and profile-state tables plus a generation manifest.

    Returns:
        events_df: one row per raw event, columns matching section 5.1.
        profile_df: one row per entity, columns matching section 5.2/5.5 profile fields.
        manifest: generation configuration and summary statistics for auditing.
    """
    config = config or SyntheticDataConfig()
    py_rng, np_rng, faker = _rng(config.seed)

    entity_plans = _entity_edge_case_plan(config)
    py_rng.shuffle(entity_plans)

    profile_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    event_totals: list[float] = []
    entity_escalations: list[bool] = []
    event_counter = 0

    for i, plan in enumerate(entity_plans):
        entity_id = f"u{i:06d}"
        signup_at = _random_timestamp(py_rng, config.history_start, config.history_end)

        if plan == "zero_events":
            n_events = 0
        elif plan == "single_event":
            n_events = 1
        elif plan == "long_history":
            n_events = config.long_history_event_count
        else:
            n_events = int(np_rng.integers(2, config.max_events_per_entity))

        entity_events: list[dict[str, object]] = []
        for _ in range(n_events):
            family = _sample_event_family(py_rng)
            fields = _build_event_fields(
                family,
                py_rng=py_rng,
                np_rng=np_rng,
                faker=faker,
                rare_rate=config.rare_value_rate,
            )

            if plan == "same_timestamp" and entity_events:
                created_at = entity_events[-1]["created_at"]
            else:
                created_at = _random_timestamp(py_rng, signup_at, config.history_end)

            event_counter += 1
            row: dict[str, object] = {
                "entity_id": entity_id,
                "event_id": f"e{event_counter:08d}",
                "created_at": created_at,
                "type": family,
                **fields,
            }
            entity_events.append(row)

        # Same-timestamp entities need >= 2 colliding events to be a useful fixture.
        if plan == "same_timestamp" and len(entity_events) < 2:
            continue

        event_rows.extend(entity_events)

        milestone_families = {
            "first_topup_at": "topup",
            "first_card_payment_at": "card_payment",
            "first_p2p_at": "p2p_transfer",
        }
        milestones: dict[str, datetime | None] = {}
        for milestone_key, family_name in milestone_families.items():
            matching = [
                cast(datetime, e["created_at"]) for e in entity_events if e["type"] == family_name
            ]
            milestones[milestone_key] = min(matching) if matching else None

        profile_rows.append(
            {
                "entity_id": entity_id,
                "signup_at": signup_at,
                "country": py_rng.choice(COUNTRIES),
                "plan": py_rng.choice(PLANS),
                "kyc_level": py_rng.choice(KYC_LEVELS),
                "age_band": py_rng.choice(AGE_BANDS),
                "balance_quantile": py_rng.choice(BALANCE_QUANTILES),
                "is_active": bool(py_rng.random() > 0.05),
                "first_topup_at": milestones["first_topup_at"],
                "first_card_payment_at": milestones["first_card_payment_at"],
                "first_p2p_at": milestones["first_p2p_at"],
                "_edge_case_profile": plan,
            }
        )
        entity_total = 0.0
        for e in entity_events:
            amount = e.get("amount")
            if isinstance(amount, int | float):
                entity_total += float(amount)
        event_totals.append(entity_total)
        entity_escalations.append(_is_escalating_spender(entity_events))

    events_df = pd.DataFrame(event_rows)
    profile_df = pd.DataFrame(profile_rows)

    if not profile_df.empty:
        # A downstream-task label, not a model input (Phase 9 needs one to check
        # "at least one checkpoint produces useful downstream probe signal" -
        # ADR-free choice since it's evaluation-only scaffolding, not a pretraining
        # decision). Deliberately derived from the entity's own event history
        # (total transaction volume) so it correlates with the sequence the model
        # actually sees, unlike `is_active`/`plan`, which are assigned independent
        # of events. Underscore-prefixed like `_edge_case_profile` so it is
        # automatically excluded from `SchemaRegistry` validation and never reaches
        # `PointInTimeRecordBuilder` (which only reads `registry.profile_fields`).
        totals = np.array(event_totals)
        median_total = float(np.median(totals))
        profile_df["_downstream_is_high_value"] = totals > median_total
        profile_df["_downstream_is_escalating_spender"] = entity_escalations

    if not events_df.empty:
        events_df = events_df.sort_values(["entity_id", "created_at", "event_id"]).reset_index(
            drop=True
        )

    manifest = {
        "seed": config.seed,
        "n_entities": len(profile_df),
        "n_events": len(events_df),
        "history_start": config.history_start.isoformat(),
        "history_end": config.history_end.isoformat(),
        "downstream_is_high_value_counts": (
            profile_df["_downstream_is_high_value"].value_counts().to_dict()
            if not profile_df.empty
            else {}
        ),
        "downstream_is_escalating_spender_counts": (
            profile_df["_downstream_is_escalating_spender"].value_counts().to_dict()
            if not profile_df.empty
            else {}
        ),
        "edge_case_counts": (
            profile_df["_edge_case_profile"].value_counts().to_dict()
            if not profile_df.empty
            else {}
        ),
        "event_family_counts": (
            events_df["type"].value_counts().to_dict() if not events_df.empty else {}
        ),
        "schema_version": 1,
    }

    return events_df, profile_df, manifest


def validate_corpus(events_df: pd.DataFrame, profile_df: pd.DataFrame) -> dict[str, object]:
    """Run raw schema/leakage sanity checks and return a JSON-serializable report."""
    registry = SchemaRegistry.default()

    schema_issues: list[str] = []
    event_value_columns = [
        c for c in events_df.columns if c not in ("entity_id", "event_id", "created_at", "type")
    ]
    for row in events_df.itertuples(index=False):
        row_dict = row._asdict()
        fields = {c: row_dict[c] for c in event_value_columns if pd.notna(row_dict[c])}
        issues = registry.validate_event(row_dict["type"], fields)
        schema_issues.extend(issues)

    profile_value_columns = [
        c for c in profile_df.columns if c != "entity_id" and not c.startswith("_")
    ]
    for row in profile_df.itertuples(index=False):
        row_dict = row._asdict()
        fields = {c: row_dict[c] for c in profile_value_columns if pd.notna(row_dict[c])}
        issues = registry.validate_profile(fields)
        schema_issues.extend(issues)

    leakage_issues: list[str] = []
    if not events_df.empty:
        merged = events_df.merge(profile_df[["entity_id", "signup_at"]], on="entity_id", how="left")
        before_signup = merged[merged["created_at"] < merged["signup_at"]]
        if len(before_signup) > 0:
            leakage_issues.append(
                f"{len(before_signup)} events occur before their entity's signup_at"
            )

        family_for_milestone = {
            "first_topup_at": "topup",
            "first_card_payment_at": "card_payment",
            "first_p2p_at": "p2p_transfer",
        }
        # A milestone must equal the earliest created_at among that entity's
        # events of the matching family.
        for col, family in family_for_milestone.items():
            family_events = events_df[events_df["type"] == family]
            if family_events.empty:
                continue
            true_min = family_events.groupby("entity_id")["created_at"].min()
            joined = profile_df.set_index("entity_id")[col]
            comparable = joined.reindex(true_min.index)
            mismatched = (comparable != true_min) & comparable.notna()
            if mismatched.any():
                leakage_issues.append(
                    f"{int(mismatched.sum())} entities have a '{col}' milestone "
                    f"inconsistent with their earliest '{family}' event"
                )

    null_rates: dict[str, float] = {}
    for col in event_value_columns:
        if events_df.empty:
            null_rates[col] = 0.0
        else:
            null_rates[col] = float(events_df[col].isna().mean())

    duplicate_timestamp_entities = 0
    if not events_df.empty:
        dup_counts = events_df.groupby(["entity_id", "created_at"]).size()
        duplicate_timestamp_entities = int((dup_counts > 1).groupby("entity_id").any().sum())

    report = {
        "n_schema_issues": len(schema_issues),
        "schema_issues_sample": schema_issues[:20],
        "n_leakage_issues": len(leakage_issues),
        "leakage_issues": leakage_issues,
        "null_rate_by_field": null_rates,
        "entities_with_duplicate_event_timestamps": duplicate_timestamp_entities,
        "n_events": len(events_df),
        "n_entities": len(profile_df),
        "passed": len(schema_issues) == 0 and len(leakage_issues) == 0,
    }
    return report
