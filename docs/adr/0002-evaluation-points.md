# ADR 0002: Evaluation points and point-in-time correctness

**Status:** Accepted

## Context

Section 5.3–5.4 of the implementation plan requires every pretraining
observation to be a leakage-safe `EvaluationRecord`: profile state and event
history as they existed at one `evaluation_time`, with no information from
after that point.

## Decision

- Point-in-time rules are enforced at the raw-data validation stage, before
  any tokenization: no event may have `created_at` before its entity's
  `signup_at`, and lifelong-milestone timestamps (`first_topup_at`,
  `first_card_payment_at`, `first_p2p_at`) must equal the earliest event of
  the matching family for that entity — never an earlier or later value.
- Tie-breaking for equal timestamps is deterministic: events are sorted by
  `(entity_id, created_at, event_id)`, and `event_id` is a monotonically
  assigned, globally unique string, so ordering never depends on insertion
  order or table scan order.
- `SyntheticDataConfig` (`src/pragma/data/synthetic.py`) deliberately
  generates a `same_timestamp` cohort of entities with two or more events at
  an identical `created_at`, so tie-breaking logic has a guaranteed fixture
  to be tested against in every generation run.
- The full construction of an `EvaluationRecord` (choosing evaluation points,
  slicing `events_before_evaluation`, reconstructing `profile_state` as of
  that point) is deferred to Phase 2's `PointInTimeRecordBuilder`. Phase 1
  only guarantees the raw tables are internally consistent enough for that
  builder to be correct.

## Consequences

- `validate_corpus()` treats milestone/signup inconsistency as a leakage
  failure, not a warning — a corpus that fails this check must not be used to
  fit the processor or train the model.
- How evaluation points themselves will be *sampled* per entity (single vs.
  multiple per customer, sampling frequency) remains an open project decision
  per section 18 and is out of scope for Phase 1.
