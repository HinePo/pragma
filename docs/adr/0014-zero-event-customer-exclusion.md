# ADR 0014: Zero-event customers are excluded from pretraining and inference

**Status:** Accepted

## Context

`PointInTimeRecordBuilder.build()` (ADR 0008, decision 5) produces exactly
one `EvaluationRecord` per entity, including entities with no events at all
— `evaluation_time` falls back to `signup_at` and `events_before_evaluation`
is empty. This is deliberate at the record-builder level: the builder is a
faithful point-in-time snapshot of the raw tables and must be able to
represent a zero-event entity structurally (`n_zero_event_entities` is a
dedicated edge-case cohort in `SyntheticDataConfig`, and
`test_zero_event_entities_have_no_events_and_no_milestones` locks this in).

The paper, however, explicitly discards customers with zero events during
pretraining. Section 5.3 of the implementation plan states PRAGMA's
architecture generalizes to customers absent from the training data
*because there is no customer-ID lookup table* — record embeddings are
computed from `profile_state` and `events_before_evaluation`. A zero-event
record has an empty `events_before_evaluation`; the History Encoder has
nothing to encode beyond "nothing has happened yet," which is not the
cold-start capability the paper or this project claims. Section 10.2
separately expects the data-loading stage to "track ... dropped zero-event
users," implying the drop happens downstream of record building, not inside
it. Prior to this ADR, no code actually performed that drop: `fit_processor`,
`tokenize_shards`, and every embedding-extraction/probe path fed zero-event
records straight through, which is a genuine gap from the paper's stated
pretraining protocol.

## Decision

- `PointInTimeRecordBuilder` keeps building a record for every entity,
  zero-event ones included — this stays the leakage-safe, structurally
  complete snapshot layer.
- A new helper, `drop_zero_event_records()` (`src/pragma/data/records.py`),
  filters a list of `EvaluationRecord`s down to those with at least one
  event, returning the kept list and the dropped count.
- Every consumer that feeds records into pretraining or inference calls this
  helper immediately after building records and before doing anything else
  with them:
  - `scripts/fit_processor.py` — the processor must never be fit against
    zero-event customers.
  - `scripts/tokenize_shards.py` — zero-event customers must never reach a
    training shard; the dropped count is printed per section 10.2's
    tracking requirement.
  - Any script or notebook that builds records directly from raw tables for
    inference/probing (`scripts/run_probe.py` and the corresponding
    notebooks) applies the same filter, or already gets the same effect for
    free because it restricts to entity IDs present in embeddings extracted
    from the (already-filtered) shards.
- A completely new customer with zero events remains a genuine cold-start
  case this model does not support; scoring such a customer would require a
  separate profile-only model, which is out of scope here.

## Consequences

- `data/shards/` no longer contains zero-event records for any split —
  train, val, or test. A customer must have at least one event before
  `evaluation_time` to be scored by PRAGMA at all.
- `PragmaProcessor.fit()`'s existing `split == "train"` filter
  (ADR 0008, decision 6) and the new zero-event filter are independent and
  both apply — an entity must be in the train split *and* have events to
  contribute to fitted vocabulary/bucket statistics.
- `notebooks/001_point_in_time_records.ipynb` is unaffected by this decision
  — it demonstrates `PointInTimeRecordBuilder` itself, including its
  zero-event edge case, and is not a pretraining/inference consumer.
