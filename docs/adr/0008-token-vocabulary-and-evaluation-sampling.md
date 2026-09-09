# ADR 0008: Token ID space, milestone tokenization, and evaluation-point sampling

**Status:** Accepted (Phase 2)

## Context

Section 18 of the implementation plan leaves several processor-level
questions explicitly open: whether categorical value tokens are global or
namespaced per key, how lifelong-milestone profile fields are tokenized
given that their *value* is a timestamp, and how pretraining evaluation
points are sampled per entity. Phase 2 (`PragmaProcessor`) cannot be built
without resolving these.

## Decisions

1. **Numeric value tokens are global, shared across all numeric keys** —
   directly per section 6.1's "global percentile-value tokens combined with
   the key embedding." `NumericBucketizer` fits per-key percentile
   boundaries, but the resulting bucket *token IDs* (`[ZERO]`, `[P0]` ..
   `[P{n-1}]`) come from one shared range reused by every numeric key; the
   added key embedding disambiguates which key's distribution the bucket
   index refers to.

2. **Categorical value tokens are namespaced per key** (not deduplicated
   globally). Section 6.4's worked example assigns `V_CARD_PAYMENT` as a
   value token specific to the `type` key. Sharing categorical value IDs
   across unrelated keys would conflate unrelated meanings (e.g. a
   `direction` value and a `channel` value that happen to share a string).
   Each key gets its own contiguous vocabulary block; unknown/out-of-vocab
   categorical values map to the shared `[UNK]` special token.

3. **Text (BPE) value tokens are global** by construction — the BPE model is
   trained once across all approved text fields (ADR 0003), so subword IDs
   are naturally shared. The BPE model uses a byte-level pre-tokenizer,
   which structurally cannot produce out-of-vocabulary fragments (worst case
   falls back to single-byte tokens), so text OOV rate is defined but
   expected to be ~0.

4. **Lifelong-milestone profile fields tokenize as presence, not value.**
   A milestone key's semantic content for the model is its *timing*,
   already carried by the profile temporal RoPE coordinate (section 7.3).
   The value slot is filled by one of two shared special tokens,
   `[MILESTONE_PRESENT]` or `[MILESTONE_ABSENT]`, depending on whether the
   milestone timestamp is defined and `<= evaluation_time`. This avoids
   inventing a value vocabulary for timestamps while still letting the
   Profile State Encoder attend to "did this happen yet."

5. **One evaluation point per entity for the MVP.** `PointInTimeRecordBuilder`
   sets `evaluation_time` to the entity's latest observed event timestamp
   (or `signup_at` for zero-event entities), producing exactly one
   `EvaluationRecord` per entity. This is the simplest sampling rule
   consistent with point-in-time correctness and gives each record the
   fullest available history. Multi-point-per-entity sampling is deferred;
   the builder's split-assignment-by-entity-hash already prevents the
   leakage that multi-point sampling would otherwise introduce.

6. **Train/validation/test partitioning is by entity**, via a deterministic
   hash of `entity_id` (default 80/10/10). All processor artifacts
   (`KeyVocabulary`, `NumericBucketizer`, `CategoricalEncoder`,
   `TextBPEEncoder`) are fit only on records whose entity falls in the
   `train` split, per section 5.4 and the Phase 2 exit gate.

## Consequences

- The processor bundle must store, per numeric key, only boundary arrays
  (the bucket token IDs themselves are fixed by configuration); per
  categorical key, an explicit value-vocabulary block; and one BPE model
  shared across text fields.
- Adding a new numeric key never grows the value vocabulary; adding a new
  categorical key always does.
- If a future phase samples multiple evaluation points per entity, the
  split-by-entity-hash rule must be preserved so all records from one
  entity stay in the same partition.
