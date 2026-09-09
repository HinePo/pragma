# ADR 0005: Masking overlap, corruption, and label semantics

**Status:** Accepted (interface frozen; implementation is Phase 4)

## Context

The paper defines three masking sources (individual token 15%, whole event
10%, semantic key 10%) but does not define how they interact when they
select overlapping positions, nor the exact `[UNK]` input-dropout fraction
(section 8, section 18).

## Decision

- The three mask sources are sampled independently and combined by **union**.
  The collator records the origin of every selected position (token / event /
  key) so losses and accuracy can be broken down by masking strategy. This is
  an explicit MVP interpretation, not a claim about the paper's exact
  behavior.
- Semantic-key masking is **record-wide**: selecting a key masks every
  eligible occurrence of that key across the record's entire event history,
  not just within one event.
- Corruption rule (section 8.2, section 16.3): for a selected position, first
  decide whether it is a `[UNK]`-dropout position (probability
  `unk_dropout_frac`) or a standard masked-prediction position.
  - Standard masked positions: input replaced with `[MASK]`, label set to the
    original value ID — this is what the loss trains on.
  - `[UNK]`-dropout positions: input replaced with `[UNK]`, label set to
    `-100`. These positions perturb the input but are **excluded from the
    MLM loss** — this is what makes it "input dropout" rather than another
    masked-prediction target, and is stated explicitly in section 16.3
    ("`[UNK]` input-dropout positions have label `-100`"). *(Correction:
    an earlier version of this ADR incorrectly stated the original value is
    saved to `mlm_labels` in both cases — it is not; `[UNK]`-dropout
    positions always get `-100`.)*
  - Every non-objective (never-selected) position also has label `-100`.
  - In no case does the input ever leak the target value.
- The `[UNK]` input-dropout fraction defaults to **5%**, to be ablated over
  {0%, 5%, 10%} once a real pretraining corpus exists (section 8.2, 13.2).
- Identifier, timestamp, and special-token positions are never eligible for
  masking (enforced structurally by `FieldPolicy.maskable`, ADR 0001) —
  eligibility is a schema property, not a per-batch decision.
- A position whose *current value* is already `[UNK]` (out-of-vocabulary at
  tokenization time — ADR 0003's categorical/BPE fallback) is also never
  eligible, discovered in Phase 6 when it produced an out-of-range label:
  `[UNK]`'s ID sits below `value_vocab_start` (ADR 0009), outside the slice
  `PragmaMLMHead` computes logits over, and there is no real value left to
  predict anyway — the information was already lost before masking ran.

## Consequences

- `MaskingPlanner` diagnostics must expose mask-origin and collision-rate
  metrics from the start (section 14.1), since overlap-handling was an
  assumption, not a given.
- Changing the union-vs.-other-combination policy later is a training-data
  semantics change and must be versioned like a schema change, not silently
  swapped in the collator.
