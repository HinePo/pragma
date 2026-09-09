# ADR 0004: Packed variable-length sequences as the canonical batch contract

**Status:** Accepted (interface frozen; implementation is Phase 3/7)

## Context

Customer histories range from zero events to thousands of events (the
synthetic corpus intentionally spans this full range — see ADR 0002's
`same_timestamp`/`long_history` fixtures and the `zero_events` cohort).
Padding every batch to the longest history wastes memory and compute at this
scale.

## Decision

- `PragmaBatch` (section 9.1) is the canonical batch representation: flat
  token/event buffers plus cumulative offsets, not padded tensors.
- Padded tensors are retained *only* as a reference implementation
  (`PaddedAttentionBackend`) for small fp32 correctness tests and toy runs —
  never for real pretraining.
- Boundaries in `PragmaBatch` (event-token offsets, event-to-record mapping,
  history offsets) are safety controls, not just performance metadata: event
  tokens must never attend across event boundaries in the Event Encoder, and
  one customer's history must never attend into another customer's history
  in the History Encoder. Both are asserted by dedicated isolation tests
  before either attention backend is trusted (section 16.4).
- Batches are formed by a `TokenBudgetBatchSampler` under token/event budgets
  rather than a fixed record count per batch, because history length varies
  by orders of magnitude (section 9.3).

## Consequences

- Storage and model code must agree on the packed contract from Phase 3
  onward; changing it later is a cross-cutting change, which is why it is
  frozen here rather than left implicit in the first storage implementation.
- The zero-event entity cohort in the synthetic corpus is a required fixture
  for testing that `PragmaBatch` handles an entity with an empty event
  history without a special-cased code path.
