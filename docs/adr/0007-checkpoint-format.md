# ADR 0007: Checkpoint contents and validity

**Status:** Accepted (interface frozen; implementation is Phase 8)

## Context

A checkpoint that can be resumed without materially changing results and
evaluated trustworthily must carry more than model weights: optimizer/scheduler state,
RNG state, the exact processor used to create its token IDs, and enough
lineage to reproduce the run (section 13.4, section 19).

## Decision

Every resumable checkpoint is atomic and contains, as one unit:

- Model weights (safetensors) and `PragmaConfig`.
- The exact `PragmaProcessor` bundle identity/compatibility hash used to
  produce the training token IDs (ADR 0003). **A checkpoint without its
  matching processor bundle is invalid** — this is a hard rule, not a
  recommendation.
- Optimizer, scheduler, and Accelerate/distributed state.
- Python, NumPy, CPU, and GPU RNG states, plus sampler epoch/position, so
  resume does not materially change results.
- Global token/event/record/update counters (tokens processed is the
  primary scale coordinate, not epochs — section 13.3).
- Data manifest, shard list, code revision, and dependency lock fingerprint.
- Validation metrics and best-checkpoint status.

## Consequences

- `CheckpointManager` must fail loudly if any of the above is missing on
  load, rather than resuming with partial state.
- Promoting a checkpoint (section 19) additionally requires a model card,
  data card, padded-vs-varlen parity results, and a frozen-probe evaluation
  report — the checkpoint itself is necessary but not sufficient for
  promotion.
