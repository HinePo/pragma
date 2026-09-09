# ADR 0006: Dual attention-backend interface

**Status:** Accepted (interface frozen; implementation is Phase 5/7)

## Context

The MVP needs both a trustworthy correctness oracle and a realistic
variable-length execution path, and must not get permanently coupled to one
attention kernel vendor (section 9.2).

## Decision

- Define one `AttentionBackend` interface with two implementations:
  `PaddedAttentionBackend` (standard SDPA + explicit masks, fp32, permanent
  reference) and `VarLenAttentionBackend` (packed buffers + cumulative
  sequence lengths, used for real pretraining).
- The optimized backend is never enabled for training until its forward
  outputs and backward gradients match the padded backend within defined
  numerical tolerances on randomized small batches (section 9.2, 16.4). This
  parity check is a standing gate, not a one-time validation.
- The specific packed kernel (PyTorch varlen attention, FlashAttention, or a
  future replacement) is isolated behind the interface so it can change
  without touching encoder code (section 3, "Attention" row).

## Consequences

- Encoder modules (`ProfileStateEncoder`, `EventEncoder`, `HistoryEncoder`)
  depend only on the `AttentionBackend` interface, never on a concrete
  kernel, so backend swaps are a config change.
- Every new attention-affecting change (new masking rule, new packing
  scheme) must re-run the padded/varlen parity suite before it can ship.
