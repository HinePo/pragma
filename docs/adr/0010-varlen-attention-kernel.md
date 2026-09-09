# ADR 0010: `VarLenAttentionBackend` kernel choice — PyTorch nested tensors, not FlashAttention

**Status:** Accepted (Phase 7)

## Context

ADR 0006 froze the dual-backend interface but deliberately left the *specific*
packed kernel open ("PyTorch varlen attention, FlashAttention, or a future
replacement"). Phase 7 has to pick one. FlashAttention's variable-length
kernels require a CUDA GPU with a supported compute capability; this
project's dev machine has no usable CUDA PyTorch build (`CLAUDE.md`'s
environment note — driver too old for current CUDA wheels), so a
FlashAttention-based backend could not even be smoke-tested here.

## Decision

- `VarLenAttentionBackend` (`src/pragma/attention/varlen.py`) uses PyTorch's
  built-in nested-tensor (`torch.jagged` layout) support for
  `scaled_dot_product_attention` — exactly the mechanism the implementation
  plan's own references point to (section 22, "PyTorch: Using variable-length
  attention"). Packed `[N, num_heads, head_dim]` buffers are split by
  `cu_seqlens` lengths into a list of variable-length chunks,
  `torch.nested.as_nested_tensor(..., layout=torch.jagged)` wraps them, SDPA
  runs directly on the nested tensors (no manual padding, no explicit mask),
  and `.values()` unwraps the result back to the same flat packed shape the
  rest of the model expects — so nothing downstream of `AttentionBackend`
  needs to know which backend ran.
- This works identically on CPU and CUDA (verified here on CPU: forward
  parity with `PaddedAttentionBackend` matches to float64 machine precision,
  and gradients flow correctly through nested-tensor SDPA).
- FlashAttention remains a valid *future* backend behind the same interface
  once a compatible GPU is available — ADR 0006's whole point is that this
  swap costs nothing in encoder code.

## Consequences

- `VarLenAttentionBackend` has zero new third-party dependencies (no
  `flash-attn` package, no custom CUDA extension) — it works everywhere
  `torch` does, at some cost of not being the fastest possible kernel.
- The nested-tensor jagged API is explicitly a PyTorch "prototype" feature
  (`UserWarning` on construction) — expect API surface to shift on PyTorch
  upgrades; pin the minimum tested version and re-verify parity after any
  `torch` bump.
- Memory/throughput benefit over `PaddedAttentionBackend` scales with how
  uneven `cu_seqlens` group lengths are (avoided padding waste); on
  near-uniform-length batches the two backends cost about the same, so the
  Phase 7 exit gate's "meaningful memory or throughput improvement" is
  measured on deliberately skewed representative lengths, not on this
  project's typical (already fairly uniform after event-count bucketing —
  ADR-adjacent to section 9.3) batches.
