"""Analytical memory/compute comparison between padded and packed (varlen) execution.

Deterministic and hardware-independent, unlike a wall-clock benchmark — this is
what backs the Phase 7 exit gate's "packed execution demonstrates a meaningful
memory or throughput improvement on representative lengths." A wall-clock
benchmark is still worth running (see `008_varlen_attention.ipynb`), but ADR
0010 notes PyTorch's CPU nested-tensor SDPA kernel is not yet as optimized as
its dense batched path, so wall-clock time on CPU can actually *regress* even
though the padded backend allocates and computes over far more elements —
the element-count comparison here is the metric that survives that caveat.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionCostComparison:
    n_sequences: int
    total_tokens: int
    max_length: int
    padded_qkv_elements: int
    """Elements `PaddedAttentionBackend` allocates for one of q/k/v: `n_seq * max_len * H * D`."""
    packed_qkv_elements: int
    """Elements `VarLenAttentionBackend` actually needs: `total_tokens * H * D` — no padding."""
    padded_score_elements: int
    """Attention-score matrix elements the padded backend computes: `n_seq * H * max_len^2`."""
    packed_score_elements: int
    """Attention-score elements a packed/varlen kernel actually needs: `H * sum(length_i^2)`."""

    @property
    def qkv_memory_ratio(self) -> float:
        return self.padded_qkv_elements / self.packed_qkv_elements

    @property
    def score_memory_ratio(self) -> float:
        return self.padded_score_elements / self.packed_score_elements


def compare_attention_cost(
    lengths: list[int], *, num_heads: int, head_dim: int
) -> AttentionCostComparison:
    """Computes the padded-vs-packed element-count comparison for one batch's group lengths."""
    n_sequences = len(lengths)
    total_tokens = sum(lengths)
    max_length = max(lengths) if lengths else 0

    padded_qkv = n_sequences * max_length * num_heads * head_dim
    packed_qkv = total_tokens * num_heads * head_dim

    padded_scores = n_sequences * num_heads * max_length * max_length
    packed_scores = num_heads * sum(length * length for length in lengths)

    return AttentionCostComparison(
        n_sequences=n_sequences,
        total_tokens=total_tokens,
        max_length=max_length,
        padded_qkv_elements=padded_qkv,
        packed_qkv_elements=packed_qkv,
        padded_score_elements=padded_scores,
        packed_score_elements=packed_scores,
    )
