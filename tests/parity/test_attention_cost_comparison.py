"""Deterministic memory-cost comparison (implementation plan Phase 7 exit gate: "packed
execution demonstrates a meaningful memory or throughput improvement on representative
lengths"). See `pragma.attention.analysis`'s docstring for why this is element-count
based rather than a wall-clock benchmark."""

import random

from pragma.attention import compare_attention_cost


def _representative_event_lengths() -> list[int]:
    """Mimics real event-token-count skew: mostly short events (1-6 tokens, matching
    typical field counts) with a handful near `max_event_tokens=24` (long, multi-field
    events with a multi-token BPE description)."""
    rng = random.Random(0)
    short_events = [rng.randint(1, 6) for _ in range(200)]
    long_events = [24] * 10
    return short_events + long_events


def test_packed_execution_needs_far_fewer_elements_on_skewed_lengths() -> None:
    lengths = _representative_event_lengths()
    comparison = compare_attention_cost(lengths, num_heads=3, head_dim=64)

    assert comparison.qkv_memory_ratio > 2.0
    assert comparison.score_memory_ratio > 5.0


def test_uniform_lengths_show_no_padding_waste() -> None:
    """The padded and packed costs converge when every group is the same length —
    padding waste is specifically a function of length *variance*, not batch size."""
    lengths = [8] * 50
    comparison = compare_attention_cost(lengths, num_heads=3, head_dim=64)

    assert comparison.qkv_memory_ratio == 1.0
    assert comparison.score_memory_ratio == 1.0
