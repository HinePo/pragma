"""Attention backends: padded reference and packed variable-length implementations."""

from pragma.attention.analysis import AttentionCostComparison, compare_attention_cost
from pragma.attention.backend import AttentionBackend
from pragma.attention.padded import PaddedAttentionBackend
from pragma.attention.varlen import VarLenAttentionBackend

__all__ = [
    "AttentionBackend",
    "AttentionCostComparison",
    "PaddedAttentionBackend",
    "VarLenAttentionBackend",
    "compare_attention_cost",
]
