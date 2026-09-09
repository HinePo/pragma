"""`AttentionBackend`: interface for self-attention over packed, bounded sequences.

Implementation plan, section 9.2; ADR 0006. Every encoder depends only on
this interface — never on padding or kernel details — so the packed-varlen
backend (Phase 7) can replace `PaddedAttentionBackend` without any encoder
code changing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class AttentionBackend(ABC):
    """Self-attention over `N` packed tokens grouped into bounded sequences.

    `q`, `k`, `v` are `[N, num_heads, head_dim]` — every token in the batch,
    concatenated across sequences in order. `cu_seqlens` is `[n_seq + 1]`:
    sequence `i`'s tokens are `q[cu_seqlens[i]:cu_seqlens[i+1]]`. A token must
    never attend to another sequence's tokens (ADR 0004's isolation
    requirement) — this is what makes packing safe to use in place of one
    independent forward pass per sequence.
    """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Returns attention output, same shape as `q`: `[N, num_heads, head_dim]`."""
        ...

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(q, k, v, cu_seqlens)
