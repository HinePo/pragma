"""`VarLenAttentionBackend`: the packed-execution attention path used for real
pretraining (implementation plan, section 9.2; ADR 0006, ADR 0010).

Unlike `PaddedAttentionBackend`, this never materializes a padded tensor:
packed `[N, num_heads, head_dim]` buffers are split into per-sequence chunks
and wrapped as a PyTorch nested tensor (`torch.jagged` layout), so
`scaled_dot_product_attention` computes exactly the tokens that exist — no
wasted compute or memory on padding when `cu_seqlens` group lengths vary
widely. `.values()` unwraps the result back to the same flat packed shape
`PaddedAttentionBackend` returns, so callers never need to know which
backend ran (ADR 0006's whole point).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pragma.attention.backend import AttentionBackend


class VarLenAttentionBackend(AttentionBackend):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        n_seq = cu_seqlens.shape[0] - 1
        if n_seq == 0 or q.shape[0] == 0:
            return q.new_zeros(q.shape)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        # [N, H, D] -> nested [n_seq, ragged, H, D] -> [n_seq, H, ragged, D] for SDPA.
        nq = torch.nested.as_nested_tensor(list(torch.split(q, lengths)), layout=torch.jagged)
        nk = torch.nested.as_nested_tensor(list(torch.split(k, lengths)), layout=torch.jagged)
        nv = torch.nested.as_nested_tensor(list(torch.split(v, lengths)), layout=torch.jagged)
        nq, nk, nv = nq.transpose(1, 2), nk.transpose(1, 2), nv.transpose(1, 2)

        out = F.scaled_dot_product_attention(nq, nk, nv)
        return out.transpose(1, 2).values()
