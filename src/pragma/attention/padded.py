"""`PaddedAttentionBackend`: the permanent correctness oracle (section 9.2; ADR 0006).

Pads each `cu_seqlens`-bounded group to the batch's longest group, runs
standard `scaled_dot_product_attention` with an explicit boolean key-padding
mask, then discards the padding on the way back out. Every group produced by
this codebase's encoders has length >= 1 (a `[USR]`/`[EVT]` slot is always
prepended before attention runs — ADR 0009), so there is no all-masked-row
edge case to guard against here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from pragma.attention.backend import AttentionBackend


class PaddedAttentionBackend(AttentionBackend):
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

        q_pad = pad_sequence(list(torch.split(q, lengths)), batch_first=True)  # [n_seq, L, H, D]
        k_pad = pad_sequence(list(torch.split(k, lengths)), batch_first=True)
        v_pad = pad_sequence(list(torch.split(v, lengths)), batch_first=True)

        max_len = q_pad.shape[1]
        length_tensor = torch.tensor(lengths, device=q.device)
        # [n_seq, max_len], True where the key position is real (not padding).
        valid = torch.arange(max_len, device=q.device).unsqueeze(0) < length_tensor.unsqueeze(1)

        q_pad = q_pad.transpose(1, 2)  # [n_seq, H, L, D] — SDPA's expected layout
        k_pad = k_pad.transpose(1, 2)
        v_pad = v_pad.transpose(1, 2)

        # Broadcasts over heads and query rows; only masks which *keys* a query may see.
        attn_mask = valid[:, None, None, :]
        out = F.scaled_dot_product_attention(q_pad, k_pad, v_pad, attn_mask=attn_mask)
        out = out.transpose(1, 2)  # [n_seq, L, H, D]

        # Boolean-index with the same [n_seq, max_len] mask to drop padding and
        # restore the original packed (sequence-major, then position) order.
        return out[valid]
