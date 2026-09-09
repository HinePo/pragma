"""`PragmaTransformerBlock`: pre-norm self-attention + GELU feed-forward (section 7.1).

Q/K/V/output and MLP projections keep stable, explicit names
(`q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2`) per section 15.2's
requirement that LoRA can later target named linear projections without the
encoder code changing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pragma.attention.backend import AttentionBackend
from pragma.modeling.rope import apply_rotary


class PragmaTransformerBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.ln1 = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.ln2 = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_backend: AttentionBackend,
        *,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_tokens = x.shape[0]
        residual = x
        h = self.ln1(x)

        q = self.q_proj(h).view(n_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(n_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(h).view(n_tokens, self.num_heads, self.head_dim)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rotary(q, rope_cos, rope_sin)
            k = apply_rotary(k, rope_cos, rope_sin)

        attn_out = attention_backend(q, k, v, cu_seqlens)
        attn_out = attn_out.reshape(n_tokens, self.num_heads * self.head_dim)
        x = residual + self.dropout(self.out_proj(attn_out))

        residual = x
        h = self.ln2(x)
        ff = self.fc2(self.dropout(F.gelu(self.fc1(h))))
        return residual + self.dropout(ff)
