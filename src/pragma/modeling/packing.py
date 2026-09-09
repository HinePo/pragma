"""Generic packed-sequence utilities for inserting/removing one vector per group.

Every `[USR]`/`[EVT]` prepend in the model (Profile State Encoder's `[USR]`,
Event Encoder's `[EVT]`, History Encoder's profile-`[USR]`-seed) is the same
operation at the tensor level: insert one vector at the start of each
`cu_seqlens`-bounded group. ADR 0009 decided to implement this once here and
reuse it at all three call sites, rather than duplicating index arithmetic
per encoder.
"""

from __future__ import annotations

import torch


def new_cu_seqlens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """`cu_seqlens` after inserting exactly one extra token into every group."""
    n_groups = cu_seqlens.shape[0] - 1
    offsets = torch.arange(n_groups + 1, device=cu_seqlens.device, dtype=cu_seqlens.dtype)
    return cu_seqlens + offsets


def cu_seqlens_from_group_ids(group_ids: torch.Tensor, n_groups: int) -> torch.Tensor:
    """Inverse of `group_index_per_token`: per-token group indices -> `cu_seqlens`.

    Used for `PragmaBatch.profile_token_to_record`, which stores one record
    index per profile token rather than a `cu_seqlens` array directly.
    Requires `group_ids` to already be non-decreasing (record-major order —
    `PragmaBatch.validate()` enforces this).
    """
    counts = torch.bincount(group_ids, minlength=n_groups)
    return torch.cat([torch.zeros(1, dtype=torch.long, device=group_ids.device), counts.cumsum(0)])


def group_index_per_token(cu_seqlens: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """Which group (0-indexed) each of `n_tokens` flat tokens belongs to."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    n_groups = lengths.shape[0]
    return torch.repeat_interleave(
        torch.arange(n_groups, device=cu_seqlens.device), lengths, output_size=n_tokens
    )


def prepend_vector(
    x: torch.Tensor, cu_seqlens: torch.Tensor, vec: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Insert one vector per group at the start of that group's tokens.

    `x`: `[N, H]` flat tokens. `vec`: `[n_groups, H]`, one vector per group
    (e.g. `[USR]`'s embedding, or the profile encoder's `[USR]` output being
    fed into the History Encoder). Returns `(new_x, new_cu_seqlens)` with
    `new_x`: `[N + n_groups, H]`.
    """
    new_cu = new_cu_seqlens(cu_seqlens)
    n_new = int(new_cu[-1])
    out = x.new_zeros((n_new, x.shape[-1]))
    starts = new_cu[:-1]
    out[starts] = vec
    keep_mask = torch.ones(n_new, dtype=torch.bool, device=x.device)
    keep_mask[starts] = False
    out[keep_mask] = x
    return out, new_cu


def group_starts(x_new: torch.Tensor, cu_seqlens_old: torch.Tensor) -> torch.Tensor:
    """The prepended vector's post-attention state for every group: `[n_groups, H]`."""
    new_cu = new_cu_seqlens(cu_seqlens_old)
    return x_new[new_cu[:-1]]


def unprepend_vector(x_new: torch.Tensor, cu_seqlens_old: torch.Tensor) -> torch.Tensor:
    """Inverse of `prepend_vector`: drop the prepended slot, restore original order/shape."""
    new_cu = new_cu_seqlens(cu_seqlens_old)
    n_new = int(new_cu[-1])
    keep_mask = torch.ones(n_new, dtype=torch.bool, device=x_new.device)
    keep_mask[new_cu[:-1]] = False
    return x_new[keep_mask]
