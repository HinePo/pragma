"""Stable special-token registry (implementation plan, section 6.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    """Fixed IDs for special tokens, stable across all processor versions.

    `MILESTONE_PRESENT`/`MILESTONE_ABSENT` are PRAGMA-specific additions
    (ADR 0008) used as the value token for lifelong-milestone profile fields,
    whose real information content is carried by the RoPE temporal
    coordinate rather than by a value token.
    """

    PAD: int = 0
    UNK: int = 1
    MASK: int = 2
    USR: int = 3
    EVT: int = 4
    MILESTONE_PRESENT: int = 5
    MILESTONE_ABSENT: int = 6

    @property
    def count(self) -> int:
        return 7

    def as_dict(self) -> dict[str, int]:
        return {
            "PAD": self.PAD,
            "UNK": self.UNK,
            "MASK": self.MASK,
            "USR": self.USR,
            "EVT": self.EVT,
            "MILESTONE_PRESENT": self.MILESTONE_PRESENT,
            "MILESTONE_ABSENT": self.MILESTONE_ABSENT,
        }
