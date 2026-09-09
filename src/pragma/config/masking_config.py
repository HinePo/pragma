"""`MaskingConfig`: mask probabilities, `[UNK]` input-dropout fraction, seed.

Implementation plan, section 11.1. Defaults match section 8.1's probabilities
and ADR 0005's `[UNK]` input-dropout default.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MaskingConfig:
    token_mask_prob: float = 0.15
    """Individual value-token masking probability (section 8.1)."""
    event_mask_prob: float = 0.10
    """Whole-event masking probability (section 8.1)."""
    key_mask_prob: float = 0.10
    """Record-wide semantic-key masking probability (section 8.1, ADR 0005)."""
    unk_dropout_frac: float = 0.05
    """Fraction of selected positions replaced with `[UNK]` (label -100) instead
    of `[MASK]` (label = original value) — ADR 0005's corrected corruption rule."""
    seed: int = 20260101

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaskingConfig:
        return cls(**data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> MaskingConfig:
        return cls.from_dict(json.loads(path.read_text()))
