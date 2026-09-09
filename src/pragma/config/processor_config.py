"""`ProcessorConfig`: field types, vocabulary rules, numeric buckets, BPE settings, truncation.

Implementation plan, section 11.1.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessorConfig:
    n_numeric_buckets: int = 16
    bpe_vocab_size: int = 512
    bpe_min_frequency: int = 1
    max_within_field_tokens: int = 8
    """Truncation cap for a single multi-token BPE field value."""
    max_profile_tokens: int = 200
    """Matches the paper's PRAGMA-S maximum profile-state tokens (section 7.1)."""
    max_event_tokens: int = 24
    """Matches the paper's PRAGMA-S maximum event tokens (section 7.1)."""
    max_history_events: int = 6500
    """Matches the paper's PRAGMA-S maximum history events (section 7.1)."""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessorConfig:
        return cls(**data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> ProcessorConfig:
        return cls.from_dict(json.loads(path.read_text()))
