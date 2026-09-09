"""Per-key categorical value vocabularies (implementation plan, section 6.3).

Per ADR 0008, categorical value tokens are namespaced per key: each key gets
its own contiguous ID block, assigned in the order keys are fitted. Unknown
values at transform time fall back to the shared `[UNK]` special token
rather than growing the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pragma.processing.special_tokens import SpecialTokens


@dataclass
class CategoricalKeyStats:
    vocab: dict[str, int]
    n_train_values: int
    oov_count: int = 0
    lookup_count: int = 0

    @property
    def oov_rate(self) -> float:
        return (self.oov_count / self.lookup_count) if self.lookup_count else 0.0


@dataclass
class CategoricalEncoder:
    """Fits a per-key value vocabulary and encodes/decodes categorical values."""

    unk_id: int = field(default_factory=lambda: SpecialTokens().UNK)
    _next_id: dict[str, int] = field(default_factory=dict)
    _stats: dict[str, CategoricalKeyStats] = field(default_factory=dict)

    def fit(self, key: str, values: list[str], *, base_id: int) -> None:
        """Fit a stable value vocabulary for one key from training-partition values."""
        vocab: dict[str, int] = {}
        next_id = base_id
        for value in sorted(set(values)):
            vocab[value] = next_id
            next_id += 1
        self._stats[key] = CategoricalKeyStats(vocab=vocab, n_train_values=len(values))
        self._next_id[key] = next_id

    def vocab_size(self, key: str) -> int:
        return len(self._stats[key].vocab)

    def base_id(self, key: str) -> int:
        vocab = self._stats[key].vocab
        return min(vocab.values()) if vocab else self._next_id[key]

    def transform(self, key: str, value: str) -> int:
        stats = self._stats.get(key)
        if stats is None:
            raise KeyError(f"CategoricalEncoder has no fitted vocab for key '{key}'")
        stats.lookup_count += 1
        token_id = stats.vocab.get(value)
        if token_id is None:
            stats.oov_count += 1
            return self.unk_id
        return token_id

    def stats_for(self, key: str) -> CategoricalKeyStats:
        return self._stats[key]

    def fitted_keys(self) -> list[str]:
        return sorted(self._stats)

    def to_dict(self) -> dict[str, object]:
        return {
            "unk_id": self.unk_id,
            "keys": {
                key: {"vocab": stats.vocab, "n_train_values": stats.n_train_values}
                for key, stats in self._stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoricalEncoder:
        encoder = cls(unk_id=int(data["unk_id"]))
        for key, payload in data["keys"].items():
            vocab = {str(v): int(i) for v, i in payload["vocab"].items()}
            encoder._stats[key] = CategoricalKeyStats(
                vocab=vocab, n_train_values=int(payload["n_train_values"])
            )
            encoder._next_id[key] = (max(vocab.values()) + 1) if vocab else 0
        return encoder
