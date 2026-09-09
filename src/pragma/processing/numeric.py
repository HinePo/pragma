"""Per-key percentile bucketing for numerical values (implementation plan, section 6.2).

Per ADR 0008, bucket *token IDs* are global and shared across every numeric
key (`[ZERO]`, `[P0]` .. `[P{n-1}]`); only the percentile boundaries used to
choose a bucket index are fit per key. This keeps the numeric value
vocabulary fixed in size regardless of how many numeric keys exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NumericKeyStats:
    """Fitted boundaries and audit statistics for one numeric key."""

    boundaries: list[float]
    n_train_values: int
    n_zero_values: int
    null_rate: float


@dataclass
class NumericBucketizer:
    """Fits and applies per-key percentile bucket boundaries plus a zero bucket."""

    n_buckets: int
    value_base_id: int = 0
    _stats: dict[str, NumericKeyStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_buckets < 2:
            raise ValueError("n_buckets must be >= 2")

    @property
    def vocab_size(self) -> int:
        """Zero bucket plus `n_buckets` percentile buckets."""
        return 1 + self.n_buckets

    @property
    def zero_id(self) -> int:
        return self.value_base_id

    def percentile_id(self, bucket_index: int) -> int:
        if not 0 <= bucket_index < self.n_buckets:
            raise ValueError(f"bucket_index out of range: {bucket_index}")
        return self.value_base_id + 1 + bucket_index

    def fit(self, key: str, values: np.ndarray, *, null_count: int = 0) -> None:
        """Fit percentile boundaries for one key using training-partition values only.

        `values` must already exclude nulls; zeros are excluded from the
        percentile fit (they always route to the dedicated zero bucket) but
        counted for auditing.
        """
        non_null_total = len(values) + null_count
        n_zero = int(np.sum(values == 0.0))
        non_zero = values[values != 0.0]

        if len(non_zero) == 0:
            boundaries: list[float] = []
        else:
            edges = np.linspace(0, 100, self.n_buckets + 1)[1:-1]
            boundaries = sorted({float(b) for b in np.percentile(non_zero, edges)})

        self._stats[key] = NumericKeyStats(
            boundaries=boundaries,
            n_train_values=len(values),
            n_zero_values=n_zero,
            null_rate=(null_count / non_null_total) if non_null_total else 0.0,
        )

    def transform(self, key: str, value: float) -> int:
        """Map a value to its global bucket token ID, clipping outliers to edge buckets."""
        if key not in self._stats:
            raise KeyError(f"NumericBucketizer has no fitted stats for key '{key}'")
        if value == 0.0:
            return self.zero_id

        boundaries = self._stats[key].boundaries
        if not boundaries:
            return self.percentile_id(0)

        # side="right": a value exactly on a boundary falls into the *upper*
        # bucket, matching np.percentile's convention for the edge used to fit
        # that boundary. `min(...)` clips outliers above the top boundary to
        # the last bucket instead of raising (section 6.2: map outliers to
        # edge buckets, never refit at inference time).
        bucket_index = int(np.searchsorted(boundaries, value, side="right"))
        bucket_index = min(bucket_index, self.n_buckets - 1)
        return self.percentile_id(bucket_index)

    def stats_for(self, key: str) -> NumericKeyStats:
        return self._stats[key]

    def fitted_keys(self) -> list[str]:
        return sorted(self._stats)

    def to_dict(self) -> dict[str, object]:
        return {
            "n_buckets": self.n_buckets,
            "value_base_id": self.value_base_id,
            "stats": {
                key: {
                    "boundaries": stats.boundaries,
                    "n_train_values": stats.n_train_values,
                    "n_zero_values": stats.n_zero_values,
                    "null_rate": stats.null_rate,
                }
                for key, stats in self._stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NumericBucketizer:
        bucketizer = cls(n_buckets=int(data["n_buckets"]), value_base_id=int(data["value_base_id"]))
        for key, stats in data["stats"].items():
            bucketizer._stats[key] = NumericKeyStats(
                boundaries=list(stats["boundaries"]),
                n_train_values=int(stats["n_train_values"]),
                n_zero_values=int(stats["n_zero_values"]),
                null_rate=float(stats["null_rate"]),
            )
        return bucketizer
