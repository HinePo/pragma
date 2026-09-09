"""`TokenBudgetBatchSampler`: dynamic, budget-constrained batching (implementation
plan, section 9.3). Replaces Phase 3's fixed-batch-size `DataLoader` for real
pretraining runs — a fixed record count per batch is unstable when history
lengths span zero to thousands of events (ADR 0004's isolation note).
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from dataclasses import dataclass

from pragma.config.training_config import TokenBudgetConfig
from pragma.data.dataset import TokenizedRecordDataset


@dataclass(frozen=True)
class BatchStats:
    """Diagnostics for one epoch's batch formation (section 9.3's reporting requirement)."""

    n_batches: int
    n_records: int
    n_events: int
    n_event_tokens: int
    records_per_batch: list[int]
    event_tokens_per_batch: list[int]
    events_per_batch: list[int]

    @property
    def mean_records_per_batch(self) -> float:
        return self.n_records / self.n_batches if self.n_batches else 0.0

    @property
    def mean_event_tokens_per_batch(self) -> float:
        return self.n_event_tokens / self.n_batches if self.n_batches else 0.0


def _epoch_rng(seed: int, epoch: int) -> random.Random:
    digest = hashlib.sha256(f"token_budget:{seed}:{epoch}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


class TokenBudgetBatchSampler:
    """Greedily packs records into batches under token/event/record budgets.

    Records are bucketed into coarse length groups, buckets and within-bucket
    order are shuffled reproducibly per `(seed, epoch)`, then packed greedily
    in that order. Distributed ranks each get an equal-length, disjoint slice
    of the resulting batch list (`num_replicas`/`rank`) — the sampler handles
    this itself rather than relying on `Accelerator.prepare()`'s automatic
    `DataLoader` sharding, which assumes fixed-size shards (ADR 0011).
    """

    def __init__(
        self,
        dataset: TokenizedRecordDataset,
        config: TokenBudgetConfig,
        *,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas}), got {rank}")
        self.dataset = dataset
        self.config = config
        self.num_replicas = num_replicas
        self.rank = rank
        self._epoch = 0

        self._event_tokens = [
            sum(len(e.value_ids()) for e in dataset[i].events) for i in range(len(dataset))
        ]
        self._event_counts = [len(dataset[i].events) for i in range(len(dataset))]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _bucketed_order(self, rng: random.Random) -> list[int]:
        n = len(self.dataset)
        by_length = sorted(range(n), key=lambda i: self._event_tokens[i])
        n_buckets = max(1, min(self.config.n_length_buckets, n))
        bucket_size = -(-n // n_buckets)  # ceil division

        buckets = [by_length[i : i + bucket_size] for i in range(0, n, bucket_size)]
        for bucket in buckets:
            rng.shuffle(bucket)
        rng.shuffle(buckets)
        return [idx for bucket in buckets for idx in bucket]

    def _pack(self, order: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_tokens = 0
        current_events = 0

        for idx in order:
            tokens, events = self._event_tokens[idx], self._event_counts[idx]
            over_budget = current and (
                current_tokens + tokens > self.config.max_event_tokens_per_batch
                or current_events + events > self.config.max_events_per_batch
                or len(current) + 1 > self.config.max_records_per_batch
            )
            if over_budget:
                batches.append(current)
                current, current_tokens, current_events = [], 0, 0
            current.append(idx)
            current_tokens += tokens
            current_events += events

        if current:
            batches.append(current)
        return batches

    def _all_batches(self) -> list[list[int]]:
        rng = _epoch_rng(self.config.seed, self._epoch)
        return self._pack(self._bucketed_order(rng))

    def _batches_for_this_rank(self) -> list[list[int]]:
        batches = self._all_batches()
        if not batches:
            return []
        # Pad so every rank gets the same number of batches (required for DDP's
        # collective gradient sync to stay aligned across ranks).
        remainder = len(batches) % self.num_replicas
        if remainder:
            batches = batches + batches[: self.num_replicas - remainder]
        return batches[self.rank :: self.num_replicas]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches_for_this_rank()

    def __len__(self) -> int:
        return len(self._batches_for_this_rank())

    def stats(self) -> BatchStats:
        """Reports over *all* ranks' batches combined (pre-padding), for one epoch."""
        batches = self._all_batches()
        records_per_batch = [len(b) for b in batches]
        event_tokens_per_batch = [sum(self._event_tokens[i] for i in b) for b in batches]
        events_per_batch = [sum(self._event_counts[i] for i in b) for b in batches]
        return BatchStats(
            n_batches=len(batches),
            n_records=sum(records_per_batch),
            n_events=sum(events_per_batch),
            n_event_tokens=sum(event_tokens_per_batch),
            records_per_batch=records_per_batch,
            event_tokens_per_batch=event_tokens_per_batch,
            events_per_batch=events_per_batch,
        )
