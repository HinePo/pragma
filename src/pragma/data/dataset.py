"""`TokenizedRecordDataset`: the reference (non-dynamic) dataset for Phase 3.

Wraps an in-memory list of `TokenizedRecord`s loaded from a `PragmaRecordStore`
as a `torch.utils.data.Dataset`, so it plugs directly into a standard
`DataLoader` with `PragmaCollator` for small, fixed-size batches. The dynamic
`TokenBudgetBatchSampler` (section 9.3) that groups records by token/event
budget instead of a fixed batch size is a Phase 8 concern — this dataset is
deliberately the simple reference path Phase 8 will be benchmarked against.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from pragma.data.storage import PragmaRecordStore, read_dataset
from pragma.processing.tokenized_record import TokenizedRecord


class TokenizedRecordDataset(Dataset[TokenizedRecord]):
    def __init__(self, records: list[TokenizedRecord]) -> None:
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> TokenizedRecord:
        return self._records[index]

    @classmethod
    def from_store(
        cls, shard_dir: Path, store: PragmaRecordStore, *, split: str | None = None
    ) -> TokenizedRecordDataset:
        """Load every shard for `split` (or all splits) from `shard_dir`'s manifest."""
        records = read_dataset(shard_dir, store, split=split)
        return cls(records)
