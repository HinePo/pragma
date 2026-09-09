"""`EmbeddingExtractor`: frozen-backbone `[USR]`/last-`[EVT]`/concatenation extraction.

Implementation plan section 14.2, steps 1-2 ("Freeze the backbone. Extract
`[USR]`, final contextual `[EVT]`, and their concatenation.") and section
11.4's `EmbeddingExtractor` row. Phase 9 only needed the `[USR]`
(`record_embeddings`) variant (`extract_record_embeddings`, kept below for
the callers that only want that one); Phase 10 needs all three, since
`LinearProbeRunner` compares them against each other and against
conventional baselines (`pragma.evaluation.baselines`).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from pragma.data.batch import PragmaCollator
from pragma.data.dataset import TokenizedRecordDataset
from pragma.modeling.model import PragmaModel


@dataclass(frozen=True)
class EmbeddingBundle:
    """One extraction pass's output, aligned position-for-position by `entity_ids`."""

    entity_ids: list[str]
    usr: torch.Tensor
    """Contextual `[USR]` state per record - `[n, hidden_size]`."""
    last_event: torch.Tensor
    """Final contextual `[EVT]` state per record - `[n, hidden_size]`. A
    zero-event evaluation record (no history yet) has no event to draw from,
    so its row is all zeros - a real, valid embedding for "nothing has
    happened yet", not a missing value."""
    concat: torch.Tensor
    """`torch.cat([usr, last_event], dim=-1)` - `[n, 2 * hidden_size]`."""


class EmbeddingExtractor:
    """Runs a frozen `PragmaModel` backbone over a dataset with no masking
    applied (a plain `PragmaCollator()`, per section 14.2's "freeze the
    backbone" - clean input, not the pretraining objective)."""

    def __init__(
        self,
        model: PragmaModel,
        *,
        batch_size: int = 16,
        device: torch.device | str = "cpu",
    ) -> None:
        """`model` should be the unwrapped backbone (`PragmaModel`, e.g.
        `accelerator.unwrap_model(engine.model).pragma`), not
        `PragmaForMaskedModeling` - only the backbone's forward is needed here."""
        self.model = model
        self.batch_size = batch_size
        self.device = device

    @torch.no_grad()
    def extract(self, dataset: TokenizedRecordDataset) -> EmbeddingBundle:
        self.model.eval()
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, collate_fn=PragmaCollator()
        )

        hidden_size = self.model.config.hidden_size
        entity_ids: list[str] = []
        usr_chunks: list[torch.Tensor] = []
        last_event_chunks: list[torch.Tensor] = []
        for batch in loader:
            batch = batch.to(self.device)
            _, event_embeddings, record_embeddings = self.model(batch)
            entity_ids.extend(batch.entity_ids)
            usr_chunks.append(record_embeddings.detach().cpu())
            last_event_chunks.append(
                _last_event_per_record(
                    event_embeddings.detach().cpu(), batch.history_cu_seqlens.cpu(), hidden_size
                )
            )

        usr = torch.cat(usr_chunks, dim=0) if usr_chunks else torch.zeros((0, hidden_size))
        last_event = (
            torch.cat(last_event_chunks, dim=0)
            if last_event_chunks
            else torch.zeros((0, hidden_size))
        )
        concat = torch.cat([usr, last_event], dim=1)
        return EmbeddingBundle(entity_ids=entity_ids, usr=usr, last_event=last_event, concat=concat)


def _last_event_per_record(
    event_embeddings: torch.Tensor, history_cu_seqlens: torch.Tensor, hidden_size: int
) -> torch.Tensor:
    n_records = history_cu_seqlens.shape[0] - 1
    out = torch.zeros((n_records, hidden_size), dtype=torch.float32)
    starts = history_cu_seqlens[:-1]
    ends = history_cu_seqlens[1:]
    for i in range(n_records):
        if ends[i] > starts[i]:
            out[i] = event_embeddings[ends[i] - 1]
    return out


@torch.no_grad()
def extract_record_embeddings(
    model: PragmaModel,
    dataset: TokenizedRecordDataset,
    *,
    batch_size: int = 16,
    device: torch.device | str = "cpu",
) -> tuple[list[str], torch.Tensor]:
    """The `[USR]`-only variant of `EmbeddingExtractor` - kept as a plain
    function for Phase 9's callers (`scripts/pretrain.py`'s periodic probe,
    `010_pilot_pretraining.ipynb`) that only need `record_embeddings`, not
    the full `EmbeddingBundle`."""
    bundle = EmbeddingExtractor(model, batch_size=batch_size, device=device).extract(dataset)
    return bundle.entity_ids, bundle.usr
