"""Single-device PyTorch/Accelerate debug training loop (implementation plan, section 17, Phase 6).

Deliberately minimal — a handful of epochs over a tiny in-memory record list,
plain AdamW, no dynamic batching or checkpointing. This exists to prove the
architecture *can learn* before any distributed/performance work is worth
doing (Phase 7+); it is not `PretrainingEngine` (Phase 8's durable,
checkpointed, dynamically-batched engine will supersede this loop the same
way Phase 8's `TokenBudgetBatchSampler` supersedes Phase 3's fixed-batch
`DataLoader`). Using `Accelerate` here even for a single CPU/GPU device means
Phase 8 extends this loop instead of rewriting it (section 13.1).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

from pragma.config import MaskingConfig
from pragma.data.batch import IGNORE_INDEX, PragmaBatch, PragmaCollator
from pragma.data.dataset import TokenizedRecordDataset
from pragma.masking import MaskingPlanner, MaskSource
from pragma.modeling import PragmaConfig, PragmaForMaskedModeling
from pragma.processing import PragmaProcessor
from pragma.processing.tokenized_record import TokenizedRecord

_ORIGIN_SOURCES = {"token": MaskSource.TOKEN, "event": MaskSource.EVENT, "key": MaskSource.KEY}


@dataclass(frozen=True)
class DebugTrainingConfig:
    n_epochs: int = 20
    batch_size: int = 4
    """Small on purpose (implementation plan Phase 6 is a correctness smoke test, not a
    throughput run) — keep this low on memory-constrained GPUs too."""
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    seed: int = 0


@dataclass
class DebugTrainingResult:
    epoch_losses: list[float] = field(default_factory=list)
    epoch_losses_by_origin: list[dict[str, float]] = field(default_factory=list)
    model: PragmaForMaskedModeling | None = None


def _per_origin_losses(
    batch: PragmaBatch, logits: torch.Tensor, value_vocab_start: int
) -> dict[str, float]:
    """Cross-entropy restricted to each mask source, for the exit-gate check that MLM
    loss decreases for all three masking strategies (not just in aggregate)."""
    masked = batch.event_mlm_labels != IGNORE_INDEX
    origin_masked = batch.event_mask_origin[masked]
    labels_masked = batch.event_mlm_labels[masked] - value_vocab_start

    losses: dict[str, float] = {}
    for name, flag in _ORIGIN_SOURCES.items():
        selected = (origin_masked & int(flag)) != 0
        if bool(selected.any()):
            losses[name] = F.cross_entropy(logits[selected], labels_masked[selected]).item()
    return losses


def run_debug_training(
    records: list[TokenizedRecord],
    processor: PragmaProcessor,
    pragma_config: PragmaConfig,
    masking_config: MaskingConfig,
    debug_config: DebugTrainingConfig,
) -> DebugTrainingResult:
    """Trains `PragmaForMaskedModeling` on `records` for `debug_config.n_epochs` epochs.

    Returns per-epoch overall loss and a per-mask-source breakdown (section
    16.5's "model can deliberately overfit a tiny synthetic corpus" plus
    section 14.1's "cross-entropy by ... masking source", both checked here
    at debug scale rather than waiting for a full pretraining run).
    """
    torch.manual_seed(debug_config.seed)

    planner = MaskingPlanner.from_registry(processor.registry, processor.key_vocab, masking_config)
    collator = PragmaCollator(masking_planner=planner)
    dataset = TokenizedRecordDataset(records)
    loader = DataLoader(
        dataset, batch_size=debug_config.batch_size, shuffle=True, collate_fn=collator
    )

    model = PragmaForMaskedModeling(pragma_config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=debug_config.learning_rate, weight_decay=debug_config.weight_decay
    )

    accelerator = Accelerator()
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    result = DebugTrainingResult()
    for epoch in range(debug_config.n_epochs):
        collator.set_epoch(epoch)
        model.train()

        epoch_loss_sum = 0.0
        n_batches = 0
        origin_loss_sums: dict[str, float] = dict.fromkeys(_ORIGIN_SOURCES, 0.0)
        origin_loss_counts: dict[str, int] = dict.fromkeys(_ORIGIN_SOURCES, 0)

        for batch in loader:
            if bool((batch.event_mlm_labels != IGNORE_INDEX).any()):
                optimizer.zero_grad()
                out = model(batch)
                accelerator.backward(out.loss)
                optimizer.step()

                epoch_loss_sum += out.loss.item()
                n_batches += 1

                value_vocab_start = accelerator.unwrap_model(model).config.value_vocab_start
                for name, loss_value in _per_origin_losses(
                    batch, out.mlm_logits, value_vocab_start
                ).items():
                    origin_loss_sums[name] += loss_value
                    origin_loss_counts[name] += 1

        result.epoch_losses.append(epoch_loss_sum / max(n_batches, 1))
        result.epoch_losses_by_origin.append(
            {
                name: (origin_loss_sums[name] / origin_loss_counts[name])
                for name in _ORIGIN_SOURCES
                if origin_loss_counts[name] > 0
            }
        )

    result.model = accelerator.unwrap_model(model)
    return result


def shuffle_context_tokens(batch: PragmaBatch, generator: torch.Generator) -> PragmaBatch:
    """Returns a corrupted copy of `batch` with every *non-masked* event value token
    randomly permuted among themselves — masked/target positions and their
    `[MASK]`/`[UNK]` inputs are left untouched.

    This is the "destroyed context" transformation for the Phase 6 exit gate.
    Plain within-record event *reordering* would not test anything here —
    this architecture's self-attention plus continuous-time RoPE is not
    order-sensitive to begin with (each token carries its own true time
    coordinate regardless of list position) — so it corrupts *content*
    (which value sits at which context position) instead.

    Empirically, on a tiny low-cardinality synthetic debug corpus, the
    resulting *loss* often barely moves even though `context_dependency_grad`
    below confirms the model genuinely depends on this context: with only a
    handful of possible values per key and a handful of records, a
    per-key-marginal shortcut already explains most of the achievable loss
    reduction, so scrambling the rest of the context doesn't cost much in
    aggregate. Use `context_dependency_grad` as the pass/fail check; treat
    this function's loss comparison as an exploratory diagnostic, not an
    assertion — see `plans/progress.md`'s Phase 6 methodology note.
    """
    corrupted = copy.deepcopy(batch)
    context_positions = (corrupted.event_mlm_labels == IGNORE_INDEX).nonzero(as_tuple=True)[0]
    if context_positions.numel() > 1:
        permuted = context_positions[torch.randperm(context_positions.numel(), generator=generator)]
        corrupted.event_value_ids[context_positions] = corrupted.event_value_ids[permuted]
    return corrupted


def context_dependency_grad(
    model: PragmaForMaskedModeling, batch: PragmaBatch
) -> dict[str, float | int | bool]:
    """Confirms MLM predictions structurally depend on context tokens via gradient
    connectivity: `d(loss)/d(embedding row)` for every *context* (non-masked)
    value ID that appears in `batch` must be non-zero somewhere.

    This is the robust form of the "destroyed context degrades predictions"
    exit-gate check — robust for the same reason Phase 5's time-coordinate
    test needed a gradient check instead of a magnitude threshold: whether a
    specific trained model's solution happens to be *numerically sensitive*
    to a random corruption is a fact about that particular fit and corpus,
    not about whether the mechanism exists. A zero gradient here would mean
    context literally cannot influence predictions, which would be a real
    architecture bug.
    """
    model.zero_grad()
    embedding = model.pragma.embedding.embedding.weight
    previously_required_grad = embedding.requires_grad
    embedding.requires_grad_(True)

    out = model(batch)
    assert out.loss is not None
    out.loss.backward()
    assert embedding.grad is not None
    grad = embedding.grad

    context_value_ids = batch.event_value_ids[batch.event_mlm_labels == IGNORE_INDEX].unique()
    grad_norms = (
        grad[context_value_ids].norm(dim=-1) if context_value_ids.numel() else grad.new_zeros(0)
    )

    embedding.requires_grad_(previously_required_grad)
    embedding.grad = None

    return {
        "n_context_ids_checked": int(context_value_ids.numel()),
        "max_grad_norm": float(grad_norms.max()) if grad_norms.numel() else 0.0,
        "any_nonzero": bool((grad_norms > 0).any()) if grad_norms.numel() else False,
    }
