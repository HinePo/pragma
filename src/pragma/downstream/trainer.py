"""`DownstreamTrainer`: Accelerate-based LoRA fine-tuning loop (section 11.4, 15.2).

Mirrors `PretrainingEngine`'s hardware posture exactly (ADR 0011, reaffirmed
in ADR 0013 for this phase): hardware/DDP selection is entirely
`Accelerate`'s job (`Accelerator()`, `resolve_mixed_precision`), never
hardcoded here. One deliberate simplification versus `PretrainingEngine`:
downstream fine-tuning datasets are small and fixed-size, so this uses a
plain fixed-batch `DataLoader` through `accelerator.prepare()` (which shards
it automatically under DDP) instead of `TokenBudgetBatchSampler`'s own
rank-splitting — that dynamic sampler exists specifically for pretraining's
variable, much larger token budgets (section 9.3), which downstream
fine-tuning does not need.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from accelerate import Accelerator
from peft import PeftModel
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from pragma.artifacts.checkpoint import CheckpointManager, hash_processor_bundle
from pragma.config import DownstreamConfig
from pragma.data.batch import PragmaBatch, PragmaCollator
from pragma.data.dataset import TokenizedRecordDataset
from pragma.downstream.lora import build_lora_model
from pragma.downstream.task_model import PragmaForTask
from pragma.modeling import PragmaConfig, PragmaForMaskedModeling
from pragma.processing.tokenized_record import TokenizedRecord
from pragma.training.hardware import resolve_mixed_precision
from pragma.training.scheduler import WarmupDecayScheduler

MANIFEST_FILE = "adapter_manifest.json"


class DownstreamCollator:
    """Builds a plain (unmasked) `PragmaBatch` plus an aligned label tensor."""

    def __init__(self, labels: dict[str, bool]) -> None:
        self.labels = labels
        self._collate = PragmaCollator()

    def __call__(self, records: list[TokenizedRecord]) -> tuple[PragmaBatch, torch.Tensor]:
        batch = self._collate(records)
        label_tensor = torch.tensor(
            [float(self.labels[r.entity_id]) for r in records], dtype=torch.float32
        )
        return batch, label_tensor


@dataclass
class DownstreamEpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float | None = None
    val_auc: float | None = None


def load_frozen_backbone(
    pragma_config: PragmaConfig,
    checkpoint_dir: Path,
    checkpoint_name: str,
    *,
    processor_dir: Path,
) -> PragmaForTask:
    """Builds a fresh `PragmaForTask` and loads only its `.pragma` backbone
    weights from a `PretrainingEngine` checkpoint — the `classifier` head
    stays randomly initialized, to be trained fresh for this task. Verifies
    the checkpoint's processor hash first, the same way `CheckpointManager.load`
    does (ADR 0007) — a checkpoint is invalid without its exact processor,
    whether resuming pretraining or reusing it for a downstream task.
    """
    manager = CheckpointManager(checkpoint_dir)
    manifest = manager.read_manifest(checkpoint_name)
    expected_hash = manifest["processor_hash"]
    actual_hash = hash_processor_bundle(processor_dir)
    if expected_hash != actual_hash:
        raise ValueError(
            f"processor bundle at {processor_dir} does not match checkpoint "
            f"{checkpoint_dir / checkpoint_name} (expected hash {expected_hash}, "
            f"got {actual_hash}) — a checkpoint is invalid without its exact "
            "matching processor bundle (ADR 0007)"
        )

    loader_accelerator = Accelerator()
    pretraining_model = loader_accelerator.prepare(PragmaForMaskedModeling(pragma_config))
    loader_accelerator.load_state(str(checkpoint_dir / checkpoint_name))
    backbone_state = loader_accelerator.unwrap_model(pretraining_model).pragma.state_dict()

    task_model = PragmaForTask(pragma_config)
    task_model.pragma.load_state_dict(backbone_state)
    return task_model


def _hash_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


class DownstreamTrainer:
    """Fine-tunes a LoRA adapter + task head on top of a frozen pretrained
    backbone for one binary classification task."""

    def __init__(
        self,
        train_dataset: TokenizedRecordDataset,
        labels: dict[str, bool],
        pragma_config: PragmaConfig,
        downstream_config: DownstreamConfig,
        *,
        base_checkpoint_dir: Path,
        base_checkpoint_name: str,
        processor_dir: Path,
        val_dataset: TokenizedRecordDataset | None = None,
    ) -> None:
        self.downstream_config = downstream_config
        self.processor_dir = processor_dir
        self.base_checkpoint_dir = base_checkpoint_dir
        self.base_checkpoint_name = base_checkpoint_name

        resolved_precision = resolve_mixed_precision(downstream_config.mixed_precision)
        self.accelerator = Accelerator(mixed_precision=resolved_precision)

        torch.manual_seed(downstream_config.seed)
        base_model = load_frozen_backbone(
            pragma_config,
            base_checkpoint_dir,
            base_checkpoint_name,
            processor_dir=processor_dir,
        )
        self.model: PeftModel = build_lora_model(base_model, downstream_config)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=downstream_config.learning_rate,
            weight_decay=downstream_config.weight_decay,
            betas=downstream_config.adam_betas,
        )

        collator = DownstreamCollator(labels)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=downstream_config.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        self.val_loader = (
            DataLoader(
                val_dataset,
                batch_size=downstream_config.batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            if val_dataset is not None
            else None
        )

        total_updates = max(1, len(self.train_loader) * downstream_config.n_epochs)
        warmup_steps = max(1, int(total_updates * downstream_config.warmup_ratio))
        self.scheduler = WarmupDecayScheduler(
            self.optimizer, warmup_steps=warmup_steps, total_updates=total_updates, kind="cosine"
        )

        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader
        )
        if self.val_loader is not None:
            self.val_loader = self.accelerator.prepare(self.val_loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Returns `(mean_loss, auc)` over `loader` — `auc` is `nan` if the
        loader's labels ended up single-class (mirrors `run_linear_probe`'s
        degenerate-split handling)."""
        self.model.eval()
        loss_sum, n_batches = 0.0, 0
        all_labels: list[float] = []
        all_probs: list[float] = []
        for batch, label_tensor in loader:
            batch = batch.to(self.accelerator.device)
            label_tensor = label_tensor.to(self.accelerator.device)
            out = self.model(batch, labels=label_tensor)
            loss_sum += out.loss.item()
            n_batches += 1
            all_labels.extend(label_tensor.cpu().tolist())
            all_probs.extend(torch.sigmoid(out.logits).cpu().tolist())
        self.model.train()

        mean_loss = loss_sum / max(n_batches, 1)
        auc = (
            float(roc_auc_score(all_labels, all_probs))
            if len(set(all_labels)) > 1
            else float("nan")
        )
        return mean_loss, auc

    def train(self) -> list[DownstreamEpochMetrics]:
        history: list[DownstreamEpochMetrics] = []
        for epoch in range(self.downstream_config.n_epochs):
            self.model.train()
            loss_sum, n_batches = 0.0, 0
            start = time.perf_counter()
            for batch, label_tensor in self.train_loader:
                batch = batch.to(self.accelerator.device)
                label_tensor = label_tensor.to(self.accelerator.device)

                out = self.model(batch, labels=label_tensor)
                self.accelerator.backward(out.loss)
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(), self.downstream_config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                loss_sum += out.loss.item()
                n_batches += 1

            val_loss, val_auc = (
                self.evaluate(self.val_loader) if self.val_loader is not None else (None, None)
            )
            history.append(
                DownstreamEpochMetrics(
                    epoch=epoch,
                    train_loss=loss_sum / max(n_batches, 1),
                    val_loss=val_loss,
                    val_auc=val_auc,
                )
            )
            _ = time.perf_counter() - start
        return history

    def save_adapter(self, path: Path) -> Path:
        """Saves the LoRA adapter + task head (via `modules_to_save`) to
        `path`, plus a sidecar manifest with the exact base-checkpoint and
        processor identity (section 15.2) — never touches or duplicates the
        frozen base checkpoint itself."""
        path.mkdir(parents=True, exist_ok=True)
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(str(path))

        base_checkpoint_path = self.base_checkpoint_dir / self.base_checkpoint_name
        manifest = {
            "base_checkpoint_dir": str(self.base_checkpoint_dir),
            "base_checkpoint_name": self.base_checkpoint_name,
            "base_checkpoint_hash": _hash_file(base_checkpoint_path / "model.safetensors"),
            "processor_dir": str(self.processor_dir),
            "processor_hash": hash_processor_bundle(self.processor_dir),
            "downstream_config": self.downstream_config.to_dict(),
        }
        (path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, default=str))
        return path
