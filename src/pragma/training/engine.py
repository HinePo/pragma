"""`PretrainingEngine`: Accelerate-based train loop (implementation plan, section 13.1,
11.4; ADR 0011).

Automatically single-device or multi-GPU DDP depending only on how the
process is launched — plain `python script.py` gives single-device execution,
`accelerate launch --multi_gpu --num_processes=N script.py` gives DDP across
N GPUs, with zero code changes either way. Nothing in this module branches on
`torch.cuda.is_available()` except indirectly through `resolve_mixed_precision`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from pragma.artifacts.checkpoint import CheckpointManager, TrainingCounters
from pragma.config import MaskingConfig, TokenBudgetConfig, TrainingConfig
from pragma.data.batch import PragmaCollator
from pragma.data.dataset import TokenizedRecordDataset
from pragma.data.token_budget_sampler import TokenBudgetBatchSampler
from pragma.masking import MaskingPlanner
from pragma.modeling import PragmaConfig, PragmaForMaskedModeling
from pragma.processing import PragmaProcessor
from pragma.training.hardware import resolve_mixed_precision
from pragma.training.optimizer import build_optimizer
from pragma.training.scheduler import build_scheduler


@dataclass
class EpochMetrics:
    """Per-epoch throughput/loss report (section 14.1's "tokens per second" and
    section 13.3's progress units, at the granularity a training loop actually emits)."""

    epoch: int
    mean_loss: float
    n_steps: int
    n_records: int
    n_events: int
    n_event_tokens: int
    seconds: float
    val_loss: float | None = None
    """Mean loss over `val_dataset` (`train`'s optional argument), if one was given."""

    @property
    def event_tokens_per_second(self) -> float:
        return self.n_event_tokens / self.seconds if self.seconds > 0 else 0.0


class PretrainingEngine:
    """Wires the dynamic sampler, masking, model, optimizer/scheduler, checkpointing,
    and MLflow logging into one train loop. Construction order matters: the model
    and optimizer must be built (and the optimizer given `model.parameters()`)
    *before* `accelerator.prepare()`, so DDP-wrapping the model afterward still
    shares the exact same parameter tensors the optimizer already holds references to.
    """

    def __init__(
        self,
        train_dataset: TokenizedRecordDataset,
        processor: PragmaProcessor,
        pragma_config: PragmaConfig,
        masking_config: MaskingConfig,
        training_config: TrainingConfig,
        token_budget_config: TokenBudgetConfig,
        *,
        processor_dir: Path,
        checkpoint_dir: Path | None = None,
        data_manifest_dir: Path | None = None,
    ) -> None:
        self.train_dataset = train_dataset
        self.processor = processor
        self.pragma_config = pragma_config
        self.masking_config = masking_config
        self.training_config = training_config
        self.token_budget_config = token_budget_config
        self.processor_dir = processor_dir
        self.data_manifest_dir = data_manifest_dir

        resolved_precision = resolve_mixed_precision(training_config.mixed_precision)
        self.accelerator = Accelerator(
            mixed_precision=resolved_precision,
            gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        )

        torch.manual_seed(training_config.seed)
        self.model = PragmaForMaskedModeling(pragma_config)
        self.optimizer = build_optimizer(self.model, training_config)

        self.sampler = TokenBudgetBatchSampler(
            train_dataset,
            token_budget_config,
            num_replicas=self.accelerator.num_processes,
            rank=self.accelerator.process_index,
        )
        steps_per_epoch = max(1, len(self.sampler))
        total_updates = training_config.total_optimizer_updates or max(
            1,
            (steps_per_epoch * training_config.n_epochs)
            // training_config.gradient_accumulation_steps,
        )
        self.scheduler = build_scheduler(
            self.optimizer, training_config, total_updates=total_updates
        )

        self.masking_planner = MaskingPlanner.from_registry(
            processor.registry, processor.key_vocab, masking_config
        )
        self.collator = PragmaCollator(masking_planner=self.masking_planner)

        self.model = self.accelerator.prepare(self.model)
        self.optimizer = self.accelerator.prepare(self.optimizer)

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir or Path(training_config.checkpoint_dir)
        )
        self.counters = TrainingCounters()

        self._mlflow_run = None
        if self.accelerator.is_main_process:
            # `mlflow`'s fluent API keeps a thread-local "active run" that survives
            # across `set_tracking_uri`/`set_experiment` calls — without explicitly
            # starting (and later closing) our own run, a second `PretrainingEngine`
            # built in the same process inherits a stale run ID pointing at a
            # *different* tracking store and every `log_metrics` call raises
            # "Run not found" (hit directly while testing resume). Starting our own
            # run here makes each engine instance self-contained regardless of what
            # ran before it in this process.
            if mlflow.active_run() is not None:
                mlflow.end_run()
            mlflow.set_tracking_uri(training_config.mlflow_tracking_uri)
            mlflow.set_experiment(training_config.mlflow_experiment_name)
            self._mlflow_run = mlflow.start_run()

    @property
    def mlflow_run_id(self) -> str | None:
        """This engine's MLflow run ID, for pulling back logged metric history after
        the fact (e.g. via `MlflowClient().get_metric_history(run_id, "loss")`) —
        `None` on non-main processes, which never start a run."""
        return self._mlflow_run.info.run_id if self._mlflow_run is not None else None

    def close(self) -> None:
        """Ends this engine's MLflow run, if any. Call when done training in a
        process that may go on to build another `PretrainingEngine` (e.g. tests,
        or a resume script that constructs a fresh engine) — a real training job
        that exits after `train()` doesn't strictly need this."""
        if self._mlflow_run is not None:
            mlflow.end_run()
            self._mlflow_run = None

    def _dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_sampler=self.sampler, collate_fn=self.collator)

    def resume(self, checkpoint_name: str) -> None:
        self.counters = self.checkpoint_manager.load(
            checkpoint_name,
            accelerator=self.accelerator,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            processor_dir=self.processor_dir,
        )
        self.sampler.set_epoch(self.counters.epoch)
        self.collator.set_epoch(self.counters.epoch)

    def save_checkpoint(
        self,
        name: str,
        *,
        validation_metrics: dict[str, float] | None = None,
        is_best: bool = False,
    ) -> Path | None:
        if not self.accelerator.is_main_process:
            return None
        return self.checkpoint_manager.save(
            name,
            accelerator=self.accelerator,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            processor_dir=self.processor_dir,
            pragma_config=self.pragma_config,
            masking_config=self.masking_config,
            training_config=self.training_config,
            counters=self.counters,
            data_manifest_dir=self.data_manifest_dir,
            validation_metrics=validation_metrics,
            is_best=is_best,
        )

    @torch.no_grad()
    def evaluate(self, dataset: TokenizedRecordDataset) -> float:
        """Mean MLM loss over `dataset`, no gradient/optimizer step — the "evaluate"
        half of section 13.1's "explicit Accelerate-based train/evaluate loop."

        Uses the same `MaskingPlanner`/epoch (masking is still applied — there is
        no unmasked validation objective to fall back to), a fixed epoch (`0`) so
        every call masks `dataset` identically and validation loss is comparable
        across training epochs, and a plain fixed-size `DataLoader` rather than
        the dynamic sampler, since evaluation only needs *a* consistent pass, not
        a training-shaped one.
        """
        self.model.eval()
        eval_collator = PragmaCollator(masking_planner=self.masking_planner)
        eval_collator.set_epoch(0)
        loader = DataLoader(
            dataset,
            batch_size=min(8, max(1, len(dataset))),
            shuffle=False,
            collate_fn=eval_collator,
        )

        loss_sum, n_batches = 0.0, 0
        for batch in loader:
            batch = batch.to(self.accelerator.device)
            out = self.model(batch)
            if out.loss is None:
                continue
            loss_sum += out.loss.item()
            n_batches += 1

        self.model.train()
        return loss_sum / max(n_batches, 1)

    def train(
        self,
        *,
        start_epoch: int = 0,
        end_epoch: int | None = None,
        val_dataset: TokenizedRecordDataset | None = None,
    ) -> list[EpochMetrics]:
        """Trains epochs `[start_epoch, end_epoch)` (default: through `training_config.n_epochs`).

        `end_epoch` lets a caller stop early (e.g. to simulate an interruption
        in a test) *without* changing `training_config.n_epochs` — the
        scheduler's warmup/decay horizon is derived from `n_epochs`
        (`total_optimizer_updates`), so a resumed run must keep the same
        `n_epochs` the original run used, even if it only trains a subset of
        those epochs before checkpointing. Confirmed directly while building
        the resume test: varying `n_epochs` between the "full" and "resumed"
        runs silently changes the LR schedule and makes their losses diverge
        for a real reason, not a resume bug.

        If `val_dataset` is given, `evaluate()` runs against it at the end of
        every epoch and the result is logged as `epoch_val_loss` and returned
        in that epoch's `EpochMetrics.val_loss`.
        """
        history: list[EpochMetrics] = []
        stop_epoch = self.training_config.n_epochs if end_epoch is None else end_epoch

        for epoch in range(start_epoch, stop_epoch):
            self.sampler.set_epoch(epoch)
            self.collator.set_epoch(epoch)
            self.model.train()

            loss_sum, n_steps, n_records, n_events, n_tokens = 0.0, 0, 0, 0, 0
            start_time = time.perf_counter()

            for batch in self._dataloader():
                batch = batch.to(self.accelerator.device)
                with self.accelerator.accumulate(self.model):
                    out = self.model(batch)
                    if out.loss is None:
                        continue
                    self.accelerator.backward(out.loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.training_config.max_grad_norm
                        )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                loss_sum += out.loss.item()
                n_steps += 1
                n_records += batch.n_records
                n_events += batch.n_events
                n_tokens += batch.n_event_tokens
                self.counters.global_step += 1
                self.counters.tokens_processed += batch.n_event_tokens
                self.counters.events_processed += batch.n_events
                self.counters.records_processed += batch.n_records

                self._maybe_log_step(out.loss.item())
                self._maybe_checkpoint(epoch)

            self.counters.epoch = epoch + 1
            elapsed = time.perf_counter() - start_time
            val_loss = self.evaluate(val_dataset) if val_dataset is not None else None
            metrics = EpochMetrics(
                epoch=epoch,
                mean_loss=loss_sum / max(n_steps, 1),
                n_steps=n_steps,
                n_records=n_records,
                n_events=n_events,
                n_event_tokens=n_tokens,
                seconds=elapsed,
                val_loss=val_loss,
            )
            history.append(metrics)
            if self.accelerator.is_main_process:
                epoch_log = {
                    "epoch_mean_loss": metrics.mean_loss,
                    "epoch_seconds": elapsed,
                    "epoch_event_tokens_per_second": metrics.event_tokens_per_second,
                }
                if val_loss is not None:
                    epoch_log["epoch_val_loss"] = val_loss
                mlflow.log_metrics(epoch_log, step=epoch)

        return history

    def _maybe_log_step(self, loss_value: float) -> None:
        if not self.accelerator.is_main_process:
            return
        if self.counters.global_step % self.training_config.log_every_n_steps != 0:
            return
        mlflow.log_metrics(
            {"loss": loss_value, "lr": self.optimizer.param_groups[0]["lr"]},
            step=self.counters.global_step,
        )

    def _maybe_checkpoint(self, epoch: int) -> None:
        if not self.accelerator.is_main_process:
            return
        every = self.training_config.checkpoint_every_n_steps
        if every <= 0 or self.counters.global_step % every != 0:
            return
        self.counters.epoch = epoch
        self.save_checkpoint("latest")
