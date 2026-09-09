"""`SchedulerFactory`: warmup + cosine decay or constant-with-warmup (section 11.4, 13.2).

Built from optimizer-update count (`total_updates`), not epochs — dynamic
batches contain different numbers of records per step, so epochs are not a
stable scale coordinate (section 13.3).

A small custom scheduler, not `torch.optim.lr_scheduler.LambdaLR`: `LambdaLR`
requires `isinstance(optimizer, torch.optim.Optimizer)`, which `HybridOptimizer`
deliberately isn't (it composes two independent real optimizers rather than
faking a single flat parameter list — ADR 0011). This scheduler only needs
`optimizer.param_groups` to be a list of dicts with an `'lr'` key, which both
`torch.optim.AdamW` and `HybridOptimizer` satisfy identically.
"""

from __future__ import annotations

import math
from typing import Any

from torch.optim import AdamW

from pragma.config.training_config import TrainingConfig
from pragma.training.optimizer import HybridOptimizer

Optimizer = AdamW | HybridOptimizer


def _cosine_factor(step: int, *, warmup_steps: int, total_updates: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_updates - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _constant_factor(step: int, *, warmup_steps: int) -> float:
    return min(1.0, step / max(1, warmup_steps))


class WarmupDecayScheduler:
    """Multiplies each param group's own initial LR by a shared warmup/decay factor."""

    def __init__(
        self, optimizer: Optimizer, *, warmup_steps: int, total_updates: int, kind: str
    ) -> None:
        if kind not in ("cosine", "constant"):
            raise ValueError(f"unknown scheduler: {kind!r}")
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_updates = total_updates
        self.kind = kind
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.last_step = -1
        self.step()

    def _factor(self, step: int) -> float:
        if self.kind == "cosine":
            return _cosine_factor(
                step, warmup_steps=self.warmup_steps, total_updates=self.total_updates
            )
        return _constant_factor(step, warmup_steps=self.warmup_steps)

    def step(self) -> None:
        self.last_step += 1
        self._apply_current_factor()

    def _apply_current_factor(self) -> None:
        factor = self._factor(self.last_step)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs, strict=True):
            group["lr"] = base_lr * factor

    def state_dict(self) -> dict[str, Any]:
        return {"last_step": self.last_step, "base_lrs": self.base_lrs}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.last_step = state_dict["last_step"]
        self.base_lrs = state_dict["base_lrs"]
        # A freshly (re)constructed optimizer's param groups still hold their
        # pre-scheduler LR until a step() call re-derives it from base_lrs — do
        # that now rather than waiting for the next training step, so the
        # restored optimizer is immediately consistent with the restored counters.
        self._apply_current_factor()


def build_scheduler(
    optimizer: Optimizer, config: TrainingConfig, *, total_updates: int
) -> WarmupDecayScheduler:
    warmup_steps = max(1, int(total_updates * config.warmup_ratio))
    return WarmupDecayScheduler(
        optimizer, warmup_steps=warmup_steps, total_updates=total_updates, kind=config.scheduler
    )
