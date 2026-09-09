"""`TrainingConfig`: optimizer, scheduler, precision, batching, logging, checkpointing.

Implementation plan, section 11.1, 13.2. Deliberately has no `device` or
`use_ddp` field — hardware selection is entirely `Accelerate`'s job
(ADR 0011); this config only controls things that are true policy choices
regardless of what hardware ends up running them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class TokenBudgetConfig:
    """Dynamic batch-sampler budget (section 9.3)."""

    max_event_tokens_per_batch: int = 4096
    max_events_per_batch: int = 2048
    max_records_per_batch: int = 256
    n_length_buckets: int = 8
    seed: int = 0


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: Literal["adamw", "muon_adamw"] = "adamw"
    learning_rate: float = 3e-4
    muon_learning_rate: float = 0.02
    """Only for `optimizer == "muon_adamw"` — Muon's LR scale differs from AdamW's (ADR 0011)."""
    weight_decay: float = 0.01
    muon_momentum: float = 0.95
    adam_betas: tuple[float, float] = (0.9, 0.999)

    scheduler: Literal["cosine", "constant"] = "cosine"
    warmup_ratio: float = 0.01
    """Fraction of total optimizer updates spent warming up (section 13.2)."""

    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    """Requested precision; `resolve_mixed_precision` downgrades this to what the
    current accelerator can actually support (ADR 0011)."""
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0

    n_epochs: int = 1
    total_optimizer_updates: int | None = None
    """If set, the scheduler's decay horizon; otherwise derived from `n_epochs` x steps/epoch."""

    checkpoint_every_n_steps: int = 100
    log_every_n_steps: int = 10
    checkpoint_dir: str = "checkpoints"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    """Local SQLite file, no tracking server — MLflow 3.x deprecated the plain
    `file:./mlruns` store (confirmed directly: it now raises unless
    `MLFLOW_ALLOW_FILE_STORE=true` is set); sqlite keeps the "no server
    dependency" property ADR 0011 calls for without relying on that escape hatch."""
    mlflow_experiment_name: str = "pragma-pretraining"

    seed: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        data = dict(data)
        if "adam_betas" in data:
            data["adam_betas"] = tuple(data["adam_betas"])
        return cls(**data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> TrainingConfig:
        return cls.from_dict(json.loads(path.read_text()))
