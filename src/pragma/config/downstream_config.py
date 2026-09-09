"""`DownstreamConfig`: LoRA rank/target-modules and fine-tuning hyperparameters.

Implementation plan section 15.2, 11.4's `LoRAAdapterFactory`/`DownstreamTrainer`
rows, and section 18's "LoRA and downstream evaluation" open-ambiguity list
(LoRA dropout/learning rate/batch size/weight decay/training steps, exact
target-module names beyond QKV+MLP). ADR 0013 records the actual decisions;
this dataclass is where they live as configuration, not hardcoded constants,
per section 18's closing instruction ("must be recorded in ... configuration
files ... not remain implicit in code").
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DownstreamConfig:
    lora_r: int = 8
    """LoRA rank — section 15.2's "start with rank 8 and alpha 8"."""
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    )
    """QKV and MLP projection names (section 15.2) — every `PragmaTransformerBlock`
    across the profile/event/history encoders shares these names (ADR 0011),
    so PEFT's name-suffix matching finds them at every layer without listing
    layer indices explicitly."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    adam_betas: tuple[float, float] = (0.9, 0.999)
    warmup_ratio: float = 0.05
    batch_size: int = 8
    n_epochs: int = 10
    max_grad_norm: float = 1.0

    mixed_precision: str = "bf16"
    """Requested precision; `resolve_mixed_precision` downgrades it to what the
    current accelerator can actually support (ADR 0011, reused here)."""

    seed: int = 0
    modules_to_save: tuple[str, ...] = field(default_factory=lambda: ("classifier",))
    """Non-LoRA modules PEFT should also train and save (section 15.2's "save
    adapters and task heads separately from the immutable base checkpoint") —
    the task head lives in the same adapter directory as the LoRA weights,
    never inside the frozen base checkpoint."""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DownstreamConfig:
        data = dict(data)
        for key in ("target_modules", "modules_to_save"):
            if key in data:
                data[key] = tuple(data[key])
        if "adam_betas" in data:
            data["adam_betas"] = tuple(data["adam_betas"])
        return cls(**data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> DownstreamConfig:
        return cls.from_dict(json.loads(path.read_text()))
