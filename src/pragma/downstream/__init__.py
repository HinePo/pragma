"""LoRA adaptation and supervised task heads for downstream fine-tuning."""

from pragma.downstream.lora import build_lora_model, load_lora_model
from pragma.downstream.task_model import PragmaForTask, PragmaTaskOutput
from pragma.downstream.trainer import (
    DownstreamCollator,
    DownstreamEpochMetrics,
    DownstreamTrainer,
    load_frozen_backbone,
)

__all__ = [
    "DownstreamCollator",
    "DownstreamEpochMetrics",
    "DownstreamTrainer",
    "PragmaForTask",
    "PragmaTaskOutput",
    "build_lora_model",
    "load_frozen_backbone",
    "load_lora_model",
]
