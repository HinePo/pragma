"""Configuration dataclasses for model, processor, masking, and training."""

from pragma.config.downstream_config import DownstreamConfig
from pragma.config.masking_config import MaskingConfig
from pragma.config.processor_config import ProcessorConfig
from pragma.config.training_config import TokenBudgetConfig, TrainingConfig

__all__ = [
    "DownstreamConfig",
    "MaskingConfig",
    "ProcessorConfig",
    "TokenBudgetConfig",
    "TrainingConfig",
]
