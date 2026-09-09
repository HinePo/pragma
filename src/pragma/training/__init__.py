"""Accelerate-based pretraining engine, optimizer/scheduler factories, checkpointing."""

from pragma.training.debug_loop import (
    DebugTrainingConfig,
    DebugTrainingResult,
    context_dependency_grad,
    run_debug_training,
    shuffle_context_tokens,
)
from pragma.training.engine import EpochMetrics, PretrainingEngine
from pragma.training.hardware import resolve_mixed_precision
from pragma.training.muon import Muon
from pragma.training.optimizer import HybridOptimizer, build_optimizer
from pragma.training.scheduler import WarmupDecayScheduler, build_scheduler

__all__ = [
    "DebugTrainingConfig",
    "DebugTrainingResult",
    "EpochMetrics",
    "HybridOptimizer",
    "Muon",
    "PretrainingEngine",
    "WarmupDecayScheduler",
    "build_optimizer",
    "build_scheduler",
    "context_dependency_grad",
    "resolve_mixed_precision",
    "run_debug_training",
    "shuffle_context_tokens",
]
