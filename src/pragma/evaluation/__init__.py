"""Embedding extraction, linear probes, conventional baselines, reports."""

from pragma.evaluation.baselines import run_baselines
from pragma.evaluation.embeddings import (
    EmbeddingBundle,
    EmbeddingExtractor,
    extract_record_embeddings,
)
from pragma.evaluation.features import build_aggregated_features
from pragma.evaluation.probe import LinearProbeRunner, ProbeResult, run_linear_probe
from pragma.evaluation.report import build_comparison_report

__all__ = [
    "EmbeddingBundle",
    "EmbeddingExtractor",
    "LinearProbeRunner",
    "ProbeResult",
    "build_aggregated_features",
    "build_comparison_report",
    "extract_record_embeddings",
    "run_baselines",
    "run_linear_probe",
]
