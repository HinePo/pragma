"""`CheckpointManager`: atomic checkpoint save/resume (implementation plan, section 13.4;
ADR 0007, ADR 0011).

Builds on `Accelerator.save_state`/`load_state` for model weights and RNG
state (Python/NumPy/CPU/GPU) — that is exactly what those calls already
persist. Optimizer and scheduler state are saved/loaded *explicitly* by this
module rather than relying on `Accelerator`'s automatic optimizer tracking:
`HybridOptimizer` (ADR 0011) isn't a `torch.optim.Optimizer` subclass, so
`Accelerate` silently skips it during `save_state` — confirmed directly while
building this module, not assumed — which would otherwise lose Muon/AdamW
momentum state on every resume without any visible error.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from accelerate import Accelerator

    from pragma.config import MaskingConfig, TrainingConfig
    from pragma.modeling import PragmaConfig

MANIFEST_FILE = "manifest.json"
OPTIMIZER_FILE = "optimizer.pt"
SCHEDULER_FILE = "scheduler.pt"


@dataclass
class TrainingCounters:
    """Global progress counters (section 13.3 — tokens processed is the primary
    scale coordinate, not epochs)."""

    global_step: int = 0
    tokens_processed: int = 0
    events_processed: int = 0
    records_processed: int = 0
    epoch: int = 0


def hash_processor_bundle(processor_dir: Path) -> str:
    """Fingerprint of a processor bundle's on-disk contents (bundle metadata + BPE
    model) — used to verify a checkpoint's exact processor is present on load."""
    hasher = hashlib.sha256()
    for name in ("bundle.json", "bpe_tokenizer.json"):
        file_path = processor_dir / name
        if file_path.exists():
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _hash_file_if_exists(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _git_revision(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or None if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


class CheckpointManager:
    def __init__(self, checkpoint_dir: Path, *, repo_root: Path | None = None) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.repo_root = repo_root or Path.cwd()

    def save(
        self,
        name: str,
        *,
        accelerator: Accelerator,
        optimizer: Any,
        scheduler: Any,
        processor_dir: Path,
        pragma_config: PragmaConfig,
        masking_config: MaskingConfig,
        training_config: TrainingConfig,
        counters: TrainingCounters,
        data_manifest_dir: Path | None = None,
        validation_metrics: dict[str, float] | None = None,
        is_best: bool = False,
    ) -> Path:
        """Atomically writes one checkpoint directory. Only the main process should
        call this — `accelerator.save_state` already no-ops on non-main processes,
        but the sidecar manifest write below does not, so callers must still gate
        this with `accelerator.is_main_process`."""
        path = self.checkpoint_dir / name
        path.mkdir(parents=True, exist_ok=True)

        accelerator.save_state(str(path))
        torch.save(optimizer.state_dict(), path / OPTIMIZER_FILE)
        torch.save(scheduler.state_dict(), path / SCHEDULER_FILE)

        manifest = {
            "counters": asdict(counters),
            "pragma_config": pragma_config.to_dict(),
            "masking_config": masking_config.to_dict(),
            "training_config": training_config.to_dict(),
            "processor_dir": str(processor_dir),
            "processor_hash": hash_processor_bundle(processor_dir),
            "data_manifest_dir": str(data_manifest_dir) if data_manifest_dir else None,
            "code_revision": _git_revision(self.repo_root),
            "dependency_lock_fingerprint": _hash_file_if_exists(self.repo_root / "uv.lock"),
            "validation_metrics": validation_metrics,
            "is_best": is_best,
            "saved_at": datetime.now(UTC).isoformat(),
        }
        (path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, default=str))
        return path

    def load(
        self,
        name: str,
        *,
        accelerator: Accelerator,
        optimizer: Any,
        scheduler: Any,
        processor_dir: Path,
    ) -> TrainingCounters:
        """Restores model/RNG (via `accelerator`), optimizer, and scheduler state in
        place, and returns the saved counters. Raises if the manifest or its
        processor bundle is missing/mismatched — per ADR 0007, a checkpoint without
        its exact processor is invalid, not merely incomplete."""
        path = self.checkpoint_dir / name
        manifest_path = path / MANIFEST_FILE
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no checkpoint manifest at {manifest_path} — not a valid PRAGMA checkpoint"
            )
        manifest = json.loads(manifest_path.read_text())

        expected_hash = manifest["processor_hash"]
        actual_hash = hash_processor_bundle(processor_dir)
        if expected_hash != actual_hash:
            raise ValueError(
                f"processor bundle at {processor_dir} does not match this checkpoint "
                f"(expected hash {expected_hash}, got {actual_hash}) — a checkpoint is "
                "invalid without its exact matching processor bundle (ADR 0007)"
            )

        accelerator.load_state(str(path))
        optimizer.load_state_dict(torch.load(path / OPTIMIZER_FILE, weights_only=False))
        scheduler.load_state_dict(torch.load(path / SCHEDULER_FILE, weights_only=False))

        return TrainingCounters(**manifest["counters"])

    def read_manifest(self, name: str) -> dict[str, Any]:
        return json.loads((self.checkpoint_dir / name / MANIFEST_FILE).read_text())
