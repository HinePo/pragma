"""`run_linear_probe` / `LinearProbeRunner`: frozen-embedding linear probes.

Implementation plan section 14.2, steps 3-4 ("Standard-scale embeddings
using only the downstream training partition. Fit logistic or linear probes
using fixed splits.") and section 11.4's `LinearProbeRunner` row.
`run_linear_probe` is the single-variant primitive Phase 9 introduced to
check its own exit gate; `LinearProbeRunner` (Phase 10) runs it across every
`EmbeddingExtractor` variant (`usr`, `last_event`, `concat`) so they can be
compared against each other and against `pragma.evaluation.baselines` in one
report (`pragma.evaluation.report.build_comparison_report`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from pragma.evaluation.embeddings import EmbeddingBundle


@dataclass(frozen=True)
class ProbeResult:
    name: str = "probe"
    """Which model/variant this result is for (e.g. `"probe_usr"`,
    `"baseline_logreg"`) - `pragma.evaluation.baselines` reuses this same
    dataclass shape for conventional baselines so both fit in one
    comparison report."""
    auc: float = float("nan")
    """`ROC AUC` on the validation partition; `nan` if a partition ended up
    single-class (too few records to probe meaningfully - see `n_train`/`n_val`)."""
    n_train: int = 0
    n_val: int = 0


def run_linear_probe(
    entity_ids: list[str],
    embeddings: torch.Tensor,
    labels: dict[str, bool],
    *,
    val_entity_ids: set[str],
    seed: int = 0,
    name: str = "probe",
) -> ProbeResult:
    """Fits a standard-scaled logistic-regression probe on the entities *not*
    in `val_entity_ids` and reports AUC on those that are.

    `val_entity_ids` should be the same entity split already used as the
    pretraining validation set (e.g. `{r.entity_id for r in val_dataset}`) -
    reusing the existing point-in-time train/val split means the probe never
    fits on embeddings the val AUC is computed from, without introducing a
    second split to keep consistent with the first.
    """
    y = np.array([int(labels[eid]) for eid in entity_ids])
    x = embeddings.numpy()
    is_val = np.array([eid in val_entity_ids for eid in entity_ids])

    x_train, y_train = x[~is_val], y[~is_val]
    x_val, y_val = x[is_val], y[is_val]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        # A degenerate split (e.g. a tiny debug corpus) can't support a
        # meaningful probe - report it honestly rather than raising or faking
        # a score.
        return ProbeResult(name=name, auc=float("nan"), n_train=len(y_train), n_val=len(y_val))

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(scaler.transform(x_train), y_train)
    probs = clf.predict_proba(scaler.transform(x_val))[:, 1]
    return ProbeResult(
        name=name,
        auc=float(roc_auc_score(y_val, probs)),
        n_train=len(y_train),
        n_val=len(y_val),
    )


class LinearProbeRunner:
    """Runs `run_linear_probe` on every `EmbeddingExtractor` variant in an
    `EmbeddingBundle` against the same labels/split - section 11.4's
    `LinearProbeRunner`."""

    def __init__(self, *, seed: int = 0) -> None:
        self.seed = seed

    def run(
        self,
        bundle: EmbeddingBundle,
        labels: dict[str, bool],
        *,
        val_entity_ids: set[str],
    ) -> list[ProbeResult]:
        variants = {"usr": bundle.usr, "last_event": bundle.last_event, "concat": bundle.concat}
        return [
            run_linear_probe(
                bundle.entity_ids,
                embeddings,
                labels,
                val_entity_ids=val_entity_ids,
                seed=self.seed,
                name=f"probe_{variant_name}",
            )
            for variant_name, embeddings in variants.items()
        ]
