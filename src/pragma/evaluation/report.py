"""`build_comparison_report`: one table for probes, LoRA runs, and baselines.

Implementation plan section 11.4's `ModelEvaluator` ("produces comparable
offline reports across checkpoints, probes, LoRA runs, and baselines") and
Phase 10/11's exit gates ("the team can state where pretraining helps",
"LoRA is compared against the frozen probe and conventional baseline on the
same split"). Deliberately just the slice needed so far, not the full
`ModelEvaluator` class — there is nothing to poll across *multiple*
checkpoints yet (only ever one checkpoint's results are compared per call).
"""

from __future__ import annotations

import pandas as pd

from pragma.evaluation.probe import ProbeResult


def _kind(name: str) -> str:
    if name.startswith("probe_"):
        return "probe"
    if name.startswith("baseline_"):
        return "baseline"
    return "lora"


def build_comparison_report(results: list[ProbeResult], *, checkpoint_name: str) -> pd.DataFrame:
    """Combines probe (`"probe_usr"`, `"probe_last_event"`, `"probe_concat"`),
    baseline (`"baseline_logreg"`, `"baseline_gbdt"`), and any other
    (Phase 11's `"lora_finetuned"`) `ProbeResult`s into one table, sorted by
    AUC descending (`NaN` sorts last — a degenerate split's result never
    outranks a real one). `checkpoint_name` tags every row that depends on a
    specific pretrained checkpoint (probes and LoRA runs both do; baselines
    don't, since they never see the backbone at all) — keeps every row
    traceable to exactly which comparison run produced it.
    """
    rows = [
        {
            "checkpoint": "n/a (baseline)" if r.name.startswith("baseline_") else checkpoint_name,
            "kind": _kind(r.name),
            "name": r.name,
            "auc": r.auc,
            "n_train": r.n_train,
            "n_val": r.n_val,
        }
        for r in results
    ]
    report = pd.DataFrame(rows)
    return report.sort_values("auc", ascending=False, na_position="last").reset_index(drop=True)
