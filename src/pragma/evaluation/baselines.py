"""`run_baselines`: conventional-feature baselines for the section 14.3 comparison.

Implementation plan section 14.3: "Aggregated features plus LightGBM or
another strong GBDT" and "Logistic regression on conventional aggregated
features" must be evaluated on the same point-in-time inputs, labels, and
splits as the PRAGMA embedding probes (`pragma.evaluation.probe`), so the
final report (`pragma.evaluation.report`) compares like with like.

**GBDT choice**: `sklearn.ensemble.HistGradientBoostingClassifier`, not
LightGBM. The plan explicitly allows "LightGBM or another strong GBDT";
`HistGradientBoostingClassifier` is a strong histogram-based GBDT already
available through the `scikit-learn` dependency Phase 9 added for the linear
probe, avoiding a second, heavier native-code dependency (LightGBM ships its
own compiled library) for a project that already runs entirely on this one
dev machine's CPU. See ADR 0012.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pragma.evaluation.probe import ProbeResult

_CATEGORICAL_COLUMNS = ("country", "plan", "kyc_level", "age_band", "balance_quantile")


def _preprocessor(features_df: pd.DataFrame) -> ColumnTransformer:
    categorical = [c for c in _CATEGORICAL_COLUMNS if c in features_df.columns]
    numeric = [c for c in features_df.columns if c not in categorical]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )


def run_baselines(
    features_df: pd.DataFrame,
    labels: dict[str, bool],
    *,
    val_entity_ids: set[str],
    seed: int = 0,
) -> list[ProbeResult]:
    """Fits logistic regression and a GBDT on `features_df` (from
    `build_aggregated_features`, indexed by `entity_id`) against `labels`,
    using the same `val_entity_ids` split every embedding probe uses.

    Returns one `ProbeResult` per model (`"baseline_logreg"`,
    `"baseline_gbdt"`) - the same dataclass `pragma.evaluation.probe` uses,
    so both fit in one `pragma.evaluation.report.build_comparison_report` call.
    """
    entity_ids = list(features_df.index)
    y = np.array([int(labels[eid]) for eid in entity_ids])
    is_val = np.array([eid in val_entity_ids for eid in entity_ids])

    x_train_df, y_train = features_df.iloc[~is_val], y[~is_val]
    x_val_df, y_val = features_df.iloc[is_val], y[is_val]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        # A degenerate split (e.g. a tiny debug corpus) can't support a
        # meaningful baseline either - report it honestly, same as `run_linear_probe`.
        return [
            ProbeResult(name=name, auc=float("nan"), n_train=len(y_train), n_val=len(y_val))
            for name in ("baseline_logreg", "baseline_gbdt")
        ]

    results = []
    models = {
        "baseline_logreg": LogisticRegression(max_iter=1000, random_state=seed),
        "baseline_gbdt": HistGradientBoostingClassifier(random_state=seed),
    }
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", _preprocessor(features_df)), ("model", model)])
        pipeline.fit(x_train_df, y_train)
        probs = pipeline.predict_proba(x_val_df)[:, 1]
        results.append(
            ProbeResult(
                name=name,
                auc=float(roc_auc_score(y_val, probs)),
                n_train=len(y_train),
                n_val=len(y_val),
            )
        )
    return results
