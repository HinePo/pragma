# ADR 0012: Evaluation harness decisions (embedding probes and baselines)

**Status:** Accepted (Phase 10)

## Context

Section 14.2/14.3 and section 11.4 leave several Phase 10 questions open:
which conventional baselines count as "agreed," which GBDT implementation to
use (the plan says "LightGBM or another strong GBDT"), what a "reproducible
comparison report" actually looks like, and what feature set a fair
conventional baseline gets to see. Phase 9 already shipped a single-function
probe (`run_linear_probe`) to check its own exit gate; Phase 10 needs the
fuller harness section 11.4 names (`EmbeddingExtractor`, `LinearProbeRunner`,
`BaselineRunner`-equivalent, a comparison report) without introducing new
native-code dependencies on a project that only ever runs on this one dev
machine's CPU.

## Decisions

1. **GBDT choice: `sklearn.ensemble.HistGradientBoostingClassifier`, not
   LightGBM.** The plan explicitly allows "another strong GBDT." LightGBM
   ships its own compiled native library; `HistGradientBoostingClassifier`
   is a strong histogram-based GBDT already available through the
   `scikit-learn` dependency Phase 9 added for the linear probe, so no new
   dependency is needed. Revisit if a later phase needs LightGBM-specific
   features (e.g. native categorical splits at larger scale, GPU training).
2. **The conventional-baseline feature set (`build_aggregated_features`) is
   deliberately simple and hand-built**: event counts/totals (`n_events`,
   `n_distinct_event_types`, `total_amount`, `mean_amount`), tenure
   (`days_since_signup`), lifelong-milestone flags, and the static
   categorical profile attributes — read from the exact same
   `EvaluationRecord`s the tokenizer sees (`events_before_evaluation`,
   `profile_state`), so the comparison is apples-to-apples on point-in-time
   correctness. It intentionally does *not* try to be a strong, tuned
   feature-engineering baseline — the point of section 14.3 is "does the
   shared backbone beat a simple conventional approach," and a weak
   conventional baseline that PRAGMA still needs to beat is a stronger,
   more honest signal than not having one at all.
3. **`EmbeddingExtractor` returns `[USR]`, last `[EVT]`, and their
   concatenation in one pass** (`EmbeddingBundle`), reusing
   `PragmaModel.forward`'s existing `event_embeddings`/`history_cu_seqlens`
   to slice out each record's final event state — no model change needed.
   A zero-event evaluation record's `last_event` is an all-zero vector, not
   a dropped row or a NaN: "no history yet" is a real, valid state a probe
   or baseline can (and should) see, and dropping those rows would silently
   change what population is being evaluated.
4. **Probes and baselines share one result shape** (`ProbeResult`: `name`,
   `auc`, `n_train`, `n_val`) so `build_comparison_report` can combine and
   rank them without a separate baseline-specific type. `run_baselines`
   constructs the same dataclass `pragma.evaluation.probe` defines, tagged
   `"baseline_logreg"`/`"baseline_gbdt"`, rather than inventing a parallel
   `BaselineResult`.
5. **The comparison report is a single, checkpoint-scoped table, not yet a
   full `ModelEvaluator`.** Section 11.4's `ModelEvaluator` is described as
   producing comparable reports "across checkpoints, probes, LoRA runs, and
   baselines" — the LoRA-run column has nothing to populate until Phase 11
   builds `PragmaForTask`, and there is no multi-checkpoint sweep yet (only
   ever one checkpoint compared per call). `build_comparison_report` is
   deliberately just the probes-vs-baselines slice needed now; extending it
   to accept a LoRA row and multiple checkpoints is Phase 11's job, not a
   rewrite — the shared `ProbeResult` shape (decision 4) is exactly what
   makes that extension additive.
6. **AUC is the one comparison metric across every probe/baseline
   variant.** `_downstream_is_high_value` (Phase 9's synthetic label) is
   binary and roughly balanced by construction (median split), so ROC AUC
   is well-defined and comparable across every result the same way section
   14.1 already avoids picking a single interpretable perplexity number for
   a heterogeneous vocabulary.

## Consequences

- Adding a real downstream label later (e.g. from Phase 1's outstanding real
  client extract) only requires implementing a new
  `dict[str, bool]`-shaped labels source — every probe/baseline/report
  function already takes labels as a plain dict, never coupled to
  `_downstream_is_high_value` specifically.
- `build_aggregated_features`' feature set is intentionally not
  comprehensive — if PRAGMA's embeddings ever *lose* to this baseline, that
  is a real, actionable signal (per section 14.3, "the project is
  successful only if the shared representation provides value relative to
  the cost and maintenance of these baselines"), not evidence the baseline
  was engineered unfairly strong.
- `HistGradientBoostingClassifier` and `LogisticRegression` both run through
  the same `ColumnTransformer` (impute + scale/one-hot) for simplicity; a
  GBDT does not strictly need scaled numeric features, so this is a minor,
  acceptable simplification rather than a maximally-tuned baseline.
