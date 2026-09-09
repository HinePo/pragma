# PRAGMA Implementation Progress

Tracks status against the phased development order in
[PRAGMA-Implementation-Plan.md](PRAGMA-Implementation-Plan.md), section 17.
This file is the single place to check "where did we stop" and "what's
next" — see `CLAUDE.md` for how it's meant to be used and kept up to date.

**Current status: Phase 11 complete. Next up: Phase 12 (optimize and decide on scale-up).**

---

## Done and validated

### Phase 0 — Freeze contracts and decisions

**Build**
- [x] ADRs for data schema, evaluation points, processor, packed batches, masking semantics, attention backends, checkpoint format — `docs/adr/0001`–`0007`.
- [x] ADR 0008 added later (Phase 2) for token ID space and evaluation-point sampling decisions the plan left open.
- [x] Repository structure (`src/pragma/`, `tests/`, `configs/`, `scripts/`), dependency lock (`uv`/`pyproject.toml`), linting (`ruff`), typing (`mypy`, `disallow_untyped_defs`), unit-test framework (`pytest`).
- [x] `CLAUDE.md` written (three-layer architecture, documentation standard, this progress-tracking convention).

**Exit gate:** met — canonical record/batch contracts agreed (ADRs), every plan ambiguity found so far is either decided (ADR) or config.

### Phase 1 — Create synthetic and curated sample data

**Build**
- [x] Synthetic histories covering all value types and edge cases — `src/pragma/data/synthetic.py` (zero-event, single-event, long-history, same-timestamp cohorts; rare/OOV categorical values).
- [x] Schema registry — `src/pragma/schema/registry.py` (`SchemaRegistry.default()`), raw validation reports (`validate_corpus`).
- [ ] A small de-identified **real** client extract with known evaluation points — not started; MVP is running on synthetic data only so far.

**Exit gate:** met for synthetic data — deterministic reconstruction, leakage/schema tests pass (`tests/unit/test_synthetic_data.py`). Real-data field ownership/privacy approval is N/A until a real extract exists.

### Phase 2 — Implement the fitted processor

**Build**
- [x] Key vocabulary — `src/pragma/processing/vocabulary.py`.
- [x] Numeric bucketizer (global shared bucket tokens, per-key boundaries) — `src/pragma/processing/numeric.py`.
- [x] Categorical encoder (per-key namespaced vocab, shared `[UNK]` fallback) — `src/pragma/processing/categorical.py`.
- [x] Shared byte-level BPE encoder — `src/pragma/processing/text_bpe.py`.
- [x] Temporal features (soft-log transform, calendar cyclical features) — `src/pragma/processing/temporal.py`.
- [x] Special-token registry — `src/pragma/processing/special_tokens.py`.
- [x] `PragmaProcessor` coordinator + versioned save/load bundle — `src/pragma/processing/processor.py`.
- [x] `PointInTimeRecordBuilder` (`EvaluationRecord`, `ProfileState`, `EventRecord`) — `src/pragma/data/records.py` (pulled forward from Phase 1/ADR 0002 since the processor needs it to be testable).
- [x] `drop_zero_event_records()` (`src/pragma/data/records.py`) — added later (post-Phase 11) per ADR 0014, once a review found zero-event customers (which the paper discards during pretraining) were reaching `fit_processor.py`/`tokenize_shards.py`/`run_probe.py` unfiltered. Wired into all three; `data/shards/` now has zero zero-event records in any split (verified in `003_tokenize_shards.ipynb`).
- [x] Offline tokenized-record writer + data manifest — `scripts/fit_processor.py`, `scripts/tokenize_shards.py` (superseded by Phase 3's Parquet format; JSONL was a placeholder).
- [x] ADR 0008: token ID-space layout, milestone tokenization, evaluation-point sampling, train/val/test split policy.
- [x] Notebooks: `001_point_in_time_records.ipynb`, `002_fit_processor.ipynb`.

**Exit gate:** met — save/load round-trip is byte-identical (tested), fitting provably train-split-only, OOV/truncation/coverage inspected in `002_fit_processor.ipynb` and covered by `tests/unit/test_processor.py`, `test_numeric_bucketizer.py`, `test_categorical_encoder.py`, `test_text_bpe.py`, `test_records.py`. `test_split_assignment_is_pairwise_disjoint_across_entities` (added post-Phase 11, `tests/unit/test_records.py`) directly asserts the train/val/test entity-ID sets never intersect, rather than relying on `_entity_split`'s hash implementation being read carefully by future changes.

### Phase 3 — Implement record storage and reference batching

**Build**
- [x] Tokenized Parquet/Arrow shards with nested-list columns — `src/pragma/data/storage.py` (`PragmaRecordStore` interface, `ParquetShardStore`), sharded by `(split, event-count bucket)` per section 9.3.
- [x] Data manifest + shard index with checksums — `DataManifest`, `ShardInfo` in `storage.py`.
- [x] Reference dataset — `TokenizedRecordDataset` (`src/pragma/data/dataset.py`), a plain `torch.utils.data.Dataset` for use with a standard `DataLoader`.
- [x] `PragmaBatch` contract (packed buffers, `event_cu_seqlens`, `history_cu_seqlens`, `event_to_record`, temporal/calendar features, placeholder `event_mlm_labels`) — `src/pragma/data/batch.py`.
- [x] `PragmaCollator` + `PragmaBatch.validate()` for boundary/mapping validation.
- [x] `scripts/tokenize_shards.py` updated to the Parquet format.
- [x] Notebooks: `003_tokenize_shards.ipynb` (rebuilt for Parquet), `004_reference_batching.ipynb` (new).

**Exit gate:** met — `tests/unit/test_batch.py` and `004_reference_batching.ipynb` both prove batches reconstruct source records exactly and cross-event/cross-record boundaries hold, including for zero-event entities and mixed batches. `tests/unit/test_storage.py` covers round-trip, split filtering, and checksum-corruption detection.

**Note:** the padded collator called out in section 17's Phase 3 "Build" list is not separately implemented — `PragmaBatch`/`PragmaCollator` here *is* the packed reference path. A `PaddedAttentionBackend` (section 9.2) is still pending and belongs to Phase 5, since padding is a model/attention concern, not a data-batching one.

### Phase 4 — Implement structured masking

**Build**
- [x] `MaskingPlanner`: samples individual-token, whole-event, and semantic-key masks per ADR 0005 (independent sampling, union combination, record-wide key masking) — `src/pragma/masking/planner.py`.
- [x] `MaskingConfig` (probabilities, `unk_dropout_frac`, seed) — `src/pragma/config/masking_config.py`.
- [x] Corruption logic: `[MASK]` for standard objective positions (label = original value), `[UNK]` for a configurable input-dropout fraction (default 5%, label `-100`, excluded from loss), `-100` for every non-selected position.
- [x] Wired `MaskingPlanner` into `PragmaCollator` — populates `PragmaBatch.event_mlm_labels` for real and adds a new `event_mask_origin` bitflag field (`MaskSource`) for diagnostics — `src/pragma/data/batch.py`.
- [x] Deterministic per-`(seed, entity_id, epoch)` masking; `PragmaCollator.set_epoch()` (mirrors `DistributedSampler.set_epoch`) reshuffles masks across epochs.
- [x] Notebook: `005_masking_inspection.ipynb` — token-by-token inspection of one record, corpus-wide mask-source coverage/overlap, `PragmaCollator` integration, determinism/epoch-variation check.

**Exit gate:** met — `tests/unit/test_masking.py` (11 tests) and `tests/unit/test_batch_masking.py` (5 tests) cover every section 16.3 property (corruption, `-100` labeling, `[UNK]`-dropout labeling, whole-event/key coverage, determinism, epoch variation); `005_masking_inspection.ipynb` visually confirms no target leaks into the input on real records.

**Correction made during this phase:** ADR 0005 originally stated the original value ID is saved to `mlm_labels` for *both* `[MASK]` and `[UNK]`-dropout positions. That contradicted the plan's section 8.2 and 16.3, which are explicit that `[UNK]`-dropout positions get label `-100` (excluded from the loss — that's what makes it "dropout" rather than another prediction target). ADR 0005 has been corrected in place; the implementation follows the corrected rule.

### Phase 5 — Implement the padded reference model

**Build**
- [x] `SharedKeyValueEmbedding`, `WithinFieldPositionEncoding` — `src/pragma/modeling/embeddings.py`. Keys/values/special tokens share one embedding table (Phase 2's ID layout already makes this possible); a `[USR]`/`[EVT]` slot reuses the same `E(k)+E(v)+P(0)` formula with `k=v=special_id` (ADR 0009).
- [x] `ContinuousRoPE` (GPT-NeoX/RoFormer rotate-half, base 10000.0) — `src/pragma/modeling/rope.py`.
- [x] `CalendarEncoder` (`6 -> hidden_size -> hidden_size`, GELU) — `src/pragma/modeling/calendar.py`.
- [x] Generic packed-sequence primitives (`prepend_vector`, `group_starts`, `unprepend_vector`, `group_index_per_token`, `cu_seqlens_from_group_ids`) reused by all three encoders — `src/pragma/modeling/packing.py` (ADR 0009).
- [x] `ProfileStateEncoder`, `EventEncoder`, `HistoryEncoder`, `PragmaTransformerBlock` (explicit `q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2` naming for future LoRA targeting) — `src/pragma/modeling/encoders.py`, `transformer_block.py`.
- [x] `PragmaMLMHead`: concatenates local/contextual-event/contextual-user states, projects, and matches against the tied **value-vocabulary slice only** (`value_vocab_start:`, not the full vocab — ADR 0009) — `src/pragma/modeling/mlm_head.py`.
- [x] `AttentionBackend` interface + `PaddedAttentionBackend` (split-pad-mask-SDPA-unpad) — `src/pragma/attention/`.
- [x] `PragmaConfig`, `PragmaModel`, `PragmaForMaskedModeling` as HF-compatible (`PreTrainedConfig`/`PreTrainedModel`) classes, `PragmaOutput` (`ModelOutput`) — `src/pragma/modeling/config.py`, `model.py`, `outputs.py`.
- [x] ADR 0009: RoPE base, special-token embedding convention, `[USR]`/`[EVT]` RoPE coordinate (`0.0`), `CalendarEncoder` dims, MLM logit scope, weight init (BERT/GPT-2-style), padded-backend implementation approach.
- [x] Notebook: `006_model_architecture.ipynb` — builds config from a fitted processor, runs forward+backward on a real masked batch, checks param count at both corpus and paper vocab scale, demonstrates isolation.

**Exit gate:** met — `tests/unit/test_modeling.py` (7 tests) covers tensor shapes, ~10M params at paper-scale vocab (9.10M measured, within 8–12M), Event/History Encoder isolation (solo vs. batched), profile/event summary position mapping, MLM head local/contextual state correctness, and gradient finiteness/connectivity for temporal coordinates. `tests/unit/test_temporal.py` (10 tests, closing a Phase 2 test gap) covers calendar-feature periodicity. `006_model_architecture.ipynb` confirms all of this end to end on real data.

**Methodology note:** two architecture tests initially used output-magnitude thresholds (`torch.allclose` on perturbed vs. unperturbed inputs) and failed intermittently — not from a bug, but because pre-norm attention is nearly uniform at random initialization (a well-known property of untrained Pre-LN transformers), making genuine-but-tiny effects indistinguishable from "broken" under a magnitude threshold. Fixed by testing gradient connectivity instead (`d(output)/d(input) != 0`), which is robust regardless of untrained-model attention saturation.

### Phase 6 — Prove learnability on tiny data

**Build**
- [x] Single-device (CPU — see the environment note below, and "Open items") PyTorch/Accelerate debug training loop: `run_debug_training` — `src/pragma/training/debug_loop.py`. Deliberately minimal (plain AdamW, no dynamic batching/checkpointing); explicitly *not* `PretrainingEngine` — Phase 8 supersedes this loop the same way Phase 8's dynamic sampler supersedes Phase 3's fixed-batch `DataLoader`.
- [x] Per-mask-source loss breakdown (`epoch_losses_by_origin`) reusing `PragmaBatch.event_mask_origin` (Phase 4) — no model changes needed.
- [x] `context_dependency_grad`: gradient-connectivity check that predictions structurally depend on non-masked context tokens.
- [x] `shuffle_context_tokens`: magnitude-based context-corruption diagnostic (kept as an exploratory tool, not the pass/fail check — see methodology note below).
- [x] Tiny synthetic pretraining task: a 12-entity slice of the Phase 1 synthetic corpus, generated fresh inside the debug loop/notebook (no dependency on `data/raw/`).
- [x] Bug fix: `MaskingPlanner` could select an already-`[UNK]` (out-of-vocabulary) value for masking, producing an out-of-range label since `[UNK]`'s ID sits below `value_vocab_start`. Fixed by excluding `value_id == UNK` from eligibility; documented in ADR 0005.
- [x] Notebook: `007_debug_training.ipynb` — `CFG.DEBUG` toggle (fraction of corpus, model size, epoch count), loss curves overall + per masking source, both context-degradation checks.

**Exit gate:** met — `tests/integration/test_overfit.py` (4 tests, slower than `tests/unit/` since it runs real training steps) covers: loss drops to <70% of its initial value on a 12-record corpus in 40 epochs; loss decreases for all three masking sources individually; predictions have non-zero gradient w.r.t. every context token's embedding; training is deterministic for a fixed seed. `007_debug_training.ipynb` confirms all of this on real (freshly generated) data.

**Methodology note (same pattern as Phase 5's fix):** the literal "shuffle/destroy context and check loss goes up" comparison (`shuffle_context_tokens`) often barely moves the aggregate loss on this tiny, low-cardinality synthetic corpus — verified directly (per-position loss deltas, per-field-type breakdown, destroying profile identity too), not assumed. This is because a per-key-marginal shortcut ("guess this key's typical value") already explains most of the achievable loss reduction at this scale, not because the mechanism is broken: `context_dependency_grad` confirms real (up to norm ~1–2.4) gradients flow from the loss back through context-token embeddings. The gradient check is the pass/fail criterion; the magnitude comparison is kept in the notebook as an honestly-caveated exploratory diagnostic.

**Environment note:** dev machine's GPU (GTX 1650 Ti, 4GB) has a driver too old for current PyTorch CUDA wheels (max CUDA 11.0 reported; modern wheels need 12.x) — confirmed via `nvidia-smi` and `torch.cuda.is_available()`. User chose to proceed on CPU rather than chase an old CUDA-11.8 wheel against a borderline-incompatible driver. All Phase 6 work (and by extension anything before a driver upgrade) runs CPU-only; `DebugTrainingConfig.batch_size` defaults small regardless of device.

### Phase 7 — Implement packed varlen execution

**Build**
- [x] `VarLenAttentionBackend`: PyTorch nested-tensor (`torch.jagged` layout) `scaled_dot_product_attention` — `src/pragma/attention/varlen.py`. Zero new third-party dependencies; works identically on CPU and CUDA (ADR 0010 — chosen over FlashAttention specifically because this dev machine has no usable CUDA PyTorch build to test one against).
- [x] Event-to-history and token-to-event gather/broadcast operations — reused Phase 5's `prepend_vector`/`group_starts`/`unprepend_vector`/`group_index_per_token` (`src/pragma/modeling/packing.py`) as-is; nothing new was needed here.
- [x] `PragmaConfig.attention_backend` ("padded" | "varlen") + `PragmaModel`'s `_resolve_backend` — lets Phase 8's training engine pick the backend via config instead of always constructing `PaddedAttentionBackend` — `src/pragma/modeling/config.py`, `model.py`.
- [x] `pragma.attention.compare_attention_cost`: deterministic, hardware-independent element-count comparison (padded vs. packed QKV and attention-score buffer sizes) — `src/pragma/attention/analysis.py`.
- [x] ADR 0010: kernel choice (nested tensors, not FlashAttention) and its consequences.
- [x] Notebook: `008_varlen_attention.ipynb` — forward/backward parity, isolation, memory-ratio exit-gate check, and an honest wall-clock benchmark.

**Exit gate:** met — `tests/parity/test_attention_backend_parity.py` (8 tests: backend-level and full-model forward/backward parity across uniform/skewed/zero-length/single-group length distributions, plus isolation) and `tests/parity/test_attention_cost_comparison.py` (2 tests: representative-skew and uniform-length element-count ratios) all pass. Forward parity matches to float64 machine precision at the backend level and ~1e-7 through the full model; backward parity matches within 5e-3 (fp32 accumulation across 8 layers). `008_varlen_attention.ipynb` confirms all of this on real data.

**Methodology finding (recorded, not hidden):** on this CPU-only dev machine, `VarLenAttentionBackend` is measured *slower* in wall-clock time than `PaddedAttentionBackend` on the same skewed-length benchmark that shows a 5.5x QKV / 13.9x attention-score element-count advantage for packed execution — PyTorch's nested-tensor (`torch.jagged`) SDPA kernel is not yet as optimized on CPU as the dense batched math path (ADR 0010). The Phase 7 exit gate's "meaningful memory or throughput improvement" is therefore satisfied via the deterministic element-count comparison (`compare_attention_cost`), not a wall-clock benchmark; a GPU-side re-verification is needed once the driver/hardware situation (see "Open items") is resolved, since that's where FlashAttention-class kernels are expected to actually translate the element-count saving into a wall-clock one.

### Phase 8 — Implement durable distributed pretraining

**Build**
- [x] `TokenBudgetBatchSampler` (dynamic, token/event/record-budget based; length-bucketed, reproducibly shuffled per `(seed, epoch)`; distributed rank-splitting done by the sampler itself, not `Accelerator.prepare()` — ADR 0011) — `src/pragma/data/token_budget_sampler.py`. Replaces Phase 3's fixed-batch `DataLoader` for real runs.
- [x] Accelerate-based training engine (`PretrainingEngine`) — `src/pragma/training/engine.py`. Never branches on hardware except mixed-precision resolution (`resolve_mixed_precision`); single-device vs. multi-GPU DDP is decided entirely by how the process is launched.
- [x] `OptimizerFactory` (plain AdamW, or hybrid Muon+AdamW with 2D-weight-matrix routing via a minimal direct `Muon` implementation) — `src/pragma/training/optimizer.py`, `muon.py`.
- [x] `SchedulerFactory` (warmup + cosine/constant, built from optimizer-update count) — `src/pragma/training/scheduler.py`. A small custom `WarmupDecayScheduler`, not `torch.optim.lr_scheduler.LambdaLR` (see methodology note).
- [x] `CheckpointManager` (atomic save/resume; model+RNG via `Accelerator.save_state`/`load_state`, optimizer+scheduler saved explicitly; sidecar manifest with processor hash, configs, counters, code revision, dependency-lock fingerprint; fails loudly on a missing/mismatched processor bundle per ADR 0007) — `src/pragma/artifacts/checkpoint.py`.
- [x] MLflow logging, local SQLite-backed (see methodology note), main-process-only, persisted at `data/mlflow/mlflow.db` (gitignored, survives notebook re-runs — the checkpoint dir does not).
- [x] `PretrainingEngine.evaluate()` + `train(val_dataset=...)`: a real no-grad validation pass (section 13.1's "train/evaluate loop"), logged as `epoch_val_loss` per epoch. Added after Phase 8 was otherwise complete, when asked for train/valid loss plots and there was no validation loss to plot yet.
- [x] `PretrainingEngine.mlflow_run_id` property, so a notebook/script can pull back exactly what was logged via `MlflowClient().get_metric_history(run_id, key)` instead of tracking a parallel copy of every metric itself.
- [x] ADR 0011: hardware auto-detection philosophy, sampler's own distributed splitting, Muon/AdamW routing, checkpoint strategy, MLflow backend.
- [x] Notebook: `009_distributed_pretraining.ipynb` — hardware/precision probe, sampler diagnostics, a full train→checkpoint→resume→continue cycle with train/val loss tracked throughout, the processor-bundle-mismatch failure case, **three plots built entirely from metrics read back out of MLflow** (per-step loss + LR schedule; per-epoch train-vs-validation loss; the full loss curve stitched across the resume boundary using the fact that `global_step` keeps counting up across a resume), and exact instructions for opening the full MLflow UI (`mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db`, then http://127.0.0.1:5000) or querying it programmatically (`mlflow.search_runs()`).

**Exit gate:** met on the single-device case this machine can run — `tests/unit/test_token_budget_sampler.py` (7 tests: budget/record caps, deterministic reproducible shuffling, distributed rank-splitting disjointness), `tests/integration/test_checkpoint_resume.py` (2 tests: **exact** loss-trajectory match between an uninterrupted 4-epoch run and a stop-at-epoch-2/resume/continue run: single-threaded to isolate the mechanism from unrelated CPU floating-point noise — see methodology note; and loud failure on a corrupted processor bundle), and `tests/integration/test_engine_evaluate.py` (5 tests: finite val loss, `val_loss` populated every epoch when requested and left `None` when not, `evaluate()` restores train mode, `mlflow_run_id` lifecycle) all pass. `009_distributed_pretraining.ipynb` confirms all of this on real data plus prints/plots throughput, token-count, and train/val loss metrics per epoch.

**Multi-GPU comparability not verified — structural, not empirical.** Per ADR 0011, "one- and multi-GPU runs are comparable" could not be tested on this dev machine (no usable CUDA build). The engine is built strictly to `Accelerate`'s standard contract (never constructs `DistributedDataParallel` itself, never touches `torch.cuda` device indices, lets the sampler's own `num_replicas`/`rank` come from `accelerator.num_processes`/`process_index`) specifically so that claim is true by construction rather than proven by a run — it needs re-verification the first time real multi-GPU hardware is available (see "Open items").

**Methodology/bug findings from this phase (all fixed, all documented in ADR 0011 or inline):**
- `torch.optim.lr_scheduler.LambdaLR` requires `isinstance(optimizer, torch.optim.Optimizer)`, which `HybridOptimizer` deliberately isn't — confirmed directly, replaced with a small custom `WarmupDecayScheduler` that only needs `optimizer.param_groups`.
- `Accelerator.save_state` silently skips a custom (non-`torch.optim.Optimizer`) optimizer with no error — confirmed directly (no `optimizer.bin` written for `HybridOptimizer`) — would have silently lost Muon/AdamW momentum on every resume. Fixed by having `CheckpointManager` save/load optimizer and scheduler state explicitly, never relying on `Accelerate`'s automatic tracking for them.
- A restored scheduler's `load_state_dict` must immediately re-apply the restored LR factor to the optimizer's param groups — found via a stale `0.0` LR right after a resume in manual testing — fixed in `WarmupDecayScheduler.load_state_dict`.
- MLflow 3.x deprecated the plain `file:./mlruns` store (raises unless `MLFLOW_ALLOW_FILE_STORE=true`) — switched the default `mlflow_tracking_uri` to a local SQLite file, keeping "no server dependency" without the escape hatch.
- MLflow's fluent API keeps a process-wide "active run" that survives across engines — a second `PretrainingEngine` in the same process inherited a stale run ID pointing at a different tracking store and every `log_metrics` call raised "Run not found." Fixed by having the engine explicitly `mlflow.start_run()`/`end_run()` (`PretrainingEngine.close()`).
- `Accelerator`/`AcceleratorState` is a **process-wide singleton**: the first `Accelerator(...)` call fixes `mixed_precision` for the rest of the process, and a later call requesting a different value raises. Not a bug — just means one process builds one `PretrainingEngine` with one precision setting in practice; documented in ADR 0011 and the notebook rather than "fixed," since there's nothing to fix.
- A resumed run's `TrainingConfig.n_epochs` must match the original run's — the scheduler's decay horizon (`total_optimizer_updates`) is derived from it, so accidentally varying it between the "original" and "resumed" configs changes the LR schedule and makes losses diverge for a real reason. Added `PretrainingEngine.train(end_epoch=...)` so a caller can stop early without changing `n_epochs`, and documented the trap directly in `train()`'s docstring.

### Phase 9 — Run a PRAGMA-S pilot corpus

**Build**
- [x] `scripts/pretrain.py`: thin CLI wrapping `PretrainingEngine` (deferred from Phase 8's open items until a real pilot run needed one) — loads a fitted processor + tokenized shards, builds `TrainingConfig`/`MaskingConfig`/`TokenBudgetConfig` from JSON + CLI overrides, trains in chunks of `--probe-every-n-epochs`, saves a final checkpoint. Always points MLflow at the persistent `data/mlflow/mlflow.db`, never `TrainingConfig`'s bare `sqlite:///mlflow.db` default (confirmed directly: that default writes to whatever the CWD happens to be — caught while smoke-testing this script, fixed before it ever ran against real data).
- [x] `_downstream_is_high_value` label added to `pragma.data.synthetic`'s profile table — a downstream-task target (not a model input), deliberately derived from an entity's own event history (total transaction volume above the corpus median) so a probe on frozen embeddings has real signal to detect, unlike `is_active`/`plan` (assigned independent of events). Underscore-prefixed like `_edge_case_profile` so `SchemaRegistry` validation ignores it and `PointInTimeRecordBuilder` never surfaces it as an input (it only reads `registry.profile_fields`) — `src/pragma/data/synthetic.py`.
- [x] `pragma.evaluation.extract_record_embeddings` — runs a frozen `PragmaModel` backbone with a plain (unmasked) `PragmaCollator()` over a dataset and returns `[USR]` (`record_embeddings`) per entity, per section 14.2 steps 1–2 — `src/pragma/evaluation/embeddings.py`.
- [x] `pragma.evaluation.run_linear_probe` — standard-scaled logistic-regression probe with a fixed train/val split (reusing the pretraining entity split, so the probe never fits on embeddings its own validation AUC is computed from), per section 14.2 steps 3–4 — `src/pragma/evaluation/probe.py`. Deliberately a single function, not yet Phase 10's fuller `LinearProbeRunner`/`BaselineRunner` — just enough to check this phase's own exit gate.
- [x] Fixed train/validation pilot corpus, materially larger than every prior phase's smoke-test data (12–40 entities) — 300 entities / 9,418 events at pilot scale (`010_pilot_pretraining.ipynb`'s `CFG.DEBUG=False`), kept in its own `data/pilot/` directory rather than overwriting `data/raw/`.
- [x] AdamW pilot, then hybrid Muon+AdamW comparison — same model/data/seed/epoch count, only `optimizer` differs; both produced stable, monotonically-decreasing validation loss (see exit-gate evidence below).
- [x] Periodic frozen-embedding extraction — `run_linear_probe`/`extract_record_embeddings` run after training in the notebook, and automatically every `--probe-every-n-epochs` in `scripts/pretrain.py` (`probe_auc`/`probe_n_train`/`probe_n_val` logged to MLflow alongside the loss metrics).
- [x] Notebook: `010_pilot_pretraining.ipynb` — generates the pilot corpus, fits/tokenizes via the existing scripts pointed at `data/pilot/*`, runs the AdamW and Muon+AdamW pilots with per-epoch validation loss, plots train/val loss per optimizer and probe AUC per optimizer (all read back from MLflow), exercises `scripts/pretrain.py` directly via `subprocess` (including `--probe-every-n-epochs`), and documents exact `DEBUG=False` pilot-run numbers in a markdown cell since the committed notebook executes at `DEBUG=True` smoke-test scale (same convention as `007`/`009`).

**Exit gate:** met, on the `DEBUG=False` pilot run captured in `010_pilot_pretraining.ipynb` (300 entities, 9,418 events, hidden_size=64, 3 event + 2 history layers, 5 epochs, CPU-only per `CLAUDE.md`'s environment note):
- *Stable validation curves* — both AdamW and Muon+AdamW show smooth, monotonically-decreasing validation loss across all 5 epochs (AdamW: 4.37 → 3.71; Muon+AdamW: 4.56 → 2.55), no divergence, NaNs, or oscillation.
- *No material data-quality/OOV/truncation/instability blocker* — `validate_corpus` passed (0 schema issues, 0 leakage issues) on the generated pilot corpus; both optimizer runs completed with finite losses throughout.
- *At least one checkpoint shows useful downstream probe signal* — both did: frozen-embedding probe AUC of 0.972 (AdamW) and 0.955 (Muon+AdamW) against `_downstream_is_high_value` on validation entities the probe never trained on, far above the 0.5 chance baseline.

**AdamW-vs-hybrid finding (not yet a tuning conclusion):** at the published reference Muon hyperparameters (`lr=0.02, momentum=0.95` — ADR 0011, not tuned for PRAGMA-S), Muon+AdamW reached a materially lower validation loss than plain AdamW over the same 5 epochs, while AdamW's frozen embeddings scored marginally higher on the probe — both signals are strong and comparable. Five epochs on 300 entities is far too little data to call one optimizer definitively better for downstream transfer from this alone; revisit once Phase 1's real client extract or a materially larger synthetic pilot is available.

**Methodology note (same "state it honestly" pattern as Phases 6–8):** the notebook is committed executing at `CFG.DEBUG=True` (a ~40-entity, 1-epoch smoke test, consistent with every prior phase's notebook convention) so it runs in well under a minute and its probe AUC is degenerate (1.0 on 4 validation entities — too few to mean anything). The pilot-scale numbers this exit gate is actually verified against were captured from one real `DEBUG=False` run on this dev machine and are recorded as a markdown table in the notebook rather than re-executed on every run, the same way Phase 7's wall-clock benchmark numbers are reported without being asserted on in a test.

### Phase 10 — Build embedding probes and baselines

**Build**
- [x] `EmbeddingExtractor` (`[USR]`, last `[EVT]`, and their concatenation, in one pass — `EmbeddingBundle`) — generalizes Phase 9's `[USR]`-only `extract_record_embeddings` (kept as a thin wrapper for its existing callers) — `src/pragma/evaluation/embeddings.py`. A zero-event evaluation record's `last_event` is an all-zero vector by design (ADR 0012) — "no history yet" is a real state, not a missing value.
- [x] `LinearProbeRunner` — runs `run_linear_probe` across all three `EmbeddingBundle` variants against the same labels/split — `src/pragma/evaluation/probe.py`.
- [x] `build_aggregated_features` — conventional, hand-built point-in-time feature table (event counts/totals, tenure, milestone flags, static categorical attributes) read from the same `EvaluationRecord`s the tokenizer sees — `src/pragma/evaluation/features.py`.
- [x] `run_baselines` (the plan's `BaselineRunner`) — logistic regression and `HistGradientBoostingClassifier` (ADR 0012's GBDT choice, not LightGBM — avoids a second native-code dependency) on `build_aggregated_features`'s table, same labels/split as the probes, returning the same `ProbeResult` shape so both fit in one report — `src/pragma/evaluation/baselines.py`.
- [x] `build_comparison_report` (the plan's `ModelEvaluator`, scoped to what Phase 10 actually needs — see ADR 0012 decision 5) — combines probe + baseline results into one ranked table — `src/pragma/evaluation/report.py`.
- [x] `scripts/extract_embeddings.py` and `scripts/run_probe.py` (both named explicitly in section 12's repo structure) — thin CLIs wrapping the above; `extract_embeddings.py` rebuilds only the model from a checkpoint's own saved `PragmaConfig` (confirmed directly that `accelerator.load_state` works without the optimizer/scheduler ever being `accelerator.prepare`d), writes embeddings to Parquet; `run_probe.py` reads those back plus the raw profile/event tables and writes a CSV comparison report.
- [x] ADR 0012: GBDT choice, aggregated-feature-set scope, `EmbeddingExtractor`'s zero-event convention, shared `ProbeResult` shape across probes and baselines, `build_comparison_report`'s deliberately-partial `ModelEvaluator` scope, AUC as the one comparison metric.
- [x] Notebook: `011_embedding_probes_and_baselines.ipynb` — trains one checkpoint (same recipe as `010`'s AdamW pilot), runs the full probe/baseline/report pipeline via direct library calls (for plotting control) and again via the two new CLI scripts (`subprocess`), plots probes vs. baselines as one ranked bar chart, and documents actual `DEBUG=False` pilot-run numbers in a markdown cell (same convention as `007`/`009`/`010`).

**Exit gate:** met — the comparison report gives a clear, specific answer, not just a probe number in isolation. On the `DEBUG=False` pilot run (300 entities, same checkpoint as `010`'s AdamW pilot): `baseline_gbdt` (AUC 1.000) and `baseline_logreg` (0.994) edge out `probe_usr`/`probe_concat` (0.972) and `probe_last_event` (0.955) against `_downstream_is_high_value`. This has a specific, understood cause, not an unexplained gap: the label is *defined* as `total_amount` above the corpus median, and `build_aggregated_features` hands the baseline `total_amount` directly — a GBDT given the exact statistic a label thresholds is expected to win. The positive result underneath that: PRAGMA's frozen embeddings reach 0.95–0.97 AUC on this label **without ever seeing `total_amount` as an input**, only the raw tokenized event sequence — real, recoverable transfer signal, just not (yet, on this label) evidence the backbone beats a GBDT that was handed the defining feature. Both the "where it doesn't help" and the "how much of its signal is genuine anyway" halves of the exit gate are answered, not just one.

**Methodology note:** the mechanics tests (`tests/integration/test_probes_and_baselines.py`) verify shapes/alignment/report structure on an untrained model; they deliberately do not assert an AUC ordering, since that is a property of a trained checkpoint and a specific label, not of the harness itself. The actual "does pretraining help" claim is evidenced by the pilot run above, following the same "state findings honestly" pattern as Phases 6–9 rather than asserting a numeric threshold that would either be circular (tuned to always pass) or brittle (fail non-deterministically on a different corpus).

### Phase 11 — Implement one LoRA task

**Build**
- [x] `PragmaForTask` (`PragmaModel` backbone + `Linear(hidden_size, 1)` binary head on `record_embeddings`/`[USR]` — ADR 0013) — `src/pragma/downstream/task_model.py`.
- [x] `build_lora_model` (the plan's `LoRAAdapterFactory`, a plain function per this project's established "*Factory" -> function" pattern) — wraps `PragmaForTask` with a PEFT `LoraConfig` targeting `q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2` (rank 8/alpha 8 default, section 15.2), `modules_to_save=["classifier"]` so the task head trains and saves alongside the adapter — `src/pragma/downstream/lora.py`. Confirmed directly (not assumed) that PEFT's name-suffix matching works against this custom, non-Hugging-Face architecture.
- [x] `load_lora_model` — reloads a saved adapter **independently**: rebuilds the frozen backbone from scratch from the base checkpoint (re-verifying its processor hash, ADR 0007) and reattaches the adapter via `PeftModel.from_pretrained` — `src/pragma/downstream/lora.py`.
- [x] `DownstreamTrainer` (the plan's row of the same name) — Accelerate-based fine-tuning loop reaffirming ADR 0011's hardware posture exactly (auto GPU/DDP/mixed-precision via `Accelerator`, never hardcoded); plain AdamW (not Muon — ADR 0013) over only the trainable LoRA+head parameters; a plain fixed-batch `DataLoader` through `accelerator.prepare()` instead of `TokenBudgetBatchSampler` (small downstream datasets don't need dynamic token budgeting); `evaluate()` reports loss + AUC — `src/pragma/downstream/trainer.py`.
- [x] `load_frozen_backbone` — loads only a checkpoint's `.pragma` submodule weights into a fresh `PragmaForTask` (classifier stays randomly initialized), verifying the processor hash first — `src/pragma/downstream/trainer.py`.
- [x] `DownstreamConfig` (LoRA rank/alpha/dropout/target-modules, fine-tuning hyperparameters — section 18's open list, resolved in ADR 0013) — `src/pragma/config/downstream_config.py`, `configs/downstream/default.json`.
- [x] `hash_processor_bundle` made a public, shared helper (`pragma.artifacts`) instead of pretraining-checkpoint-private, so adapter manifests can reuse the exact same processor-identity check `CheckpointManager` uses.
- [x] `scripts/finetune_lora.py` (named explicitly in section 12's repo structure) — thin CLI wrapping `DownstreamTrainer`.
- [x] `build_comparison_report` (Phase 10) extended with a third `"lora"` kind, alongside `"probe"`/`"baseline"` — found and fixed during notebook execution: a `ProbeResult` named `"lora_finetuned"` was being silently misclassified as a baseline by the old two-way `startswith("probe_")` check.
- [x] New downstream label `_downstream_is_escalating_spender` (`pragma.data.synthetic`, ADR 0013) — order-dependent (second-half vs. first-half spend), specifically so it can't be trivially recovered by an aggregate feature the way Phase 9/10's `_downstream_is_high_value` could (Phase 10's finding).
- [x] ADR 0013: LoRA vs. QLoRA (plain LoRA — no quantization needed at PRAGMA-S's ~10M-parameter scale; documents what each technique is and why, with the paper's own section 15.2 guidance as the anchor), target modules, task-head architecture, hardware/Accelerate/DDP reaffirmation, optimizer choice, hyperparameter defaults, the new downstream label's motivation.
- [x] Notebook: `012_lora_finetuning.ipynb` — extensive markdown explaining LoRA's mechanism and the LoRA-vs-QLoRA reasoning (matching the user's explicit request for this phase to "take some time explain and document well"), pretrains a checkpoint, checks section 15.1's entry condition (frozen probe vs. baseline) on the new label, LoRA fine-tunes via `DownstreamTrainer`, verifies base-checkpoint immutability and independent adapter reload with byte/tensor-level checks, builds the full probe-vs-baseline-vs-LoRA comparison report, exercises `scripts/finetune_lora.py` directly, and documents actual `DEBUG=False` pilot-run numbers (same convention as `007`/`009`/`010`/`011`).

**Exit gate:** mechanically met, scientifically inconclusive on this pilot — and said so directly rather than glossed over. `tests/integration/test_downstream_trainer.py` (4 tests) directly verifies the two structural exit-gate claims: the base checkpoint's `model.safetensors` is byte-identical before and after a full fine-tuning run (immutability), and a freshly reloaded adapter (`load_lora_model`, no in-memory state from the training run) reproduces the trained model's exact logits (independent reload). `tests/unit/test_lora.py` (3 tests) confirms only LoRA+classifier parameters ever receive gradients or move during a forward/backward pass. LoRA-vs-frozen-probe-vs-baseline is compared on the identical split via `build_comparison_report` (`tests/integration/test_probes_and_baselines.py`, 5 tests including the new `"lora"` kind). On the pilot run itself (300 entities, same pretraining recipe as `010`/`011`, 10 LoRA epochs at rank 8): every method — probes, both baselines, and LoRA — scored close to the 0.5 chance line on `_downstream_is_escalating_spender` (best: baseline_gbdt 0.582; LoRA ~0.555; probe_usr 0.505). Section 15.1's entry condition ("frozen embeddings beat or materially complement a baseline") is **not** cleanly satisfied here — documented honestly rather than cherry-picking a favorable split or quietly lowering the bar.

**Methodology/bug findings from this phase (documented, not hidden):**
- The order-dependent label needed for this phase to be scientifically interesting (not trivially solved by an aggregate, per Phase 10's finding) also turned out to need more data/compute than this synthetic pilot supplies to actually *learn* — 238 training entities and 5 pretraining epochs on a hidden_size=64 model aren't enough for any method, including LoRA, to reliably capture a subtle temporal-trend pattern. This is a genuine scale limitation, not a bug: LoRA's own training loss does fall (~0.72 → ~0.62 over 10 epochs), it just hasn't found a validation-generalizing signal yet. Revisit once Phase 1's real client extract (still outstanding) or a materially larger synthetic pilot is available — real behavioral trends should carry far more supporting signal than 300 synthetic entities ever will.
- `build_comparison_report`'s two-way `kind` classification (`startswith("probe_")` else `"baseline"`) silently mislabeled `ProbeResult(name="lora_finetuned", ...)` as a baseline — caught while executing `012_lora_finetuning.ipynb`'s final comparison table, fixed by classifying on `startswith("baseline_")` too and defaulting everything else to `"lora"` (`tests/integration/test_probes_and_baselines.py` now asserts this directly).

### Phase 12 — Optimize and decide on scale-up

**Possible work**
- [ ] Kernel/backend tuning, better length bucketing/prefetching, `torch.compile`, gradient checkpointing, FSDP/DeepSpeed if justified.
- [ ] Larger corpus / PRAGMA-M feasibility study / additional downstream tasks.

**Scale-up gate:** measured downstream value justifies additional compute; bottlenecks identified via profiling; data volume/diversity sufficient for a larger model.

---

## Open items not tied to a specific phase

- Phase 1's real de-identified client extract is still outstanding — everything so far has run on the synthetic corpus only.
- Section 18's "what the paper does not make clear" list is progressively resolved via ADRs as each phase forces a decision (0001–0013 so far); Muon/AdamW routing and hyperparameters are now decided (ADR 0011, defaults only — not yet ablated); evaluation-harness decisions are now decided too (ADR 0012); LoRA target modules/task-head/hyperparameters are now decided too (ADR 0013, defaults only — not yet swept); the rank 4/8/16 sweep section 15.2 calls for is still open (only rank 8 has been run).
- `LMDBProfileStore` (optional, section 11.2) has not been needed yet — `ParquetShardStore` has been sufficient at this data scale.
- Dev machine's GPU (GTX 1650 Ti, 4GB) can't run current PyTorch CUDA wheels (driver reports max CUDA 11.0; modern wheels need 12.x) — all training so far (Phases 6–11) is CPU-only. Phase 7's memory-vs-throughput finding and Phase 8's multi-GPU comparability are both **structurally** satisfied (deterministic element-count comparison; strict adherence to `Accelerate`'s standard contract) but not empirically verified on real GPU hardware — revisit both once the driver is updated or training moves to a cloud GPU. `DownstreamTrainer` (Phase 11) inherits this same structural-not-empirical status for DDP.
- Muon/AdamW hyperparameters (`lr=0.02, momentum=0.95` for Muon) are still the reference implementation's published defaults, not tuned for PRAGMA-S — Phase 9's pilot ran the AdamW-vs-hybrid comparison section 13.2 calls for (Muon+AdamW reached materially lower validation loss at these defaults), but 5 epochs on 300 synthetic entities is not enough evidence to tune from; revisit with more data.
- `_downstream_is_high_value` (Phase 9's synthetic label) turned out to be close to a solved problem for a simple aggregate feature (Phase 10's finding: it's a threshold of `total_amount`, which `build_aggregated_features` hands the baseline directly). `_downstream_is_escalating_spender` (Phase 11's label, order-dependent) fixed that specific problem but turned out to need more data/compute than this synthetic pilot supplies for *any* method to learn well (best AUC ~0.58, near chance) — every synthetic label tried so far is either too easy (aggregate-trivial) or too hard at this data scale (order-dependent). Phase 1's outstanding real client extract remains the most likely way to get a downstream label that is both non-trivial *and* learnable at a reasonable data scale.
- Phase 10's `run_baselines` runs both models through the same `ColumnTransformer` (impute + scale/one-hot) for simplicity (ADR 0012) — a GBDT does not strictly need scaled numeric features; revisit if baseline tuning becomes a priority.
- LoRA rank sweep (4, 8, 16 — section 15.2) not yet run; only rank 8/alpha 8 (the stated starting point) has been evaluated so far (Phase 11).
- `DownstreamConfig`'s hyperparameters (`learning_rate=1e-3`, `batch_size=8`, `n_epochs=10`, etc. — ADR 0013) are reasonable LoRA fine-tuning defaults from the wider literature, not tuned for PRAGMA-S or for any specific downstream task.
- No out-of-time test partition (customers acquired after the training period) exists yet — `SplitConfig`'s train/val/test split (ADR 0008) is a random per-entity hash, not a time-based cohort split. The current split already gives strict unseen-customer evaluation (entity IDs are disjoint across splits, ADR 0008 decision 6, tested directly per the Phase 2 note above), but it does not yet prove generalization to customers who didn't exist during training. Candidate for a new ADR + `SplitConfig` extension when Phase 12/13 evaluation-harness work resumes.
- ADR 0014 (zero-event customer exclusion): `PointInTimeRecordBuilder` still builds a record for zero-event entities (needed for point-in-time correctness testing), but every pretraining/inference consumer now drops them via `drop_zero_event_records()` before use.
