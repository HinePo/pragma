# PRAGMA — Technical Map

A from-scratch orientation to the codebase for new collaborators: what every
piece is, what it does, and how it fits together. This is a snapshot of the
*implementation*, organized in the same phase order the project was built in
(see [plans/PRAGMA-Implementation-Plan.md](plans/PRAGMA-Implementation-Plan.md)
for the spec and [plans/progress.md](plans/progress.md) for the current
phase/status). Design *decisions* (the "why", when the plan left something
ambiguous) live in [docs/adr/](docs/adr/README.md) — this document points at
the relevant ADR instead of re-deriving the reasoning.

> Read [CLAUDE.md](CLAUDE.md) first for the three-layer convention
> (`src/pragma/` = logic, `scripts/` = thin CLIs, `notebooks/` = narrative) —
> everything below assumes it.

---

## 1. Repository layout

```
pragma/
├── src/pragma/            # ALL business logic (the only tested, importable layer)
│   ├── schema/            # canonical field registry (raw data contract)
│   ├── data/              # synthetic generation, point-in-time records, packed batches
│   ├── processing/        # fitted structured processor (tokenization)
│   ├── masking/           # MLM masking planner
│   ├── modeling/          # PRAGMA-S model: embeddings, encoders, RoPE, MLM head
│   ├── attention/         # padded vs. varlen(packed) attention backends
│   ├── training/          # Accelerate-based pretraining engine, optimizers, scheduler
│   ├── artifacts/         # checkpoint save/resume
│   ├── evaluation/        # frozen-embedding probes, conventional baselines, reports
│   ├── downstream/        # LoRA task head + fine-tuning trainer
│   └── config/            # every *Config dataclass (JSON-serializable)
├── scripts/               # thin CLI entry points (SageMaker-job shaped)
├── notebooks/             # one notebook per phase, numbered 000-012
├── tests/
│   ├── unit/              # fast, one module per library module
│   ├── integration/       # cross-component, checkpoint resume, end-to-end pilot pieces
│   ├── parity/            # padded-vs-varlen attention backend parity/cost
│   └── distributed/       # reserved for later multi-GPU tests (empty so far)
├── configs/               # JSON defaults for every *Config (model/processor/pretraining/downstream)
├── docs/adr/              # 13 ADRs — the "why" behind every ambiguous design choice
├── plans/
│   ├── PRAGMA-Implementation-Plan.md   # source-of-truth spec, 13 phases
│   └── progress.md                     # phase-by-phase build/exit-gate/evidence log
└── data/                  # generated, gitignored (raw/, processor/, shards/, checkpoints, mlflow)
```

### Data flow, end to end

```
raw events + profile (pragma.data.synthetic)
        │
        ▼
EvaluationRecord  (pragma.data.records.PointInTimeRecordBuilder)   ← point-in-time correctness
        │
        ▼
TokenizedRecord   (pragma.processing.PragmaProcessor.transform)     ← fitted vocab/buckets/BPE
        │
        ▼
PragmaBatch       (pragma.data.batch.PragmaCollator)                ← packed, masked, offsets
        │
        ▼
PragmaForMaskedModeling.forward(batch)  (pragma.modeling)            ← the model
        │
        ├─► pretraining loss/MLflow   (pragma.training.PretrainingEngine)
        ├─► frozen embeddings/probes  (pragma.evaluation)
        └─► LoRA task fine-tuning     (pragma.downstream.DownstreamTrainer)
```

---

## 2. Schema layer — `src/pragma/schema/`

**`registry.py`**
- `FieldType` (`NUMERICAL`/`CATEGORICAL`/`TEXT`/`TIMESTAMP`/`IDENTIFIER`/`IGNORED`), `NullPolicy` (`ALLOWED`/`FORBIDDEN`/`ZERO_BUCKET`).
- `FieldPolicy` — canonical definition of one semantic key (type, aliases, null policy, `maskable`, `regulated`, `pii_risk`). Identifiers/timestamps are forced non-maskable in `__post_init__`.
- `EventFamily` — the allowed required/optional fields for one raw event type (e.g. `card_payment`).
- `SchemaRegistry` — the whole canonical vocabulary:
  - `register_field` / `register_event_family` / `register_profile_field` (marks lifelong milestones).
  - `resolve(source_field_name)` — raw column name → canonical key (alias resolution).
  - `validate_event` / `validate_profile` — schema-conformance checks used by `synthetic.validate_corpus`.
  - `SchemaRegistry.default()` — the concrete schema for this project's synthetic domain: 7 event families (`topup`, `card_payment`, `p2p_transfer`, `atm_withdrawal`, `fx_exchange`, `direct_debit`, `app_event`), profile attributes + 3 lifelong milestones (`first_topup_at`, `first_card_payment_at`, `first_p2p_at`).

Every raw source field must resolve through this registry before it can reach a record or the processor — this is the single control point for "what is a valid field" (ADR 0001).

---

## 3. Data layer — `src/pragma/data/`

### `synthetic.py` — corpus generator
- `SyntheticDataConfig` — size/edge-case knobs (`n_entities`, event-family weights baked in as `EVENT_FAMILY_WEIGHTS`, zero/single/long-history/same-timestamp entity counts, rare-value injection rate).
- `generate_synthetic_corpus(config) -> (events_df, profile_df, manifest)` — the corpus builder. Deliberately manufactures edge cases (zero events, one event, 3000-event "whale", colliding timestamps, rare currency/MCC) rather than relying on chance.
- Two **downstream labels** are attached to `profile_df` (evaluation-only scaffolding, not model inputs, underscore-prefixed so the schema registry never sees them):
  - `_downstream_is_high_value` (Phase 9) — `total_amount > median`. Found in Phase 10 to be trivially recoverable by a baseline that includes `total_amount` as a feature.
  - `_downstream_is_escalating_spender` (Phase 11, `_is_escalating_spender`) — spent more in the second (time-ordered) half of history than the first. Deliberately **order-dependent**: a conventional aggregate-feature baseline can't recover it without engineering a first/second-half split itself. See ADR 0013.
- `validate_corpus(events_df, profile_df) -> report` — schema conformance + leakage checks (no event before signup, milestone timestamps consistent with actual first events of that family) + null-rate/duplicate-timestamp stats.

### `records.py` — point-in-time correctness (ADR 0002)
- `EventRecord`, `ProfileState` (attributes + `milestones: dict[key, datetime|None]`, already clipped to point-in-time), `EvaluationRecord` (one leakage-safe observation per entity: `evaluation_time`, `profile_state`, `events_before_evaluation`, `split`).
- `SplitConfig` — `train_frac`/`val_frac`/`seed`; `_entity_split` assigns train/val/test by **hashing the entity ID** (stable across reruns, independent of row order).
- `PointInTimeRecordBuilder.build(events_df, profile_df) -> list[EvaluationRecord]` — for each entity: `evaluation_time` = last event time (or signup if none); milestones after that time are reported as `None`, never leaked; events are sorted and tie-broken by `(created_at, event_id)`.

### `batch.py` — the packed batch contract (ADR 0004)
- `PragmaBatch` — flat token/event buffers + cumulative offsets (`*_cu_seqlens`), **never padded tensors**. Three levels of grouping: profile tokens → record, event tokens → event (`event_cu_seqlens`), events → record (`history_cu_seqlens`, `event_to_record`).
  - `.to(device)` — moves every tensor field in place (used by `PretrainingEngine` after `Accelerate` picks a device).
  - `.validate()` — internal consistency checks (monotonic offsets, in-range mappings) — this is what a real attention backend relies on for isolation guarantees.
- `PragmaCollator(masking_planner=None)` — turns a `list[TokenizedRecord]` into one `PragmaBatch`. Without a planner, masking fields are all-`IGNORE_INDEX`/zero (same shape either way). `set_epoch(epoch)` reshuffles masks deterministically per epoch (mirrors `DistributedSampler.set_epoch`).

### `dataset.py`
- `TokenizedRecordDataset(Dataset[TokenizedRecord])` — thin wrapper over an in-memory record list; `.from_store(shard_dir, store, split=...)` loads via `PragmaRecordStore`. This is the *reference* (fixed-batch) path Phase 8's dynamic sampler is benchmarked against.

### `storage.py` — durable shard storage (section 10.1)
- `PragmaRecordStore` (ABC: `write_shard`/`read_shard`) and its local implementation `ParquetShardStore` (nested-list Arrow schema `_ARROW_SCHEMA`).
- `event_count_bucket(n_events, edges=(1,50,500))` — coarse length-bucket label; shards are written one-per-`(split, bucket)`.
- `write_dataset(...) -> DataManifest` / `read_dataset(...)` — write/read a whole tokenized dataset plus a `manifest.json` (schema version, processor fingerprint, per-shard SHA-256 — `read_dataset` refuses a shard whose checksum doesn't match).

### `token_budget_sampler.py` — dynamic batching (section 9.3)
- `TokenBudgetBatchSampler` — greedily packs records into batches under `TokenBudgetConfig`'s token/event/record budgets, after bucketing by length and shuffling reproducibly per `(seed, epoch)`. Handles its **own** distributed rank-splitting (`num_replicas`/`rank`, batch-list padding so every rank gets an equal count) rather than trusting `Accelerator.prepare()`'s automatic sharding, which assumes fixed-size shards (ADR 0011).
- `BatchStats` / `.stats()` — per-epoch batch-formation diagnostics (mean records/tokens per batch).

---

## 4. Processing layer — `src/pragma/processing/`

Converts `EvaluationRecord` → `TokenizedRecord` via a **fitted structured processor**, not a Hugging Face tokenizer (ADR 0003) — because fields are typed (numeric/categorical/text/timestamp), not raw strings.

- **`special_tokens.py`** — `SpecialTokens`: fixed IDs `PAD=0, UNK=1, MASK=2, USR=3, EVT=4, MILESTONE_PRESENT=5, MILESTONE_ABSENT=6`.
- **`vocabulary.py`** — `KeyVocabulary`: one stable token ID per tokenizable canonical key (numeric/categorical/text fields + lifelong-milestone fields). `.fit(registry, base_id)`, `.id_for(key)`/`.key_for(id)`.
- **`numeric.py`** — `NumericBucketizer`: per-key **percentile** bucket boundaries fit on train values only; bucket *token IDs* are global/shared across keys (`[ZERO]` + `n_buckets` percentile buckets) — ADR 0008. `.fit(key, values)`, `.transform(key, value) -> token_id` (outliers clip to edge buckets, never refit at inference).
- **`categorical.py`** — `CategoricalEncoder`: per-key value vocab, namespaced into its own contiguous ID block (ADR 0008); unseen values at transform time fall back to `[UNK]`, tracked via `CategoricalKeyStats.oov_rate`.
- **`temporal.py`** — stateless: `soft_log_time(delta_seconds, scale=8.0)` = `scale * ln(1 + t/scale)`; `calendar_features(timestamp) -> CalendarFeatures` (sin/cos of hour/weekday/day-of-month); `time_to_latest`, `time_to_evaluation` (both soft-log elapsed time, feeding `ContinuousRoPE`).
- **`text_bpe.py`** — `TextBPEEncoder`: one shared byte-level BPE model (via `tokenizers`) across every `TEXT` field (ADR 0003), byte-level pre-tokenizer/decoder so there's never an OOV fragment (worst case: single bytes).
- **`tokenized_record.py`** — `FieldTokens` (key/value/within-field-pos triples, equal length enforced in `__post_init__`), `TokenizedEvent`, `TokenizedRecord` — the stable serialized shape (`to_dict`/`from_dict`).
- **`processor.py`** — `PragmaProcessor`, the coordinator:
  - Lays out the **full token ID space once** at construction: `[special][keys][numeric buckets][categorical...][bpe]` — key IDs and numeric-bucket range size are schema/config-derived (fixed immediately); categorical/BPE ranges depend on fitted training data (finalized only in `fit()`).
  - `.fit(train_records) -> FitReport` — filters to `split == "train"` internally (never trusts the caller), fits every sub-encoder, computes a `train_fingerprint` (SHA-256 of sorted entity IDs).
  - `.transform(record) -> TokenizedRecord` — tokenizes profile + events; truncates events from the **end of `event.fields`'s iteration order** (whole fields, never mid-BPE-subword) when over `max_event_tokens`; keeps the **most recent** events when over `max_history_events`.
  - `.save(path)` / `.load(path, registry)` — the versioned artifact bundle (`bundle.json` + `bpe_tokenizer.json`) a checkpoint requires to be reproducible (ADR 0003, ADR 0007).

---

## 5. Masking layer — `src/pragma/masking/`

**`planner.py`**
- `MaskSource` (`IntFlag`: `NONE`/`TOKEN`/`EVENT`/`KEY`) — bitflags recording *which* mask source(s) selected a position; three independent sources combined by union (ADR 0005).
- `RecordMaskPlan` — `input_value_ids`, `labels` (`-100` = ignore), `origin` (per-position `MaskSource` flags), aligned with the record's flattened event-token order.
- `MaskingPlanner`:
  - `.from_registry(registry, key_vocab, config)` — precomputes which key IDs are event-maskable once (schema-level fact, not per-call).
  - `.plan_record(record, epoch=0)` — deterministic per-`(seed, entity_id, epoch)` RNG (`_record_rng`, SHA-256-seeded). Applies, in order: (1) per-token masking, (2) whole-event masking, (3) record-wide semantic-key masking (masks *every* occurrence of a chosen key across the record's full history). Each selected position then independently rolls `unk_dropout_frac`: becomes `[UNK]` (input-only corruption, excluded from loss) or `[MASK]` (loss target = original value) — ADR 0005's corrected corruption rule.
  - Only event **value** tokens are ever masked; profile-state values stay visible as context (section 7.5). Positions already `[UNK]` (true OOV) are never selected — there's no real class to predict.

Profile-state masking never happens; only `PragmaCollator`'s event-token path consults a `MaskingPlanner`.

---

## 6. Modeling layer — `src/pragma/modeling/` (PRAGMA-S architecture)

This is the heart of the project. Every class here is a genuinely custom, from-scratch implementation — no pretrained weights are loaded from anywhere.

### Hugging Face integration
- `PragmaConfig` (**subclasses `transformers.PreTrainedConfig`**, `config.py`) — every model hyperparameter (`hidden_size`, `num_heads`, `profile_layers`/`event_layers`/`history_layers`, `intermediate_size`, `dropout`, `rope_base`, `max_within_field_position`, `attention_backend`, special-token IDs, `value_vocab_start`). `.from_processor(processor, **overrides)` is the *only* place a fitted processor's vocabulary layout/special-token IDs are read into the model config — the model never touches the processor at runtime after that.
- `PragmaModel` and `PragmaForMaskedModeling` (**both subclass `transformers.PreTrainedModel`**, `model.py`) — get `save_pretrained`/`from_pretrained`, standard weight init hooks (`_init_weights`), and Hub compatibility "for free". This is also the base class `downstream/task_model.py`'s `PragmaForTask` reuses (and what PEFT's `get_peft_model` wraps in Phase 11 — see §10).

### Embeddings — `embeddings.py`
- `WithinFieldPositionEncoding` — fixed (non-learned) sinusoidal encoding of a token's position *within* its field (only matters for multi-token BPE text).
- `SharedKeyValueEmbedding` — **one shared `nn.Embedding` table** for keys, values, and special tokens (they all live in one contiguous ID space laid out by the processor). `forward(key_ids, value_ids, within_field_pos) = E(key) + E(value) + P(pos)`. `.embed_special(special_id, n, device)` reuses the exact same formula for a `[USR]`/`[EVT]` slot with `k = v = special_id`.

### RoPE — `rope.py`
- `ContinuousRoPE` — standard GPT-NeoX/RoFormer rotate-half rotary encoding, generalized to accept **arbitrary continuous elapsed-time coordinates** instead of integer positions (profile milestone time, event time-to-latest). `apply_rotary(x, cos, sin)`.

### Calendar features — `calendar.py`
- `CalendarEncoder` — tiny 2-layer MLP (`6 → hidden_size → hidden_size`, GELU) mapping `pragma.processing.temporal.calendar_features`'s 6 cyclical features to model width; added onto the Event Encoder's `[EVT]` summary only (not every token).

### Packing utilities — `packing.py`
Generic, reused by all three encoders — insert/remove exactly one summary vector per `cu_seqlens`-bounded group:
- `new_cu_seqlens`, `cu_seqlens_from_group_ids`, `group_index_per_token`
- `prepend_vector(x, cu_seqlens, vec)` — insert one vector (e.g. `[USR]`'s embedding) at the start of every group; returns `(new_x, new_cu_seqlens)`.
- `group_starts` — the post-attention state of the prepended slot, per group.
- `unprepend_vector` — inverse of `prepend_vector`.

### Transformer block — `transformer_block.py`
- `PragmaTransformerBlock` — pre-norm self-attention + GELU MLP. Projections are named `q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2` **deliberately**, so PEFT's LoRA (Phase 11) can target them by name suffix without any encoder code changing (section 15.2, ADR 0011/0013). Delegates the actual attention computation to whichever `AttentionBackend` is passed in — this block has no idea whether it's running padded or packed/varlen.

### The three encoders — `encoders.py`
All three share the packing pattern: embed → prepend one group summary vector (`[USR]`/`[EVT]`) → run `N` transformer blocks with per-group isolation → split back into local-token-states + group-summary via `packing.py`.

1. **`ProfileStateEncoder`** — bidirectional, one `[USR]` + profile key/value tokens per record. `ContinuousRoPE` applied to milestone elapsed-time coordinates (static attributes and `[USR]` itself get `0.0`). Returns the contextual `[USR]` summary per record.
2. **`EventEncoder`** — encodes every event **independently**, fully isolated from every other event (no RoPE — within-event order comes from the within-field positional encoding, not a temporal coordinate). Returns `(local_token_states, event_summaries)`; `event_summaries` gets the `CalendarEncoder` output added.
3. **`HistoryEncoder`** — encodes `[USR]` (profile encoder's output) followed by the ordered event summaries per record, with `ContinuousRoPE` over each event's time-to-latest coordinate. This is where the "complete customer history" gets contextualized. Returns `(contextual_usr, contextual_event)`.

### The MLM head — `mlm_head.py`
- `PragmaMLMHead` — for every masked value-token position, concatenates `[local_token_state, contextual_event_state, contextual_usr_state]` (`3 × hidden_size`), projects to `hidden_size`, and matches against the **value-vocabulary-only slice** of the shared embedding table (`embedding.weight[value_vocab_start:]`) — logits are never computed over the full vocabulary, since a masked value can never equal a special/key ID (ADR 0009). Only masked positions get logits materialized (never a full batch×seq×vocab tensor).

### Top-level models — `model.py`
- `PragmaModel(PreTrainedModel)` — **the reusable backbone**: `ProfileStateEncoder → EventEncoder → HistoryEncoder`. `forward(batch, attention_backend=None) -> (local_token_states, event_embeddings, record_embeddings)`. Resolves the backend string (`"padded"`/`"varlen"`) from config if none is passed — the *only* place that string is interpreted.
- `PragmaForMaskedModeling(PreTrainedModel)` — `PragmaModel` + `PragmaMLMHead` + loss, the randomly-initialized pretraining model. `forward(batch) -> PragmaOutput`.
- `PragmaOutput` (`outputs.py`, a `transformers.utils.ModelOutput` dataclass) — `loss`, `mlm_logits`, `record_embeddings` (contextual `[USR]` per record — the customer-level summary used everywhere downstream), `event_embeddings` (contextual `[EVT]` per event).

---

## 7. Attention layer — `src/pragma/attention/` (ADR 0006, ADR 0010)

- **`backend.py`** — `AttentionBackend` (ABC): `forward(q, k, v, cu_seqlens) -> out`, all shaped `[N, num_heads, head_dim]` (packed across the whole batch). Every encoder depends only on this interface.
- **`padded.py`** — `PaddedAttentionBackend`: the **permanent correctness oracle**. Pads each group to the batch's longest, runs standard `scaled_dot_product_attention` with an explicit boolean key-padding mask, discards padding on the way out.
- **`varlen.py`** — `VarLenAttentionBackend`: the **real pretraining path**. Never materializes padded tensors — wraps packed buffers as PyTorch nested tensors (`torch.jagged` layout) so SDPA computes exactly the tokens that exist. `.values()` unwraps back to the same flat shape the padded backend returns, so callers never know which ran.
- **`analysis.py`** — `compare_attention_cost(lengths, num_heads, head_dim) -> AttentionCostComparison`: deterministic, hardware-independent element-count comparison (QKV memory ratio, attention-score memory ratio) between padded and packed execution — this is what backs the Phase 7 exit gate rather than a wall-clock benchmark (ADR 0010 notes PyTorch's CPU nested-tensor kernel isn't as optimized as the dense path, so wall-clock time can even *regress* on CPU despite computing far less).

Which backend is used is a pure runtime choice (`PragmaConfig.attention_backend` or an explicit override) — no model code changes between them.

---

## 8. Training layer — `src/pragma/training/` (ADR 0011)

**Central hardware posture, reaffirmed everywhere in this project (Phase 8 → Phase 11):** device and DDP selection are **never** hardcoded — they're entirely `accelerate.Accelerator`'s job. `pragma.training.hardware.resolve_mixed_precision` is the *only* place actual hardware capability is checked (bf16/fp16 support), and only to gracefully downgrade a requested precision, not to pick a device.

- **`hardware.py`** — `resolve_mixed_precision(requested) -> "no"|"fp16"|"bf16"` — downgrades gracefully (e.g. `bf16` on non-bf16 CUDA → `fp16`; `fp16` on CPU → `no`).
- **`muon.py`** — `Muon(Optimizer)`, a minimal direct port of Keller Jordan's Newton–Schulz-orthogonalized momentum optimizer for 2D hidden-layer weight matrices (`_zeropower_via_newton_schulz`).
- **`optimizer.py`** — `build_optimizer(model, config)`:
  - `"adamw"` → plain `torch.optim.AdamW` over everything.
  - `"muon_adamw"` → `HybridOptimizer(Muon, AdamW)` — every 2D `Linear` weight (`q_proj`/.../`fc2`/`PragmaMLMHead.proj`) routes to Muon via `_is_muon_eligible` (`ndim==2 and "embedding" not in name`); the shared embedding table and every 1D param (LayerNorm, biases) route to AdamW. `HybridOptimizer` exposes the minimal `Optimizer`-like surface (`param_groups`, `step`, `zero_grad`, `state_dict`/`load_state_dict`) both `Accelerate` and the custom scheduler need — but it is **not** a `torch.optim.Optimizer` subclass (composes two real optimizers instead of faking one flat list), which is exactly why...
- **`scheduler.py`** — `WarmupDecayScheduler` is a small **custom** scheduler, not `LambdaLR` (which requires `isinstance(optimizer, torch.optim.Optimizer)`, incompatible with `HybridOptimizer`). Multiplies each param group's own base LR by a shared warmup/cosine-or-constant decay factor. Built from **optimizer-update count**, not epochs (dynamic batches have variable record counts per step).
- **`engine.py`** — `PretrainingEngine`, the durable train loop:
  - Construction order matters: model + optimizer built with real `model.parameters()` **before** `accelerator.prepare()`, so DDP-wrapping afterward still shares the same tensors the optimizer already references.
  - Wires: `TokenBudgetBatchSampler` (rank/replica-aware) → `MaskingPlanner`/`PragmaCollator` → `PragmaForMaskedModeling` → `build_optimizer`/`build_scheduler` → `CheckpointManager` → MLflow (one run per engine instance, explicitly started/ended to avoid stale-run bugs across multiple engines in one process).
  - `.train(start_epoch, end_epoch, val_dataset=None) -> list[EpochMetrics]` — supports partial-epoch runs (checkpoint-resume tests exploit this) while keeping the scheduler's warmup/decay horizon fixed to the *original* `n_epochs`.
  - `.evaluate(dataset) -> float` — fixed-epoch-0 masking, plain `DataLoader`, no grad — a consistent, comparable-across-epochs validation loss.
  - `.resume(checkpoint_name)` / `.save_checkpoint(name, ...)` — delegate to `CheckpointManager`.
  - `EpochMetrics` — per-epoch loss/throughput (`event_tokens_per_second`).
- **`debug_loop.py`** — the **Phase 6 correctness smoke test**, deliberately separate from `PretrainingEngine` (minimal, plain AdamW, no dynamic batching/checkpointing, tiny in-memory record list):
  - `run_debug_training(...) -> DebugTrainingResult` — trains a few epochs, reports per-epoch loss **and a per-mask-source breakdown** (`_per_origin_losses`) to prove all three masking strategies (token/event/key) individually drive the loss down.
  - `shuffle_context_tokens(batch, generator)` — corrupts *content* (permutes non-masked value tokens among themselves) to test context dependence; kept as an exploratory diagnostic, not an assertion (documented caveat: on tiny low-cardinality data the loss barely moves even though the model provably depends on context).
  - `context_dependency_grad(model, batch)` — the actual pass/fail check: confirms `d(loss)/d(embedding row)` is non-zero for every context (non-masked) value ID — a gradient-connectivity check, robust against a specific fit's numerical insensitivity.

---

## 9. Artifacts layer — `src/pragma/artifacts/checkpoint.py` (ADR 0007, ADR 0011)

- `TrainingCounters` — `global_step`, `tokens_processed`, `events_processed`, `records_processed`, `epoch` (tokens/events processed is the primary progress scale, not epochs — section 13.3).
- `hash_processor_bundle(processor_dir) -> sha256` — fingerprints `bundle.json` + `bpe_tokenizer.json` together; used to verify a checkpoint's exact processor is present, both on resume and when a downstream task reuses the checkpoint.
- `CheckpointManager`:
  - `.save(name, accelerator, optimizer, scheduler, ...) -> Path` — `accelerator.save_state` covers model weights + RNG state (Python/NumPy/CPU/GPU); optimizer/scheduler state are saved **explicitly** via plain `torch.save`, because `HybridOptimizer` isn't a real `torch.optim.Optimizer` so `Accelerate` silently skips it during `save_state` (confirmed directly, not assumed — see file docstring). Also writes a `manifest.json`: configs, processor hash, git revision, `uv.lock` hash, validation metrics, `is_best` flag.
  - `.load(name, accelerator, optimizer, scheduler, processor_dir)` — refuses to load if the processor hash doesn't match (a checkpoint is invalid without its exact processor, per ADR 0007) or if no manifest exists.
  - `.read_manifest(name)` — used by scripts/`load_frozen_backbone` to pull `pragma_config` back out without a full state load.

---

## 10. Evaluation layer — `src/pragma/evaluation/` (Phase 9-10, ADR 0012)

Answers "does pretraining actually help downstream tasks, compared to a data scientist's usual approach?" — a frozen backbone, never gradient-updated here.

- **`embeddings.py`**:
  - `EmbeddingBundle` — `entity_ids` + `usr` (contextual `[USR]`), `last_event` (final contextual `[EVT]`, zeros for a zero-event record — a real, valid "nothing happened yet" embedding, not a missing value), `concat` (`cat([usr, last_event])`).
  - `EmbeddingExtractor(model, batch_size, device)` — runs a **frozen** `PragmaModel` backbone (not the masked-modeling wrapper) with **no masking** (plain `PragmaCollator()`) and extracts all three variants.
  - `extract_record_embeddings(...)` — the `[USR]`-only convenience function kept for Phase 9 callers.
- **`probe.py`**:
  - `ProbeResult(name, auc, n_train, n_val)` — shared result shape reused by baselines and LoRA results too (so everything fits one comparison table).
  - `run_linear_probe(entity_ids, embeddings, labels, val_entity_ids, seed)` — standard-scaled `LogisticRegression`, fit on train-split entities, AUC on val. Reports `nan` honestly on a degenerate (single-class) split rather than raising or faking a score.
  - `LinearProbeRunner` — runs `run_linear_probe` across all three `EmbeddingBundle` variants (`probe_usr`, `probe_last_event`, `probe_concat`).
- **`features.py`** — `build_aggregated_features(records) -> DataFrame` — the *conventional* feature set a data scientist would hand-build without a foundation model: `n_events`, `n_distinct_event_types`, `total_amount`, `mean_amount`, `days_since_signup`, milestone-presence flags, static categorical attributes. Read from the same leakage-safe `EvaluationRecord`s the tokenizer sees, so the comparison is apples-to-apples.
- **`baselines.py`** — `run_baselines(features_df, labels, val_entity_ids, seed) -> [ProbeResult]`: `LogisticRegression` + `HistGradientBoostingClassifier` (chosen over LightGBM to avoid a second heavy native dependency — ADR 0012) inside a `ColumnTransformer`/`Pipeline` (median-impute + scale numeric, constant-impute + one-hot categorical). Results named `baseline_logreg`/`baseline_gbdt`.
- **`report.py`** — `build_comparison_report(results, checkpoint_name) -> DataFrame`: combines probe/baseline/LoRA `ProbeResult`s into one table sorted by AUC descending (NaN last). `_kind(name)` classifies `"probe_*"` → `"probe"`, `"baseline_*"` → `"baseline"`, anything else (e.g. Phase 11's `"lora_finetuned"`) → `"lora"` — **note**: this three-way branch replaced an earlier two-way version that silently mis-classified LoRA results as baselines; a regression test (`test_build_comparison_report_classifies_lora_results_as_their_own_kind`) now guards it.

---

## 11. Downstream layer — `src/pragma/downstream/` (Phase 11, ADR 0013)

**LoRA, not QLoRA** — see ADR 0013 for the full reasoning; the short version: QLoRA adds 4-bit quantization of the frozen base specifically to fit fine-tuning of multi-billion-parameter LLMs into limited GPU memory. PRAGMA-S is ~10M parameters — there is no memory pressure to trade away, and the plan's own section 15.2 only calls for PEFT, not quantization. Plain LoRA is what "follow the paper's direction" resolves to at this scale.

**What LoRA does here, mechanically**: every targeted `nn.Linear` (`q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2` — the exact names `PragmaTransformerBlock` was deliberately given, ADR 0011) gets a frozen `W` plus a trainable low-rank update `W_effective = W_frozen + (alpha/r) * B @ A`, with `B` initialized to zero so the adapted model starts numerically identical to the frozen base. Only `A`/`B` (and `modules_to_save` — the task head) ever receive gradients.

- **`task_model.py`**:
  - `PragmaTaskOutput(ModelOutput)` — `loss`, `logits` (`[n_records]`, pre-sigmoid), `record_embeddings`.
  - `PragmaForTask(PreTrainedModel)` — `PragmaModel` backbone + single `nn.Linear(hidden_size, 1)` binary classification head on `record_embeddings` (the `[USR]` state — the same variant Phase 10's probe comparison showed carries as much signal as `concat`). Binary-only for now; multiclass/multilabel/regression would be a mechanical extension (swap loss + output width), not built speculatively.
- **`lora.py`**:
  - `build_lora_model(model, config) -> PeftModel` — wraps `PragmaForTask` with `peft.LoraConfig`/`get_peft_model`; freezes everything except the injected LoRA matrices and `modules_to_save`.
  - `load_lora_model(pragma_config, adapter_dir, base_checkpoint_dir, base_checkpoint_name, processor_dir) -> PeftModel` — reloads an adapter **independently** of the training run that produced it: rebuilds a fresh frozen backbone from the base checkpoint (verifying its processor hash), then reattaches the saved adapter via `PeftModel.from_pretrained`. Imports `load_frozen_backbone` from `trainer.py` **inside the function body**, not at module level, specifically to break a circular import (`trainer.py` imports `build_lora_model` from this module).
- **`trainer.py`**:
  - `DownstreamCollator(labels)` — wraps a plain (unmasked) `PragmaCollator` and returns `(batch, label_tensor)`.
  - `DownstreamEpochMetrics(epoch, train_loss, val_loss, val_auc)`.
  - `load_frozen_backbone(pragma_config, checkpoint_dir, checkpoint_name, processor_dir) -> PragmaForTask` — verifies the checkpoint's processor hash (same check as `CheckpointManager.load`), then restores **only** the `.pragma` backbone's `state_dict` into a fresh `PragmaForTask` via a throwaway `Accelerator().prepare(PragmaForMaskedModeling(...))` + `load_state`; the classifier head stays randomly initialized.
  - `DownstreamTrainer` — mirrors `PretrainingEngine`'s hardware posture exactly (`Accelerator()`, `resolve_mixed_precision`, `accelerator.prepare(model, optimizer, train_loader)`). Deliberate simplification vs. pretraining: a plain fixed-batch `DataLoader` (relies on `accelerator.prepare()`'s automatic DDP sharding) instead of `TokenBudgetBatchSampler`, since downstream fine-tuning datasets are small and fixed-size.
    - `.train() -> list[DownstreamEpochMetrics]` — standard loop; `.evaluate(loader) -> (mean_loss, auc)`.
    - `.save_adapter(path) -> Path` — `unwrapped.save_pretrained(path)` (LoRA weights + task head only) plus a sidecar `adapter_manifest.json` (base checkpoint identity/hash, processor hash, the `DownstreamConfig` used) — never touches or duplicates the frozen base checkpoint.

---

## 12. Config layer — `src/pragma/config/`

Every config is a frozen dataclass with `to_dict`/`from_dict`/`save`/`load` (JSON), so nothing is a hardcoded constant buried in code (an explicit project convention — see `DownstreamConfig`'s docstring quoting the plan's section 18 closing instruction).

| Config | File | Key fields |
|---|---|---|
| `ProcessorConfig` | `processor_config.py` | `n_numeric_buckets`, `bpe_vocab_size`, `max_profile_tokens`(200), `max_event_tokens`(24), `max_history_events`(6500) — the last three match the paper's PRAGMA-S caps. |
| `MaskingConfig` | `masking_config.py` | `token_mask_prob`(.15), `event_mask_prob`(.10), `key_mask_prob`(.10), `unk_dropout_frac`(.05), `seed`. |
| `TokenBudgetConfig` | `training_config.py` | `max_event_tokens_per_batch`, `max_events_per_batch`, `max_records_per_batch`, `n_length_buckets`. |
| `TrainingConfig` | `training_config.py` | optimizer (`"adamw"`\|`"muon_adamw"`), LR/betas/weight decay, `scheduler`, `mixed_precision`, `n_epochs`, checkpoint/log intervals, MLflow URI/experiment. **No `device`/`use_ddp` field by design** — that's `Accelerate`'s job. |
| `DownstreamConfig` | `downstream_config.py` | `lora_r`(8)/`lora_alpha`(8)/`lora_dropout`, `target_modules`, LR/epochs/batch size, `modules_to_save`. |
| `PragmaConfig` | `modeling/config.py` (not here, but the model's config) | dimensions + special-token IDs; `.from_processor(processor, **overrides)`. |

`configs/` on disk holds JSON instances: `configs/processor/default.json`, `configs/downstream/default.json` populated; `configs/model/` and `configs/pretraining/` currently hold only `.gitkeep` placeholders (see §14, "things to look at").

---

## 13. Scripts — thin CLI entry points (`scripts/`)

Each wires config → library calls → disk I/O, no logic of its own (per `CLAUDE.md`). In rough pipeline order:

1. `generate_synthetic_data.py` — `SyntheticDataConfig` → `generate_synthetic_corpus` + `validate_corpus` → `data/raw/{events,profile_state}.parquet`.
2. `fit_processor.py` — raw parquet → `PointInTimeRecordBuilder` → `PragmaProcessor.fit` → `data/processor/` bundle + fit report.
3. `tokenize_shards.py` — raw parquet + processor bundle → `PragmaProcessor.transform` → `write_dataset` → `data/shards/` (Parquet shards + manifest).
4. `pretrain.py` — shards + processor → `PretrainingEngine.train()`, optional periodic frozen-embedding probe (`--probe-every-n-epochs`) via `extract_record_embeddings`/`run_linear_probe`; defaults MLflow to a persistent `data/mlflow/mlflow.db` (not `TrainingConfig`'s bare relative default). What `accelerate launch scripts/pretrain.py ...` runs on real hardware.
5. `extract_embeddings.py` — a checkpoint + shard split → `EmbeddingExtractor` → a Parquet file of `{usr, last_event, concat}` per entity.
6. `run_probe.py` — saved embeddings + raw tables → `LinearProbeRunner` + `run_baselines` + `build_comparison_report` → a CSV report.
7. `finetune_lora.py` — a base checkpoint + shards + label column → `DownstreamTrainer.train()` → `save_adapter()`.

---

## 14. Notebooks — `notebooks/` (narrative/exploration only, never a source of truth)

One per phase, matching `plans/progress.md`'s phase numbering:

| # | Notebook | Demonstrates |
|---|---|---|
| 000 | `synthetic_data_generation` | `pragma.data.synthetic` end-to-end + validation report |
| 001 | `point_in_time_records` | `PointInTimeRecordBuilder`, leakage checks |
| 002 | `fit_processor` | `PragmaProcessor.fit`, vocabulary/bucket inspection |
| 003 | `tokenize_shards` | `write_dataset`/`read_dataset`, shard bucketing |
| 004 | `reference_batching` | `PragmaCollator`, `PragmaBatch.validate()` |
| 005 | `masking_inspection` | `MaskingPlanner`, mask-source visualization |
| 006 | `model_architecture` | `PragmaModel` forward pass, shape/gradient sanity |
| 007 | `debug_training` | `run_debug_training`, `context_dependency_grad` |
| 008 | `varlen_attention` | `PaddedAttentionBackend` vs `VarLenAttentionBackend` parity + `compare_attention_cost` |
| 009 | `distributed_pretraining` | `PretrainingEngine`, checkpoint save/resume |
| 010 | `pilot_pretraining` | AdamW vs Muon+AdamW pilot comparison (`scripts/pretrain.py`) |
| 011 | `embedding_probes_and_baselines` | `EmbeddingExtractor`, `LinearProbeRunner`, `run_baselines`, `build_comparison_report` |
| 012 | `lora_finetuning` | LoRA mechanism/math explanation, LoRA-vs-QLoRA reasoning, full `DownstreamTrainer` pipeline, adapter reload verification |

Each opens with a markdown cell stating what it demonstrates and which plan section/ADR it follows; all real logic lives in the library, notebooks only call it and inspect results. Notebooks default to a fast `DEBUG=True` scale for CI-speed execution but document real pilot-scale (e.g. 300-entity) numbers captured from separate background runs in a markdown cell.

---

## 15. Tests — `tests/`

- **`unit/`** — one module per library module (`test_records.py`, `test_processor.py`, `test_masking.py`, `test_modeling.py`, `test_batch.py`, `test_token_budget_sampler.py`, `test_lora.py`, `test_task_model.py`, `test_synthetic_data.py`, etc.) — fast, tmp-dir I/O only.
- **`integration/`** — cross-component and slower: `test_engine_evaluate.py`, `test_checkpoint_resume.py` (proves resumed training reproduces the uninterrupted run given the same `n_epochs`), `test_embedding_probe.py`, `test_probes_and_baselines.py`, `test_downstream_trainer.py` (adapter save/reload byte-for-byte base-checkpoint immutability + exact-logit reload match), `test_overfit.py`.
- **`parity/`** — `test_attention_backend_parity.py` (padded vs. varlen numerical agreement), `test_attention_cost_comparison.py`.
- **`distributed/`** — reserved (currently only `__init__.py`) for later real multi-GPU/DDP tests per the phase plan; not a leftover, an intentional placeholder (`CLAUDE.md` documents this explicitly).

---

## 16. Cross-cutting decisions worth knowing up front

- **Point-in-time correctness is enforced structurally, not by convention** — `PointInTimeRecordBuilder` clips milestones itself; `PragmaProcessor.fit` filters to `split=="train"` itself; `read_dataset` checksums every shard. None of these rely on a caller remembering to do the right thing.
- **One shared token ID space** — special tokens, keys, and every value type share one embedding table and one contiguous ID layout (`[special][keys][numeric][categorical][bpe]`), decided once at `PragmaProcessor.__init__`/`fit()` and read into `PragmaConfig.from_processor`.
- **Packed, never padded, for real execution** — `PragmaBatch` is flat buffers + offsets; `PaddedAttentionBackend` exists purely as a correctness oracle for tests, `VarLenAttentionBackend` is what training actually uses.
- **Hardware selection is `Accelerate`'s job, everywhere** — `PretrainingEngine`, `debug_loop.run_debug_training`, and `DownstreamTrainer` all construct a plain `Accelerator()` and never branch on `torch.cuda.is_available()` except inside `resolve_mixed_precision`. Multi-GPU DDP is `accelerate launch --multi_gpu --num_processes=N script.py` — zero code changes.
- **A checkpoint is invalid without its exact processor bundle** (ADR 0007) — enforced by hash comparison in three independent places: `CheckpointManager.load`, `load_frozen_backbone`, and (implicitly) anything that calls either.
- **LoRA reload is fully independent of the training process that produced it** — `load_lora_model` rebuilds everything from disk artifacts only, which is exactly the path a real deployment would use.

---

## 17. Things to look at (possible leftovers — not deleted, flagging for a decision)

Found while building this map; none of these break anything, but they're worth a conscious keep/delete call:

1. **`tests/_scratch_checkpoint_resume/`** — contains `mlflow_resumed.db` and `mlflow_uninterrupted.db` (~856 KB each), sitting directly under `tests/`, untracked and not matching any `tests/unit`/`tests/integration` naming pattern. This looks like a stray output directory from a manual/offline checkpoint-resume experiment rather than a fixture the test suite depends on (the actual `test_checkpoint_resume.py` uses `tmp_path`). Likely safe to delete, but worth confirming nothing local still points at it.
2. **`configs/model/.gitkeep` and `configs/pretraining/.gitkeep`** — intentionally empty per `configs/README.md` ("populated starting Phase 2" / model variants "and later PRAGMA-M/L") — not a leftover, just flagging that they're still empty as of Phase 11. No action needed unless you want to seed them with the actual configs currently only living as constructor defaults (`PragmaConfig`'s defaults, `TrainingConfig`'s defaults) and `scripts/pretrain.py`'s CLI-flag defaults.
3. **`tests/distributed/`** — only `__init__.py`, no test modules yet. Also intentional (`CLAUDE.md` reserves it for later multi-GPU work), not a leftover.
4. **`__pycache__/` directories throughout `src/pragma/**`** — already covered by `.gitignore`, harmless, not worth mentioning again except to note they showed up in a raw filesystem scan and are not actually tracked.

Nothing else in the scan looked like dead code or an abandoned module — every file under `src/pragma/` is imported from at least one script, notebook, or test.
