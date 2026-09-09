# PRAGMA: Paper → Code Walkthrough (1h team presentation)

Primary reference: Ostroukhov et al., *PRAGMA: Revolut Foundation Model* (2026).
This doc maps the paper's key ideas to where they live in this repo, in the
order the system is actually built and run (data → processor → model →
masking → batching → training → evaluation → adaptation). Use it as the
talk outline; each row is a "here's the paper concept, here's the code"
beat.

For deeper "why did we decide it this way" context on any row, the linked
ADR is the paper-ambiguity resolution; `plans/PRAGMA-Implementation-Plan.md`
section numbers are cited for the paper-faithful design description.

## Topics

- Framing: what PRAGMA is, MVP scope, non-goals (Sections 1-2)
- Data contract & point-in-time correctness (rows 1-3)
- Structured processor & vocabulary (rows 4-8)
- Model architecture: encoders, embeddings, temporal encoding (rows 9-15)
- Masking objective (rows 16-18)
- Packed batching, attention backends, training loop (rows 19-24)
- Evaluation & LoRA downstream adaptation, questions (rows 25-27)

---

## 1. Data contract & point-in-time correctness

| # | Paper concept | Plan section | Code (library) | Scripts / Notebooks | ADR |
|---|---|---|---|---|---|
| 1 | Raw event & profile-state records (heterogeneous banking events) | §5.1–5.2 | [src/pragma/data/records.py](src/pragma/data/records.py), [src/pragma/schema/registry.py](src/pragma/schema/registry.py) | [scripts/generate_synthetic_data.py](scripts/generate_synthetic_data.py), [notebooks/000_synthetic_data_generation.ipynb](notebooks/000_synthetic_data_generation.ipynb) | [0001](docs/adr/0001-data-schema.md) |
| 2 | Point-in-time evaluation records — no future leakage, deterministic tie-breaking | §5.3–5.4 | [src/pragma/data/records.py](src/pragma/data/records.py) | [notebooks/001_point_in_time_records.ipynb](notebooks/001_point_in_time_records.ipynb) | [0002](docs/adr/0002-evaluation-points.md) |
| 3 | Synthetic data covering value types & edge cases (zero/single-event, long history, same-timestamp, rare/OOV categories); zero-event customer exclusion | §Phase 1 / new decision | [src/pragma/data/synthetic.py](src/pragma/data/synthetic.py) | [notebooks/000_synthetic_data_generation.ipynb](notebooks/000_synthetic_data_generation.ipynb) | [0014](docs/adr/0014-zero-event-customer-exclusion.md) |

## 2. Structured processor & token vocabulary

| # | Paper concept | Plan section | Code (library) | Scripts / Notebooks | ADR |
|---|---|---|---|---|---|
| 4 | Fitted structured processor (not a generic NLP tokenizer) | §6.1, §6.5 | [src/pragma/processing/processor.py](src/pragma/processing/processor.py) | [scripts/fit_processor.py](scripts/fit_processor.py), [notebooks/002_fit_processor.ipynb](notebooks/002_fit_processor.ipynb) | [0003](docs/adr/0003-structured-processor.md) |
| 5 | Key vocabulary & shared token ID space | §6.4 | [src/pragma/processing/vocabulary.py](src/pragma/processing/vocabulary.py), [src/pragma/processing/special_tokens.py](src/pragma/processing/special_tokens.py) | [notebooks/002_fit_processor.ipynb](notebooks/002_fit_processor.ipynb) | [0008](docs/adr/0008-token-vocabulary-and-evaluation-sampling.md) |
| 6 | Numeric bucketization (global shared bucket tokens, per-key boundaries) | §6.2 | [src/pragma/processing/numeric.py](src/pragma/processing/numeric.py) | [notebooks/002_fit_processor.ipynb](notebooks/002_fit_processor.ipynb) | — |
| 7 | Categorical vs. free-text values (per-key namespaced vocab + shared `[UNK]`; byte-level BPE for text) | §6.3 | [src/pragma/processing/categorical.py](src/pragma/processing/categorical.py), [src/pragma/processing/text_bpe.py](src/pragma/processing/text_bpe.py) | [notebooks/002_fit_processor.ipynb](notebooks/002_fit_processor.ipynb) | — |
| 8 | Temporal features (soft-log transform, calendar cyclical features) & tokenized record output | §6.2, §7.3 | [src/pragma/processing/temporal.py](src/pragma/processing/temporal.py), [src/pragma/processing/tokenized_record.py](src/pragma/processing/tokenized_record.py) | [notebooks/002_fit_processor.ipynb](notebooks/002_fit_processor.ipynb), [notebooks/003_tokenize_shards.ipynb](notebooks/003_tokenize_shards.ipynb) | — |
| 9 | Sharded, tokenized on-disk corpus | §10.1 | [src/pragma/data/storage.py](src/pragma/data/storage.py) | [scripts/tokenize_shards.py](scripts/tokenize_shards.py), [notebooks/003_tokenize_shards.ipynb](notebooks/003_tokenize_shards.ipynb) | — |

## 3. Model architecture (PRAGMA-S)

| # | Paper concept | Plan section | Code (library) | Scripts / Notebooks | ADR |
|---|---|---|---|---|---|
| 10 | PRAGMA-S overall config / hierarchical encoder stack | §7.1, §7.4 | [src/pragma/modeling/model.py](src/pragma/modeling/model.py), [src/pragma/modeling/config.py](src/pragma/modeling/config.py) | [notebooks/006_model_architecture.ipynb](notebooks/006_model_architecture.ipynb) | [0009](docs/adr/0009-model-architecture-decisions.md) |
| 11 | Profile State Encoder, Event Encoder, History Encoder (three-level hierarchy) | §7.4 | [src/pragma/modeling/encoders.py](src/pragma/modeling/encoders.py) | [notebooks/006_model_architecture.ipynb](notebooks/006_model_architecture.ipynb) | [0009](docs/adr/0009-model-architecture-decisions.md) |
| 12 | Shared key/value token embeddings | §7.2 | [src/pragma/modeling/embeddings.py](src/pragma/modeling/embeddings.py) | [notebooks/006_model_architecture.ipynb](notebooks/006_model_architecture.ipynb) | — |
| 13 | Continuous-time RoPE, calendar features, within-field positions | §7.3 | [src/pragma/modeling/rope.py](src/pragma/modeling/rope.py), [src/pragma/modeling/calendar.py](src/pragma/modeling/calendar.py) | [notebooks/006_model_architecture.ipynb](notebooks/006_model_architecture.ipynb) | — |
| 14 | Transformer block (attention + FFN) used by each encoder level | §7.4 | [src/pragma/modeling/transformer_block.py](src/pragma/modeling/transformer_block.py) | [notebooks/006_model_architecture.ipynb](notebooks/006_model_architecture.ipynb) | — |
| 15 | Three-level MLM head (key/value/whole-event prediction) | §7.5 | [src/pragma/modeling/mlm_head.py](src/pragma/modeling/mlm_head.py), [src/pragma/modeling/outputs.py](src/pragma/modeling/outputs.py) | [notebooks/006_model_architecture.ipynb](notebooks/006_model_architecture.ipynb) | — |

## 4. Masking objective

| # | Paper concept | Plan section | Code (library) | Scripts / Notebooks | ADR |
|---|---|---|---|---|---|
| 16 | Mask sources: individual-token, whole-event, semantic-key masking | §8.1 | [src/pragma/masking/planner.py](src/pragma/masking/planner.py) | [notebooks/005_masking_inspection.ipynb](notebooks/005_masking_inspection.ipynb) | [0005](docs/adr/0005-masking-semantics.md) |
| 17 | Corruption strategy & label semantics (overlap handling) | §8.2 | [src/pragma/masking/planner.py](src/pragma/masking/planner.py), [src/pragma/config/masking_config.py](src/pragma/config/masking_config.py) | [notebooks/005_masking_inspection.ipynb](notebooks/005_masking_inspection.ipynb) | [0005](docs/adr/0005-masking-semantics.md) |
| 18 | Masking applied at batch-build time | §8 | [src/pragma/data/batch.py](src/pragma/data/batch.py) | [notebooks/005_masking_inspection.ipynb](notebooks/005_masking_inspection.ipynb) | — |

## 5. Variable-length representation, batching & training

| # | Paper concept | Plan section | Code (library) | Scripts / Notebooks | ADR |
|---|---|---|---|---|---|
| 19 | Canonical packed batch (cumulative sequence offsets, no cross-record leakage) | §9.1 | [src/pragma/data/batch.py](src/pragma/data/batch.py), [src/pragma/modeling/packing.py](src/pragma/modeling/packing.py) | [notebooks/004_reference_batching.ipynb](notebooks/004_reference_batching.ipynb) | [0004](docs/adr/0004-packed-batches.md) |
| 20 | Dual attention backends: padded reference (correctness oracle) vs. packed varlen | §9.2 | [src/pragma/attention/backend.py](src/pragma/attention/backend.py), [src/pragma/attention/padded.py](src/pragma/attention/padded.py), [src/pragma/attention/varlen.py](src/pragma/attention/varlen.py) | [notebooks/008_varlen_attention.ipynb](notebooks/008_varlen_attention.ipynb), [tests/parity/test_attention_backend_parity.py](tests/parity/test_attention_backend_parity.py) | [0006](docs/adr/0006-attention-backends.md), [0010](docs/adr/0010-varlen-attention-kernel.md) |
| 21 | Dynamic token/event-budget batch sampler | §9.3 | [src/pragma/data/token_budget_sampler.py](src/pragma/data/token_budget_sampler.py), [src/pragma/data/dataset.py](src/pragma/data/dataset.py) | [notebooks/004_reference_batching.ipynb](notebooks/004_reference_batching.ipynb) | — |
| 22 | Training engine: Accelerate-driven loop, bf16, checkpoint contents (model/optimizer/scheduler/RNG/config/processor/data manifest) | §13 | [src/pragma/training/engine.py](src/pragma/training/engine.py), [src/pragma/artifacts/checkpoint.py](src/pragma/artifacts/checkpoint.py), [src/pragma/training/hardware.py](src/pragma/training/hardware.py) | [scripts/pretrain.py](scripts/pretrain.py), [notebooks/007_debug_training.ipynb](notebooks/007_debug_training.ipynb), [notebooks/009_distributed_pretraining.ipynb](notebooks/009_distributed_pretraining.ipynb) | [0007](docs/adr/0007-checkpoint-format.md), [0011](docs/adr/0011-distributed-training-engine-decisions.md) |
| 23 | Hybrid Muon+AdamW optimizer / LR scheduler | §13.2 | [src/pragma/training/muon.py](src/pragma/training/muon.py), [src/pragma/training/optimizer.py](src/pragma/training/optimizer.py), [src/pragma/training/scheduler.py](src/pragma/training/scheduler.py) | [notebooks/007_debug_training.ipynb](notebooks/007_debug_training.ipynb) | [0011](docs/adr/0011-distributed-training-engine-decisions.md) |
| 24 | Pilot pretraining run on PRAGMA-S corpus | §Phase 9 | — | [notebooks/010_pilot_pretraining.ipynb](notebooks/010_pilot_pretraining.ipynb) | — |

## 6. Evaluation & downstream adaptation

| # | Paper concept | Plan section | Code (library) | Scripts / Notebooks | ADR |
|---|---|---|---|---|---|
| 25 | Frozen embedding extraction & MLM diagnostics | §14.1 | [src/pragma/evaluation/embeddings.py](src/pragma/evaluation/embeddings.py) | [scripts/extract_embeddings.py](scripts/extract_embeddings.py), [notebooks/011_embedding_probes_and_baselines.ipynb](notebooks/011_embedding_probes_and_baselines.ipynb) | [0012](docs/adr/0012-evaluation-harness-decisions.md) |
| 26 | Transfer evaluation: linear probes & conventional baselines | §14.2–14.3 | [src/pragma/evaluation/probe.py](src/pragma/evaluation/probe.py), [src/pragma/evaluation/baselines.py](src/pragma/evaluation/baselines.py), [src/pragma/evaluation/features.py](src/pragma/evaluation/features.py), [src/pragma/evaluation/report.py](src/pragma/evaluation/report.py) | [scripts/run_probe.py](scripts/run_probe.py), [notebooks/011_embedding_probes_and_baselines.ipynb](notebooks/011_embedding_probes_and_baselines.ipynb) | [0012](docs/adr/0012-evaluation-harness-decisions.md) |
| 27 | LoRA downstream adaptation of the pretrained backbone | §15 | [src/pragma/downstream/lora.py](src/pragma/downstream/lora.py), [src/pragma/downstream/task_model.py](src/pragma/downstream/task_model.py), [src/pragma/downstream/trainer.py](src/pragma/downstream/trainer.py) | [scripts/finetune_lora.py](scripts/finetune_lora.py), [notebooks/012_lora_finetuning.ipynb](notebooks/012_lora_finetuning.ipynb) | [0013](docs/adr/0013-lora-downstream-adaptation-decisions.md) |

---

## Where we are right now

Per [plans/progress.md](plans/progress.md): **Phase 11 complete, next up is Phase 12**
(optimize and decide on scale-up — the last phase before this MVP is "done").
Everything in the tables above through row 27 is implemented and tested; the
open items are optimization/scale-up decisions, not new capabilities.

## What the paper leaves ambiguous (good discussion material)

See §18 of the plan for the full list (data/tokenization, masking/objective,
architecture, pretraining, LoRA/downstream — things the paper doesn't fully
specify). Every ambiguity that affects this implementation was resolved as
one of the 14 ADRs in [docs/adr/](docs/adr/README.md) — worth pointing at a
couple of these live as examples of "the paper said X but not how", e.g.
[0008](docs/adr/0008-token-vocabulary-and-evaluation-sampling.md) (token ID
space) or [0014](docs/adr/0014-zero-event-customer-exclusion.md) (zero-event
customers).

## Explicit non-goals (set expectations early)

PRAGMA-M/L scale, the paper's 207B-token corpus, cross-customer graph/AML
modelling, the optional frozen external text encoder, federated training,
and exact reproduction of Revolut's unpublished hyperparameters/results are
all out of scope for this MVP (plan §2.2). This is an architecture and
training-recipe reproduction on synthetic data, not a metrics reproduction.
