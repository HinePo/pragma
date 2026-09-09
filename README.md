# PRAGMA

PRAGMA is a from-scratch reproduction of a **PRAGMA-style financial foundation model**: a hierarchical Transformer that learns reusable customer representations from heterogeneous, time-ordered banking events (transactions, top-ups, app events, profile changes, etc.).

The primary reference is Ostroukhov et al., *PRAGMA: Revolut Foundation Model* (2026). This project reproduces the architecture and training recipe described in that paper — it does not attempt to reproduce Revolut's data, exact hyperparameters, or reported metrics, which are unpublished.

## What this is

A durable, production-shaped MVP built around the **PRAGMA-S** (~10M parameter) configuration, covering the full pipeline end to end:

- **Point-in-time record construction** — converting raw event and profile-state tables into leakage-safe evaluation records.
- **A fitted structured processor** — turning typed key/value fields (numerical, categorical, text, temporal) into token IDs, without relying on a standard NLP tokenizer.
- **The PRAGMA-S backbone** — a Profile State Encoder, Event Encoder, and History Encoder, using continuous-time RoPE, calendar features, and a shared key/value embedding scheme.
- **Structured masked-modelling pretraining** — individual-token, whole-event, and semantic-key masking, trained on packed variable-length sequences.
- **Scalable training** — dynamic token-budget batching, multi-GPU bf16 training via Hugging Face Accelerate, and fully resumable checkpoints.
- **Downstream evaluation** — frozen embedding extraction, linear probes, conventional baselines, and LoRA-based fine-tuning for at least one supervised task.

Record-level embeddings are computed from a customer's event history and profile state up to an evaluation point, not looked up from a learned customer-ID table — so the model can produce embeddings for customers it never saw during training.

## What this isn't (yet)

This MVP intentionally excludes larger model scales (PRAGMA-M/L), the full-size training corpus from the paper, cross-customer graph modelling, federated training, and production online serving. These are scale and productization concerns layered on top of an architecture that is designed to support them without a rewrite.

## Project status

Early stage — the implementation plan and architectural decisions are defined; core components (schema, processor, model, training loop) are being built incrementally, phase by phase.

## Documentation

See [PRAGMA-Implementation-Plan.md](PRAGMA-Implementation-Plan.md) for the full technical plan: data contracts, model architecture, masking design, training system, testing strategy, and the phased development roadmap.
