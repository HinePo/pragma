# Configs

Configuration files for model, processor, pretraining, and downstream runs
(implementation plan, section 12). Populated starting Phase 2, once
`ProcessorConfig` and `PragmaConfig` exist to load them.

- `model/` — `PragmaConfig` variants (PRAGMA-S, and later PRAGMA-M/L).
- `processor/` — `ProcessorConfig` variants (vocabulary rules, numeric buckets, BPE settings).
  `default.json` is loaded by `scripts/fit_processor.py`'s defaults (n_numeric_buckets=16,
  bpe_vocab_size=512); override via CLI flags or `ProcessorConfig.load(...)` for other runs.
- `pretraining/` — `TrainingConfig` and `MaskingConfig` variants.
- `downstream/` — `DownstreamConfig` variants (LoRA rank/alpha/target modules,
  fine-tuning hyperparameters — ADR 0013). `default.json` is
  `scripts/finetune_lora.py`'s default (`lora_r=8, lora_alpha=8`, section 15.2's
  starting point); override via CLI flags or `DownstreamConfig.load(...)`.
