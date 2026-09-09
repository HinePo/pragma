# CLI Guide

Order-of-operations guide to the `scripts/` entry points, in the sequence
you'd actually run them for a full pipeline pass — synthetic data through
LoRA fine-tuning. Each script is a thin wrapper (per `CLAUDE.md`'s
three-layer architecture); the real logic lives in `src/pragma/`. This
guide is about *how to invoke them*, not what they do internally — see
`plans/progress.md` for the phase-by-phase implementation history and
`docs/adr/` for the design decisions behind each step.

All commands assume `uv` and are run from the repo root:

```
uv run python scripts/<script>.py [args]
```

Commands below are written as single lines so they paste cleanly into any
shell. If you split one across lines yourself: Bash/Git Bash uses `\` as
the continuation character, PowerShell uses `` ` `` — they are not
interchangeable, and mixing them produces a parser error rather than a
Python one.

---

## 1. Generate synthetic data

Once you've done: nothing — this is the starting point. It creates the raw
corpus everything downstream reads from.

```
uv run python scripts/generate_synthetic_data.py --n-entities 500 --seed 1337
```

This will generate synthetic entity histories (events + profile state)
covering all value types and edge cases, validate the corpus (schema +
leakage checks), and write `events.parquet`, `profile_state.parquet`,
`manifest.json`, and `validation_report.json` to `data/raw/` (override with
`--out-dir`). Fails loudly if validation doesn't pass.

## 2. Fit the processor

Once you've generated `data/raw/` (step 1), you can run this to learn the
tokenization scheme from the train split only.

```
uv run python scripts/fit_processor.py
```

This will build point-in-time records from `data/raw/`, drop zero-event
records (ADR 0014), fit `PragmaProcessor` (key vocabulary, numeric
bucketizer, categorical encoder, BPE text encoder, temporal features) using
**only the train-split entities** (no leakage), and save the versioned
processor bundle + a fit report to `data/processor/` (override with
`--out-dir`). Use `--config` to point at a different `ProcessorConfig` JSON
(default: `configs/processor/default.json`), or `--n-numeric-buckets` /
`--bpe-vocab-size` to override specific fields without a config file.

## 3. Tokenize shards

Once you've fit a processor (step 2), you can run this to turn the raw
corpus into the tokenized Parquet shards training actually reads.

```
uv run python scripts/tokenize_shards.py
```

This will re-build point-in-time records, drop zero-event records, tokenize
every record with the fitted processor, and write one Parquet shard per
`(split, event-count bucket)` plus a data manifest with checksums to
`data/shards/` (override with `--out-dir`). Reads `data/raw/` and
`data/processor/` by default (override with `--raw-dir` / `--processor-dir`).

## 4. Pretrain

Once you've tokenized shards (step 3), you can run this to actually train
the PRAGMA backbone.

```
uv run python scripts/pretrain.py --n-epochs 5 --optimizer adamw
```

This will build the model/training/masking config (JSON files under
`configs/pretraining/` plus any CLI overrides — `--optimizer`, `--n-epochs`,
`--hidden-size`, `--num-heads`, `--intermediate-size`, `--event-layers`,
`--history-layers`, dynamic-batching budgets like
`--max-event-tokens-per-batch`), run `PretrainingEngine.train()` over the
tokenized shards, log metrics (loss, LR, and optionally probe AUC) to MLflow
at `data/mlflow/mlflow.db`, and save checkpoints to
`data/checkpoints/pretrain` (override with `--checkpoint-dir`; see
`checkpoint_every_n_steps` in `TrainingConfig` for how often it saves).
Pass `--probe-every-n-epochs N` to also run a periodic frozen-embedding
linear probe against `--probe-label-column` between training chunks
(needs `--raw-dir` for the label). This is the same entry point
`accelerate launch scripts/pretrain.py ...` would run on real multi-GPU
hardware (ADR 0011); on this dev machine it runs single-process on CPU.

To view training curves: `mlflow ui --backend-store-uri sqlite:///data/mlflow/mlflow.db`.

**Quick smoke tests:** `pretrain.py` has no dataset-size flag of its own —
it just reads whatever's in `--shards-dir`. The default `data/shards/`
(from steps 1–3 at their defaults) is pilot-scale, ~500 entities / ~108k
events, which is slow to iterate on for CPU-only dev. For a fast end-to-end
check, generate a small side corpus (steps 1–3 pointed at separate `_small`
dirs so you don't overwrite your real data) and shrink the model too:

```
uv run python scripts/generate_synthetic_data.py --n-entities 40 --out-dir data/raw_small
uv run python scripts/fit_processor.py --raw-dir data/raw_small --out-dir data/processor_small
uv run python scripts/tokenize_shards.py --raw-dir data/raw_small --processor-dir data/processor_small --out-dir data/shards_small

uv run python scripts/pretrain.py --shards-dir data/shards_small --processor-dir data/processor_small --raw-dir data/raw_small --n-epochs 2 --hidden-size 64 --num-heads 2 --event-layers 2 --history-layers 1 --max-event-tokens-per-batch 1024
```

Same pattern the debug/pilot notebooks (`007_debug_training.ipynb`,
`010_pilot_pretraining.ipynb`) use for their `CFG.DEBUG=True` smoke-test
path — small corpus + small model, not just fewer epochs, since epoch count
alone doesn't shrink per-step cost.

**`--hidden-size`/`--num-heads` must stay compatible:** `hidden_size /
num_heads` (the per-head dimension `ContinuousRoPE` rotates) must be an
even integer, or the model raises `ValueError: ContinuousRoPE requires an
even head_dim` at construction. The default pairing is `192 / 3 = 64`. If
you override `--hidden-size` for a smoke test, override `--num-heads` to
match — e.g. `--hidden-size 64 --num-heads 2` (head_dim 32).

## 5. Extract embeddings

Once you've pretrained a checkpoint (step 4), you can run this to pull
frozen embeddings out of it for downstream evaluation.

```
uv run python scripts/extract_embeddings.py --checkpoint-dir data/checkpoints/pretrain --split train --out-dir data/embeddings
uv run python scripts/extract_embeddings.py --checkpoint-dir data/checkpoints/pretrain --split val --out-dir data/embeddings
```

This will rebuild only the model (not optimizer/scheduler) from the
checkpoint's own saved `PragmaConfig`, run a frozen forward pass with a
plain (unmasked) collator over the tokenized shards, and write `[USR]`,
last-`[EVT]`, and their concatenation as `<split>_embeddings.parquet` to
`--out-dir` — so `run_probe.py` (or ad-hoc analysis) never needs to re-run
the model. Run it once per `--split` (default `"val"`; step 6 needs both
`train` and `val`). `--checkpoint-name` selects which saved checkpoint
under `--checkpoint-dir` to load (default `"final"`); `--batch-size`
controls the extraction batch size (default 16).

## 6. Run probes and baselines

Once you've extracted embeddings (step 5), you can run this to see whether
those embeddings actually carry useful signal, compared to conventional
baselines.

```
uv run python scripts/run_probe.py --embeddings-dir data/embeddings --out-path data/reports/probe_comparison.csv
```

This will run linear probes on each embedding variant (`[USR]`,
last-`[EVT]`, concat) against `--label-column` (default
`_downstream_is_high_value`), run conventional baselines
(logistic regression + `HistGradientBoostingClassifier`) on a hand-built
aggregated feature table from `data/raw/`, and write one ranked comparison
report (probes vs. baselines, same `ProbeResult` shape, AUC as the
metric) to `--out-path`. Uses the same train/val entity split pretraining
used, so the probe never trains on its own validation embeddings.

## 7. LoRA fine-tune a downstream task

Once you've pretrained a checkpoint (step 4) — extraction/probing (steps
5–6) are optional first checks, not a hard dependency — you can run this
to adapt the frozen backbone to one downstream task via LoRA.

```
uv run python scripts/finetune_lora.py --base-checkpoint-dir data/checkpoints/pretrain --adapter-out-dir data/checkpoints/lora_escalating_spender
```

This will re-verify the base checkpoint's processor hash, freeze the
backbone, attach a LoRA adapter (rank/alpha from
`configs/downstream/default.json` or `--lora-r` / `--lora-alpha`
overrides) targeting `q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2`, and
fine-tune only the adapter + a small classifier head against
`--label-column` (default `_downstream_is_escalating_spender`) via
`DownstreamTrainer`. Saves the trained adapter (and classifier head) to
`--adapter-out-dir`, leaving the base checkpoint on disk untouched
(verified byte-identical in `tests/integration/test_downstream_trainer.py`).
`--n-epochs`, `--batch-size`, `--learning-rate` override
`DownstreamConfig` defaults.

---

## Notes

- Every script has full `--help` output (`uv run python scripts/<script>.py --help`)
  listing every flag and default — this guide covers the common path, not
  every override.
- Each step has a matching notebook under `notebooks/` (numbered in
  pipeline order, e.g. `000_synthetic_data_generation.ipynb` through
  `012_lora_finetuning.ipynb`) that runs the same logic interactively with
  inspection/plots — useful for understanding what a step actually produces
  before trusting the CLI run.
- See `plans/progress.md` for what's actually been validated at each phase
  (exit gates, test coverage, pilot-run numbers) — this file only documents
  how to invoke the commands, not what's proven to work yet.
