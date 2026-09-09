# ADR 0011: Distributed pretraining engine decisions

**Status:** Accepted (Phase 8)

## Context

Section 18 leaves several Phase 8 questions open (Muon/AdamW parameter
routing and hyperparameters, distributed strategy/communication settings,
global token budget), and the dev machine cannot run or test multi-GPU
DDP or a real CUDA build at all (`CLAUDE.md`'s environment note). Phase 8
still has to ship a *correct, automatically-scaling* engine — one that uses
GPU/DDP when the infrastructure supports it and falls back to CPU/single-process
without any code change, so the same `PretrainingEngine` that runs here on
CPU today runs unmodified on a multi-GPU box later.

## Decisions

1. **Hardware selection is never hardcoded — it's entirely `Accelerate`'s
   job.** `PretrainingEngine` constructs a plain `Accelerator(...)` and asks
   it for `accelerator.device`, `accelerator.num_processes`,
   `accelerator.process_index`. Running the training script directly gives
   single-CPU-or-single-GPU execution; running it via `accelerate launch
   --multi_gpu --num_processes=N` gives DDP across N GPUs — with zero
   changes to `pragma.training` code either way. `TrainingConfig` has no
   `device` or `use_ddp` field; that would fight Accelerate's own detection
   rather than delegate to it.
2. **Mixed precision is resolved against actual hardware support, not
   requested blindly.** `resolve_mixed_precision(requested)` downgrades
   `"bf16"`/`"fp16"` to `"no"` when the current accelerator can't actually
   use it (e.g. `"fp16"` requested on CPU), so a `TrainingConfig` written for
   a GPU box degrades gracefully instead of erroring on this dev machine.
3. **`TokenBudgetBatchSampler` does its own distributed rank-splitting**,
   rather than relying on `accelerator.prepare()`'s automatic `DataLoader`
   sharding. Section 9.3 explicitly assigns "balanced work across
   distributed ranks" to the sampler itself, and a fixed-size-shard
   assumption (what `accelerator.prepare(dataloader)` expects) doesn't fit
   variable-token-budget batches — each rank can have a different number of
   records per batch by design. Consequently `PretrainingEngine` calls
   `accelerator.prepare(model, optimizer)` but *not* the `DataLoader`;
   gradient synchronization on `accelerator.backward()` still works
   correctly once the model itself is DDP-wrapped, independent of how the
   data arrived.
4. **Muon+AdamW parameter routing**: 2D weight matrices belonging to
   `PragmaTransformerBlock`'s attention/MLP projections
   (`q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/`fc2`) and `PragmaMLMHead.proj`
   route to Muon (Keller Jordan's Newton–Schulz-orthogonalized momentum
   optimizer — `src/pragma/training/muon.py`, a minimal, direct
   implementation of the public reference algorithm, not a new dependency).
   Everything else — the shared embedding table, all `LayerNorm`
   weights/biases, and every `Linear` bias — routes to AdamW. This matches
   Muon's own documented scope (it is designed for hidden 2D matrices, not
   embeddings or gain/bias parameters) and is the most direct reading of
   "hybrid Muon+AdamW" available without the paper's actual routing table.
   Muon/AdamW hyperparameters default to the reference implementation's
   published values (`lr=0.02, momentum=0.95` for Muon) until Phase 9's
   pilot run ablates them (section 13.2).
5. **`CheckpointManager` builds on `Accelerator.save_state`/`load_state`**
   for model/optimizer/scheduler weights and RNG state (that's exactly what
   those calls already checkpoint) and adds one sidecar
   `manifest.json` per checkpoint directory with everything ADR 0007 requires
   beyond that: processor bundle identity/hash, `PragmaConfig`/`MaskingConfig`/
   `TrainingConfig` dumps, sampler epoch, global token/event/record/update
   counters, data manifest reference, code revision (git SHA if available),
   dependency lock fingerprint (hash of `uv.lock`), and validation metrics.
   Loading fails loudly (raises) if the sidecar or its referenced processor
   bundle is missing or its hash doesn't match — per ADR 0007, a checkpoint
   without its exact processor is invalid, not merely incomplete.
6. **MLflow tracking defaults to a local SQLite file** (`sqlite:///mlflow.db`),
   not a tracking server. The plain `file:./mlruns` store — MLflow's older
   "no server" option — is deprecated as of MLflow 3.x and now raises unless
   `MLFLOW_ALLOW_FILE_STORE=true` is set (confirmed directly while building
   the engine, not assumed); SQLite keeps the "no server dependency" property
   without relying on that escape hatch. Only the main process logs —
   `accelerator.is_main_process` gates every `mlflow.log_*` call, avoiding N
   duplicate runs under DDP.

## Consequences

- Nothing in `pragma.training`/`pragma.modeling` ever branches on
  `torch.cuda.is_available()` directly except `resolve_mixed_precision`'s
  narrow support check — device/process-count questions always go through
  `Accelerator`.
- Multi-GPU behavior (the Phase 8 exit gate's "one- and multi-GPU runs are
  comparable") could not be executed or measured on this dev machine — the
  engine is built to Accelerate's standard contract specifically so that
  claim is *structurally* true rather than verified empirically here; it
  needs re-verification the first time real multi-GPU hardware is available
  (tracked in `plans/progress.md`'s "Open items").
- Changing the Muon/AdamW routing rule later is a training-recipe change,
  not a refactor — it must be versioned (new ADR or an explicit amendment
  here), since it changes what a resumed checkpoint's optimizer state means.
