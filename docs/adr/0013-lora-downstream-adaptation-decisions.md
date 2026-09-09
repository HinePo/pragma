# ADR 0013: LoRA downstream adaptation decisions

**Status:** Accepted (Phase 11)

## Context

Section 15 requires adapting the frozen, pretrained PRAGMA-S backbone to one
supervised downstream task using LoRA, via Hugging Face PEFT rather than a
hand-rolled implementation (15.2). Section 18's "LoRA and downstream
evaluation" list leaves several concrete questions open: LoRA vs. a
quantized variant, exact target modules beyond the stated QKV/MLP families,
LoRA dropout/learning rate/batch size/weight decay/training steps, and the
task-head architecture. This phase also needs to keep the same hardware
posture every training component has had since Phase 8 (ADR 0011): never
hardcode a device, auto-detect GPU/DDP through `Accelerate`.

### What LoRA actually is, and why the paper reaches for it here

Full fine-tuning of a pretrained model updates *every* parameter for the
downstream task — for a foundation model meant to serve many downstream
tasks off one shared backbone, that means either keeping N full copies of
the model (one per task) or accepting that each new task's fine-tuning run
overwrites what came before. LoRA (Hu et al., 2021, *LoRA: Low-Rank
Adaptation of Large Language Models*) instead freezes every pretrained
weight matrix `W` and adds a *parallel*, low-rank update: for a targeted
linear layer, the effective weight becomes `W + (alpha/r) * B @ A`, where
`A` (`r x in_features`) and `B` (`out_features x r`) are the only new,
trainable parameters, `A` is initialized so the update starts at exactly
zero (the adapted model equals the frozen base until training moves it),
and `r` (rank) is a small number — 4 to 64 in practice — that controls how
much new capacity the adapter has. Because `r` is tiny relative to
`in_features`/`out_features`, the trainable parameter count is a small
fraction of the full model (confirmed directly for PRAGMA-S: rank-4 LoRA on
every QKV+MLP projection is ~35% of the backbone's own already-small
parameter count in this project's own smoke test — see the Phase 11 build
notes in `plans/progress.md`).

This is exactly the shape of adaptation section 15 asks for: `PragmaForTask`
(the frozen backbone plus a task head) gets a LoRA adapter injected into its
attention/MLP projections; the frozen base checkpoint stays completely
unmodified on disk (`build_lora_model` freezes every non-adapter parameter -
see decision 3); only the small adapter (plus the task head) is trained and
saved per task; and the same immutable base checkpoint can grow an
unbounded number of independent, cheap task-specific adapters (LoRA
"stacking" as a business model, not just a training trick — one ~10M
parameter PRAGMA-S base could in principle carry dozens of task adapters at
a fraction of the storage/serving cost of dozens of fully fine-tuned
copies). Section 15.2's own instructions read directly off this mental
model: "keep stable, explicit names for Q, K, V, output, and MLP linear
projections" (so PEFT's name-based injection can find them — already true
here since ADR 0011/0009 named `q_proj`/`k_proj`/`v_proj`/`out_proj`/`fc1`/
`fc2` explicitly for this reason), "apply LoRA to QKV and MLP projections,
matching the paper's stated target families" (the paper's own reported
practice), and "save adapters and task heads separately from the immutable
base checkpoint" (the entire point of freezing the base in the first
place).

## Decisions

1. **Plain LoRA, not QLoRA.** QLoRA (Dettmers et al., 2023) is LoRA applied
   on top of a base model quantized to 4-bit (NF4) precision specifically
   to fit fine-tuning of very large (multi-billion-parameter) LLMs into
   limited GPU memory — the quantization is a memory-capacity trade-off,
   not a LoRA variant with different adaptation behavior. Neither the
   implementation plan nor the paper's stated approach (section 15.2:
   "Use Hugging Face PEFT rather than implementing LoRA mathematics
   manually" — no mention anywhere of quantization) calls for it. PRAGMA-S
   is a ~10M parameter model (Phase 5's exit gate measured 9.10M at paper
   vocab scale) — multiple orders of magnitude below where quantization's
   memory savings would matter, and this dev machine's GPU situation
   (`CLAUDE.md`'s environment note) means fine-tuning has never once been
   memory-bound. Quantizing a model this small would only add complexity
   (a `bitsandbytes` dependency with historically poor Windows support) and
   precision loss for a memory problem that does not exist here. **If a
   future PRAGMA-M/L scale-up (Phase 12) makes the backbone large enough
   that GPU memory becomes the binding constraint during fine-tuning,
   revisit this decision — that is exactly the situation QLoRA is for.**
2. **Target modules: exactly the QKV and MLP projection names section 15.2
   states** — `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`.
   These names are shared verbatim across every `PragmaTransformerBlock` in
   the profile/event/history encoders (ADR 0011), so PEFT's suffix-based
   module matching injects a LoRA adapter into every layer of every
   encoder without listing layer indices — confirmed directly against this
   custom (non-Hugging-Face) architecture. `PragmaMLMHead.proj` (the
   pretraining-only MLM output projection) is correctly never touched:
   `PragmaForTask` does not have or use an MLM head at all.
3. **Task-head architecture: a single `Linear(hidden_size, 1)` on
   `record_embeddings` (the contextual `[USR]` state), trained via PEFT's
   `modules_to_save`** rather than a separate save path. `[USR]` already
   plays the "customer-level summary" role section 7 designs it for, and
   Phase 10's probe comparison found `probe_usr` and the richer `probe_concat`
   (`[USR]` + last `[EVT]`) scored identically on this corpus — so the
   extra complexity of concatenating a second embedding into the task head
   bought nothing on the evidence available, and the simpler head is
   preferred. Multiclass/multilabel/regression heads (section 11.3's
   fuller description of `PragmaForTask`) are a mechanical extension of the
   same pattern (a different output width and loss) whenever a task
   actually needs one — not written speculatively now.
4. **A plain fixed-batch `DataLoader`, not `TokenBudgetBatchSampler`, for
   downstream fine-tuning** — `accelerator.prepare(dataloader)` shards it
   automatically under DDP with no custom sampler needed. The dynamic
   token-budget sampler (section 9.3, ADR 0011) exists specifically for
   pretraining's much larger and highly variable per-batch token counts;
   downstream fine-tuning datasets are small enough, and this is a genuine
   simplification, not a missed reuse opportunity.
5. **`DownstreamTrainer` reaffirms ADR 0011's hardware posture exactly**:
   a plain `Accelerator(mixed_precision=resolve_mixed_precision(...))`,
   never a hardcoded device or DDP flag. Running a fine-tuning script
   directly gives single-device execution; `accelerate launch --multi_gpu
   --num_processes=N scripts/finetune_lora.py` gives DDP across N GPUs with
   zero code changes — identical to how `PretrainingEngine` behaves. On
   this dev machine, that means single-process CPU execution, same as
   every other training component in this project so far.
6. **Optimizer: plain `torch.optim.AdamW` over only the trainable (LoRA +
   task-head) parameters, not Muon.** ADR 0011's Muon/AdamW hybrid routing
   is a *pretraining* decision, specifically for the large from-scratch
   2D weight matrices being optimized from random initialization; LoRA
   fine-tuning only ever trains small, already-near-zero-initialized `A`/`B`
   matrices and a fresh linear head — plain AdamW is the standard, well
   understood choice for LoRA fine-tuning in the wider literature, and
   introducing Muon here would be solving a problem (large-matrix
   optimization from scratch) that does not exist in this setting.
   `WarmupDecayScheduler` (ADR 0011, already optimizer-agnostic — it only
   needs `optimizer.param_groups`) is reused unmodified for the LR
   schedule, since `torch.optim.AdamW` satisfies that same minimal
   contract `HybridOptimizer` does.
7. **Default hyperparameters** (`DownstreamConfig`, section 18's
   open list): `lora_r=8`, `lora_alpha=8` (section 15.2's stated starting
   point), `lora_dropout=0.05`, `learning_rate=1e-3`, `weight_decay=0.01`,
   `batch_size=8`, `n_epochs=10`, `warmup_ratio=0.05`, cosine schedule.
   These are reasonable, commonly-used LoRA fine-tuning defaults, not
   independently tuned for PRAGMA-S — recorded as configuration (this ADR
   plus `configs/downstream/`), not hardcoded, exactly as section 18
   requires, so they can be revised once a real downstream dataset (not
   the synthetic pilot) is available to tune against.
8. **A new synthetic downstream label:
   `_downstream_is_escalating_spender`** (`pragma.data.synthetic`), in
   addition to Phase 9's `_downstream_is_high_value`. Phase 10's finding
   was that `_downstream_is_high_value` (a threshold of `total_amount`) is
   trivially recoverable by any baseline that includes `total_amount` as a
   feature, making the probe-vs-baseline comparison uninteresting for that
   label (`ADR 0012`'s cross-reference, `plans/progress.md`'s Phase 10 open
   item). The new label is **order-dependent**: `True` if an entity spent
   more in the second (time-ordered) half of its history than the first —
   a property `total_amount`/`n_events`/any simple aggregate cannot recover
   without a baseline builder explicitly engineering a first-half/
   second-half split themselves (exactly the kind of extra feature-
   engineering cost section 14.3 asks whether PRAGMA's sequence-aware
   representation makes unnecessary). Still a well-defined, plausible
   binary business question ("is this customer's spending trending up") —
   not AML, per section 15.2's explicit exclusion (record-isolated
   architecture, no cross-customer network modeling).

## Consequences

- Every non-adapter, non-task-head parameter is verifiably frozen: `build_lora_model`
  never sees or touches them, and `tests/integration/test_downstream_trainer.py`
  asserts the base checkpoint's `model.safetensors` is byte-identical before
  and after a full fine-tuning run — section 15.1's immutability entry
  condition is a tested property, not an assumption.
- An adapter is a self-contained, independently reloadable artifact: PEFT's
  own `save_pretrained`/`from_pretrained` handle the LoRA weights and the
  `modules_to_save` task head together, and `load_lora_model` rebuilds the
  frozen backbone from the base checkpoint from scratch (verifying the
  processor hash again, per ADR 0007) rather than trusting any in-memory
  state — `tests/integration/test_downstream_trainer.py` confirms a
  freshly reloaded adapter reproduces the trained model's exact predictions.
  Every adapter's sidecar manifest records the exact base-checkpoint hash
  and processor hash it was trained against (section 15.2's logging
  requirement).
- If PRAGMA scales up enough that GPU memory becomes a binding constraint
  during downstream fine-tuning (unlike today), decision 1 should be
  revisited — QLoRA's quantize-the-base approach is designed for exactly
  that situation, and PEFT already supports it as a drop-in `LoraConfig`
  change (a `bitsandbytes`-backed quantized base model) without altering
  anything else in `DownstreamTrainer`.
- `_downstream_is_high_value` is kept, not removed — Phase 9/10 evidence
  and notebooks still reference it, and it remains a legitimate (if
  aggregate-trivial) label for future comparisons that specifically want an
  "easy" baseline-favoring case.
