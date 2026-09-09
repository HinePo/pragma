# CLAUDE.md

Guidance for Claude Code (and any other agent or contributor) working in this
repository.

## What this project is

An implementation of the PRAGMA financial foundation model, following
[plans/PRAGMA-Implementation-Plan.md](plans/PRAGMA-Implementation-Plan.md).
That plan is the source of truth for scope, architecture, and the phased
development order (section 17) — read it before making architectural
changes. Project-specific decisions that resolve ambiguities the plan leaves
open live in [docs/adr/](docs/adr/README.md) as numbered ADRs; check there
before re-deciding something (data schema, evaluation points, processor
design, batching, masking, attention backends, checkpoints, token vocabulary
and evaluation-point sampling are already decided as of ADR 0001-0008).

Current phase progress is tracked in
[plans/progress.md](plans/progress.md), not duplicated here — see the
dedicated section below.

## Tracking progress: `plans/progress.md`

This is a long-running, multi-session project spanning 13 phases (section 17
of the implementation plan). `plans/progress.md` is the single file that
answers "where did we stop" and "what's next":

- It mirrors every phase's **Build** checklist and **Exit gate** from the
  plan, split into two sections: **Done and validated** and **To do**.
- Each completed item is checked off with a pointer to the file(s)/ADR(s)
  that implement it and a one-line note on how the exit gate was verified
  (which tests, which notebook) — not just a checkmark, so the claim is
  auditable later without re-deriving it.
- A one-line "Current status" banner at the top names the last completed
  phase and the next one, so a session can start with just "continue from
  @plans/progress.md" and know exactly where to pick up.

**Rule: update `plans/progress.md` immediately after finishing a phase** (or
a substantial chunk of one) — move items from "To do" to "Done and
validated" with their evidence, update the "Current status" banner, and add
any new open items or ADRs that phase produced. Treat this the same as
updating a companion notebook: part of finishing the work, not a separate
follow-up task. If a phase's scope changes from what the plan originally
described (e.g. an item turns out to belong to a different phase), note that
explicitly in `progress.md` rather than silently reshuffling — see Phase 3's
note on the padded collator for the pattern to follow.

## Three-layer architecture: library, scripts, notebooks

This project targets an eventual AWS deployment (likely SageMaker Processing
/ Training jobs, or Batch). That target — plus wanting both a maintainable
codebase and an easy way to explore data and results interactively — decides
the structure:

1. **`src/pragma/` (library) — where all logic lives.** Every non-trivial
   piece of behavior (data generation, record building, processor fitting,
   model code, training loops, evaluation) is a tested, importable function
   or class here. This is the only layer with real business logic.
2. **`scripts/` (thin CLI entry points) — what actually runs in prod.**
   Each script wires config → library calls → disk I/O and nothing else. No
   business logic here. These are what will eventually become SageMaker
   job entry points; keeping them thin means the AWS migration is a wiring
   change, not a rewrite.
3. **`notebooks/` (narrative + exploration) — never a source of truth.**
   Notebooks import from `src/pragma`/`scripts` to demonstrate, explore, and
   sanity-check; they must never contain logic that isn't already in the
   library. If a notebook cell has more than a few lines of real logic that
   isn't a plot or a print, that logic belongs in `src/pragma` instead and
   the notebook should just call it.

**Rule: whenever you add or materially change a module or script, add or
update a companion notebook** that imports it and exercises it end-to-end —
generate/load a small amount of data, run the new code, inspect the output,
sanity-check edge cases. This is the fast feedback loop for this project;
don't skip it just because unit tests pass. Notebooks matter here as much as
scripts.

### Notebook naming and location

All notebooks live in `notebooks/`, numbered with a zero-padded three-digit
prefix in rough phase/dependency order, e.g.:

```
000_synthetic_data_generation.ipynb
001_point_in_time_records.ipynb
002_fit_processor.ipynb
003_tokenize_shards.ipynb
004_reference_batching.ipynb
005_masking_inspection.ipynb
006_model_architecture.ipynb
007_debug_training.ipynb
008_varlen_attention.ipynb
009_distributed_pretraining.ipynb
```

Reuse the next free number for a new step in the pipeline; don't renumber
existing notebooks to make room unless the dependency order genuinely
changed. Each notebook should open with a short markdown cell stating what
it demonstrates and which plan section / ADR it corresponds to, matching the
style already used in `000_synthetic_data_generation.ipynb`.

## Code documentation standard (overrides the default terse style)

This is a research/reproduction project where the *why* behind a design
choice is often more valuable than the code itself — most of the hard
decisions here resolve ambiguity in the source paper or the implementation
plan. Because of that, **write more documentation here than you would by
default**:

- Every module gets a module-level docstring stating its responsibility,
  which section of `plans/PRAGMA-Implementation-Plan.md` it implements, and
  which ADR (if any) it follows.
- Non-trivial classes and functions get a short docstring — one line is
  fine, but say *what it's for*, not just what it's named.
- Add a comment whenever a piece of code encodes a non-obvious decision:
  why this bucketing strategy, why this ID range layout, why this fallback
  when the paper doesn't specify one. Point at the ADR or plan section
  instead of re-explaining the whole rationale inline when one exists.
- Still don't narrate the obvious — a comment restating what a well-named
  line already says is noise. The bar is "would the next person wonder
  why," not "does this line do something."

## Environment and tooling

- Package/env management: `uv` (`uv sync`, `uv run <cmd>`). Python 3.12.
- Lint: `uv run ruff check src scripts tests`. Type-check: `uv run mypy`.
  Test: `uv run pytest`. All three must pass before considering a change
  done; `mypy` runs with `disallow_untyped_defs = true`, so new functions
  need type hints.
- Dev machine is Windows; the Bash tool here runs Git Bash, PowerShell is
  also available. `uv run` works from either.
- Dev machine's GPU (GTX 1650 Ti, 4GB) has a driver too old for current
  PyTorch's CUDA wheels (driver reports max CUDA 11.0; modern wheels need
  12.x) — `torch` here is the CPU build, and all training runs (Phase 6+)
  are CPU-only until the driver is updated. Keep debug/training batch sizes
  small regardless of device; don't assume GPU is available in scripts or
  notebooks without checking `torch.cuda.is_available()`.
- Generated data (`data/raw/*.parquet`, `data/processor/`, `data/shards/`)
  is gitignored and reproducible from `scripts/generate_synthetic_data.py`,
  `scripts/fit_processor.py`, and `scripts/tokenize_shards.py` — never hand
  edit it, regenerate it.

## Testing conventions

- `tests/unit/` — one test module per library module, fast, no I/O beyond
  tmp dirs.
- `tests/integration/`, `tests/parity/`, `tests/distributed/` — reserved for
  later phases (packed/padded attention parity, multi-GPU, end-to-end
  pipeline runs) per the implementation plan's phase structure; keep unit
  tests fast and put anything slower or cross-component there instead.
- Point-in-time correctness and leakage checks (train-only fitting, no
  future data visible, deterministic tie-breaking) are correctness
  properties, not nice-to-haves — always test them explicitly when touching
  `pragma.data.records` or `pragma.processing`.

## When you hit a design ambiguity

If the implementation plan doesn't specify something and the choice isn't
purely mechanical, write it down as a new ADR in `docs/adr/` (see existing
ADRs for the format: Context / Decision / Consequences) rather than making a
silent, undocumented choice. Update `docs/adr/README.md`'s index table too.
