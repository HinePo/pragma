# Architecture decision records

Phase 0 exit gate requires every paper ambiguity that affects implementation
to be either decided or represented as configuration (see
[plans/PRAGMA-Implementation-Plan.md](../../plans/PRAGMA-Implementation-Plan.md),
section 17). These records capture those decisions.

| ADR | Topic |
|---|---|
| [0001](0001-data-schema.md) | Canonical data schema and raw record shape |
| [0002](0002-evaluation-points.md) | Evaluation points and point-in-time correctness |
| [0003](0003-structured-processor.md) | Fitted structured processor vs. a Hugging Face tokenizer |
| [0004](0004-packed-batches.md) | Packed variable-length sequences as the canonical batch contract |
| [0005](0005-masking-semantics.md) | Masking overlap, corruption, and label semantics |
| [0006](0006-attention-backends.md) | Dual attention-backend interface |
| [0007](0007-checkpoint-format.md) | Checkpoint contents and validity |
| [0008](0008-token-vocabulary-and-evaluation-sampling.md) | Token ID space, milestone tokenization, and evaluation-point sampling |
| [0009](0009-model-architecture-decisions.md) | Padded reference model architecture decisions |
| [0010](0010-varlen-attention-kernel.md) | `VarLenAttentionBackend` kernel choice — PyTorch nested tensors, not FlashAttention |
| [0011](0011-distributed-training-engine-decisions.md) | Distributed pretraining engine decisions |
| [0012](0012-evaluation-harness-decisions.md) | Evaluation harness decisions (embedding probes and baselines) |
| [0013](0013-lora-downstream-adaptation-decisions.md) | LoRA downstream adaptation decisions (LoRA vs. QLoRA, target modules, task head, hyperparameters) |
| [0014](0014-zero-event-customer-exclusion.md) | Zero-event customers are excluded from pretraining and inference |

Each ADR marks its implementation status. "Interface frozen" means the
contract is decided even though the code lands in a later phase — later
phases must conform to it rather than re-litigate it.
