# ADR 0003: Fitted structured processor over a Hugging Face tokenizer

**Status:** Accepted (interface frozen; implementation is Phase 2)

## Context

Most PRAGMA inputs are semantic keys, categorical values, numerical buckets,
and timestamps rather than natural-language text. A standard NLP tokenizer
cannot represent this without losing the key/value structure the model
depends on (section 6).

## Decision

- Build a fitted `PragmaProcessor` as the single conversion path from an
  `EvaluationRecord` to token IDs. It owns a `KeyVocabulary`, a
  `NumericBucketizer` (per-key percentile boundaries + dedicated zero
  bucket), a `CategoricalEncoder`, and a `TextBPEEncoder`.
- Hugging Face `tokenizers` is used only for the BPE sub-component, applied
  exclusively to fields the `SchemaRegistry` marks as approved text fields
  (`FieldType.TEXT`). It is not used as the top-level tokenizer.
- Every value token replicates its key ID (section 6.4): a 3-token BPE value
  produces 3 repeated key-ID entries and within-field positions `[0, 1, 2]`.
- All vocabulary/bucket/BPE fitting happens only on the training partition
  (ADR 0002's point-in-time guarantees make this partition well-defined).
- The processor bundle (schema version, vocabularies, bucket boundaries, BPE
  model, special-token IDs, truncation policy, data fingerprint) is one
  versioned, saved/loadable artifact. A checkpoint without its exact
  processor bundle is invalid (section 6.5).

## Consequences

- The `SchemaRegistry` from ADR 0001 is a required input to the processor:
  field type and null policy decide which fitted artifact (bucketizer,
  categorical encoder, or BPE) a key routes through.
- `AutoModelForMaskedLM`/AutoTokenizer integration is explicitly deferred
  until the processor and model artifact formats are stable, per section 3.
