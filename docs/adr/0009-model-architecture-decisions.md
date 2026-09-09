# ADR 0009: Padded reference model architecture decisions

**Status:** Accepted (Phase 5)

## Context

Section 18 ("Architecture") lists several open items the paper does not
specify: weight initialization, RoPE base/frequency, `CalendarEncoder`'s
hidden dimension, exact embedding/output-projection weight sharing, and
attention-kernel numerical settings. Phase 5 (the padded reference model)
cannot be built without resolving these.

## Decisions

1. **`[USR]`/`[EVT]` insertion is a generic packed-sequence operation**, not
   ID-array surgery. `pragma.modeling.packing` provides `prepend_vector`,
   `group_starts`, and `unprepend_vector` operating on already-embedded
   `[N, H]` tensors plus `cu_seqlens`. The same three functions are reused
   for `[USR]` in the Profile State Encoder, `[EVT]` in the Event Encoder,
   and the profile-`[USR]`-seeded sequence entering the History Encoder —
   one packing primitive, three call sites, per CLAUDE.md's preference for
   reuse over three bespoke implementations.

2. **Special-token embeddings reuse the key+value+position formula.** A
   `[USR]`/`[EVT]` slot is embedded as `E(special_id) + E(special_id) +
   P(0)` — the same `x_i = E(k_i) + E(v_i) + P_within-field(i)` formula
   section 6.4 defines for real fields, with `k_i = v_i = special_id`. This
   keeps one embedding code path for every token instead of a special-cased
   one, and is expressively equivalent to a dedicated single embedding
   (gradients still update exactly one table row, `embedding_table[special_id]`,
   so the model learns a unique, fixed vector per special token either way).

3. **RoPE coordinate for `[USR]`/`[EVT]` slots is `0.0`.** In the Profile
   State Encoder, `[USR]` gets `0.0` alongside static (non-milestone)
   attributes — it is not itself a milestone. In the History Encoder, the
   `[USR]`-seed position also gets `0.0`, the same value the most-recent
   event's `time_to_latest` converges to, i.e. "now."

4. **`ContinuousRoPE` uses the standard GPT-NeoX/RoFormer rotate-half
   formulation with base 10000.0** — the paper does not specify a base;
   10000 is the near-universal default and there is no evidence PRAGMA's
   continuous-time variant needs a different one. `head_dim` must be even.

5. **`CalendarEncoder`'s two-layer MLP is `6 -> hidden_size -> hidden_size`**
   (input is the 6 sine/cosine features from `pragma.processing.temporal`),
   GELU activation between layers, no normalization or dropout inside it —
   its output is added directly to the Event Encoder's `[EVT]` summary,
   which already passes through the transformer blocks' own LayerNorm/dropout.

6. **MLM logits are computed only over the value-vocabulary slice**, matching
   section 7.2's "expose the value-vocabulary slice for tied MLM output
   projection." `PragmaMLMHead` slices the shared embedding table at
   `config.value_vocab_start` (the first numeric-bucket token ID, i.e.
   `KeyVocabulary.next_id` / `NumericBucketizer.value_base_id` from Phase 2)
   through the end of the vocabulary, and shifts non-ignored labels by that
   offset before computing cross-entropy. This keeps the softmax from
   wasting probability mass on special/key IDs that a value token can never
   equal, per section 7.5's "logits should be materialized only for masked
   positions" efficiency goal.

7. **Weight initialization**: `nn.Linear`/`nn.Embedding` weights ~
   `Normal(0, 0.02)` (truncated at ±2 std), biases zero, `LayerNorm` weight
   1 / bias 0 — the standard BERT/GPT-2 recipe. No evidence in the paper
   favors a different scheme, and this one is well-understood at PRAGMA-S's
   scale (~10M parameters).

8. **`PaddedAttentionBackend` pads by splitting the packed sequence at
   `cu_seqlens`, running `torch.nn.functional.scaled_dot_product_attention`
   with a boolean key-padding mask (`True` = attend), and discarding padded
   query-row outputs on unpad** — those rows are never selected by the
   caller, so their (numerically valid but unused) content does not need to
   be masked out of the output itself.

## Consequences

- Every encoder (`ProfileStateEncoder`, `EventEncoder`, `HistoryEncoder`)
  depends only on `pragma.modeling.packing`'s three functions and the
  `AttentionBackend` interface (ADR 0006) — never on padding details
  directly, so Phase 7's `VarLenAttentionBackend` swap requires no encoder
  changes.
- `PragmaConfig.from_processor(processor)` is the only place `value_vocab_start`
  and the `[USR]`/`[EVT]` special-token IDs get read from a fitted
  `PragmaProcessor` bundle, keeping the model config's vocabulary knowledge
  in one place.
