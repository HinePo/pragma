"""Phase 5 architecture tests (implementation plan, section 16.4, subset relevant to the
padded reference model) and the Phase 5 exit gate (~10M params, fp32 tensor-shape/
isolation/mapping/temporal/gradient checks)."""

import torch

from pragma.config import ProcessorConfig
from pragma.data.batch import PragmaCollator
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.modeling import PragmaConfig, PragmaForMaskedModeling
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry


def _tokenized_records(**overrides: object):
    defaults = dict(
        n_entities=40,
        seed=777,
        max_events_per_entity=15,
        n_zero_event_entities=4,
        n_single_event_entities=4,
        n_long_history_entities=1,
        long_history_event_count=20,
        n_same_timestamp_entities=3,
    )
    defaults.update(overrides)
    events_df, profile_df, _ = generate_synthetic_corpus(SyntheticDataConfig(**defaults))
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=6, bpe_vocab_size=100))
    processor.fit(records)
    return [processor.transform(r) for r in records], processor


def _tiny_model(processor) -> PragmaForMaskedModeling:
    torch.manual_seed(0)
    config = PragmaConfig.from_processor(
        processor,
        hidden_size=24,
        num_heads=2,
        intermediate_size=48,
        profile_layers=1,
        event_layers=2,
        history_layers=1,
    )
    model = PragmaForMaskedModeling(config)
    model.eval()  # disable dropout so isolation checks are deterministic
    return model


def test_forward_backward_produce_finite_shapes_and_gradients() -> None:
    from pragma.config import MaskingConfig
    from pragma.masking import MaskingPlanner

    tokenized, processor = _tokenized_records()
    records = [r for r in tokenized if r.events][:16]
    planner = MaskingPlanner.from_registry(
        SchemaRegistry.default(), processor.key_vocab, MaskingConfig(token_mask_prob=1.0)
    )
    batch = PragmaCollator(masking_planner=planner)(records)

    config = PragmaConfig.from_processor(
        processor, hidden_size=24, num_heads=2, intermediate_size=48, event_layers=2
    )
    model = PragmaForMaskedModeling(config)
    out = model(batch)

    assert out.record_embeddings.shape == (batch.n_records, config.hidden_size)
    assert out.event_embeddings.shape == (batch.n_events, config.hidden_size)
    assert torch.isfinite(out.record_embeddings).all()
    assert torch.isfinite(out.event_embeddings).all()

    assert out.loss is not None
    out.loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"


def test_parameter_count_is_approximately_10m_at_paper_scale_vocab() -> None:
    config = PragmaConfig(
        vocab_size=28060,
        value_vocab_start=64,
        hidden_size=192,
        num_heads=3,
        intermediate_size=768,
        profile_layers=1,
        event_layers=5,
        history_layers=2,
    )
    model = PragmaForMaskedModeling(config)
    n_params = sum(p.numel() for p in model.parameters())
    assert 8_000_000 <= n_params <= 12_000_000


def test_event_encoder_output_is_invariant_to_unrelated_events_in_the_batch() -> None:
    """An event's contextual embedding must not change depending on which other
    (unrelated) events happen to share its packed batch — ADR 0004's isolation
    requirement, tested at the model level."""
    tokenized, processor = _tokenized_records()
    target = next(r for r in tokenized if len(r.events) >= 2)
    others = [r for r in tokenized if r.entity_id != target.entity_id and r.events][:5]

    model = _tiny_model(processor)

    solo_batch = PragmaCollator()([target])
    group_batch = PragmaCollator()(others + [target])

    with torch.no_grad():
        _, solo_event_emb, _ = model.pragma(solo_batch)
        _, group_event_emb, _ = model.pragma(group_batch)

    n_target_events = len(target.events)
    group_target_events = group_event_emb[-n_target_events:]
    assert torch.allclose(solo_event_emb, group_target_events, atol=1e-5)


def test_history_encoder_output_is_invariant_to_unrelated_records_in_the_batch() -> None:
    tokenized, processor = _tokenized_records()
    target = next(r for r in tokenized if len(r.events) >= 2)
    others = [r for r in tokenized if r.entity_id != target.entity_id][:5]

    model = _tiny_model(processor)

    solo_batch = PragmaCollator()([target])
    group_batch = PragmaCollator()(others + [target])

    with torch.no_grad():
        _, _, solo_record_emb = model.pragma(solo_batch)
        _, _, group_record_emb = model.pragma(group_batch)

    assert torch.allclose(solo_record_emb[0], group_record_emb[-1], atol=1e-5)


def test_profile_and_event_summary_positions_are_correctly_mapped() -> None:
    """`group_starts`/`unprepend_vector` must place the `[USR]`/`[EVT]` summary at
    exactly the right position relative to that record's/event's own tokens.

    Checked via gradient connectivity rather than an output-magnitude
    comparison: at random initialization, pre-norm attention is nearly
    uniform (see `test_time_coordinates_...`'s docstring), so a real mapping
    bug and "the effect is just numerically tiny before training" can look
    identical under a magnitude threshold. Gradient flow does not have that
    problem — record 0's embedding must depend only on record 0's own
    profile tokens, never on record 1's.
    """
    tokenized, processor = _tokenized_records()
    records = [r for r in tokenized if r.events][:6]
    model = _tiny_model(processor)

    batch = PragmaCollator()(records)
    batch.profile_time_coords.requires_grad_(True)
    _, _, record_emb = model.pragma(batch)
    record_emb[0].sum().backward()

    grad = batch.profile_time_coords.grad
    assert grad is not None
    own_tokens = batch.profile_token_to_record == 0
    other_tokens = batch.profile_token_to_record != 0
    assert bool((grad[own_tokens] != 0).any()), "record 0's embedding must depend on its own tokens"
    assert bool((grad[other_tokens] == 0).all()), (
        "record 0's embedding must not depend on other records"
    )


def test_masked_logits_use_the_correct_local_event_and_user_states() -> None:
    """Manually recompute one masked position's MLM input vector from the backbone's
    raw outputs and confirm it matches what `PragmaMLMHead` actually used."""
    tokenized, processor = _tokenized_records()
    records = [r for r in tokenized if len(r.events) >= 2][:8]
    model = _tiny_model(processor)

    from pragma.config import MaskingConfig
    from pragma.masking import MaskingPlanner

    planner = MaskingPlanner.from_registry(
        SchemaRegistry.default(), processor.key_vocab, MaskingConfig(token_mask_prob=1.0)
    )
    batch = PragmaCollator(masking_planner=planner)(records)
    assert bool((batch.event_mlm_labels != -100).any())

    with torch.no_grad():
        local_token_states, event_emb, record_emb = model.pragma(batch)

        from pragma.modeling.packing import group_index_per_token

        token_to_event = group_index_per_token(batch.event_cu_seqlens, batch.n_event_tokens)
        masked_positions = (batch.event_mlm_labels != -100).nonzero(as_tuple=True)[0]
        i = int(masked_positions[0])
        event_idx = int(token_to_event[i])
        record_idx = int(batch.event_to_record[event_idx])

        expected = torch.cat([local_token_states[i], event_emb[event_idx], record_emb[record_idx]])
        expected_logits = (
            model.mlm_head.proj(expected)
            @ model.pragma.embedding.embedding.weight[model.mlm_head.config.value_vocab_start :].T
        )

        _, logits = model.mlm_head(
            local_token_states,
            event_emb,
            record_emb,
            batch.event_cu_seqlens,
            batch.event_to_record,
            batch.event_mlm_labels,
        )
        row = int(masked_positions.tolist().index(i))
        assert torch.allclose(expected_logits, logits[row], atol=1e-5)


def test_time_coordinates_change_output_and_zero_time_is_stable() -> None:
    """At random initialization, pre-norm attention is nearly uniform (a well-known
    property of untrained Pre-LN transformers), so a magnitude-based "output changes
    a lot" comparison is flaky by construction here — a real RoPE effect can be
    numerically tiny before training without being broken. Test the structural
    property instead: gradients must flow from the output back through
    `event_time_to_latest`, proving the coordinate is actually wired into attention,
    not silently ignored. Separately confirm the all-zero-time case is numerically
    stable (finite, no NaN/Inf)."""
    tokenized, processor = _tokenized_records()
    record = next(r for r in tokenized if len(r.events) >= 1)
    model = _tiny_model(processor)

    batch = PragmaCollator()([record])
    batch.event_time_to_latest.requires_grad_(True)
    _, event_emb, _ = model.pragma(batch)
    event_emb.sum().backward()

    assert batch.event_time_to_latest.grad is not None
    assert bool((batch.event_time_to_latest.grad != 0).any())

    # Re-running with identical (non-perturbed) inputs must reproduce identical output.
    batch_b = PragmaCollator()([record])
    with torch.no_grad():
        _, event_emb_b, _ = model.pragma(batch_b)
    assert torch.allclose(event_emb.detach(), event_emb_b, atol=1e-6)

    # Zero-time (all events "now") must be numerically stable, not NaN/Inf.
    batch_zero = PragmaCollator()([record])
    batch_zero.event_time_to_latest = torch.zeros_like(batch_zero.event_time_to_latest)
    with torch.no_grad():
        _, event_emb_zero, record_emb_zero = model.pragma(batch_zero)
    assert torch.isfinite(event_emb_zero).all()
    assert torch.isfinite(record_emb_zero).all()
