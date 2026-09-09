"""`PaddedAttentionBackend` vs. `VarLenAttentionBackend` parity (implementation plan,
section 9.2, 16.4; ADR 0006, ADR 0010) — the standing gate ADR 0006 requires before the
optimized backend can ever be used for training. Runs on randomized small batches, at
the backend level and through the full model, in both directions (forward and gradients)."""

import torch

from pragma.attention import PaddedAttentionBackend, VarLenAttentionBackend
from pragma.config import MaskingConfig, ProcessorConfig
from pragma.data.batch import PragmaCollator
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.masking import MaskingPlanner
from pragma.modeling import PragmaConfig, PragmaForMaskedModeling
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry


def _random_qkv(
    lengths: list[int], num_heads: int, head_dim: int, *, dtype: torch.dtype, seed: int
):
    generator = torch.Generator().manual_seed(seed)
    n = sum(lengths)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.long)
    shape = (n, num_heads, head_dim)
    q = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, generator=generator, dtype=dtype, requires_grad=True)
    return q, k, v, cu


def test_forward_parity_uniform_lengths() -> None:
    q, k, v, cu = _random_qkv([4, 4, 4, 4], num_heads=3, head_dim=8, dtype=torch.float64, seed=0)
    padded = PaddedAttentionBackend().forward(q, k, v, cu)
    varlen = VarLenAttentionBackend().forward(q, k, v, cu)
    assert torch.allclose(padded, varlen, atol=1e-10)


def test_forward_parity_highly_skewed_lengths() -> None:
    q, k, v, cu = _random_qkv(
        [1, 50, 3, 1, 27], num_heads=2, head_dim=16, dtype=torch.float64, seed=1
    )
    padded = PaddedAttentionBackend().forward(q, k, v, cu)
    varlen = VarLenAttentionBackend().forward(q, k, v, cu)
    assert torch.allclose(padded, varlen, atol=1e-10)


def test_forward_parity_includes_a_zero_length_group() -> None:
    q, k, v, cu = _random_qkv([3, 0, 5, 1], num_heads=2, head_dim=8, dtype=torch.float64, seed=2)
    padded = PaddedAttentionBackend().forward(q, k, v, cu)
    varlen = VarLenAttentionBackend().forward(q, k, v, cu)
    assert torch.allclose(padded, varlen, atol=1e-10)


def test_forward_parity_single_group() -> None:
    q, k, v, cu = _random_qkv([7], num_heads=2, head_dim=8, dtype=torch.float64, seed=3)
    padded = PaddedAttentionBackend().forward(q, k, v, cu)
    varlen = VarLenAttentionBackend().forward(q, k, v, cu)
    assert torch.allclose(padded, varlen, atol=1e-10)


def test_backward_parity_matches_within_tolerance() -> None:
    lengths = [2, 9, 4, 1, 13]
    q1, k1, v1, cu = _random_qkv(lengths, num_heads=3, head_dim=8, dtype=torch.float32, seed=4)
    q2, k2, v2 = (
        q1.detach().clone().requires_grad_(),
        k1.detach().clone().requires_grad_(),
        v1.detach().clone().requires_grad_(),
    )

    out_padded = PaddedAttentionBackend().forward(q1, k1, v1, cu)
    out_padded.sum().backward()

    out_varlen = VarLenAttentionBackend().forward(q2, k2, v2, cu)
    out_varlen.sum().backward()

    assert torch.allclose(out_padded, out_varlen, atol=1e-5)
    assert torch.allclose(q1.grad, q2.grad, atol=1e-4)
    assert torch.allclose(k1.grad, k2.grad, atol=1e-4)
    assert torch.allclose(v1.grad, v2.grad, atol=1e-4)


def _tokenized_records():
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=40,
            seed=99,
            max_events_per_entity=15,
            n_zero_event_entities=4,
            n_single_event_entities=4,
            n_long_history_entities=1,
            long_history_event_count=25,
            n_same_timestamp_entities=3,
        )
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=6, bpe_vocab_size=100))
    processor.fit(records)
    tokenized = [processor.transform(r) for r in records if r.events_before_evaluation]
    return tokenized, processor


def test_full_model_forward_parity() -> None:
    tokenized, processor = _tokenized_records()
    planner = MaskingPlanner.from_registry(
        SchemaRegistry.default(), processor.key_vocab, MaskingConfig(token_mask_prob=0.3)
    )
    batch = PragmaCollator(masking_planner=planner)(tokenized[:20])

    config = PragmaConfig.from_processor(
        processor,
        hidden_size=24,
        num_heads=2,
        intermediate_size=48,
        event_layers=2,
        history_layers=1,
    )
    torch.manual_seed(0)
    model = PragmaForMaskedModeling(config)
    model.eval()

    with torch.no_grad():
        out_padded = model(batch, attention_backend=PaddedAttentionBackend())
        out_varlen = model(batch, attention_backend=VarLenAttentionBackend())

    assert torch.allclose(out_padded.record_embeddings, out_varlen.record_embeddings, atol=1e-5)
    assert torch.allclose(out_padded.event_embeddings, out_varlen.event_embeddings, atol=1e-5)
    assert torch.allclose(out_padded.mlm_logits, out_varlen.mlm_logits, atol=1e-4)
    assert torch.allclose(out_padded.loss, out_varlen.loss, atol=1e-5)


def test_full_model_backward_parity() -> None:
    import copy

    tokenized, processor = _tokenized_records()
    planner = MaskingPlanner.from_registry(
        SchemaRegistry.default(), processor.key_vocab, MaskingConfig(token_mask_prob=0.3)
    )
    batch = PragmaCollator(masking_planner=planner)(tokenized[:20])

    config = PragmaConfig.from_processor(
        processor,
        hidden_size=24,
        num_heads=2,
        intermediate_size=48,
        event_layers=2,
        history_layers=1,
    )
    torch.manual_seed(0)
    model_padded = PragmaForMaskedModeling(config)
    model_varlen = copy.deepcopy(model_padded)

    model_padded(batch, attention_backend=PaddedAttentionBackend()).loss.backward()
    model_varlen(batch, attention_backend=VarLenAttentionBackend()).loss.backward()

    for (name, p_padded), (_, p_varlen) in zip(
        model_padded.named_parameters(), model_varlen.named_parameters(), strict=True
    ):
        assert p_padded.grad is not None and p_varlen.grad is not None, f"{name} missing a gradient"
        assert torch.allclose(p_padded.grad, p_varlen.grad, atol=5e-3, rtol=5e-3), (
            f"gradient mismatch for {name}"
        )


def test_varlen_backend_never_crosses_group_boundaries() -> None:
    """Same isolation property Phase 5 checked for the padded backend, now for varlen:
    an event's contextual embedding must not depend on unrelated batch-mates."""
    tokenized, processor = _tokenized_records()
    target = next(r for r in tokenized if len(r.events) >= 2)
    others = [r for r in tokenized if r.entity_id != target.entity_id and r.events][:5]

    config = PragmaConfig.from_processor(
        processor,
        hidden_size=24,
        num_heads=2,
        intermediate_size=48,
        event_layers=2,
        history_layers=1,
    )
    torch.manual_seed(0)
    model = PragmaForMaskedModeling(config)
    model.eval()

    solo_batch = PragmaCollator()([target])
    group_batch = PragmaCollator()(others + [target])

    with torch.no_grad():
        _, solo_event_emb, _ = model.pragma(solo_batch, attention_backend=VarLenAttentionBackend())
        _, group_event_emb, _ = model.pragma(
            group_batch, attention_backend=VarLenAttentionBackend()
        )

    n_target_events = len(target.events)
    group_target_events = group_event_emb[-n_target_events:]
    assert torch.allclose(solo_event_emb, group_target_events, atol=1e-5)
