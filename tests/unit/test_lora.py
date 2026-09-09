"""`build_lora_model` (Phase 11, section 11.4's `LoRAAdapterFactory`, ADR 0013) -
confirms PEFT actually freezes the backbone and only trains LoRA + the task
head on this custom (non-Hugging-Face) architecture, and that a
forward/backward pass only moves the parameters it should."""

from __future__ import annotations

import torch

from pragma.config import DownstreamConfig, ProcessorConfig
from pragma.data.batch import PragmaCollator
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.downstream.lora import build_lora_model
from pragma.downstream.task_model import PragmaForTask
from pragma.modeling import PragmaConfig
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry


def _batch_and_model():
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(n_entities=20, seed=3, max_events_per_entity=10)
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    records = [r for r in records if r.events_before_evaluation]
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=6, bpe_vocab_size=80))
    processor.fit(records)
    config = PragmaConfig.from_processor(
        processor,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        event_layers=2,
        history_layers=1,
    )
    tokenized = [processor.transform(r) for r in records[:8]]
    batch = PragmaCollator()(tokenized)
    return batch, PragmaForTask(config)


def test_only_lora_and_classifier_params_are_trainable() -> None:
    _, base_model = _batch_and_model()
    n_before = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    lora_model = build_lora_model(base_model, DownstreamConfig(lora_r=4, lora_alpha=4))
    trainable_names = [n for n, p in lora_model.named_parameters() if p.requires_grad]

    assert trainable_names  # something is trainable
    assert all(("lora_" in n) or ("classifier" in n) for n in trainable_names)
    n_trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in lora_model.parameters())
    assert 0 < n_trainable < n_total
    assert n_total >= n_before  # LoRA only adds parameters, never removes any


def test_forward_backward_only_moves_trainable_parameters() -> None:
    batch, base_model = _batch_and_model()
    lora_model = build_lora_model(base_model, DownstreamConfig(lora_r=4, lora_alpha=4))

    frozen_before = {
        n: p.detach().clone() for n, p in lora_model.named_parameters() if not p.requires_grad
    }

    labels = torch.randint(0, 2, (batch.n_records,)).float()
    out = lora_model(batch, labels=labels)
    out.loss.backward()

    for n, p in lora_model.named_parameters():
        if p.requires_grad:
            continue
        assert p.grad is None, f"frozen parameter {n} unexpectedly received a gradient"
        assert torch.equal(p, frozen_before[n])


def test_trainable_parameters_receive_gradients() -> None:
    batch, base_model = _batch_and_model()
    lora_model = build_lora_model(base_model, DownstreamConfig(lora_r=4, lora_alpha=4))

    labels = torch.randint(0, 2, (batch.n_records,)).float()
    out = lora_model(batch, labels=labels)
    out.loss.backward()

    trainable = [(n, p) for n, p in lora_model.named_parameters() if p.requires_grad]
    assert all(p.grad is not None for _, p in trainable)
