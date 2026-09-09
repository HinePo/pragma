"""`PragmaForTask` (Phase 11's binary-classification downstream model,
section 11.3, ADR 0013) - forward shape/loss correctness and gradient flow
on a real tokenized batch, independent of LoRA (`tests/unit/test_lora.py`
covers the PEFT-wrapped case)."""

from __future__ import annotations

import torch

from pragma.config import ProcessorConfig
from pragma.data.batch import PragmaCollator
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.downstream.task_model import PragmaForTask
from pragma.modeling import PragmaConfig
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry


def _batch_and_config():
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
        event_layers=1,
        history_layers=1,
    )
    tokenized = [processor.transform(r) for r in records[:8]]
    batch = PragmaCollator()(tokenized)
    return batch, config


def test_forward_without_labels_returns_finite_logits_and_no_loss() -> None:
    batch, config = _batch_and_config()
    model = PragmaForTask(config)
    out = model(batch)
    assert out.loss is None
    assert out.logits.shape == (batch.n_records,)
    assert torch.isfinite(out.logits).all()


def test_forward_with_labels_returns_finite_positive_loss() -> None:
    batch, config = _batch_and_config()
    model = PragmaForTask(config)
    labels = torch.randint(0, 2, (batch.n_records,)).float()
    out = model(batch, labels=labels)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    assert out.loss.item() > 0


def test_gradient_flows_from_loss_to_backbone_embeddings() -> None:
    batch, config = _batch_and_config()
    model = PragmaForTask(config)
    labels = torch.randint(0, 2, (batch.n_records,)).float()
    out = model(batch, labels=labels)
    out.loss.backward()
    embedding_grad = model.pragma.embedding.embedding.weight.grad
    assert embedding_grad is not None
    assert torch.any(embedding_grad != 0)
    assert model.classifier.weight.grad is not None
