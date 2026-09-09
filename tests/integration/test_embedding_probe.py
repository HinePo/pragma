"""`extract_record_embeddings` / `run_linear_probe` (Phase 9's minimal slice of
section 14.2's transfer-evaluation protocol) - checks the mechanics (shapes,
alignment, split handling) work correctly. Whether a *trained* checkpoint's
embeddings carry real signal is a pilot-run claim, made in
`010_pilot_pretraining.ipynb`, not asserted here on an untrained model."""

from __future__ import annotations

from pathlib import Path

import torch

from pragma.config import ProcessorConfig
from pragma.data.dataset import TokenizedRecordDataset
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.evaluation import extract_record_embeddings, run_linear_probe
from pragma.modeling import PragmaConfig
from pragma.modeling.model import PragmaModel
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry


def _dataset_labels_and_processor(
    tmp_dir: Path,
) -> tuple[TokenizedRecordDataset, dict[str, bool], PragmaProcessor]:
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=40,
            seed=11,
            max_events_per_entity=12,
            n_zero_event_entities=2,
            n_single_event_entities=2,
            n_long_history_entities=0,
            n_same_timestamp_entities=1,
        )
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=6, bpe_vocab_size=80))
    processor.fit(records)
    tokenized = [processor.transform(r) for r in records if r.events_before_evaluation]

    labels = dict(
        zip(profile_df["entity_id"], profile_df["_downstream_is_high_value"], strict=True)
    )
    return TokenizedRecordDataset(tokenized), labels, processor


def test_extract_record_embeddings_shape_and_alignment(tmp_path: Path) -> None:
    dataset, _, processor = _dataset_labels_and_processor(tmp_path)
    config = PragmaConfig.from_processor(
        processor,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        event_layers=1,
        history_layers=1,
    )
    model = PragmaModel(config)

    entity_ids, embeddings = extract_record_embeddings(model, dataset, batch_size=8)
    assert len(entity_ids) == len(dataset)
    assert embeddings.shape == (len(dataset), config.hidden_size)
    assert torch.isfinite(embeddings).all()


def test_run_linear_probe_reports_finite_auc_on_a_real_split(tmp_path: Path) -> None:
    dataset, labels, processor = _dataset_labels_and_processor(tmp_path)
    config = PragmaConfig.from_processor(
        processor,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        event_layers=1,
        history_layers=1,
    )
    model = PragmaModel(config)

    entity_ids, embeddings = extract_record_embeddings(model, dataset, batch_size=8)
    val_entity_ids = set(entity_ids[: max(1, len(entity_ids) // 4)])

    result = run_linear_probe(entity_ids, embeddings, labels, val_entity_ids=val_entity_ids, seed=0)
    assert result.n_train + result.n_val == len(entity_ids)
    if result.n_train and result.n_val:
        assert result.auc == result.auc or (result.n_val == 0)  # not NaN unless degenerate


def test_run_linear_probe_reports_nan_on_a_single_class_split() -> None:
    entity_ids = ["a", "b", "c", "d"]
    embeddings = torch.randn(4, 8)
    labels = {"a": True, "b": True, "c": True, "d": True}  # every label identical
    result = run_linear_probe(entity_ids, embeddings, labels, val_entity_ids={"c", "d"})
    assert result.auc != result.auc  # NaN
    assert result.n_train == 2
    assert result.n_val == 2
