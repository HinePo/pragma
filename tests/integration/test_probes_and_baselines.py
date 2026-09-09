"""`EmbeddingExtractor` variants, `LinearProbeRunner`, `run_baselines`, and
`build_comparison_report` (Phase 10, section 14.2/14.3/11.4). Mechanics only
(shapes, alignment, report structure) - whether a *trained* checkpoint's
embeddings beat the conventional baselines is a pilot-run claim made in
`011_embedding_probes_and_baselines.ipynb`, not asserted here on an untrained
model."""

from __future__ import annotations

from pathlib import Path

import torch

from pragma.config import ProcessorConfig
from pragma.data.dataset import TokenizedRecordDataset
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.evaluation import (
    EmbeddingExtractor,
    LinearProbeRunner,
    build_aggregated_features,
    build_comparison_report,
    run_baselines,
)
from pragma.evaluation.probe import ProbeResult
from pragma.modeling import PragmaConfig
from pragma.modeling.model import PragmaModel
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry


def _fixture(tmp_dir: Path):
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=40,
            seed=11,
            max_events_per_entity=12,
            n_zero_event_entities=3,
            n_single_event_entities=2,
            n_long_history_entities=0,
            n_same_timestamp_entities=1,
        )
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=6, bpe_vocab_size=80))
    processor.fit(records)
    kept_records = [r for r in records if r.events_before_evaluation]
    tokenized = [processor.transform(r) for r in kept_records]

    config = PragmaConfig.from_processor(
        processor,
        hidden_size=16,
        num_heads=2,
        intermediate_size=32,
        event_layers=1,
        history_layers=1,
    )
    model = PragmaModel(config)

    labels = dict(
        zip(profile_df["entity_id"], profile_df["_downstream_is_high_value"], strict=True)
    )
    dataset = TokenizedRecordDataset(tokenized)
    features_df = build_aggregated_features(kept_records)
    return model, dataset, kept_records, features_df, labels


def test_embedding_extractor_produces_aligned_usr_last_event_and_concat(tmp_path: Path) -> None:
    model, dataset, records, _, _ = _fixture(tmp_path)
    bundle = EmbeddingExtractor(model, batch_size=8).extract(dataset)

    assert len(bundle.entity_ids) == len(dataset)
    assert bundle.usr.shape == (len(dataset), model.config.hidden_size)
    assert bundle.last_event.shape == (len(dataset), model.config.hidden_size)
    assert bundle.concat.shape == (len(dataset), 2 * model.config.hidden_size)
    assert torch.equal(bundle.concat, torch.cat([bundle.usr, bundle.last_event], dim=1))
    assert torch.isfinite(bundle.usr).all()
    assert torch.isfinite(bundle.last_event).all()

    zero_event_ids = {r.entity_id for r in records if not r.events_before_evaluation}
    for entity_id, row in zip(bundle.entity_ids, bundle.last_event, strict=True):
        if entity_id in zero_event_ids:
            assert torch.all(row == 0.0)


def test_linear_probe_runner_covers_all_three_variants(tmp_path: Path) -> None:
    model, dataset, _, _, labels = _fixture(tmp_path)
    bundle = EmbeddingExtractor(model, batch_size=8).extract(dataset)
    val_entity_ids = set(bundle.entity_ids[: max(1, len(bundle.entity_ids) // 4)])

    results = LinearProbeRunner(seed=0).run(bundle, labels, val_entity_ids=val_entity_ids)
    names = {r.name for r in results}
    assert names == {"probe_usr", "probe_last_event", "probe_concat"}
    for r in results:
        assert r.n_train + r.n_val == len(bundle.entity_ids)


def test_run_baselines_returns_logreg_and_gbdt_results(tmp_path: Path) -> None:
    _, _, _, features_df, labels = _fixture(tmp_path)
    entity_ids = list(features_df.index)
    val_entity_ids = set(entity_ids[: max(1, len(entity_ids) // 4)])

    results = run_baselines(features_df, labels, val_entity_ids=val_entity_ids, seed=0)
    names = {r.name for r in results}
    assert names == {"baseline_logreg", "baseline_gbdt"}
    for r in results:
        assert r.n_train + r.n_val == len(entity_ids)
        if r.n_train and r.n_val:
            assert r.auc == r.auc or r.n_val == 0  # not NaN unless degenerate


def test_build_comparison_report_combines_probes_and_baselines(tmp_path: Path) -> None:
    model, dataset, _, features_df, labels = _fixture(tmp_path)
    bundle = EmbeddingExtractor(model, batch_size=8).extract(dataset)
    val_entity_ids = set(bundle.entity_ids[: max(1, len(bundle.entity_ids) // 4)])

    probe_results = LinearProbeRunner(seed=0).run(bundle, labels, val_entity_ids=val_entity_ids)
    baseline_results = run_baselines(features_df, labels, val_entity_ids=val_entity_ids, seed=0)

    report = build_comparison_report(probe_results + baseline_results, checkpoint_name="untrained")
    assert set(report["name"]) == {
        "probe_usr",
        "probe_last_event",
        "probe_concat",
        "baseline_logreg",
        "baseline_gbdt",
    }
    assert set(report["kind"]) == {"probe", "baseline"}
    assert (report.loc[report["kind"] == "probe", "checkpoint"] == "untrained").all()
    assert (report.loc[report["kind"] == "baseline", "checkpoint"] == "n/a (baseline)").all()
    # Sorted descending by AUC (NaN last) - non-NaN rows must be non-increasing.
    non_nan_aucs = report["auc"].dropna().tolist()
    assert non_nan_aucs == sorted(non_nan_aucs, reverse=True)


def test_build_comparison_report_classifies_lora_results_as_their_own_kind() -> None:
    # Phase 11's `ProbeResult(name="lora_finetuned", ...)` must not be
    # misclassified as a baseline just because it isn't named "probe_*".
    lora_result = ProbeResult(name="lora_finetuned", auc=0.7, n_train=10, n_val=3)
    report = build_comparison_report([lora_result], checkpoint_name="ckpt-a")
    assert report.loc[0, "kind"] == "lora"
    assert report.loc[0, "checkpoint"] == "ckpt-a"
