import tempfile
from pathlib import Path

from pragma.config import ProcessorConfig
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.storage import (
    ParquetShardStore,
    event_count_bucket,
    read_dataset,
    write_dataset,
)
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry


def _tokenized_records():
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=100,
            seed=99,
            max_events_per_entity=25,
            n_zero_event_entities=5,
            n_single_event_entities=5,
            n_long_history_entities=2,
            long_history_event_count=60,
            n_same_timestamp_entities=5,
        )
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=8, bpe_vocab_size=150))
    processor.fit(records)
    return [processor.transform(r) for r in records]


def test_event_count_bucket_labels() -> None:
    assert event_count_bucket(0) == "0-0"
    assert event_count_bucket(1) == "1-49"
    assert event_count_bucket(49) == "1-49"
    assert event_count_bucket(50) == "50-499"
    assert event_count_bucket(1000) == "500+"


def test_write_read_round_trip_preserves_every_record_field() -> None:
    tokenized = _tokenized_records()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "shards"
        write_dataset(tokenized, out_dir, ParquetShardStore())
        restored = read_dataset(out_dir, ParquetShardStore())

    before = {r.entity_id: r.to_dict() for r in tokenized}
    after = {r.entity_id: r.to_dict() for r in restored}
    assert before == after


def test_split_filtering_reads_only_requested_split() -> None:
    tokenized = _tokenized_records()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "shards"
        write_dataset(tokenized, out_dir, ParquetShardStore())
        train_only = read_dataset(out_dir, ParquetShardStore(), split="train")

    assert train_only
    assert all(r.split == "train" for r in train_only)
    expected = {r.entity_id for r in tokenized if r.split == "train"}
    assert {r.entity_id for r in train_only} == expected


def test_manifest_checksums_detect_corruption() -> None:
    tokenized = _tokenized_records()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "shards"
        manifest = write_dataset(tokenized, out_dir, ParquetShardStore())
        corrupted_path = out_dir / manifest.shards[0].path
        corrupted_path.write_bytes(b"not a valid parquet file")

        try:
            read_dataset(out_dir, ParquetShardStore())
            raise AssertionError("expected a checksum-mismatch error")
        except ValueError as exc:
            assert "checksum mismatch" in str(exc)


def test_records_are_grouped_by_split_and_event_count_bucket() -> None:
    tokenized = _tokenized_records()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "shards"
        manifest = write_dataset(tokenized, out_dir, ParquetShardStore())

    seen = {(s.split, s.length_bucket) for s in manifest.shards}
    assert len(seen) == len(manifest.shards), "one shard per (split, bucket) group expected"
    assert sum(s.n_records for s in manifest.shards) == len(tokenized)
