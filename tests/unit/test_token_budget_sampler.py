from pragma.config import ProcessorConfig, TokenBudgetConfig
from pragma.data.dataset import TokenizedRecordDataset
from pragma.data.records import PointInTimeRecordBuilder, SplitConfig
from pragma.data.synthetic import SyntheticDataConfig, generate_synthetic_corpus
from pragma.data.token_budget_sampler import TokenBudgetBatchSampler
from pragma.processing import PragmaProcessor
from pragma.schema import SchemaRegistry


def _dataset() -> TokenizedRecordDataset:
    events_df, profile_df, _ = generate_synthetic_corpus(
        SyntheticDataConfig(
            n_entities=80,
            seed=11,
            max_events_per_entity=30,
            n_zero_event_entities=5,
            n_single_event_entities=5,
            n_long_history_entities=2,
            long_history_event_count=200,
            n_same_timestamp_entities=5,
        )
    )
    registry = SchemaRegistry.default()
    records = PointInTimeRecordBuilder(registry, SplitConfig()).build(events_df, profile_df)
    processor = PragmaProcessor(registry, ProcessorConfig(n_numeric_buckets=6, bpe_vocab_size=100))
    processor.fit(records)
    tokenized = [processor.transform(r) for r in records]
    return TokenizedRecordDataset(tokenized)


def test_every_record_appears_exactly_once_per_epoch_single_replica() -> None:
    dataset = _dataset()
    sampler = TokenBudgetBatchSampler(dataset, TokenBudgetConfig(max_event_tokens_per_batch=500))
    all_indices = [i for batch in sampler for i in batch]
    assert sorted(all_indices) == list(range(len(dataset)))


def test_batches_respect_the_token_and_event_budgets_except_oversized_singletons() -> None:
    dataset = _dataset()
    config = TokenBudgetConfig(max_event_tokens_per_batch=200, max_events_per_batch=100)
    sampler = TokenBudgetBatchSampler(dataset, config)
    for batch in sampler:
        n_tokens = sum(sampler._event_tokens[i] for i in batch)
        n_events = sum(sampler._event_counts[i] for i in batch)
        if len(batch) > 1:
            assert n_tokens <= config.max_event_tokens_per_batch
            assert n_events <= config.max_events_per_batch


def test_max_records_per_batch_is_respected() -> None:
    dataset = _dataset()
    config = TokenBudgetConfig(
        max_event_tokens_per_batch=1_000_000,
        max_events_per_batch=1_000_000,
        max_records_per_batch=3,
    )
    sampler = TokenBudgetBatchSampler(dataset, config)
    for batch in sampler:
        assert len(batch) <= 3


def test_distributed_ranks_are_disjoint_up_to_padding_duplicates() -> None:
    dataset = _dataset()
    config = TokenBudgetConfig(max_event_tokens_per_batch=500)
    sampler_0 = TokenBudgetBatchSampler(dataset, config, num_replicas=3, rank=0)
    sampler_1 = TokenBudgetBatchSampler(dataset, config, num_replicas=3, rank=1)
    sampler_2 = TokenBudgetBatchSampler(dataset, config, num_replicas=3, rank=2)

    assert len(sampler_0) == len(sampler_1) == len(sampler_2)

    all_batches = list(sampler_0) + list(sampler_1) + list(sampler_2)
    all_indices = [i for batch in all_batches for i in batch]
    # Every real record must appear at least once across all ranks; padding may
    # cause a handful of records to appear more than once, never zero times.
    assert set(all_indices) == set(range(len(dataset)))


def test_set_epoch_changes_batch_order() -> None:
    dataset = _dataset()
    sampler = TokenBudgetBatchSampler(dataset, TokenBudgetConfig(max_event_tokens_per_batch=500))
    batches_epoch_0 = list(sampler)
    sampler.set_epoch(1)
    batches_epoch_1 = list(sampler)
    assert batches_epoch_0 != batches_epoch_1


def test_deterministic_for_a_fixed_seed_and_epoch() -> None:
    dataset = _dataset()
    config = TokenBudgetConfig(max_event_tokens_per_batch=500, seed=42)
    sampler_a = TokenBudgetBatchSampler(dataset, config)
    sampler_b = TokenBudgetBatchSampler(dataset, config)
    assert list(sampler_a) == list(sampler_b)


def test_stats_report_matches_dataset_totals() -> None:
    dataset = _dataset()
    sampler = TokenBudgetBatchSampler(dataset, TokenBudgetConfig(max_event_tokens_per_batch=500))
    stats = sampler.stats()
    assert stats.n_records == len(dataset)
    assert stats.n_batches == len(stats.records_per_batch) == len(stats.event_tokens_per_batch)
    assert sum(stats.records_per_batch) == stats.n_records
