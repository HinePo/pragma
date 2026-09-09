import numpy as np
import pytest

from pragma.processing.numeric import NumericBucketizer


def test_zero_has_its_own_dedicated_bucket() -> None:
    bucketizer = NumericBucketizer(n_buckets=4, value_base_id=100)
    bucketizer.fit("amount", np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0]))
    assert bucketizer.transform("amount", 0.0) == bucketizer.zero_id
    assert bucketizer.transform("amount", 0.0) != bucketizer.percentile_id(0)


def test_bucket_ids_come_from_the_shared_global_range_regardless_of_key() -> None:
    bucketizer = NumericBucketizer(n_buckets=4, value_base_id=100)
    bucketizer.fit("amount", np.array([1.0, 10.0, 100.0, 1000.0]))
    bucketizer.fit("balance", np.array([5.0, 50.0, 500.0, 5000.0]))
    # Same relative position in each key's own distribution -> same global token id.
    assert bucketizer.transform("amount", 1.0) == bucketizer.transform("balance", 5.0)


def test_outliers_clip_to_edge_buckets_rather_than_erroring() -> None:
    bucketizer = NumericBucketizer(n_buckets=4, value_base_id=0)
    bucketizer.fit("amount", np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    low = bucketizer.transform("amount", -1000.0)
    high = bucketizer.transform("amount", 1_000_000.0)
    assert low == bucketizer.percentile_id(0)
    assert high == bucketizer.percentile_id(bucketizer.n_buckets - 1)


def test_transform_unknown_key_raises() -> None:
    bucketizer = NumericBucketizer(n_buckets=4)
    with pytest.raises(KeyError):
        bucketizer.transform("unknown", 1.0)


def test_save_load_round_trip_preserves_boundaries_and_ids() -> None:
    bucketizer = NumericBucketizer(n_buckets=6, value_base_id=50)
    bucketizer.fit("amount", np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    restored = NumericBucketizer.from_dict(bucketizer.to_dict())

    for value in [0.0, 1.5, 3.3, 6.9, 100.0]:
        assert bucketizer.transform("amount", value) == restored.transform("amount", value)


def test_single_value_key_still_produces_a_valid_bucket() -> None:
    bucketizer = NumericBucketizer(n_buckets=4)
    bucketizer.fit("amount", np.array([5.0]))
    bucket_id = bucketizer.transform("amount", 5.0)
    assert 0 <= bucket_id - bucketizer.zero_id - 1 < bucketizer.n_buckets
    assert bucketizer.transform("amount", 0.0) == bucketizer.zero_id
