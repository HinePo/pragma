from pragma.processing.categorical import CategoricalEncoder
from pragma.processing.special_tokens import SpecialTokens


def test_known_values_get_stable_distinct_ids() -> None:
    encoder = CategoricalEncoder()
    encoder.fit("currency", ["GBP", "EUR", "GBP", "USD"], base_id=10)
    ids = {v: encoder.transform("currency", v) for v in ["GBP", "EUR", "USD"]}
    assert len(set(ids.values())) == 3
    assert all(10 <= i < 13 for i in ids.values())


def test_unknown_value_falls_back_to_shared_unk_token() -> None:
    encoder = CategoricalEncoder()
    encoder.fit("currency", ["GBP", "EUR"], base_id=0)
    assert encoder.transform("currency", "ISK") == SpecialTokens().UNK


def test_oov_rate_is_tracked_per_key() -> None:
    encoder = CategoricalEncoder()
    encoder.fit("currency", ["GBP", "EUR"], base_id=0)
    encoder.transform("currency", "GBP")
    encoder.transform("currency", "ISK")
    encoder.transform("currency", "ISK")
    assert encoder.stats_for("currency").oov_rate == 2 / 3


def test_different_keys_get_non_overlapping_id_blocks() -> None:
    encoder = CategoricalEncoder()
    encoder.fit("currency", ["GBP", "EUR"], base_id=10)
    encoder.fit("direction", ["in", "out"], base_id=encoder.base_id("currency") + 2)
    currency_ids = {encoder.transform("currency", v) for v in ["GBP", "EUR"]}
    direction_ids = {encoder.transform("direction", v) for v in ["in", "out"]}
    assert currency_ids.isdisjoint(direction_ids)


def test_transform_unfitted_key_raises() -> None:
    encoder = CategoricalEncoder()
    try:
        encoder.transform("unknown_key", "x")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_save_load_round_trip_preserves_vocab() -> None:
    encoder = CategoricalEncoder()
    encoder.fit("currency", ["GBP", "EUR", "USD"], base_id=5)
    restored = CategoricalEncoder.from_dict(encoder.to_dict())
    for value in ["GBP", "EUR", "USD", "ISK"]:
        assert encoder.transform("currency", value) == restored.transform("currency", value)
