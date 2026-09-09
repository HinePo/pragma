import tempfile
from pathlib import Path

from pragma.processing.text_bpe import TextBPEEncoder

TEXTS = [
    "monthly subscription",
    "grocery shopping",
    "dinner with friends",
    "metal plan",
    "rent payment",
] * 20


def test_multi_token_value_repeats_positions_from_zero() -> None:
    encoder = TextBPEEncoder(vocab_size=100, value_base_id=1000)
    encoder.fit(TEXTS)
    ids = encoder.transform("grocery shopping")
    assert len(ids) >= 1
    assert all(i >= 1000 for i in ids)


def test_byte_level_bpe_has_no_out_of_vocabulary_fragments() -> None:
    encoder = TextBPEEncoder(vocab_size=80, value_base_id=0)
    encoder.fit(TEXTS)
    # Encode text containing characters never seen during fit.
    encoder.transform("a completely unseen string with 12345 !@#$%")
    assert encoder.oov_rate == 0.0


def test_save_load_round_trip_produces_identical_ids() -> None:
    encoder = TextBPEEncoder(vocab_size=100, value_base_id=500)
    encoder.fit(TEXTS)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bpe.json"
        encoder.save(path)
        restored = TextBPEEncoder.load(path, vocab_size=100, value_base_id=500)
        for text in TEXTS[:5] + ["novel merchant name"]:
            assert encoder.transform(text) == restored.transform(text)


def test_transform_before_fit_raises() -> None:
    encoder = TextBPEEncoder(vocab_size=50)
    try:
        encoder.transform("hello")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
