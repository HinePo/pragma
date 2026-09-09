"""Shared BPE encoder for approved free-text fields (implementation plan, section 6.3).

Trained once across every field the `SchemaRegistry` marks `FieldType.TEXT`,
per ADR 0003. Uses a byte-level pre-tokenizer/decoder so the vocabulary is
global and, by construction, cannot produce out-of-vocabulary fragments
(worst case falls back to single-byte tokens) — see ADR 0008.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


class TextBPEEncoder:
    """Fits and applies one shared byte-level BPE vocabulary."""

    def __init__(self, vocab_size: int, *, value_base_id: int = 0) -> None:
        self.vocab_size = vocab_size
        self.value_base_id = value_base_id
        self._tokenizer: Tokenizer | None = None
        self._oov_lookups = 0
        self._oov_hits = 0

    @property
    def is_fitted(self) -> bool:
        return self._tokenizer is not None

    def fit(self, texts: list[str], *, min_frequency: int = 1) -> None:
        """Train the shared BPE model on approved training-partition text values."""
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<unk>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tokenizer.train_from_iterator(texts or [""], trainer=trainer)
        self._tokenizer = tokenizer

    def transform(self, text: str) -> list[int]:
        """Encode `text` into global BPE value token IDs, preserving subword order."""
        if self._tokenizer is None:
            raise RuntimeError("TextBPEEncoder must be fit or loaded before transform()")
        encoding = self._tokenizer.encode(text)
        unk_local_id = self._tokenizer.token_to_id("<unk>")
        ids: list[int] = []
        for local_id in encoding.ids:
            self._oov_lookups += 1
            if local_id == unk_local_id:
                self._oov_hits += 1
            ids.append(self.value_base_id + local_id)
        return ids

    @property
    def local_vocab_size(self) -> int:
        if self._tokenizer is None:
            raise RuntimeError("TextBPEEncoder must be fit or loaded before local_vocab_size")
        return self._tokenizer.get_vocab_size()

    @property
    def oov_rate(self) -> float:
        return (self._oov_hits / self._oov_lookups) if self._oov_lookups else 0.0

    def save(self, path: Path) -> None:
        if self._tokenizer is None:
            raise RuntimeError("cannot save an unfit TextBPEEncoder")
        self._tokenizer.save(str(path))

    @classmethod
    def load(cls, path: Path, *, vocab_size: int, value_base_id: int) -> TextBPEEncoder:
        encoder = cls(vocab_size=vocab_size, value_base_id=value_base_id)
        encoder._tokenizer = Tokenizer.from_file(str(path))
        return encoder
