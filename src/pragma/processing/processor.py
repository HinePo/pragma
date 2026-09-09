"""`PragmaProcessor`: coordinates all fitted sub-encoders (implementation plan, section 6.5).

Converts `EvaluationRecord`s into `TokenizedRecord`s and owns the single
versioned artifact bundle (schema version, key vocabulary, numeric buckets,
categorical vocabularies, BPE model, special tokens, truncation policy, data
fingerprint) that a model checkpoint requires to be reproducible (ADR 0003).
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pragma.config.processor_config import ProcessorConfig
from pragma.data.records import EvaluationRecord, EventRecord
from pragma.processing.categorical import CategoricalEncoder
from pragma.processing.numeric import NumericBucketizer
from pragma.processing.special_tokens import SpecialTokens
from pragma.processing.temporal import calendar_features, time_to_evaluation, time_to_latest
from pragma.processing.text_bpe import TextBPEEncoder
from pragma.processing.tokenized_record import FieldTokens, TokenizedEvent, TokenizedRecord
from pragma.processing.vocabulary import KeyVocabulary
from pragma.schema import FieldType, SchemaRegistry

SCHEMA_VERSION = 1
BUNDLE_METADATA_FILE = "bundle.json"
BPE_MODEL_FILE = "bpe_tokenizer.json"


@dataclass(frozen=True)
class FitReport:
    """Auditing summary produced by `PragmaProcessor.fit` (Phase 2 exit gate)."""

    n_train_records: int
    numeric_stats: dict[str, dict[str, object]]
    categorical_oov_rates: dict[str, float]
    text_oov_rate: float
    text_local_vocab_size: int
    total_vocab_size: int


class PragmaProcessor:
    """Fits and applies the structured processor; owns the versioned artifact bundle."""

    def __init__(self, registry: SchemaRegistry, config: ProcessorConfig) -> None:
        self.registry = registry
        self.config = config
        # The full token ID space is laid out once, at construction time, as
        # contiguous ranges: [special][keys][numeric buckets][categorical...][bpe].
        # Key IDs and the numeric bucket range have a size that only depends on
        # the schema/config, so they're fixed here; the categorical and BPE
        # ranges depend on what values are actually seen in the training data
        # and are only finalized in `fit()` (see ADR 0008 for why numeric
        # buckets are global but categorical values are namespaced per key).
        self.special_tokens = SpecialTokens()
        self.key_vocab = KeyVocabulary.fit(registry, base_id=self.special_tokens.count)

        numeric_base = self.key_vocab.next_id
        self.numeric_bucketizer = NumericBucketizer(
            n_buckets=config.n_numeric_buckets, value_base_id=numeric_base
        )
        self._categorical_base = numeric_base + self.numeric_bucketizer.vocab_size
        self.categorical_encoder = CategoricalEncoder(unk_id=self.special_tokens.UNK)
        self._text_base: int | None = None  # set once fit() knows the categorical range's size
        self.text_encoder = TextBPEEncoder(vocab_size=config.bpe_vocab_size, value_base_id=0)
        self._fit_report: FitReport | None = None
        self._train_fingerprint: str | None = None
        self._fit_timestamp: str | None = None

    # -- field classification -------------------------------------------

    def _numeric_keys(self) -> list[str]:
        return [k for k, p in self.registry.fields.items() if p.field_type is FieldType.NUMERICAL]

    def _categorical_keys(self) -> list[str]:
        return [k for k, p in self.registry.fields.items() if p.field_type is FieldType.CATEGORICAL]

    def _text_keys(self) -> list[str]:
        return [k for k, p in self.registry.fields.items() if p.field_type is FieldType.TEXT]

    # -- fitting ----------------------------------------------------------

    def fit(self, train_records: list[EvaluationRecord]) -> FitReport:
        """Fit every sub-encoder using only records whose `split == 'train'`.

        Filtering here (rather than trusting the caller to pre-filter) is
        deliberate: it makes "never fit on val/test" a property of the
        processor itself instead of a convention callers must remember, per
        the Phase 2 exit gate in section 17 of the implementation plan.
        """
        train_records = [r for r in train_records if r.split == "train"]
        if not train_records:
            raise ValueError("PragmaProcessor.fit requires at least one training-split record")

        numeric_values: dict[str, list[float]] = {k: [] for k in self._numeric_keys()}
        numeric_nulls: dict[str, int] = {k: 0 for k in self._numeric_keys()}
        categorical_values: dict[str, list[str]] = {k: [] for k in self._categorical_keys()}
        text_values: list[str] = []

        for record in train_records:
            for event in record.events_before_evaluation:
                for raw_key, value in event.fields.items():
                    canonical = self.registry.resolve(raw_key)
                    if canonical is None:
                        continue
                    field_type = self.registry.get_field(canonical).field_type
                    if field_type is FieldType.NUMERICAL:
                        numeric_values.setdefault(canonical, []).append(float(value))  # type: ignore[arg-type]
                    elif field_type is FieldType.CATEGORICAL:
                        categorical_values.setdefault(canonical, []).append(str(value))
                    elif field_type is FieldType.TEXT:
                        text_values.append(str(value))

        for key in self._numeric_keys():
            numeric_array = np.asarray(numeric_values.get(key, []), dtype=float)
            self.numeric_bucketizer.fit(key, numeric_array, null_count=numeric_nulls.get(key, 0))

        # Each categorical key claims a contiguous block sized to its own
        # fitted vocabulary (ADR 0008), so the running offset only advances
        # once a key's vocab is known - this is why categorical keys must be
        # fit in the registry's fixed iteration order rather than in parallel.
        next_categorical_id = self._categorical_base
        for key in self._categorical_keys():
            categorical_list = categorical_values.get(key, [])
            self.categorical_encoder.fit(key, categorical_list, base_id=next_categorical_id)
            next_categorical_id += self.categorical_encoder.vocab_size(key)
        self._text_base = next_categorical_id

        self.text_encoder = TextBPEEncoder(
            vocab_size=self.config.bpe_vocab_size, value_base_id=self._text_base
        )
        self.text_encoder.fit(text_values, min_frequency=self.config.bpe_min_frequency)

        self._train_fingerprint = self._fingerprint(train_records)
        self._fit_timestamp = datetime.now(UTC).isoformat()

        total_vocab_size = self._text_base + self.text_encoder.local_vocab_size
        self._fit_report = FitReport(
            n_train_records=len(train_records),
            numeric_stats={
                k: {
                    "boundaries": self.numeric_bucketizer.stats_for(k).boundaries,
                    "n_train_values": self.numeric_bucketizer.stats_for(k).n_train_values,
                    "n_zero_values": self.numeric_bucketizer.stats_for(k).n_zero_values,
                }
                for k in self._numeric_keys()
            },
            categorical_oov_rates={
                k: self.categorical_encoder.stats_for(k).oov_rate for k in self._categorical_keys()
            },
            text_oov_rate=self.text_encoder.oov_rate,
            text_local_vocab_size=self.text_encoder.local_vocab_size,
            total_vocab_size=total_vocab_size,
        )
        return self._fit_report

    @staticmethod
    def _fingerprint(records: list[EvaluationRecord]) -> str:
        entity_ids = sorted(r.entity_id for r in records)
        digest = hashlib.sha256("|".join(entity_ids).encode()).hexdigest()
        return digest

    @property
    def is_fitted(self) -> bool:
        return self._text_base is not None

    @property
    def total_vocab_size(self) -> int:
        if self._fit_report is None:
            raise RuntimeError("processor must be fit before total_vocab_size is available")
        return self._fit_report.total_vocab_size

    @property
    def train_fingerprint(self) -> str | None:
        """Hash of the training-split entity IDs used to fit this processor.

        Recorded in the data manifest (section 10.1) alongside tokenized
        shards so a shard directory can be checked against the processor
        bundle that produced it.
        """
        return self._train_fingerprint

    # -- transform ----------------------------------------------------------

    def _tokenize_value(self, canonical_key: str, raw_value: object) -> FieldTokens:
        field_type = self.registry.get_field(canonical_key).field_type
        key_id = self.key_vocab.id_for(canonical_key)

        if field_type is FieldType.NUMERICAL:
            value_id = self.numeric_bucketizer.transform(
                canonical_key,
                float(raw_value),  # type: ignore[arg-type]
            )
            return FieldTokens(key_ids=(key_id,), value_ids=(value_id,), within_field_pos=(0,))

        if field_type is FieldType.CATEGORICAL:
            value_id = self.categorical_encoder.transform(canonical_key, str(raw_value))
            return FieldTokens(key_ids=(key_id,), value_ids=(value_id,), within_field_pos=(0,))

        if field_type is FieldType.TEXT:
            value_ids = self.text_encoder.transform(str(raw_value))
            if not value_ids:
                value_ids = [self.special_tokens.UNK]
            value_ids = value_ids[: self.config.max_within_field_tokens]
            return FieldTokens(
                key_ids=tuple(key_id for _ in value_ids),
                value_ids=tuple(value_ids),
                within_field_pos=tuple(range(len(value_ids))),
            )

        raise ValueError(f"field '{canonical_key}' is not tokenizable (type={field_type})")

    def _tokenize_profile(
        self, record: EvaluationRecord
    ) -> tuple[tuple[FieldTokens, ...], tuple[float, ...]]:
        fields: list[FieldTokens] = []
        time_coords: list[float] = []

        for key in sorted(self.registry.profile_fields):
            if key in self.registry.lifelong_milestone_fields:
                milestone_time = record.profile_state.milestones.get(key)
                value_id = (
                    self.special_tokens.MILESTONE_PRESENT
                    if milestone_time is not None
                    else self.special_tokens.MILESTONE_ABSENT
                )
                field_tokens = FieldTokens(
                    key_ids=(self.key_vocab.id_for(key),),
                    value_ids=(value_id,),
                    within_field_pos=(0,),
                )
                fields.append(field_tokens)
                time_coords.append(time_to_evaluation(milestone_time, record.evaluation_time))
                continue

            if key not in self._tokenizable_keys():
                continue
            raw_value = record.profile_state.attributes.get(key)
            if raw_value is None:
                continue
            field_tokens = self._tokenize_value(key, raw_value)
            fields.append(field_tokens)
            time_coords.extend([0.0] * len(field_tokens.key_ids))

        fields = fields[: self.config.max_profile_tokens]
        n_tokens_kept = sum(len(f.key_ids) for f in fields)
        return tuple(fields), tuple(time_coords[:n_tokens_kept])

    def _tokenizable_keys(self) -> set[str]:
        return set(self.key_vocab.keys())

    def _tokenize_event(self, event: EventRecord, *, latest_time: datetime) -> TokenizedEvent:
        fields: list[FieldTokens] = []
        for raw_key, raw_value in event.fields.items():
            canonical = self.registry.resolve(raw_key)
            if canonical is None or canonical not in self._tokenizable_keys():
                continue
            fields.append(self._tokenize_value(canonical, raw_value))

        # Truncation policy (section 6.5): drop whole fields from the end of
        # `event.fields`'s iteration order rather than cutting a multi-token
        # BPE value mid-subword, which would silently corrupt its within-field
        # positions.
        n_tokens = sum(len(f.key_ids) for f in fields)
        truncated = n_tokens > self.config.max_event_tokens
        if truncated:
            kept: list[FieldTokens] = []
            running = 0
            for f in fields:
                if running + len(f.key_ids) > self.config.max_event_tokens:
                    break
                kept.append(f)
                running += len(f.key_ids)
            fields = kept

        return TokenizedEvent(
            event_id=event.event_id,
            fields=tuple(fields),
            time_to_latest=time_to_latest(event.created_at, latest_time),
            calendar_features=calendar_features(event.created_at).as_tuple(),
            truncated=truncated,
        )

    def transform(self, record: EvaluationRecord) -> TokenizedRecord:
        if not self.is_fitted:
            raise RuntimeError("PragmaProcessor must be fit (or loaded) before transform()")

        profile_fields, profile_time_coords = self._tokenize_profile(record)

        events_sorted = sorted(
            record.events_before_evaluation, key=lambda e: (e.created_at, e.event_id)
        )
        n_events_total = len(events_sorted)
        # Keep the most recent events, not the earliest (section 7.4's note on
        # histories near the 6,500-event cap) - a slice from the end of a
        # chronologically sorted list.
        events_kept = events_sorted[-self.config.max_history_events :]
        events_truncated = len(events_kept) < n_events_total
        latest_time = events_kept[-1].created_at if events_kept else record.evaluation_time

        tokenized_events = tuple(
            self._tokenize_event(event, latest_time=latest_time) for event in events_kept
        )

        return TokenizedRecord(
            entity_id=record.entity_id,
            split=record.split,
            profile_fields=profile_fields,
            profile_time_coords=profile_time_coords,
            events=tokenized_events,
            n_events_total=n_events_total,
            n_events_kept=len(events_kept),
            events_truncated=events_truncated,
        )

    # -- persistence ----------------------------------------------------------

    def save(self, path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("cannot save an unfit PragmaProcessor")
        path.mkdir(parents=True, exist_ok=True)

        self.text_encoder.save(path / BPE_MODEL_FILE)

        bundle = {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.to_dict(),
            "special_tokens": self.special_tokens.as_dict(),
            "key_vocab": self.key_vocab.to_dict(),
            "numeric_bucketizer": self.numeric_bucketizer.to_dict(),
            "categorical_encoder": self.categorical_encoder.to_dict(),
            "text_base_id": self._text_base,
            "train_fingerprint": self._train_fingerprint,
            "fit_timestamp": self._fit_timestamp,
            "python_version": sys.version,
            "platform": platform.platform(),
            "fit_report": {
                "n_train_records": self._fit_report.n_train_records,
                "numeric_stats": self._fit_report.numeric_stats,
                "categorical_oov_rates": self._fit_report.categorical_oov_rates,
                "text_oov_rate": self._fit_report.text_oov_rate,
                "text_local_vocab_size": self._fit_report.text_local_vocab_size,
                "total_vocab_size": self._fit_report.total_vocab_size,
            }
            if self._fit_report
            else None,
        }
        (path / BUNDLE_METADATA_FILE).write_text(json.dumps(bundle, indent=2, default=str))

    @classmethod
    def load(cls, path: Path, registry: SchemaRegistry) -> PragmaProcessor:
        bundle = json.loads((path / BUNDLE_METADATA_FILE).read_text())
        if bundle["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"processor bundle schema_version {bundle['schema_version']} != "
                f"expected {SCHEMA_VERSION}"
            )

        config = ProcessorConfig.from_dict(bundle["config"])
        processor = cls(registry, config)
        processor.key_vocab = KeyVocabulary.from_dict(bundle["key_vocab"])
        processor.numeric_bucketizer = NumericBucketizer.from_dict(bundle["numeric_bucketizer"])
        processor.categorical_encoder = CategoricalEncoder.from_dict(bundle["categorical_encoder"])
        processor._categorical_base = processor.numeric_bucketizer.value_base_id + (
            processor.numeric_bucketizer.vocab_size
        )
        processor._text_base = bundle["text_base_id"]
        processor.text_encoder = TextBPEEncoder.load(
            path / BPE_MODEL_FILE,
            vocab_size=config.bpe_vocab_size,
            value_base_id=bundle["text_base_id"],
        )
        processor._train_fingerprint = bundle["train_fingerprint"]
        processor._fit_timestamp = bundle["fit_timestamp"]
        fit_report = bundle.get("fit_report")
        if fit_report:
            processor._fit_report = FitReport(**fit_report)
        return processor
