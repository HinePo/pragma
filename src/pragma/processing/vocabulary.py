"""Semantic-key vocabulary (implementation plan, section 6.1: "one token per key")."""

from __future__ import annotations

from typing import Any

from pragma.processing.special_tokens import SpecialTokens
from pragma.schema import FieldType, SchemaRegistry

TOKENIZABLE_FIELD_TYPES = (FieldType.NUMERICAL, FieldType.CATEGORICAL, FieldType.TEXT)


class KeyVocabulary:
    """Assigns one stable token ID to every tokenizable canonical key.

    Fields the `SchemaRegistry` marks as numerical, categorical, or text
    become key/value tokens with a data-derived value token (section 6.4).
    Lifelong-milestone fields (registered via
    `register_profile_field(..., lifelong_milestone=True)`) also get a key
    token, but their value is always one of the shared
    `[MILESTONE_PRESENT]`/`[MILESTONE_ABSENT]` tokens (ADR 0008) rather than
    a value derived from the timestamp itself. Plain identifiers and
    structural timestamps (`entity_id`, `event_id`, `created_at`,
    `signup_at`) are never tokenized directly; their timing is carried by
    RoPE coordinates instead (section 7.3).
    """

    def __init__(self, keys: list[str], *, base_id: int) -> None:
        self._base_id = base_id
        self._key_to_id: dict[str, int] = {key: base_id + i for i, key in enumerate(keys)}
        self._id_to_key: dict[int, str] = {v: k for k, v in self._key_to_id.items()}

    @classmethod
    def fit(
        cls, registry: SchemaRegistry, *, base_id: int = SpecialTokens().count
    ) -> KeyVocabulary:
        keys = sorted(
            {
                key
                for key, policy in registry.fields.items()
                if policy.field_type in TOKENIZABLE_FIELD_TYPES
            }
            | set(registry.lifelong_milestone_fields)
        )
        return cls(keys, base_id=base_id)

    def __len__(self) -> int:
        return len(self._key_to_id)

    @property
    def base_id(self) -> int:
        return self._base_id

    @property
    def next_id(self) -> int:
        return self._base_id + len(self)

    def id_for(self, key: str) -> int:
        try:
            return self._key_to_id[key]
        except KeyError as exc:
            raise KeyError(f"'{key}' is not a tokenizable key in this vocabulary") from exc

    def key_for(self, token_id: int) -> str:
        return self._id_to_key[token_id]

    def keys(self) -> list[str]:
        return sorted(self._key_to_id, key=lambda k: self._key_to_id[k])

    def to_dict(self) -> dict[str, object]:
        return {"base_id": self._base_id, "keys": self.keys()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KeyVocabulary:
        return cls(list(data["keys"]), base_id=int(data["base_id"]))
