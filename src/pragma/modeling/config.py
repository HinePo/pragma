"""`PragmaConfig`: PRAGMA-S model dimensions and vocabulary/special-token identity.

Implementation plan, section 7.1, 11.1. Subclasses `PreTrainedConfig` for
standard `save_pretrained`/`from_pretrained` and future Hub compatibility
(section 3). `from_processor` is the only place a fitted `PragmaProcessor`'s
vocabulary layout and special-token IDs get read into the model config
(ADR 0009), so the model never depends on the processor at runtime.
"""

from __future__ import annotations

from transformers import PreTrainedConfig

from pragma.processing import PragmaProcessor


class PragmaConfig(PreTrainedConfig):
    model_type = "pragma"

    def __init__(
        self,
        vocab_size: int = 512,
        value_vocab_start: int = 64,
        hidden_size: int = 192,
        intermediate_size: int = 768,
        num_heads: int = 3,
        profile_layers: int = 1,
        event_layers: int = 5,
        history_layers: int = 2,
        dropout: float = 0.1,
        max_within_field_position: int = 32,
        rope_base: float = 10000.0,
        usr_token_id: int = 3,
        evt_token_id: int = 4,
        label_smoothing: float = 0.0,
        attention_backend: str = "padded",
        **kwargs: object,
    ) -> None:
        if attention_backend not in ("padded", "varlen"):
            raise ValueError(
                f"attention_backend must be 'padded' or 'varlen', got {attention_backend!r}"
            )
        self.vocab_size = vocab_size
        self.value_vocab_start = value_vocab_start
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.profile_layers = profile_layers
        self.event_layers = event_layers
        self.history_layers = history_layers
        self.dropout = dropout
        self.max_within_field_position = max_within_field_position
        self.rope_base = rope_base
        self.usr_token_id = usr_token_id
        self.evt_token_id = evt_token_id
        self.label_smoothing = label_smoothing
        self.attention_backend = attention_backend
        super().__init__(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_processor(cls, processor: PragmaProcessor, **overrides: object) -> PragmaConfig:
        """Builds a config from a fitted `PragmaProcessor`'s vocabulary layout.

        `processor.numeric_bucketizer.value_base_id` is the first value-token
        ID (numeric buckets are the first value range laid out after keys —
        ADR 0008) and doubles as `value_vocab_start`, per ADR 0009's
        value-vocabulary-only MLM logit scope.
        """
        kwargs: dict[str, object] = {
            "vocab_size": processor.total_vocab_size,
            "value_vocab_start": processor.numeric_bucketizer.value_base_id,
            "usr_token_id": processor.special_tokens.USR,
            "evt_token_id": processor.special_tokens.EVT,
        }
        kwargs.update(overrides)
        return cls(**kwargs)  # type: ignore[arg-type]
