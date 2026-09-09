"""Canonical schema registry for PRAGMA raw events and profile state.

Defines, for every semantic key, the canonical name, value type, null policy,
masking eligibility, and regulatory eligibility described in the implementation
plan (section 5.5). Raw source fields must resolve to a canonical key through
this registry before they are eligible for point-in-time record construction
or processor fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FieldType(StrEnum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    TIMESTAMP = "timestamp"
    IDENTIFIER = "identifier"
    IGNORED = "ignored"


class NullPolicy(StrEnum):
    """How a missing value for this key should be treated."""

    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"
    ZERO_BUCKET = "zero_bucket"


@dataclass(frozen=True)
class FieldPolicy:
    """Canonical definition of one semantic key."""

    key: str
    field_type: FieldType
    aliases: tuple[str, ...] = ()
    null_policy: NullPolicy = NullPolicy.ALLOWED
    maskable: bool = True
    regulated: bool = False
    pii_risk: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if self.field_type in (FieldType.IDENTIFIER, FieldType.TIMESTAMP) and self.maskable:
            object.__setattr__(self, "maskable", False)


@dataclass(frozen=True)
class EventFamily:
    """The canonical keys that may appear on one event source/type."""

    name: str
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    description: str = ""

    @property
    def allowed_fields(self) -> tuple[str, ...]:
        return self.required_fields + self.optional_fields


class SchemaRegistry:
    """Canonical key vocabulary, source-alias resolution, and raw validation."""

    def __init__(self) -> None:
        self._fields: dict[str, FieldPolicy] = {}
        self._alias_to_key: dict[str, str] = {}
        self._event_families: dict[str, EventFamily] = {}
        self._profile_fields: set[str] = set()
        self._lifelong_milestone_fields: set[str] = set()

    # -- registration ---------------------------------------------------

    def register_field(self, policy: FieldPolicy) -> None:
        if policy.key in self._fields:
            raise ValueError(f"field '{policy.key}' already registered")
        self._fields[policy.key] = policy
        self._alias_to_key[policy.key] = policy.key
        for alias in policy.aliases:
            if alias in self._alias_to_key:
                raise ValueError(f"alias '{alias}' already mapped to a canonical key")
            self._alias_to_key[alias] = policy.key

    def register_event_family(self, family: EventFamily) -> None:
        if family.name in self._event_families:
            raise ValueError(f"event family '{family.name}' already registered")
        for key in family.allowed_fields:
            if key not in self._fields:
                raise ValueError(f"event family '{family.name}' references unknown field '{key}'")
        self._event_families[family.name] = family

    def register_profile_field(self, key: str, *, lifelong_milestone: bool = False) -> None:
        if key not in self._fields:
            raise ValueError(f"profile field references unknown key '{key}'")
        self._profile_fields.add(key)
        if lifelong_milestone:
            self._lifelong_milestone_fields.add(key)

    # -- lookup -----------------------------------------------------------

    def resolve(self, source_field_name: str) -> str | None:
        """Map a raw source column name to its canonical key, or None if unknown."""
        return self._alias_to_key.get(source_field_name)

    def get_field(self, key: str) -> FieldPolicy:
        canonical = self.resolve(key)
        if canonical is None:
            raise KeyError(f"unknown field or alias: '{key}'")
        return self._fields[canonical]

    def get_event_family(self, name: str) -> EventFamily:
        return self._event_families[name]

    @property
    def fields(self) -> dict[str, FieldPolicy]:
        return dict(self._fields)

    @property
    def event_families(self) -> dict[str, EventFamily]:
        return dict(self._event_families)

    @property
    def profile_fields(self) -> frozenset[str]:
        return frozenset(self._profile_fields)

    @property
    def lifelong_milestone_fields(self) -> frozenset[str]:
        return frozenset(self._lifelong_milestone_fields)

    # -- validation ---------------------------------------------------------

    def validate_event(self, event_type: str, fields: dict[str, object]) -> list[str]:
        """Return a list of human-readable schema violations for one raw event.

        Empty list means the event conforms to the registered schema for its family.
        """
        issues: list[str] = []
        family = self._event_families.get(event_type)
        if family is None:
            return [f"unknown event family: '{event_type}'"]

        for required_key in family.required_fields:
            if required_key not in fields or fields[required_key] is None:
                policy = self._fields[required_key]
                if policy.null_policy is NullPolicy.FORBIDDEN or required_key not in fields:
                    issues.append(f"missing required field '{required_key}' for '{event_type}'")

        for raw_key, value in fields.items():
            canonical = self.resolve(raw_key)
            if canonical is None:
                issues.append(f"unmapped source field '{raw_key}' on event '{event_type}'")
                continue
            if canonical not in family.allowed_fields:
                issues.append(f"field '{canonical}' not allowed on event family '{event_type}'")
                continue
            policy = self._fields[canonical]
            if value is None and policy.null_policy is NullPolicy.FORBIDDEN:
                issues.append(f"null not allowed for field '{canonical}' on '{event_type}'")

        return issues

    def validate_profile(self, fields: dict[str, object]) -> list[str]:
        issues: list[str] = []
        for raw_key, value in fields.items():
            canonical = self.resolve(raw_key)
            if canonical is None:
                issues.append(f"unmapped source field '{raw_key}' on profile state")
                continue
            if canonical not in self._profile_fields:
                issues.append(f"field '{canonical}' not a registered profile field")
                continue
            policy = self._fields[canonical]
            if value is None and policy.null_policy is NullPolicy.FORBIDDEN:
                issues.append(f"null not allowed for profile field '{canonical}'")
        return issues

    # -- default synthetic-domain schema ------------------------------------

    @classmethod
    def default(cls) -> SchemaRegistry:
        """Build the canonical schema used for the PRAGMA synthetic corpus.

        Covers identifiers, one numerical key (amount), several categorical
        keys, two free-text keys, and profile attributes with lifelong
        milestones, matching the raw-event shape described in section 5.1.
        """
        registry = cls()

        identifiers = [
            FieldPolicy("entity_id", FieldType.IDENTIFIER),
            FieldPolicy("event_id", FieldType.IDENTIFIER),
            FieldPolicy("created_at", FieldType.TIMESTAMP, aliases=("event_time",)),
            FieldPolicy("type", FieldType.CATEGORICAL, aliases=("event_type", "source")),
        ]
        for policy in identifiers:
            registry.register_field(policy)

        event_value_fields = [
            FieldPolicy(
                "amount",
                FieldType.NUMERICAL,
                null_policy=NullPolicy.ZERO_BUCKET,
                description="Transaction amount in minor-unit-free decimal form.",
            ),
            FieldPolicy(
                "currency",
                FieldType.CATEGORICAL,
                null_policy=NullPolicy.FORBIDDEN,
            ),
            FieldPolicy(
                "direction",
                FieldType.CATEGORICAL,
                description="'in' or 'out' relative to the entity.",
            ),
            FieldPolicy("channel", FieldType.CATEGORICAL),
            FieldPolicy("mcc", FieldType.CATEGORICAL, description="Merchant category code."),
            FieldPolicy(
                "counterparty_id",
                FieldType.IDENTIFIER,
                regulated=True,
                description="Pseudonymous P2P counterparty identifier.",
            ),
            FieldPolicy(
                "merchant_name",
                FieldType.TEXT,
                pii_risk=True,
                description="Free-text merchant name; approved for BPE fitting.",
            ),
            FieldPolicy(
                "description",
                FieldType.TEXT,
                pii_risk=True,
                description="Free-text note; approved for BPE fitting.",
            ),
            FieldPolicy("view", FieldType.CATEGORICAL, description="In-app screen/event name."),
            FieldPolicy("source_currency", FieldType.CATEGORICAL),
        ]
        for policy in event_value_fields:
            registry.register_field(policy)

        registry.register_event_family(
            EventFamily(
                "topup",
                required_fields=("amount", "currency", "direction", "channel"),
                optional_fields=("description",),
            )
        )
        registry.register_event_family(
            EventFamily(
                "card_payment",
                required_fields=("amount", "currency", "direction", "mcc"),
                optional_fields=("description", "merchant_name"),
            )
        )
        registry.register_event_family(
            EventFamily(
                "p2p_transfer",
                required_fields=("amount", "currency", "direction", "counterparty_id"),
                optional_fields=("description",),
            )
        )
        registry.register_event_family(
            EventFamily(
                "atm_withdrawal",
                required_fields=("amount", "currency", "direction", "channel"),
            )
        )
        registry.register_event_family(
            EventFamily(
                "fx_exchange",
                required_fields=("amount", "currency", "source_currency", "direction"),
                optional_fields=("description",),
            )
        )
        registry.register_event_family(
            EventFamily(
                "direct_debit",
                required_fields=("amount", "currency", "direction"),
                optional_fields=("merchant_name", "description"),
            )
        )
        registry.register_event_family(
            EventFamily(
                "app_event",
                required_fields=("view",),
            )
        )

        profile_fields = [
            FieldPolicy("country", FieldType.CATEGORICAL, null_policy=NullPolicy.FORBIDDEN),
            FieldPolicy("plan", FieldType.CATEGORICAL, null_policy=NullPolicy.FORBIDDEN),
            FieldPolicy("kyc_level", FieldType.CATEGORICAL, null_policy=NullPolicy.FORBIDDEN),
            FieldPolicy("age_band", FieldType.CATEGORICAL),
            FieldPolicy("balance_quantile", FieldType.CATEGORICAL),
            FieldPolicy("is_active", FieldType.CATEGORICAL, null_policy=NullPolicy.FORBIDDEN),
            FieldPolicy("signup_at", FieldType.TIMESTAMP, null_policy=NullPolicy.FORBIDDEN),
            FieldPolicy("first_topup_at", FieldType.TIMESTAMP),
            FieldPolicy("first_card_payment_at", FieldType.TIMESTAMP),
            FieldPolicy("first_p2p_at", FieldType.TIMESTAMP),
        ]
        for policy in profile_fields:
            registry.register_field(policy)

        for key in ("country", "plan", "kyc_level", "age_band", "balance_quantile", "is_active"):
            registry.register_profile_field(key)
        for key in ("signup_at", "first_topup_at", "first_card_payment_at", "first_p2p_at"):
            registry.register_profile_field(key, lifelong_milestone=True)

        return registry
