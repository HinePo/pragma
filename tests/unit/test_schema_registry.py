import pytest

from pragma.schema import EventFamily, FieldPolicy, FieldType, NullPolicy, SchemaRegistry


def test_default_registry_resolves_aliases() -> None:
    registry = SchemaRegistry.default()
    assert registry.resolve("created_at") == "created_at"
    assert registry.resolve("event_time") == "created_at"
    assert registry.resolve("nonexistent_field") is None


def test_default_registry_validates_conforming_event() -> None:
    registry = SchemaRegistry.default()
    issues = registry.validate_event(
        "card_payment",
        {"amount": 12.5, "currency": "GBP", "direction": "out", "mcc": "5411"},
    )
    assert issues == []


def test_validate_event_flags_missing_required_field() -> None:
    registry = SchemaRegistry.default()
    issues = registry.validate_event(
        "card_payment",
        {"amount": 12.5, "direction": "out", "mcc": "5411"},
    )
    assert any("currency" in issue for issue in issues)


def test_validate_event_flags_unknown_family() -> None:
    registry = SchemaRegistry.default()
    issues = registry.validate_event("not_a_real_family", {})
    assert len(issues) == 1
    assert "unknown event family" in issues[0]


def test_validate_event_flags_field_not_allowed_on_family() -> None:
    registry = SchemaRegistry.default()
    issues = registry.validate_event(
        "app_event",
        {"view": "home", "amount": 5.0},
    )
    assert any("amount" in issue for issue in issues)


def test_identifier_and_timestamp_fields_are_never_maskable() -> None:
    registry = SchemaRegistry.default()
    assert registry.get_field("entity_id").maskable is False
    assert registry.get_field("created_at").maskable is False


def test_duplicate_field_registration_raises() -> None:
    registry = SchemaRegistry()
    registry.register_field(FieldPolicy("amount", FieldType.NUMERICAL))
    with pytest.raises(ValueError):
        registry.register_field(FieldPolicy("amount", FieldType.NUMERICAL))


def test_duplicate_alias_raises() -> None:
    registry = SchemaRegistry()
    registry.register_field(FieldPolicy("amount", FieldType.NUMERICAL, aliases=("value",)))
    with pytest.raises(ValueError):
        registry.register_field(FieldPolicy("total", FieldType.NUMERICAL, aliases=("value",)))


def test_event_family_referencing_unknown_field_raises() -> None:
    registry = SchemaRegistry()
    registry.register_field(FieldPolicy("amount", FieldType.NUMERICAL))
    with pytest.raises(ValueError):
        registry.register_event_family(EventFamily("topup", required_fields=("amount", "currency")))


def test_forbidden_null_flagged_on_profile() -> None:
    registry = SchemaRegistry.default()
    issues = registry.validate_profile({"country": None, "plan": "standard", "kyc_level": "full"})
    assert any("country" in issue for issue in issues)


def test_null_policy_enum_values_are_stable_strings() -> None:
    assert NullPolicy.FORBIDDEN == "forbidden"
    assert NullPolicy.ZERO_BUCKET == "zero_bucket"
