# ADR 0001: Canonical data schema and raw record shape

**Status:** Accepted

## Context

PRAGMA consumes heterogeneous event and profile-state tables from potentially
multiple sources. Raw field names, types, and null conventions cannot be
trusted to be consistent, and the model's key/value token scheme (section 6.4
of the implementation plan) requires every field to resolve to one stable
semantic key before it can be tokenized.

## Decision

- A `SchemaRegistry` (`src/pragma/schema/registry.py`) is the single source of
  truth for canonical semantic keys. Every raw source column must resolve to
  exactly one canonical key through a registered alias before it is used
  anywhere downstream.
- Each canonical key carries a `FieldPolicy`: value type (numerical,
  categorical, text, timestamp, identifier), null policy, MLM-masking
  eligibility, and regulated-decision eligibility — matching section 5.5.
- Raw events are grouped into `EventFamily` definitions (e.g. `card_payment`,
  `topup`, `app_event`) that declare required vs. optional canonical keys.
  Different families expose different fields; sparse/null columns in a wide
  representation are expected, not an error.
- Identifier and timestamp fields are never MLM-maskable by construction
  (enforced in `FieldPolicy.__post_init__`).
- Client/source identity is not a model feature by default. Pooling data
  across clients is a separate governance decision, not an automatic
  consequence of sharing one registry.

## Consequences

- Any new raw field must be registered (canonical key + alias + policy)
  before it can appear in an event or profile record. Unregistered fields are
  flagged by `SchemaRegistry.validate_event` / `validate_profile` rather than
  silently ignored or silently included.
- The registry is intentionally decoupled from the processor's vocabulary
  IDs (`KeyVocabulary`, Phase 2). The registry answers "is this field known
  and well-typed"; the processor answers "what token ID does this become."
