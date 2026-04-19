# Semantic Contract Target And Trade Envelope

## Status

Draft. Cross-cutting design document. Not yet an execution mirror.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- `doc/plan/draft__external-prior-art-adoption-map.md`
- `doc/plan/draft__valuation-session-and-request-surface.md`
- `doc/plan/draft__valuation-result-identity-and-provenance.md`
- `doc/plan/draft__portfolio-path-and-result-set-surface.md`
- existing repo surfaces:
  - `trellis.agent.semantic_contracts.SemanticContract`
  - `trellis.agent.valuation_context.ValuationContext`
  - `trellis.book.Book`

## Purpose

Strengthen the architectural boundary between:

- semantic contract meaning
- trade / execution envelope metadata
- portfolio position or aggregation context

The immediate inspiration is Strata's separation of `Product` and
`Trade`, but the Trellis goal is narrower and more semantic:

- the compiler should know what contract is being priced
- it should also know what surrounding trade or portfolio metadata
  exists
- but it must not let trade metadata become a shadow route selector

## Why This Matters

Route retirement removes one old source of accidental authority:

- instrument name
- route id
- route family

If Trellis does not replace that with a clear boundary, the old product
split can simply reappear in other shapes:

- "trade type" wrappers
- desk tags
- position metadata
- counterparty or booking envelopes

That would preserve the old architecture under a new name.

## Core Distinctions

### 1. Contract meaning vs trade metadata

The semantic contract says what was agreed economically.

Trade metadata says things such as:

- trade date
- booking desk
- counterparty
- external identifiers
- sales / workflow annotations

Those matter operationally, but they are not the contract's semantic
payoff authority.

### 2. Contract meaning vs position scaling

The semantic contract is not the same thing as:

- quantity
- holdings sign
- booked premium
- package-level allocation

Position context may scale or aggregate a contract, but it should not
change the contract-family identity that the compiler matches.

### 3. Valuation target vs valuation context

The thing being priced and the policy under which it is priced are
distinct:

- the **target** is the semantic contract plus any explicit envelope
  needed for bookkeeping or aggregation
- the **valuation context** is the model / measure / discounting /
  reporting choice

This keeps the compiler contract factored cleanly.

## Candidate Surface

Exact names may change, but the useful next shape is close to:

```text
ContractTarget =
    { contract: SemanticContract | ContractIR
    ; trade_envelope: TradeEnvelope | None
    ; position_context: PositionContext | None
    }

TradeEnvelope =
    { trade_date: date | None
    ; counterparty: str | None
    ; identifiers: dict
    ; desk_tags: dict
    ; metadata: dict
    }

PositionContext =
    { quantity: float | None
    ; scaling_notional: float | None
    ; portfolio_path: ResultPath | None
    ; metadata: dict
    }
```

The important architectural point is not the exact dataclass list. It
is that Trellis should have a named place for trade and position
metadata that is not the semantic contract itself.

## Compiler Rule

For migrated fresh-build surface, solver selection should be a function
of:

- semantic contract structure
- normalized generic term environment
- valuation context
- market capabilities

and not of opaque trade-envelope labels.

Trade or position metadata may still affect:

- output naming
- aggregation
- reporting
- booked-premium or fee handling if those are modeled explicitly

but they should not silently choose a different structural declaration.

## Relationship To Existing Trellis Surfaces

### `SemanticContract`

`SemanticContract` remains the contract-meaning surface.

This note is about what should wrap around it, not about replacing its
semantic job.

### `Book` and result sets

`Book` and later result-set surfaces are natural homes for
`PositionContext` or path metadata, not for semantic contract
definition.

### Phase 3 / Phase 4

Phase 3 should reserve this boundary so that declaration selection does
not quietly drift onto trade metadata.

Phase 4 should enforce that route retirement removes product authority
without accidentally transferring it to trade-envelope fields.

## Non-Goals

- Do not require every Trellis user-facing API to expose a trade
  envelope immediately.
- Do not invent a new large object hierarchy for trades or positions.
- Do not let trade metadata bypass semantic normalization.
- Do not turn portfolio wrappers into semantic product classifiers.

## Ordered Follow-On Queue

### T1 — Document the contract / trade / position split

Objective:

Make the separation explicit in the semantic-contract docs and planning
surfaces.

Acceptance:

- one stable note exists for later implementers
- compiler docs no longer blur semantic contract and trade wrapper

### T2 — Introduce a thin trade envelope where operationally useful

Objective:

Add a bounded wrapper for trade metadata without altering contract
authority.

Acceptance:

- at least one workflow can carry external ids or trade date separately
  from contract meaning
- route-free selection remains invariant under envelope-only changes

### T3 — Position context and result-path alignment

Objective:

Line up position metadata with the later path-aware result-set model.

Acceptance:

- portfolio path or quantity context can be attached without changing
  semantic family selection
- result explain surfaces can reference position context cleanly

## Risks To Avoid

- **Trade-envelope relapse.** If selection falls back to trade type or
  desk tags, route retirement has failed in substance.
- **Contract pollution.** If external ids or booking metadata get pushed
  into the semantic contract, the contract surface becomes less reusable
  and less canonical.
- **Position confusion.** Quantity or aggregation metadata should not be
  mistaken for contract meaning.
