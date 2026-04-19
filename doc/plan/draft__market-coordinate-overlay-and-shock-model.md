# Market Coordinate, Overlay, And Shock Model

## Status

Draft. Cross-cutting design document. Not yet an execution mirror.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- `doc/plan/draft__quoted-observable-contract-ir-foundation.md`
- `doc/plan/draft__valuation-result-identity-and-provenance.md`
- `doc/plan/active__semantic-simulation-substrate.md`
- existing repo surfaces:
  - `trellis.core.market_state.MarketState`
  - `trellis.curves.scenario_packs`
  - `trellis.book.ScenarioResultCube`

## Purpose

Define a Trellis-native market-coordinate and market-overlay model that
supports three nearby needs with one vocabulary:

1. future quoted-observable contract nodes such as `CurveQuote` and
   `SurfaceQuote`
2. scenario and stress workflows that shock or override selected parts
   of the market state
3. route-free valuation provenance in Phase 3 and Phase 4

The design is inspired by the useful parts of `gs-quant`'s
`MarketDataCoordinate`, `MarketDataPattern`, `MarketDataShock`,
`OverlayMarket`, and `RelativeMarket`, but it must fit Trellis' local
semantic-compiler architecture rather than a remote risk-service client.

## Why This Is Its Own Surface

Trellis already has:

- `MarketState` as the valuation-facing market snapshot
- named rate scenario packs
- scenario-result cubes with per-scenario provenance
- bounded synthetic market-state shifts in mock data

What it does **not** yet have is a first-class coordinate language for
"which quoted point or quoted family of points" a workflow is talking
about.

That gap matters in at least three places:

- quoted-observable contracts need semantic leaves that reference quote
  points honestly
- scenario workflows need a reusable selector surface for shocks and
  overrides
- route-retired traces need non-route identity for the market snapshot
  and any overlay that produced the value

## Design Objectives

The market-side contract should be:

- explicit about quote identity and quote convention
- compatible with `MarketState`, not a second market universe
- reusable across pricing, scenario, and provenance surfaces
- route-free and product-free
- honest about the boundary between market identity and contract meaning

## Core Distinctions

### 1. Contract observable vs market coordinate

These are related, but not the same:

- `CurveQuote` / `SurfaceQuote` in contract IR say what the contract
  settles on
- a market coordinate says how a market snapshot identifies the quoted
  point

The same contract node may lower through different market-coordinate
mechanisms depending on market source or valuation policy. So market
coordinates belong in market and lowering surfaces, not in generic
contract algebra.

### 2. Base market vs overlay market

A market overlay is not a new contract and not a new route.

It is a valuation-state transformation of the form:

```text
M' = Overlay(M_base, overrides)
```

where the override may be:

- an explicit point replacement
- a pattern-based shock
- a named scenario-pack projection

### 3. Relative market vs overlay market

These should stay distinct:

- `MarketStateOverlay`: one concrete shocked market used for valuation
- `RelativeMarketState`: an ordered pair `(M_from, M_to)` used to define
  relative analytics, attribution, or P&L-explain style outputs

## Candidate Surface

Exact class names may change, but the first useful family should be
close to:

```text
MarketCoordinate =
    { market_type: str
    ; market_asset: str
    ; market_class: str | None
    ; point: tuple[str, ...] | None
    ; quoting_convention: str | None
    }

MarketCoordinatePattern =
    { market_type: str | Wildcard
    ; market_asset: str | Wildcard
    ; market_class: str | Wildcard
    ; point: tuple[str, ...] | Prefix | Wildcard
    ; quoting_convention: str | Wildcard
    }

MarketShock =
    | AbsoluteShock(value)
    | RelativeShock(scale)
    | Override(value)

MarketStateOverlay =
    { base_market_identity: MarketIdentity
    ; point_overrides: tuple[(MarketCoordinate, scalar_or_surface_value), ...]
    ; pattern_shocks: tuple[(MarketCoordinatePattern, MarketShock), ...]
    ; metadata: dict
    }

RelativeMarketState =
    { from_market_identity: MarketIdentity
    ; to_market_identity: MarketIdentity
    ; metadata: dict
    }
```

The important architectural point is not the exact field spelling. It
is that:

- quote-point identity is typed
- shock selection is typed
- overlays are explicit market objects
- relative-vs-overlaid valuation is explicit

## Immediate Relevance To Phase 3 And 4

Even before quoted-observable contracts land, the Phase 3 / Phase 4
compiler outputs should reserve space for:

- `market_identity`
- `market_overlay_identity`
- `resolved_market_coordinates`

That gives route-free traces a stable market-facing vocabulary today and
avoids another migration later when `CurveQuote` / `SurfaceQuote`
families arrive.

## Relationship To Existing Trellis Surfaces

### `MarketState`

This note does not replace `MarketState`.

Instead:

- `MarketState` remains the valuation-facing snapshot
- `MarketCoordinate` identifies quoted contents inside or alongside a
  market-state implementation
- `MarketStateOverlay` is the market-state transformation layer

### Scenario packs

Current named rate scenario packs should be interpretable as one special
case of:

- pattern-based market shocks
- projected onto a market overlay

That gives scenario packs and future quoted-point shocks the same market
language.

### `ScenarioResultCube`

`ScenarioResultCube` already carries per-scenario provenance. This note
does not replace it. It supplies the missing typed language for what a
"scenario" did to the market.

## Non-Goals

- Do not turn `MarketCoordinate` into a generic "query anything"
  escape hatch.
- Do not encode interpolation, smoothing, or calibration policy inside
  the coordinate object.
- Do not make market coordinates the semantic authority for contract
  meaning.
- Do not require every current Trellis helper to expose exact point
  coordinates immediately.

## Ordered Follow-On Queue

### M1 — Coordinate dataclasses and identity helpers

Objective:

Land the bounded typed surfaces for market coordinates, patterns, and
market identity.

Acceptance:

- one stable coordinate representation exists for curve/surface points
- string parsing or formatting is additive, not authoritative
- Phase 3/4 provenance surfaces can reference the type without needing
  the full shock stack

### M2 — `MarketStateOverlay` and `RelativeMarketState`

Objective:

Introduce explicit overlay and relative-market objects over the current
market-state surface.

Acceptance:

- one bounded overlay path can override or shock selected coordinates
- relative-market analytics can distinguish base-vs-shocked identity
- route-free traces can name the overlay without route aliases

### M3 — Pattern shocks and scenario-pack alignment

Objective:

Unify current named scenario packs with typed coordinate/pattern shock
semantics.

Acceptance:

- at least one existing scenario-pack family lowers onto the typed
  pattern/shock surface
- `ScenarioResultCube` can carry the typed market-overlay provenance
  instead of only string labels

### M4 — Quoted-observable lowering integration

Objective:

Make future `CurveQuote` / `SurfaceQuote` lowering consume the typed
market-coordinate surface.

Acceptance:

- quoted-observable contract nodes lower without inventing a second
  quote identity language
- lowering can distinguish contract meaning from market lookup policy

## Risks To Avoid

- **Contract/market collapse.** A market coordinate is not itself a
  contract observable.
- **Interpolation leakage.** Coordinate identity should not smuggle
  interpolation policy into contract or provenance surfaces.
- **Scenario proliferation.** Named scenario packs should compile onto
  one typed shock language rather than each inventing their own payload
  schema.
- **Fake precision.** Phase 3/4 should permit optional coordinate
  provenance; helpers that do not naturally resolve exact point-level
  reads should not fabricate them.
