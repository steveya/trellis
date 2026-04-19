# Portfolio Path And Result-Set Surface

## Status

Draft. Cross-cutting design document. Not yet an execution mirror.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- `doc/plan/draft__external-prior-art-adoption-map.md`
- `doc/plan/draft__semantic-contract-target-and-trade-envelope.md`
- `doc/plan/active__semantic-simulation-substrate.md`
- `doc/plan/draft__valuation-result-identity-and-provenance.md`
- existing repo surfaces:
  - `trellis.book.Book`
  - `trellis.book.BookResult`
  - `trellis.book.ScenarioResultCube`
  - `trellis.book.FutureValueCube`
  - `trellis.analytics.result.AnalyticsResult`
  - `trellis.analytics.result.BookAnalyticsResult`

## Purpose

Capture the useful `gs-quant` ideas around `Portfolio`,
`PortfolioPath`, and `PortfolioRiskResult` in a Trellis-native form.

As Trellis moves toward route-free pricing of arbitrary decomposable
derivatives, the result side also needs a vocabulary for nested
structure:

- books containing positions
- decomposition trees containing structural subclaims
- scenario and historical projections of those values
- multiple requested outputs over the same result tree

The current flat `BookResult` and scenario-cube surfaces are useful, but
they are only the first layer of that result model.

## Why This Matters

If representation, decomposition, and lowering close over more product
families, result structure will also become richer.

Without an explicit result-set design, Trellis risks one of two bad
outcomes:

- each workflow invents a one-off nested payload schema
- deeper structures are flattened too early, and operators lose the
  path back to the originating sub-position or subclaim

That would recreate the same observability problem that Phase 4 is
trying to solve on the route side.

## Design Objectives

The future result-set surface should be:

- path-aware
- compatible with current flat book and cube projections
- explicit about the axis represented by each path segment
- route-free
- usable for both portfolio aggregation and semantic decomposition
  outputs without pretending they are the same thing

## Core Distinctions

### 1. Portfolio path vs decomposition path

These should not collapse into one unlabeled tuple.

Examples:

- `book -> position["swap_3"]`
- `contract -> decomposition_leg[1]`
- `scenario -> shocked_parallel_up`
- `date -> 2026-06-30`

All are "paths," but they refer to different axes. The result-set
surface should preserve the axis kind, not just the key.

This also interacts with the contract-target / trade-envelope split:

- portfolio position path is not semantic contract identity
- decomposition path is not booking path
- both may need to appear on one result surface without being confused

### 2. Result tree vs cube

A result tree organizes nested ownership or composition.

A cube organizes repeated valuations across one or more axes such as:

- scenario
- date
- path
- requested output

Some projections are naturally trees, some are naturally cubes, and
some workflows need both. `gs-quant`'s result containers are useful
because they preserve path-aware navigation. Trellis should preserve
that idea without forcing every workflow into one generic object.

### 3. Path identity vs display label

Display labels are useful for notebooks and UI, but path identity must
stay stable even when names change.

That means a future path surface should prefer typed segments over
free-form strings embedded in display text.

## Candidate Surface

Exact names may change, but the useful next shape is close to:

```text
ResultPathSegment =
    { axis: str
    ; key: str | int | date
    }

ResultPath = tuple[ResultPathSegment, ...]

ValuationResultNode =
    { path: ResultPath
    ; payload: object | None
    ; metadata: dict
    ; children: tuple[ValuationResultNode, ...]
    }

ValuationResultSet =
    { root: ValuationResultNode
    ; valuation_identity: ValuationIdentityKey | None
    ; provenance: ValuationProvenance | None
    ; projections: dict
    }
```

The important architectural point is not the exact node class. It is
that:

- path segments are typed by axis
- nested results have a stable navigation surface
- result projections can preserve provenance instead of flattening it
  away

## Relationship To Existing Trellis Surfaces

### `Book` and `BookResult`

These remain valid and useful for flat position books.

This note says they should later be understandable as one projection of
a more general result-set surface, rather than the only possible result
shape.

### Contract-target and trade-envelope boundary

The result-path surface should preserve where a path segment came from:

- semantic decomposition
- trade or booking envelope
- portfolio aggregation
- scenario or historical projection

That boundary is described more directly in
`doc/plan/draft__semantic-contract-target-and-trade-envelope.md`.

### `ScenarioResultCube`

`ScenarioResultCube` already proves Trellis can publish stable
cube-like outputs with per-scenario provenance.

This note complements that surface by asking:

- what is the stable path within one cube cell?
- how do scenario/date/output projections compose with nested position or
  decomposition structure?

### `FutureValueCube`

`FutureValueCube` should remain a specialized path/date valuation cube,
not a generic replacement for every nested result type.

The likely relationship is:

- `ValuationResultSet` carries the navigable result tree and identity
- `FutureValueCube` is one specialized payload or projection within that
  broader surface

### `AnalyticsResult`

`AnalyticsResult` and `BookAnalyticsResult` remain the lightweight
leaf/book ergonomics layer.

This note is about what happens when workflows need deeper structural
navigation and a stable path identity, not about replacing the simple
front-door result wrappers.

## Non-Goals

- Do not copy `gs-quant`'s future/promise result machinery.
- Do not force decomposition trees and portfolio trees into the same
  semantic object.
- Do not replace flat result containers before there is a concrete need.
- Do not use path identity as a substitute for valuation identity or
  contract semantics.

## Ordered Follow-On Queue

### P1 — Result-path taxonomy

Objective:

Define the bounded axis vocabulary for result paths.

Acceptance:

- one typed segment shape exists
- scenario/date/position/decomposition axes are distinguishable
- downstream docs stop inventing ad hoc path encodings

### P2 — Path-aware wrappers over current book and cube outputs

Objective:

Show how `BookResult` and `ScenarioResultCube` project onto a typed
result-path surface.

Acceptance:

- at least one current result family can expose stable typed paths
- flat access remains available for ordinary workflows

### P3 — Decomposition-aware result navigation

Objective:

Support route-free decomposition outputs without flattening away the
subclaim structure.

Acceptance:

- one decomposition-based workflow publishes path-aware sub-results
- provenance can identify the originating structural subclaim

### P4 — Identity and cube integration

Objective:

Align path-aware result containers with valuation identity and later
historical/scenario cube surfaces.

Acceptance:

- one result-set path can carry `ValuationIdentityKey`
- scenario/date projections can preserve path-aware provenance
- future value and scenario cubes have a clean relationship to the path
  model

## Risks To Avoid

- **Path ambiguity.** Unlabeled tuple paths will become impossible to
  interpret once books, decompositions, scenarios, and dates mix.
- **Premature flattening.** If decomposition outputs are flattened into
  anonymous rows too early, route-free explainability will be weaker
  than route-based explainability.
- **One-object syndrome.** Not every result should be forced into one
  universal mega-container.
- **UI leakage.** Human-friendly labels are useful, but they should not
  be the authoritative identity surface.
