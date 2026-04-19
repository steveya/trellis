# Valuation Session And Requested-Output Surface

## Status

Draft. Cross-cutting design document. Not yet an execution mirror.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- `doc/plan/draft__external-prior-art-adoption-map.md`
- `doc/plan/draft__semantic-contract-target-and-trade-envelope.md`
- `doc/plan/draft__market-coordinate-overlay-and-shock-model.md`
- `doc/plan/draft__valuation-result-identity-and-provenance.md`
- existing repo surfaces:
  - `trellis.agent.valuation_context.ValuationContext`
  - `trellis.core.types.DslMeasure`
  - `trellis.analytics.result.RiskMeasureOutput`
  - `docs/quant/pricing_stack.rst`
  - `docs/user_guide/session.rst`

## Purpose

Capture the useful `gs-quant` ideas around `PricingContext`,
`HistoricalPricingContext`, and structured `RiskMeasure` objects, but in
a Trellis-native form that respects the semantic compiler architecture.

The goal is not to copy `gs-quant`'s remote-pricing client model. The
goal is to make Trellis' valuation request surface more explicit and
more stable as route authority disappears:

- valuation policy should remain typed and inspectable
- requested outputs should remain normalized and composable
- historical sweeps and execution hints should be explicit rather than
  hidden in ambient session state

## Why This Is Separate From Phase 3 And Phase 4

Phase 3 and Phase 4 need one thing immediately: a stable valuation
identity and provenance packet for the value that was produced.

They do **not** need, as a hard prerequisite, a full redesign of the
user-facing valuation-session API.

So this note is intentionally adjacent rather than blocking:

- Phase 3/4 reserve the identity boundary now
- this note captures the later control-plane and ergonomics layer

That avoids overloading the route-retirement tranche with API polish
work while still preserving the good design ideas.

## Design Objectives

The eventual valuation-session surface should be:

- explicit about valuation policy vs execution policy
- additive on top of `ValuationContext`, not a second authority object
- explicit about historical date sweeps vs market overlays vs scenario
  comparisons
- compatible with the existing `DslMeasure` vocabulary, but able to
  carry richer output options later
- independent of route ids, provider ids, and remote-service-only flags

## Core Distinctions

### 1. Valuation policy vs execution policy

Trellis already has the beginnings of this split in `ValuationContext`.
That boundary should become sharper:

- valuation policy: model, measure, discounting, collateral, reporting,
  requested outputs, market selection
- execution policy: batching, caching, concurrency, timeout, notebook/UI
  preferences

Execution hints may influence how the run happens, but they are not part
of the mathematical or compiler contract.

### 2. Valuation identity vs session convenience

Session-scoped defaults are useful, but they must not become the only
place where valuation meaning lives.

The compiler and result identity layers should continue to depend on
explicit resolved values, not on ambient mutable session state. In
particular:

- `ValuationIdentityKey` identifies one resolved valuation
- session config is only a convenience surface that resolves into that
  identity

### 3. Historical sweep vs scenario overlay

These are different operations:

- **historical sweep**: evaluate across a sequence of valuation dates or
  market snapshots
- **scenario overlay**: apply a typed shock or override to a base market
  identity

`gs-quant` distinguishes historical contexts from shocked markets. We
should keep that distinction too, so later cube/result surfaces do not
collapse "time-series valuation" and "scenario valuation" into one
ambiguous axis.

### 4. Requested output identity vs result payload

Requested outputs are part of valuation identity, but they are not the
same thing as the returned result container.

For example:

- `price`
- `dv01`
- `vega`
- `future_value_cube`
- `scenario_pnl`

belong to the request side as output identities or output specs, while
the corresponding returned objects may be scalars, surfaces, cubes, or
structured payloads.

### 5. Requested output support vs market requirements

Another useful Strata-style distinction is:

- a request names the outputs it wants
- a calculation surface names the outputs it supports
- the supported outputs imply a market-requirement footprint

Trellis should preserve that split explicitly. Otherwise one opaque
"request spec" object ends up doing too many jobs at once:

- naming measures
- proving support
- planning market data
- choosing a solver

Those should stay factored.

## Candidate Surface

Exact names may change, but the next useful shape is close to:

```text
ValuationSessionConfig =
    { default_context: ValuationContext
    ; execution_hints: ExecutionHints
    ; historical_grid: HistoricalValuationGrid | None
    ; metadata: dict
    }

ExecutionHints =
    { batch_mode: str | None
    ; timeout_ms: int | None
    ; cache_policy: str | None
    ; async_mode: bool | None
    }

HistoricalValuationGrid =
    { dates: tuple[date, ...] | DateRule
    ; market_roll_policy: str
    ; metadata: dict
    }

RequestedOutputSpec =
    { output_id: str
    ; canonical_measure: DslMeasure | None
    ; options: dict
    ; aggregation: str | None
    ; presentation: str | None
    }
```

The important architectural point is not the exact field spelling. It
is that Trellis should eventually distinguish:

- resolved valuation policy
- session defaults and execution hints
- historical valuation grids
- typed requested-output specs

and should be able to project market requirements from supported output
families without letting requested-output names become route selectors.

without turning any one of those into an opaque catch-all object.

## Relationship To Existing Trellis Surfaces

### `ValuationContext`

`ValuationContext` should remain the authoritative valuation-policy
object for compiler work.

This note is about later layering:

- session config resolves into `ValuationContext`
- it does not replace or bypass `ValuationContext`

### `DslMeasure`

`DslMeasure` already gives Trellis a canonical measure vocabulary. That
is a strong base.

The likely evolution is:

- keep `DslMeasure` as the canonical scalar/risk measure identity where
  it fits
- add richer `RequestedOutputSpec` wrappers only when output families
  need options or projections that plain strings cannot carry honestly

### Trade-envelope boundary

This note should also be read with
`doc/plan/draft__semantic-contract-target-and-trade-envelope.md`.

Session config and requested-output specs live on the valuation-policy
side. Trade-envelope and position metadata live on the target side.
Keeping those separate prevents the request surface from becoming a
second product-classification mechanism.

### `RiskMeasureOutput`

`RiskMeasureOutput` is a leaf-level result payload with metadata. This
note does not replace it.

Instead, a richer request surface should make it clearer which metadata
belongs to:

- the request
- the valuation identity
- the returned payload

## Non-Goals

- Do not copy `gs-quant`'s remote provider abstraction or futures API.
- Do not put route or helper selection into session config.
- Do not replace every current `requested_outputs` string immediately.
- Do not make execution hints part of semantic contract identity.

## Ordered Follow-On Queue

### S1 — Document the current `ValuationContext` boundary

Objective:

Make the existing valuation-policy object explicit in docs as the
compiler-facing authority, separate from any future session wrapper.

Acceptance:

- one short design note or doc patch explains what belongs in
  `ValuationContext`
- route/compiler work does not need to guess where policy lives

### S2 — Typed requested-output spec over canonical measures

Objective:

Introduce a bounded typed output-spec surface only where plain measure
strings are no longer honest enough.

Acceptance:

- existing `DslMeasure` strings remain valid
- richer outputs can express options without ad hoc payloads
- requested-output identity can serialize cleanly into valuation
  provenance

### S2.5 — Supported-output and market-requirement projection

Objective:

Make it explicit which output families imply which market-data
requirements, without collapsing support, request, and selection into
one object.

Acceptance:

- one bounded planning surface can answer "what market requirements does
  this supported output set induce?"
- requested outputs remain distinct from structural declaration
  authority

### S3 — Historical valuation grid and time-axis semantics

Objective:

Define a local, route-free surface for historical valuation sweeps.

Acceptance:

- historical sweeps are distinct from scenario overlays
- result identity can name the date/grid semantics explicitly
- future cube/result containers can project historical outputs

### S4 — Optional execution hints layer

Objective:

Add bounded local execution hints without polluting valuation semantics.

Acceptance:

- cache/batch/async hints are explicit and typed
- compiler selection and valuation identity do not depend on those hints

## Risks To Avoid

- **Ambient-state drift.** If the meaning of a valuation depends on an
  implicit mutable context, route retirement will lose auditability.
- **Request/result collapse.** Requested output identity should not be
  inferred only from the shape of the returned object.
- **Execution leakage.** Cache or batching controls should not become
  part of pricing semantics.
- **Over-abstraction.** Trellis does not need a provider-agnostic remote
  risk-client framework just because `gs-quant` has one.
