# Quoted-Observable Contract IR — Future Track Foundation

## Status

Draft. Parking-lot design document. Not yet an execution mirror and not
yet tied to a filed Linear child issue.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-904 — Phase 2 umbrella for payoff-expression Contract IR
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/draft__contract-ir-phase-2-ast-foundation.md`
- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`
- `doc/plan/draft__contract-ir-phase-4-route-retirement.md`
- `doc/plan/draft__market-coordinate-overlay-and-shock-model.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `docs/quant/contract_algebra.rst`

## Purpose

Capture the future track for products whose payoff depends on explicit
quoted market points observed at one or more observation surfaces.

Examples include:

- vol-skew products paying a function of two implied-vol surface points
- terminal curve-spread products paying a function of two par-rate or
  zero-rate quote points
- options whose underlier is itself a quoted spread, skew, or surface
  differential

The goal is to extend the payoff-expression semantic program without
polluting it with route-local product buckets, and without collapsing
quoted snapshot products into the leg-based cashflow track.

## Why This Is A Separate Track

The current payoff-expression `ContractIR` already carries semantic
observables such as `Spot`, `Forward`, `SwapRate`, `Annuity`, and
`VarianceObservable`.

Those are *not* generic market-map lookup nodes:

- `SwapRate(schedule)` is a semantic quantity tied to a contract-defined
  schedule
- `VarianceObservable(interval)` is a contract-defined variance
  quantity, even if one future lowering uses static option replication

Quoted-observable products need a different kind of leaf:

- a direct reference to a quoted point on a market object
- with explicit coordinates and explicit quote convention

That is semantically distinct from all of:

- payoff-expression observables defined by financial identity
- leg-based contracts defined by coupon / accrual / payment rules
- event/state/control programs that wrap static semantic bases with
  running state or stopping logic

## Design Objective

The quoted-observable track should extend the payoff-expression `ContractIR`
so Trellis can represent snapshot-style quote products in a way that is:

- additive alongside the current payoff-expression nodes
- explicit about quote-map identity, coordinates, and quote convention
- independent of route ids, instrument strings, and backend-binding ids
- composable inside the existing payoff algebra
- honest about what is contractual quote meaning versus what is
  interpolation, calibration, or pricing method

## Non-Goals For The First Slice

- Do not widen `VarianceObservable` into a generic market-query escape
  hatch.
- Do not encode interpolation rules, smile fitting choices, or
  replication formulas directly in quote nodes.
- Do not collapse scheduled coupon products into this track just because
  their desk label contains words such as "basis" or "spread."
- Do not treat quote-linked coupon notes as snapshot quote products when
  the contract really has scheduled cashflows, interruption logic, or
  callability.
- Do not invent product-keyed nodes such as `VolSkewSwap` or
  `CurveSteepenerOption` when the payoff is structurally a function of
  quote points.

## Semantic Requirements

Any serious quoted-observable extension must make the following surfaces
explicit:

1. **Quoted object identity.** Which curve, surface, or quote map is
   referenced.
2. **Coordinate system.** Tenor, expiry, strike, delta, maturity, or
   another explicit coordinate basis.
3. **Quote convention.** For example: Black vol, normal vol, par rate,
   zero rate, spread in basis points, log-moneyness strike, delta
   convention.
4. **Observation-time semantics.** The contract root still determines
   *when* the quote is observed; the quote node determines *which point*
   is read from the map at that observation.
5. **Units and scaling.** Percent vs absolute vol, decimal vs basis
   points, and any contractual scaling must be explicit somewhere in the
   semantic representation, not guessed by route.

If any of those are hidden in an opaque leaf or helper-specific
payload, the track will not support route-free fresh builds.

## Candidate Surface

This document does not lock the final ADT, but the first useful shape
looks like explicit quoted-observable leaves inside the existing
payoff-expression algebra.

Pseudo-ADT sketch:

```text
PayoffExpr =
    ...
    | CurveQuote(curve_id: str, coordinate: CurveCoordinate, convention: CurveQuoteConvention)
    | SurfaceQuote(surface_id: str, coordinate: SurfaceCoordinate, convention: SurfaceQuoteConvention)

CurveCoordinate =
    | ParRateTenor(tenor: str)
    | ZeroRateTenor(tenor: str)
    | ForwardRateInterval(start_tenor: str, end_tenor: str)

SurfaceCoordinate =
    | VolPoint(option_tenor: str, strike: float, strike_style: str)
    | VolDeltaPoint(option_tenor: str, delta: float, delta_style: str)
```

The contract root's observation schedule says when the quote is
observed. The quote leaf says which market-map coordinate is observed at
that time.

The important architectural point is not the exact field list. It is
that quote coordinates and quote conventions are semantic structure, not
buried in route-local adapters.

Those contract-side coordinates should later lower onto the market-side
coordinate language described in
`doc/plan/draft__market-coordinate-overlay-and-shock-model.md`.
Contract coordinates say what the product settles on. Market
coordinates say how a market snapshot identifies or shocks that quoted
point. The two must align, but they are not the same layer.

## Examples

### Example 1 — Terminal vol-skew payoff

At observation time `T`, pay
`N * (σ_{1Y,90%}(T) - σ_{1Y,110%}(T))`.

Possible shape:

```text
Scaled(
    Constant(N),
    Sub(
        SurfaceQuote("SPX_IV", VolPoint("1Y", 0.90, "moneyness"), "black_vol"),
        SurfaceQuote("SPX_IV", VolPoint("1Y", 1.10, "moneyness"), "black_vol"),
    ),
)
```

This is a quoted-observable product because it settles on explicit
surface points at one observation surface.

### Example 2 — Terminal 10Y-2Y curve-spread option

At observation time `T`, pay
`max(S_{10Y}(T) - S_{2Y}(T) - K, 0)` where `S_tau` is a quoted par swap
rate for tenor `tau`.

Possible shape:

```text
Max(
    Sub(
        Sub(
            CurveQuote("USD_SWAP", ParRateTenor("10Y"), "par_rate"),
            CurveQuote("USD_SWAP", ParRateTenor("2Y"), "par_rate"),
        ),
        Strike(K),
    ),
    Constant(0),
)
```

This is still payoff-expression semantics, not leg-based cashflow
semantics.

### Example 3 — Boundary against basis swaps

"10Y-2Y basis" is not enough to classify the product.

- if the contract settles once on a terminal quote spread, it belongs
  here
- if the contract exchanges scheduled coupons across floating legs, it
  belongs to the leg-based IR track

The desk nickname is not the semantic authority.

### Example 4 — Boundary against quote-linked coupon notes

A callable CMS-spread range-accrual note is also not a quoted-observable
contract in this narrow sense.

Even if the coupon references explicit quote points, the contract still
contains:

- a scheduled coupon program
- possible in-range counting or interruption state
- call dates and issuer choice

So the quote coordinates belong here, but the full product belongs to a
later combination of:

- static leg semantics
- quoted-observable leaves
- the event/state/control wrapper

## Closure Requirements

This track should explicitly satisfy all three semantic closures from
`draft__semantic-contract-closure-program.md`.

### Representation closure

Quoted products are representationally closed only when:

- quote-map identity, coordinates, and quote convention are explicit
- no product-keyed node or adapter blob is required to know which quote
  is being referenced
- interpolation or calibration policy is not smuggled into the contract
  node

### Decomposition closure

Quoted products are decomposition-closed only when:

- the decomposer can classify snapshot quote products against the
  leg-based and event/state/control tracks deterministically
- semantically equivalent requests normalize to the same canonical quote
  nodes
- route ids and instrument strings are not needed to emit those nodes

### Lowering closure

Quoted products are lowering-closed only when:

- declarations can consume the quote nodes plus generic term
  environment, valuation context, and market capabilities
- quote lookup, interpolation, and calibration policy live in lowering
  or valuation layers rather than in product-local route logic
- parity or validation evidence exists for the selected checked helper
  or checked assembly path

## Dependency On Phases 3 And 4

This track should reuse the Phase 3 / Phase 4 authority model, not
invent a separate dispatch regime.

Concretely:

- future quoted-observable declarations should be selected from
  structural IR plus valuation / market surfaces
- fresh builds for migrated quoted-observable families should retire
  route-local product authority in the same way as payoff-expression,
  leg-based, and future event/state/control families

## First Implementable Slice

When this track is eventually promoted from parking lot to active work,
the first useful scope should stay narrow:

1. Terminal linear quote-spread payoffs on curve quotes
2. Terminal linear skew / spread payoffs on surface quotes
3. Options on those quoted spreads only when a checked lowering surface
   already exists or can be assembled transparently from existing
   checked primitives

Deferred from that first slice:

- path-dependent quoted surfaces
- multi-snapshot realized quote products
- quote-coupled coupon products that naturally belong in the combined
  static-leg plus event/state/control track
- generic "query any market object" escape hatches

## Relationship To The Dynamic Track

This quoted-observable note owns the semantic meaning of explicit quote
points, not the whole dynamic contract whenever those quote points are
used inside a scheduled coupon or event-driven structure.

So a future dynamic family may legitimately reuse these nodes inside:

- coupon formulas
- trigger conditions
- settlement expressions

without changing the ownership boundary of this document. This track
defines "what quote point is observed," while the dynamic track defines
"how the contract evolves through time around that observation."

## Pricing Boundary

This track should follow the same discipline as the rest of the
semantic-contract program:

- contract nodes represent contractual quote meaning
- lowerings represent quote retrieval, interpolation, calibration, and
  pricing method

Examples:

- `SurfaceQuote(..., convention="black_vol")` belongs in the contract
  when the contract explicitly settles on Black vol
- spline choice, interpolation stencil, or arbitrage-cleaning policy do
  not belong in the contract node
- a future static-replication or local-vol lowering is not part of the
  node's name or semantics

## Risks To Avoid

- **Generic market-query escape hatches.** A node like
  `MarketValue(kind="anything")` would just recreate route authority in
  a less reviewable form.
- **Method leakage.** A name such as `VolReplicationPoint` would smuggle
  a pricing method into the contract boundary.
- **Interpolation leakage.** Off-grid quote handling is a lowering or
  valuation-policy question, not a contract-node field.
- **Boundary confusion with leg products.** Terminal quote-spread
  products and scheduled basis swaps may share desk language but do not
  share the same semantics.

## Next Steps

1. Keep this document as the parking-lot spec for the quoted-observable
   track while the payoff-expression Phase 3 / Phase 4 work lands.
2. Use this document and the leg-based companion doc together when
   classifying future "spread" or "basis" products.
3. Use the event/state/control companion doc when a quote-linked
   product also has scheduled coupons, state, or callability.
4. File a future Linear child issue under QUA-887 for the first active
   quoted-observable slice.
