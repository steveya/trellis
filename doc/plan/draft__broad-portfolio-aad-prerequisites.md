# Broad Portfolio AAD Prerequisites

## Status

Execution mirror for `QUA-1011`, the first AAD substrate epic under the broader
portfolio-AAD program, and `QUA-1019`, the first second-product-family adapter
epic. The child tickets below were created from this plan and implemented as
local commits on their epic branches before the final PR.

| Ticket | Scope | Status |
|---|---|---|
| `QUA-1012` | Canonical `RiskFactorId`, `RiskFactorCoordinate`, and `SparseRiskVector` primitives | Done |
| `QUA-1013` | Market-object coordinate registry for yield curves plus discovery-only credit, volatility, and model-parameter coordinates | Done |
| `QUA-1014` | `PortfolioAADRequest`, `PortfolioAADResult`, and `UnsupportedAADPosition` support contract | Done |
| `QUA-1015` | Migrate the existing shared-curve bond-book VJP lane onto factorized metadata | Done |
| `QUA-1016` | Trade-AAD adapter protocol and default unsupported-position policy | Done |
| `QUA-1017` | Aggregation, selected-factor filtering, and finite-difference verification | Done |
| `QUA-1018` | Session reporting, docs, limitations, and closeout validation | Done |
| `QUA-1020` | Reusable `RiskAggregationMap` bucket aggregation payloads | Done |
| `QUA-1021` | Concrete bond-curve AAD adapter ownership | Done |
| `QUA-1022` | Vanilla equity flat-vol portfolio-AAD adapter and public book entrypoint | Done |
| `QUA-1023` | Independent finite-difference verification for option flat-vol AAD | Done |
| `QUA-1024` | Second-product-family docs, limitations, plan mirror, and PR gate | Done |

This document still describes the prerequisites for broad portfolio AAD. The
first prerequisite, stable factor identity plus typed result contracts, is now
implemented. Broad portfolio AAD remains open until product-family adapters,
hybrid factor graphs, scenario/bucket aggregation, and scale validation exist.

## Current Shipped Boundary

Trellis currently has bounded book-level reverse-mode lanes:

- `trellis.book.portfolio_aad_curve_risk(...)`
- supported bond positions on a shared `YieldCurve`
- `trellis.book.portfolio_aad_equity_option_vol_risk(...)`
- supported European vanilla call/put specs on one shared `FlatVol`
- canonical `RiskFactorId` coordinates and sparse risk vectors in metadata
- typed `PortfolioAADRequest` / `PortfolioAADResult` payloads
- `RiskAggregationMap` bucket maps and bucket totals for factor reporting
- unsupported positions excluded and reported in metadata
- runtime reporting integrated through the derivative-method taxonomy
- backend VJP support is checked and executable

This is useful, but it is not broad portfolio AAD. It does not yet cover
general asset classes, grid-vol option surfaces, early-exercise or
path-dependent option products, hybrid objects, large mixed books, or a full
risk aggregation graph.

## Missing Gap Before Implementation

Broad portfolio AAD cannot be built until Trellis has a unified risk factor
identity and coordinate registry.

The first missing prerequisite is now implemented:

`RiskFactorRegistry`: a stable mapping from market objects and model
parameters to differentiable coordinates, factor IDs, aggregation buckets, and
reporting labels.

Without stable factor identity, AAD results cannot be safely aggregated across
trades. Two trades may refer to the same curve node, surface bucket, credit
hazard, or correlation parameter using different object paths. Conversely, two
objects may share a name while representing different curve dates, currencies,
tenors, or provenance. Broad AAD needs identity before it needs more
derivative code.

## Mathematical Contract

For a book with positions `i = 1..N`:

```text
B(theta) = sum_i q_i V_i(theta)
```

Where:

- `q_i` is the signed notional, quantity, or exposure scale
- `V_i(theta)` is the value of trade `i`
- `theta` is the global vector of market and model risk coordinates

The portfolio gradient is:

```text
grad B(theta) = sum_i q_i grad V_i(theta)
```

For vector-valued trade outputs:

```text
y_i = F_i(theta)
L_i = w_i^T y_i
grad L_i = J_i(theta)^T w_i
```

That is a VJP:

```text
VJP_F_i(theta, w_i) = J_i(theta)^T w_i
```

Broad AAD should avoid materializing dense Jacobians unless required. The
preferred representation is sparse:

```text
RiskVector = {
  RiskFactorId -> sensitivity
}
```

Aggregation is:

```text
RiskBook[f] = sum_i q_i RiskTrade_i[f]
```

Bucket reporting is a linear map:

```text
bucket_risk = A^T grad B
```

Where `A` maps low-level factors to reporting buckets such as currency, tenor,
surface expiry, strike, credit name, tranche point, or correlation key.

Second-order support should be directional first:

```text
HVP_B(theta, v) = H_B(theta) v
```

Full Hessians are not a broad portfolio target unless the factor set is tiny.

## Computational Contract

Required core objects:

- `RiskFactorId`: canonical factor identity
- `RiskFactorCoordinate`: object path, date, tenor, strike, currency, name,
  scenario axis, and coordinate transform
- `RiskFactorRegistry`: maps `MarketState` objects to coordinates
- `TradeAADAdapter`: declares supported derivative lane for one trade family
- `PortfolioAADRequest`: book, valuation context, factor selection, method
  policy, unsupported-position policy
- `PortfolioAADResult`: portfolio value, sparse risk vector, unsupported
  positions, method metadata, diagnostics
- `RiskAggregationMap`: maps low-level factors to reporting buckets

`QUA-1011` delivered the identity, request/result, adapter-protocol, and
verification substrate. `QUA-1019` added `RiskAggregationMap`, concrete
bond-adapter ownership, and the first non-bond product-family adapter for
flat-vol vanilla equity option books.

The first broad architecture should be adapter-based:

```text
TradeAADAdapter:
  supports(trade, market_state, request) -> SupportDecision
  factor_dependencies(trade, market_state) -> set[RiskFactorId]
  value(theta_subset) -> scalar or vector
  vjp(weight) -> sparse risk vector
```

Unsupported products must be explicit:

```text
unsupported_position = {
  trade_id,
  reason,
  requested_factors,
  fallback_method,
  included_in_value,
  included_in_risk
}
```

The default policy should be:

- include unsupported trades in value if pricing is available
- exclude unsupported trades from AAD risk
- optionally compute bump fallback only when requested
- always report the unsupported set

## Factor Identity Requirements

A factor ID must be stable and unambiguous. Examples:

```text
curve:USD:usd_ois:zero_rate:2027-11-15
curve:USD:usd_sofr_3m:zero_rate:5Y
vol_surface:SPX:spx_surface_authority:total_variance:2027-06-15:4000
credit_curve:ACME:hazard:2029-06-20
correlation:EURUSD:EUR_underlier:scalar
model_parameter:heston_equity:kappa
```

The ID must distinguish:

- object type
- currency or issuer
- curve/surface/parameter-set name
- coordinate type
- node date, expiry, tenor, strike, tranche point, or correlation key
- provenance or scenario namespace when needed

Risk factor IDs should not depend on Python object identity.

## Product Coverage Gating

Broad portfolio AAD should widen by product family, not by a single generic
adapter.

Suggested order:

1. fixed-rate and floating-rate bond books on shared curves
2. vanilla rate derivatives using public curve coordinates
3. equity/FX vanilla options on flat or grid vol coordinates
4. credit single-name instruments on credit-curve coordinates
5. calibrated object risk for curve and surface materializations
6. bounded hybrid quanto routes after hybrid factor graph support exists
7. path-dependent products only with explicit discontinuity policy

Each product family must define:

- value inclusion policy
- supported factors
- derivative method
- fallback method
- validation reference
- runtime metadata payload

## Required Validation Before Broad Claims

The broad AAD program should not claim support until it has:

- factor identity tests across two trades sharing the same market object
- aggregation tests proving shared factors add correctly
- unsupported-position tests
- finite-difference parity on representative books
- VJP parity against dense Jacobian on small books
- scenario/bucket aggregation tests
- serialization tests for risk results
- latency benchmarks against bump/reprice on a bounded book

Minimum mathematical validation:

```text
AADRisk[f] ~= (B(theta + eps e_f) - B(theta - eps e_f)) / (2 eps)
```

for selected smooth factors.

Minimum aggregation validation:

```text
Risk(BookA + BookB) = Risk(BookA) + Risk(BookB)
```

when the books share the same `MarketState` and factor registry.

## Implementation Phases

### Phase 1: Risk Factor Registry

Add canonical factor IDs and coordinate extraction for curves, credit curves,
vol surfaces, scalar model parameters, and scalar correlations.

Deliverables:

- registry module
- stable ID tests
- `MarketState` factor discovery tests

Status: delivered in `QUA-1012` and `QUA-1013` for canonical identity,
sparse vectors, supported yield-curve nodes, and discovery-only credit,
volatility, and scalar model-parameter coordinates. Scalar correlation
coordinates remain part of the hybrid-factor follow-on.

### Phase 2: Portfolio AAD Request And Result Schema

Define public request/result dataclasses and metadata conventions. Keep
existing `portfolio_aad_curve_risk(...)` working.

Deliverables:

- request/result types
- serialization payloads
- unsupported-position schema
- derivative-method payload integration

Status: delivered in `QUA-1014`, `QUA-1015`, and `QUA-1018`. The existing
tenor-keyed result remains compatible, and the metadata now carries
`risk_factor_coordinates`, `sparse_risk_vector`, and a serialized
`portfolio_aad_result`.

### Phase 3: Adapter Migration For Existing Bond Lane

Move the current supported bond-book lane behind the adapter interface without
changing output.

Deliverables:

- parity tests
- risk report compatibility tests
- finite-difference comparison

Status: delivered. `QUA-1016` added the protocol and unsupported policy,
`QUA-1015` migrated the current bond-book output to the factor registry, and
`QUA-1021` added a concrete `BondCurveAADAdapter` /
`BondCurveAADMarketContext` ownership surface for the existing bond lane.

### Phase 4: Second Product Family

Add one new product family with a small smooth book. A good first target is a
vanilla option book on a shared flat-vol or grid-vol surface.

Status: delivered for the first bounded second-family slice. `QUA-1022` added
`VanillaEquityOptionVolAADAdapter` /
`VanillaEquityOptionVolAADMarketContext` plus
`portfolio_aad_equity_option_vol_risk(...)` for European call/put specs on one
shared `FlatVol`. `QUA-1023` verifies the sparse flat-vol VJP against an
independent central finite-difference Black76 bump/reprice reference.

Deliverables:

- product adapter
- factor dependency tests
- aggregation and unsupported-position tests
- finite-difference verification

### Phase 5: Runtime And Benchmark Integration

Expose broad AAD through `Session.risk_report(...)` only for supported books.
Add latency and coverage reporting.

Status: partially delivered for session reporting of the existing bounded
bond-book lane, direct typed result exposure for the bounded flat-vol option
lane, and documentation of the current support boundary. Broad runtime
exposure, grid/surface support, and latency benchmarking remain open.

Deliverables:

- benchmark artifact
- docs in `docs/quant/differentiable_pricing.rst`
- support matrix row updates

## Explicit Non-Goals

- Do not claim universal portfolio AAD from the bounded bond-book or flat-vol
  option lanes.
- Do not silently bump unsupported trades inside an AAD result.
- Do not aggregate risk by display labels alone.
- Do not include path-dependent discontinuous products until their derivative
  policy is explicit.
- Do not require dense Jacobian materialization for large books.

## Open Design Questions

- Should factor IDs live in `trellis.core`, `trellis.analytics`, or a new
  `trellis.risk` module?
- Should a risk result include both low-level factors and bucketed factors by
  default?
- Should bump fallback be part of the AAD result or a separate comparison
  report?
- How should risk factor identity handle multiple market snapshots or scenario
  namespaces in one session?

## First Ticket Shape

Suggested future ticket:

`Portfolio AAD: risk factor registry and adapter contract`

Acceptance criteria:

- factor IDs are stable for curves, surfaces, credit curves, and scalar
  parameter sets
- existing bond-book reverse-mode risk migrates behind the adapter interface
  with no numeric regression
- unsupported positions are represented in result metadata
- no broad `portfolio_aad=True` capability is advertised yet
