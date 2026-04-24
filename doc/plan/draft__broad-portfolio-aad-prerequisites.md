# Broad Portfolio AAD Prerequisites

## Status

Draft planning document. No Linear tickets have been created from this plan.

This document describes the missing contracts and implementation work required
before Trellis can claim broad portfolio AAD.

## Current Shipped Boundary

Trellis currently has a bounded book-level reverse-mode lane:

- `trellis.book.portfolio_aad_curve_risk(...)`
- supported bond positions on a shared `YieldCurve`
- unsupported positions excluded and reported in metadata
- runtime reporting integrated through the derivative-method taxonomy
- backend VJP support is checked and executable

This is useful, but it is not broad portfolio AAD. It does not yet cover
general asset classes, general factor sets, hybrid objects, path-dependent
products, or a full risk aggregation graph.

## Missing Gap Before Implementation

Broad portfolio AAD cannot be built until Trellis has a unified risk factor
identity and coordinate registry.

The missing prerequisite is:

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

### Phase 2: Portfolio AAD Request And Result Schema

Define public request/result dataclasses and metadata conventions. Keep
existing `portfolio_aad_curve_risk(...)` working.

Deliverables:

- request/result types
- serialization payloads
- unsupported-position schema
- derivative-method payload integration

### Phase 3: Adapter Migration For Existing Bond Lane

Move the current supported bond-book lane behind the adapter interface without
changing output.

Deliverables:

- parity tests
- risk report compatibility tests
- finite-difference comparison

### Phase 4: Second Product Family

Add one new product family with a small smooth book. A good first target is a
vanilla option book on a shared flat-vol or grid-vol surface.

Deliverables:

- product adapter
- factor dependency tests
- aggregation and unsupported-position tests

### Phase 5: Runtime And Benchmark Integration

Expose broad AAD through `Session.risk_report(...)` only for supported books.
Add latency and coverage reporting.

Deliverables:

- benchmark artifact
- docs in `docs/quant/differentiable_pricing.rst`
- support matrix row updates

## Explicit Non-Goals

- Do not claim universal portfolio AAD from the existing bond-book lane.
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
