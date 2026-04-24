# Universal Hybrid AD Prerequisites

## Status

Draft planning document. No Linear tickets have been created from this plan.

This document describes the missing mathematical and computational contracts
required before Trellis can honestly claim universal hybrid automatic
differentiation.

## Current Shipped Boundary

Trellis now has a bounded hybrid quanto-correlation calibration route:

- `calibrate_quanto_correlation_workflow(...)` fits one scalar correlation for
  the checked `bounded_quanto_correlation` route
- the route consumes already-bound domestic/foreign curves, underlier and FX
  spots, a vol surface, and quanto option quotes
- solve provenance records
  `resolved_derivative_method="scipy_2point_residual_jacobian"`
- derivative reporting classifies this as a governed finite-difference
  fallback, not as hybrid AD or AAD

Autograd Phase 2 also added a truthful backend capability surface:

- `grad=True`
- `jacobian=True`
- `hessian=True`
- `vjp=True`
- `hessian_vector_product=True`
- `jvp=False`
- `portfolio_aad=False`

The current system can compute derivatives for many smooth single-route
pricing maps and bounded calibration representatives. It cannot yet
differentiate arbitrary cross-asset hybrid systems end to end.

## Missing Gap Before Implementation

Universal hybrid AD cannot be started until Trellis has a typed factor graph
and coordinate chart for hybrid market objects.

The missing prerequisite is:

`HybridFactorGraph`: a representation of curves, surfaces, spots, FX rates,
correlations, basis bridges, parameter sets, and derived market objects as one
typed differentiable dependency graph with explicit coordinate ownership.

Without that graph, "hybrid AD" would trace whichever floats happen to pass
through a route. That is not enough to define what a derivative means, how risk
factors aggregate, how constraints are enforced, or which unsupported
dependencies were held fixed.

## Mathematical Contract

A hybrid value should be represented as:

```text
V = F(S(theta), C(theta), Sigma(theta), R(theta), rho(theta), p(theta))
```

Where:

- `S` are spot and forward observables
- `C` are discount, forecast, and basis curves
- `Sigma` are vol surfaces or model volatility parameters
- `R` are credit or survival objects
- `rho` are correlations or correlation surfaces
- `p` are route-local model parameters

Hybrid AD must define the full coordinate vector:

```text
theta = [theta_curve, theta_surface, theta_credit, theta_fx, theta_corr, theta_model]
```

and the derivative:

```text
dV / dtheta
```

only for coordinates owned by the graph. If a dependency is external,
constant, stale, or unsupported, the derivative report must say so.

For directional derivatives:

```text
JVP_F(theta, v) = J_F(theta) v
VJP_F(theta, w) = J_F(theta)^T w
HVP_F(theta, v) = H_F(theta) v
```

These are only meaningful after the coordinate chart is explicit.

Correlation parameters need a constrained chart. Scalar correlations can use:

```text
rho = tanh(x)
```

For correlation matrices:

```text
R = L L^T
diag(R) = 1
R positive semidefinite
```

Possible parameterizations:

- hyperspherical coordinates
- Cholesky-like coordinates with normalization
- Fisher or tanh transforms plus projection
- nearest-correlation projection with explicit non-smooth fallback

The chosen chart must define both:

```text
dR / dx
```

and the behavior at bounds or projection singularities.

Quanto-style hybrid maps also need bridge semantics:

```text
F_foreign_domestic = S_fx * D_foreign / D_domestic * basis_bridge
```

The derivative must know whether a foreign curve, FX forward, or basis bridge
is canonical, user-supplied, inferred, or explicitly held fixed.

## Backend Gap

The current backend intentionally reports `jvp=False`. Stock `autograd` lacks
forward-mode coverage for pricing primitives Trellis uses, including normal
CDF based pricing routes. Universal hybrid AD cannot depend on a backend
operator that is not executable.

Before claiming hybrid AD, Trellis needs one of:

- owned primitive rules for missing forward-mode operations
- a backend switch or multi-backend layer with checked JVP coverage
- a hybrid derivative policy that restricts itself to VJP/HVP-backed lanes and
  explicitly excludes JVP

The support contract must stay executable truth, not roadmap intent.

## Computational Contract

Required core objects:

- `HybridFactorGraph`
- `RiskFactorCoordinate`
- `MarketObjectCoordinateChart`
- `HybridDependencyNode`
- `HybridDerivativeRequest`
- `HybridDerivativeResult`
- `HybridUnsupportedDependency`

The graph must represent:

- market object identity
- coordinate ownership
- transforms from unconstrained coordinates to market objects
- upstream dependency edges
- differentiability class: smooth, piecewise, discontinuous, projected, held
  fixed, or unsupported
- derivative method selected: AD, VJP, HVP, bump, custom adjoint, smoothed,
  finite-difference fallback, or unsupported

The graph should not hide route-local resolvers. For example,
`resolve_quanto_inputs(...)` should become a graph-producing or graph-consuming
boundary, so provenance for spot, FX spot, domestic curve, foreign curve, vol
surface, and correlation remains visible in derivative output.

## Required Validation Before Universal Claims

Before the first universal-hybrid claim, Trellis needs tests for:

- scalar correlation chart derivatives away from bounds
- correlation-matrix chart validity and derivative behavior
- quanto input resolver provenance under derivative requests
- held-fixed dependencies reported explicitly
- unsupported forward-mode requests fail closed when JVP is unavailable
- VJP/HVP requests match finite-difference checks on smooth hybrid fixtures
- boundary behavior near correlation limits
- no derivative path silently traces stale or inferred market data

The first validation target should be small:

```text
V(theta) = price_quanto_option(
    domestic_curve(theta_d),
    foreign_curve(theta_f),
    vol_surface(theta_sigma),
    spot(theta_s),
    fx(theta_fx),
    rho(theta_rho)
)
```

Use explicit coordinates and compare selected derivatives against independent
finite differences.

## Implementation Phases

### Phase 1: Hybrid Factor Graph Prototype

Define the graph and coordinate dataclasses. Do not change pricing behavior.
Add tests that build the graph for a bounded quanto route and inspect
dependencies.

Deliverables:

- graph objects
- coordinate chart objects for scalar spot, curve nodes, vol nodes, and scalar
  correlation
- provenance round-trip tests

### Phase 2: Resolver Integration

Teach one resolver, likely the quanto resolver, to produce or consume graph
metadata while preserving current pricing behavior.

Deliverables:

- graph-aware `resolve_quanto_inputs(...)` path
- tests proving all resolved inputs map to graph nodes or explicit held-fixed
  constants

### Phase 3: Smooth VJP Route

Implement one smooth VJP-backed hybrid derivative lane where all coordinates
are explicit and supported.

Deliverables:

- derivative result with method metadata
- finite-difference comparison
- fail-closed behavior for unsupported coordinates

### Phase 4: Correlation Matrix Policy

Extend from scalar correlation to matrix or surface correlation only after the
chart is chosen and validated.

Deliverables:

- positive-semidefinite chart validation
- derivative tests away from singularities
- explicit projection or unsupported policy at invalid regions

### Phase 5: Backend Decision For JVP

Either implement checked JVP primitive coverage or document and enforce a
VJP/HVP-only hybrid derivative contract.

Deliverables:

- backend capability tests
- support matrix update
- runtime metadata update

## Explicit Non-Goals

- Do not claim universal hybrid AD from the existing finite-difference quanto
  calibration route.
- Do not trace through arbitrary Python object mutation.
- Do not infer derivatives for vendor or mock data that is not represented in
  the factor graph.
- Do not smooth correlation constraints or discontinuous events silently.
- Do not report `jvp` support until the backend wrapper computes checked
  values on pricing primitives.

## Open Design Questions

- Should hybrid coordinates reuse portfolio risk factor IDs, or should risk
  IDs be derived from the hybrid factor graph?
- Should correlation matrices use a differentiable chart only, or allow a
  projected chart with explicit non-smooth fallback?
- Is a multi-backend layer required before forward-mode hybrid derivatives?
- How should path-dependent hybrid products report event discontinuities and
  held-fixed path state?

## First Ticket Shape

Suggested future ticket:

`Hybrid AD: factor graph and scalar quanto derivative prototype`

Acceptance criteria:

- a bounded quanto route can build a typed hybrid factor graph
- every resolved market input is represented as differentiable, held fixed, or
  unsupported
- one smooth derivative lane is checked against finite differences
- unsupported `jvp` requests fail closed and report why
