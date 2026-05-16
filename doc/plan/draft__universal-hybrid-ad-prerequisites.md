# Universal Hybrid AD Prerequisites

## Status

First prototypes delivered under the `QUA-1034`, `QUA-1040`, `QUA-1045`, and
`QUA-1049` epics. Universal hybrid AD is still not claimed; this document now
records the shipped bounded quanto scalar-coordinate prototypes, the checked
correlation matrix policy surface, and the remaining prerequisites.

| Ticket | Status | Outcome |
|---|---|---|
| `QUA-1035` | Done | Added typed hybrid factor graph primitives and scalar-correlation charts. |
| `QUA-1036` | Done | Added an opt-in graph-producing path to the quanto resolver. |
| `QUA-1037` | Done | Added a bounded scalar quanto VJP derivative lane. |
| `QUA-1038` | Done | Added fail-closed JVP and matrix/surface correlation policy. |
| `QUA-1039` | Done | Closeout docs, limitations, validation, and final PR preparation. |
| `QUA-1041` | Done | Added executable chart context for quanto graph curve and vol nodes. |
| `QUA-1042` | Done | Added graph-owned multi-factor scalar quanto VJP over supported scalar coordinates. |
| `QUA-1043` | Done | Hardened selected-factor filtering, zero-sensitivity selections, and fail-closed diagnostics. |
| `QUA-1044` | Done | Closeout docs, limitations, validation, and final PR preparation. |
| `QUA-1046` | Done | Added HVP request direction contract and derivative-method taxonomy. |
| `QUA-1047` | Done | Added graph-owned scalar quanto HVP execution over supported scalar coordinates. |
| `QUA-1048` | Done | Closeout docs, limitations review, validation, and final PR preparation. |
| `QUA-1050` | Done | Added checked correlation matrix chart-policy construction and payload round trips. |
| `QUA-1051` | Done | Hardened matrix/surface correlation fail-closed diagnostics with chart metadata. |
| `QUA-1052` | Done | Added invalid matrix diagnostic-code coverage and public-surface checks. |
| `QUA-1053` | Done | Closeout docs, limitations review, validation, and final PR preparation. |
| `QUA-1055` | Done | Added ContractIR-backed semantic admission dataclasses and classifier for bounded graph-owned hybrid AD lanes. |
| `QUA-1056` | Done | Bridged supported/planned/unsupported semantic admissions into bounded quanto derivative result metadata. |

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

Trellis also has one bounded portfolio-AAD scalar-correlation lane:

- `portfolio_aad_quanto_correlation_risk(...)` differentiates an already
  resolved single-name quanto option book with respect to one scalar
  underlier/FX correlation
- the lane uses canonical `RiskFactorId` coordinates with
  `object_type="model_parameter"` and `coordinate_type="correlation"`
- all curves, spots, FX spot, volatility inputs, and broader hybrid factor
  graph dependencies are held fixed outside this narrow lane

Trellis now also has a bounded graph-backed quanto hybrid-AD prototype:

- `HybridFactorGraph`, `HybridDependencyNode`,
  `MarketObjectCoordinateChart`, and `HybridUnsupportedDependency` define the
  typed graph and coordinate payloads
- `resolve_quanto_inputs(..., include_hybrid_factor_graph=True)` can attach a
  bounded quanto graph whose nodes describe underlier spot, FX spot,
  domestic/foreign curves, underlier vol, FX vol, and scalar correlation
- `differentiate_quanto_scalar_correlation(...)` computes one VJP-backed
  sensitivity to the graph-owned scalar underlier/FX correlation coordinate
- `differentiate_quanto_scalar_inputs(...)` computes a VJP-backed sparse risk
  vector over supported graph-owned scalar coordinates for the same bounded
  quanto route: underlier spot, FX spot, domestic/foreign curve zero-rate
  nodes, flat/grid vol nodes, and scalar correlation
- `differentiate_quanto_scalar_inputs(..., HybridDerivativeRequest(
  derivative_method="hvp", hvp_direction=...))` computes a bounded
  graph-owned directional HVP, `H @ v`, over the same scalar-coordinate chart
- the scalar correlation chart supports both constrained `rho` and
  unconstrained `x` coordinates through `rho = tanh(x)`
- `HybridDerivativeResult` returns a sparse risk vector, graph payload,
  method metadata, unsupported dependency records, and diagnostics
- selected-factor requests filter the returned sparse vector without changing
  the full-factor metadata; missing selected factors and unsupported graph
  dependencies are reported explicitly
- `MarketObjectCoordinateChart.correlation_matrix_policy(...)` can validate a
  correlation matrix policy payload, derive deterministic off-diagonal
  `RiskFactorId` coordinates, record the minimum eigenvalue, and enforce the
  no-projection policy
- `fail_closed_correlation_structure_derivative(...)` distinguishes valid but
  non-executable matrix charts, invalid matrix charts, and unsupported surface
  charts through typed diagnostics and `unsupported_hybrid_structure` metadata
- `jvp` requests and correlation matrix/surface requests fail closed through
  explicit unsupported derivative-method metadata
- `admit_hybrid_ad_lane(...)` admits only ContractIR terminal quanto VJP/HVP
  requests into the bounded graph-owned scalar-coordinate lanes; JVP,
  matrix/surface correlation, composite-underlier, path-dependent, and
  early-exercise hybrid shapes are classified as unsupported or planned before
  runtime AD is invoked
- `HybridDerivativeRequest.semantic_admission` carries that decision into
  `differentiate_quanto_scalar_inputs(...)`; supported admissions are preserved
  in result metadata while planned or unsupported admissions fail closed with
  empty risk and typed diagnostics

This is still a prototype, not universal hybrid AD. The shipped derivative
lanes differentiate a bounded scalar-coordinate vector and a bounded
directional HVP for one single-name quanto route. They do not differentiate
arbitrary cross-asset hybrid systems, executable matrix/surface correlations,
path-dependent hybrid state, or broad product families end to end.

Autograd Phase 2 also added a truthful backend capability surface:

- `grad=True`
- `jacobian=True`
- `hessian=True`
- `vjp=True`
- `hessian_vector_product=True`
- `jvp=False`
- `portfolio_aad=False`

The current system can compute derivatives for many smooth single-route
pricing maps, bounded calibration representatives, bounded portfolio-AAD
lanes, and the first graph-backed bounded quanto scalar-vector hybrid
VJP/HVP derivatives. It cannot yet differentiate arbitrary cross-asset hybrid
systems end to end.

## Missing Gap Before Implementation

Universal hybrid AD still cannot be claimed until the typed graph and
coordinate chart expand beyond the first scalar quanto prototype.

The first prerequisite now exists in bounded form:

`HybridFactorGraph`: a representation of curves, surfaces, spots, FX rates,
correlations, basis bridges, parameter sets, and derived market objects as one
typed differentiable dependency graph with explicit coordinate ownership.

The remaining gap is breadth and executable derivative ownership. Without a
complete graph for the requested product family, "hybrid AD" would trace
whichever floats happen to pass through a route. That is not enough to define
what a derivative means, how risk factors aggregate, how constraints are
enforced, or which unsupported dependencies were held fixed.

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

The first shipped subset covers these objects for a bounded single-name quanto
graph and a scalar-coordinate vector over spot, FX spot, curve zero-rate
nodes, flat/grid vol nodes, and scalar correlation. Matrix correlations,
surface correlations, path-dependent hybrid state, and other product-family
graphs remain future work unless a lane explicitly declares support.

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

The `QUA-1034` prototype covers scalar correlation chart derivatives,
bounded quanto resolver provenance, held-fixed and unsupported dependency
reporting, JVP fail-closed behavior, and a finite-difference check for the
smooth scalar correlation VJP lane. The `QUA-1040` prototype widens that route
to a bounded scalar-coordinate VJP vector over graph-owned spot, FX spot,
curve-node, vol-node, and correlation factors, with selected-factor and
fail-closed policy tests. The `QUA-1045` prototype adds an explicit sparse HVP
direction contract and checks the bounded scalar-coordinate directional HVP
against independent finite differences of VJP vectors. Matrix/surface charts,
path-dependent hybrid state, and larger hybrid fixtures remain open
prerequisites.

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

[done in `QUA-1035`]

Define the graph and coordinate dataclasses. Do not change pricing behavior.
Add tests that build the graph for a bounded quanto route and inspect
dependencies.

Deliverables:

- graph objects
- coordinate chart objects for scalar spot, curve nodes, vol nodes, and scalar
  correlation
- provenance round-trip tests

### Phase 2: Resolver Integration

[done in `QUA-1036` for the bounded quanto resolver]

Teach one resolver, likely the quanto resolver, to produce or consume graph
metadata while preserving current pricing behavior.

Deliverables:

- graph-aware `resolve_quanto_inputs(...)` path
- tests proving all resolved inputs map to graph nodes or explicit held-fixed
  constants

### Phase 3: Smooth VJP Route

[done in `QUA-1037` for one scalar quanto correlation coordinate]

Implement one smooth VJP-backed hybrid derivative lane where all coordinates
are explicit and supported.

Deliverables:

- derivative result with method metadata
- finite-difference comparison
- fail-closed behavior for unsupported coordinates

### Phase 3b: Scalar Coordinate Vector VJP Route

[done in `QUA-1040` for the bounded single-name quanto scalar-coordinate
vector]

Extend the bounded quanto route from one scalar correlation coordinate to the
graph-owned scalar coordinates that can be reconstructed from executable
charts.

Deliverables:

- executable chart context for curve and vol nodes
- VJP-backed sparse vector for supported scalar coordinates
- selected-factor filtering and fail-closed diagnostics
- finite-difference checks for representative smooth coordinates

### Phase 3c: Scalar Coordinate Vector HVP Route

[done in `QUA-1045` for the bounded single-name quanto scalar-coordinate
vector]

Extend the bounded quanto scalar-coordinate lane from first-order VJP to one
checked second-order directional HVP over the same executable graph chart.

Deliverables:

- explicit sparse `hvp_direction` request contract keyed by `RiskFactorId`
- `hybrid_scalar_vector_hvp` derivative-method metadata
- HVP-backed sparse `H @ v` vector for supported scalar coordinates
- fail-closed diagnostics for empty or unavailable HVP directions
- finite-difference checks against VJP-vector perturbations

### Phase 4: Correlation Matrix Policy

[checked policy and fail-closed diagnostics delivered in `QUA-1049`;
derivative support still open]

Extend from scalar correlation to matrix or surface correlation only after the
chart is chosen and validated.

Deliverables:

- positive-semidefinite chart validation [done in `QUA-1049`]
- deterministic off-diagonal matrix `RiskFactorId` coordinates [done in
  `QUA-1049`]
- explicit projection or unsupported policy at invalid regions [done in
  `QUA-1049` as fail-closed no-projection diagnostics]
- executable derivative tests away from singularities [open]

### Phase 5: Backend Decision For JVP

[fail-closed runtime policy delivered in `QUA-1038`; checked JVP support still
open]

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

## Completed First Ticket Shape

Delivered epic:

`Hybrid AD: factor graph and scalar quanto derivative prototype`

Acceptance criteria:

- a bounded quanto route can build a typed hybrid factor graph
- every resolved market input is represented as differentiable, held fixed, or
  unsupported
- one smooth derivative lane is checked against finite differences
- unsupported `jvp` requests fail closed and report why

Second delivered epic:

`Hybrid AD: graph-owned quanto scalar coordinate VJP`

Acceptance criteria:

- the bounded quanto graph carries executable scalar chart context for curves
  and vol surfaces
- the route-level helper can return a sparse VJP risk vector over supported
  graph-owned scalar coordinates
- selected-factor requests, known zero-sensitivity factors, unsupported graph
  dependencies, and `jvp` requests are explicitly governed

Third delivered epic:

`Hybrid AD: scalar graph HVP lane`

Acceptance criteria:

- the bounded scalar-coordinate quanto helper accepts an explicit sparse HVP
  direction over graph-owned `RiskFactorId` coordinates
- the route-level helper returns a sparse directional second derivative
  `H @ v` with `hybrid_scalar_vector_hvp` metadata
- empty or unavailable HVP directions fail closed with typed diagnostics
- selected-factor requests filter the returned HVP vector without changing
  full-factor metadata

Fourth delivered epic:

`Hybrid AD: correlation matrix chart policy and validation`

Acceptance criteria:

- valid correlation matrix policy payloads are checked for factor labels,
  shape, finite entries, symmetry, unit diagonal, bounds, and PSD tolerance
- the policy chart records deterministic off-diagonal factor coordinates,
  matrix dimension, factor labels, minimum eigenvalue, and no-projection policy
- valid matrix derivative requests fail closed as
  `correlation_matrix_derivative_not_implemented` with chart metadata
- invalid matrix and unsupported surface requests fail closed with typed
  diagnostics and `unsupported_hybrid_structure` metadata

## Follow-On Ticket Candidates

- `Hybrid AD: ContractIR admission for graph-owned hybrid derivative lanes`
- `Hybrid AD: executable matrix-coordinate derivative lane away from PSD boundary`
- `Hybrid AD: path-dependent hybrid state and event policy`
- `Hybrid AD: multi-product graph-owned derivative fixtures`
