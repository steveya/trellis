# Universal Hybrid AD Prerequisites

## Status

First prototypes delivered under the `QUA-1034`, `QUA-1040`, `QUA-1045`,
`QUA-1049`, `QUA-1054`, `QUA-1059`, `QUA-1065`, `QUA-1071`,
`QUA-1076`, `QUA-1081`, and `QUA-1086` epics. Universal hybrid
AD is still not claimed; this document now records the shipped bounded quanto
scalar-coordinate prototypes, the checked correlation matrix policy surface,
the executable matrix-coordinate lane, the ContractIR admission boundary, the
typed path-state/event policy guardrail, the first executable smooth
path-summary lane, the first executable early-exercise smooth-interior lane, and
the Phase 5 backend decision that enforces a VJP/HVP-only hybrid derivative
contract while JVP remains fail-closed. It also records the bounded
multi-product fixture surface that linearly aggregates already-computed
lane-local VJP outputs while preserving unsupported-lane diagnostics.

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
| `QUA-1057` | Done | Locked public exports and fail-closed admission matrix coverage across supported, planned, and unsupported shapes. |
| `QUA-1058` | Done | Closeout docs, limitations review, validation, and final PR preparation. |
| `QUA-1060` | Done | Added executable, well-conditioned correlation-matrix coordinate context. |
| `QUA-1061` | Done | Added bounded matrix-coordinate VJP over off-diagonal matrix coordinates. |
| `QUA-1062` | Done | Added bounded matrix-coordinate directional HVP with finite-difference checks. |
| `QUA-1063` | Done | Wired terminal quanto matrix-coordinate VJP/HVP into semantic admission and runtime metadata. |
| `QUA-1064` | Done | Closeout docs, limitations review, validation, and final PR preparation. |
| `QUA-1066` | Done | Added immutable path-state/event policy admission payloads. |
| `QUA-1067` | Done | Added ContractIR path-summary and discontinuous-event policy classification. |
| `QUA-1068` | Done | Added DynamicContractIR and early-exercise state policy classification. |
| `QUA-1069` | Done | Bridged state-policy payloads into runtime fail-closed metadata. |
| `QUA-1070` | Done | Closeout docs, limitations review, validation, and final PR preparation. |
| `QUA-1072` | Done | Added supported admission for the bounded arithmetic-average smooth path-summary VJP lane. |
| `QUA-1073` | Done | Added the executable arithmetic-Asian path-summary VJP runtime helper. |
| `QUA-1074` | Done | Added independent flat-vol finite-difference verification and unsupported-shape hardening. |
| `QUA-1075` | Done | Closeout docs, limitations review, validation, and final PR preparation. |
| `QUA-1077` | Done | Added supported admission for the bounded vanilla early-exercise flat-vol VJP lane. |
| `QUA-1078` | Done | Added the executable vanilla early-exercise VJP runtime helper. |
| `QUA-1079` | Done | Added independent flat-vol finite-difference verification and unsupported-shape hardening. |
| `QUA-1080` | Done | Closeout docs, limitations review, validation, and final PR preparation. |
| `QUA-1082` | Done | Added backend operator support records and explicit JVP unsupported reasons. |
| `QUA-1083` | Done | Normalized hybrid JVP fail-closed runtime and admission metadata through `unsupported_hybrid_jvp`. |
| `QUA-1084` | Done | Documented the VJP/HVP-only backend decision and updated limitations/plan closeout. |
| `QUA-1087` | Done | Added frozen multi-product Hybrid AD request, lane-result, and result contracts. |
| `QUA-1088` | Done | Added bounded sparse VJP aggregation across lane-local Hybrid AD results. |
| `QUA-1089` | Done | Added structured mixed unsupported-shape diagnostics and strict fail-closed policy behavior. |
| `QUA-1090` | Done | Added executable multi-product verification plus docs, limitations, and plan closeout. |
| `QUA-1091` | In Progress | Plan grid-vol path-summary and early-exercise state/control derivative policy epic. |
| `QUA-1092` | Done | Add semantic Hybrid AD admission for grid-vol path-summary and early-exercise state/control requests. |
| `QUA-1093` | Done | Define graph-owned grid-vol coordinate policy payloads for state/control derivative lanes. |
| `QUA-1094` | Done | Add the checked grid-vol path-summary runtime lane or explicit fail-closed runtime result. |
| `QUA-1095` | Todo | Harden grid-vol early-exercise fail-closed runtime and admission diagnostics. |
| `QUA-1096` | Todo | Close the grid-vol state/control epic with verification, docs, limitations, and plan updates. |

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
- `build_correlation_matrix_coordinate_context(...)` promotes valid,
  well-conditioned matrix payloads into executable off-diagonal coordinate
  context while failing closed near the PSD boundary
- `differentiate_quanto_correlation_matrix(...)` computes bounded terminal
  quanto VJP and directional HVP risk for the active underlier/FX off-diagonal
  matrix coordinate, with sparse direction validation and finite-difference
  verification away from singularities
- `fail_closed_correlation_structure_derivative(...)` distinguishes valid but
  non-executable matrix-policy calls, invalid matrix charts, and unsupported
  surface charts through typed diagnostics and `unsupported_hybrid_structure`
  metadata
- `jvp` requests, correlation surfaces, matrix requests at the PSD boundary,
  and matrix requests that require projection or smoothing fail closed through
  explicit unsupported derivative-method metadata
- `admit_hybrid_ad_lane(...)` admits only ContractIR terminal quanto VJP/HVP
  requests into the bounded graph-owned scalar-coordinate lanes or the bounded
  matrix-coordinate lane; JVP, correlation surfaces, composite-underlier,
  path-dependent, early-exercise, and dynamic hybrid shapes are classified as
  unsupported or planned before runtime AD is invoked
- `HybridADStatePolicy` records the semantic state/event policy for those
  stateful shapes: arithmetic-average flat-vol path summaries can be supported
  by the bounded VJP lane, other smooth path summaries remain planned,
  discontinuous event monitors remain unsupported, vanilla flat-vol
  early-exercise controls can be supported by the bounded VJP lane, and
  DynamicContractIR state/control requests remain planned except for
  backend-unsupported JVP
- `HybridDerivativeRequest.semantic_admission` carries that decision into the
  scalar and matrix quanto derivative helpers; supported admissions are
  preserved in result metadata while wrong-lane, planned, or unsupported
  admissions fail closed with empty risk, typed diagnostics, and searchable
  `semantic_state_policy` metadata when the admission carries a state policy
- `differentiate_arithmetic_asian_path_summary(...)` is the first executable
  smooth path-summary lane: bounded arithmetic-average European call/put
  requests over one `FlatVol` coordinate return `hybrid_path_summary_vjp`
  metadata, graph-owned sparse risk, semantic admission metadata, and
  supported `HybridADStatePolicy` payloads
- `differentiate_vanilla_early_exercise(...)` is the first executable
  smooth-interior early-exercise lane: bounded vanilla American/Bermudan
  call/put requests over one `FlatVol` coordinate return
  `hybrid_early_exercise_vjp` metadata, graph-owned sparse risk, semantic
  admission metadata, and supported hard-exercise-projection
  `HybridADStatePolicy` payloads
- non-arithmetic path summaries, grid-vol path summaries, discontinuous event
  monitors, grid-vol or boundary-kink early-exercise controls, dynamic state,
  path-summary/early-exercise HVP, and JVP remain fail closed with typed
  diagnostics
- `HybridADMultiProductRequest`, `HybridADMultiProductLaneResult`,
  `HybridADMultiProductResult`, and
  `aggregate_hybrid_ad_lane_results(...)` now compose already-executed
  lane-local `HybridDerivativeResult` values. The aggregate helper sums
  supported sparse VJP vectors by stable `RiskFactorId`, scales lane value and
  risk by explicit quantity, preserves lane-level semantic admission and
  derivative-method metadata, and records unsupported lanes as structured
  diagnostics. It is composition over supported lane-local outputs, not one
  global hybrid tape.
- permissive multi-product requests can collect supported aggregate risk while
  reporting unsupported lanes; strict requests fail closed and suppress
  aggregate value/risk when any unsupported lane is present

This is still a prototype, not universal hybrid AD. The shipped derivative
lanes differentiate a bounded scalar-coordinate vector and a bounded
directional HVP for one single-name quanto route plus one bounded
well-conditioned matrix-coordinate lane for that same terminal quanto shape,
one bounded arithmetic-average flat-vol path-summary VJP lane, and one
bounded vanilla flat-vol early-exercise VJP lane. The shipped multi-product
surface aggregates those lane-local results; it does not create a cross-product
AD tape or admit unsupported product families.
They do not differentiate arbitrary cross-asset hybrid systems, correlation
surfaces, matrix projections or PSD-boundary behavior, grid-vol or
event-monitor path-state execution, dynamic state execution, grid-vol or
boundary-kink early-exercise controls, path-summary/early-exercise HVP/JVP, or
broad product families end to end.

Autograd Phase 2 also added a truthful backend capability surface:

- `grad=True`
- `jacobian=True`
- `hessian=True`
- `vjp=True`
- `hessian_vector_product=True`
- `jvp=False`
- `portfolio_aad=False`

The backend payload also exposes a `support_matrix`, `operator_support(...)`
records, and `unsupported_reasons`. Hybrid JVP requests now resolve through
`unsupported_hybrid_jvp` metadata with `requested_backend_operator="jvp"` and
the backend support record, not an executable JVP `backend_operator`.

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
graph, a scalar-coordinate vector over spot, FX spot, curve zero-rate nodes,
flat/grid vol nodes, scalar correlation, and a direct off-diagonal
correlation-matrix coordinate lane away from the PSD boundary. Surface
correlations, path-dependent hybrid state, projected matrix charts, and other
product-family graphs remain future work unless a lane explicitly declares
support.

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
against independent finite differences of VJP vectors. The `QUA-1059`
prototype adds a well-conditioned direct matrix-coordinate context plus VJP
and directional HVP verification for the terminal quanto active off-diagonal
coordinate. Surface charts, projected or boundary matrix charts,
broader path-dependent hybrid state, and larger hybrid fixtures remain open
prerequisites. The `QUA-1065` prototype adds typed state/event policy payloads
for smooth path summaries, discontinuous event monitors, early-exercise
controls, and DynamicContractIR state/control requests, and carries those
payloads into runtime fail-closed metadata. Those policies are not executable
pathwise or dynamic derivative lanes. The `QUA-1071` prototype turns the
arithmetic-average flat-vol subset into an executable `hybrid_path_summary_vjp`
lane and verifies it against independent finite-difference bumps; all other
path-state families remain planned or unsupported.

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

### Phase 4: Correlation Matrix Policy And Executable Matrix Lane

[checked policy and fail-closed diagnostics delivered in `QUA-1049`; bounded
direct matrix-coordinate VJP/HVP delivered in `QUA-1059`]

Extend from scalar correlation to matrix or surface correlation only after the
chart is chosen and validated.

Deliverables:

- positive-semidefinite chart validation [done in `QUA-1049`]
- deterministic off-diagonal matrix `RiskFactorId` coordinates [done in
  `QUA-1049`]
- explicit projection or unsupported policy at invalid regions [done in
  `QUA-1049` as fail-closed no-projection diagnostics]
- executable derivative tests away from singularities for the terminal quanto
  active off-diagonal matrix coordinate [done in `QUA-1059`]

### Phase 4b: Path-State And Event Policy Guardrail

[semantic/runtime fail-closed policy delivered in `QUA-1065`]

Classify path-dependent, discontinuous-event, early-exercise, and dynamic
hybrid shapes before runtime derivative helpers execute.

Deliverables:

- immutable `HybridADStatePolicy` payload with JSON round trips [done in
  `QUA-1066`]
- ContractIR path-summary and discontinuous-event classification [done in
  `QUA-1067`]
- DynamicContractIR and early-exercise control policy classification [done in
  `QUA-1068`]
- runtime fail-closed metadata bridge for planned/unsupported state policies
  [done in `QUA-1069`]

### Phase 4c: Executable Smooth Path-Summary Lane

[bounded arithmetic-average flat-vol VJP delivered in `QUA-1071`]

Turn one smooth path-summary policy into an executable bounded derivative
lane without claiming broad pathwise hybrid AD.

Deliverables:

- arithmetic-average ContractIR path summaries admit the bounded VJP lane
  [done in `QUA-1072`]
- arithmetic-Asian flat-vol path-summary requests return
  `hybrid_path_summary_vjp` metadata and graph-owned sparse risk [done in
  `QUA-1073`]
- independent finite-difference verification covers call and put flat-vol VJP
  and unsupported shapes remain fail-closed [done in `QUA-1074`]
- official docs, limitations, final validation, and PR closeout [done in
  `QUA-1075`]

### Phase 4d: Executable Early-Exercise Smooth-Interior Lane

[admission contract delivered in `QUA-1077`; runtime delivered in `QUA-1078`]

Turn one early-exercise control policy into an executable bounded derivative
lane without claiming broad dynamic-state or pathwise hybrid AD.

Deliverables:

- vanilla American/Bermudan ContractIR shapes admit the bounded flat-vol VJP
  lane with supported early-exercise state-policy metadata [done in
  `QUA-1077`]
- American/Bermudan flat-vol requests return `hybrid_early_exercise_vjp`
  metadata and graph-owned sparse risk [done in `QUA-1078`]
- independent finite-difference verification covers American and Bermudan
  flat-vol VJP and unsupported shapes remain fail-closed [done in
  `QUA-1079`]
- official docs, limitations, final validation, and PR closeout [done in
  `QUA-1080`]

### Phase 5: Backend Decision For JVP

[VJP/HVP-only backend contract delivered in `QUA-1081`; checked JVP support
remains a future backend/primitive-rule prerequisite]

Document and enforce a VJP/HVP-only hybrid derivative contract. Trellis does
not report JVP support until a backend wrapper computes checked values on
pricing primitives such as `norm.cdf`.

Deliverables:

- backend capability tests [done in `QUA-1082`]
- support matrix update [done in `QUA-1082` and documented in `QUA-1084`]
- runtime metadata update [done in `QUA-1083`]

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
- valid matrix derivative requests through the policy-only helper fail closed
  as `correlation_matrix_derivative_not_implemented` with chart metadata
- invalid matrix and unsupported surface requests fail closed with typed
  diagnostics and `unsupported_hybrid_structure` metadata

Fifth delivered epic:

`Semantic hybrid AD: ContractIR admission for graph-owned lanes`

Acceptance criteria:

- immutable admission dataclasses round-trip through JSON-friendly payloads
- terminal quanto ContractIR VJP/HVP requests are admitted into the existing
  graph-owned scalar-coordinate lanes
- JVP, matrix/surface correlation as of that epic, composite-underlier,
  path-dependent, and early-exercise hybrid shapes are classified as
  unsupported or planned before runtime AD executes
- supported admissions can be carried through
  `HybridDerivativeRequest.semantic_admission` into derivative result metadata
- planned or unsupported admissions return empty risk and typed diagnostics

Sixth delivered epic:

`Hybrid AD: executable matrix-coordinate derivative lane`

Acceptance criteria:

- well-conditioned correlation-matrix payloads produce executable
  off-diagonal coordinate context without projection or repair
- terminal quanto matrix-coordinate VJP returns `hybrid_matrix_vector_vjp`
  metadata and sparse risk keyed by matrix `RiskFactorId` coordinates
- terminal quanto matrix-coordinate HVP returns `hybrid_matrix_vector_hvp`
  metadata and sparse `H @ v` values for explicit sparse directions
- empty, unknown, unsafe, JVP, surface, and PSD-boundary requests fail closed
  with typed diagnostics
- ContractIR admission supports only bounded terminal quanto matrix-coordinate
  VJP/HVP and carries the supported admission payload into runtime metadata

Seventh delivered epic:

`Hybrid AD: path-dependent state and event policy`

Acceptance criteria:

- `HybridADStatePolicy` captures state kind, differentiability class, support
  status, event/control policy, state-variable roles, metadata, and diagnostics
- ContractIR path summaries, discontinuous event monitors, and early-exercise
  shapes carry deterministic state-policy payloads
- DynamicContractIR VJP/HVP admissions remain planned with dynamic-state
  policy, while JVP remains unsupported because backend JVP is unavailable
- scalar and matrix quanto helpers preserve state-policy payloads in runtime
  fail-closed diagnostics and method metadata
- no pathwise, dynamic, or early-exercise hybrid derivative execution is
  claimed

Eighth delivered epic:

`Hybrid AD: executable smooth path-summary derivative lane`

Acceptance criteria:

- bounded arithmetic-average ContractIR path summaries admit the flat-vol VJP
  lane with supported `HybridADStatePolicy` metadata
- `differentiate_arithmetic_asian_path_summary(...)` returns
  `hybrid_path_summary_vjp` metadata, a graph-owned flat-vol risk coordinate,
  sparse VJP risk, and semantic admission/state-policy payloads
- call and put flat-vol VJP results match independent finite-difference bumps
- non-arithmetic averages, grid-vol path summaries, discontinuous event
  monitors, early exercise, HVP, JVP, and wrong-lane semantic admissions fail
  closed with searchable diagnostics
- no broad pathwise, dynamic, early-exercise, or grid-vol path-summary hybrid
  derivative execution is claimed

Ninth delivered epic:

`Hybrid AD: executable early-exercise smooth-interior derivative lane`

Acceptance criteria:

- bounded vanilla American/Bermudan ContractIR shapes admit the flat-vol VJP
  lane with supported hard-exercise-projection `HybridADStatePolicy` metadata
- `differentiate_vanilla_early_exercise(...)` returns
  `hybrid_early_exercise_vjp` metadata, a graph-owned flat-vol risk
  coordinate, sparse VJP risk, and semantic admission/state-policy payloads
- American and Bermudan flat-vol VJP results match independent
  finite-difference bumps
- grid-vol early exercise, exercise-boundary ties, HVP, JVP, and wrong-lane
  semantic admissions fail closed with searchable diagnostics
- no broad pathwise, dynamic, grid-vol early-exercise, boundary-kink
  early-exercise, or early-exercise HVP/JVP hybrid derivative execution is
  claimed

Tenth delivered epic:

`Hybrid AD: enforce VJP-HVP-only backend contract`

Acceptance criteria:

- backend capability payloads expose stable `support_matrix` records and
  explicit unsupported reasons for `jvp` and `portfolio_aad`
- hybrid JVP runtime and admission paths resolve to
  `unsupported_hybrid_jvp` with `requested_backend_operator="jvp"` and backend
  support metadata
- unsupported JVP payloads do not advertise an executable
  `backend_operator="jvp"`
- docs and limitations state that checked JVP support remains a future
  backend or primitive-rule prerequisite

## Follow-On Ticket Candidates

- `Hybrid AD: dynamic-state executable derivative lane`
- `Hybrid AD: correlation-surface chart policy and fail-closed diagnostics`

## Next Planned Epic: Grid-Vol State/Control Derivative Policy

`QUA-1091` is the next planned Hybrid AD epic. It should resolve the closest
remaining support-boundary gap: grid-vol path-summary and grid-vol
early-exercise state/control derivative requests. The work must stay bounded:
it may add a narrow executable grid-vol path-summary VJP lane only if the
coordinate policy and finite-difference verification make that mathematically
defensible. Grid-vol early-exercise, exercise-boundary kinks, event monitors,
JVP, HVP, and broad dynamic state remain fail-closed unless a child ticket
explicitly proves otherwise.

| Ticket | Status | Scope |
|---|---|---|
| `QUA-1091` | In Progress | Parent epic for grid-vol state/control derivative policy. |
| `QUA-1092` | Done | Semantic admission for grid-vol path-summary, grid-vol early-exercise, event-monitor, boundary-kink, HVP, and JVP cases. |
| `QUA-1093` | Done | Graph-owned grid-vol coordinate policy payloads, selected-factor behavior, and unsupported dependency reasons. |
| `QUA-1094` | Done | Checked grid-vol path-summary runtime result surface: executable sparse node VJP if defensible, otherwise first-class fail-closed result. |
| `QUA-1095` | Todo | Grid-vol early-exercise fail-closed diagnostics with state/control policy and boundary-kink distinctions. |
| `QUA-1096` | Todo | Verification, docs, limitations, final validation, and plan mirror closeout. |
