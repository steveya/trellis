# Universal Calibration Engine Prerequisites

## Status

Draft planning document. No Linear tickets have been created from this plan.

This document describes the missing contracts and implementation work required
before Trellis can honestly claim a universal calibration engine. It is a
follow-on planning artifact after the completed calibration sleeve umbrella and
the Autograd Phase 2 umbrella.

## Current Shipped Boundary

Trellis now has several concrete calibration workflows:

- typed solve requests and solver provenance in
  `trellis.models.calibration.solve_request`
- dependency-order support in `trellis.models.calibration.dependency_graph`
- quote maps in `trellis.models.calibration.quote_maps`
- runtime materialization helpers in
  `trellis.models.calibration.materialization`
- workflow-specific calibration modules for rates, rates vol, equity vol,
  Heston, SABR, local vol, credit, basket credit, and bounded quanto
- benchmark and replay coverage in
  `trellis.models.calibration.benchmarking` and
  `tests/test_verification/test_calibration_replay.py`

The current stack is still workflow-oriented. Each family owns its own quote
schema, objective construction, diagnostics, materialization payload, and
support boundary. That is correct for the completed pass, but it is not a
universal engine.

## Missing Gap Before Implementation

The engine cannot be built safely until Trellis has a common calibration IR
that can represent asset-specific workflows without flattening away their
mathematics.

The missing prerequisite is:

`CalibrationProblemIR`: a typed intermediate representation for quotes,
instruments, target transforms, calibration variables, constraints, dependency
nodes, solver hints, derivative availability, materialization outputs, replay
artifacts, and diagnostics.

Without this IR, a "universal engine" would be a thin dispatcher over unrelated
functions. That would add surface area without making calibration more
composable, auditable, or reusable.

## Mathematical Contract

A calibration problem should be represented as:

```text
Given:
  theta in Theta                      calibration coordinates
  m in M                              fixed market state and upstream objects
  I_i                                 instrument or calibration target i
  q_i                                 observed market quote i
  T_i                                 quote transform for target i
  w_i                                 confidence or scaling weight
  C(theta, m) >= 0                    hard or soft constraints

Find:
  theta* = argmin_theta Phi(theta; m, I, q)
```

The objective should be explicit:

```text
r_i(theta) = w_i * (T_i(P_i(theta, m; I_i)) - T_i(q_i))

Phi(theta) = 0.5 * sum_i rho_i(r_i(theta)) + R(theta, m)
```

Where:

- `P_i` is the pricing, par-spread, implied-vol, expected-loss, or other
  model-to-quote map for target `I_i`
- `T_i` maps market conventions into objective space, such as price, vol,
  spread, log-price, zero rate, survival probability, or expected loss
- `rho_i` is the loss function, usually squared loss but eventually robust
  losses for stale or noisy quotes
- `R` is regularization, smoothness, monotonicity, or prior penalty

The IR must preserve the coordinate chart:

```text
theta = chart^{-1}(x)
x in R^n
```

Examples:

- rates: unconstrained zero-rate nodes or discount-factor transforms
- credit: non-negative hazards via log hazards or positive increments
- vol surfaces: total variance or volatility nodes with calendar/strike
  constraints
- correlations: bounded scalar via `rho = tanh(x)`, or matrix coordinates via
  Cholesky, hyperspherical, or nearest-correlation projection

The engine must also represent dependency structure:

```text
G = (V, E)
v in V: calibrated object or parameter set
u -> v: v consumes materialized output from u
```

Each node should have:

```text
node_result_v = calibrate(problem_v, upstream_results_{parents(v)})
```

The graph is not optional. Hybrid and basket workflows are invalid if they
silently re-fit or bypass upstream calibrated objects.

## Computational Contract

The first implementation target should not replace existing workflows. It
should wrap them behind a shared problem shape.

Required core objects:

- `CalibrationProblemIR`: immutable dataclass for one calibration node
- `CalibrationVariableSpec`: name, coordinate chart, bounds, initial guess,
  scaling, and warm-start semantics
- `CalibrationTargetSpec`: instrument identity, quote convention, target
  transform, quote value, confidence, and validation tags
- `CalibrationObjectiveSpec`: residual vector, loss, regularization, and
  derivative contract
- `CalibrationDependencySpec`: upstream object references and materialization
  requirements
- `CalibrationMaterializationSpec`: object type, destination field on
  `MarketState`, provenance, and replay payload
- `CalibrationDiagnosticSpec`: required fit metrics, residual tolerances,
  stability checks, and latency envelope

Required services:

- problem validator
- dependency graph compiler
- solver adapter over existing `SolveRequest`
- derivative-method selector using `trellis.analytics.derivative_methods`
- materialization adapter to current `MarketState` helpers
- replay serializer and comparator
- benchmark fixture builder

The engine should delegate product mathematics to family adapters:

```text
FamilyAdapter:
  build_targets(market_snapshot_or_state) -> tuple[CalibrationTargetSpec, ...]
  build_problem(targets, upstream_objects) -> CalibrationProblemIR
  price_or_quote(theta, target, market_state) -> model quote
  materialize(result, market_state) -> MarketState
```

That keeps the universal layer from becoming a route-local formula registry.

## Required Validation Before A Real Engine

The prerequisite phase is complete only when the IR can losslessly represent
at least these existing workflows:

- dated multi-curve rates bootstrap
- caplet strip or swaption cube assembly
- equity vol surface authority
- Heston smile or surface compression
- schedule-aware single-name credit curve calibration
- basket-credit tranche correlation surface
- bounded quanto correlation

Each migrated workflow must prove:

- existing numeric results are unchanged inside current tolerances
- the generated `SolveRequest` is replayable
- derivative-method provenance is unchanged or explicitly improved
- materialized `MarketState` payloads are byte-stable where currently checked
- diagnostics and latency envelopes remain present

## Implementation Phases

### Phase 1: Calibration IR Skeleton

Add the immutable dataclasses and validators, but keep all existing workflow
functions as the source of truth. Create translation tests from current
workflow inputs into `CalibrationProblemIR`.

Deliverables:

- new IR module under `trellis/models/calibration/`
- typed validators for variables, quotes, transforms, and materialization
- tests showing existing workflow metadata can be represented without solving

### Phase 2: One Workflow Adapter

Adapt one low-risk workflow, preferably SABR smile or single-name credit, to
build an IR and then execute through the existing solver path.

Deliverables:

- adapter implementation
- parity tests against the old workflow result
- replay artifact comparison

### Phase 3: Dependency Graph Integration

Make multi-node calibration explicit by compiling the IR dependency graph into
execution order and materialized upstream objects.

Deliverables:

- graph compiler tests
- basket-credit or bounded quanto route represented as a two-node graph
- failure diagnostics for missing upstream materializations

### Phase 4: Benchmark And Replay Migration

Move benchmark metadata and replay payloads to consume IR fields instead of
workflow-local dictionaries.

Deliverables:

- benchmark report still covers the existing workflow set
- replay tests compare problem IR, solve request, result payload, and
  materialization output

### Phase 5: Universal Orchestrator

Introduce a public orchestrator only after multiple adapters prove the IR.

Deliverables:

- `calibrate(problem_ir, market_state)` entrypoint
- support matrix listing which families are engine-backed
- fail-closed behavior for unsupported families

## Explicit Non-Goals

- Do not build a single pricing formula registry inside the engine.
- Do not claim vendor-market-data integration.
- Do not replace product-specific conventions with generic strings.
- Do not collapse all quote conventions into price residuals.
- Do not remove existing workflow functions until adapters have parity and
  migration coverage.

## Open Design Questions

- Should the IR live entirely under `trellis.models.calibration`, or should
  reusable factor/coordinate objects live in `trellis.core`?
- Should robust quote loss be in the first implementation, or should it wait
  for the market-data plant?
- How should the engine represent families whose authoritative output is not a
  parameter vector, such as surfaces, curves, and grids?
- Should replay compare the IR directly, or compare a canonical serialized
  problem payload?

## First Ticket Shape

Suggested future ticket:

`Calibration engine: common problem IR and one migrated workflow adapter`

Acceptance criteria:

- `CalibrationProblemIR` exists and can represent at least one current
  workflow without changing its result.
- The migrated workflow emits the same solve request, materialization payload,
  and diagnostics as before.
- Unsupported workflows remain on their current direct functions and are not
  advertised as universal-engine-backed.
