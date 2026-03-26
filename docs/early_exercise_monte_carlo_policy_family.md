# Early-Exercise Monte Carlo Policy Family

This note turns the `exercise_monte_carlo` route into a small policy family
instead of a single-algorithm assumption.

## Why

Historically, Trellis treated Monte Carlo early exercise as if it meant
`longstaff_schwartz`. That is a reasonable default, but it is too narrow as a
contract.

For Trellis, the better contract is:

- Monte Carlo early exercise must use an approved optimal-stopping policy class
- the route should expose enough structure to validate:
  - stopping-policy construction
  - continuation estimation
  - lower bound
  - upper bound where applicable

## Approved Policy Classes

### `longstaff_schwartz`

Status:
- implemented

General construct:
- simulate paths
- regress discounted continuation values onto basis functions backward in time
- exercise when intrinsic value exceeds estimated continuation value

Current implementation anchor:
- `trellis.models.monte_carlo.lsm.longstaff_schwartz`

### `tsitsiklis_van_roy`

Status:
- implemented

General construct:
- approximate the continuation value recursion directly by fitted regression /
  approximate dynamic programming
- keep the regression contract explicit and separable from the path engine

Current implementation anchor:
- `trellis.models.monte_carlo.tv_regression.tsitsiklis_van_roy`

### `primal_dual_mc`

Status:
- implemented

General construct:
- compute a primal lower bound from an admissible stopping policy
- compute an optimistic upper-bound diagnostic from pathwise discounted exercise
  payoffs on the same simulated mesh
- use the bound pair as both pricing output and route-level validation

Current implementation anchor:
- `trellis.models.monte_carlo.primal_dual.primal_dual_mc`

Implementation note:
- the current upper bound is a simple pathwise perfect-information diagnostic,
  not a full Andersen-Broadie nested dual construction

### `stochastic_mesh`

Status:
- implemented

General construct:
- estimate continuation values using stochastic-mesh weighting across simulated
  states
- keep transition-weight logic explicit so the route stays auditable

Current implementation anchor:
- `trellis.models.monte_carlo.stochastic_mesh.stochastic_mesh`

## Shared Library Constructs

Before implementing all three missing classes, add common constructs in the
library:

- `EarlyExercisePolicyResult`
  - `price_lower`
  - `price_upper` when available
  - `exercise_policy_summary`
  - `diagnostics`

- continuation-estimator protocol
  - fit
  - predict continuation

- stopping-policy contract
  - derive exercise decision from intrinsic vs continuation

- optional dual-bound contract
  - compute upper bound when the policy class supports it

Suggested home:
- `trellis.models.monte_carlo.early_exercise`

## Implementation Sequence

### Phase EEMC.1: Shared Contracts

Add:
- result dataclasses
- continuation-estimator protocol
- stopping-policy protocol
- diagnostics bundle

Validation:
- deterministic unit tests for protocol / dataclass behavior

Status:
- implemented March 25, 2026

### Phase EEMC.2: Refactor `longstaff_schwartz` onto shared contracts

Goal:
- make the existing implementation the first concrete instance of the family

Validation:
- existing MC / benchmark tests stay green

Status:
- implemented March 25, 2026

### Phase EEMC.3: `tsitsiklis_van_roy`

Goal:
- add the continuation-regression ADP variant as a sibling primitive

Validation:
- side-by-side tests against `longstaff_schwartz` on standard American put
  benchmarks

Status:
- implemented March 26, 2026

### Phase EEMC.4: `primal_dual_mc`

Goal:
- add lower/upper bound output for high-value validation paths

Validation:
- deterministic bound-order tests
- task reruns for American / Bermudan comparison families

Status:
- implemented March 25, 2026

### Phase EEMC.5: `stochastic_mesh`

Goal:
- add an alternative continuation estimator suitable for higher-dimensional
  stopping problems

Validation:
- benchmark and convergence-shape tests

Status:
- implemented March 25, 2026

### Phase EEMC.6: Route and Task Integration

Goal:
- let `exercise_monte_carlo` plan around a policy class, not one symbol

Concrete work:
- extend deterministic route planning
- extend semantic validation
- extend task comparison support for policy-family cross-checks

Status:
- implemented March 25, 2026

## Guardrails

- Do not claim a policy class is supported until the library primitive exists
- Do not let prompt guidance imply planned classes are valid imports yet
- Keep import validation strict even if semantic validation recognizes the
  policy family conceptually
- Prefer lower-bound plus upper-bound-aware validation where available

## Trigger Tasks

Good proving-ground tasks for this family:
- `T07`
- American/Bermudan swaption comparison tasks
- later, harder early-exercise exotics after the vanilla family is stable
