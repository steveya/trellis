# Semantic Simulation Substrate Plan

## Purpose

This document defines the implementation queue for `QUA-886`, the new
standalone simulation-substrate epic under the Trellis roadmap.

The goal is to add a reusable factor-state simulation abstraction that sits
between the current family-level numerical lanes and later institutional
workflows.

This plan is intentionally:

- broader than xVA
- narrower than "replace ORE"
- upstream of future-value, collateral, netting, and xVA tickets
- independent enough to be worth doing even if institutional valuation stays
  later-phase work

## Decision Summary

The roadmap direction is:

- keep `QUA-886` separate from the institutional valuation umbrella
- treat the current single-state Monte Carlo lane as an important precursor,
  not the final simulation abstraction
- expose simulated factor state, projected market state, conditional valuation,
  and future-value cubes as first-class runtime contracts
- let later institutional tickets consume this substrate instead of inventing
  their own local abstractions

The short version is:

- current Trellis is good at bounded pricing lanes
- current Trellis is not yet good at generic simulation programs over factor
  state
- `QUA-886` exists to close that gap

## Linear Ticket Mirror

These tables mirror the current Linear state for this workstream and record the
intended local execution order.

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This file is the repo-local execution mirror for `QUA-886`.
- Do not implement the umbrella epic directly.
- Use this file to shape child tickets and later execution order.
- Once child tickets exist, implement the earliest child whose prerequisites
  are satisfied.
- Update this file only after the relevant Linear ticket is updated.

Status mirror last synced: `2026-04-18`

### Epic Row

| Ticket | Status |
| --- | --- |
| `QUA-886` Semantic simulation substrate: factor-state valuation, market projection, and future-value cube | Backlog |

### Child Ticket Queue

| Ticket | Status | Ordered role | Hard prerequisites |
| --- | --- | --- | --- |
| `QUA-889` Semantic simulation substrate: factor-state family IR and runtime contract | Backlog | first checked factor-state family boundary | none |
| `QUA-890` Semantic simulation substrate: event and observable lowering onto factor-state programs | Backlog | universal event/control lowering onto simulation | `QUA-889` |
| `QUA-891` Semantic simulation substrate: factor-state projection onto valuation-facing market views | Backlog | explicit `Phi_t` market projection | `QUA-889` |
| `QUA-892` Conditional valuation: reusable intermediate-date regression and continuation contract | Backlog | reusable `V_a(t_i, x)` service | `QUA-890`, `QUA-891` |
| `QUA-893` Future-value cube: trade-date-path valuation tensor and projections | Backlog | stable `FutureValueCube` output contract | `QUA-892` |
| `QUA-894` Simulation proof: interest-rate swap future-value runtime through `T52` | Backlog | first proving path through task-runtime / benchmark surfaces | `QUA-893` |
| `QUA-895` Simulation substrate: docs, benchmark surfaces, and limitation truth | Backlog | epic closeout, docs, and validation truth | `QUA-894` |

### Adjacent Ticket Context

| Ticket | Status | Relationship |
| --- | --- | --- |
| `QUA-719` | Done | landed the bounded single-state event-aware Monte Carlo lane that this plan builds on |
| `QUA-734` | Done | landed the universal `EventProgramIR` / `ControlProgramIR` boundary that this plan must extend rather than duplicate |
| `QUA-635` | Done | landed `ScenarioResultCube`, which is adjacent to but not the same as a future-value cube |
| `QUA-594` | Backlog | later institutional consumer umbrella, not the right parent abstraction |
| `QUA-638` | Backlog | first swap-portfolio future-value consumer that should narrow onto this substrate instead of redefining it |
| `QUA-639` | Backlog | collateral-state consumer of the future-value substrate |
| `QUA-640` | Backlog | netting-set consumer of the future-value substrate |
| `QUA-641` | Backlog | exposure-statistics consumer of the future-value substrate |
| `QUA-619` | Backlog | xVA integration ticket that should remain downstream of the reusable substrate |
| `T52` | `proof_only_hold` | existing held proof task for MC exposure simulation on an interest-rate swap; likely the right first proving task-runtime surface |
| `T95` | `proof_only_hold` | later xVA-style proof task that should stay downstream of the substrate and `T52`-class future-value work |

## Mathematical Target

The target computational contract is:

```text
Choose one numeraire N per substrate instance

Latent factor state
  X_t in E subseteq R^d
  dX_t = mu(t, X_t) dt + Sigma(t, X_t) dW_t

Simulated market projection
  M_t = Phi_t(X_t)

Observation grid
  0 = t_0 < t_1 < ... < t_m

Event / observable program
  O_{i,ell} = h_{i,ell}(X_{t_i})
  Y_{i,0} given
  Y_{i,ell} = T_{i,ell}(Y_{i,ell-1}, O_{i,ell})
  Y_{i+1,0} = Y_{i,p_i}

Trade-level future value under numeraire N
  H_a = sum_{u in U_a} (N_{T_a} / N_u) CF_a(u; X_[0,u], Y_[0,u])
  V_a^N(t_i, x) = N_{t_i} E^{Q^N}[ H_a / N_{T_a} | X_{t_i} = x ]

Conditional valuation approximation where needed
  V_{a,i}(x) ~= sum_{k=1}^K beta_{a,i,k} phi_k(x)

Future-value cube
  C_{a,i,n} = V_a^{clean,N}(t_i^+, X_{t_i}^{(n)})
```

The cube semantics should be interpreted as:

```text
Observation grid
  0 <= t_0 < t_1 < ... < t_m

Trade-level clean future value
  C_{a,i,n} = V_a^{clean,N}(t_i^+, X_{t_i}^{(n)})
```

with the following explicit meaning:

- `t_i^+` means after applying contractual events whose effective phase is at
  `t_i`, using the already-landed universal event/control ordering
- values are time-`t_i` clean trade values, not discounted-to-as-of values
- values are pre-netting, pre-collateral, pre-CVA, pre-DVA, and pre-FVA
- values should be `0` once the trade has fully matured and settled on the cube
  grid
- the observation grid must be a stable shared cube grid or a documented
  projection onto one, so later netting-set aggregation is mathematically
  well-defined across trades
- if holder optionality is active at `t_i`, the reported `t_i^+` value is
  post-decision when the decision phase is included at `t_i`

If conditional valuation is fitted rather than exact, the fit regime should be
explicit whenever the fitted surface feeds exposure statistics:

```text
D_{a,i}^{train} = {(X_{t_i}^{(n)}, Z_{a,i}^{(n)})}_{n in I_train}
D_{a,i}^{eval}  = {(X_{t_i}^{(n)}, Z_{a,i}^{(n)})}_{n in I_eval}
I_train intersect I_eval = emptyset
```

The downstream institutional algebra is deliberately outside the core substrate
and should consume it rather than redefine it:

```text
Netting set aggregation
  N_{g,i,n} = sum_{a in g} C_{a,i,n}

Collateral recursion
  K_{g,i+1,n} = Gamma(K_{g,i,n}, N_{g,i,n}; Theta_g)

Positive / negative exposure
  E^+_{g,i,n} = max(N_{g,i,n} - K_{g,i,n}, 0)
  E^-_{g,i,n} = max(K_{g,i,n} - N_{g,i,n}, 0)

Exposure statistics
  EE_g(t_i) = (1 / Npaths) sum_n E^+_{g,i,n}
  EPE_g(t_i) = (1 / i) sum_{j <= i} EE_g(t_j)
  PFE_{g,alpha}(t_i) = inf { x : P(E^+_{g,i} <= x) >= alpha }
```

### Current bounded contract

Today Trellis is closer to the following bounded Monte Carlo contract:

```text
X_t in R
H = G(X_T, rho(X_{tau_1}, ..., X_{tau_m}))
```

where `rho` is built from bounded snapshots, barrier flags, reducers, or event
replay.

That is a legitimate family surface. It is not yet a generic factor-state
simulation substrate.

## Why This Plan Exists

The repo now proves two things at once:

1. low-level simulation support is stronger than the current exposed
   computational abstraction, and
2. later workflows such as exposure, future value, or ORE-style simulation will
   become structurally awkward unless the abstraction is widened first.

Concretely, the current repo already has:

- a bounded single-state event-aware Monte Carlo runtime in
  `trellis/models/monte_carlo/event_aware.py`
- explicit acknowledgment that this runtime is "single-state" in the module
  docstring
- raw vector-state simulation support in
  `MonteCarloEngine._simulate_state_euler_vector()` in
  `trellis/models/monte_carlo/engine.py`
- multi-dimensional processes such as `Heston` in
  `trellis/models/processes/heston.py`
- localized continuation / regression machinery in
  `trellis/models/monte_carlo/lsm.py`,
  `tv_regression.py`, `primal_dual.py`, and `stochastic_mesh.py`
- a scenario-batch repricing cube in `trellis/book.py`
- a task-runtime and benchmark workflow that is now a first-class validation
  surface for new computational lanes

But the current stack does not yet expose:

- a first-class factor-state family for `d > 1`
- an explicit market projection `Phi_t`
- a generic event / observable program over latent factor state
- a reusable conditional-valuation service for intermediate dates
- a future-value cube over `(trade, date, path)`

That is the real gap this plan exists to close.

## Repo-Grounded Current State

### Strengths to preserve

- `EventAwareMonteCarloIR` and its runtime are already a useful bounded family
  for schedule-driven single-state problems.
- The raw Monte Carlo engine can already stream vector-state processes into
  reduced storage.
- Early-exercise code already proves Trellis can fit
  `continuation ~ basis(state)`-style approximations.
- `ScenarioResultCube` already proves Trellis can publish stable cube-like
  outputs with provenance and projection helpers.

### Gaps to close

#### 1. Missing generic factor-state family

The compiler/runtime boundary does not yet expose a family whose primary state
contract is:

```text
E subseteq R^d, d >= 1
```

rather than a bounded one-factor family.

#### 2. Missing simulated-market projection

There is no first-class runtime map

```text
Phi_t : E -> M_t
```

that turns latent factor coordinates into projected market views consumed by
downstream valuation.

#### 3. Missing generic event / observable program

There is no reusable program-level contract of the form:

```text
O_i = h_i(X_{t_i})
Y_{i+1} = T_i(Y_i, O_i)
L = G(Y_m, X_{t_m})
```

that is not already product-local or lane-local.

#### 4. Missing conditional valuation service

The stack does not yet expose:

```text
V_a(t_i, x) = N_{t_i} E^{Q^N}[ H_a / N_T | X_{t_i} = x ]
```

as a reusable runtime service. Existing regression machinery is still tied
mostly to early-exercise policy code paths.

#### 5. Missing future-value cube

`ScenarioResultCube` has a scenario axis, not a path-and-date valuation axis.
The missing stable object is:

```text
C_{a,i,n} = V_a(t_i, X_{t_i}^{(n)})
```

with clear semantics for trade, date, path, metadata, projections, and later
aggregation.

## Design Guardrails

### 1. This is not an xVA umbrella

`QUA-886` should stop before closeout, collateral, netting, and xVA become the
main implementation surface.

### 2. This is not a universal solver IR

The current family-shaped architecture remains correct. The target is one new
simulation substrate or family boundary, not a single IR for every backend.

### 3. Existing bounded lanes should survive the migration

`EventAwareMonteCarloIR`, `CorrelatedBasketMonteCarloIR`, and the current
product lanes should remain valid while the broader substrate is introduced.

### 4. The new substrate must extend the existing universal compiler program

The repo already has a shared compiler program above family lowering:

- `EventProgramIR`
- `ControlProgramIR`

This plan must project from those shared objects into the new simulation
substrate. It must not create a second unrelated event/control semantics stack.

Likewise, where the current MC family already has useful typed surfaces such as
`MCStateSpec`, `MCProcessSpec`, `MCMeasureSpec`, and `MCEventTimeSpec`, this
plan should refine or widen them where practical rather than fork them into
parallel naming-only abstractions.

### 5. `FutureValueCube` and `ScenarioResultCube` should remain distinct

The repo should not overload one object with both:

- scenario-batch repricing semantics, and
- path/date conditional-valuation semantics

unless that unification is mathematically and operationally explicit.

### 6. The first reusable contract must admit `d > 1`

The important widening is not "support one more product." It is "make
vector-state processes first-class at the semantic/runtime boundary."

### 7. `Phi_t` must be explicit

The runtime should not force downstream consumers to reinterpret raw state
coordinates ad hoc.

### 8. Regression / continuation must be reusable beyond early exercise

The target is not just Bermudan continuation. The target is intermediate-date
conditional valuation as a stable contract.

## Target Typed Surfaces

Exact names may change, but the conceptual surfaces should be close to:

- `FactorStateSimulationIR`
- `SimulationStateSpec`
- `SimulationFactorSpec`
- `SimulationProcessBundleSpec`
- `SimulatedMarketProjectionSpec`
- `ObservationProgramSpec`
- `StateTransitionSpec`
- `ConditionalValuationSpec`
- `ConditionalValuationModel`
- `FutureValueCube`
- `FutureValueCubeMetadata`

The important requirement is not the exact class names. The important
requirement is that the public contract can express:

- latent factor state
- projected market state
- observable extraction
- event/state transitions
- conditional future value
- cube emission

Where possible, these should be expressed as:

- widenings or companions of the current Monte Carlo typed surfaces in
  `trellis/agent/family_lowering_ir.py`
- projections from the existing `EventProgramIR` / `ControlProgramIR`
- runtime companions to the existing market-binding and valuation-context
  surfaces rather than a second independent market semantics stack

## Proposed Ordered Delivery Queue

The umbrella is now decomposed into concrete child tickets. The intended local
execution order is the same as the Linear dependency graph above.

Downstream institutional tickets remain outside this child queue:

- `QUA-638` to `QUA-641` remain consumers of the reusable substrate
- `QUA-619` remains the later xVA integration ticket
- the purpose of `QUA-886` is to supply their shared computational base, not to
  subsume them

### `QUA-889` Semantic simulation substrate: factor-state family IR and runtime contract

Objective:

Land the first checked simulation-family boundary whose primary state contract
is factor-state rather than bounded single-state.

Mathematical payload:

```text
X_t in E subseteq R^d
dX_t = mu(t, X_t) dt + Sigma(t, X_t) dW_t
```

Scope:

- widen or companion the current Monte Carlo typed surfaces so the public family
  boundary admits `d > 1`
- add explicit state/factor/process metadata and runtime problem-spec semantics
- preserve a compatibility path from the current single-state event-aware lane

Acceptance:

- at least one vector-state process is representable as a first-class family
  instance
- existing bounded single-state routes remain valid
- the docs state the supported dimensional boundary honestly

### `QUA-890` Semantic simulation substrate: event and observable lowering onto factor-state programs

Objective:

Project the existing universal `EventProgramIR` / `ControlProgramIR` boundary
onto factor-state simulation without creating a second semantic timeline.

Mathematical payload:

```text
Y_{i,0} given
O_{i,ell} = h_{i,ell}(X_{t_i})
Y_{i,ell} = T_{i,ell}(Y_{i,ell-1}, O_{i,ell})
Y_{i+1,0} = Y_{i,p_i}
L = G(Y_{m,p_m}, X_{t_m})
```

Scope:

- preserve same-date phase ordering from the universal event program
- separate latent state `X_t`, derived observables `O_{i,ell}`, and contract
  state `Y_{i,ell}`
- identify the widening points in `trellis/agent/family_lowering_ir.py` and
  `trellis/models/monte_carlo/event_aware.py`

Acceptance:

- same-date phase ordering is explicit and testable
- the contract admits vector-state inputs and does not collapse to scalar-only
  assumptions
- the current bounded event-aware lane has a documented compatibility path onto
  the broader substrate

### `QUA-891` Semantic simulation substrate: factor-state projection onto valuation-facing market views

Objective:

Introduce the explicit simulated-market projection

```text
Phi_t : E -> M_t
```

so downstream valuation consumes projected market views instead of raw state
coordinates.

Mathematical payload:

```text
M_{i,n} = Phi_{t_i}(X_{t_i}^{(n)})
q_j(t_i) = phi_j(t_i, X_{t_i})
P(t, T) = phi_P(t, T, X_t)
L(t; T_1, T_2) = phi_L(t, T_1, T_2, X_t)
```

Scope:

- define the typed projection contract from factor-state onto valuation-facing
  market views
- make explicit whether the result is a full `MarketState`, a lightweight
  simulated market view, or a narrower valuation projection
- support the rate-facing projection surfaces needed by later swap valuation

Acceptance:

- projected market values are reproducible from `(t, X_t)` plus metadata
- at least one rate-facing projection path is representable
- raw factor coordinates stop being the implicit downstream market interface

### `QUA-892` Conditional valuation: reusable intermediate-date regression and continuation contract

Objective:

Generalize the existing continuation machinery into a reusable intermediate-date
conditional valuation service.

Mathematical payload:

```text
V_a^N(t, x) = N_t E^{Q^N}[ H_a / N_T | X_t = x ]
V_{a,i}(x) = V_a^N(t_i, x)
V_{a,i}(x) ~= sum_{k=1}^K beta_{a,i,k} phi_k(x)
```

Scope:

- expose a runtime API for intermediate-date conditional valuation
- reuse existing continuation / regression implementations where possible
- publish estimator diagnostics rather than route-local regression glue

Acceptance:

- intermediate-date valuation is a reusable runtime surface over factor-state
- at least one proving workflow uses the surface outside a Bermudan-only code
  path
- the contract admits exact, fitted, and later alternative conditional models,
  with explicit train/eval metadata whenever fitted surfaces feed exposure
  statistics

### `QUA-893` Future-value cube: trade-date-path valuation tensor and projections

Objective:

Land the stable output object for simulated future values over trade, date, and
path axes.

Mathematical payload:

```text
C in R^{|A| x (m+1) x Npaths}
C_{a,i,n} = V_a^clean(t_i^+, X_{t_i}^{(n)})
```

Required semantics:

```text
1. time-t_i value, not as-of PV
2. post-event value at t_i^+
3. pre-netting, pre-collateral, pre-CVA, pre-DVA, pre-FVA
4. C_{a,i,n} = 0 after final settlement on the cube grid
```

Scope:

- define `FutureValueCube` and its metadata/provenance surface
- keep it distinct from `ScenarioResultCube`
- add projections and summarizers that preserve explicit trade/date/path
  semantics

Acceptance:

- one checked cube object carries stable trade/date/path semantics
- post-settlement zeroing and phase ordering are explicit and tested
- downstream consumers can read projections without reconstructing hidden
  semantics ad hoc

### `QUA-894` Simulation proof: interest-rate swap future-value runtime through `T52`

Objective:

Prove the substrate on a real workflow anchored on held legacy task `T52`.

Mathematical payload:

```text
C_{i,n} = V_swap^clean(t_i^+, X_{t_i}^{(n)})
E_i^+(n) = max(C_{i,n}, 0)
EE(t_i) = (1 / Npaths) sum_n E_i^+(n)
```

Scope:

- route one interest-rate swap future-value workflow through the new substrate
- use task-runtime / benchmark / proof machinery rather than an ad hoc harness
- document exactly which slice of `T52` is now supported and which xVA work
  remains downstream

Acceptance:

- one checked swap workflow emits a future-value cube on an observation grid
- the proof uses the same validation machinery recent computational lanes use
- positive-exposure sanity is available only as a validation projection, not as
  a substitute for downstream institutional tickets

### `QUA-895` Simulation substrate: docs, benchmark surfaces, and limitation truth

Objective:

Make the substrate measurable, documented, and truthful at closeout.

Mathematical contract to document:

```text
X_t in E subseteq R^d
M_t = Phi_t(X_t)
V_a(t, x) = N_t E^{Q^N}[ H_a / N_T | X_t = x ]
C_{a,i,n} = V_a^clean(t_i^+, X_{t_i}^{(n)})
```

Scope:

- update official docs across `docs/quant/`, `docs/developer/`, and
  `docs/user_guide/`
- add or harden benchmark / task-runtime coverage for the first proving path
- update `LIMITATIONS.md`
- sync the plan mirror and closeout notes with the actual supported boundary

Acceptance:

- docs explain the supported mathematical contract and non-goals clearly
- benchmark or task-runtime coverage exists for the first proving path
- `LIMITATIONS.md` and the plan mirror match the real support boundary

## Recommended Sequencing Constraints

- Do not start the simulated-market projection slice before the factor-state
  family boundary exists.
- Do not start the future-value cube slice before at least one proving
  conditional-valuation path exists.
- Do not treat the institutional bridge slice as the place to invent the core
  substrate.
- Keep netting, collateral, and xVA outside the first three slices unless a
  smaller upstream primitive is truly required.

## Agent Pickup Directive

If you hand this file to a coding agent before child tickets exist, the agent
should:

1. read `QUA-886`
2. audit the current codebase against the mathematical target above
3. propose or create the smallest first child ticket from the ordered queue
4. avoid coding directly against the umbrella epic without narrowing the scope

If child tickets already exist, the agent should:

1. pick the earliest non-`Done` child ticket whose prerequisites are satisfied
2. implement only that slice
3. update Linear first
4. then update this file

## Success Condition

This plan is successful when Trellis can say, precisely and honestly, that it
has a reusable factor-state simulation substrate with:

- explicit latent state
- explicit market projection
- explicit conditional valuation
- explicit future-value cube semantics

and that statement is true even if xVA is still unfinished.
