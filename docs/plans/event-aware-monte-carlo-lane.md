# Event-Aware Monte Carlo Lane Plan

## Purpose

This document defines the implementation queue for a generic event-aware,
state-aware Monte Carlo family in Trellis.

The goal is to stop treating Monte Carlo support as a mix of:

- one generic GBM fallback route
- a few product-specific checked helpers
- a narrow basket-specific lowering IR

Instead, Trellis should compile a bounded class of products into one typed
Monte Carlo family that can express:

- a typed Markov state model
- deterministic event timelines
- explicit reduced-storage path requirements
- payoff reducers assembled from event/path state
- calibration and quote-regime prerequisites when method comparison depends on
  them

This plan is intentionally bounded. It is not a universal Monte Carlo program
for every future exotic.

## Why This Plan Exists

The current repo proves two things at once:

1. the low-level Monte Carlo substrate is already richer than the compiler
   currently exposes, and
2. live canary failures still fall back to mathematically wrong generic routes
   when a product needs typed state/event semantics.

The clearest evidence is `T73`:

- the product is a European rate-style swaption with analytical, tree, and
  Monte Carlo comparison targets
- the current compiler has no typed Monte Carlo family IR for schedule-driven
  rate products
- the runtime therefore falls back to the generic `monte_carlo_paths` route,
  which binds `GBM + MonteCarloEngine`
- that route is not the right mathematical family for a Hull-White short-rate
  swaption

At the same time, the repo already ships reusable lower-level pieces:

- reduced-storage Monte Carlo path-state contracts in
  `trellis/models/monte_carlo/path_state.py`
- deterministic event replay in
  `trellis/models/monte_carlo/event_state.py`
- early-exercise control primitives in `trellis/models/monte_carlo/`
- typed model/calibration bindings elsewhere in the stack

So the missing layer is not “one more swaption helper.” The missing layer is a
typed Monte Carlo compiler family that can lower semantic contracts into those
existing runtime substrates.

## Repo-Grounded Current State

### Current strengths to preserve

Trellis already ships:

- `SemanticContract`, `ProductIR`, and family-level lowering boundaries
- a reduced-storage Monte Carlo engine and path-state contracts
- deterministic path-event replay helpers
- typed basket Monte Carlo lowering through `CorrelatedBasketMonteCarloIR`
- typed calibration, quote-map, and `EngineModelSpec` surfaces elsewhere in the
  pricing stack

This plan must build on those boundaries. It should not replace them with a
single universal numerical IR.

### Current gap

The Monte Carlo compiler path is still fragmented:

- `CorrelatedBasketMonteCarloIR` is the only real typed Monte Carlo family IR
- `monte_carlo_paths` is broad in name but narrow in actual mathematics
- route assembly does not preserve typed process/model semantics for products
  like rate-style swaptions
- deterministic event schedules are not lowered into explicit MC timelines for
  generic single-state products
- method comparison flows do not always enforce a shared quote/calibration
  regime before comparing outputs

That is why a task like `T73` can have healthy analytical and tree legs but a
broken Monte Carlo leg and a misleading cross-validation failure.

## Design Summary

The target shape is:

```text
SemanticContract
  -> ProductIR
  -> EventAwareMonteCarloIR
  -> MonteCarloProblemSpec
  -> generic event/state-aware simulation engine
```

The key idea is:

- keep product semantics in the existing contract/compiler layers
- add one richer Monte Carlo family IR for bounded single-state problems
- compile schedules and typed event-machine semantics into explicit Monte Carlo
  timelines and path requirements
- express vanilla Monte Carlo pricing as the simplest instance of this family
- keep comparison/calibration normalization explicit instead of implicit in
  generated adapters

## Design Guardrails

### 1. This is a bounded family IR, not a universal solver IR

The shipped compiler boundary remains:

`SemanticContract -> ProductIR -> family IR -> numerical backend`

This work should add a new Monte Carlo family IR sibling, not reopen the whole
pricing stack.

### 2. The first tranche is single-state / one-factor only

The initial lane should support only one-factor Markov state models.

That still covers useful products such as:

- vanilla European equity options
- local-vol vanilla equity options
- European rate-style swaptions under one-factor short-rate models

### 3. Event support must be explicit and typed

The Monte Carlo lane should not consume raw schedule state indefinitely. It
should consume explicit event buckets, replay specs, and reduced path
requirements compiled from typed semantics.

### 4. Control support is intentionally bounded in v1

The first tranche must support:

- `identity`

It may preserve the shape for future controller extensions, but it should not
attempt Longstaff-Schwartz, issuer-control, or multi-controller problems on the
critical path for this workstream.

### 5. Quote and calibration regime must be explicit for comparison tasks

When an analytical method and a model-based method are compared, the Monte
Carlo family should carry the calibration/quote-map preconditions needed to put
both outputs in the same valuation regime before comparison.

### 6. Existing specialized routes are migrated by subsumption, not immediate removal

Do not delete `monte_carlo_paths` or `local_vol_monte_carlo` first.

Instead:

- introduce the new Monte Carlo family IR
- compile the simple vanilla MC routes into it
- keep existing route ids as compatibility wrappers while traces, docs, and
  runtime readers migrate
- only collapse the wrappers once the event-aware MC path is stable

## Target Typed Surfaces

The exact names may change, but the conceptual surfaces should be:

- `EventAwareMonteCarloIR`
- `MCStateSpec`
- `MCProcessSpec`
- `MCEventTimelineSpec`
- `MCEventSpec`
- `MCPathRequirementSpec`
- `MCPayoffReducerSpec`
- `MCControlSpec`
- `MCMeasureSpec`
- `MCCalibrationBindingSpec`
- `MonteCarloProblemSpec`

The v1 event/path contract should be small and explicit:

- terminal-only payoff evaluation
- snapshot-at-event evaluation
- deterministic event replay from typed schedules
- bounded running reducers for schedule-driven payoffs

## Supported v1 Boundary

### In scope

- one-factor Markov state only
- deterministic event schedules
- process families:
  - `gbm_1d`
  - `local_vol_1d`
  - `hull_white_1f`
- control styles:
  - `identity`
- path/state requirements:
  - terminal-only
  - explicit event snapshots
  - bounded event replay
  - bounded running reducers
- explicit calibration/quote bindings for comparison-critical rate products

### Out of scope

- Longstaff-Schwartz / Bermudan MC in this tranche
- issuer-control Monte Carlo
- multi-controller game structures
- multi-asset basket/path-selection migration
- stochastic-volatility or multi-factor short-rate simulation families
- hybrid credit/equity/rates Monte Carlo families

## Proof Products

The first proving set should be:

1. vanilla European equity option
   - `identity`
   - terminal-only reduced-state payoff

2. local-vol vanilla equity option
   - same family IR, different process mapping

3. European payer swaption under Hull-White 1-factor
   - deterministic exercise/swap schedule
   - event-aware payoff reducer at exercise
   - explicit Black76/Hull-White calibration normalization
   - live canary `T73`

This gives one simplest identity case, one alternative process family, and one
schedule-driven rate case that proves the abstraction is not equity-specific.

## Ordered Delivery Queue

1. umbrella: event-aware/state-aware Monte Carlo lane
2. Monte Carlo IR: event-aware family IR and admissibility contract
3. Monte Carlo compiler: lower schedules and event machines into MC timelines
   and path requirements
4. Monte Carlo numerics: generic event-aware problem assembly and payoff
   reducers
5. Monte Carlo migration: express generic vanilla MC routes through the new
   family IR
6. Semantic swaption: preserve rate conventions, model bindings, and
   quote-map/calibration prerequisites through comparison assembly
7. Monte Carlo proof route: Hull-White European swaption MC and `T73` recovery
8. Monte Carlo lane: docs, observability, and compatibility cleanup

## Critical Path

The critical path is:

`MC IR -> MC compiler -> MC numerics -> MC vanilla migration -> {swaption semantics, T73 proof route} -> cleanup`

The swaption semantics slice and the generic vanilla migration can overlap once
the new family IR is stable, but `T73` should not attempt live recovery before
both are in place.

## Dependency Notes

- this plan is related to `docs/plans/canary-suite-stabilization.md` because
  `T73` should be recovered through the new MC family rather than through a
  one-off swaption adapter
- this plan is intentionally parallel to `docs/plans/event-aware-pde-lane.md`
  so the PDE and Monte Carlo stacks converge on the same compiler shape:
  bounded reusable family IRs instead of proliferating product lanes
- `CorrelatedBasketMonteCarloIR` remains in place for now; basket migration is a
  separate follow-on once the single-state family is stable

## Agent Intake Bundle

Each coding agent assigned to this workstream should begin with:

- this plan doc
- `AGENTS.md`
- `docs/quant/contract_algebra.rst`
- `docs/quant/pricing_stack.rst`
- `docs/developer/dsl_system_design_review.md`
- `docs/plans/canary-suite-stabilization.md`
- the current Linear ticket body plus any upstream blockers
- the code paths under:
  - `trellis/agent/family_lowering_ir.py`
  - `trellis/agent/dsl_lowering.py`
  - `trellis/agent/semantic_contracts.py`
  - `trellis/agent/route_registry.py`
  - `trellis/agent/lane_obligations.py`
  - `trellis/models/monte_carlo/path_state.py`
  - `trellis/models/monte_carlo/event_state.py`
  - `trellis/models/monte_carlo/engine.py`
  - `trellis/models/calibration/rates.py`
- the latest `T73` task-run artifacts and diagnostics

## Linear Mirror

Status mirror last synced: `2026-04-10`

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-719` | Monte Carlo lane: event-aware single-state path family | Done |

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-720` | Monte Carlo IR: event-aware family IR and admissibility contract | Done |
| `QUA-721` | Monte Carlo compiler: lower schedules and event machines into MC timelines and path requirements | Done |
| `QUA-722` | Monte Carlo numerics: generic event-aware problem assembly and payoff reducers | Done |
| `QUA-723` | Monte Carlo migration: express vanilla MC routes through `EventAwareMonteCarloIR` | Done |
| `QUA-724` | Semantic swaption: preserve conventions and calibration bindings through comparison assembly | Done |
| `QUA-725` | Monte Carlo proof route: Hull-White European swaption MC and `T73` recovery | Done |
| `QUA-726` | Monte Carlo lane: docs, observability, and compatibility cleanup | Done |
