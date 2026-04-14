# Event-Aware PDE Lane Plan

## Purpose

This document defines the implementation queue for a generic event-aware
one-dimensional PDE lane in Trellis.

The goal is to stop treating PDE support as a mix of one narrow vanilla route
plus ad hoc product-specific workarounds. Instead, Trellis should compile a
bounded class of products into one typed PDE family that can express:

- state evolution between event dates
- deterministic event-time transforms
- single-controller obstacle projections

This plan is intentionally bounded. It is not a universal PDE program for all
future exotics.

## Why This Plan Exists

The current repo proves two things at once:

1. the numerical method exists for at least one important schedule-driven PDE
   problem, and
2. the production compiler/runtime does not yet expose that method through a
   general enough PDE lane.

The clearest evidence is the callable-bond Hull-White PDE:

- [test_t10_callable_pde.py](../../tests/test_tasks/test_t10_callable_pde.py)
  already carries a working hand-written reference implementation
- live canary `T17` still fails because the runtime lowers the PDE side into
  the generic `pde_theta_1d` route, which has no typed event/control lowering
  for `issuer_min` and `schedule_state`

At the same time, the lattice side already has the right architectural shape:

- `ExerciseLatticeIR` in
  [family_lowering_ir.py](../../trellis/agent/family_lowering_ir.py)
- `EventOverlaySpec` in
  [trees/algebra.py](../../trellis/models/trees/algebra.py)
- richer control/event obligations in
  [lane_obligations.py](../../trellis/agent/lane_obligations.py)

The PDE side should be brought up to the same typed semantic standard rather
than patched with more one-off helpers.

## Repo-Grounded Current State

### Current strengths to preserve

Trellis already ships:

- `SemanticContract`, typed event-machine normalization, and controller
  protocols
- `ProductIR` as the shared checked compiler summary
- family-specific lowering IRs as the executable compiler boundary
- typed route admissibility and validation contracts
- a stable vanilla-equity PDE path through `VanillaEquityPDEIR`
- a stable lattice path for schedule-driven control through `ExerciseLatticeIR`

This plan must build on those boundaries. It should not replace them with a
single universal numerical IR.

### Current gap

The current PDE path is still too narrow:

- `VanillaEquityPDEIR` is the only typed PDE family IR
- the generic `pde_theta_1d` route is broad in name but narrow in actual
  admissibility
- event-time schedule semantics are not lowered into explicit PDE transforms
- PDE lane obligations are still generic

That is why a product like callable bond can succeed through the tree lane and
fail through the PDE lane in the same task.

## Design Summary

The target shape is:

```text
SemanticContract
  -> ProductIR
  -> EventAwarePDEIR
  -> PDEProblemSpec
  -> generic 1D rollback engine
```

The key idea is:

- keep product semantics in the existing contract/compiler layers
- add one richer PDE family IR
- compile schedules and typed event-machine semantics into explicit PDE event
  transforms
- use one bounded rollback substrate for multiple products

Vanilla-equity PDE should become the simplest instance of this new family, not
an unrelated permanent branch.

## Design Guardrails

### 1. This is a bounded family IR, not a universal solver IR

The shipped compiler boundary remains:

`SemanticContract -> ProductIR -> family IR -> numerical backend`

This work should add a new PDE family IR sibling, not reopen the whole pricing
stack.

### 2. The first tranche is 1D only

The initial lane should support only one-dimensional Markov problems.

That keeps the numerical substrate reviewable while still covering valuable
products such as:

- vanilla European equity options
- event-aware holder-control equity options
- callable bonds under one-factor short-rate models

### 3. Event support must be explicit and typed

The PDE lane should not consume raw schedule state indefinitely. It should
consume explicit event buckets and transforms compiled from typed semantics.

### 4. Control support is bounded to one controller

The first tranche supports:

- `identity`
- `holder_max`
- `issuer_min`

It does not attempt multi-controller game problems.

### 5. `VanillaEquityPDEIR` is migrated by subsumption

Do not delete it first.

Instead:

- introduce the new PDE family IR
- compile vanilla-equity PDE into it
- keep `VanillaEquityPDEIR` as a short-lived compatibility wrapper
- remove it only after the migrated path is stable

## Target Typed Surfaces

The exact names may change, but the conceptual surfaces should be:

- `EventAwarePDEIR`
- `PDEStateSpec`
- `PDEOperatorSpec`
- `PDETerminalSpec`
- `PDEEventTimelineSpec`
- `PDEEventTransformSpec`
- `PDEControlSpec`
- `PDEBoundarySpec`
- `PDEProblemSpec`

The event transform set in tranche 1 should be small and explicit:

- `add_cashflow`
- `project_max`
- `project_min`
- `state_remap`

## Supported v1 Boundary

### In scope

- one-dimensional Markov state only
- deterministic event schedules
- operator families:
  - `black_scholes_1d`
  - `local_vol_1d`
  - `hull_white_1f`
- control styles:
  - `identity`
  - `holder_max`
  - `issuer_min`
- event transforms:
  - `add_cashflow`
  - `project_max`
  - `project_min`
  - bounded `state_remap`

### Out of scope

- high-dimensional PDE systems
- multi-controller game structures
- stochastic-volatility PDE systems
- hybrid credit/equity/rates PDE systems
- rough-vol or non-Markov PDE families

## Proof Products

The first proving set should be:

1. vanilla European equity option
   - `identity`
   - no event transforms beyond maturity

2. bounded holder-controlled event-aware equity option
   - `holder_max`
   - deterministic event schedule

3. callable bond under Hull-White 1-factor
   - `issuer_min`
   - coupon and call-date event transforms
   - live canary `T17`

This gives one identity case, one holder-control case, and one issuer-control
case.

## Ordered Delivery Queue

1. `QUA-711` PDE lane: event-aware 1D rollback family
2. `QUA-712` PDE IR: event-aware family IR and admissibility contract
3. `QUA-713` PDE compiler: lower schedules and event machines into PDE timelines
4. `QUA-714` PDE numerics: generic 1D event-aware rollback problem assembly
5. `QUA-715` PDE migration: express vanilla theta-method route through `EventAwarePDEIR`
6. `QUA-716` PDE proof route: holder-max event-aware equity path
7. `QUA-717` PDE proof route: issuer-min Hull-White callable bond and `T17` recovery
8. `QUA-718` PDE lane: docs, observability, and compatibility cleanup

## Critical Path

The critical path is:

`QUA-712 -> QUA-713 -> QUA-714 -> QUA-715 -> {QUA-716, QUA-717} -> QUA-718`

`QUA-716` and `QUA-717` can run in parallel once the new family IR, lowering,
and generic rollback substrate are in place and the vanilla route migration has
stabilized.

## Dependency Notes

- `QUA-711` is related to `QUA-700` because the canary workstream depends on
  this architecture for `T17`
- `QUA-706` remains the canary acceptance ticket for `T17`, but it is now
  blocked by `QUA-717`
- this plan intentionally avoids a helper-only local fix for `T17`

## Agent Intake Bundle

Each coding agent assigned to this workstream should begin with:

- this plan doc
- `AGENTS.md`
- `docs/quant/contract_algebra.rst`
- `docs/quant/lattice_algebra.rst`
- `docs/developer/dsl_system_design_review.md`
- `doc/plan/done__canary-suite-stabilization.md`
- the current Linear ticket body plus any upstream blockers
- the code paths under:
  - `trellis/agent/family_lowering_ir.py`
  - `trellis/agent/dsl_lowering.py`
  - `trellis/agent/lane_obligations.py`
  - `trellis/agent/semantic_contracts.py`
  - `trellis/models/equity_option_pde.py`
  - `trellis/models/pde/`
  - `tests/test_tasks/test_t10_callable_pde.py`

## Linear Ticket Mirror

These tables mirror the current Linear tickets for this workstream and their
intended implementation order.

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This file is the repo-local mirror for subsequent agents.
- Implement the earliest ticket in the ordered queue below whose status is not
  `Done` and whose blockers are satisfied.
- Do not mark a row `Done` here before the corresponding Linear issue is
  actually closed.
- When a ticket changes behavior, APIs, compiler traces, runtime workflow, or
  operator expectations, update the relevant official docs in the same
  implementation closeout.

Status mirror last synced: `2026-04-07`

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-711` | Done |

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-712` | Event-aware PDE family IR and admissibility contract | Done |
| `QUA-713` | Lower schedules and event machines into PDE timelines | Done |
| `QUA-714` | Generic 1D event-aware rollback problem assembly | Done |
| `QUA-715` | Migrate vanilla theta-method PDE onto `EventAwarePDEIR` | Done |
| `QUA-716` | Holder-max event-aware equity proof route | Done |
| `QUA-717` | Issuer-min Hull-White callable-bond proof route and `T17` recovery | Done |
| `QUA-718` | Docs, observability, and compatibility cleanup | Done |

### Adjacent Shared Work

| Ticket | Reason |
| --- | --- |
| `QUA-700` | Canary stabilization umbrella that owns `T17` as a release-signal concern |
| `QUA-706` | Canary acceptance ticket for `T17`, now architecture-blocked on `QUA-717` |
| `QUA-707` | Adjacent analytical canary workstream, not owned by this PDE lane |
| `QUA-708` | Adjacent transform/Heston canary workstream, not owned by this PDE lane |

## Validation Posture

This queue should validate at three levels.

### Local

- family-lowering IR tests
- DSL/lane-obligation tests
- PDE event-transform and control-projection tests
- operator/problem-assembly tests

### Regional

- vanilla PDE migration tests
- bounded holder-control event-aware PDE tests
- callable-bond PDE/tree parity tests

### Global

- live canary reruns for `T13` and `T17`
- trace/review-surface validation for the new family IR
- final docs and observability closeout

## Official Docs Expected To Change

If this plan is implemented as intended, the main official docs that should be
updated during closeout are:

- `docs/quant/contract_algebra.rst`
- `docs/quant/pricing_stack.rst`
- `docs/quant/lattice_algebra.rst`
- `docs/developer/dsl_system_design_review.md`
- `docs/developer/task_and_eval_loops.rst`
- `docs/user_guide/pricing.rst`
