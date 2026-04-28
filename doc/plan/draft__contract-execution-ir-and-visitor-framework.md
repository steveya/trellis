# Contract Execution IR And Visitor Framework

## Status

Draft execution mirror. Linear epic and phased child queue filed for
implementation sequencing.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-886 — Semantic simulation substrate: factor-state valuation,
  market projection, and future-value cube
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `doc/plan/draft__external-prior-art-adoption-map.md`
- `doc/plan/active__semantic-simulation-substrate.md`
- `docs/quant/contract_algebra.rst`
- `docs/quant/contract_ir.rst`
- `docs/quant/static_leg_contract_ir.rst`
- `docs/quant/dynamic_contract_ir.rst`
- Existing implementation surfaces:
  - `trellis/agent/semantic_contract_compiler.py`
  - `trellis/agent/family_lowering_ir.py`
  - `trellis/core/payoff.py`
  - `trellis/models/contingent_cashflows.py`
  - `trellis/models/cashflow_engine/*`
  - `trellis/models/monte_carlo/event_aware.py`
  - `trellis/models/pde/event_aware.py`
  - `trellis/models/monte_carlo/simulation_substrate.py`
  - `trellis/instruments/_agent/*`

## Linked Linear

- `QUA-975` - Semantic execution: contract execution IR and visitor framework

## Linear Ticket Mirror

Status mirror last synced: `2026-04-28`.

Umbrella:

| Ticket | Status | Scope |
| --- | --- | --- |
| `QUA-975` | Backlog | Semantic execution: contract execution IR and visitor framework |

Implementation queue:

| Order | Ticket | Status | Objective | Hard blocker |
| --- | --- | --- | --- | --- |
| 1 | `QUA-976` | Done | XIR.0 - execution seam and authority boundary | none |
| 2 | `QUA-977` | Backlog | XIR.1 - static-leg execution visitors and repricing slice | `QUA-976` |
| 3 | `QUA-978` | Backlog | XIR.2 - execution-backed payoff and adapter migration | `QUA-977` |
| 4 | `QUA-979` | Backlog | XIR.3 - dynamic execution bridge for callable structures | `QUA-978` |
| 5 | `QUA-980` | Backlog | XIR.4 - simulation and future-value bridge | `QUA-979` |
| 6 | `QUA-981` | Backlog | XIR.5 - aggregation and xVA precursor visitors | `QUA-980` |

## Purpose

Introduce one reusable execution substrate between Trellis' semantic
authority surfaces and its family-specific numerical lanes.

The target is not "add a scripting language." The target is:

- keep `SemanticContract`, `ContractIR`, `StaticLegContractIR`, and
  `DynamicContractIR` as the authoritative statement of contract meaning
- lower those semantic objects into a shared execution IR
- run reusable visitor passes over that execution IR to derive:
  - schedules
  - market requirements
  - deterministic or contingent cashflow programs
  - event and state programs
  - bridges onto existing Monte Carlo, PDE, tree, and future-value
    runtimes
- reduce the amount of product-local repricing logic currently living in
  `_agent/{product}.py` adapters

This plan exists because current Trellis has strong semantic boundaries
and increasingly reusable numerical runtimes, but it still lacks a
stable common execution object between them.

## Current Problem

Today the repository has three important strengths:

1. semantic meaning is already separated from valuation policy in
   `SemanticContract` and `ValuationContext`
2. bounded sibling semantic tracks now exist for payoff-expression,
   static-leg, and dynamic contracts
3. reusable lower runtimes now exist for event-aware Monte Carlo, 1D
   event-aware PDE, and the factor-state simulation substrate

But there is still a structural execution gap:

- some `_agent` adapters are already thin helper shells, but others
  still implicitly own schedule assembly, event wiring, or market-input
  resolution
- deterministic cashflow expansion, event compilation, and runtime
  requirement derivation are still spread across family helpers rather
  than one shared compiler/runtime layer
- repricing, bump-and-reprice, future-value, and later netting/xVA work
  do not yet share one contract execution artifact

The practical effect is that `_agent/{product}.py` is still too often a
real repricing boundary instead of a compatibility shell.

## Design Rule

The new layer must obey four rules.

### 1. Semantic authority stays upstream

The new execution layer must never replace:

- `SemanticContract`
- `ContractIR`
- `StaticLegContractIR`
- `DynamicContractIR`

Those remain the source-of-truth semantic objects.

### 2. Execution IR is model-free

The execution IR should represent undiscounted contractual obligations,
events, observables, and state evolution. It must not directly encode:

- route ids
- backend-binding ids as semantics
- discounting policy
- model choice
- measure choice

Those remain downstream concerns owned by `ValuationContext`,
market-binding, and family-specific lowerings.

### 3. Visitor passes are reusable compiler/runtime passes

Visitors should derive stable artifacts such as schedules, event
timelines, requirements, cashflow expansions, and simulation bridges.
They should not become a second prompt layer or a free-form product
taxonomy.

### 4. Family IRs remain narrow consumers

`EventAwareMonteCarloIR`, `FactorStateSimulationIR`,
`TransformPricingIR`, `ExerciseLatticeIR`, and future family IRs remain
useful narrow numerical contracts. The new execution IR sits above them
and feeds them. It does not replace them.

## Target Architecture

The intended flow is:

```text
SemanticContract
  + ContractIR / StaticLegContractIR / DynamicContractIR
    -> ContractExecutionIR
    -> visitor passes
    -> family lowering IRs / execution runtime / future-value substrate
    -> checked numerical helpers
```

Or, in more operational terms:

```text
semantic authority
  -> execution authority
  -> reusable compiler/runtime passes
  -> numerical execution
```

## What To Add

### New package

Add a new package:

- `trellis/execution/`

This package should be deterministic library/runtime code. It should not
live under `trellis/agent/knowledge/`, and it should not be LLM-generated
source.

### New modules

The first module set should be:

| Path | Purpose |
| --- | --- |
| `trellis/execution/ir.py` | Frozen dataclasses for `ContractExecutionIR` and supporting execution nodes |
| `trellis/execution/compiler.py` | Lower semantic sibling IRs onto execution IR |
| `trellis/execution/bindings.py` | Bind valuation/runtime requirements onto execution observables and schedule nodes |
| `trellis/execution/runtime.py` | Execute deterministic and contingent compiled obligations |
| `trellis/execution/summary.py` | Stable operator-facing summary for traces, diagnostics, and review |
| `trellis/execution/visitors/normalize.py` | Canonical execution-form normalization |
| `trellis/execution/visitors/schedule.py` | Schedule and event-time materialization |
| `trellis/execution/visitors/requirements.py` | Route-free market and state requirement derivation |
| `trellis/execution/visitors/cashflow_expand.py` | Deterministic and contingent cashflow expansion |
| `trellis/execution/visitors/event_compile.py` | Projection onto event-aware family runtimes |
| `trellis/execution/visitors/simulation_bridge.py` | Projection onto factor-state / future-value workflows |

## Candidate Surface

This document does not lock the final ADT, but the minimal useful shape
should look like:

```text
ContractExecutionIR =
    { source_track: SourceTrack
    ; obligations: tuple[ExecutionObligation, ...]
    ; observables: tuple[ObservableBinding, ...]
    ; event_plan: ExecutionEventPlan
    ; state_schema: ExecutionStateSchema
    ; decision_program: DecisionProgram | None
    ; settlement_program: SettlementProgram
    ; requirement_hints: RequirementHints
    ; execution_metadata: ExecutionMetadata
    }

ExecutionObligation =
    | KnownCashflow(...)
    | CouponLegExecution(...)
    | PeriodRateOptionStripExecution(...)
    | ContingentSettlement(...)
    | PrincipalExchange(...)

ObservableBinding =
    | SpotObservableRef(...)
    | ForwardRateObservableRef(...)
    | SwapRateObservableRef(...)
    | CurveQuoteObservableRef(...)
    | SurfaceQuoteObservableRef(...)

ExecutionEvent =
    | FixingEvent(...)
    | ObservationEvent(...)
    | AccrualBoundaryEvent(...)
    | CouponEvent(...)
    | PaymentEvent(...)
    | DecisionEvent(...)
    | TerminationEvent(...)
    | StateResetEvent(...)
```

Key design intent:

- payoff-expression contracts lower into one or more
  `ContingentSettlement` obligations plus observation events
- static-leg contracts lower into explicit coupon, option-strip, and
  exchange obligations
- dynamic contracts lower into a static execution substrate plus
  explicit decision/state/termination programs

## First Visitor Pass Set

### `normalize`

Purpose:

- canonicalize execution structure
- remove duplicate schedule/event declarations
- stabilize ordering for traces, matching, and tests

Why it matters:

- semantically equivalent contracts should produce the same execution
  artifact even when they arrived through different upstream wording or
  sibling IR paths

### `schedule`

Purpose:

- compile dated event buckets and schedule roles from execution nodes
- produce concrete accrual, fixing, payment, and settlement sequences
- preserve same-day phase ordering

Why it matters:

- the repo already has schedule-dependent and event-aware runtimes, but
  schedule extraction is still too family-local

### `requirements`

Purpose:

- derive market requirements from execution structure
- derive state obligations and timeline-role requirements
- eventually back `Payoff.requirements` and route admissibility from one
  route-free source

Why it matters:

- product adapters should stop hand-writing capability sets that are
  already implicit in the contract structure

### `cashflow_expand`

Purpose:

- expand static coupon and principal structure into deterministic or
  contingent cashflow programs
- reuse `trellis.models.contingent_cashflows` and
  `trellis.models.cashflow_engine/*` instead of reproducing those
  mechanics product-locally

Why it matters:

- this is the shortest path from semantic leg structure to reusable
  repricing and future-value preparation

### `event_compile`

Purpose:

- project execution events and state updates onto existing narrow
  runtime families
- bridge onto:
  - `trellis.models.monte_carlo.event_aware`
  - `trellis.models.pde.event_aware`
  - later tree/event runtimes

Why it matters:

- Trellis already has event-aware runtimes; this pass lets semantic
  contract execution feed them without product-local hand assembly

### `simulation_bridge`

Purpose:

- project execution state, observables, and continuation structure onto
  the factor-state simulation substrate in
  `trellis.models.monte_carlo.simulation_substrate`
- emit the reusable observation/state contracts needed for
  `FutureValueCube`

Why it matters:

- repricing and future-value should eventually consume the same compiled
  contract execution artifact

## Integration With Existing Code

### Semantic compiler integration

Update:

- `trellis/agent/semantic_contract_compiler.py`

Add:

- `SemanticImplementationBlueprint.execution_ir`

Purpose:

- once a semantic track is admitted, the compiler should attach a route-
  free execution artifact beside existing `contract_ir`,
  `static_leg_contract_ir`, and dynamic/family-lowering surfaces

### Family lowering integration

Update:

- `trellis/agent/family_lowering_ir.py`
- family-specific lowering compilers

Purpose:

- family IRs should be able to consume visitor outputs from
  `ContractExecutionIR` instead of re-deriving schedules, event
  semantics, or requirements locally

The intended relationship is:

- semantic IRs say what the contract means
- execution IR says how the contract operationally unfolds
- family IRs say what a bounded numerical lane needs

### Runtime payoff integration

Update:

- `trellis/core/payoff.py`

Add:

- `ExecutionBackedPayoff` or `CompiledExecutionPayoff`

Purpose:

- let the public repricing boundary consume a compiled execution object
  directly
- reduce the amount of product-local logic in `_agent/{product}.py`

### `_agent` compatibility integration

Update:

- `trellis/instruments/_agent/*`
- builder/executor surfaces that cache or generate those modules

Purpose:

- the generated module becomes a compatibility shell:
  - carry spec
  - satisfy public payoff interface
  - delegate to execution-backed runtime or exact helper binding

This is the medium-term migration target, not a day-1 cutover.

### Cashflow and contingent-payment integration

Reuse:

- `trellis/models/contingent_cashflows.py`
- `trellis/models/cashflow_engine/*`

Purpose:

- deterministic and contingent payment expansion should reuse existing
  kernels rather than inventing a second private cashflow engine inside
  agent adapters or family routes

### Simulation and future-value integration

Reuse:

- `trellis/models/monte_carlo/simulation_substrate.py`
- `FactorStateSimulationIR`
- `FutureValueCube`

Purpose:

- let future-value and later netting/collateral/xVA work consume the
  same execution artifact as repricing

## First Proving Cohort

The first cohort should stay inside products that are already mostly
semantically explicit and where the missing piece is reusable execution
assembly rather than new pricing math.

Recommended proving cohort:

1. fixed-float IRS
2. float-float basis swap
3. fixed coupon bond
4. scheduled cap/floor strip
5. issuer-callable fixed coupon bond

Why this cohort:

- it exercises schedules, notionals, coupon formulas, payment events,
  and one bounded decision/termination wrapper
- it uses surfaces Trellis already has:
  - `StaticLegContractIR`
  - dynamic callable-bond decomposition
  - rate-cap-floor helpers
  - callable-bond tree/PDE helpers
- it avoids starting with barrier or basket exotics where the current
  representation challenge is different

## Phased Queue

### `XIR.0` — Execution seam and authority boundary

Objective:

- create `trellis/execution/`
- define `ContractExecutionIR`
- add `execution_ir` to the semantic blueprint
- add stable execution summaries for traces and diagnostics

Files:

- `trellis/execution/ir.py`
- `trellis/execution/compiler.py`
- `trellis/execution/summary.py`
- `trellis/agent/semantic_contract_compiler.py`

Exit criteria:

- admitted semantic builds can attach `execution_ir`
- no runtime cutover claim yet

### `XIR.1` — Static deterministic execution slice

Objective:

- lower `StaticLegContractIR` into execution IR
- land `normalize`, `schedule`, `requirements`, and `cashflow_expand`
- support execution-backed repricing for the static proving cohort

Files:

- `trellis/execution/visitors/normalize.py`
- `trellis/execution/visitors/schedule.py`
- `trellis/execution/visitors/requirements.py`
- `trellis/execution/visitors/cashflow_expand.py`
- `trellis/execution/runtime.py`

Exit criteria:

- IRS, bond, and cap/floor-strip repricing can run from execution-backed
  runtime without product-local schedule logic

### `XIR.2` — Public payoff and adapter migration

Objective:

- add `ExecutionBackedPayoff`
- migrate the static proving cohort `_agent` wrappers into compatibility
  shells over compiled execution

Files:

- `trellis/core/payoff.py`
- `trellis/instruments/_agent/*` for admitted cohort
- build/executor cache/reuse boundaries

Exit criteria:

- admitted `_agent` wrappers in the cohort no longer own real repricing
  logic

### `XIR.3` — Dynamic execution bridge

Objective:

- lower bounded `DynamicContractIR` wrappers onto execution IR
- land `event_compile`
- prove callable-bond execution through existing checked tree/PDE paths

Files:

- `trellis/execution/visitors/event_compile.py`
- dynamic lowering bridges
- callable-bond route/runtime integration

Exit criteria:

- the callable-bond cohort reuses one explicit execution event/state
  structure rather than product-local event assembly

### `XIR.4` — Simulation and future-value bridge

Objective:

- land `simulation_bridge`
- feed `FactorStateSimulationIR`, `ObservationProgram`, and
  `FutureValueCube` from execution artifacts for the admitted cohort

Files:

- `trellis/execution/visitors/simulation_bridge.py`
- `trellis/models/monte_carlo/simulation_substrate.py`
- relevant book/future-value consumers

Exit criteria:

- repricing and future-value workflows share one compiled execution
  substrate for the admitted cohort

### `XIR.5` — Aggregation and xVA precursors

Objective:

- add reusable aggregation/compression visitors
- prepare later consumers for netting, collateral, exposure, and xVA

Possible later passes:

- `compression.py`
- `netting.py`
- `xva_decorate.py`

Exit criteria:

- later institutional tickets can consume execution artifacts instead of
  inventing their own contract operational layer

## Non-Goals

- Do not replace semantic sibling IRs with one new universal execution
  language.
- Do not encode discounting, model choice, or measure choice directly in
  execution nodes.
- Do not claim route retirement from this layer alone.
- Do not start by rewriting all existing exotic adapters.
- Do not make visitor patterns the new product taxonomy.

## Acceptance Gates

This plan is only successful if it improves real runtime boundaries.

Minimum acceptance gates:

1. a bounded static-leg contract can compile into `execution_ir` and be
   repriced from it
2. the same execution artifact can support at least one bump-and-reprice
   path without rebuilding product-local schedule logic
3. the callable-bond cohort can project onto an existing dynamic
   numerical lane from execution structure
4. future-value workflows can consume the same execution artifact rather
   than a separate ad hoc contract representation
5. admitted `_agent` wrappers shrink into compatibility shells over
   execution-backed runtime or exact helper bindings

## Relationship To Prior Art

The relevant external guidance is:

- Peyton Jones / combinator contracts for the small compositional
  payoff-expression core
- ACTUS for the term / event / state split
- Marlowe for explicit continuation and event semantics
- Antoine Savine for the idea of an internal cashflow execution
  substrate with reusable preprocessing and transformation passes

The Trellis-specific difference is deliberate:

- semantic authority remains explicit and typed upstream
- execution IR is a lowering/runtime substrate, not the top semantic
  model
- valuation policy remains separate
- route admissibility, provenance, and review traces remain first-class

That is the boundary that keeps the execution layer useful without
collapsing the current semantic architecture into "just another
script-tree runtime."
