# Event / State / Control Contract Foundation

## Status

Draft. Parking-lot design document. Not yet an execution mirror and not
yet tied to a filed Linear child issue.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/draft__quoted-observable-contract-ir-foundation.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`
- `doc/plan/draft__contract-ir-phase-4-route-retirement.md`
- `doc/plan/draft__semantic-contract-target-and-trade-envelope.md`
- `doc/plan/draft__post-phase-4-semantic-closure-execution-plan.md`
- `doc/plan/draft__automatic-event-state-lowering-plan.md`
- `doc/plan/draft__discrete-control-lowering-plan.md`
- `doc/plan/draft__continuous-singular-control-lowering-plan.md`
- `doc/plan/draft__insurance-contract-overlay-foundation.md`
- `doc/plan/done__semantic-platform-hardening.md`
- `doc/plan/done__event-aware-pde-lane.md`
- `doc/plan/done__event-aware-monte-carlo-lane.md`
- `doc/plan/active__semantic-simulation-substrate.md`
- existing repo surfaces:
  - `trellis.agent.event_machine`
  - `trellis.agent.dsl_algebra`
  - `trellis.agent.semantic_contract_compiler`

## Purpose

Capture the future semantic track for contracts whose meaning depends on
ordered events, running state, automatic stopping rules, or explicit
holder/issuer control.

This track exists to prevent two failure modes:

- forcing dynamic products into static payoff or static leg shapes that
  erase their real semantics
- leaving dynamic semantics buried in route-local numerical code so the
  route-free compiler can never become authoritative

## Why This Is A Separate Track

The existing and planned static semantic homes are necessary but not
sufficient:

- payoff-expression IR says what a static payoff expression is
- quoted-observable IR says which quoted points a static payoff settles
  on
- static leg-based IR says which dated coupon or cashflow obligations
  exist

Many important products require more than any one of those:

- **autocallables / phoenix / snowball products**
  barrier or threshold observations, coupon memory, early redemption,
  and stateful termination
- **TARN / TARF structures**
  running accumulated coupon or gain state, target-triggered stopping,
  and schedule-dependent continuation
- **PRDC and related FX/rate hybrids**
  schedule-linked contingent coupons, sometimes with issuer call or
  target features
- **callable CMS-spread range-accrual notes**
  scheduled contingent coupon formula plus quote-linked condition plus
  issuer control
- **swing options**
  remaining-right inventory, refraction or volume constraints, and
  repeated exercise decisions
- **GMWB / GMxB contracts**
  account-value state, benefit-base state, and policyholder withdrawal
  control

Those are not just "harder lowerings." They are contracts with
additional semantic structure.

## Design Objective

The event/state/control track should let Trellis represent dynamic
contracts in a way that is:

- additive alongside payoff-expression, quoted-observable, and static
  leg-based semantic homes
- explicit about event ordering, state evolution, stopping rules, and
  controller role
- composable over static semantic subcontracts rather than creating a
  separate product-name universe
- independent of route ids, instrument strings, and backend binding ids
- suitable for multiple numerical lowerings: tree, PDE, Monte Carlo,
  dynamic programming, or future hybrid solvers

## Core Design Claim

The right long-run shape is not one product-specific node per exotic.

The right semantic target is closer to:

- a **base semantic contract** from the static domains
- plus an **event program**
- plus a **state schema**
- plus an optional **control program**

That keeps the dynamic layer reusable across product families.

Examples:

- an autocallable may use payoff-style redemption fragments plus an
  automatic stopping program
- a callable range-accrual note may use a static leg contract plus an
  issuer-control program
- a swing contract may use a base delivery or payoff fragment plus a
  holder-control program and inventory state

## Relationship To Existing Trellis Work

This plan is not inventing event/control semantics from scratch. The
repo already points in this direction:

- `trellis.agent.event_machine` provides a bounded event-machine surface
  for canonical autocallable and TARF patterns
- `trellis.agent.dsl_algebra` already distinguishes control-free
  algebra from explicit holder/issuer choice
- prior event-aware PDE and Monte Carlo plans established that events
  and control should lower into generic numerical timelines rather than
  remain product-local

What is missing is a closure-program-quality semantic foundation that
ties those ideas to route-free representation, decomposition, and
lowering.

## Closure Requirements

This track must satisfy the same three closures defined in
`doc/plan/draft__semantic-contract-closure-program.md`.

### Representation closure

Dynamic families are representationally closed only when the semantic
surface makes the following explicit:

1. **Base semantic substrate.**
   Which static contract or subcontracts the dynamic program acts on.
2. **Event ordering.**
   Observation, coupon accrual, payment, exercise, state-reset, and
   termination phases.
3. **Running state.**
   State variables, domains, initial values, and update rules.
4. **Stopping rules.**
   Automatic knock-out, target redemption, maturity settlement, or other
   contract-level termination logic.
5. **Controller role.**
   Automatic, holder, issuer, or other contractual decision-maker.
6. **Admissible actions.**
   Exercise, withdraw, continue, call, put, cancel, or no-op where
   applicable.

If any of those remain implicit in a route-local helper or a generated
adapter, representation closure has not been achieved.

### Decomposition closure

Dynamic families are decomposition-closed only when Trellis can:

- classify a request as static vs dynamic deterministically
- emit the same event/state/control structure for semantically
  equivalent requests
- separate automatic stopping from true holder/issuer control
- separate contract state from valuation policy or numerical
  approximation choices
- emit the dynamic semantic object without needing route ids or product
  strings as hidden discriminators

### Lowering closure

Dynamic families are lowering-closed only when the numerical lane can
consume:

- the base semantic contract
- the event ordering
- the state schema and updates
- the control program if present
- generic term groups
- valuation context
- market capabilities

without falling back to product-family blobs.

For dynamic families, lowerings should be explicit about which class of
semantics they support:

- automatic stateful contracts without control
- discrete holder/issuer control with bounded action sets
- continuous or singular control
- family-specific hybrid state dimensions such as FX/rate joint state

## Expected Lowering Map

The post-Phase-4 dynamic closure work should not treat all eventful
products as one generic numerical problem. A useful first map is:

- **automatic event/state, no control**
  event-aware PDE, tree, or Monte Carlo lanes can be admissible if they
  preserve event ordering and state updates explicitly
- **discrete holder/issuer control**
  backward-induction trees, control-aware PDEs, or LSMC-style Monte
  Carlo lanes can be admissible if the controller role and action set
  are explicit semantic inputs
- **continuous or singular control**
  these likely need a dedicated control solver family such as QVI,
  penalty, or another bounded control formulation; do not blur them into
  the same admission class as Bermudan-style stopping
- **hybrid factor-state families**
  PRDC-like FX/rate products or related hybrids may require additional
  admissibility constraints on factor dimensionality, quote conventions,
  and state interaction

The semantic track should therefore hand later lowering work a typed
problem class, not just a generic promise that "this product is
path-dependent."

## Candidate Surface

This document does not lock the final ADT, but a minimally honest shape
looks closer to:

```text
DynamicContractIR =
    { base_contract: BaseSemanticContract | None
    ; state_schema: StateSchema
    ; event_program: EventProgramIR
    ; control_program: ControlProgramIR | None
    ; settlement: SettlementRule
    ; observation: ObservationContext | None
    }

BaseSemanticContract =
    | PayoffContractIR
    | QuotedObservableContractIR
    | LegContractIR

StateSchema =
    { fields: tuple[StateFieldSpec, ...] }

StateFieldSpec =
    { name: str
    ; domain: str
    ; initial_value: object
    ; tags: tuple[str, ...]
    }

EventProgramIR =
    { events: tuple[ContractEvent, ...]
    ; ordering: tuple[str, ...]
    ; termination_rules: tuple[TerminationRule, ...]
    }

ContractEvent =
    | ObservationEvent(label, schedule, observed_terms)
    | CouponEvent(label, schedule, coupon_expr, state_updates)
    | PaymentEvent(label, schedule, cashflow_expr)
    | DecisionEvent(label, schedule, action_set, controller_role)
    | AutomaticTerminationEvent(label, trigger, settlement_expr)
    | StateResetEvent(label, schedule, state_updates)

ControlProgramIR =
    { controller_role: str
    ; decision_style: str
    ; decision_events: tuple[str, ...]
    ; admissible_actions: tuple[ActionExpr, ...]
    }
```

The important architectural point is not the exact field spelling. It
is that event ordering, state, and control are semantic structure above
the static contract basis.

## Mathematical Boundaries

### Automatic stopping is not the same thing as control

An autocall barrier check or target-redemption trigger is contractual
logic, but it is not yet holder or issuer optimization. That difference
matters because automatic stopping can often lower to event-aware PDE or
Monte Carlo without solving a dynamic-programming problem, while
holder/issuer control usually needs backward induction, LSMC, or a
control-aware PDE/tree formulation.

### Discrete control is not the same thing as continuous or singular control

Swing options and Bermudan callability usually fit a discrete decision
grid. GMWB-style withdrawal contracts introduce a harder control class:
the state includes account value and guarantee base, and the controller
chooses withdrawal magnitude, sometimes with penalty regions and in some
formulations a continuous-time or singular-control limit.

The closure plan should therefore not use one generic phrase such as
"optionality" to hide materially different mathematical problems.

### Dynamic state can sit above any static semantic base

Not every dynamic family is leg-based:

- autocallables may be best modeled as event logic over payoff-like
  redemption fragments
- TARF/TARN can mix schedule-driven coupon formulas with target state
- swing can sit over a delivery or payoff fragment
- GMWB involves account-value state that is not naturally a static leg
  schedule

Likewise, not every leg-based product is dynamic. Vanilla IRS and basis
swaps should remain in the static leg track unless they truly add
stateful stopping or control.

## Examples

### Example 1 — Worst-of autocallable

Semantic requirements:

- observation schedule
- worst-of multi-asset observation rule
- coupon/redemption formulas
- optional coupon-memory state for phoenix/snowball variants
- automatic termination when the autocall condition is met

This is not representationally closed if the event order and coupon
memory live only inside a Monte Carlo payoff callback.

### Example 2 — TARN / TARF

Semantic requirements:

- scheduled coupon or gain accrual formula
- running accumulated target state
- target cap or knock-out threshold
- settlement rule on target hit vs maturity exhaustion

This is not representationally closed if the running target lives only
as an unnamed local variable inside PDE or Monte Carlo code.

### Example 3 — Callable CMS-spread range-accrual note

Semantic requirements:

- static scheduled coupon structure
- quote-linked coupon condition or formula
- range-accrual or in-range counting state
- issuer decision schedule
- call settlement and accrual-interruption rules

This example demonstrates why the static leg and quoted-observable
tracks need a dynamic wrapper rather than trying to absorb everything
into either one alone.

### Example 4 — Swing option

Semantic requirements:

- decision schedule
- remaining-right inventory state
- optional minimum spacing or refraction state
- holder-maximizing control semantics

This is a control program, not just a path-dependent payoff.

### Example 5 — GMWB

Semantic requirements:

- account-value state
- guarantee-base or remaining-benefit state
- withdrawal action semantics and penalty policy
- maturity or surrender settlement

This family likely needs a dedicated later slice because the control
problem is materially stronger than the earlier automatic-stopping or
discrete-exercise families.

## Ordered Post-Phase-4 Queue

### ESC.1 — Dynamic semantic foundation

Objective:

Define the admitted event/state/control semantic surface and its
relationship to the static semantic bases.

Acceptance:

- one stable semantic note exists
- static vs dynamic boundary is explicit
- controller role and stopping semantics are documented

### ESC.2 — Automatic event/state families

Objective:

Land the first decomposition and representation slices for automatic
stateful products without explicit holder/issuer optimization.

Candidate proving families:

- autocallables / phoenix / snowball
- TARN / TARF

Acceptance:

- explicit state and event fixtures exist
- dynamic semantics do not rely on product-name nodes

### ESC.3 — Quote-linked scheduled contingent coupons

Objective:

Compose static leg and quoted-observable semantics under a dynamic
wrapper for structured coupon notes.

Candidate proving families:

- CMS-spread range accrual
- callable CMS-spread range accrual
- PRDC-style schedule-linked FX/rate coupons

Acceptance:

- no orphan family remains between quoted-observable and leg-based
  tracks
- quote-linked coupon formulas are semantic, not route-local

### ESC.4 — Discrete control families

Objective:

Add explicit holder/issuer control semantics with bounded action sets and
decision dates.

Candidate proving families:

- swing options
- callable coupon products beyond automatic stopping

Acceptance:

- controller role is explicit
- dynamic-programming or LSMC lowerings have a typed semantic input

### ESC.5 — Continuous or singular control families

Objective:

Handle the stronger control class exemplified by GMWB / GMxB.

Acceptance:

- control class is documented explicitly
- the plan does not blur discrete and singular control into one fake
  generic lane

## Risks To Avoid

- **Product-name relapse.** Nodes such as `AutocallableProgram`,
  `TarfProgram`, or `GmwbProgram` should not become disguised route ids.
- **Static-shape overfitting.** Forcing dynamic products into static leg
  or static payoff shapes will only move missing semantics into helper
  code.
- **Numerical leakage.** LSM basis choice, PDE stencil, or Monte Carlo
  regression policy are lowering concerns, not contract nodes.
- **Control confusion.** Automatic stopping and true holder/issuer
  optimization must remain distinct.
- **One-lane overclaim.** Automatic event-aware Monte Carlo support does
  not prove PDE, tree, or continuous-control support.

## Next Steps

1. Keep this document as the parking-lot foundation for the dynamic
   semantic track while current payoff-expression Phase 4 closes.
2. Update the static leg and quoted-observable docs so they hand off
   dynamic products explicitly rather than deferring them ambiguously.
3. Use the companion execution docs for the actual queueing once the
   post-Phase-4 strand becomes active:
   - `draft__post-phase-4-semantic-closure-execution-plan.md`
   - `draft__automatic-event-state-lowering-plan.md`
   - `draft__discrete-control-lowering-plan.md`
   - `draft__continuous-singular-control-lowering-plan.md`
   - `draft__insurance-contract-overlay-foundation.md`
4. File a future Linear child issue under QUA-887 for ESC.1 once the
   current Phase 4 slice is complete.
5. Split the eventual implementation into separate representation,
   decomposition, and lowering workstreams rather than one generic
   "dynamic IR" ticket.
