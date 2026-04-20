# Semantic Contract Closure Program

## Status

Draft. Cross-phase design document. Not yet an execution mirror, but
intended to define the closure vocabulary used by the Phase 2, Phase 3,
Phase 4, quoted-observable, and leg-based planning strands.

## Linked Linear

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-904 — Phase 2 umbrella (payoff-expression IR substrate)
- QUA-905 — Phase 3 umbrella (structural solver compiler)
- QUA-906 — Phase 4 umbrella (route retirement / dispatch phaseout)

## Companion Docs

- `doc/plan/draft__external-prior-art-adoption-map.md`
- `doc/plan/draft__contract-ir-phase-2-ast-foundation.md`
- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`
- `doc/plan/draft__contract-ir-phase-4-route-retirement.md`
- `doc/plan/draft__quoted-observable-contract-ir-foundation.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `doc/plan/draft__post-phase-4-semantic-closure-execution-plan.md`
- `doc/plan/draft__semantic-contract-target-and-trade-envelope.md`
- `doc/plan/draft__contract-ir-normalization-and-rewrite-discipline.md`

## Purpose

Define what it means for Trellis to support "arbitrary decomposable
derivatives" without falling back to route-local product buckets.

The core claim is narrower, and more useful, than "any imaginable
instrument string is supported." The real target is:

- if a contract can be decomposed into Trellis semantic representations
- and those representations have route-free lowering surfaces
- then fresh builds can price the contract without a direct hard-coded
  route keyed on instrument name

That target needs explicit closure gates. Without them, the program can
accidentally over-claim after any one of:

- an IR exists but no decomposer emits it
- a decomposer emits an IR but no checked lowering consumes it
- a lowering exists but the selector still depends on route-local
  product metadata

## Grand End Goal

The long-run architecture should satisfy:

1. semantic requests normalize into explicit contract representations
2. those representations are canonical and route-independent
3. structural declarations lower them into checked solver or assembly
   calls
4. fresh-build authority depends on semantic structure plus valuation
   context plus market capabilities, not on instrument strings or route
   ids

This is the sense in which Trellis can eventually price arbitrary
derivatives that are decomposable *within Trellis*. "Arbitrary" means
"arbitrary inside the supported semantic basis," not "arbitrary with no
semantic closure boundary."

## Prior-Art Guidance

The closure program should borrow selectively from strong external
designs rather than rediscovering every boundary from scratch. The
working adoption map lives in
`doc/plan/draft__external-prior-art-adoption-map.md`.

High-conviction guidance:

- **Strata:** preserve the separation between contract meaning,
  trade-envelope metadata, requested outputs, and market requirements
- **ACTUS:** future leg-based and event/state/control semantics should
  distinguish terms, events, and state instead of hiding them inside
  product labels
- **SymPy:** normalization needs explicit rewrite-strategy discipline,
  not a bag of local simplifiers
- **gs-quant:** valuation identity, market identity, overlays, and
  path-aware result navigation deserve explicit surfaces
- **Peyton Jones / Marlowe:** keep constructor budgets disciplined and
  prefer small explicit semantic cores over product-name sprawl, with
  explicit continuations or control where the contract really has
  stopping or decision semantics

## The Three Closures

For a semantic family `F`, define three distinct closure questions.

### 1. Representation closure

`Rep(F)` holds when Trellis has a route-free semantic representation for
the family that captures contractual meaning without opaque
product-specific payloads.

Concretely, representation closure requires:

- an admitted IR domain for the family
- clear well-formedness and canonicalization rules
- node names that describe contractual meaning or quoted market
  quantities, not pricing methods or route-local helper names
- if the family has endogenous running state, event ordering, automatic
  termination, or contractual holder/issuer choice, those surfaces are
  explicit semantic structure rather than helper-local behavior
- no hidden product discriminator such as `kind="swaption"` or
  `kind="sofr_ff_basis"` inside a supposedly generic node

Representation closure answers:

- "Can Trellis say what this contract *is* without naming a route?"

### 2. Decomposition closure

`Dec(F)` holds when supported request surfaces can be normalized into the
family's semantic representation deterministically and route-
independently.

Concretely, decomposition closure requires:

- fixture-driven normalization from request / semantic contract input to
  the relevant IR plus any generic non-structural term environment
- canonical output for semantically equivalent requests
- explicit boundary classification between sibling semantic domains
  where ambiguity is plausible, for example:
  - quoted-observable vs leg-based
  - payoff-expression vs event-coupled leg structures
  - static leg schedules vs leg-plus-event/control structures
  - automatic stopping programs vs holder/issuer control programs
- proof that route ids or instrument strings are not required to emit
  the representation

Decomposition closure answers:

- "Can Trellis reliably produce the right semantic object from the
  upstream request surface?"

### 3. Lowering closure

`Low(F, \mu)` holds for a requested method cohort `\mu` when Trellis can
compile the family's semantic representation into a checked solver call
or checked assembly without route-local product authority.

Concretely, lowering closure requires:

- structural declarations or equivalent lowering rules
- explicit admissibility conditions over representation, valuation
  context, and market capabilities
- normalization from semantic representation into helper or kernel
  inputs
- checked solver or assembly surfaces with family-appropriate parity or
  validation evidence
- for state-bearing or controlled families, checked lowering of event
  ordering, state updates, and controller semantics into the numerical
  lane rather than helper-local branching
- no product-shaped adapter blobs smuggled in as a substitute for
  missing representation work

Lowering closure answers:

- "Can Trellis price this semantic family from the representation alone,
  plus generic terms, valuation context, and market state?"

## Combined Closure And Route Retirement

For a family `F` and method cohort `\mu`, the semantic closure gate is:

$$
\operatorname{Closed}(F, \mu)
\;:=\;
\operatorname{Rep}(F)
\land
\operatorname{Dec}(F)
\land
\operatorname{Low}(F, \mu).
$$

This is necessary, but not sufficient, for route retirement.

Phase 4 route-free fresh-build authority should require:

$$
\operatorname{Migratable}(F, \mu)
\;:=\;
\operatorname{Closed}(F, \mu)
\land
\operatorname{Parity}(F, \mu)
\land
\operatorname{Prov}(F, \mu),
$$

where:

- `Parity(F, \mu)` means the structural path has passed the agreed
  valuation parity policy against the incumbent checked path
- `Prov(F, \mu)` means provenance / replay / observability remain
  adequate after route-local authority is removed

## Decomposable Arbitrary Products

The program goal is not limited to single-family products.

A composite product is in scope when it can be decomposed into a finite
combination of semantically closed subcontracts under supported
composition operators.

Examples:

- payoff-expression algebra already provides additive and multiplicative
  composition via `Add`, `Sub`, `Mul`, `Scaled`, `Max`, `Min`, and
  indicator structures
- future leg-based IR will provide composition through signed legs,
  coupon schedules, and settlement rules
- future quoted-observable products stay inside payoff-expression
  algebra as functions of explicit quote points
- future event/state/control semantics should provide composition
  through ordered events, explicit state updates, and holder/issuer or
  automatic stopping programs over a base semantic contract

So "arbitrary decomposable derivative" means:

- the contract can be normalized into the supported semantic basis
- each required semantic component is closed on representation,
  decomposition, and lowering
- the composition operator itself is supported by the relevant IR /
  lowering surface

If any of those fail, Trellis should block honestly rather than rescue
the request with a hidden product route.

## Stateful And Controlled Products

Many important post-Phase-4 families are not static expressions or
static leg schedules. They require endogenous state, event ordering, or
explicit control.

Representative examples include:

- autocallables, phoenix notes, and snowballs with discrete observation
  dates, coupon memory, barriers, and early redemption
- TARN / TARF structures with running accumulated coupon or gain state
  and target-triggered termination
- PRDC-style and callable structured rate products with quote-linked
  scheduled coupons and issuer call overlays
- swing options with remaining-right inventory, refraction rules, and
  multiple exercise decisions
- GMWB / GMxB contracts with account-value state, guarantee-base state,
  and withdrawal control

For these families, "decomposable" must be interpreted more carefully.
The right semantic target is often:

- a base semantic contract from one of the static sibling domains
  (payoff-expression, quoted-observable, or leg-based)
- plus an explicit event/state/control program that says how
  observations, payments, state updates, stopping rules, and controller
  actions evolve through time

This is a first-class semantic representation problem, not only a later
lowering detail. A product that needs running target state or multiple
exercise rights is not representationally closed if those semantics live
only inside Monte Carlo code or route-local helper assembly.

## Program Responsibilities By Phase

### Phase 2 — Payoff-expression representation and bounded decomposition

Phase 2 primarily owns:

- representation closure for the bounded payoff-expression cohort
- bounded decomposition closure for the same cohort

Phase 2 does **not** own lowering closure. It prepares the contract
surface that Phase 3 will consume.

### Quoted-observable follow-on — Representation extension for quote-map products

This track extends representation closure for products whose semantics
depend on explicit curve / surface quote points at an observation
surface.

It also owns the decomposition boundary between:

- quoted-observable products
- leg-based cashflow products with similar desk labels

### Leg-based follow-on — Representation extension for coupon / cashflow products

This track extends representation closure for contracts defined by legs,
cashflows, accrual rules, fixing rules, and settlement conventions.

It owns the decomposition boundary between:

- schedule-of-cashflows products
- one-shot quoted-observable spread products

This track should own the **static** leg semantics that later stateful
tracks can embed. It should not be forced to absorb every callable,
interruptible, or target-accumulating product into one oversized leg
schema.

### Event/state/control follow-on — Representation extension for stateful and controlled contracts

This track extends representation closure for contracts whose meaning
depends on ordered events, running state, automatic stopping rules, or
explicit holder/issuer control.

It owns the decomposition boundary between:

- static payoff-expression or static leg products
- those same semantic bases wrapped in event/state/control programs
- automatic termination logic and true contractual control
- contractual control semantics and valuation-policy choices

This track should be able to reuse payoff-expression, quoted-observable,
and leg-based subcontracts as building blocks. Not every stateful exotic
is naturally leg-based, and not every event-coupled product should be
forced into a coupon-leg root.

### Phase 3 — Lowering closure for migrated families

Phase 3 owns lowering closure for the currently represented payoff-
expression cohort.

It must consume upstream representations honestly. If a family still
requires a product-shaped blob to reach a helper, that is a failure of
representation closure or decomposition closure, not a reason to relax
the lowering contract.

### Phase 4 — Authority closure after semantic closure exists

Phase 4 owns the selector-authority change once a family is already
closed and parity / provenance evidence exists.

Phase 4 should never be used to "discover" missing representation or
lowering pieces. If those are missing, the family is not yet migratable.

## Required Artifact Checklist Per Family

No family should be called "migrated" until it has all of the following:

1. A semantic home.
   Example: payoff-expression `ContractIR`, quoted-observable
   extension, static leg-based IR, or event/state/control foundation.
2. A well-formedness and canonicalization contract.
3. A boundary-classification matrix against adjacent semantic homes.
4. Fixture-driven decomposition evidence.
5. Route-independence tests for decomposition.
6. Checked lowering declarations or checked assembly surface.
7. Admissibility / ambiguity policy for the lowering.
8. Parity evidence on the agreed benchmark or fixture cohort.
9. Provenance and route-masked selector evidence for fresh builds.

If the family carries state or control, the semantic home must also make
the following explicit:

- state fields and their initial conditions
- event ordering and observation/payment phases
- stopping or termination rules
- controller role and admissible decision set when control exists
- the base semantic contract or subcontracts on which those dynamic
  rules operate

## Bounded-Claim Discipline

Closure is always family-scoped and method-scoped.

Do not convert:

- `Closed(vanilla_terminal_ramp, analytical)`

into:

- "arbitrary equity exotics are route-free now"

Likewise, a proof on one representation track must not be stretched
across the others:

- payoff-expression closure does not imply quoted-observable closure
- quoted-observable closure does not imply leg-based closure
- static leg-based closure does not imply event/state/control closure
- quoted quote-node closure does not imply quote-linked scheduled coupon
  closure
- automatic stopping-program closure does not imply holder/issuer
  control closure
- discrete control closure does not imply continuous or singular control
  closure
- deterministic analytical lowering closure does not imply Monte Carlo
  lowering closure

## Immediate Planning Consequences

1. Every phase document should state which closure(s) it owns.
2. The quoted-observable track needs its own planning artifact rather
   than living only as a boundary note.
3. The static leg-based and quoted-observable tracks should state
   explicitly how they compose under the later event/state/control
   foundation rather than orphaning quote-linked coupon products
   between them.
4. The event/state/control track needs its own planning artifact rather
   than remaining only an optional note inside leg-based planning.
5. Closure work after Phase 4 should include a fixture-backed semantic
   classifier matrix for at least:
   - terminal quote spreads vs basis swaps
   - static coupon notes vs callable / interruptible coupon notes
   - autocallables / phoenix / snowball structures
   - TARN / TARF target-accumulation structures
   - PRDC and related FX/rate coupon hybrids
   - swing and other multiple-exercise contracts
   - GMWB / GMxB and other control-heavy insurance-style contracts
6. Families entering Phase 3 shadow mode should already have documented
   representation and decomposition closure evidence.
7. Families entering Phase 4 route retirement should already have a
   closure record plus parity and provenance evidence.

## Post-Phase-4 Closure Queue

The most useful next queue after the current payoff-expression Phase 4
slice is:

1. quoted-observable snapshot closure
2. static leg-based closure
3. event/state/control foundation
4. quote-linked scheduled coupon closure on top of static leg and
   quoted-observable leaves
5. discrete stopping/control closure for autocallable, target-
   accumulation, callable coupon, and swing-style products
6. continuous/singular control closure for GMWB / GMxB-style families

This ordering keeps the semantic basis honest:

- static quote and static leg semantics land first
- dynamic wrappers then compose over them explicitly
- Phase 4 for those later families can inherit one authority model
  instead of re-inventing route retirement family by family
