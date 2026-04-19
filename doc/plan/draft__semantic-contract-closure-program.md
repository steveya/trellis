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
- **ACTUS:** future leg-based semantics should distinguish terms,
  events, and state instead of hiding them inside product labels
- **SymPy:** normalization needs explicit rewrite-strategy discipline,
  not a bag of local simplifiers
- **gs-quant:** valuation identity, market identity, overlays, and
  path-aware result navigation deserve explicit surfaces
- **Peyton Jones / Marlowe:** keep constructor budgets disciplined and
  prefer small explicit semantic cores over product-name sprawl

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

So "arbitrary decomposable derivative" means:

- the contract can be normalized into the supported semantic basis
- each required semantic component is closed on representation,
  decomposition, and lowering
- the composition operator itself is supported by the relevant IR /
  lowering surface

If any of those fail, Trellis should block honestly rather than rescue
the request with a hidden product route.

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
   extension, or leg-based IR.
2. A well-formedness and canonicalization contract.
3. Fixture-driven decomposition evidence.
4. Route-independence tests for decomposition.
5. Checked lowering declarations or checked assembly surface.
6. Admissibility / ambiguity policy for the lowering.
7. Parity evidence on the agreed benchmark or fixture cohort.
8. Provenance and route-masked selector evidence for fresh builds.

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
- deterministic analytical lowering closure does not imply Monte Carlo
  lowering closure

## Immediate Planning Consequences

1. Every phase document should state which closure(s) it owns.
2. The quoted-observable track needs its own planning artifact rather
   than living only as a boundary note.
3. Families entering Phase 3 shadow mode should already have documented
   representation and decomposition closure evidence.
4. Families entering Phase 4 route retirement should already have a
   closure record plus parity and provenance evidence.
