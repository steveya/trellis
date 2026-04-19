# External Prior-Art Adoption Map For The Semantic Contract Program

## Status

Draft. Cross-cutting design note. Not yet an execution mirror.

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-904 — Phase 2 umbrella for payoff-expression Contract IR
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/draft__contract-ir-phase-2-ast-foundation.md`
- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`
- `doc/plan/draft__contract-ir-phase-4-route-retirement.md`
- `doc/plan/draft__quoted-observable-contract-ir-foundation.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__market-coordinate-overlay-and-shock-model.md`
- `doc/plan/draft__valuation-result-identity-and-provenance.md`
- `doc/plan/draft__valuation-session-and-request-surface.md`
- `doc/plan/draft__portfolio-path-and-result-set-surface.md`

## Purpose

Record which external libraries or prior-art sources Trellis should
deliberately learn from during the semantic-contract program, and which
parts of those systems Trellis should explicitly avoid copying.

This note exists to prevent two failure modes:

- vague name-dropping with no design consequence
- accidental cargo-culting of product taxonomies or provider
  architectures that would cut against Trellis' route-retirement goal

## Design Rule

An external system is relevant only if it helps Trellis strengthen one
or more of the following:

- semantic representation
- decomposition discipline
- lowering / requirements declaration
- market identity and provenance
- result navigation and output identity

If it does not improve one of those, it is background reading, not an
adoption candidate.

## High-Conviction Sources

### 1. OpenGamma Strata

Primary lessons:

- strict separation of contract economics (`Product`) from trade
  wrapper (`Trade`)
- explicit requested-measure vocabulary (`Measure`)
- explicit market requirement declaration (`FunctionRequirements`)
- calculation dispatch by target type plus measure support, not by
  sprawling handwritten route ladders

Adopt in Trellis:

- semantic contract meaning should stay separate from trade envelope and
  portfolio position metadata
- structural declarations should expose supported outputs and market
  requirements explicitly
- valuation output identity should be a first-class contract, not an ad
  hoc string pile

Do **not** adopt:

- Strata's product-class taxonomy as Trellis' semantic authority
- a Java-style class-per-product hierarchy as the future of Trellis IR
- type-only dispatch as a substitute for semantic structural matching

Where this should land:

- `draft__semantic-contract-target-and-trade-envelope.md`
- `draft__contract-ir-phase-3-solver-compiler.md`
- `draft__valuation-session-and-request-surface.md`

### 2. ACTUS

Primary lessons:

- contract semantics should distinguish:
  - contract terms
  - generated events
  - evolving contract state
- many apparently different products share the same deeper event/state
  skeleton
- schedules and business events deserve explicit treatment, not product
  prose

Adopt in Trellis:

- future leg-based IR should make terms, scheduled events, and state
  transition surfaces explicit
- party role / pay-receive / contract direction must be semantic data
  rather than a product label
- event ordering and state evolution should be design-time objects, not
  helper-local conventions

Do **not** adopt:

- ACTUS contract-type names as Trellis' mandatory semantic taxonomy
- blind one-to-one mirroring of the ACTUS dictionary into Trellis node
  names
- forcing the payoff-expression Phase 2 AST into an event/state machine
  before needed

Where this should land:

- `draft__leg-based-contract-ir-foundation.md`

### 3. SymPy

Primary lessons:

- treat rewriting as a programmable strategy layer, not a bag of random
  simplifications
- distinguish:
  - local rewrite rules
  - traversal order (`top_down`, `bottom_up`)
  - first-success selection (`do_one`)
  - fixed-point application (`exhaust`)
- write tests for idempotence and fixed-point behavior, not just spot
  examples

Adopt in Trellis:

- Contract IR normalization needs an explicit rewrite strategy contract
- canonicalization should be deterministic, idempotent, and
  property-tested
- rule order should be deliberate and reviewable

Do **not** adopt:

- symbolic-general-purpose simplification ambition beyond Trellis'
  bounded contract algebra
- algebraically clever rewrites that outrun semantic proof or test
  coverage

Where this should land:

- `draft__contract-ir-phase-2-ast-foundation.md`
- `draft__contract-ir-normalization-and-rewrite-discipline.md`

### 4. gs-quant

Primary lessons:

- explicit market-coordinate identity matters
- overlay and relative-market semantics matter
- valuation results need a compact identity key plus richer provenance
- path-aware result navigation is useful once outputs become nested

Adopt in Trellis:

- market coordinate / overlay / shock vocabulary
- valuation identity and provenance packet
- later session/request surface hardening
- path-aware result-set design

Do **not** adopt:

- provider-driven remote pricing as the semantic authority
- generated target schemas as the center of the architecture
- service-client object graphs as a replacement for Trellis' local
  semantic compiler

Where this should land:

- `draft__market-coordinate-overlay-and-shock-model.md`
- `draft__valuation-result-identity-and-provenance.md`
- `draft__valuation-session-and-request-surface.md`
- `draft__portfolio-path-and-result-set-surface.md`

### 5. Peyton Jones / Combinator Contracts

Primary lessons:

- a small core algebra can express a large class of payoff structures
- constructor budget matters
- composition should beat product-specific node proliferation

Adopt in Trellis:

- keep the payoff-expression AST small and compositional
- justify every new constructor by semantic necessity, not desk label
- prefer algebraic combination plus semantic observables over
  instrument-named nodes

Do **not** adopt:

- assuming the small combinator core alone solves schedules,
  conventions, market coordinates, or leg-state evolution

Where this should land:

- `draft__contract-ir-phase-2-ast-foundation.md`
- `draft__semantic-contract-closure-program.md`

### 6. Marlowe

Primary lessons:

- a small contract language with explicit step semantics can still be
  operationally useful
- continuation/event semantics should be explicit when a domain needs
  them
- analysis gets easier when the execution model is finite and explicit

Adopt in Trellis:

- event-coupled contract tracks should prefer explicit operational
  surfaces over hidden helper logic
- when Trellis introduces event/state semantics, those phases and
  continuations should be first-class and reviewable

Do **not** adopt:

- blockchain-specific execution constraints
- forcing the payoff-expression IR into a continuation DSL before the
  leg/event track needs it

Where this should land:

- `draft__leg-based-contract-ir-foundation.md`
- `draft__semantic-contract-closure-program.md`

## Adoption Priority

In order of immediate usefulness to the current Trellis program:

1. Strata
2. SymPy
3. gs-quant
4. ACTUS
5. Peyton Jones
6. Marlowe

This is not a judgment of overall quality. It is a judgment of what
most directly hardens the current closure and route-retirement program.

## Cross-Cutting Rejections

Across all prior art, Trellis should reject:

- product-family payload blobs that smuggle back route logic
- giant product taxonomies as a substitute for semantic decomposition
- remote-provider or service-client architecture as semantic authority
- "universal IR" ambition before the closure boundaries are proven

## Follow-On Use

When a future plan doc says it is "inspired by" one of these systems, it
should also answer:

1. which exact idea is being adopted
2. which exact idea is being rejected
3. which closure or provenance problem this solves for Trellis

If a doc cannot answer those three questions, the reference is probably
decorative rather than architectural.
