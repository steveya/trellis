# Validation Contract Loop Design

This document is the current design reference for Trellis' validation process
after the contract-algebra, DSL, admissibility, and analytics-support work.
It supersedes the older deterministic-first review sketches as the primary
architecture note for validation.

Read this together with:

- `docs/quant/contract_algebra.rst`
- `docs/quant/dsl_algebra.rst`
- `docs/agent/critic_agent.rst`
- `docs/developer/audit_and_observability.rst`

## Why This Needs A Real Design

The old validation story was directionally right but structurally incomplete.
It said "deterministic first" while leaving too much work to reviewer stages
that still had to infer semantics, route risk, comparison meaning, and even
what deterministic checks should exist.

That created a recurring failure pattern:

- time was burned in critic JSON retries and text fallbacks instead of pricing
- codegen errors reached reviewer stages because semantic and lowering context
  was not propagated deeply enough into validation
- arbiter had to support brittle reviewer-authored `test_code`
- comparison and cross-validation logic silently assumed equality or `<=`
  relations where the route semantics were actually lower-bound, upper-bound,
  or otherwise one-sided
- prompts remained too large and too open-ended to serve as a reliable builder
  or reviewer loop

The contract-algebra and DSL work changes the available design space. Trellis
now has enough typed structure to make validation itself a compiled artifact
instead of an after-the-fact reviewer guess.

## Current Executable Boundary

The shipped compilation boundary is now:

```text
SemanticContract
  -> semantic validation
  -> ValuationContext
  -> RequiredDataSpec / MarketBindingSpec
  -> ProductIR
  -> typed route admissibility
  -> family lowering IR
  -> helper-backed target bindings
  -> lowering errors / warnings / trace metadata
```

The important point is that validation no longer has to infer this structure
from code alone. The current request metadata already persists:

- semantic contract summary
- semantic blueprint summary
- family IR type and payload
- target bindings
- lowering errors
- requested outputs
- valuation context and required data summaries

The current shipped loop now uses those artifacts in a bounded order:

- quant/compiler selects the method and route under the typed semantic surface
- the compiler emits a validation contract keyed to that exact route instance
- deterministic checks and relation-aware comparisons run from the compiled
  contract, not from reviewer-authored test snippets
- any model-validator work is bounded to the residual-risk packet that remains
  after deterministic validation, admissibility, and lowering succeed

What is still missing is broader coverage across more route families, not the
basic shape of the quant -> validation contract -> residual model-validator
loop itself.

## Design Goals

- make validation a compiled contract emitted from semantic and lowering state
- reject semantic, admissibility, and lowering failures before reviewer loops
- keep quant and critic prompts small, structured, and role-specific
- remove executable reviewer output from the standard path
- make arbiter purely deterministic and relation-aware
- separate deterministic validation, critic review, and model validation into
  non-overlapping roles
- make every retry explainable by a stable failure packet
- keep standard mode cheap and thorough mode richer without changing the core
  contract

## Non-Goals

- do not build a universal formal proof system for pricing correctness
- do not replace existing invariant libraries with one giant abstract engine
- do not make critic or model validator authoritative over typed semantics
- do not let review prompts become the primary place where route capability is
  defined

## Core Design: Validation As A Compiled Contract

The central design move is to compile a validation contract from the same
semantic and lowering artifacts that now drive route selection.

Let:

- `C` be the semantic contract
- `V` be the valuation context
- `P` be the checked `ProductIR`
- `A` be the typed route admissibility decision
- `L` be the family lowering summary, target bindings, and lowering errors

Then validation should consume a compiled object:

```text
X = Γ(C, V, P, A, L)
```

where `X` is the validation contract.

The validation contract is not "extra metadata." It is the executable
specification of what validation means for one compiled route instance.

### Proposed Validation Contract Fields

The validation contract should minimally contain:

- `deterministic_checks`
  The exact invariant and comparison checks that apply.
- `comparison_relations`
  The intended comparator relation for each reference or cross-validation edge.
- `harness_requirements`
  The factories, fixtures, market perturbations, and helper inputs needed to
  run each deterministic check.
- `residual_risks`
  The small set of semantic or numerical concerns not already discharged by
  deterministic checks.
- `review_policy`
  Standard-vs-thorough budgets and whether critic or model validation is
  `skip`, `advisory`, or `required`.
- `trace_contract`
  The exact fields that must be persisted for replay and debugging.

One useful formal reading is:

```text
X = (I, R, H, E, P, T)
```

where:

- `I` is the deterministic invariant family
- `R` is the relation set for reference and comparison checks
- `H` is the executable harness specification
- `E` is the residual-risk set
- `P` is the review-budget and escalation policy
- `T` is the required trace payload

## Mathematical View

Validation should be understood as layered property checking, not as one
monolithic pass/fail test.

### Layer 1: Semantic Validity

The first question is whether the request has a coherent typed meaning.

This layer checks properties such as:

- phase ordering consistency
- controller protocol consistency
- state-field admissibility
- output admissibility
- valuation-context completeness

Failures here are semantic errors, not pricing failures. The builder and critic
should never see them as if they were code bugs.

### Layer 2: Admissibility And Lowering Validity

Given a valid semantic request, the next question is whether a route family can
 legally support it.

This layer checks:

- route admissibility against control, state, output, and reporting tags
- family lowering success
- target-binding completeness
- lowering warnings versus hard errors

Again, these are not reviewer concerns. They are typed compiler outcomes.

### Layer 3: Deterministic Numerical Properties

Once a route exists, Trellis should check explicit numerical properties
selected by the validation contract.

Let `u` be the realized pricing implementation for one route instance.
Each deterministic check is a property:

```text
φ_i(u; h_i) ∈ {pass, fail, skip}
```

where `h_i` is the harness material required for check `i`.

Examples:

- non-negativity
- volatility sensitivity or monotonicity
- rate sensitivity or monotonicity
- reference bounds
- family-specific invariants such as CDS quote normalization

The key point is that the contract defines the property family first. Critic
does not invent `φ_i`.

### Layer 4: Relation-Aware Comparison Semantics

Cross-validation is not always equality.

For a route output `u` and a reference output `r`, the intended
relation can be:

- equality within tolerance: `|u - r| <= eps`
- lower-bound relation: `u <= r + eps`
- upper-bound relation: `u >= r - eps`
- interval membership
- monotone directional agreement across perturbations

This should be compiled from comparison semantics rather than hard-coded by the
arbiter or validation bundle.

In the current implementation, the deterministic layer now normalizes these
relations into stable runtime ids:

- ``within_tolerance`` for equality-style comparisons
- ``<=`` for upper-bound checks such as callable-bond versus straight-bond
- ``>=`` for lower-bound checks such as puttable-bond versus straight-bond

Those relation ids are carried in the compiled validation contract and replayed
through validation bundles and task cross-validation results.

### Layer 5: Residual Review

Only after the first four layers pass should LLM reviewers be asked anything.

The residual reviewer problem is:

```text
review(u, σ(X), D)
```

where:

- `u` is the realized implementation
- `σ(X)` is a compact summary of the validation contract
- `D` is the deterministic evidence packet

The reviewer should answer only unresolved questions. It should not rediscover
semantic structure or invent new validation objectives.

## Computational Design

### Compile-Time Artifacts

The validation pipeline should emit four durable artifacts before any critic
call:

1. `CompiledRequest`
   Semantic contract, valuation context, product IR, route plan.
2. `LoweringSummary`
   Family IR, target bindings, lowering warnings, lowering errors.
3. `ValidationContract`
   Deterministic checks, comparison relations, residual risks, budgets.
4. `ValidationEvidencePacket`
   Deterministic outcomes, skipped reasons, stage latencies, and replay fields.

`ValidationContract` should be treated as a sibling of route selection, not as
an afterthought inside the critic.

### Prompt Surfaces

### Quant / Builder Prompt

The quant or builder stage should receive:

- route card
- family IR summary
- target bindings summary
- admissibility summary
- validation contract summary
- prior failure packet if this is a retry

It should not receive:

- long prose about every validation policy in the system
- free-form reviewer speculation
- large irrelevant docs dumps

The retry packet should be machine-oriented:

- stable failure code
- failing check ids
- compact evidence values
- requested remediation target

### Critic Prompt

The critic prompt should become smaller than the builder prompt.

It should receive only:

- generated code
- route family and instrument summary
- compact validation contract summary
- deterministic evidence packet
- residual-risk ids
- allowed finding ids

It should not receive:

- unresolved semantic compilation work
- large knowledge dumps
- the authority to invent checks, formulas, or executable code

### Model Validator Prompt

`model_validator` should become a residual conceptual-review stage, mostly for
thorough mode. Its job is to reason about approximation quality, calibration
assumptions, and material blind spots that deterministic validation does not
cover.

It should not duplicate critic or arbiter.

### Critic Contract

Critic output should be a bounded structured finding schema, for example:

```text
CriticFinding(
  finding_id,
  check_family,
  severity,
  status,
  evidence_refs,
  rationale,
  remediation,
)
```

Important constraints:

- no `test_code`
- no new check identifiers
- no new formulas
- no broad narrative review
- no more than a small bounded number of findings

The critic's job is to point to residual deterministic concerns or unresolved
conceptual mismatches, not to author executable work.

### Arbiter Contract

Arbiter should execute only:

- validation-contract deterministic checks
- relation-aware comparison checks
- any supported critic finding families that map onto existing deterministic
  contracts

Arbiter should not:

- execute reviewer-authored Python
- infer relation semantics from route names
- silently reinterpret critic findings

The arbiter becomes a deterministic dispatcher over compiled check specs.

### Standard vs Thorough Policy

The same validation contract should support both modes, with different budgets.

### Standard

Standard mode should aim for a cheap loop:

- semantic validation, admissibility, and lowering are mandatory
- deterministic validation is mandatory
- critic is `skip` or `advisory` depending on residual risk
- no text fallback
- no model-validator LLM by default
- failures are returned as compact machine-readable packets

### Thorough

Thorough mode may add:

- larger deterministic bundles
- broader comparison matrices
- required critic on selected residual risks
- model-validator conceptual review
- richer trace payloads

The important rule is that thorough mode should extend the contract, not change
its ontology.

## Practical Design

### Practical Failure Taxonomy

Trellis should classify validation outcomes into stable buckets:

1. `semantic_invalid`
   Typed contract is inconsistent or incomplete.
2. `admissibility_failed`
   Requested route family cannot support the typed request.
3. `lowering_failed`
   Family IR or target binding could not be completed.
4. `codegen_invalid`
   Generated code failed sanitation, imports, or basic execution readiness.
5. `deterministic_validation_failed`
   Invariant or relation-aware check failed.
6. `critic_schema_failed`
   Critic could not return a valid bounded finding packet.
7. `critic_deterministic_confirmation_failed`
   Critic raised a supported concern and arbiter confirmed it.
8. `model_validation_advisory`
   Deep conceptual review found non-blocking limitations or concerns.

These buckets matter because retries differ by class:

- builder retry can fix `codegen_invalid` and `deterministic_validation_failed`
- compiler work is needed for `semantic_invalid`, `admissibility_failed`, or
  `lowering_failed`
- critic schema failures should not masquerade as pricing failures

### Observability And Replay

Every build attempt should persist:

- compiled validation contract summary
- deterministic check list and per-check status
- comparison relations and tolerances
- residual-risk ids
- critic mode, retry counts, and stage latency
- model-validator mode and latency
- builder retry reason

That makes replay possible without reconstructing the request from chat text or
guessing why a reviewer was invoked.

### Recommended Loop

The target loop is:

```text
request
  -> compile semantics and lowering
  -> emit validation contract
  -> build code against route card + validation contract
  -> sanitize and import-check
  -> run deterministic validation
  -> if fail: return structured failure packet to builder
  -> if pass and residual risk exists: run bounded critic
  -> arbiter confirms only supported deterministic findings
  -> if thorough and conceptual risk remains: run model validator
  -> return final evidence packet
```

This loop is deliberately narrow. The quant stage writes code. The validation
contract defines what must be true. The critic only comments on what remains
unresolved. The arbiter only executes supported checks.

## Current Gap To Close

The main architectural gaps described in this note are now implemented. The
remaining work is maintenance rather than redesign:

- keep the docs aligned as new validation-contract fields or review reasons are added
- keep regression coverage broad enough to catch prompt-surface or trace drift
- keep the compatibility-only paths from leaking back into the standard path

## Upgrade Plan

This should be executed as one coherent redesign under `QUA-461`.

### Phase 0: Epic Reframe

Refresh the umbrella issue so the objective is:

- compiler-emitted validation contract
- deterministic execution first
- bounded residual review loop
- explicit standard versus thorough policy

Artifacts:

- this design note
- issue-body refresh for the remaining open `QUA-461` children

### Phase 1: Validation Contract Compiler

Primary issue: `QUA-466`

Deliver:

- `ValidationContract` dataclass and summary helpers
- compiler step from semantic blueprint, family IR, target bindings,
  lowering errors, requested outputs, and route admissibility
- trace persistence

Key files:

- `trellis/agent/platform_requests.py`
- `trellis/agent/executor.py`
- `trellis/agent/validation_bundles.py`
- any new validation-contract module

### Phase 2: Relation-Aware Comparison Semantics

Primary issue: `QUA-505`

Deliver:

- explicit comparison relation objects or normalized specs
- validation-bundle and arbiter support for `=`, `<=`, `>=`, and interval-style
  relations
- route or family-level comparator plans compiled into the validation contract

Landed now:

- compiled validation contracts persist normalized comparison relations in trace
  metadata
- deterministic validation bundles consume the compiled relation instead of
  hard-coding callable-style `<=`
- task cross-validation supports equality-style and directional (`<=`, `>=`)
  semantics with tolerance slack
- puttable-bond bound validation now compiles as a lower-bound check rather
  than incorrectly reusing the callable-bond upper-bound direction

Remaining follow-on:

- arbiter still needs to consume the same compiled relation surface instead of
  its transitional hard-coded reference semantics

### Phase 3: Review Policy Over Compiled Validation State

Primary issue: `QUA-462`

Deliver:

- review policy keyed off validation-contract residual risks, lowering errors,
  admissibility status, and comparison-plan complexity
- standard mode bounded critic settings
- thorough mode full critic + model-validator settings

Landed now:

- review policy consumes compiled validation-contract state before falling back
  to coarse route heuristics
- low-risk analytical routes still skip reviewer LLM stages when the contract
  has no residual risks
- residual-risk and directional-relation signals now escalate the critic for
  the right reason instead of because the route family merely looked risky
- lowering or admissibility failures are now treated as contract-blocking
  review reasons rather than as invitations for free-form reviewer guesses

This removes another large source of noisy reviewer work while keeping explicit
scrutiny on semantically risky builds.

### Phase 3: Review Policy Over Compiled Validation State

Primary issue: `QUA-462`

Deliver:

- review policy keyed off validation-contract residual risks, lowering errors,
  admissibility status, and comparison-plan complexity
- stable `skip` / `advisory` / `required` semantics
- explicit retry and latency budgets

### Phase 4: Critic And Arbiter Refactor

Primary issues: `QUA-464`, `QUA-465`

Deliver:

- compact critic finding schema
- critic prompt generation from validation contract and evidence packet
- deterministic arbiter dispatch over validation-contract check families
- standard-path removal of legacy `test_code` execution

This is the main loop-hardening phase.

Landed now:

- critic check menus can be restricted directly from compiled deterministic
  validation checks instead of only from coarse instrument buckets
- critic output is filtered to allowed contract-backed check ids by default
- arbiter dispatch now executes only explicitly allowed check ids in the
  standard path
- reviewer-authored `test_code` is no longer executable in the standard path;
  only an explicit compatibility mode can still replay it offline

### Phase 5: Model Validator Role Narrowing

Primary issue: `QUA-467`

Deliver:

- model-validator prompt driven by residual conceptual risk
- standard mode stays cheap
- thorough mode owns deeper MRM-style review without duplicating critic

Landed now:

- model-validator prompt can be narrowed around compiled validation-contract
  residual risks instead of rehashing generic code-review concerns
- prompt text now explicitly tells the model validator not to repeat
  deterministic checks already covered by the validation contract
- executor threads validation-contract residual risks into the thorough-mode
  model-validator path

### Phase 6: Cleanup, Docs, And Latency Budgets

Primary issue: `QUA-468`

Deliver:

- cleanup of transitional branches
- docs refresh across quant, developer, and agent docs
- canary reruns and latency-budget capture

Landed now:

- docs refreshed across the validation design note, developer observability
  docs, user pricing guide, and critic/arbiter docs
- stale standard-path references to reviewer-authored execution were removed
  from the core validation docs
- the updated validation stack passes the full non-integration regression sweep
  on April 2, 2026:
  `2316 passed, 18 skipped, 5 deselected`

## Ticket Mapping

The remaining `QUA-461` child set should read as:

- `QUA-466`
  compiler-emitted validation contract
- `QUA-505`
  relation-aware comparison semantics
- `QUA-462`
  review policy over compiled validation state
- `QUA-464`
  compact critic finding contract and prompt surface
- `QUA-465`
  deterministic arbiter dispatcher over compiled checks
- `QUA-467`
  residual conceptual review role for model validator
- `QUA-468`
  cleanup, latency hardening, and docs closeout

`QUA-463` stays valid as a completed family-specific deterministic bundle slice.

## Design Rules

- validation must be compiled from typed semantic and lowering artifacts
- deterministic checks own executable confirmation
- reviewer stages own only residual uncertainty
- prompts should be smaller after the compiler, not larger
- standard mode must fail quickly and explainably
- thorough mode may be richer, but it must remain structured and bounded

## What This Document Replaces

This note replaces the architecture role that was previously spread across the
older deterministic-review planning notes and ad hoc ticket comments.

Those sketches were useful during the initial tightening work, but they are no
longer the right source of truth now that contract algebra, admissibility, DSL
lowering, and analytics support are real compiled artifacts in the system.
