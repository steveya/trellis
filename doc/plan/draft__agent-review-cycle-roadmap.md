# Agent Review Cycle Roadmap

## Status

Draft execution mirror. Linear epic and phased child queue filed for
implementation sequencing.

## Linked Context

- `ARCHITECTURE.md`
- `docs/user_guide/pricing.rst`
- `trellis/agent/quant.py`
- `trellis/agent/planner.py`
- `trellis/agent/builder.py`
- `trellis/agent/executor.py`
- `trellis/agent/review_policy.py`
- `trellis/agent/validation_contract.py`
- `trellis/agent/critic.py`
- `trellis/agent/arbiter.py`
- `trellis/agent/model_validator.py`
- `trellis/agent/platform_traces.py`
- `trellis/agent/linear_tracker.py`
- `tests/test_agent/test_lite_review.py`
- `tests/test_agent/test_critic.py`
- `tests/test_agent/test_model_validator.py`
- `tests/test_agent/test_platform_traces.py`

## Linked Linear

- `QUA-982` - Agent cycle: governed quant, critic, model-validator, and arbiter loop
- `QUA-461` - Validation pipeline: deterministic-first review redesign
- `QUA-669` - Validation review: executor, critic, arbiter, lite review, model validator, and audit paths

## Linear Ticket Mirror

Status mirror last synced: `2026-04-28`.

Umbrella:

| Ticket | Status | Scope |
| --- | --- | --- |
| `QUA-982` | Backlog | Agent cycle: governed quant, critic, model-validator, and arbiter loop |

Implementation queue:

| Order | Ticket | Status | Objective | Hard blocker |
| --- | --- | --- | --- | --- |
| 1 | `QUA-983` | Done | ARC.0 - explicit cycle report contract | none |
| 2 | `QUA-984` | Done | ARC.1 - quant challenger packet | `QUA-983` |
| 3 | `QUA-985` | Done | ARC.2 - executable critic and arbiter claims | `QUA-984` |
| 4 | `QUA-986` | Backlog | ARC.3 - residual-risk model validation | `QUA-985` |
| 5 | `QUA-987` | Backlog | ARC.4 - promotion and adoption governance | `QUA-986` |
| 6 | `QUA-988` | Backlog | ARC.5 - product-grade cycle result surface | `QUA-987` |

## Purpose

This document defines the product roadmap for Trellis' quant, critic,
model-validator, and arbiter cycle.

The main goal is not merely to keep those stages alive. The goal is to make the
cycle itself the core governed product surface: one auditable loop that can
explain how Trellis chose a method, what assumptions it made, what deterministic
checks it ran, what conceptual risks remain, and why a route was or was not
trusted.

## Why This Plan Exists

The repo already contains serious pieces of the cycle:

- `quant.py` selects methods from canonical decompositions instead of relying on
  a second ad hoc policy table.
- `validation_contract.py` compiles route-ready deterministic checks,
  comparison relations, residual risks, and review hints.
- `review_policy.py` already uses deterministic rules to decide whether critic
  and model-validator LLM stages should run.
- `critic.py` is bounded to selecting deterministic check families rather than
  inventing arbitrary arbiter logic.
- `arbiter.py` executes deterministic checks without LLM judgment.
- `model_validator.py` combines deterministic sensitivity and benchmark checks
  with bounded conceptual review.
- `executor.py` already orchestrates the full sequence and emits lifecycle
  events such as `quant_selected_method`, `critic_completed`,
  `arbiter_completed`, and `model_validator_completed`.
- `platform_traces.py` persists the semantic boundary, validation contract, and
  lifecycle events for replay and audit.

That is enough machinery to justify treating the cycle as a first-class Trellis
product capability.

What is still missing is one coherent end-state program for this loop.

## Repo-Grounded Current State

### What is already strong

- Quant selection is canonical-first and already carries method reasoning,
  required market data, sensitivity support, and assumption summaries.
- Validation bundles and validation contracts already give the loop a typed
  deterministic substrate.
- Critic escalation is already bounded by deterministic review policy rather
  than running on every route.
- Arbiter execution is already fail-closed and deterministic.
- Model-validator review is already separated from code review and can focus on
  conceptual and calibration risk.
- Platform traces already persist lifecycle events and compact semantic /
  generation / validation boundaries.

### What is still weak

1. There is no single first-class cycle report or verdict object. The executor
   has stage outputs, events, and `gate_results`, but they are not yet one
   explicit product contract.
2. Quant emits one chosen method, but not yet a full challenge packet for the
   downstream reviewer stages. The critic and model validator still recover too
   much context indirectly.
3. Critic coverage is still narrow. The check library is small, and too much of
   the surface is still generic instead of family-specific and contract-driven.
4. Arbiter coverage is still narrow. It executes deterministic checks, but the
   executable claim surface is still much smaller than the semantic and
   numerical support surface.
5. Model-validator review still lacks a strong “residual-risk only” contract at
   the overall cycle level. It is better than a free-form reviewer, but it is
   not yet clearly the last stage in a layered claim system.
6. Promotion and adoption are not yet framed as explicit outputs of the cycle.
   The repo has promotion machinery, but the review cycle is not yet the
   authoritative graduation gate.
7. Observability is rich but still fragmented. Traces persist the pieces, yet
   there is no stable product-level scorecard that a desk, reviewer, or future
   runtime can consume directly.

## End State

The target system should look like this:

```text
request
  -> quant hypothesis selection
  -> compiled validation contract
  -> deterministic bundle / oracle checks
  -> critic-selected executable challenges
  -> arbiter verdicts
  -> model-validator residual conceptual review
  -> cycle report
  -> trace / audit / promotion decision
```

In the end state:

- Quant emits a ranked, reviewable pricing hypothesis rather than only one
  method label.
- The validation contract is the central authority for deterministic checks,
  comparison relations, residual risks, and review escalation.
- Critic can only select from executable checks already admitted by the
  validation contract.
- Arbiter executes those checks and records stable verdicts with no LLM
  judgment.
- Model validator reviews only the residual conceptual risk left after the
  deterministic layers.
- The executor returns a stable cycle report that the trace, Linear sync, and
  later promotion/adoption surfaces can all consume.
- Route promotion, adoption, and future approved-model lifecycle decisions are
  downstream consequences of the cycle, not separate informal judgments.

## Non-Goals

This plan does not attempt to:

- replace the deterministic pricing substrate
- turn critic or model-validator into open-ended code reviewers
- let LLM stages invent new deterministic checks on the fly
- make every supported pricing route immediately eligible for promotion
- solve external interoperability or document-ingestion work in the same wave

## Reverse Dependency Ladder

Work backward in this order:

1. stable cycle report and audit contract
2. validation-contract centrality
3. critic and arbiter executable-claim breadth
4. model-validator residual-risk discipline
5. quant hypothesis richness and challenger inputs
6. promotion / adoption / approved-route closeout

The critical point is that the cycle becomes real only when its outputs are
stable, auditable, and promotable. Richer agent behavior comes after the
contract, not before it.

## Delivery Rule

Do not widen any one stage in isolation.

- Do not make quant produce richer arguments if the downstream cycle cannot
  persist and judge them.
- Do not widen critic prompts without widening the executable arbiter surface.
- Do not widen model-validator prose if deterministic layers should have owned
  the claim instead.
- Do not treat promotion as complete if the cycle report is still informal.

## Cohorts

| Cohort | Objective | Core repo surfaces |
|--------|-----------|--------------------|
| `ARC.0` | Establish the explicit cycle contract | `executor.py`, `platform_traces.py`, `validation_contract.py` |
| `ARC.1` | Make quant output the right challenger packet | `quant.py`, `planner.py`, `platform_requests.py`, `validation_contract.py` |
| `ARC.2` | Harden critic and arbiter around executable claims | `critic.py`, `arbiter.py`, `validation_bundles.py`, `reference_oracles.py` |
| `ARC.3` | Narrow model-validator to residual conceptual risk | `model_validator.py`, `review_policy.py`, `validation_contract.py` |
| `ARC.4` | Wire cycle outputs into promotion and adoption | `knowledge/promotion.py`, `platform_traces.py`, `linear_tracker.py` |
| `ARC.5` | Expose the cycle as a product-grade result surface | `trellis.platform`, trace readers, docs, UI-facing projections |

## `ARC.0` — Explicit Cycle Contract

Objective: make the review cycle itself a stable runtime artifact.

Scope:

- define a first-class cycle report / scorecard shape
- project stage outcomes from executor into that shape
- persist the same projection into platform traces
- make stage identity and verdicts stable enough for audit, promotion, and UI

Queue:

- `ARC.0.1` Add a compact cycle-report projection over quant, bundle, oracle,
  critic, arbiter, and model-validator outcomes.
- `ARC.0.2` Persist that projection in `platform_traces.py`.
- `ARC.0.3` Expose stable stage identifiers, statuses, and failure summaries.
- `ARC.0.4` Add regression tests proving trace round-trips preserve the cycle
  report.
- `ARC.0.5` Remove any remaining places where stage meaning has to be inferred
  from ad hoc event strings alone.

Exit criteria:

- one stable cycle report exists
- executor and traces agree on stage outcomes
- downstream code no longer needs to reconstruct the loop from free-form event
  history

## `ARC.1` — Quant Challenger Packet

Objective: make quant a real hypothesis stage rather than only a method picker.

Scope:

- chosen method
- candidate alternatives
- assumption basis
- expected executable checks
- comparison / oracle expectations
- residual-risk handoff to later stages

Queue:

- `ARC.1.1` Extend quant output with explicit alternatives and rejection
  reasons.
- `ARC.1.2` Push quant-side assumptions and challenge expectations into the
  validation contract instead of keeping them prompt-local.
- `ARC.1.3` Make pricing-method identity, route family, and exact-binding
  identity consistent across quant, validation, and model validation.
- `ARC.1.4` Add tests proving downstream stages consume the compiled packet
  rather than reconstructing it heuristically.

Exit criteria:

- quant produces a reviewable hypothesis packet
- critic and model validator can rely on compiled upstream context rather than
  loose prompt reconstruction

## `ARC.2` — Critic And Arbiter Breadth

Objective: widen the executable challenge surface and make critic selection more
contract-driven.

Scope:

- richer check families
- stronger validation-contract mapping
- more reference and relation checks
- sharper deterministic failure taxonomy

Queue:

- `ARC.2.1` Expand critic-visible checks from generic sensitivities into
  family-aware executable claim types.
- `ARC.2.2` Move more check availability decisions behind
  `validation_contract.py`.
- `ARC.2.3` Broaden arbiter execution for relation-style and comparison-style
  claims.
- `ARC.2.4` Record structured arbiter verdicts instead of only failure strings.
- `ARC.2.5` Add coverage proving critic cannot select checks that arbiter cannot
  execute.

Exit criteria:

- critic selection is bounded by admitted executable claims
- arbiter verdicts cover a materially broader and more product-relevant cohort

## `ARC.3` — Residual-Risk Model Validation

Objective: make the model-validator the final conceptual gate, not a second
generic reviewer.

Scope:

- residual-risk-driven review prompts
- deterministic-first stage ownership
- stronger linkage between conceptual findings and upstream executed evidence

Queue:

- `ARC.3.1` Make residual risks explicit and stable at the overall cycle level.
- `ARC.3.2` Tighten review-policy rules so deterministic claim ownership is
  always preferred over conceptual prose.
- `ARC.3.3` Feed model-validator prompts the executed arbiter / oracle evidence,
  not only code and generic context.
- `ARC.3.4` Distinguish “conceptual blocker”, “calibration blocker”, and
  “residual limitation” in the stable report surface.

Exit criteria:

- model-validator findings are clearly residual conceptual findings
- the stage no longer duplicates deterministic checks that should belong to the
  arbiter path

## `ARC.4` — Promotion And Adoption Closeout

Objective: make cycle success or failure govern promotion, adoption, and later
approved-route lifecycle decisions.

Scope:

- promotion candidate review
- adoption safety
- stale / deprecated adapter lifecycle signals
- issue / audit handoff

Queue:

- `ARC.4.1` Define which cycle outcomes are required for promotion eligibility.
- `ARC.4.2` Persist those outcomes in promotion review artifacts.
- `ARC.4.3` Make stale / deprecated lifecycle state visible as cycle-level
  governance, not only knowledge-local hygiene.
- `ARC.4.4` Align Linear / audit comments with the stable cycle report.

Exit criteria:

- promotion and adoption are downstream consequences of the governed cycle
- reviewers can explain exactly why a route was promoted, blocked, or kept
  advisory-only

## `ARC.5` — Product Surface

Objective: expose the loop as a product-grade differentiator rather than an
internal implementation detail.

Scope:

- stable result projection
- desk / operator readable explanations
- UI / API consumable scorecards
- documentation and benchmark narrative

Queue:

- `ARC.5.1` Add a stable cycle-summary projection to user-facing result
  envelopes.
- `ARC.5.2` Document the cycle contract in `docs/user_guide/` and
  `docs/developer/`.
- `ARC.5.3` Publish benchmark and proving evidence for the cycle itself, not
  only for pricing routes.
- `ARC.5.4` Define the product claim carefully: what the cycle can honestly
  certify today and what still remains advisory.

Exit criteria:

- the cycle is visible and legible as part of the Trellis product contract
- the main selling point is documented as an auditable runtime capability, not
  only inferred from code

## Immediate Next Slice

The smallest honest near-term queue is:

1. `ARC.0.1` explicit cycle report
2. `ARC.0.2` trace persistence for that report
3. `ARC.1.3` stable pricing-method identity across quant, executor, and model
   validation
4. `ARC.2.4` structured arbiter verdicts
5. `ARC.3.3` feed executed deterministic evidence into model-validator review

As of April 23, 2026, the first hardening step in `ARC.1.3` is already in
progress: executor-side model validation now needs to consume the quant-selected
pricing method rather than unrelated spec-schema requirement labels.
