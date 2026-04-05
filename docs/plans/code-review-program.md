# Trellis Code Review Program

## Purpose

This document defines a repo-wide code review program for Trellis.

The goal is not to do one giant "review the whole repo" pass. The goal is to
turn review into a durable, high-signal workflow that:

1. breaks the codebase into reviewable slices
2. prioritizes the highest-risk and highest-churn areas first
3. reviews code and tests together rather than treating tests as an afterthought
4. improves local, regional, and global interaction coverage as part of review
5. produces concrete findings, follow-on issues, and documentation updates
6. can be decomposed into Linear issues without rewriting the plan later

This is a planning document. It is intentionally repo-grounded and should be
used as the source document for future review tickets.

## Why This Plan Exists

Trellis is growing quickly across several distinct architectures at once:

- deterministic pricing engines
- checked-in reference instruments
- a large agent/compiler/validation stack
- a governed platform and MCP surface
- a self-maintaining knowledge system

At the same time, other sessions are actively removing compatibility and
deprecated shims. That means a good review program must do more than look for
style issues. It must explicitly check:

- correctness and numerical soundness
- public API and runtime contract drift
- interaction behavior across modules, classes, and functions
- whether tests only prove local behavior or also prove subsystem and workflow behavior
- legacy-path and shim removal opportunities
- mismatch between code, tests, docs, and `LIMITATIONS.md`

## Repo-Grounded Current State

Snapshot collected on `2026-04-04` from the checked-in repo:

- `trellis/` contains roughly `98k` lines of Python
- `tests/` contains roughly `61k` lines of Python
- `docs/` contains roughly `20k` lines of Markdown / reStructuredText
- `trellis/agent/` alone contains roughly `54k` lines of Python plus roughly
  `139k` lines of knowledge YAML and related declarative assets

Recent 90-day churn is concentrated in:

- `trellis/agent`
- `trellis/models`
- `trellis/instruments`
- `trellis/platform/services`
- supporting tests in `tests/test_agent`, `tests/test_models`, and
  `tests/test_mcp`

Large files that must not be reviewed as one undifferentiated unit include:

- `trellis/agent/executor.py`
- `trellis/agent/semantic_contracts.py`
- `trellis/agent/knowledge/promotion.py`
- `trellis/agent/task_runtime.py`
- `trellis/agent/semantic_contract_validation.py`
- `trellis/agent/codegen_guardrails.py`
- `trellis/platform/services/pricing_service.py`
- `trellis/models/trees/algebra.py`
- `trellis/models/trees/lattice.py`
- `trellis/models/monte_carlo/engine.py`

That shape is the reason for this program. A credible review process here must
be risk-based, wave-based, and aggressively split.

## Review Objectives

Every review slice should answer the same questions:

1. Is the implementation correct for the contract it claims to implement?
2. Are numerical assumptions, edge cases, and failure modes explicit?
3. Are runtime contracts, protocols, dataclasses, and public surfaces coherent?
4. Are module, class, and function interactions explicit and defendable?
5. Are tests proving the important behavior at local, regional, and global scope?
6. Are docs and `LIMITATIONS.md` honest about what is and is not supported?
7. Can compatibility shims, deprecated entry points, or dead routes be removed?

## Testing Posture This Program Requires

Local TDD is necessary, but it is not sufficient for Trellis.

The repo already has the beginnings of a broader testing pyramid:

- local/unit tests in `tests/test_core`, `tests/test_models`, `tests/test_agent`,
  `tests/test_platform`, and similar directories
- bounded interaction tests such as `tests/test_pipeline.py`,
  `tests/test_platform/test_executor.py`,
  `tests/test_agent/test_dsl_integration.py`, and
  `tests/test_agent/test_platform_loop.py`
- broader workflow and oracle surfaces in `tests/test_tasks`,
  `tests/test_crossval`, `tests/test_verification`, and selected MCP tests

This program treats that broader testing surface as part of the review target,
not as optional polish.

### Required Test Layers

Every substantive review ticket must classify coverage at these three layers:

- `Local`
  Single function, class, or module behavior. Typical TDD scope.
- `Regional`
  Bounded subsystem interactions across multiple modules, classes, or
  functions. Examples: request compilation to execution, calibration workflow
  to `MarketState` handoff, route lowering to helper-backed kernel.
- `Global`
  User-visible or workflow-visible behavior across major layers. Examples:
  `Session.price(...)`, `Pipeline.run()`, MCP `price.trade`, task
  cross-validation, and verification/benchmark replay.

### Review Rule

For any behavior that matters to users, desk credibility, governance, or
regression safety, the reviewer must ask:

1. Is there a local test?
2. Is there a regional interaction test?
3. Is there a global or workflow-level regression surface?

If the answer is "no" at regional or global scope for an important behavior,
that is a review finding, not a note for later.

### Test Quality Questions

When reviewing tests, do not stop at coverage count. Check:

- whether the test asserts contracts or just implementation details
- whether the seam between modules is exercised with real objects rather than
  excessive mocking
- whether failure messages are diagnostic
- whether the test would catch authority drift after shim removal
- whether the test proves provenance, audit, warning, and fallback behavior
  where those are part of the contract
- whether cross-validation and verification surfaces still match the checked
  scope in docs and `LIMITATIONS.md`

## Non-Goals

This program is not intended to:

- perform repo-wide style churn or mass formatting
- rewrite modules during the review ticket unless the fix is trivial
- line-review every generated or declarative artifact before triage
- replace domain verification or benchmark work with code reading alone

## Linear Ticket Mirror

These tables mirror the current Linear tickets for the code-review program and
their intended implementation order.

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This plan file is the repo-local mirror for subsequent agents.
- Implement the earliest ticket in the ordered queue below whose status is not
  `Done`.
- Tickets inside the same wave may run in parallel when their write scopes do
  not conflict.
- Treat each review ticket as a review-and-fix slice, not as a read-only audit:
  low-risk in-scope code, test, doc, and `LIMITATIONS.md` changes should land
  inside the ticket; larger remediations should be split into follow-on issues.
- After a ticket is closed in Linear, update the corresponding row in this
  table in the same closeout.

Status mirror last synced: `2026-04-05`

### Review Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-647` Code review program: repo-wide review backlog and interaction-test hardening | Done |

### Wave 0 Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-648` | `CR-00` Review process: dossier template, severity rubric, and closeout template | Done |

### Wave 1 Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-649` | `CR-01` Public API review: top-level entry points and session surface | Done |
| `QUA-650` | `CR-02` Platform core review: governed execution context, policies, runs, and results | Done |
| `QUA-651` | `CR-03` Platform services review: model, provider, snapshot, and pricing services | Done |
| `QUA-652` | `CR-04` MCP surface review: tool registry, resources, prompts, and transport | Done |

### Wave 2 Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-653` | `CR-05` Core contract review: protocols, types, market state, and differentiability | Done |
| `QUA-654` | `CR-06` Convention review: dates, schedules, day count, and calendars | Done |
| `QUA-655` | `CR-07` Curve review: interpolation, shocks, bootstrap, and scenario packs | Done |
| `QUA-657` | `CR-08` Market data review: resolver, snapshots, providers, and mock handling | Done |
| `QUA-656` | `CR-09` Runtime analytics review: pricing engine, payoff pricer, and measures | Done |

### Wave 3 Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-658` | `CR-10` Instrument review: checked-in reference instruments | Done |
| `QUA-659` | `CR-11` Tree review: lattice algebra, trees, and early-exercise products | Done |
| `QUA-660` | `CR-12` Monte Carlo review: engine, state, schemes, variance reduction, and exercise policy | Done |
| `QUA-661` | `CR-13` PDE and transform review: grid methods, theta/PSOR, FFT, and COS | Done |
| `QUA-662` | `CR-14` Analytical and process review: Black, analytical kernels, processes, and vol surfaces | Done |
| `QUA-663` | `CR-15` Calibration review: rates, local vol, SABR, Heston, and solve requests | Done |
| `QUA-664` | `CR-16` Structured model review: cashflow, credit, copula, resolution, and structured helpers | Done |
| `QUA-665` | `CR-17` Generated route review: agent-generated instrument adapters and freshness policy | Done |

### Wave 4 Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-666` | `CR-18` Semantic review: product IR, contracts, family lowering, and validators | Done |
| `QUA-667` | `CR-19` Routing review: platform requests, route registry, route scoring, and market binding | Done |
| `QUA-668` | `CR-20` Generation review: planner, quant, builder, prompts, and guardrails | Done |
| `QUA-669` | `CR-21` Validation review: executor, critic, arbiter, lite review, model validator, and audit paths | Done |
| `QUA-670` | `CR-22` Task runtime review: runtime loops, diagnostics, checkpoints, evals, and traces | Done |
| `QUA-671` | `CR-23` Knowledge runtime review: store, retrieval, decomposition, reflection, and promotion code | Done |
| `QUA-672` | `CR-24` Knowledge schema review: canonical maps, feature taxonomy, requirements, and blockers | Done |
| `QUA-673` | `CR-25` Knowledge hygiene review: lessons index, promotion hygiene, and trace retention | Done |

### Wave 5 Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-674` | `CR-26` Test architecture review: local, regional, global, cross-validation, verification, and legacy markers | Done |
| `QUA-675` | `CR-27` Documentation review: public guides, developer docs, plans, and limitations alignment | Done |
| `QUA-676` | `CR-28` Compatibility review: deprecated paths, wrappers, shims, and removal candidates | Done |
| `QUA-677` | `CR-29` Review synthesis: aggregate findings, remediation roadmap, and policy updates | Done |

## Best-Practice Review Operating Model

### Review Unit Size

A review ticket should normally include:

- `4-12` production files
- the directly related tests at local, regional, and global scope where they exist
- the directly related docs or `LIMITATIONS.md` entries

Target size:

- `400-1200` lines of production code per ticket
- or one coherent subsystem if the subsystem is naturally smaller

Hard rule:

- any file above roughly `1000` lines must be reviewed in concern-based
  sub-slices, not as a single ticket

### Reviewer Roles

Each review ticket should have:

- one primary reviewer responsible for the deep read
- one challenger reviewer who did not recently author most of the slice
- AI assistance allowed for dossier preparation, grep, and test mapping, but
  not as a substitute for grounded findings

### Required Review Workflow

Every ticket should follow this sequence:

1. Build the review dossier.
   Include scope, churn summary, known limitations, relevant docs, the
   interaction map, and the test commands that should prove the area at local,
   regional, and global scope.
2. Run the static review.
   Read code for contracts, invariants, assumptions, edge cases, and obvious
   dead or duplicate paths. Review the tests with the same seriousness as the
   production code.
3. Run the execution review.
   Execute the targeted tests when practical. If local tests exist but regional
   or global interaction tests are missing, misleading, or too broad, record
   that as a finding.
4. Record findings with severity.
   Findings are the primary output, not narrative summary.
5. Close with a disposition.
   Mark the slice `cleared`, `cleared with follow-ons`, or `blocked`.

### Required Dossier Artifacts

Every dossier should include:

- the code files in scope
- the directly relevant tests in scope
- the public or internal contracts being defended
- the inbound and outbound module interactions
- the current local / regional / global test map
- the known limitations, docs, and benchmark artifacts that constrain claims

### Severity Rubric

Use this rubric consistently:

- `P0`: wrong result, silent corruption, or broken governance on a critical path
- `P1`: high-confidence correctness, runtime, or contract bug
- `P2`: meaningful maintainability, test, or observability gap with real risk
- `P3`: cleanup, simplification, or lower-risk consistency gap

### Required Outputs Per Review Ticket

Each ticket must produce:

- prioritized findings with file references
- missing-test and misleading-test notes
- a local / regional / global coverage assessment
- interaction seams that are currently unguarded
- doc and `LIMITATIONS.md` mismatches
- follow-on implementation tickets where needed
- an explicit statement on shim-removal opportunities

### WIP Limits And Cadence

To keep the review program sharp:

- keep at most `3` review tickets in flight at once
- do not open the next wave until the current wave has a closed findings pass
- prefer short, reviewable tickets over broad "audit everything in X" tickets

## Special Handling Rules

### Generated And Semi-Generated Route Artifacts

`trellis/instruments/_agent/` should not receive the same first-pass treatment
as hand-written pricing code.

Review focus there should be:

- freshness and divergence versus validated `_fresh` snapshots
- public contract shape and import hygiene
- whether the checked-in adapter is stale, duplicate, or removable
- whether coverage exists at the route boundary

### Knowledge Lessons And Traces

`trellis/agent/knowledge/lessons/` and `trellis/agent/knowledge/traces/` are
too large and too dynamic for line-by-line first-pass review.

Review focus there should be:

- schema integrity
- index consistency
- promotion and archival policy correctness
- sampling-based content review for quality drift
- retention and noise control

### Excluded From The Review Queue

These should be excluded from the formal code-review backlog unless a ticket
specifically targets them:

- `__pycache__/`
- transient `task_runs/` outputs
- one-off local caches and generated artifacts outside versioned source

## Mandatory Interaction-Seam Review

The review program must explicitly inspect and test seams like these:

- public API surface -> request compilation -> governed executor
- `Session` / `Pipeline` -> execution context -> platform services
- trade parse -> semantic contract -> `ProductIR` -> route selection
- route lowering -> helper-backed numerical engine -> runtime result shaping
- market resolution -> provider binding -> snapshot persistence -> audit bundle
- calibration workflow -> solver provenance -> `MarketState` handoff -> runtime reuse
- analytics / scenario packs -> curve shocks -> book or payoff runtime
- task runtime -> validation bundles -> audit artifacts -> knowledge reflection
- MCP tool surface -> platform service -> run ledger -> audit/resource retrieval

Any critical seam that lacks a realistic interaction test should be tracked as
part of the review result.

## Mandatory Micro-Splits For Oversized Files

These files should be split before ticket creation:

| File | Split guidance |
| --- | --- |
| `trellis/agent/executor.py` | split into orchestration, validation/audit, and fallback/reuse behavior |
| `trellis/agent/semantic_contracts.py` | split into contract datamodel/parsing and lowering/legacy mirrors |
| `trellis/agent/knowledge/promotion.py` | split into promotion lifecycle and archival/distillation mechanics |
| `trellis/agent/task_runtime.py` | split into execution orchestration and persistence/reporting |
| `trellis/agent/semantic_contract_validation.py` | split into admissibility rules and comparison/review logic |
| `trellis/agent/codegen_guardrails.py` | split into prompt/input contracts and generated-code validation |
| `trellis/platform/services/pricing_service.py` | split into request execution and result/audit shaping |
| `trellis/models/trees/algebra.py` | split into algebra substrate and route-facing assembly helpers |
| `trellis/models/trees/lattice.py` | split into lattice construction and backward/exercise behavior |
| `trellis/models/monte_carlo/engine.py` | split into path generation/core engine and runtime greek/differentiability behavior |

## Ordered Review Waves

### Wave 0: Review Program Scaffolding

| Slice | Proposed title | Scope | Main focus |
| --- | --- | --- | --- |
| `CR-00` | `Review process: Trellis review dossier, severity rubric, and closeout template` | this plan, review templates, finding format, issue template | make review repeatable before starting deep audits |

### Wave 1: Public And Governed Runtime Surface

| Slice | Proposed title | Scope | Main focus |
| --- | --- | --- | --- |
| `CR-01` | `Public API review: top-level entry points and session surface` | `trellis/__init__.py`, `trellis/session.py`, `trellis/pipeline.py`, `trellis/book.py`, `trellis/samples.py`, corresponding tests | API coherence, runtime intent, hidden fallback paths, user-visible contract drift, public-workflow regression coverage |
| `CR-02` | `Platform core review: governed execution context, policies, runs, and results` | `trellis/platform/context.py`, `executor.py`, `policies.py`, `results.py`, `runs.py`, `audits.py`, related tests | governed execution correctness, provenance, result stability, audit integrity, cross-service interaction safety |
| `CR-03` | `Platform services review: model, provider, snapshot, and pricing services` | `trellis/platform/services/*`, `trellis/platform/models.py`, `providers.py`, `storage.py`, related tests | service boundaries, persistence rules, model-selection behavior, snapshot integrity, service-to-service workflow tests |
| `CR-04` | `MCP surface review: tool registry, resources, prompts, and transport` | `trellis/mcp/*`, `tests/test_mcp/*` | thin-surface discipline, error handling, stable contracts, governed-only behavior, end-to-end tool-to-service coverage |

### Wave 2: Deterministic Pricing Substrate

| Slice | Proposed title | Scope | Main focus |
| --- | --- | --- | --- |
| `CR-05` | `Core contract review: protocols, types, market state, and differentiability` | `trellis/core/*`, `tests/test_core/*`, `tests/test_public_api_surface.py` | protocol purity, frozen dataclasses, autograd boundary, shared invariants, downstream interaction assumptions |
| `CR-06` | `Convention review: dates, schedules, day count, and calendars` | `trellis/conventions/*`, `trellis/core/date_utils.py`, related tests | financial convention correctness, schedule edge cases, naming consistency, cross-module schedule propagation |
| `CR-07` | `Curve review: interpolation, shocks, bootstrap, and scenario packs` | `trellis/curves/*`, related tests | interpolation assumptions, off-grid behavior, scenario correctness, bootstrap realism, scenario-to-runtime coverage |
| `CR-08` | `Market data review: resolver, snapshots, providers, and mock handling` | `trellis/data/*`, related tests and user docs later referenced by closeout | provider fallback discipline, snapshot fidelity, mock leakage, schema stability, resolution-to-runtime interaction tests |
| `CR-09` | `Runtime analytics review: pricing engine, payoff pricer, and measures` | `trellis/engine/*`, `trellis/analytics/*`, `tests/test_engine/*`, `tests/test_pipeline.py`, targeted verification tests | runtime pricing path correctness, risk surface honesty, error handling, observability, scenario and analytics integration coverage |

### Wave 3: Instruments And Numerical Engines

| Slice | Proposed title | Scope | Main focus |
| --- | --- | --- | --- |
| `CR-10` | `Instrument review: checked-in reference instruments` | hand-written files in `trellis/instruments/` excluding `_agent/`, related tests | contract clarity, market input use, parity with engine capabilities, instrument-to-engine interaction coverage |
| `CR-11` | `Tree review: lattice algebra, trees, and early-exercise products` | `trellis/models/trees/*`, `callable_bond_tree.py`, `bermudan_swaption_tree.py`, `zcb_option_tree.py`, `equity_option_tree.py`, tree tests and task tests | exercise logic, node semantics, backward induction, legacy wrappers, route-to-lattice regional tests |
| `CR-12` | `Monte Carlo review: engine, state, schemes, variance reduction, and exercise policy` | `trellis/models/monte_carlo/*`, related tests and task tests | path generation, regression bias, barrier/path semantics, differentiability claims, multi-module simulation workflow tests |
| `CR-13` | `PDE and transform review: grid methods, theta/PSOR, FFT, and COS` | `trellis/models/pde/*`, `equity_option_pde.py`, `trellis/models/transforms/*`, `grid_protocols.py`, related tests | discretization correctness, operator boundaries, stability claims, doc honesty, solver-to-contract interaction coverage |
| `CR-14` | `Analytical and process review: Black, analytical kernels, processes, and vol surfaces` | `trellis/models/black.py`, `trellis/models/analytical/*`, `trellis/models/processes/*`, `vol_surface.py`, related tests | formula correctness, model assumptions, parameter semantics, process reuse, process-to-pricer compatibility coverage |
| `CR-15` | `Calibration review: rates, local vol, SABR, Heston, and solve requests` | `trellis/models/calibration/*`, `hull_white_parameters.py`, calibration tests, benchmark docs where relevant | calibration realism, diagnostics, parameter reuse, limitation truthfulness, workflow tests from solve request to runtime reuse |
| `CR-16` | `Structured model review: cashflow, credit, copula, resolution, and structured helpers` | `trellis/models/cashflow_engine/*`, `credit_default_swap.py`, `range_accrual.py`, `contingent_cashflows.py`, `trellis/models/resolution/*`, `trellis/models/copulas/*`, related tests | structured-product semantics, tranche/range logic, cross-asset glue correctness, cross-module assembly coverage |
| `CR-17` | `Generated route review: agent-generated instrument adapters and freshness policy` | `trellis/instruments/_agent/*` including `_fresh`, task tests and route docs | stale adapters, duplicate routes, import drift, checked-in versus fresh divergence, route-boundary regression safety |

### Wave 4: Agent, Compiler, And Knowledge Stack

| Slice | Proposed title | Scope | Main focus |
| --- | --- | --- | --- |
| `CR-18` | `Semantic review: product IR, contracts, family lowering, and validators` | `trellis/agent/semantic_contracts.py`, `semantic_contract_compiler.py`, `semantic_contract_validation.py`, `semantic_concepts.py`, `family_*`, `semantic_validators/*`, related tests | semantic contract integrity, lowering correctness, legacy mirror drift, blocker taxonomy, cross-layer semantic integration tests |
| `CR-19` | `Routing review: platform requests, route registry, route scoring, and market binding` | `trellis/agent/platform_requests.py`, `route_registry.py`, `route_scorer.py`, `market_binding.py`, `valuation_context.py`, `ask.py`, related tests | request compilation correctness, route determinism, market binding completeness, route-to-runtime interaction coverage |
| `CR-20` | `Generation review: planner, quant, builder, prompts, and guardrails` | `trellis/agent/planner.py`, `quant.py`, `builder.py`, `prompts.py`, `codegen_guardrails.py`, `build_gate.py`, related tests | generation contracts, prompt scope, import hygiene, guardrail coverage, end-to-end build-loop safety tests |
| `CR-21` | `Validation review: executor, critic, arbiter, lite review, model validator, and audit paths` | relevant concern-based slices of `trellis/agent/executor.py`, plus `critic.py`, `arbiter.py`, `lite_review.py`, `model_validator.py`, `validation_contract.py`, `validation_bundles.py`, `reference_oracles.py`, related tests | fallback control, validation truthfulness, best-effort behavior, review-stage coherence, validation-pipeline interaction coverage |
| `CR-22` | `Task runtime review: runtime loops, diagnostics, checkpoints, evals, and traces` | `trellis/agent/task_runtime.py`, `task_run_store.py`, `task_diagnostics.py`, `checkpoints.py`, `evals.py`, `platform_traces.py`, related tests and scripts if needed for context | reproducibility, persistence shape, failure attribution, benchmark integrity, batch and regression workflow coverage |
| `CR-23` | `Knowledge runtime review: store, retrieval, decomposition, reflection, and promotion code` | executable Python in `trellis/agent/knowledge/` excluding canonical and lessons YAML, related tests | retrieval correctness, promotion safety, import registry trust model, stale-knowledge risk, retrieval-to-build-loop interaction coverage |
| `CR-24` | `Knowledge schema review: canonical maps, feature taxonomy, requirements, and blockers` | `trellis/agent/knowledge/canonical/*`, `api_map.py`, `instructions.py`, `skills.py`, knowledge tests | code-to-canonical consistency, taxonomy drift, cookbook and requirement accuracy, schema-to-runtime alignment |
| `CR-25` | `Knowledge hygiene review: lessons index, promotion hygiene, and trace retention` | `trellis/agent/knowledge/lessons/*`, `trellis/agent/knowledge/traces/*`, supporting scripts/tests where present | sampling quality, index integrity, retention policy, noise control, trace usefulness for regional/global regression |

### Wave 5: Quality Surface And Closeout

| Slice | Proposed title | Scope | Main focus |
| --- | --- | --- | --- |
| `CR-26` | `Test architecture review: local, regional, global, cross-validation, verification, and legacy markers` | `tests/` as a system, with emphasis on structure rather than line-by-line first | duplicated coverage, misleading tests, missing negative cases, unguarded interaction seams, legacy-only tests |
| `CR-27` | `Documentation review: public guides, developer docs, plans, and limitations alignment` | `README.md`, `docs/**`, `LIMITATIONS.md` | claim-versus-code alignment, stale docs, limitation closeout discipline |
| `CR-28` | `Compatibility review: deprecated paths, wrappers, shims, and removal candidates` | repo-wide grep-backed pass across legacy wrappers, compatibility helpers, and `legacy_compat` tests | removal plan, migration blockers, residual callers, user-facing compatibility policy |
| `CR-29` | `Review synthesis: aggregate findings, remediation roadmap, and policy updates` | outputs from all prior review tickets | turn findings into prioritized implementation work and long-lived review policy |

## Completed Review Dossiers

### Wave 1 Closeout

#### `QUA-649` / `CR-01`

- Reviewed:
  - `trellis/__init__.py`
  - `trellis/session.py`
  - `trellis/pipeline.py`
  - `trellis/book.py`
  - `trellis/samples.py`
  - `tests/test_session.py`
  - `tests/test_pipeline.py`
  - `tests/test_public_api_surface.py`
  - `tests/test_samples.py`
- Findings:
  - Added missing global interaction coverage for package-level `trellis.ask(...)`.
  - New test asserts the default `sample_session()` path seeds a `FlatVol(0.20)` before delegating to `trellis.agent.ask.ask_session`.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_session.py tests/test_pipeline.py tests/test_public_api_surface.py tests/test_samples.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on package exports, session helpers, scenario helpers, and book behavior.
  - `Regional`: strong on `Session` and `Pipeline` delegation through `_run_governed_request`.
  - `Global`: strengthened by the new package-level `trellis.ask(...)` interaction test.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-650` / `CR-02`

- Reviewed:
  - `trellis/platform/context.py`
  - `trellis/platform/executor.py`
  - `trellis/platform/policies.py`
  - `trellis/platform/results.py`
  - `trellis/platform/runs.py`
  - `trellis/platform/audits.py`
  - `tests/test_platform/test_context.py`
  - `tests/test_platform/test_executor.py`
  - `tests/test_platform/test_policies.py`
  - `tests/test_platform/test_runs.py`
  - `tests/test_platform/test_audit_bundle.py`
- Findings:
  - Fixed a broken regional executor test harness in `tests/test_platform/test_executor.py` so the `build_then_price` default adapter path is exercised correctly on this interpreter.
  - No additional in-scope production-code findings remained after the full platform-core regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_platform/test_context.py tests/test_platform/test_executor.py tests/test_platform/test_policies.py tests/test_platform/test_runs.py tests/test_platform/test_audit_bundle.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on context normalization, policy evaluation, and result projection.
  - `Regional`: strong on executor-to-pricer, run-ledger-to-audit-bundle, and blocked-run failure-context seams.
  - `Global`: covered through MCP run/audit retrieval tests in the broader Wave 1 regression suite.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-651` / `CR-03`

- Reviewed:
  - `trellis/platform/providers.py`
  - `trellis/platform/models.py`
  - `trellis/platform/storage.py`
  - `trellis/platform/services/bootstrap.py`
  - `trellis/platform/services/model_service.py`
  - `trellis/platform/services/pricing_service.py`
  - `trellis/platform/services/session_service.py`
  - `trellis/platform/services/snapshot_service.py`
  - `trellis/platform/services/trade_service.py`
  - `tests/test_platform/test_provider_registry.py`
  - `tests/test_platform/test_model_service.py`
  - `tests/test_platform/test_model_execution_gate.py`
  - `tests/test_platform/test_pricing_service.py`
  - `tests/test_platform/test_snapshot_service.py`
  - `tests/test_platform/test_session_service.py`
  - `tests/test_platform/test_storage.py`
  - global service workflows in `tests/test_mcp/test_trade_service.py`, `tests/test_mcp/test_model_match_service.py`, and `tests/test_mcp/test_price_trade_tool.py`
- Findings:
  - Fixed a service-boundary bug in `SessionService.activate_market_snapshot(...)`: the session service now requires a persisted snapshot id instead of storing a broken dangling reference.
  - Session activation now uses the persisted snapshot's canonical provider binding rather than trusting the caller's provider id blindly.
  - Added direct service tests for both the success and failure path.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_platform/test_provider_registry.py tests/test_platform/test_model_service.py tests/test_platform/test_model_execution_gate.py tests/test_platform/test_pricing_service.py tests/test_platform/test_snapshot_service.py tests/test_platform/test_session_service.py tests/test_platform/test_storage.py tests/test_mcp/test_trade_service.py tests/test_mcp/test_model_match_service.py tests/test_mcp/test_price_trade_tool.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on provider identity, model lifecycle selection, snapshot persistence, and trade parsing.
  - `Regional`: improved on the session-service-to-snapshot-store seam with the new direct activation tests.
  - `Global`: strong on parse -> match -> policy -> snapshot -> price -> run/audit workflows through the governed pricing tool tests.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-652` / `CR-04`

- Reviewed:
  - `trellis/mcp/tool_registry.py`
  - `trellis/mcp/resources.py`
  - `trellis/mcp/prompts.py`
  - `trellis/mcp/http_transport.py`
  - `trellis/mcp/server.py`
  - `tests/test_mcp/*`
- Findings:
  - No additional in-scope production-code findings after reviewing the thin MCP registry, prompt, resource, and HTTP transport surfaces against the global MCP test suite.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_mcp -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on prompt/resource packaging and resource resolution.
  - `Regional`: strong on tool-registry-to-service and HTTP-transport-to-shell seams.
  - `Global`: strong on end-to-end governed pricing, session, run, and snapshot flows through MCP tools.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-653` / `CR-05`

- Reviewed:
  - `trellis/core/__init__.py`
  - `trellis/core/types.py`
  - `trellis/core/market_state.py`
  - `trellis/core/differentiable.py`
  - `trellis/core/payoff.py`
  - `trellis/core/capabilities.py`
  - `trellis/core/state_space.py`
  - `trellis/core/runtime_contract.py`
  - `tests/test_core/*`
  - `tests/test_engine/test_payoff_pricer.py`
  - `tests/test_public_api_surface.py`
- Findings:
  - Fixed a shared contract bug in `MarketState.available_capabilities`: empty mapping-valued inputs no longer advertise capabilities that are not actually present.
  - Added explicit differentiability-wrapper coverage for `get_numpy`, `gradient`, `hessian`, and `jacobian` so the autograd boundary is exercised directly rather than only through downstream pricing tests.
  - Added a capability-gap regression for `check_market_data(...)` so empty `fx_rates` dictionaries fail correctly instead of passing as if market data exists.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_core tests/test_engine/test_payoff_pricer.py tests/test_public_api_surface.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_data/test_market_snapshot.py tests/test_data/test_mock.py tests/test_session.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strengthened on capability enumeration, capability-gap reporting, and differentiability wrappers.
  - `Regional`: improved on `MarketState` -> `check_market_data(...)` -> payoff-pricer contract enforcement.
  - `Global`: adequate through session and market-snapshot regressions that consume the same shared market-state contract.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-654` / `CR-06`

- Reviewed:
  - `trellis/conventions/calendar.py`
  - `trellis/conventions/day_count.py`
  - `trellis/conventions/schedule.py`
  - `trellis/conventions/rate_index.py`
  - `trellis/core/date_utils.py`
  - `tests/test_conventions/*`
  - `tests/test_core/test_event_schedule.py`
  - `tests/test_verification/test_conventions_edge.py`
  - `tests/test_security/test_schedule.py`
- Findings:
  - Fixed the legacy `trellis.core.date_utils.get_bracketing_dates(...)` compatibility helper so it brackets dates across multi-year schedules instead of stopping after the first coupon year.
  - `get_accrual_fraction(...)` now inherits the corrected multi-year behavior through the shared bracketing path.
  - Added local multi-year compatibility tests plus a regional interaction test that checks the legacy helper against `build_period_schedule(...)` on a multi-year schedule.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_security/test_schedule.py tests/test_core/test_event_schedule.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_conventions tests/test_core/test_event_schedule.py tests/test_verification/test_conventions_edge.py tests/test_security/test_schedule.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on day-count formulas, holiday adjustment rules, stub placement, and EOM handling; improved on legacy multi-year schedule helpers.
  - `Regional`: improved on `trellis.core.date_utils` -> `trellis.conventions.schedule` alignment through the new multi-year comparison against `build_period_schedule(...)`.
  - `Global`: adequate through convention-edge verification and pricing tests, although the corrected helper remains a compatibility path rather than the dominant route-safe scheduling API.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - `trellis.core.date_utils.get_bracketing_dates(...)` and `get_accrual_fraction(...)` remain legacy compatibility helpers
- Follow-on tickets:
  - fold the duplicate legacy schedule logic in `rate_model/security/schedule.py` into `QUA-676` / `CR-28`
- Disposition:
  - `Done`

#### `QUA-655` / `CR-07`

- Reviewed:
  - `trellis/curves/yield_curve.py`
  - `trellis/curves/interpolation.py`
  - `trellis/curves/bootstrap.py`
  - `trellis/curves/shocks.py`
  - `trellis/curves/scenario_packs.py`
  - `trellis/curves/forward_curve.py`
  - `trellis/curves/credit_curve.py`
  - `tests/test_curve/test_yield_curve.py`
  - `tests/test_curves/*`
  - curve-consuming workflows in `tests/test_pipeline.py`, `tests/test_session.py`, `tests/test_crossval/test_xv_curves.py`, and `tests/test_public_api_surface.py`
- Findings:
  - Fixed a silent mispricing risk in `YieldCurve` and `CreditCurve`: both curve classes now reject empty, length-mismatched, and non-increasing knot grids instead of letting interpolation/shock consumers operate on invalid surfaces.
  - Added explicit constructor regression tests for those invalid-grid cases.
  - Added an off-grid named scenario-pack workflow test through `Pipeline` so the review now covers the shared `scenario_pack -> curve shock -> repricing` seam on a non-native bucket grid.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_curve/test_yield_curve.py tests/test_curves/test_credit_curve.py tests/test_pipeline.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_curve tests/test_curves tests/test_crossval/test_xv_curves.py tests/test_pipeline.py tests/test_session.py tests/test_public_api_surface.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on interpolation, forward extraction, bootstrap repricing, bucket shocks, scenario pack construction, and the new invalid-grid guardrails.
  - `Regional`: strengthened on off-grid shock propagation through `YieldCurve.bump(...)`, shared shock surfaces, and pipeline scenario-pack expansion.
  - `Global`: strong on session/pipeline scenario workflows and QuantLib cross-validation for flat-curve discounting and forward extraction.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-657` / `CR-08`

- Reviewed:
  - `trellis/data/resolver.py`
  - `trellis/data/schema.py`
  - `trellis/data/mock.py`
  - `trellis/data/file_snapshot.py`
  - `tests/test_data/test_resolver.py`
  - `tests/test_data/test_mock.py`
  - `tests/test_data/test_market_snapshot.py`
  - snapshot-consuming workflows in `tests/test_session.py`, `tests/test_platform/test_snapshot_service.py`, `tests/test_platform/test_session_service.py`, `tests/test_mcp/test_trade_service.py`, `tests/test_mcp/test_model_match_service.py`, and `tests/test_mcp/test_price_trade_tool.py`
- Findings:
  - Fixed a resolver merge bug in `resolve_market_snapshot(...)`: the singular `vol_surface=` path now honors an explicit `default_vol_surface=` name instead of silently hard-coding the default slot and overwriting the inherited provider default surface key.
  - Added a focused resolver regression for the explicit singular vol-surface/default-name path.
  - Added a provider-merge interaction test on the mock snapshot path so the review now covers provider snapshot + explicit override behavior rather than only isolated resolver inputs.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_data/test_resolver.py tests/test_data/test_mock.py tests/test_data/test_market_snapshot.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_data tests/test_session.py tests/test_platform/test_snapshot_service.py tests/test_platform/test_session_service.py tests/test_mcp/test_trade_service.py tests/test_mcp/test_model_match_service.py tests/test_mcp/test_price_trade_tool.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on snapshot schema accessors, resolver merge paths, provider fallback, and mock snapshot assembly.
  - `Regional`: improved on provider snapshot -> resolver override -> canonical default-selection behavior through the new named vol-override test.
  - `Global`: strong on session-service, snapshot-service, and governed trade-pricing flows that consume resolved snapshots at runtime.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-656` / `CR-09`

- Reviewed:
  - `trellis/engine/pricer.py`
  - `trellis/engine/payoff_pricer.py`
  - `trellis/engine/analytics.py`
  - `trellis/analytics/measures.py`
  - `tests/test_engine/*`
  - runtime/workflow coverage in `tests/test_session.py`, `tests/test_pipeline.py`, `tests/test_platform/test_pricing_service.py`, `tests/test_verification/test_analytical_pricing.py`, and `tests/test_public_api_surface.py`
- Findings:
  - Fixed a runtime analytics integration bug in `trellis.analytics.measures`: cloned or shifted `MarketState` objects now preserve fixing histories, selected curve names, and market provenance instead of dropping them during rate-bump and scenario analytics.
  - Switched the remaining manual shifted-state builders in `ZSpread` and `ScenarioPnL` onto the shared `_clone_market_state(...)` path so the runtime analytics boundary uses one consistent state-preservation mechanism.
  - Added interaction tests that exercise `DV01` and `ScenarioPnL` with a payoff that requires historical fixings, proving bumped and shifted analytics repricing now carries those inputs through correctly.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_engine/test_pricer.py tests/test_engine/test_payoff_pricer.py tests/test_session.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_engine tests/test_pipeline.py tests/test_session.py tests/test_platform/test_pricing_service.py tests/test_verification/test_analytical_pricing.py tests/test_public_api_surface.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on payoff pricing, accrued-interest and YTM projections, autodiff Greeks, and direct analytics measure behavior.
  - `Regional`: improved on `MarketState` -> analytics measure -> bumped runtime repricing seams through the new fixing-aware DV01 and scenario-P&L tests.
  - `Global`: strong on session, pipeline, pricing-service, and analytical-verification workflows that consume the shared runtime analytics layer.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-658` / `CR-10`

- Reviewed:
  - hand-written files in `trellis/instruments/` excluding `_agent/`
  - `tests/test_instruments/*`
  - `tests/test_core/test_payoff_adapters.py`
  - `tests/test_engine/test_payoff_pricer.py`
  - broader interaction coverage in `tests/test_verification/test_analytical_pricing.py`, `tests/test_crossval/test_xv_bonds.py`, `tests/test_crossval/test_xv_options.py`, `tests/test_session.py`, and `tests/test_public_api_surface.py`
- Findings:
  - Fixed a cross-currency valuation bug in `trellis.instruments.fx.FXForwardPayoff.evaluate(...)`: the wrapped payoff now reprices against the configured foreign discount curve instead of silently reusing the domestic discount path.
  - The FX conversion path now clones the underlying `MarketState`, preserves the selected discount-curve name, and re-wraps the cloned state with the inner payoff runtime contract so proxy-enforced evaluation still holds after the currency-specific override.
  - Updated the instrument interaction test so a foreign-currency zero-coupon bond now asserts foreign discounting before spot FX conversion, and added a direct failure-path regression for a missing foreign discount curve binding.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_instruments tests/test_core/test_payoff_adapters.py tests/test_engine/test_payoff_pricer.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_instruments tests/test_verification/test_analytical_pricing.py tests/test_crossval/test_xv_bonds.py tests/test_crossval/test_xv_options.py tests/test_session.py tests/test_public_api_surface.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on checked-in bond, swap, cap, FX, mortgage, callable, barrier, and basket-style reference instruments, with improved direct coverage on FX payoff conversion semantics.
  - `Regional`: improved on the instrument -> payoff-pricer -> runtime-contract seam because the foreign-discount conversion path is now exercised through `price_payoff(...)` rather than only by direct payoff evaluation.
  - `Global`: strong on cross-validation, analytical verification, session, and public API flows that consume the same checked-in instrument layer.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-659` / `CR-11`

- Reviewed:
  - `trellis/models/trees/*`
  - `trellis/models/callable_bond_tree.py`
  - `trellis/models/bermudan_swaption_tree.py`
  - `trellis/models/zcb_option_tree.py`
  - `trellis/models/equity_option_tree.py`
  - `tests/test_models/test_trees/*`
  - `tests/test_models/test_callable_bond_tree.py`
  - `tests/test_models/test_bermudan_swaption_tree.py`
  - `tests/test_models/test_equity_option_tree.py`
  - route and workflow coverage in `tests/test_tasks/test_t02_bdt_callable.py`, `tests/test_tasks/test_t04_bermudan_swaption.py`, `tests/test_tasks/test_t07_american_put_3way.py`, `tests/test_agent/test_lattice_admissibility.py`, `tests/test_agent/test_rate_tree_generation.py`, `tests/test_agent/test_callable_bond.py`, `tests/test_agent/test_american_generation.py`, and `tests/test_agent/test_early_exercise_policy.py`
- Findings:
  - Fixed a Bermudan route/compiler mismatch in `trellis.models.bermudan_swaption_tree`: once settlement moves past the first exercise date, the contract compiler now drops expired exercise dates instead of anchoring the underlying swap to an already-dead start date.
  - Added a direct regional regression proving that a post-settlement Bermudan contract prices identically whether expired exercise dates remain on the spec or are removed explicitly.
  - No additional in-scope production-code findings remained after the broader tree, task, and generation regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_bermudan_swaption_tree.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_trees tests/test_models/test_callable_bond_tree.py tests/test_models/test_bermudan_swaption_tree.py tests/test_models/test_equity_option_tree.py tests/test_tasks/test_t02_bdt_callable.py tests/test_tasks/test_t04_bermudan_swaption.py tests/test_tasks/test_t07_american_put_3way.py tests/test_agent/test_lattice_admissibility.py tests/test_agent/test_rate_tree_generation.py tests/test_agent/test_callable_bond.py tests/test_agent/test_american_generation.py tests/test_agent/test_early_exercise_policy.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on lattice topology, rollback algebra, exercise-policy normalization, callable-bond helpers, Bermudan swaption helpers, and vanilla equity tree helpers.
  - `Regional`: improved on route-to-lattice exercise mapping because the new regression exercises the post-settlement Bermudan schedule filter all the way through contract compilation and tree pricing.
  - `Global`: strong on task-level callable-bond, Bermudan-swaption, and American-option workflows plus the generated-route admissibility checks.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - legacy `build_rate_lattice(...)` and `build_spot_lattice(...)` compatibility paths remain and still surface deprecation warnings in the tree suite
- Follow-on tickets:
  - carry the legacy lattice-builder cleanup into `QUA-676` / `CR-28`
- Disposition:
  - `Done`

#### `QUA-660` / `CR-12`

- Reviewed:
  - `trellis/models/monte_carlo/*`
  - `tests/test_models/test_monte_carlo/*`
  - workflow coverage in `tests/test_tasks/test_t07_american_put_3way.py`, `tests/test_tasks/test_t09_barrier.py`, `tests/test_tasks/test_t12_variance_reduction.py`, `tests/test_models/test_barrier.py`, `tests/test_agent/test_american_generation.py`, and `tests/test_agent/test_early_exercise_policy.py`
- Findings:
  - Fixed a shared early-exercise policy bug in the Monte Carlo stack: expiry is now treated as an implicit exercise date across `longstaff_schwartz`, `stochastic_mesh`, `tsitsiklis_van_roy`, and `primal_dual_mc`, so routes that only pass early exercise dates no longer drop the maturity payoff or understate the upper bound.
  - Added a direct interaction regression proving the implicit-expiry contract is consistent across all four early-exercise policy implementations.
  - No additional in-scope production-code findings remained after the broader Monte Carlo regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_monte_carlo/test_early_exercise.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_monte_carlo tests/test_tasks/test_t07_american_put_3way.py tests/test_tasks/test_t09_barrier.py tests/test_tasks/test_t12_variance_reduction.py tests/test_models/test_barrier.py tests/test_agent/test_american_generation.py tests/test_agent/test_early_exercise_policy.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on path-state storage contracts, discretization/scheme helpers, variance reduction, basket helpers, and individual early-exercise algorithms.
  - `Regional`: improved on engine -> payoff-state -> early-exercise-policy seams because the new regression now checks multiple policy implementations against the same implicit-expiry contract.
  - `Global`: strong on American-option, barrier-option, variance-reduction, and generated early-exercise workflow tests that consume the Monte Carlo substrate.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope beyond legacy tree-builder usage in generated American tests
- Follow-on tickets:
  - carry the remaining legacy tree-builder compatibility cleanup into `QUA-676` / `CR-28`
- Disposition:
  - `Done`

#### `QUA-661` / `CR-13`

- Reviewed:
  - `trellis/models/pde/*`
  - `trellis/models/equity_option_pde.py`
  - `trellis/models/transforms/*`
  - `trellis/models/grid_protocols.py`
  - `tests/test_models/test_equity_option_pde.py`
  - `tests/test_models/test_pde/*`
  - `tests/test_models/test_transforms/*`
  - verification/task coverage in `tests/test_tasks/test_t06_pde_convergence.py` and `tests/test_tasks/test_t10_callable_pde.py`
- Findings:
  - Fixed a PDE route-boundary bug: already-expired vanilla equity options now return intrinsic value without requiring a live discount curve or vol surface.
  - Fixed a numerical-hygiene bug in the COS transform helper so the zero-frequency coefficient no longer emits a divide-by-zero runtime warning under strict warning handling.
  - No additional in-scope production-code findings remained after the broader PDE/transform regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_equity_option_pde.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_transforms/test_transforms.py tests/test_models/test_equity_option_pde.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_equity_option_pde.py tests/test_models/test_pde tests/test_models/test_transforms tests/test_tasks/test_t06_pde_convergence.py tests/test_tasks/test_t10_callable_pde.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on theta-method, PSOR, tridiagonal solves, rate operators, grid assembly, and FFT/COS closed-form pricing helpers.
  - `Regional`: improved on PDE-helper route boundaries and transform coefficient generation because the new regressions exercise expired-option routing and strict-warning transform execution directly.
  - `Global`: strong on PDE convergence, callable-bond PDE versus tree cross-checks, and transform-vs-analytical pricing tests.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-662` / `CR-14`

- Reviewed:
  - `trellis/models/black.py`
  - `trellis/models/analytical/*`
  - `trellis/models/processes/*`
  - `trellis/models/vol_surface.py`
  - `tests/test_models/test_black.py`
  - `tests/test_models/test_vol_surface.py`
  - `tests/test_models/test_analytical_support.py`
  - `tests/test_models/test_fx_analytical_support.py`
  - `tests/test_models/test_processes/*`
  - broader verification/workflow coverage in `tests/test_models/test_barrier.py`, `tests/test_models/test_zcb_option_helpers.py`, `tests/test_tasks/test_t01_zcb_option.py`, `tests/test_tasks/test_t08_cev.py`, `tests/test_tasks/test_t09_barrier.py`, `tests/test_verification/test_analytical_pricing.py`, and `tests/test_agent/test_analytical_traces.py`
- Findings:
  - Hardened the shared volatility-surface contract: `FlatVol` now rejects negative vols, and `GridVolSurface` now rejects duplicate expiry/strike coordinates plus negative grid nodes.
  - Added targeted regressions so invalid vol-surface inputs fail at construction time instead of leaking into analytical, process, PDE, or calibration consumers.
  - No additional in-scope production-code findings remained after the broader analytical/process regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_vol_surface.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_black.py tests/test_models/test_vol_surface.py tests/test_models/test_analytical_support.py tests/test_models/test_fx_analytical_support.py tests/test_models/test_processes tests/test_models/test_barrier.py tests/test_models/test_zcb_option_helpers.py tests/test_tasks/test_t01_zcb_option.py tests/test_tasks/test_t08_cev.py tests/test_tasks/test_t09_barrier.py tests/test_verification/test_analytical_pricing.py tests/test_agent/test_analytical_traces.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on Black-style formulas, analytical support kernels, process drift/diffusion contracts, Heston runtime bindings, and vol-surface interpolation.
  - `Regional`: improved on process/analytical consumer boundaries because invalid vol inputs are now rejected at the shared surface rather than failing later in pricing routines.
  - `Global`: strong on analytical-verification suites, barrier/zcb task coverage, and generated analytical trace tests that consume the same foundational kernels.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-663` / `CR-15`

- Reviewed:
  - `trellis/models/calibration/*`
  - `trellis/models/hull_white_parameters.py`
  - `tests/test_models/test_calibration/*`
  - `tests/test_verification/test_calibration_replay.py`
  - `tests/test_verification/test_numerical_calibration.py`
  - `tests/test_agent/test_calibration_contract.py`
  - `tests/test_agent/test_recalibration.py`
  - `tests/test_models/test_trees/test_local_vol_lattice.py`
- Findings:
  - Fixed a calibration-diagnostics bug in the Heston workflow: ATM smile diagnostics and default seeding now use the carry-adjusted forward implied by spot, rate, dividend yield, and expiry, rather than incorrectly anchoring on spot.
  - Added a regression proving the Heston smile payload and warning text choose the forward-nearest strike when carry moves the forward away from spot.
  - No additional in-scope production-code findings remained after the calibration, replay, and verification regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_calibration/test_calibration.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_calibration tests/test_verification/test_calibration_replay.py tests/test_verification/test_numerical_calibration.py tests/test_agent/test_calibration_contract.py tests/test_agent/test_recalibration.py tests/test_models/test_trees/test_local_vol_lattice.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on solve-request contracts, replay artifacts, SABR/Heston/local-vol workflow builders, and rates/Hull-White calibration helpers.
  - `Regional`: improved on smile-surface -> diagnostics -> default-seed interactions because the new Heston regression now checks the carry-adjusted ATM path end to end.
  - `Global`: strong on calibration replay, numerical-verification, local-vol lattice, and recalibration workflow tests that consume the same supported calibration outputs.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - legacy tree-builder verification coverage still emits deprecation warnings
- Follow-on tickets:
  - carry legacy tree-builder warning cleanup into `QUA-676` / `CR-28`
- Disposition:
  - `Done`

#### `QUA-664` / `CR-16`

- Reviewed:
  - `trellis/models/cashflow_engine/*`
  - `trellis/models/credit_default_swap.py`
  - `trellis/models/contingent_cashflows.py`
  - `trellis/models/resolution/*`
  - `trellis/models/copulas/*`
  - `tests/test_models/test_cashflow/*`
  - `tests/test_models/test_credit_default_swap.py`
  - `tests/test_models/test_contingent_cashflows.py`
  - `tests/test_models/test_copulas/*`
  - broader coverage in `tests/test_crossval/test_xv_credit.py`, `tests/test_verification/test_credit_structured.py`, and `tests/test_instruments/test_nth_to_default_kernels.py`
- Findings:
  - Fixed a structured cashflow-engine state bug: an explicitly exhausted tranche (`balance=0.0`) now stays exhausted instead of being silently reset to full notional during construction.
  - Added a regression proving exhausted tranches do not re-enter the principal waterfall and instead leave incoming principal in residual cash when no balance remains.
  - No additional in-scope production-code findings remained after the broader structured-model regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_cashflow/test_cashflow.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_cashflow tests/test_models/test_credit_default_swap.py tests/test_models/test_contingent_cashflows.py tests/test_models/test_copulas tests/test_crossval/test_xv_credit.py tests/test_verification/test_credit_structured.py tests/test_instruments/test_nth_to_default_kernels.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on cashflow projection, prepayment, tranche waterfalling, CDS helpers, contingent cashflow kernels, and copula loss/default-generation helpers.
  - `Regional`: improved on stateful cashflow-engine assembly because the new regression now checks explicit tranche state flowing through the waterfall rather than only default-construction paths.
  - `Global`: strong on credit cross-validation, structured-product verification, and nth-to-default kernel tests that consume the same contingent-cashflow and copula substrate.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-665` / `CR-17`

- Reviewed:
  - `trellis/instruments/_agent/fxvanillaanalytical.py`
  - `trellis/instruments/_agent/fxvanillamontecarlo.py`
  - `trellis/instruments/_agent/quantooptionanalytical.py`
  - `trellis/instruments/_agent/quantooptionmontecarlo.py`
  - corresponding `_fresh/*` snapshots
  - `trellis/agent/knowledge/promotion.py`
  - `trellis/agent/knowledge/retrieval.py`
  - `tests/test_agent/test_promotion_candidates.py`
  - route-boundary and freshness coverage in `tests/test_agent/test_codegen_guardrails.py`, `tests/test_agent/test_route_registry.py`, `tests/test_agent/test_task_runtime.py`, `tests/test_agent/test_rate_tree_generation.py`, `tests/test_agent/test_callable_bond.py`, `tests/test_agent/test_american_generation.py`, `tests/test_agent/test_lattice_admissibility.py`, `tests/test_agent/test_calibration_contract.py`, `tests/test_agent/test_analytical_traces.py`, `tests/test_agent/test_route_scorer.py`, `tests/test_agent/test_route_learning.py`, `tests/test_agent/test_ir_retrieval.py`, `tests/test_agent/test_reference_oracles.py`, `tests/test_agent/test_family_contracts.py`, `tests/test_agent/test_family_lowering_ir.py`, `tests/test_agent/test_semantic_contracts.py`, `tests/test_agent/test_semantic_validation.py`, `tests/test_agent/test_market_binding.py`, `tests/test_agent/test_platform_requests.py`, and `tests/test_contracts/test_cassette_freshness.py`
- Findings:
  - Reviewed the four checked-in generated adapters currently flagged as stale against their validated `_fresh` counterparts and confirmed the checked-in routes are intentionally thinner helper-backed shells rather than silently divergent pricing logic.
  - No additional in-scope production-code defects remained after the route-boundary regression pass; the existing lifecycle warnings already surface the stale-vs-fresh inventory without forcing unsafe automatic replacement.
  - The current stale inventory remains: `fxvanillaanalytical`, `fxvanillamontecarlo`, `quantooptionanalytical`, and `quantooptionmontecarlo`.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_contracts/test_cassette_freshness.py tests/test_agent/test_codegen_guardrails.py tests/test_agent/test_rate_tree_generation.py tests/test_agent/test_callable_bond.py tests/test_agent/test_american_generation.py tests/test_agent/test_lattice_admissibility.py tests/test_agent/test_calibration_contract.py tests/test_agent/test_analytical_traces.py tests/test_agent/test_route_registry.py tests/test_agent/test_route_scorer.py tests/test_agent/test_route_learning.py tests/test_agent/test_ir_retrieval.py tests/test_agent/test_reference_oracles.py tests/test_agent/test_family_contracts.py tests/test_agent/test_family_lowering_ir.py tests/test_agent/test_semantic_contracts.py tests/test_agent/test_semantic_validation.py tests/test_agent/test_market_binding.py tests/test_agent/test_platform_requests.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on lifecycle detection, stale-warning formatting, and generated-adapter reuse metadata.
  - `Regional`: strong on checked-in adapter -> route registry -> task-runtime comparison seams for FX and quanto routes.
  - `Global`: strong on generation-plan, semantic, and route-boundary regressions that consume the same checked-in adapter shells during build-loop planning.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-666` / `CR-18`

- Reviewed:
  - `trellis/agent/semantic_contracts.py`
  - `trellis/agent/semantic_contract_compiler.py`
  - `trellis/agent/semantic_contract_validation.py`
  - `trellis/agent/semantic_concepts.py`
  - `trellis/agent/family_lowering_ir.py`
  - `tests/test_agent/test_semantic_contracts.py`
  - `tests/test_agent/test_family_contracts.py`
  - `tests/test_agent/test_family_lowering_ir.py`
  - `tests/test_agent/test_semantic_validation.py`
  - `tests/test_agent/test_dsl_integration.py`
  - `tests/test_agent/test_user_defined_products.py`
  - `tests/test_agent/test_blocker_planning.py`
- Findings:
  - Fixed a cross-layer semantic drift bug in `compile_semantic_contract(...)`: semantic contracts with no declared primitive route family and no semantically valid inferred route family no longer guess `analytical_black76` or pull `trellis.models.black` into `route_modules`.
  - Added a DSL integration regression for `range_accrual` proving the semantic compiler now stays on its explicit semantic target modules, emits no primitive route, and surfaces `missing_primitive_routes` instead of fabricating a vanilla-option lowering.
  - No additional in-scope semantic-contract or family-lowering defects remained after the broader semantic regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_dsl_integration.py -x -q -k range_accrual -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_semantic_contracts.py tests/test_agent/test_family_contracts.py tests/test_agent/test_family_lowering_ir.py tests/test_agent/test_semantic_validation.py tests/test_agent/test_dsl_integration.py tests/test_agent/test_user_defined_products.py tests/test_agent/test_blocker_planning.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on typed semantic validation, concept resolution, and family-IR construction.
  - `Regional`: improved on the semantic contract -> compiler -> DSL lowering seam through the new range-accrual regression.
  - `Global`: strong on the semantic DSL round-trip coverage that spans concept resolution, validation, compilation, blocker planning, and user-defined product blueprints.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - capture a checked route-family implementation for `range_accrual` in the remediation synthesis if the semantic trade-entry slice is promoted to executable routing
- Disposition:
  - `Done`

#### `QUA-667` / `CR-19`

- Reviewed:
  - `trellis/agent/platform_requests.py`
  - `trellis/agent/route_registry.py`
  - `trellis/agent/market_binding.py`
  - `trellis/agent/valuation_context.py`
  - `tests/test_agent/test_platform_requests.py`
  - `tests/test_agent/test_market_binding.py`
  - `tests/test_agent/test_route_registry.py`
  - `tests/test_agent/test_route_scorer.py`
  - `tests/test_agent/test_route_learning.py`
  - `tests/test_agent/test_ir_retrieval.py`
  - `tests/test_agent/test_reference_oracles.py`
  - `tests/test_agent/test_platform_loop.py`
- Findings:
  - Fixed a request-compilation determinism bug for semantic contracts with no primitive route: `_compile_semantic_request(...)` now clears the guessed primitive plan, blocker/new-primitive sidecars, and downstream route-authority metadata instead of reintroducing `analytical_black76` after the semantic compiler explicitly emitted `primitive_routes == ()`.
  - Added a request-layer regression for `range_accrual` proving the missing-route state now survives through the compiled request envelope, generation plan, and request metadata.
  - No additional in-scope routing or market-binding defects remained after the broader routing regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_platform_requests.py -x -q -k range_accrual -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_platform_requests.py tests/test_agent/test_market_binding.py tests/test_agent/test_route_registry.py tests/test_agent/test_route_scorer.py tests/test_agent/test_route_learning.py tests/test_agent/test_ir_retrieval.py tests/test_agent/test_reference_oracles.py tests/test_agent/test_platform_loop.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on required-data compilation, market-binding summaries, route-registry primitives, and route-scoring features.
  - `Regional`: improved on semantic blueprint -> compiled request -> generation-plan -> route-authority propagation through the new range-accrual regression.
  - `Global`: strong on platform-loop and request-compilation regressions that exercise the same routing metadata consumed by executor/build-loop flows.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-668` / `CR-20`

- Reviewed:
  - `trellis/agent/planner.py`
  - `trellis/agent/quant.py`
  - `trellis/agent/prompts.py`
  - `trellis/agent/codegen_guardrails.py`
  - `tests/test_agent/test_prompts.py`
  - `tests/test_agent/test_quant.py`
  - `tests/test_agent/test_codegen_guardrails.py`
  - `tests/test_agent/test_planner.py`
  - `tests/test_agent/test_build_loop.py`
- Findings:
  - Fixed a generation-prompt drift bug for route-less semantic requests: `_render_prompt_module_requirements(...)` now prefers compiler-selected inspected modules when no exact route-bound primitive plan exists, instead of falling back to generic family modules such as `trellis.models.black`.
  - Added a prompt regression for `range_accrual` proving the builder prompt now points at `trellis.models.range_accrual` and `trellis.models.contingent_cashflows` rather than the unrelated Black-76 helper surface.
  - No additional in-scope planner, quant-selection, or guardrail defects remained after the broader generation-stack regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_prompts.py -x -q -k route_less_semantic_request -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_prompts.py tests/test_agent/test_quant.py tests/test_agent/test_codegen_guardrails.py tests/test_agent/test_planner.py tests/test_agent/test_build_loop.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on prompt rendering, pricing-plan selection, route cards, and generation-plan assembly.
  - `Regional`: improved on semantic blueprint -> generation plan -> builder prompt module selection through the new route-less range-accrual regression.
  - `Global`: strong on build-loop tests that consume the same prompt and guardrail surfaces during end-to-end generation planning.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-669` / `CR-21`

- Reviewed:
  - `trellis/agent/validation_contract.py`
  - `trellis/agent/validation_bundles.py`
  - `trellis/agent/critic.py`
  - `trellis/agent/executor.py`
  - `tests/test_agent/test_validation_contract.py`
  - `tests/test_agent/test_validation_bundles.py`
  - `tests/test_agent/test_reference_oracles.py`
  - `tests/test_agent/test_executor.py`
  - `tests/test_agent/test_lite_review.py`
  - `tests/test_agent/test_critic.py`
  - `tests/test_agent/test_model_validator.py`
- Findings:
  - Fixed a validation-contract truthfulness bug: deterministic checks are now filtered against the compiled market-data surface, so route-less semantic contracts such as `range_accrual` no longer advertise vol-based checks when no vol surface is required.
  - Added a regression proving route-less semantic requests keep `route_id=None`, retain the lowering error, and omit `check_vol_sensitivity` / `check_vol_monotonicity` from the compiled validation contract.
  - No additional in-scope validation-pipeline defects remained after the broader executor/critic/model-validator regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_validation_contract.py -x -q -k route_less_semantic_request -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_validation_contract.py tests/test_agent/test_validation_bundles.py tests/test_agent/test_reference_oracles.py tests/test_agent/test_executor.py tests/test_agent/test_lite_review.py tests/test_agent/test_critic.py tests/test_agent/test_model_validator.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on validation bundle selection, deterministic harness requirements, critic check mapping, and model-validator review contracts.
  - `Regional`: improved on semantic blueprint -> validation contract -> critic/lite-review seams through the new route-less range-accrual regression.
  - `Global`: strong on executor and audit-path regressions that consume compiled validation contracts during real build/review flows.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-670` / `CR-22`

- Reviewed:
  - `trellis/agent/platform_traces.py`
  - `trellis/agent/checkpoints.py`
  - `trellis/agent/task_run_store.py`
  - `trellis/agent/task_diagnostics.py`
  - `trellis/agent/task_runtime.py`
  - `trellis/agent/evals.py`
  - `tests/test_agent/test_platform_traces.py`
  - `tests/test_agent/test_checkpoints.py`
  - `tests/test_agent/test_task_run_store.py`
  - `tests/test_agent/test_task_diagnostics.py`
  - broader runtime/request/validation consumers in `tests/test_agent/test_platform_loop.py`, `tests/test_agent/test_platform_requests.py`, and `tests/test_agent/test_validation_contract.py`
- Findings:
  - Fixed a trace-boundary truthfulness bug: `_generation_boundary_summary(...)` no longer falls back to `execution_plan.route_method` when semantic lowering emitted no primitive route, and instead preserves `route_id=None` / `route_family=None` for route-less semantic requests.
  - Fixed a checkpoint drift bug: the route-stage decision now prefers an actual lowering or route-authority id and stays `unknown` when no route was resolved, instead of reclassifying the high-level method as a concrete route.
  - Fixed task-run and diagnosis telemetry drift: platform-trace summaries and derived route observations no longer promote the platform action name `build_then_price` into a fake route id.
  - Added regional regressions proving route-less `range_accrual` requests stay truthful across persisted platform traces, checkpoint capture, task-run rollups, and diagnosis-packet fallback telemetry.
  - No additional in-scope task-runtime, eval, or replay defects remained after the broader runtime-facing regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_platform_traces.py -x -q -k route_less_semantic_requests_truthful -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_checkpoints.py -x -q -k route_stage_unknown_when_lowering_has_no_route -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_task_run_store.py -x -q -k fake_route_id -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_task_diagnostics.py -x -q -k platform_action_out_of_route_ids -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_platform_traces.py tests/test_agent/test_checkpoints.py tests/test_agent/test_task_run_store.py tests/test_agent/test_task_diagnostics.py tests/test_agent/test_platform_loop.py tests/test_agent/test_platform_requests.py tests/test_agent/test_validation_contract.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on trace serialization, checkpoint staging, task-run record assembly, and diagnosis-packet derivation.
  - `Regional`: improved on semantic blueprint -> trace summary -> checkpoint/task-run/diagnosis propagation through the new route-less range-accrual regressions.
  - `Global`: strong on platform-loop, request-compilation, and validation-contract suites that consume the same runtime metadata during real build and replay flows.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-671` / `CR-23`

- Reviewed:
  - `trellis/agent/knowledge/__init__.py`
  - `trellis/agent/knowledge/store.py`
  - `trellis/agent/knowledge/retrieval.py`
  - `trellis/agent/knowledge/promotion.py`
  - `trellis/agent/knowledge/decompose.py`
  - `trellis/agent/knowledge/reflect.py`
  - `tests/test_agent/test_knowledge_store.py`
  - `tests/test_agent/test_cookbooks.py`
  - `tests/test_agent/test_primitive_planning.py`
  - `tests/test_agent/test_quant.py`
  - `tests/test_agent/test_prompts.py`
  - downstream adapter-warning coverage in `tests/test_agent/test_promotion_candidates.py`
- Findings:
  - Fixed a promotion-safety bug in `resolve_adapter_lifecycle_records(...)`: persisted adapter lifecycle artifacts are loaded newest-first, and equal-status ties now keep the first-seen record instead of letting older artifacts overwrite newer review/adoption state.
  - Added a promotion regression proving the newest persisted review wins when multiple artifacts describe the same adapter lifecycle key.
  - Added a retrieval-surface regression proving prompt formatting uses the latest adapter lifecycle review and does not resurrect stale older reasons.
  - No additional in-scope retrieval, store-cache, or build-loop knowledge handoff defects remained after the broader knowledge-runtime regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_knowledge_store.py -x -q -k "newest_persisted_artifact or latest_adapter_lifecycle_review" -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_knowledge_store.py tests/test_agent/test_cookbooks.py tests/test_agent/test_primitive_planning.py tests/test_agent/test_quant.py tests/test_agent/test_prompts.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_promotion_candidates.py -x -q -k stale_adapter_warning -m "not integration"`
- Coverage assessment:
  - `Local`: strong on retrieval ranking, decomposition reuse, lesson promotion gates, and knowledge-cache invalidation.
  - `Regional`: improved on promotion artifact -> adapter lifecycle resolution -> retrieval prompt/trace summary propagation through the new newest-artifact regressions.
  - `Global`: strong on prompt and primitive-planning suites that consume shared knowledge payloads during real build and routing flows.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-672` / `CR-24`

- Reviewed:
  - `trellis/agent/knowledge/canonical/api_map.yaml`
  - `trellis/agent/knowledge/canonical/blockers.yaml`
  - `trellis/agent/knowledge/canonical/cookbooks.yaml`
  - `trellis/agent/knowledge/canonical/data_contracts.yaml`
  - `trellis/agent/knowledge/canonical/decompositions.yaml`
  - `trellis/agent/knowledge/canonical/features.yaml`
  - `trellis/agent/knowledge/canonical/method_requirements.yaml`
  - `trellis/agent/knowledge/canonical/routes.yaml`
  - `trellis/agent/knowledge/api_map.py`
  - `trellis/agent/knowledge/instructions.py`
  - `trellis/agent/knowledge/skills.py`
  - `tests/test_agent/test_api_map.py`
  - `tests/test_agent/test_knowledge_store.py`
  - `tests/test_agent/test_ir_retrieval.py`
  - `tests/test_agent/test_tools.py`
  - `tests/test_agent/test_prompts.py`
- Findings:
  - Fixed a schema-to-runtime drift bug in the API-map formatter: the canonical API map already defined `rate_style_swaption`, `jamshidian_zcb_option`, and `credit_curve` utilities, but `api_map.py` silently dropped them from prompt/tool output because `_UTILITY_ORDER` was stale.
  - Added a regression proving the formatter now surfaces all canonical utility families rather than only a truncated hard-coded subset.
  - No additional in-scope canonical-map, instruction-resolution, or skill-index defects remained after the broader schema-consumer regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_api_map.py -x -q -k canonical_utilities -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_api_map.py tests/test_agent/test_knowledge_store.py tests/test_agent/test_ir_retrieval.py tests/test_agent/test_tools.py tests/test_agent/test_prompts.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on canonical API-map parsing, import validity, and deterministic skill/instruction filtering helpers.
  - `Regional`: improved on canonical YAML -> formatter -> prompt/tool payload propagation through the new all-utilities regression.
  - `Global`: strong on prompt and tool suites that consume the API map as part of real routing and code-generation guidance surfaces.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-673` / `CR-25`

- Reviewed:
  - `trellis/agent/knowledge/promotion.py`
  - `trellis/agent/knowledge/autonomous.py`
  - `trellis/agent/platform_traces.py`
  - `tests/test_agent/test_trace_compaction.py`
  - `tests/test_agent/test_consolidation.py`
  - `tests/test_agent/test_platform_traces.py`
  - `tests/test_agent/test_knowledge_store.py`
  - sampled runtime artifacts in `trellis/agent/knowledge/lessons/entries/`, `trellis/agent/knowledge/traces/platform/`, `trellis/agent/knowledge/traces/analytical/`, `trellis/agent/knowledge/traces/checkpoints/`, and `trellis/agent/knowledge/traces/semantic_extensions/`
- Findings:
  - Fixed a retention blind spot in the knowledge hygiene loop: consolidation tier-2 was still counting only legacy flat `traces/*.yaml`, so the current platform runtime could accumulate hundreds of `platform/*.events.ndjson` sidecars without ever triggering compaction.
  - Extended `compact_traces(...)` so old platform event sidecars are compacted by inlining their full event history back into the summary YAML and removing the extra `.events.ndjson` file; this preserves replay/debug payloads while materially reducing trace-file bloat.
  - Added a regional regression that records a real platform trace, compacts it through the promotion hygiene layer, and then reloads it through the platform trace reader to prove retention stays truthful across module boundaries.
  - Sampled the lesson index inputs plus the platform, analytical, checkpoint, and semantic-extension trace directories; no additional index-integrity or promotion-state defect surfaced in-scope after the broader regression pass.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_trace_compaction.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_consolidation.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_platform_traces.py tests/test_agent/test_knowledge_store.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_agent/test_trace_compaction.py tests/test_agent/test_consolidation.py tests/test_agent/test_platform_traces.py tests/test_agent/test_knowledge_store.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: strong on lesson-payload normalization, index rebuild determinism, adapter lifecycle review state, and legacy flat-trace compaction.
  - `Regional`: improved on `autonomous._assess_consolidation_needs(...)` -> `promotion.compact_traces(...)` -> `platform_traces` reload behavior through the new platform-sidecar regression.
  - `Global`: adequate through the broader knowledge-store and platform-trace suites that consume the same persisted lesson and trace artifacts during build, reflection, and replay flows.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-674` / `CR-26`

- Reviewed:
  - `tests/conftest.py`
  - `tests/test_session.py`
  - `tests/test_pipeline.py`
  - `tests/test_contracts/test_pipeline_contracts.py`
  - `tests/test_crossval/*`
  - `tests/test_verification/*`
  - sampled legacy-marker usage in `tests/test_models/test_trees/test_lattice.py`, `tests/test_models/test_trees/test_lattice_performance_contract.py`, and `tests/test_models/test_monte_carlo/test_early_exercise.py`
- Findings:
  - Fixed a test-governance gap in the root pytest configuration: the suite already relied on `legacy_compat` as a semantic category, but the marker was never registered, and the major non-local strata (`crossval`, `verification`, `global_workflow`) were not first-class or filterable at collection time.
  - Added collection-time auto-marking so `tests/test_crossval/**`, `tests/test_verification/**`, and the shared public workflow suites (`tests/test_session.py`, `tests/test_pipeline.py`, `tests/test_contracts/test_pipeline_contracts.py`) can now be selected and audited explicitly.
  - Added architecture regressions proving the cross-validation, verification, and global-workflow strata are tagged as intended.
  - Tightened a regional/global interaction seam in `tests/test_pipeline.py`: the shared governed-runner delegation test now asserts `failure_outcome="pipeline_failed"` in addition to the success path, so the `Pipeline -> Session._run_governed_request(...)` contract is defended on both sides of the outcome mapping.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_session.py tests/test_pipeline.py tests/test_crossval/test_suite_markers.py tests/test_crossval/test_xv_bonds.py tests/test_verification/test_suite_markers.py tests/test_verification/test_greeks.py tests/test_contracts/test_pipeline_contracts.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: improved on test-governance surfaces because the suite’s architectural strata are now explicit rather than implicit directory conventions.
  - `Regional`: improved on the `Pipeline -> Session._run_governed_request(...)` seam through the new failure-outcome assertion.
  - `Global`: improved because cross-validation, verification, and workflow suites are now directly selectable for targeted review runs instead of only by broad path conventions.
- Docs / limitations mismatches:
  - none found in-scope
- Shim removal candidates:
  - legacy-compat tests are now explicitly discoverable and can be audited as a compatibility surface instead of remaining an untracked implicit bucket
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-675` / `CR-27`

- Reviewed:
  - `docs/developer/audit_and_observability.rst`
  - `docs/developer/task_and_eval_loops.rst`
  - `docs/developer/legacy_test_audit.md`
  - `docs/developer/index.rst`
  - `docs/quant/knowledge_maintenance.rst`
  - `LIMITATIONS.md`
  - checked code and tests behind the recent review slices in `trellis/agent/knowledge/promotion.py`, `trellis/agent/platform_traces.py`, `tests/conftest.py`, and `tests/test_pipeline.py`
- Findings:
  - Fixed a trace-retention documentation drift: the developer and quant docs previously described the split platform-trace layout as if the `.events.ndjson` sidecar always remained present, but the checked retention path can now compact old sidecars back into inline YAML.
  - Fixed a test-governance documentation gap: the official developer docs now describe the first-class `crossval`, `verification`, `global_workflow`, and `legacy_compat` strata introduced by the checked test architecture, including selector examples.
  - Added `legacy_test_audit` to the developer index so the compatibility-test policy is reachable from the official docs instead of existing as an unlinked note.
  - No additional doc-versus-code mismatch was found in-scope in `LIMITATIONS.md` during this pass.
- Tests checked:
  - docs-only ticket; no tests run
  - truthfulness validated against the checked code and test paths from `QUA-673` and `QUA-674`
- Coverage assessment:
  - `Local`: docs now match the concrete retention and marker behavior in the corresponding modules.
  - `Regional`: improved because the task/eval and audit docs now describe the same cross-module workflow/test strata used during review runs.
  - `Global`: improved because the developer index now surfaces the compatibility-test policy and the official docs no longer overstate the platform-trace storage shape.
- Docs / limitations mismatches:
  - resolved in-scope for trace retention and test-strata guidance
- Shim removal candidates:
  - none found in-scope
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-676` / `CR-28`

- Reviewed:
  - `tests/test_models/test_trees/test_lattice.py`
  - `tests/test_models/test_generalized_methods.py`
  - `tests/test_verification/test_numerical_calibration.py`
  - `tests/test_verification/test_literature_benchmarks.py`
  - `tests/test_models/test_trees/test_lattice_algebra.py`
  - `tests/test_models/test_trees/test_lattice_performance_contract.py`
  - `docs/developer/legacy_test_audit.md`
  - current tree-builder surfaces in `trellis/models/trees/lattice.py` and `trellis/models/trees/algebra.py`
- Findings:
  - Fixed a compatibility-boundary leak in the ordinary tree and verification suites: non-legacy tests were still building lattices through the deprecated `build_rate_lattice(...)` and `build_spot_lattice(...)` wrappers, which kept deprecation-only behavior on normal numerical and interaction paths.
  - Added `tests/lattice_builders.py` so ordinary test coverage now goes through the unified `build_lattice(...)` surface while preserving the same regional/global pricing seams those suites are meant to defend.
  - Added `tests/test_models/test_trees/test_lattice_wrapper_audit.py` to make the compatibility boundary executable: non-legacy lattice suites may not call deprecated builders, and any remaining deprecated-builder use in `tests/test_models/test_trees/test_lattice.py` must stay under `@pytest.mark.legacy_compat`.
  - Fixed a compatibility-policy doc drift in `docs/developer/legacy_test_audit.md`: removed a stale nonexistent legacy test entry and aligned the note with the checked legacy-only set, parity-oracle coverage, and the new wrapper-audit rule.
- Tests checked:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_trees/test_lattice_wrapper_audit.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_trees/test_lattice.py tests/test_models/test_trees/test_lattice_wrapper_audit.py tests/test_models/test_generalized_methods.py tests/test_verification/test_numerical_calibration.py tests/test_verification/test_literature_benchmarks.py -x -q -m "not integration"`
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest tests/test_models/test_trees/test_lattice_algebra.py tests/test_models/test_trees/test_lattice_performance_contract.py tests/test_models/test_trees/test_lattice_wrapper_audit.py -x -q -m "not integration"`
- Coverage assessment:
  - `Local`: improved on lattice-test construction because ordinary tests now exercise the current builder contract directly instead of relying on deprecated wrapper behavior.
  - `Regional`: improved on `tests -> tests/lattice_builders.py -> trellis.models.trees.build_lattice(...) -> lattice_backward_induction(...)` seams by moving existing verification suites onto the same unified route used by the modern tree algebra.
  - `Global`: improved because cross-suite numerical verification and literature benchmarks now validate the supported builder surface rather than silently pinning user-visible behavior to compatibility wrappers.
- Docs / limitations mismatches:
  - resolved in-scope for `docs/developer/legacy_test_audit.md`
- Shim removal candidates:
  - deprecated lattice builders remain only in explicit deprecation checks, parity oracles, and `legacy_compat` tests
  - no production shim removal landed in this slice
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

#### `QUA-677` / `CR-29`

- Reviewed:
  - closeout dossiers for `QUA-648` through `QUA-676`
  - the mirrored queue and exit criteria in this plan
  - `AGENTS.md`
  - `docs/developer/legacy_test_audit.md`
- Findings:
  - The highest-value recurring review themes were consistent across waves: authority drift at public/runtime boundaries, shared-state/default-selection correctness, missing regional/global interaction tests, and compatibility-only behavior leaking into ordinary coverage.
  - No unresolved `P0` or `P1` findings remain open from the completed review slices. High-severity issues discovered during the program were fixed in-scope rather than deferred into a second backlog.
  - The standing process needed one more explicit policy layer to keep the backlog from reforming: compatibility-only assertions must stay behind `legacy_compat`, ordinary review runs must use the named test strata (`crossval`, `verification`, `global_workflow`), and deprecations should gain audit tests that defend the boundary between legacy-only and ordinary behavior.
  - The reviewer instructions in `AGENTS.md` now encode those rules directly so future agents do not need to reconstruct them from the completed tickets.
- Tests checked:
  - synthesis ticket; no additional code-path tests beyond the targeted Wave 5 validation
  - relied on the completed child-ticket validation runs recorded above, especially `QUA-674` through `QUA-676`
- Coverage assessment:
  - `Local`: the review program now has explicit per-slice closeout artifacts and a concrete compatibility-policy rule set.
  - `Regional`: stronger because the standing workflow now requires reviewers to classify and defend cross-module seams rather than treating them as optional follow-up coverage.
  - `Global`: stronger because the repo now has an explicit operating policy for workflow, verification, and compatibility-test strata instead of relying on convention and memory.
- Docs / limitations mismatches:
  - resolved in-scope for review-policy guidance in `AGENTS.md` and compatibility guidance in `docs/developer/legacy_test_audit.md`
- Shim removal candidates:
  - compatibility-only coverage should remain isolated behind `legacy_compat` and explicit audit tests going forward
- Follow-on tickets:
  - none
- Disposition:
  - `Done`

### Program Closeout

#### `QUA-647`

- Reviewed:
  - all review slices `QUA-648` through `QUA-677`
  - the mirrored queue, closeout dossiers, and exit criteria in this plan
- Findings:
  - The code-review program is complete: every planned slice now has a closeout note, local/regional/global coverage has been assessed across the repo, compatibility-only paths have explicit keep-or-remove calls, and the review workflow is written back into repo instructions.
  - The most important repo-wide improvements landed during the program were:
    - stronger public/runtime boundary tests and session/pipeline/service interaction coverage
    - tighter market-state, curve, analytics, and calibration contract handling
    - broader semantic/platform/runtime route coverage across the agent stack
    - explicit test strata plus compatibility-boundary enforcement
    - doc and observability notes brought back in line with checked behavior
  - No unresolved high-severity review findings remain without either an in-scope fix or an explicit lower-priority disposition.
- Tests checked:
  - umbrella ticket; validation is the union of the child-ticket test runs recorded in this document
- Coverage assessment:
  - `Local`: repo-wide critical modules and contracts were reviewed with direct tests or direct dossier analysis.
  - `Regional`: major cross-module seams now have named review ownership and stronger interaction coverage.
  - `Global`: user-facing workflows, verification suites, and cross-validation surfaces are now part of the standing review model rather than ad hoc regression runs.
- Docs / limitations mismatches:
  - resolved or recorded in-scope during the child review slices
- Shim removal candidates:
  - compatibility-only behavior is now explicitly isolated instead of leaking through ordinary coverage
- Follow-on tickets:
  - `QUA-678` Trade parsing: restore structured schedule alias parity
  - `QUA-679` Desk review: make blocked pricing narratives truthful
- Disposition:
  - `Done`

### Post-Closeout Follow-Ons

| Ticket | Focus | Status |
| --- | --- | --- |
| `QUA-678` | Restore structured schedule alias parity on governed trade entry | Done |
| `QUA-679` | Make blocked `desk_review` pricing narratives truthful | Done |

## Standing Review Policy After Closeout

- Review code and tests together. Do not accept a slice as “reviewed” if the
  tests defending the same behavior were not inspected.
- Use the named strata during targeted review runs:
  - `crossval` for independent-engine or external-reference checks
  - `verification` for trusted analytical or literature benchmarks
  - `global_workflow` for user-visible or multi-layer runtime flows
  - `legacy_compat` only for deprecated or compatibility-only behavior
- Require a local / regional / global coverage call in every review closeout.
  Missing regional or global interaction tests are findings, not nice-to-haves.
- Keep ordinary tests on the current supported API surface. If a deprecated
  surface remains, isolate it behind `legacy_compat` or parity-oracle coverage
  and add an audit test when the boundary is easy to regress.
- Update the relevant docs and `LIMITATIONS.md` entries inside the same ticket
  when review work changes behavior, supported workflows, or compatibility
  policy.

## Ticket Template

Every review ticket should use this shape:

- Objective: what review question this ticket answers
- Why risky: why this slice deserves attention now
- Scope: exact files and tests in scope
- Interaction map: key cross-module, cross-class, and cross-function seams
- Non-goals: what the ticket will not audit
- Review checklist: correctness, edge cases, local/regional/global tests, docs, shims
- Validation: exact test commands to run or explicitly note as not run
- Coverage matrix: what exists at local, regional, and global scope
- Deliverables: findings, follow-ons, doc/limitation mismatches, shim notes, missing interaction tests

## Review Dossier Template

Use this template when starting a review ticket so the review begins with the
same shape every time.

```md
# Review Dossier

## Ticket
- Ticket:
- Slice:
- Reviewer:

## Objective
- What review question this ticket answers:

## Why risky
- Why this slice matters now:

## Scope
- Code files:
- Test files:
- Docs / `LIMITATIONS.md` entries:

## Contracts
- Public or internal contracts being defended:

## Interaction Map
- Inbound interactions:
- Outbound interactions:
- Critical seams:

## Coverage Matrix
| Behavior | Local | Regional | Global | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Validation Plan
- Targeted test commands:
- Broader regression commands:
- Benchmarks / replay artifacts, if any:

## Constraints
- Known limitations:
- Doc claims to verify:
- Shim / compatibility surfaces in scope:
```

## Recommended Closeout Template

Use this structure in the review handoff note:

- Reviewed:
- Findings:
- Tests checked:
- Coverage assessment:
- Docs / limitations mismatches:
- Shim removal candidates:
- Follow-on tickets:
- Disposition:

Recommended copy-paste form:

```md
- Reviewed:
- Findings:
- Tests checked:
- Coverage assessment:
- Docs / limitations mismatches:
- Shim removal candidates:
- Follow-on tickets:
- Disposition:
```

## Exit Criteria For The Program

This review program is complete only when:

1. every review slice above has a closeout note
2. all `P0` and `P1` findings have tracked follow-on tickets or fixes
3. docs and `LIMITATIONS.md` mismatches discovered during review are tracked
4. important behaviors have identified local, regional, and global regression
   surfaces
5. deprecated and compatibility-only paths have an explicit keep-or-remove call
6. the repo has a standing review process that prevents this backlog from
   reforming

## Recommended Next Step

Start with Wave 1, not with numerical methods.

Reason:

- the public/runtime/governed surfaces define what the rest of the codebase is
  allowed to promise
- the current repo is actively changing compatibility and governance paths
- findings there will change how later reviews classify "dead code" versus
  "still-supported path"
