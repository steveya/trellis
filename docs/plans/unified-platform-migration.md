# Unified Platform Migration Plan

## Purpose

This document defines the migration required to prepare Trellis for a governed
MCP implementation.

The immediate goal is not "add MCP." The immediate goal is to make the current
library safe to expose through an MCP surface by doing four things first:

1. remove the remaining legacy execution paths
2. make one compiled-request executor authoritative
3. introduce explicit runtime governance and provenance
4. separate research/candidate generation from approved execution

This plan is written as an implementation planning document. It is intended to
be decomposed into Linear issues later.

## Decision Summary

The migration direction is:

- keep the deterministic pricing substrate
- keep the request compiler and semantic layers
- keep the research build/validation/learning loop
- add a new governed platform core
- route all governed execution through that platform core
- treat legacy direct branches as adapters or dead code, not as parallel
  architectures

The migration is successful only when Trellis has one governed execution spine.

## Why This Plan Exists

Trellis already contains most of the hard pieces needed for the future system:

- canonical request compilation in `trellis.agent.platform_requests`
- typed semantic contracts in `trellis.agent.semantic_contracts`
- `ProductIR` in `trellis.agent.knowledge.schema`
- valuation-context and market-binding compilation
- strong build, validation, blocker, and reflection machinery for unsupported
  products
- trace and audit primitives

What the repo does not yet have is one authoritative runtime path.

Today several top-level entry points compile a request and then bypass the
compiled execution plan. That is the main architectural problem to fix before
any MCP server is built around the platform.

## Repo-Grounded Current State

This migration plan is grounded in the current codebase, not in a greenfield
server design.

### Runtime Paths That Exist Today

| Path | Current behavior | Keep / change / delete |
|------|------------------|------------------------|
| `trellis.ask(...)` / `ask_session()` | parses a `TermSheet`, compiles a request, then manually branches into known payoff match, build, or blocker handling | keep as a convenience API, but move execution to the unified executor |
| `Session.price(...)` | compiles a request for tracing, then calls direct pricing logic and may fall back to `_agent_price()` | keep the API, replace the runtime path |
| `Session.greeks(...)` | compiles a request for tracing, then executes direct analytics | keep the API, replace the runtime path |
| `Session.analyze(...)` | compiles a request for tracing, then executes direct measure logic | keep the API, replace the runtime path |
| `Pipeline.run()` | compiles a pipeline request, then scenario-loops through `Session.price(book)` | keep the API, replace the runtime path |
| `Session.price_payoff(...)` | builds `MarketState` and calls `trellis.engine.payoff_pricer.price_payoff()` | keep as the low-level deterministic substrate |
| `trellis.engine.pricer.price_instrument()` | deterministic direct instrument pricing, effectively a narrow bond-style adapter | keep as an internal route adapter, not as an architectural front door |
| `trellis.agent.executor.execute()` | older free-form agent loop | delete |
| `trellis.agent.executor.build_payoff()` | lower-level build pipeline for generating payoff code | keep, but re-scope as research/candidate-generation infrastructure |
| `trellis.agent.knowledge.build_with_knowledge()` | decompose, gap-check, retrieve, build/reuse, validate, reflect | keep, but re-scope as research/candidate-generation infrastructure |
| task runtime comparison flow | builds one artifact per target method and cross-validates them | keep, but eventually route through the same governed execution envelope |

### Request Actions That Already Exist

The current `ExecutionPlan.action` space already includes more than the first
pass migration summary captured:

- `price_book`
- `price_existing_instrument`
- `compute_greeks`
- `analyze_existing_instrument`
- `price_existing_payoff`
- `build_then_price`
- `compile_only`
- `block`
- `compare_methods`

The unified executor must treat all of these as first-class actions. The plan
must not pretend the action space is smaller than the code says it is.

### Structural Strengths To Preserve

- `compile_platform_request(...)` already normalizes request intent into one
  compiled envelope.
- `draft_semantic_contract(...)` plus `ProductIR` already give the library a
  typed identity layer for future trade parsing and model matching.
- `price_payoff(...)` already gives the runtime a clean deterministic pricing
  substrate.
- the knowledge/build loop already contains serious guardrails:
  decomposition, gap checks, route selection, reuse, validation bundles,
  oracles, critic/arbiter, reflection, and promotion pipelines.

### Structural Problems To Remove

- no authoritative `execute_compiled_request(...)` exists today
- `ask_session()` re-implements route selection manually after compilation
- `Session.price()` compiles for observability but does not execute from the
  compiled plan
- `Session._agent_price()` still points to the old free-form loop
- `Pipeline.run()` has a compile step that is mostly informational
- governed runs cannot currently express run mode, provider bindings, or policy
  bundle identity
- governed market resolution can still silently fall back to mock data
- current model audit can produce `auto_approved`, which is incompatible with a
  serious governed lifecycle
- route promotion and model approval are conceptually mixed today

## Migration Objective

Before the MCP implementation begins, Trellis should satisfy all of the
following:

1. every governed request executes through one compiled-request dispatcher
2. the dispatcher receives an explicit runtime context
3. governed runtime context includes run mode, provider bindings, and policy
   bundle identity
4. governed market resolution never silently substitutes mock data
5. every governed run returns a stable result envelope with provenance
6. approved-model execution is separated from research/candidate generation
7. public library APIs are thin projections over the platform core

## Non-Goals

This migration does not attempt to:

- redesign pricing math or stochastic engines
- replace the knowledge system
- remove mock data from notebooks, quickstarts, or explicit sandbox workflows
- build a distributed production registry backend
- make every current research artifact remotely invocable
- solve every open limitation in market coverage or numerical methods

## Desired Codebase Shape Before MCP

The codebase should look conceptually like this before MCP work begins:

```text
trellis/
  core/                 deterministic runtime protocols and value types
  curves/               deterministic market objects
  models/               deterministic numerical engines
  instruments/          checked-in instrument/payoff implementations
  data/                 provider adapters and MarketSnapshot compilation only
  platform/             governed runtime and orchestration core
    requests.py         PlatformRequest and compiled-request shims
    executor.py         authoritative compiled-request dispatcher
    context.py          RunMode / ExecutionContext / ProviderBindings
    policies.py         policy bundle records and enforcement
    results.py          ExecutionResult / provenance / output projections
    providers.py        provider registry and provider records
    models.py           model registry and lifecycle records
    runs.py             canonical run ledger and audit packaging
    services/           parse / match / validate / execute / audit services
  agent/                research, candidate generation, learning
    knowledge/
    prompts.py
    executor.py         build pipeline, not governed runtime dispatcher
  mcp/                  not implemented yet
```

### Ownership Boundary

After migration, the ownership split should be:

- `trellis.platform` owns governed request execution, runtime policy, provider
  binding, approved model selection, provenance, audit packaging, and stable
  result envelopes
- `trellis.agent` owns research workflows, candidate generation, validation
  enrichment, reflection, route discovery, and knowledge maintenance
- `trellis.core`, `trellis.models`, `trellis.curves`, and
  `trellis.instruments` remain the deterministic pricing substrate
- `trellis.data` owns provider adapters and snapshot compilation, not policy
  or fallback decisions

### Migration Constraint

Do not start by physically moving every current module.

The first implementation tranche should establish behavioral ownership through
new platform modules and compatibility re-exports. Physical relocation can come
later after runtime parity is proven.

## Target Runtime Contract

The core runtime contract after migration should be:

```python
execute_compiled_request(
    compiled_request: CompiledPlatformRequest,
    *,
    execution_context: ExecutionContext,
) -> ExecutionResult
```

### ExecutionContext

`ExecutionContext` should carry governed runtime state, not user intent.

The first governed-runtime tranche now lives in `trellis.platform.context`.
Current convenience surfaces normalize into this record through
`Session.to_execution_context()` and `Pipeline.to_execution_context()` while
leaving request compilation intent-focused.

Minimum fields:

- `session_id`
- `run_mode`
- `provider_bindings`
- `policy_bundle_id`
- `allow_mock_data`
- `require_provider_disclosure`
- `default_output_mode`
- `default_audit_mode`
- `requested_persistence`
- `requested_snapshot_policy`

### RunMode

At minimum:

- `sandbox`
- `research`
- `production`

### ProviderBindings

Typed binding slots should exist for at least:

- `market_data.primary`
- `market_data.fallback`
- `pricing_engine.primary`
- `model_store.primary`
- `validation_engine.primary`

The concrete record model uses:

- `ProviderBinding` for one stable provider id
- `ProviderBindingSet` for primary/fallback slots within one provider family
- `ProviderBindings` for the grouped market-data, pricing-engine, model-store,
  and validation-engine bindings carried by `ExecutionContext`

### ExecutionResult

Internal execution should always return one platform-level result envelope.

Minimum fields:

- `run_id`
- `request_id`
- `status`
- `action`
- `output_mode`
- `result_payload`
- `warnings`
- `provenance`
- `artifacts`
- `audit_summary`
- `trace_path`
- `policy_outcome`

### Provenance

Every governed result should carry:

- run id
- request id
- action
- valuation timestamp
- provider binding summary
- market snapshot id
- model id and version when applicable
- engine id and version
- route family and method family

## Migration Principles

1. replace alternate execution graphs with executor-owned route handlers
2. keep deterministic pricing capability, remove parallel runtime semantics
3. introduce governance before transport
4. preserve public APIs through projections and compatibility shims
5. keep research build flows, but remove their role as implicit production
   execution paths
6. separate route promotion from model lifecycle
7. prefer additive refactors before mass moves or renames

## Linear Ticket Mirror

These tables mirror the current Linear migration tickets and their
implementation order.

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This plan file is the repo-local mirror for subsequent agents.
- Implement the earliest ticket in the ordered queue below whose status is not
  `Done` and whose blockers are already satisfied.
- Skip tickets whose table row is `Done`.
- Workstream tickets are tracking rows. Do not pick them up before their
  earlier child rows unless the row has no remaining child slices.
- After a ticket is closed in Linear, update the corresponding row in this
  table in the same implementation closeout.

Status mirror last synced: `2026-04-04`

### Migration Workstream Tickets

| Ticket | Status |
| --- | --- |
| `QUA-551` Platform migration: unified governed execution spine | Done |
| `QUA-552` Platform executor: authoritative compiled-request dispatcher and result envelope | Done |
| `QUA-553` Platform governance: ExecutionContext, RunMode, and PolicyBundle enforcement | Done |
| `QUA-555` Run ledger: canonical provenance and audit packaging for governed execution | Done |
| `QUA-556` Model registry: lifecycle boundary and candidate-vs-approved execution | Done |

### Ordered Migration Implementation Queue

| Ticket | Status |
| --- | --- |
| `QUA-568` Governed runtime: RunMode, ProviderBindings, and ExecutionContext records | Done |
| `QUA-569` Model registry: record schema, local store, and lifecycle transitions | Done |
| `QUA-570` Run ledger: canonical RunRecord and artifact persistence surface | Done |
| `QUA-554` Provider registry: governed snapshot resolution and no silent mock fallback | Done |
| `QUA-573` Governed runtime: PolicyBundle evaluation and execution guards | Done |
| `QUA-574` Model execution gate: approved-model selection and lifecycle enforcement | Done |
| `QUA-575` Audit bundle: governed provenance, inputs, outputs, and policy outcomes | Done |
| `QUA-576` Platform executor: ExecutionResult envelope and action dispatcher skeleton | Done |
| `QUA-579` Platform executor: route adapters for existing compiled actions | Done |
| `QUA-557` API cutover: route ask, Session, analytics, and Pipeline through platform executor | Done |
| `QUA-558` Legacy runtime: remove free-form executor and trace-only execution branches | Done |

### Post-Migration Cleanup Queue

| Ticket | Status |
| --- | --- |
| `QUA-582` Platform cleanup: simplify governed error handling and remove executor model literal | Done |
| `QUA-583` Knowledge lessons: deduplicate repeated market-data candidate entries | Done |
| `QUA-584` Governed runtime cleanup: shared Session request runner and transition-path deletion | Done |
| `QUA-585` Knowledge retrieval: index supersedes and reduce cold-path lesson hydration | Done |
| `QUA-586` Platform traces: summary YAML plus append-only event log compatibility | Done |

## Detailed Phase Plan

## Phase 0: Establish `trellis.platform` As A Behavioral Boundary

### Objective

Create the destination platform boundary without changing runtime behavior yet.

### Deliverables

- new `trellis/platform/` package
- compatibility modules and import shims
- explicit developer documentation for the ownership split

### Specific Work

- add:
  - `trellis/platform/__init__.py`
  - `trellis/platform/requests.py`
  - `trellis/platform/context.py`
  - `trellis/platform/results.py`
  - `trellis/platform/executor.py`
  - `trellis/platform/providers.py`
  - `trellis/platform/models.py`
  - `trellis/platform/runs.py`
- re-export current request compiler symbols from
  `trellis.agent.platform_requests`
- re-export current trace helpers into the new platform namespace
- keep all call sites unchanged in this phase

### Files Likely To Change

- new `trellis/platform/*`
- `docs/developer/overview.rst`
- `docs/developer/hosting_and_configuration.rst`

### Acceptance Criteria

- the new package boundary exists and imports cleanly
- no current runtime behavior changes
- docs describe the new ownership boundary explicitly

## Phase 1: Build The Authoritative Compiled-Request Executor

### Objective

Make one dispatcher authoritative for all current compiled request actions.

### Deliverables

- `execute_compiled_request(...)`
- route-handler table keyed by `ExecutionPlan.action`
- first version of `ExecutionResult`

### Required Action Coverage

The dispatcher must handle:

- `price_book`
- `price_existing_instrument`
- `compute_greeks`
- `analyze_existing_instrument`
- `price_existing_payoff`
- `build_then_price`
- `compile_only`
- `block`
- `compare_methods`

`compare_methods` may initially delegate to a task-runtime adapter, but it must
still execute under the same result envelope and provenance model.

### Specific Work

- implement route handlers in `trellis/platform/executor.py`
- convert `ask_session()` to:
  - compile request
  - call executor
  - project `ExecutionResult` into `AskResult`
- convert `Session.price()` to:
  - compile request
  - call executor
  - project result into the historical return shape
- convert `Session.greeks()` and `Session.analyze()` similarly
- convert `Pipeline.run()` to call the executor directly instead of looping
  through a separate runtime path

The first governed executor substrate now exists as:

- `trellis.platform.results.ExecutionResult`
- `trellis.platform.executor.execute_compiled_request(...)`
- a default dispatcher table covering the full current compiled action space

At this stage, `compile_only` and `block` are concretely shaped outcomes and
the remaining action families intentionally return structured blocked envelopes
until the route-adapter ticket lands.

The next executor slice now wires thin adapters for:

- `price_book`
- `price_existing_instrument`
- `compute_greeks`
- `analyze_existing_instrument`
- `price_existing_payoff`

`build_then_price` and `compare_methods` intentionally remain structured
pending envelopes until the candidate-generation and comparison paths are
cut over.

### Adapter Strategy

Legacy deterministic logic should survive as route adapters inside the
executor:

- `price_instrument()` becomes the route adapter for the narrow direct
  instrument path
- `price_payoff()` remains the route adapter for payoff execution
- current analytics logic remains as executor-owned adapters until it can be
  reorganized cleanly

### Files Likely To Change

- `trellis/platform/executor.py`
- `trellis/platform/results.py`
- `trellis/agent/ask.py`
- `trellis/session.py`
- `trellis/pipeline.py`
- `trellis/__init__.py`

### Acceptance Criteria

- there is exactly one governed runtime dispatcher
- current public APIs still work
- compiled request execution is no longer "for traces only"

## Phase 2: Introduce Explicit Governance And Policy Enforcement

### Objective

Separate request intent from governed runtime state.

### Deliverables

- `RunMode`
- `ProviderBindings`
- `ExecutionContext`
- `PolicyBundle`
- pre-execution policy enforcement

### Specific Work

- define `ExecutionContext` in `trellis/platform/context.py`
- define executable policy rules in `trellis/platform/policies.py`
- add a policy enforcement step before runtime execution
- keep `PlatformRequest` focused on intent
- stop treating `data_source` string choices as the primary control surface for
  governed execution

### Important Compatibility Rule

`Session(data_source="mock")` and similar convenience patterns may still exist
for sandbox/notebook workflows, but they must map into explicit sandbox
contexts. They must not remain the control plane for governed execution.

### Files Likely To Change

- `trellis/platform/context.py`
- `trellis/platform/policies.py`
- `trellis/session.py`
- `trellis/agent/platform_requests.py`

### Acceptance Criteria

- governed execution requires an explicit execution context
- production and research policy checks happen before pricing begins
- run mode and provider policy show up in provenance and audit

## Phase 3: Provider Registry And Governed Market Resolution

### Objective

Replace convenience-driven snapshot resolution with explicit provider-bound
resolution for governed runs.

### Deliverables

- `ProviderRegistry`
- provider records with stable ids
- governed snapshot resolution path
- explicit snapshot identity

### Specific Work

- introduce provider registry interfaces in `trellis/platform/providers.py`
- adapt current data providers behind provider records instead of only `source`
  strings
- keep convenience `resolve_market_snapshot(source=...)` for non-governed flows
- add a governed resolver that requires provider binding ids
- remove silent fallback to mock data in governed modes
- allow mock data only when:
  - mock provider is explicitly bound
  - run mode/policy allows it

### Snapshot Identity

Governed runs need a durable snapshot handle.

Work here should add:

- explicit `snapshot_id` generation
- storage/retrieval hooks for snapshot bundles
- provenance wiring from `MarketSnapshot` into `ExecutionResult`

Current `MarketSnapshot` provenance fields are not enough by themselves.

### Files Likely To Change

- `trellis/platform/providers.py`
- `trellis/platform/runs.py`
- `trellis/data/resolver.py`
- `trellis/data/base.py`
- `trellis/data/schema.py`
- `trellis/agent/valuation_context.py`

### Acceptance Criteria

- governed resolution never silently falls back to mock data
- governed runs use provider ids, not only human source labels
- every governed run reports a snapshot id

## Phase 4: Canonical Run Ledger And Audit Packaging

### Objective

Turn traces and audits into one canonical run-ledger system.

### Deliverables

- canonical `RunRecord`
- audit package builder
- canonical storage root for governed artifacts
- compatibility projections for existing trace/task views

### Specific Work

Unify or layer the following around one canonical runtime record:

- `trellis.agent.platform_traces`
- `trellis.agent.model_audit`
- `trellis.agent.task_run_store`

The canonical run ledger should capture:

- request and action
- execution context
- parsed/semantic contract summary
- model selection decision
- provider bindings
- market snapshot identity
- engine/model provenance
- validation summary
- outputs
- warnings
- logs and linked artifacts

The first governed substrate now lives in `trellis.platform.runs` with:

- `RunRecord` as the canonical persisted run envelope
- `ArtifactReference` for durable links to traces, audits, and task-run files
- `RunLedgerStore` as the local-first persistence layer under `.trellis_state/runs`

The canonical governed review package now also lives in `trellis.platform.audits`
as `RunAuditBundle`, with stable sections for:

- `run`
- `inputs`
- `execution`
- `outputs`
- `diagnostics`
- `artifacts`

During migration, existing trace and task-run artifacts remain valid, but they
should be linked from the run ledger rather than treated as a separate source
of truth.

### Storage Direction

Introduce a configurable runtime state root, for example:

```text
.trellis_state/
  runs/
  models/
  snapshots/
  providers/
  policies/
  sessions/
```

Task-runner and knowledge traces may continue to exist during migration, but
the governed runtime must have one source of truth for run history.

### Files Likely To Change

- `trellis/platform/runs.py`
- `trellis/agent/platform_traces.py`
- `trellis/agent/model_audit.py`
- `trellis/agent/task_run_store.py`

### Acceptance Criteria

- every governed run writes one canonical run record
- audit retrieval can be reconstructed from the canonical record plus artifacts
- task/build diagnostics can be projected from the canonical record

## Phase 5: Model Registry, Typed Matching, And Lifecycle Separation

### Objective

Separate model lifecycle from knowledge/route promotion and make model
selection deterministic and typed.

### Deliverables

- `ModelRecord`
- `ModelVersionRecord`
- model registry interface
- model lifecycle:
  - `draft`
  - `validated`
  - `approved`
- `deprecated`

The first governed substrate now lives in `trellis.platform.models` with a
local-first `ModelRegistryStore`. Research build audit remains in
`trellis.agent.model_audit`, but successful audits no longer imply governed
approval by themselves.

### Specific Work

- create `trellis/platform/models.py`
- remove `auto_approved` semantics from model audit
- treat successful generated artifacts as draft or validated model versions,
  not approved production models
- keep route promotion and route adoption as a separate library-maintenance
  workflow
- define the typed model-match basis that later powers `trellis.model.match`

### Minimum Typed Match Basis

At minimum match on:

- semantic id
- semantic version
- product family
- instrument class
- payoff family
- exercise style
- underlier structure
- payout/reporting currency
- required market-data capability set
- supported method family
- lifecycle status

Do not use:

- lesson overlap
- similar-product retrieval score
- route-candidate confidence
- fuzzy text matching as a production reuse criterion

### Files Likely To Change

- `trellis/platform/models.py`
- `trellis/agent/model_audit.py`
- `scripts/approve_model.py`
- documentation for lifecycle vocabulary

### Acceptance Criteria

- approved-model execution no longer depends on route-promotion artifacts
- successful candidate generation does not create production eligibility
- typed matching rules are documented and testable

## Phase 6: Candidate Generation Boundary And Research/Production Split

### Objective

Keep the research build system, but re-scope it so it feeds candidates into the
model registry rather than acting as a shadow production executor.

### Deliverables

- explicit candidate-generation service boundary
- lifecycle handoff from research build output to model registry draft entries
- clear separation between:
  - route promotion
  - candidate generation
  - approved execution

### Specific Work

- wrap `build_with_knowledge()` and `build_payoff()` as candidate-generation
  services
- persist candidate artifacts with:
  - semantic contract summary
  - implementation path or code artifact
  - methodology summary
  - validation plan / validation result references
- keep the knowledge system free to use similarity, lessons, and route learning
  for build-time guidance
- keep those signals out of approved model matching

### Files Likely To Change

- `trellis/platform/services/`
- `trellis/agent/knowledge/autonomous.py`
- `trellis/agent/executor.py`

### Acceptance Criteria

- research build output lands as a draft candidate, not as an executable
  approved model by default
- the knowledge system remains useful without defining production selection

## Phase 7: Public API Cutover And Legacy Path Deletion

### Objective

Make all public library entry points projections over the platform core and
delete dead runtime branches.

### Deliverables

- `Session`, `Pipeline`, and top-level wrappers updated
- deprecated legacy paths removed
- docs updated to reflect the new runtime contract

### Specific Work

- keep `Session` and `Pipeline` as convenience shells
- remove their independent runtime semantics
- demote `price_instrument()` to an internal deterministic route adapter
- delete:
  - `Session._agent_price()`
  - `trellis.agent.executor.execute()`
  - compile-for-trace-only branches
  - governed reliance on `data_source` string behavior

### Files Likely To Change

- `trellis/session.py`
- `trellis/pipeline.py`
- `trellis/__init__.py`
- `trellis/agent/executor.py`

### Acceptance Criteria

- public convenience APIs still work
- they no longer define runtime behavior independently
- old free-form execution is gone

## Phase 8: Documentation And Cleanup Hardening

### Objective

Ensure the docs match the final runtime architecture and there are no shadow
paths left in the codebase.

### Deliverables

- updated developer docs
- updated user docs where behavior changed
- explicit migration notes for callers

### Specific Work

- update developer docs for:
  - governed execution model
  - provider binding and run mode
  - audit and observability
  - platform vs research ownership
- update user docs for:
  - convenience vs governed workflows
  - mock-data behavior
  - provenance fields
- remove dead code and dead docs references

### Acceptance Criteria

- docs accurately describe the code
- no alternate governed runtime graphs remain

## Testing And Validation Strategy

The migration should be gated by explicit parity and regression coverage.

### Required Test Layers

- request compiler tests:
  - action selection remains stable
  - requests stay serializable
- executor parity tests:
  - ask
  - direct price
  - greeks
  - analytics
  - payoff pricing
  - book pricing
  - build-then-price
  - compare methods
- governance tests:
  - production blocks draft/unapproved models
  - production blocks implicit mock data
  - research can use explicit mock bindings when allowed
- provenance tests:
  - every governed run emits run id, snapshot id, provider summary, model or
    engine provenance
- compatibility projection tests:
  - `AskResult`
  - `PricingResult`
  - `BookResult`

### Suggested Test Files

- `tests/test_platform/test_executor.py`
- `tests/test_platform/test_execution_context.py`
- `tests/test_platform/test_provider_registry.py`
- `tests/test_platform/test_run_ledger.py`
- `tests/test_platform/test_model_registry.py`
- `tests/test_platform/test_public_api_projections.py`

## Risks And Mitigations

### Risk: Moving files too early

Mitigation:

- establish behavioral ownership first
- use wrappers and re-exports before physical relocation

### Risk: Breaking notebook and sample workflows

Mitigation:

- preserve convenience APIs
- keep explicit sandbox defaults
- document the difference between convenience and governed execution

### Risk: Mixing research build logic with approved execution

Mitigation:

- create model lifecycle boundary before MCP work
- keep `trellis.agent` as research/candidate infrastructure

### Risk: Audit fragmentation continues

Mitigation:

- make one canonical run record mandatory before MCP
- project existing task/build summaries from the canonical record

### Risk: `compare_methods` becomes a forgotten edge path

Mitigation:

- explicitly support it in the executor contract and test plan
- do not leave it as a hidden side path in task runtime only

## Exit Criteria Before MCP

Do not start the MCP implementation until all of the following are true:

1. `execute_compiled_request(...)` exists and is authoritative for governed
   library entry points
2. `ExecutionContext` and `PolicyBundle` exist and are enforced
3. governed market resolution cannot silently fall back to mock data
4. every governed run produces a canonical run record with stable provenance
5. model lifecycle is separated from route/knowledge promotion
6. research build output lands as draft/validated candidate state, not
   production-approved state
7. `Session`, `Pipeline`, and top-level wrappers are projections over the
   platform core
8. `Session._agent_price()` and `trellis.agent.executor.execute()` are deleted

## Ticket Decomposition Guidance

When this plan is turned into Linear issues, the early tickets should follow
this order:

1. establish `trellis.platform` package and re-export shims
2. add `ExecutionResult` and `execute_compiled_request(...)`
3. cut over `ask_session()`
4. cut over `Session.price()`
5. cut over `Session.greeks()` and `Session.analyze()`
6. cut over `Pipeline.run()`
7. introduce `ExecutionContext`, `RunMode`, and `PolicyBundle`
8. add provider registry and governed snapshot resolution
9. add canonical run ledger and audit packaging
10. add model registry and lifecycle separation
11. wrap candidate generation as research-only registry input
12. delete legacy branches and update docs

That ordering minimizes the time spent with two overlapping runtimes while
still preserving safe compatibility checkpoints.
