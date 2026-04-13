# Trellis MCP Implementation Plan

## Purpose

This document describes the implementation plan for turning Trellis into an MCP
server after the unified platform migration is complete.

This plan assumes the preconditions in the
[Unified Platform Migration Plan](./done__unified-platform-migration.md) have been
satisfied. It is not a substitute for that migration.

The right mental model is:

- Trellis MCP is a governed orchestration and registry server
- deterministic pricing remains in pricing engines and checked-in route code
- candidate generation remains part of the research layer
- the MCP layer exposes tools, resources, prompts, and auditability over the
  governed platform core

## Product Interpretation

The product spec is directionally right, but in the current repo it must be
implemented with these repo-specific constraints:

- the semantic-contract stack is the canonical typed trade identity
- the request compiler and unified executor are the execution backbone
- `TermSheet` parsing is a compatibility surface, not the future canonical MCP
  trade contract
- the knowledge system is valuable for candidate generation, but its lesson and
  similarity machinery must not become the model-selection contract for
  governed pricing
- the current audit and validation machinery should be reused only after it is
  normalized behind the new governed run ledger

## Hard Product Rules

These are the non-negotiable product rules the implementation must enforce:

1. no silent fallback to mock data in governed production flows
2. no semantic "similar enough" model reuse without typed matching
3. no auto-promotion of generated candidates into production-approved status
4. every governed pricing run gets a run id, audit trail, data snapshot id,
   model version, and engine version
5. every governed output can be rendered in concise, structured, or audit mode

## Linear Ticket Mirror

These tables mirror the current Linear MCP tickets and their intended
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
- Keep the first `price.trade` milestone narrow: approved-model-only governed
  execution. Do not widen it to candidate generation in the MVP slice.

Status mirror last synced: `2026-04-10`

### MCP Workstream Tickets

| Ticket | Status |
| --- | --- |
| `QUA-559` MCP server: governed orchestration surface for Trellis | Done |
| `QUA-560` MCP foundation: state root, bootstrap config, and durable platform records | Done |
| `QUA-561` MCP session/provider tools: governed context and binding control | Done |
| `QUA-562` Semantic trade contract: typed parse and deterministic model matching services | Done |
| `QUA-567` MCP pricing tool: governed price.trade, run status, and audit retrieval | Done |

### Ordered MCP MVP Queue

| Ticket | Status |
| --- | --- |
| `QUA-571` MCP foundation: state root, config loader, and service bootstrap | Done |
| `QUA-572` Semantic trade.parse: typed contract normalization and missing-field reporting | Done |
| `QUA-577` MCP session/provider tools: context, run mode, list, and configure endpoints | Done |
| `QUA-578` Model match: deterministic approved-model selection and explain-match surface | Done |
| `QUA-580` MCP run APIs: run.get and run.get_audit over canonical ledger | Done |
| `QUA-581` MCP price.trade MVP: approved-model-only governed execution | Done |

### Ordered MCP Follow-On Queue

| Ticket | Status |
| --- | --- |
| `QUA-563` MCP lifecycle tools: candidate generation, validation, and approval controls | Done |
| `QUA-564` MCP model store tools: persistence, version history, diff, and reproducibility bundles | Done |
| `QUA-565` MCP resources: durable URIs for runs, models, snapshots, providers, and policies | Done |
| `QUA-566` MCP prompts and host packaging: thin workflows over stable Trellis tool contracts | Done |
| `QUA-587` MCP model lifecycle: require per-version validation evidence and preserve version audit artifacts | Done |
| `QUA-588` MCP transport: local streamable HTTP server for Codex and Claude Code | Done |
| `QUA-622` MCP local demo mode: sandbox mock prompt-flow bootstrap | Done |
| `QUA-683` Session context: clear active snapshot when market-data bindings change | Done |

## Preconditions

Do not start this implementation until the migration plan has produced:

1. one authoritative compiled-request executor
2. explicit `RunMode`, `ExecutionContext`, and `PolicyBundle`
3. governed provider binding and snapshot resolution
4. a canonical run ledger with stable provenance fields
5. a separate model lifecycle distinct from route/knowledge promotion

## What The MCP Server Should Own

The MCP server should own:

- session context and provider configuration
- run-mode and policy selection
- trade parsing into typed contracts
- model lookup and lifecycle control
- governed pricing orchestration
- audit retrieval
- persistence and version history exposure
- prompt and resource packaging for host workflows

The MCP server should not own:

- numeric pricing formulas invented on the fly
- hidden provider selection logic
- silent mock fallback
- automatic model approval
- host-specific business logic embedded directly into pricing code

## Non-Goals

This first MCP implementation does not need to:

- build a distributed production database
- replace notebook or direct-library workflows
- expose every current research/eval script through MCP
- redesign every internal pricing engine API
- implement every future host integration feature in the first tranche

## Repo Assets To Reuse

The implementation should explicitly reuse the strongest pieces already in the
repo.

### Typed Trade Identity

Use:

- `trellis.agent.semantic_contracts.draft_semantic_contract(...)`
- `trellis.agent.knowledge.schema.ProductIR`
- `trellis.agent.semantic_contract_compiler`

Do not use `TermSheet` as the canonical trade identity for MCP. `parse_term_sheet`
can remain a compatibility helper for older natural-language flows.

### Request Compilation

Use:

- `trellis.agent.platform_requests.PlatformRequest`
- `trellis.agent.platform_requests.ExecutionPlan`
- `trellis.agent.platform_requests.CompiledPlatformRequest`
- `trellis.agent.platform_requests.compile_platform_request(...)`

The MCP implementation should consume the platform executor built during the
migration rather than calling deep pricing code directly.

### Candidate Generation

Use:

- `trellis.agent.knowledge.build_with_knowledge(...)`
- `trellis.agent.executor.build_payoff(...)`

But only as draft-candidate generation services, not as implicit production
execution.

### Validation

Use:

- validation bundles
- reference oracles
- critic/arbiter
- model audit outputs

But treat LLM-based validation outputs as advisory research metadata unless
they are wrapped by deterministic policy gates.

### Audit Inputs

Reuse and normalize:

- `trellis.agent.platform_traces`
- `trellis.agent.model_audit`
- `trellis.agent.task_run_store`

These should become feeders into the new governed run ledger, not parallel
durable stores.

## Target Package Layout

The recommended package layout after migration and during MCP implementation is:

```text
trellis/
  platform/
    requests.py
    executor.py
    context.py
    results.py
    policies.py
    providers.py
    models.py
    runs.py
    storage.py
    services/
      session_service.py
      provider_service.py
      trade_service.py
      model_service.py
      validation_service.py
      pricing_service.py
      audit_service.py
      snapshot_service.py
  mcp/
    server.py
    tool_registry.py
    schemas.py
    resources.py
    prompts.py
    errors.py
    tools/
      session_tools.py
      provider_tools.py
      trade_tools.py
      model_tools.py
      price_tools.py
      run_tools.py
      snapshot_tools.py
```

### Layout Rules

- `trellis.platform` remains transport-neutral and testable without MCP
- `trellis.mcp` stays thin and only adapts MCP protocol concerns
- `trellis.agent` remains the research and candidate-generation layer

## State And Storage Model

## State Root

Introduce a configurable state root such as:

```text
.trellis_state/
  config/
    server.yaml
  sessions/
    {session_id}.json
  providers/
    {provider_id}.json
  policies/
    {policy_id}.yaml
  models/
    {model_id}/
      manifest.json
      versions/
        {version}/
          contract.json
          implementation.py
          methodology.json
          validation-plan.json
          validation-report.json
          lineage.json
  runs/
    {run_id}/
      summary.json
      inputs.json
      outputs.json
      audit.json
      logs.jsonl
      events.jsonl
  snapshots/
    {snapshot_id}.json
```

The state root should be configurable through:

- server config
- environment variable such as `TRELLIS_STATE_ROOT`
- explicit bootstrap arguments

### Why A New State Root Is Necessary

Current durable artifacts are spread across research-oriented locations.
That is acceptable for the library and eval loops. It is not the right
storage boundary for a governed server.

## Recommended Config Shape

The server should have a first-pass config shape close to:

```yaml
server:
  name: trellis
  transport: streamable_http

defaults:
  run_mode: research
  output_mode: structured
  audit_mode: summary
  allow_mock_data: false
  require_explicit_provider_binding: true

providers:
  market_data:
    primary: null
    fallback: null
    mock: md_mock_default
  pricing_engine:
    primary: null
  model_store:
    primary: model_store_main
  validation_engine:
    primary: validation_core_v1

policies:
  active: quant-safe-v1
  production_requires:
    - approved_model
    - explicit_market_data_provider
    - persisted_market_snapshot
    - full_run_ledger

lifecycle:
  statuses: [draft, validated, approved, deprecated]
  auto_promote: false

observability:
  persist_run_ledgers: true
  persist_logs: true
  emit_progress_events: true
```

This is a recommended starting point, not a promise that the current code
already satisfies it.

## Core Platform Records

The MCP implementation should define concrete platform records for:

- `SessionContextRecord`
- `ProviderRecord`
- `ProviderBindingRecord`
- `PolicyBundleRecord`
- `ModelRecord`
- `ModelVersionRecord`
- `RunRecord`
- `SnapshotRecord`
- `ValidationRecord`

### SessionContextRecord

Minimum fields:

- `session_id`
- `run_mode`
- `default_output_mode`
- `default_audit_mode`
- `connected_providers`
- `provider_bindings`
- `active_policy`
- `allow_mock_data`
- `require_provider_disclosure`
- `created_at`
- `updated_at`

### ProviderRecord

Minimum fields:

- `provider_id`
- `kind`
- `display_name`
- `status`
- `capabilities`
- `connection_mode`
- `config_summary`
- `is_mock`
- `supports_snapshots`
- `supports_streaming`

The current local-first substrate is implemented in `trellis.platform.providers`.
`ProviderRegistry` now exposes stable governed market-data provider ids and
`resolve_governed_market_snapshot()` enforces explicit provider binding with
canonical `snapshot_id` provenance instead of silent mock fallback.

### PolicyBundleRecord

Minimum fields:

- `policy_id`
- `name`
- `allowed_run_modes`
- `production_requirements`
- `mock_data_policy`
- `provider_disclosure_policy`
- `candidate_generation_policy`
- `audit_retention_policy`
- `lifecycle_rules`

The current local-first substrate is implemented in `trellis.platform.policies`.
`PolicyBundle` now has executable default sandbox, research, and production
bundles, while `evaluate_execution_policy()` and `enforce_execution_policy()`
return structured blocker outcomes that later executor and MCP surfaces can
attach to run records and audit responses.

### ModelRecord

Minimum fields:

- `model_id`
- `semantic_id`
- `semantic_version`
- `product_family`
- `status`
- `latest_version`
- `created_at`
- `updated_at`
- `tags`

The current local-first substrate is implemented in `trellis.platform.models`.
`ModelRecord` now persists the typed match basis plus the latest overall,
validated, and approved version pointers, and the same module now exposes the
execution-time lifecycle gate that enforces approved-only production execution
while keeping lower-mode exceptions explicit and policy-controlled.

### ModelVersionRecord

Minimum fields:

- `model_id`
- `version`
- `status`
- `contract_summary`
- `methodology_summary`
- `engine_binding`
- `validation_summary`
- `lineage`
- `artifacts`
- `created_at`

The first governed implementation also persists:

- explicit lifecycle-transition history
- validation references separate from validation summaries
- lineage back to audit/run/request artifacts without treating those artifacts
  as approval state

### RunRecord

Minimum fields:

- `run_id`
- `request_id`
- `status`
- `run_mode`
- `session_id`
- `policy_id`
- `trade_identity`
- `selected_model`
- `selected_engine`
- `provider_bindings`
- `market_snapshot_id`
- `valuation_timestamp`
- `warnings`
- `result_summary`
- `artifact_paths`
- `created_at`
- `updated_at`

The current local-first substrate is implemented in `trellis.platform.runs`.
It persists canonical `RunRecord` files plus normalized `ArtifactReference`
entries so later audit and MCP run APIs can resolve existing traces, model
audits, and task-run artifacts from one run-ledger record instead of scraping
those stores independently.

The governed audit package that now sits on top of that ledger is
`trellis.platform.audits.RunAuditBundle`. Its stable sections are:

- `run`
- `inputs`
- `execution`
- `outputs`
- `diagnostics`
- `artifacts`

## Service Architecture

The MCP tools should call explicit platform services, not reach deep into
today's modules ad hoc.

## SessionService

Responsibilities:

- load and persist session context
- return active run mode and policy bundle
- return current provider bindings
- update session-level defaults

Feeds MCP tools:

- `trellis.session.get_context`
- `trellis.run_mode.set`

## ProviderService

Responsibilities:

- list visible providers
- configure provider bindings
- validate provider capabilities
- enforce provider-binding rules for governed modes

Feeds MCP tools:

- `trellis.providers.list`
- `trellis.providers.configure`

## TradeService

Responsibilities:

- parse free-form or structured trade specs
- draft semantic contracts
- normalize to `ProductIR`
- report missing fields and parse warnings

Important implementation rule:

`trellis.trade.parse` should use the semantic stack first. It should not use
`TermSheet` as the canonical parse boundary.

Feeds MCP tools:

- `trellis.trade.parse`

## ModelService

Responsibilities:

- typed model matching
- match explanation
- candidate generation
- persistence and version history
- lifecycle transitions
- model diffing

Important implementation rule:

`trellis.model.match` must be typed and deterministic. It must not depend on
lesson similarity, similar-product retrieval, or fuzzy semantic reuse.

Feeds MCP tools:

- `trellis.model.match`
- `trellis.model.explain_match`
- `trellis.model.generate_candidate`
- `trellis.model.validate`
- `trellis.model.promote`
- `trellis.model.persist`
- `trellis.model.versions.list`
- `trellis.model.diff`

## ValidationService

Responsibilities:

- run deterministic validation bundles
- run reference or benchmark comparisons
- attach policy compliance checks
- persist validation records

Important implementation rule:

Research enrichments such as critic output or model-validator commentary may be
attached to validation records, but they must not be the sole basis for
approval.

Feeds MCP tools:

- `trellis.model.validate`

## PricingService

Responsibilities:

- orchestrate parse, select, validate, execute, and audit
- call the unified platform executor
- enforce execution policy
- project output into concise, structured, or audit modes

The current executor substrate under this service is now:

- `trellis.platform.executor.execute_compiled_request(...)`
- `trellis.platform.results.ExecutionResult`

The thin executor adapters currently cover:

- direct instrument pricing
- direct book pricing
- direct Greeks
- direct analytics
- matched-existing-payoff pricing
- candidate-generation ``build_then_price`` execution for legacy ask-style compatibility

Comparison flows still remain pending at the executor layer.

Important implementation rule:

`trellis.price.trade` must stay thin. It is an orchestrator over explicit
stages, not a new opaque super-function.

Internal stages:

1. parse
2. match or select model
3. validate policy and lifecycle eligibility
4. bind providers and snapshot
5. execute pricing
6. build provenance
7. persist run ledger
8. optionally persist reproducibility bundle

Feeds MCP tools:

- `trellis.price.trade`

## AuditService

Responsibilities:

- return run summaries
- build full audit packages from `RunAuditBundle`
- return run logs and events

Feeds MCP tools:

- `trellis.run.get`
- `trellis.run.get_audit`
- `trellis.run.stream_events`

## SnapshotService

Responsibilities:

- persist market snapshots or reproducibility bundles
- retrieve snapshot records by id
- attach snapshot identity to governed runs

Feeds MCP tools:

- `trellis.snapshot.persist_run`

## Model Matching Rules

`trellis.model.match` should use a deterministic matching algorithm built on
the existing semantic infrastructure.

### Minimum Match Basis

- semantic id
- semantic version
- product family
- instrument class
- payoff family
- exercise style
- underlier structure
- market-data capability set
- payout or reporting currency
- method family
- lifecycle status

### Optional Secondary Dimensions

- validation bundle id
- route family
- methodology tags
- engine family

### Explicit Non-Inputs

Do not include:

- lesson overlap score
- similar-product retrieval score
- route-candidate confidence from the knowledge system
- fuzzy text similarity

### Match Result Vocabulary

The service should distinguish at least:

- exact approved match
- exact validated match
- structurally compatible but not execution-eligible
- no match

The explanation tool should say why candidates were accepted or rejected.

## Candidate Generation Rules

`trellis.model.generate_candidate` should wrap the research build flow after
migration, likely through `build_with_knowledge()` or lower-level build
services.

Required behavior:

- emits a draft candidate only
- persists contract, implementation, methodology, and validation-plan artifacts
- records full provenance
- never promotes automatically

Important separation:

- route promotion or adoption remains a library-maintenance workflow
- candidate-model generation is an MCP-facing governed workflow

These are related, but they are not the same lifecycle.

## Validation Rules

The validation tool should combine:

- schema completeness
- capability completeness
- deterministic validation bundle
- reference oracle where applicable
- benchmark or comparison suite
- convergence diagnostics where relevant
- policy compliance checks

Research-only enrichments may include:

- critic findings
- model-validator commentary
- learning or reflection notes

Those enrichments should never define approval eligibility on their own.

## Pricing Output Modes

Every pricing run should be projectable into:

- concise mode
- structured mode
- audit mode

### Concise Mode

Return:

- top-level result summary
- high-signal warnings
- minimal provenance pointer

### Structured Mode

Return:

- run id
- status
- result payload
- warnings
- provenance block
- audit uri

### Audit Mode

Return the structured payload plus:

- execution stages
- provider and policy details
- validation summary
- artifact links
- snapshot and model lineage details

## Recommended MCP Tool Surface

Tool names should be namespaced under `trellis.`.

### Session And Provider Tools

- `trellis.session.get_context`
- `trellis.providers.list`
- `trellis.providers.configure`
- `trellis.run_mode.set`

### Trade And Model Selection Tools

- `trellis.trade.parse`
- `trellis.model.match`
- `trellis.model.explain_match`

### Pricing And Audit Tools

- `trellis.price.trade`
- `trellis.run.get`
- `trellis.run.get_audit`
- `trellis.run.stream_events`

### Candidate Generation And Lifecycle Tools

- `trellis.model.generate_candidate`
- `trellis.model.validate`
- `trellis.model.promote`
- `trellis.model.persist`
- `trellis.model.versions.list`
- `trellis.model.diff`
- `trellis.snapshot.persist_run`

## Recommended Resource Surface

Resources should mirror durable storage objects rather than rebuild ad hoc
views from scattered helper files.

### Model Resources

- `trellis://models/{model_id}`
- `trellis://models/{model_id}/versions`
- `trellis://models/{model_id}/versions/{version}/contract`
- `trellis://models/{model_id}/versions/{version}/code`
- `trellis://models/{model_id}/versions/{version}/validation-report`

### Run Resources

- `trellis://runs/{run_id}`
- `trellis://runs/{run_id}/audit`
- `trellis://runs/{run_id}/logs`
- `trellis://runs/{run_id}/inputs`
- `trellis://runs/{run_id}/outputs`

### Data Resources

- `trellis://market-snapshots/{snapshot_id}`
- `trellis://providers/{provider_id}`
- `trellis://policies/{policy_id}`

### Resource Rule

Every resource URI should map to one canonical stored object or one stable
projection over canonical objects. Do not let resources become wrappers over
random internal helper functions.

## Recommended Prompt Surface

Prompts are host workflows, not core contracts.

Recommended prompts:

- `price_trade`
- `price_trade_audit`
- `persist_current_model`
- `compare_model_versions`
- `explain_model_selection`
- `configure_market_data`
- `validate_candidate_model`

Prompt rule:

- prompts should call MCP tools and resources
- prompts should not bypass the structured tool surface

## Streaming And Eventing

Long-running operations should emit progress and event records suitable for MCP
progress, logging, and cancellation surfaces.

Suggested event types:

- `parse_started`
- `parse_completed`
- `model_match_started`
- `model_match_completed`
- `candidate_generation_started`
- `candidate_generation_completed`
- `validation_started`
- `validation_completed`
- `pricing_started`
- `pricing_completed`
- `audit_persisted`
- `snapshot_persisted`
- `run_failed`

These events should be backed by the canonical run ledger. Do not invent a
parallel streaming state system.

## Provider And Policy Rules

## Run Modes

At minimum:

- `sandbox`
- `research`
- `production`

### sandbox

- explicit mock usage allowed
- candidate generation allowed by default
- advisory validation enrichment allowed

### research

- real providers preferred
- explicit mock usage allowed only when configured
- candidate generation allowed only when policy says so
- outputs must disclose provider and lifecycle status

### production

- no silent mock fallback
- explicit provider binding required
- approved model required
- persisted snapshot required
- full run ledger required
- candidate generation disabled by default

## Policy Bundle Rules

Policies should be executable configuration, not narrative documentation.

At minimum, the active policy bundle should decide:

- mock-data allowance
- provider-disclosure requirement
- candidate-generation allowance
- lifecycle status required for execution
- audit persistence requirement
- snapshot persistence requirement
- validation gate thresholds

## Implementation Tranches

## Tranche 0: Platform Schemas And Storage Foundation

### Objective

Create the records, state root, and storage abstractions that everything else
depends on.

### Deliverables

- platform record dataclasses
- storage interfaces
- local filesystem-backed default implementation
- config loading and bootstrap

### Files Likely To Add

- `trellis/platform/storage.py`
- `trellis/platform/models.py`
- `trellis/platform/runs.py`
- `trellis/platform/context.py`

### Acceptance Criteria

- platform services can persist and reload sessions, models, runs, and
  snapshots without MCP transport code

## Tranche 1: Session And Provider Control

### Objective

Expose governed session context and provider binding before pricing.

### MCP Tools

- `trellis.session.get_context`
- `trellis.providers.list`
- `trellis.providers.configure`
- `trellis.run_mode.set`

### Required Behavior

- provider bindings are explicit and persisted
- run mode is explicit and persisted
- provider disclosure is visible in structured responses
- production mode blocks implicit mock behavior

### Acceptance Criteria

- a host can connect, inspect context, bind providers, and set run mode
- no pricing tool is needed to discover the active governance state

## Tranche 2: Trade Parsing And Typed Model Matching

### Objective

Establish the typed contract and deterministic match surface.

### MCP Tools

- `trellis.trade.parse`
- `trellis.model.match`
- `trellis.model.explain_match`

### Required Behavior

`trellis.trade.parse` should return:

- trade type or semantic id
- asset class
- parsed contract
- missing fields
- warnings
- contract summary

`trellis.model.match` should return:

- match type
- candidates
- match basis
- lifecycle status
- engine/model identities

### Acceptance Criteria

- model selection for governed pricing no longer depends on fuzzy similarity
- parse and match outputs are machine-readable and stable

## Tranche 3: Governed Pricing And Audit Retrieval

### Objective

Deliver the core governed pricing loop and its audit retrieval path.

### MCP Tools

- `trellis.price.trade`
- `trellis.run.get`
- `trellis.run.get_audit`
- optional `trellis.run.stream_events`

### Required Behavior

`trellis.price.trade` should remain thin and should always surface these audit
stages:

1. parse
2. model selection
3. validation and policy gating
4. execution
5. result packaging
6. persistence

### Output Requirements

Every pricing response should contain:

- run id
- completion status
- result payload
- warnings
- provenance block
- audit uri

The provenance block must include:

- model id and version
- engine id and version
- market-data provider
- market snapshot id
- valuation timestamp

### Acceptance Criteria

- a governed pricing run can be executed end to end through MCP
- the run can be fetched later by id and reconstructed from the audit package

That audit package is now concretely the canonical `RunAuditBundle`, not an
ad hoc synthesis step at the MCP layer.

## Tranche 4: Candidate Generation, Validation, And Lifecycle Control

### Objective

Expose controlled candidate generation and lifecycle operations only after the
governed pricing surface exists.

### MCP Tools

- `trellis.model.generate_candidate`
- `trellis.model.validate`
- `trellis.model.promote`
- `trellis.model.persist`
- `trellis.model.versions.list`
- `trellis.model.diff`
- `trellis.snapshot.persist_run`

### Important Lifecycle Rule

The governed lifecycle is:

- `draft`
- `validated`
- `approved`
- `deprecated`

Generated candidates must never become approved implicitly.

### Acceptance Criteria

- candidate artifacts are persistable and inspectable
- validation records are durable
- lifecycle transitions are explicit and auditable

## Tranche 5: Resource URIs

### Objective

Expose durable inspectable resources over models, runs, snapshots, providers,
and policies.

### Acceptance Criteria

- resource URIs resolve cleanly from canonical stored objects
- audit and version artifacts are inspectable without re-running pricing

## Tranche 6: Prompts And Host Workflows

### Objective

Add host-native workflows after the tool and resource surfaces stabilize.

### Acceptance Criteria

- prompts compose the same tool contracts already used elsewhere
- prompts do not introduce hidden business logic

## Tranche 7: Transport And Host Packaging

### Objective

Package the transport-neutral server core for real hosts.

### Order

1. transport-neutral platform services
2. `trellis.mcp` server adapter
3. chosen transport wiring such as `streamable_http`
4. host packaging for Claude, Codex, and ChatGPT

### Acceptance Criteria

- host packaging wraps the same MCP tool contracts
- host-specific wrappers do not fork core pricing behavior

## Suggested Test Plan

Add dedicated tests such as:

- `tests/test_platform/test_policy_enforcement.py`
- `tests/test_platform/test_model_registry.py`
- `tests/test_platform/test_snapshot_store.py`
- `tests/test_mcp/test_session_tools.py`
- `tests/test_mcp/test_provider_tools.py`
- `tests/test_mcp/test_trade_parse_tool.py`
- `tests/test_mcp/test_model_match_tool.py`
- `tests/test_mcp/test_price_trade_tool.py`
- `tests/test_mcp/test_run_audit_resources.py`

Critical scenarios:

- production run with approved model and explicit provider binding succeeds
- production run with missing provider binding fails
- production run with only mock provider available fails unless policy
  explicitly allows it
- research run can generate a candidate when policy allows it
- a validated candidate remains non-approved until explicitly promoted
- run audit retrieval contains full provenance and stage history

## Rollout Strategy

Recommended order:

1. local filesystem-backed server state
2. session/provider/policy tools
3. trade parsing service
4. typed model registry and matching
5. governed pricing tool and audit retrieval
6. candidate generation and lifecycle tools
7. resources
8. prompts
9. transport hardening
10. host packaging

This order makes Trellis credible as a governed execution surface before it
becomes ambitious as an agentic workflow surface.

## Acceptance Criteria For The Overall MCP Initiative

The MCP implementation should be considered complete only when:

1. the server exposes the planned tool families
2. run mode and provider bindings are explicit and persisted
3. `trellis.trade.parse` produces typed contract outputs from the semantic
   stack
4. `trellis.model.match` performs typed matching only
5. `trellis.price.trade` returns structured results plus provenance and audit
   uri
6. every governed run has run id, snapshot id, model version, and engine
   version
7. production mode cannot silently use mock data or draft models
8. model lifecycle transitions are explicit and persisted
9. audit resources can reconstruct the full run story from canonical records
10. prompts and resources remain thin layers over the same tool surface

## Ticket Decomposition Guidance

When this plan is broken into Linear issues, the first issue groups should be:

1. state root, storage interfaces, and platform records
2. session/provider/policy services
3. MCP session/provider tools
4. semantic trade parsing service and tool
5. typed model registry and match service
6. governed pricing service and `trellis.price.trade`
7. run retrieval and audit packaging tools
8. candidate generation and validation lifecycle tools
9. resources
10. prompts and host packaging

That sequence ensures governance and typed execution exist before the server
starts exposing the more powerful generation and lifecycle workflows.
