# Trellis Architecture

This file is the authoritative high-level architecture summary for the checked-in
codebase.

Use it together with:

- `AGENTS.md` for workflow, ownership, and implementation rules
- `LIMITATIONS.md` for the current support contract and known gaps
- `docs/quant/pricing_stack.rst` for the deterministic pricing-stack details
- `docs/developer/overview.rst` for the governed platform and runtime view
- `docs/agent/architecture.rst` for the classic multi-agent pipeline reference

## What Trellis Ships

Trellis is not only a pricing library. The current repository contains four
connected layers:

1. A deterministic quantitative pricing library for instruments, curves,
   analytics, and numerical methods.
2. An agent/request-compiler layer that turns user requests into checked
   pricing plans, route selections, and generated payoff adapters when needed.
3. A governed platform layer for sessions, providers, model lifecycle,
   policies, validation, storage, and audit trails.
4. A thin MCP server shell that exposes those platform services to external
   hosts.

The important boundary is that LLMs are not in the deterministic pricing hot
path. They participate in parsing, planning, generation, review, and knowledge
maintenance around the deterministic engines.

## Top-Level Package Map

### Public entry points

- `trellis/__init__.py` exports the package-level API, including `ask`,
  `price`, `Session`, `Pipeline`, sample objects, core market/runtime types,
  and common pricing helpers.
- `trellis/session.py` is the main immutable interactive pricing surface.
- `trellis/pipeline.py` is the declarative batch/scenario pricing surface.

### Deterministic pricing runtime

- `trellis/core/` defines the shared runtime contracts such as `MarketState`,
  payoff interfaces, runtime contracts, types, date math, and differentiable
  NumPy access.
- `trellis/conventions/` contains calendar, schedule, day-count, and rate-index
  conventions.
- `trellis/curves/` contains yield, forward, credit, bootstrap, shock, and
  scenario-pack logic.
- `trellis/data/` resolves and stores market snapshots and provider-backed data.
- `trellis/instruments/` contains hand-written reference instruments plus
  checked-in/generated agent adapters under `trellis/instruments/_agent/`.
- `trellis/engine/` contains direct pricing and payoff-pricing entry points.
- `trellis/analytics/` contains runtime measures, benchmarks, and explain/risk
  helpers.
- `trellis/models/` contains the numerical engines and reusable quantitative
  kernels:
  - `analytical/`
  - `trees/`
  - `monte_carlo/`
  - `pde/`
  - `transforms/`
  - `copulas/`
  - `processes/`
  - `calibration/`
  - `cashflow_engine/`
  - `qmc/`
  - `resolution/`

### Request compilation and agent orchestration

- `trellis/agent/platform_requests.py` is the canonical request/compiler layer.
  It normalizes ask, session, pipeline, and user-defined requests into
  `PlatformRequest` and `CompiledPlatformRequest`.
- `trellis/agent/semantic_contracts.py`,
  `trellis/agent/semantic_contract_compiler.py`,
  `trellis/agent/valuation_context.py`, and
  `trellis/agent/market_binding.py` hold the typed semantic and valuation
  boundary.
- `trellis/agent/route_registry.py`, `trellis/agent/build_gate.py`,
  `trellis/agent/family_lowering_ir.py`, and `trellis/agent/dsl_lowering.py`
  govern admissibility and lowering onto checked route families.
- `trellis/agent/quant.py`, `trellis/agent/planner.py`,
  `trellis/agent/builder.py`, `trellis/agent/critic.py`,
  `trellis/agent/arbiter.py`, and `trellis/agent/executor.py` implement the
  planning/build/review loop around the deterministic runtime.

### Knowledge system

- `trellis/agent/knowledge/` is the self-maintaining knowledge layer.
- `canonical/api_map.yaml` is the small family-level navigation map.
- `import_registry.py` is the authoritative import-path registry for anti-
  hallucination and prompt grounding.
- `canonical/features.yaml`, `canonical/decompositions.yaml`,
  `canonical/cookbooks.yaml`, `canonical/data_contracts.yaml`, and
  `canonical/method_requirements.yaml` define the structured knowledge assets.
- `store.py`, `retrieval.py`, `decompose.py`, `gap_check.py`, `reflect.py`, and
  `promotion.py` implement retrieval, auditing, reflection, and promotion.

### Governed platform layer

- `trellis/platform/` provides the transport-neutral orchestration layer.
- `context.py`, `requests.py`, and `results.py` define execution envelopes.
- `providers.py`, `policies.py`, `models.py`, `runs.py`, and `storage.py`
  govern provider resolution, execution policy, model lifecycle, run ledgers,
  and persistent state.
- `services/` bootstraps reusable session, pricing, validation, provider,
  model, audit, trade, and snapshot services for hosts.

### MCP transport layer

- `trellis/mcp/` is a thin adapter over the platform services.
- `server.py` bootstraps the MCP shell.
- `tool_registry.py`, `resources.py`, and `prompts.py` expose tool, resource,
  and prompt registries without duplicating platform logic.

## Main Execution Flows

### Direct deterministic pricing

The direct library path is:

1. Create or resolve a `Session`.
2. Price an instrument or book through `Session.price(...)` or `trellis.price`.
3. Route to direct pricing helpers in `trellis.engine.pricer` or payoff pricing
   in `trellis.engine.payoff_pricer`.
4. Compute analytics and scenario/risk projections through `trellis.analytics`
   and `trellis.pipeline`.

This is the default path for supported hand-written instruments and governed
book/scenario workflows.

### Ask / build / agent-assisted pricing

The agent-assisted path is:

1. Enter through `trellis.ask(...)` or `Session.ask(...)`.
2. Normalize the request in `trellis.agent.platform_requests`.
3. Compile semantic contracts, valuation context, required data, and route
   candidates.
4. Reuse an existing checked route when possible; otherwise plan/build/review a
   payoff adapter that lands under `trellis/instruments/_agent/`.
5. Execute pricing through the same deterministic runtime after the route is
   admitted.

### Batch and scenario execution

The batch path is:

1. Build a `Pipeline`.
2. Compile a `BookExecutionPlan`.
3. Expand scenario packs or explicit scenario specs.
4. Execute each scenario through governed session requests.
5. Return a `ScenarioResultCube` that preserves scenario specs, provenance, and
   compute-plan metadata.

### Governed host / MCP execution

The service-host path is:

1. Bootstrap platform services through `trellis.platform.services.bootstrap`.
2. Reuse those services from `trellis.mcp.server` or other host adapters.
3. Execute requests under provider, policy, model, validation, and audit
   controls.

## Architectural Boundaries That Matter When Editing

- Keep pricing math in deterministic library code. Agent and platform layers
  should orchestrate, validate, and govern, not reimplement numerical kernels.
- Treat `trellis/agent/knowledge/import_registry.py` as the authoritative source
  of valid imports.
- Use `trellis/agent/knowledge/canonical/api_map.yaml` to orient to a module
  family before drilling into specific symbols.
- Treat `trellis/agent/knowledge/` as a separately owned subsystem. Do not edit
  it during ordinary library work unless the task is explicitly knowledge work.
- Keep `Session` and market/runtime envelopes immutable in behavior and
  interface unless the change is intentional and documented.
- When behavior, support, or workflow claims change, update both the relevant
  docs and `LIMITATIONS.md` if the support contract moved.

## First Files To Read For Common Tasks

### Pricing/library work

- `AGENTS.md`
- `ARCHITECTURE.md`
- `LIMITATIONS.md`
- `docs/quant/pricing_stack.rst`
- the touched modules under `trellis/core/`, `trellis/curves/`,
  `trellis/instruments/`, `trellis/models/`, `trellis/analytics/`, or
  `trellis/engine/`

### Agent/compiler work

- `AGENTS.md`
- `ARCHITECTURE.md`
- `trellis/agent/platform_requests.py`
- `trellis/agent/semantic_contracts.py`
- `trellis/agent/route_registry.py`
- `trellis/agent/build_gate.py`
- `trellis/agent/knowledge/canonical/api_map.yaml`
- `trellis/agent/knowledge/import_registry.py`

### Platform or MCP work

- `AGENTS.md`
- `ARCHITECTURE.md`
- `docs/developer/overview.rst`
- `trellis/platform/`
- `trellis/mcp/`
