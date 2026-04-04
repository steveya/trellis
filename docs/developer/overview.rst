Developer Overview
==================

At developer scope, Trellis is more than a pricing library. It is a request
compiler, knowledge-backed build system, validation pipeline, and audit trail
around the deterministic pricing engines.

Platform Surfaces
-----------------

The main entry points all compile into a common internal representation:

- ``trellis.ask(...)`` and ``Session.ask(...)`` for natural-language pricing
- ``Session.price(...)`` and direct market-state workflows for deterministic pricing
- ``Pipeline.run()`` for declarative batch and scenario execution
- structured user-defined and comparison requests in ``trellis.agent.platform_requests``

The canonical request/compiler layer lives in ``trellis.agent.platform_requests``.
It normalizes these surfaces into ``PlatformRequest`` and ``CompiledPlatformRequest``
objects with execution plans, method selection, knowledge payloads, and blocker reports.

Governed runtime state now has a dedicated platform boundary in
``trellis.platform``. ``trellis.platform.context`` owns explicit
``RunMode``, ``ProviderBindings``, and ``ExecutionContext`` records, while
``trellis.platform.providers`` owns governed provider ids and bound snapshot
resolution. That keeps runtime governance separate from request intent and
separates governed market resolution from the old convenience ``source=...``
surface.

The transport-neutral MCP-facing service layer now also starts in
``trellis.platform.services``:

- ``SessionService`` owns durable governed session context, run-mode
  persistence, and policy-bundle projection
- ``ProviderService`` owns explicit provider listing and session-scoped binding
  updates
- ``TradeService`` owns typed trade normalization onto semantic-contract and
  ``ProductIR`` outputs with explicit missing-field reporting
- ``ModelService`` owns deterministic registry matching and explain-match
  projections plus governed candidate generation, version persistence, lifecycle
  transitions, and review diffs
- ``ValidationService`` owns deterministic model-version validation reports and
  the persisted validation records that lifecycle transitions can later require
- ``PricingService`` owns the narrow MCP ``trellis.price.trade`` orchestration
  over parse, match, provider resolution, executor dispatch, and run/audit
  persistence
- ``trellis.mcp.resources`` owns stable durable read URIs over governed models,
  runs, snapshots, providers, and policies
- ``trellis.mcp.prompts`` owns thin host workflows that compose the stable
  tools and resources instead of bypassing them
- ``trellis.mcp.http_transport`` wraps that same shell in a local streamable
  HTTP transport for Codex and Claude Code testing
- lifecycle execution eligibility remains a separate concern in
  ``trellis.platform.models.enforce_model_execution_gate(...)`` so matching does
  not quietly decide production eligibility

Execution Flow
--------------

The operational flow is:

1. create a request from an entry surface
2. normalize convenience runtime state into ``ExecutionContext`` when governed
   execution needs explicit run mode, provider bindings, or policy identity
3. resolve governed market snapshots by bound provider id when the caller is in
   the migrated platform path
4. compile the request into execution intent plus shared knowledge
5. execute deterministic pricing or agent-assisted build/validation
6. append trace events and optional external issue updates

That means developer work often crosses both the quant layer and the runtime
layer. A route-method change can alter knowledge retrieval, audit traces, and
task-batch behavior even when the underlying math is unchanged.

The governed executor boundary is now the live runtime path for the public
pricing surfaces:

- ``trellis.platform.executor.execute_compiled_request(...)`` is the single
  compiled-request dispatcher entry point
- ``trellis.platform.results.ExecutionResult`` is the stable internal result
  envelope that public API projections build on
- thin deterministic adapters now serve direct instrument pricing, book pricing,
  direct Greeks, direct analytics, matched-existing-payoff pricing, and the
  candidate-generation ``build_then_price`` path
- ``trellis.ask(...)``, ``Session.price(...)``, ``Session.greeks(...)``,
  ``Session.analyze(...)``, and ``Pipeline.run()`` now compile, execute through
  the same governed spine, and project back into their historical return types
- ``compare_methods`` remains the only structured pending executor action in
  this migration tranche

Where Things Live
-----------------

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Concern
     - Main modules
     - Notes
   * - Request compilation
     - ``trellis.agent.platform_requests``
     - Unifies ask, session, pipeline, user-defined, and comparison flows
   * - Governed runtime context
     - ``trellis.platform.context``
     - Explicit ``RunMode``, provider bindings, and serializable execution-context records
   * - Governed policy layer
     - ``trellis.platform.policies``
     - Default sandbox/research/production policy bundles plus deterministic execution guards and structured blocker outcomes
   * - Governed provider registry
     - ``trellis.platform.providers``
     - Stable provider ids, explicit governed snapshot resolution, snapshot ids, and no silent mock fallback on governed paths
   * - Governed executor
     - ``trellis.platform.executor``, ``trellis.platform.results``
     - Authoritative compiled-request dispatcher plus the stable ``ExecutionResult`` envelope used for success, blocked, and failure outcomes
   * - Governed model registry
     - ``trellis.platform.models``
     - Local-first model and version records with explicit lifecycle transitions plus execution-time lifecycle gating distinct from research audit
   * - Governed run ledger
     - ``trellis.platform.runs``
     - Canonical run records and artifact references that point to traces, audits, and task-run files
   * - Transport-neutral MCP services
     - ``trellis.platform.services.session_service``, ``trellis.platform.services.provider_service``, ``trellis.platform.services.trade_service``, ``trellis.platform.services.model_service``, ``trellis.platform.services.validation_service``, ``trellis.platform.services.pricing_service``
     - Session context, provider control, typed trade normalization, deterministic model matching, candidate persistence, deterministic validation, and narrow approved-model MCP pricing orchestration
   * - Governed audit bundle
     - ``trellis.platform.audits``
     - Deterministic audit packages built from canonical run records plus linked trace, model-audit, task-run, and diagnosis artifacts
   * - MCP resources and prompts
     - ``trellis.mcp.resources``, ``trellis.mcp.prompts``, ``trellis.mcp.server``, ``trellis.mcp.http_transport``
     - Stable URI reads, thin host workflows, the transport-neutral shell, and the local streamable HTTP wrapper over the same tool/resource contract
   * - Agent loop
     - ``trellis.agent.quant``, ``planner``, ``builder``, ``critic``, ``executor``
     - Method routing, spec planning, code generation, and validation
   * - Knowledge system
     - ``trellis.agent.knowledge``
     - Retrieval, promotion, import registry, traces, and canonical YAML assets
   * - Audit and issue sync
     - ``platform_traces``, ``github_tracker``, ``linear_tracker``
     - YAML traces plus best-effort GitHub/Linear issue creation and comments
   * - Validation and evals
     - ``model_validator``, ``validation_report``, ``evals``
     - Deterministic and LLM-assisted grading around generated artifacts
   * - Grid pricing substrate
     - ``trellis.models.trees.algebra``, ``trellis.models.grid_protocols``, ``trellis.models.pde``
     - Shared lattice/PDE rollback contracts, exercise boundaries, local-vol and two-factor lattice extensions
   * - Task runtime
     - ``task_runtime.py``, ``scripts/*.py``, ``TASKS.yaml``, ``FRAMEWORK_TASKS.yaml``
     - Batch execution, reruns, benchmarking, remediation, and separate pricing-vs-framework task inventories

Read Next
---------

- :doc:`hosting_and_configuration`
- :doc:`audit_and_observability`
- :doc:`task_and_eval_loops`
- :doc:`../quant/index`
