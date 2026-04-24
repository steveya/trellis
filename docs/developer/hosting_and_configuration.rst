Hosting And Configuration
=========================

Trellis is primarily hosted as a Python library in this repository. The repo
now also ships a first-party local MCP streamable-HTTP wrapper for host
integrations such as Codex and Claude Code, but broader production hosting
still means choosing which long-running Python process imports and executes the
package.

Operational Modes
-----------------

The repo supports four practical modes:

- embedded library mode for notebooks, scripts, workers, or services that import ``trellis``
- task-runner mode via ``scripts/run_tasks.py`` and related batch tools
- local MCP host mode via ``scripts/serve_trellis_mcp.py`` and
  ``trellis.mcp.http_transport``
- experimental UI mode via the sibling ``trellis-ui`` repo, which uses this package as its pricing backend

The repo now has a first-party localhost MCP HTTP entrypoint, but it is still a
developer/operator transport wrapper rather than a hardened public deployment
surface. Production deployments typically wrap Trellis inside another service
boundary with auth, routing, and exposure controls.

Governed Runtime Context
------------------------

The current migration path separates request intent from governed runtime state.
``trellis.platform.context`` defines:

- ``RunMode`` for ``sandbox``, ``research``, and ``production``
- ``ProviderBindings`` for explicit provider ids instead of raw source strings
- ``ExecutionContext`` as the serializable runtime carrier for policy identity,
  output defaults, disclosure requirements, and persistence preferences

``trellis.platform.providers`` now defines the governed market-data boundary:

- ``ProviderRecord`` for stable provider ids and declared capabilities
- ``ProviderRegistry`` for local-first provider lookup
- ``resolve_governed_market_snapshot()`` for explicit provider-bound snapshot
  resolution with canonical ``snapshot_id`` provenance

``trellis.platform.policies`` now turns the policy id on ``ExecutionContext``
into executable guards:

- ``PolicyBundle`` for the sandbox, research, and production default policies
- ``evaluate_execution_policy()`` for deterministic policy outcomes
- ``enforce_execution_policy()`` for a structured guard error when execution is blocked

``trellis.platform.models`` now also exposes the governed lifecycle gate used by
later execution paths:

- sandbox default: ``draft`` / ``validated`` / ``approved``
- research default: ``validated`` / ``approved``
- production default: ``approved`` only
- ``deprecated`` is not executable in the default bundles

Compatibility helpers already exist on the public library surfaces:

- ``Session.to_execution_context()`` maps session convenience state into an
  explicit runtime context
- ``Pipeline.to_execution_context()`` does the same for batch configuration

By default, mock or explicit-input market surfaces normalize to sandbox mode,
while live-provider snapshots normalize to research mode unless the caller
overrides the run mode explicitly.

This split is intentional:

- ``resolve_market_snapshot(source=...)`` remains the convenience resolver for
  library workflows, quickstarts, and explicit sandbox flows
- governed execution should resolve market data through explicit provider ids
  carried in ``ExecutionContext.provider_bindings``
- governed resolution never silently substitutes a mock provider; mock data is
  only legal when the bound provider is explicit and policy allows it
- governed execution should run the matching ``PolicyBundle`` before pricing so
  missing provider disclosure, mock-data misuse, or production lifecycle gaps
  fail with structured blocker codes
- once a governed path has identified a model id, it should pass through the
  model execution gate before pricing so lifecycle eligibility is explicit and
  deterministic

MCP State Root And Bootstrap
----------------------------

The repo now has a transport-neutral MCP bootstrap layer under
``trellis.platform``.

The shared local state root defaults to ``.trellis_state/`` at the repo root
and is resolved in this order:

- explicit bootstrap argument
- ``TRELLIS_STATE_ROOT``
- repo-local default ``.trellis_state/``

The bootstrap config lives at ``.trellis_state/config/server.yaml`` and is
managed through ``trellis.platform.storage.load_trellis_server_config()``.
The first bootstrap call writes a default config if none exists yet.

The default filesystem layout is:

.. code-block:: text

   .trellis_state/
     config/server.yaml
     sessions/
     providers/
     policies/
     models/
     runs/
     snapshots/
     validations/

The shared bootstrap entry point is
``trellis.platform.services.bootstrap_platform_services()``. It creates one
transport-neutral service container backed by:

- ``SessionContextStore`` for governed session context
- ``SnapshotStore`` for durable JSON-safe snapshot records
- ``ValidationStore`` for lifecycle validation records
- ``ModelRegistryStore`` for governed model records
- ``RunLedgerStore`` for canonical run records
- ``ProviderRegistry`` for governed provider lookup

The thin MCP shell in ``trellis.mcp.server.bootstrap_mcp_server()`` is expected
to reuse that same container rather than creating parallel MCP-only stores.

The first governed MCP control tools now hang directly off that shared
container:

- ``trellis.session.get_context`` returns the durable session record plus the
  active policy bundle
- ``trellis.run_mode.set`` persists explicit sandbox, research, or production
  mode changes and updates the default policy id
- ``trellis.providers.list`` shows the visible providers together with any
  current session binding slots
- ``trellis.providers.configure`` persists explicit provider bindings and
  rejects unknown provider ids or production-incompatible mock bindings
- rebinding ``market_data`` through ``trellis.providers.configure`` clears any
  active imported market snapshot metadata on that session so later pricing
  runs cannot mix a new ``provider_id`` with a stale imported
  ``market_snapshot_id``
- ``trellis.model.generate_candidate`` persists a governed draft candidate plus
  canonical contract, methodology, lineage, validation-plan, and optional code
  sidecars
- ``trellis.model.validate`` writes a deterministic validation report for one
  exact governed model version
- ``trellis.model.promote`` applies explicit validated, approved, or
  deprecated lifecycle transitions after review of the latest validation state
  for that exact version
- ``trellis.model.persist`` writes a new governed model version with explicit
  lineage instead of mutating the prior version in place; the new version
  starts unvalidated by default and preserves a version-specific code sidecar
  even when the revision is metadata-only and reuses the prior implementation
- ``trellis.model.versions.list`` and ``trellis.model.diff`` expose stored
  version history and review diffs from the canonical model store
- ``trellis.price.trade`` runs the narrow approved-model governed pricing path
  and persists canonical run and audit records
- ``trellis.run.get`` and ``trellis.run.get_audit`` read those canonical run
  and audit records back out by run id
- ``trellis.snapshot.persist_run`` writes a reproducibility bundle snapshot for
  one governed run and attaches that bundle URI back to the run ledger

These tools are intentionally thin. They mutate the filesystem-backed session
records under ``.trellis_state/sessions/`` and leave all pricing, matching, and
audit logic in the transport-neutral ``trellis.platform.services`` layer.

The thin MCP shell now also exposes durable read-oriented resources over the
same stores:

- ``trellis://models/{model_id}`` and ``.../versions`` for governed model and
  version history inspection
- ``trellis://models/{model_id}/versions/{version}/contract``,
  ``.../code``, and ``.../validation-report`` for canonical version sidecars
- ``trellis://runs/{run_id}``, ``.../audit``, ``.../inputs``, ``.../outputs``,
  and ``.../logs`` for governed run inspection
- ``trellis://market-snapshots/{snapshot_id}`` for persisted market snapshots
  and reproducibility bundles
- ``trellis://providers/{provider_id}`` and ``trellis://policies/{policy_id}``
  for the declared provider and policy surfaces the governed runtime uses

Prompts and host packaging stay thin over those same contracts:

- ``trellis.mcp.prompts`` provides guided workflows such as
  ``price_trade``, ``exotic_desk_one_trade``, ``price_trade_audit``, ``persist_current_model``,
  ``compare_model_versions``, ``explain_model_selection``,
  ``configure_market_data``, and ``validate_candidate_model``
- ``TrellisMcpServer.describe_host_packaging()`` returns one common manifest
  for Claude, Codex, and ChatGPT hosts, all pointing at the same bootstrap
  entrypoint, transport, tool surface, resource surface, and prompt surface
- host-specific layers are expected to wrap that common manifest, not fork the
  underlying tool or resource contracts

Local MCP HTTP Transport
------------------------

The repo now also includes a first-party streamable HTTP wrapper in
``trellis.mcp.http_transport`` plus a runnable script at
``scripts/serve_trellis_mcp.py``.

Install the optional dependency first:

.. code-block:: bash

   pip install -e ".[mcp]"

Start the local MCP endpoint:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/serve_trellis_mcp.py \
     --host 127.0.0.1 \
     --port 8000

For the first explicit exotic-desk flow, hosts should prefer the
``exotic_desk_one_trade`` prompt over ad hoc tool sequencing. That prompt
packages the supported imported-snapshot + ``range_accrual`` workflow and
points the host at the returned ``desk_review`` bundle plus the canonical run
and audit resources.

For a localhost prompt-flow smoke test, the launcher also supports an explicit
demo bootstrap:

.. code-block:: bash

   /Users/steveyang/miniforge3/bin/python3 scripts/serve_trellis_mcp.py \
     --host 127.0.0.1 \
     --port 8000 \
     --demo \
     --demo-session-id demo

That demo bootstrap is opt-in. It seeds one sandbox session bound to
``market_data.mock`` and a minimal approved vanilla-option model so local MCP
hosts can exercise the end-to-end ``trellis.price.trade`` flow without live
market connectors or a pre-populated governed registry. Non-demo launches keep
the normal governed defaults unchanged.

The default endpoint is ``http://127.0.0.1:8000/mcp``.

That same endpoint can then be registered with local MCP hosts:

.. code-block:: bash

   codex mcp add trellis --url http://127.0.0.1:8000/mcp
   claude mcp add --transport http trellis http://127.0.0.1:8000/mcp

The HTTP wrapper is intentionally thin:

- it bootstraps the same ``trellis.mcp.server.bootstrap_mcp_server()`` shell
- it exposes the same tool, resource, and prompt names over MCP
- it does not fork pricing, matching, policy, or audit logic away from
  ``trellis.platform.services``

For remote ChatGPT testing later, the same local endpoint can be exposed behind
a tunnel or public reverse proxy. That remote exposure is intentionally outside
this local-host slice.

Installation Reality
--------------------

``setup.py`` currently packages the core library plus optional extras for
agent workflows, MCP transport, live data, and cross-validation. Optional
runtime dependencies are still installed separately based on which surfaces
you intend to run.

.. code-block:: bash

   pip install trellis

   # Optional agent provider client
   pip install openai
   # or
   pip install anthropic

   # Optional local MCP transport
   pip install mcp

   # Optional live-data dependencies
   pip install requests fredapi

You may also need external libraries such as QuantLib, FinancePy, or other
comparison backends depending on which validation paths you intend to run.

Environment Loading
-------------------

``trellis.agent.config.load_env()`` searches upward for a project ``.env`` file
and loads values that are not already present in the process environment.

Key variables in the current codebase include:

- ``LLM_PROVIDER``: provider selector, currently ``openai`` or ``anthropic``
- ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``: provider credentials for direct
  SDK access
- ``GITHUB_MODELS_TOKEN``: optional OpenAI-family task-batch credential; when
  set, Trellis routes OpenAI requests through GitHub Models; when it is absent,
  Trellis uses the direct ``OPENAI_API_KEY`` path instead
- ``GITHUB_MODELS_OPENAI_MODEL``: optional override for the GitHub Models
  catalog id used for OpenAI-family requests (for example
  ``openai/gpt-5.4-mini``)
- ``OPENAI_TEXT_TIMEOUT_SECONDS``, ``OPENAI_JSON_TIMEOUT_SECONDS``, ``OPENAI_MAX_RETRIES``: OpenAI request guards
- ``FRED_API_KEY``: optional live-data access for FRED
- ``GITHUB_REQUEST_AUDIT_TOKEN`` or ``GITHUB_TOKEN``: GitHub issue-sync auth
- ``GITHUB_REQUEST_AUDIT_REPOSITORY`` or ``GITHUB_REPOSITORY``: GitHub repo for request-audit issues
- ``GITHUB_REQUEST_AUDIT_LABELS``: optional GitHub labels for created issues
- ``LINEAR_API_KEY``: Linear issue-sync auth
- ``LINEAR_REQUEST_AUDIT_TEAM_ID`` or ``LINEAR_TEAM_ID``: Linear team key or id
- ``LINEAR_REQUEST_AUDIT_PROJECT_ID``: optional Linear project target
- ``NUMBA_CACHE_DIR``: recommended when running FinancePy-backed validation in restricted environments

Build And Docs
--------------

The public docs for this repo are built from ``docs/`` using Sphinx and
``myst-nb`` for notebook-backed tutorials.

.. code-block:: bash

   cd /Users/steveyang/Projects/steveya/trellis/docs
   make clean html

Related Reading
---------------

- :doc:`overview`
- :doc:`../validation/numba_cache`
- :doc:`../quant/index`
