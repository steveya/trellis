Developer Overview
==================

At developer scope, Trellis is more than a pricing library. It is a request
compiler, knowledge-backed build system, validation pipeline, and audit trail
around the deterministic pricing engines.

The repo now also carries a checked GitHub Actions CI baseline in
``.github/workflows/ci.yml``. The primary ``build-and-test`` lane mirrors the
current local release contract on Linux/Python 3.10 by running
``python -m build``, a tracked-file whitespace check, and the non-integration
pytest command. A second advisory ``typecheck`` lane runs the scoped
``[tool.mypy]`` configuration against the thin MCP boundary modules with
``follow_imports=skip`` so the project gets immediate static-signal coverage
without pretending the full legacy/runtime surface is already type-clean.

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

The plain fixed-income pricing path now also carries desk-readable bond
reporting outputs. ``price_instrument(...)`` and the ``Session.price(...)``
projection solve a coupon-frequency nominal ``ytm`` from the reported dirty
price and compute accrued interest from the coupon schedule plus the selected
day-count convention instead of using a flat period approximation.

Calibration and inversion workflows now have a similar typed substrate in
``trellis.models.calibration.solve_request``. Rates and SABR helpers build a
serializable ``SolveRequest`` plus ``ObjectiveBundle`` first, then dispatch the
request through the current SciPy-backed executor. That keeps optimizer
metadata, bounds, warm starts, and derivative-hook availability visible in
provenance before the later backend-registry layer decides which concrete
solver implementation to use.

The semantic valuation boundary now also carries a bounded
``EngineModelSpec`` surface in ``trellis.agent.valuation_context``. That keeps
model-family binding, potential/source semantics, backend hints, calibration
requirements, and explicit rates discount/forecast curve roles in one
serializable record while preserving the legacy ``model_spec`` string as a
compatibility shim for older callers.

That backend-registry layer is now present as well. ``SolveBackendRegistry``
and ``SolveBackendRecord`` provide the adapter seam, while the capability gate
checks objective shape, bounds, constraints, warm starts, and derivative hooks
before any backend executes. Unsupported features now raise an explicit
``UnsupportedSolveCapabilityError`` unless the caller opts into a named
fallback backend, and the solve-result metadata records both the backend that
ran and any fallback path that was taken.

Calibration governance now projects that execution state into stable
``SolveProvenance`` and ``SolveReplayArtifact`` payloads. Rates and SABR
calibrations expose backend identity, solver options, termination summary, and
residual diagnostics under consistent ``solver_provenance`` and
``solver_replay_artifact`` keys so audit, replay, and validation consumers do
not have to special-case each model family.

The SABR side now also has a reusable smile-assembly layer before the final
workflow wrapper. ``build_sabr_smile_surface(...)`` normalizes one supported
smile grid into stable labeled points, optional weights, and explicit warning
flags, while ``fit_sabr_smile_surface(...)`` returns the solved process plus
persistable fit diagnostics such as residuals, RMS errors, and ATM error. That
separates smile-input shaping from the later desk-facing workflow ticket and
gives replay/validation code one stable smile artifact to inspect.

The supported workflow surface now consumes that substrate directly.
``calibrate_sabr_smile_workflow(...)`` is the raw-input entry point for the
desk-facing SABR path, while ``calibrate_sabr(...)`` stays as the compatibility
wrapper for code that still wants a plain ``SABRProcess`` object.

The first supported Heston calibration workflow now follows the same pattern.
``build_heston_smile_surface(...)`` normalizes one equity-vol smile with spot,
rate, dividend-yield, strike labels, and weights, while
``fit_heston_smile_surface(...)`` lowers it onto the shared solve-request
substrate and returns governed solver artifacts plus a runtime-ready Heston
parameter payload. ``calibrate_heston_smile_workflow(...)`` is the raw-input
wrapper for that checked path.

Dupire local-vol calibration now follows the same explicit-artifact rule at the
workflow boundary. ``dupire_local_vol_result(...)`` keeps the callable surface,
grid metadata, warning list, and sampled instability diagnostics together so
runtime consumers can inspect when the Dupire numerator or denominator became
unsafe instead of inheriting a silent implied-vol fallback.

The supported local-vol workflow now also has an explicit MarketState handoff.
``calibrate_local_vol_surface_workflow(...)`` returns the hardened
``LocalVolCalibrationResult`` together with ``apply_to_market_state(...)`` so
later MC/lattice consumers can reuse the named local-vol surface instead of
threading a route-local callable through every integration point.

Calibration throughput now also has a checked benchmark surface in
``trellis.models.calibration.benchmarking``. The supported workflow fixtures
cover Hull-White, SABR, Heston, and local-vol runs, and the persisted report in
``docs/benchmarks/calibration_workflows.{json,md}`` keeps cold-start versus
warm-start timing baselines in a stable comparison shape. The benchmark folder
now carries its own ``README.md`` contract note so checked artifacts stay
portable and do not accumulate machine-local scratch metadata.

The rates market-input side now has a similarly typed bootstrap surface in
``trellis.curves.bootstrap``. ``BootstrapCurveInputBundle`` and
``BootstrapConventionBundle`` let later calibration workflows and
``resolve_market_snapshot(..._curve_bootstraps=...)`` describe named curve
inputs with explicit currency, rate-index, convention, and instrument metadata
instead of relying on one helper's hard-coded deposit/future/swap defaults.
That surface now has a first-class solve lane as well:
``build_bootstrap_solve_request(...)`` packages the ordered curve bundle as a
typed least-squares request, and ``bootstrap_curve_result(...)`` returns the
solved curve together with the raw ``SolveRequest``/``SolveResult`` pair,
governed solver provenance, replay artifact, and Jacobian diagnostics for the
instrument repricing surface.

The market resolver persists both sides of that assembly. Snapshot provenance
still records the named ``bootstrap_inputs`` bundle for each generated curve,
and now also records a sibling ``bootstrap_runs`` payload containing the solved
bootstrap artifacts and diagnostics. That keeps replay, validation, and later
calibration workflows aligned on the same named curve assembly record instead
of reconstructing bootstrap behavior from the final zero curve alone.

The rates-risk side now has a shared bucket-shock substrate in
``trellis.curves.shocks``. ``build_curve_shock_surface(...)`` records how
requested bucket tenors sit on the base curve, including explicit support and
wide-interval warnings for sparse interpolation spans, and
``CurveShockSurface.apply_bumps(...)`` materializes a shocked ``YieldCurve``
with inserted off-grid nodes where needed. ``YieldCurve.bump(...)`` and
``Session.with_tenor_bumps(...)`` now route through that substrate instead of
silently dropping non-knot bucket requests.

The volatility side now has the matching reusable substrate in
``trellis.models.vol_surface_shocks``. ``build_vol_surface_shock_surface(...)``
projects a supported implied-vol surface onto a requested expiry/strike grid,
records exact-node versus interpolated support metadata for each bucket, and
keeps approximation warnings stable for later risk consumers. The resulting
``VolSurfaceShockSurface.apply_bumps(...)`` materializes a bumped
``GridVolSurface`` on that configured grid so bucketed-vega and
volatility-scenario routes can share one surface-bump contract instead of
hand-rolling their own.

``trellis.analytics.measures.Vega`` now consumes that substrate directly. The
default ``vega`` request still returns one coarse scalar, but callers can pass
explicit ``expiries`` and ``strikes`` to get a nested expiry/strike risk
surface plus metadata describing the bucket grid, the resolved surface type,
and any approximation warnings such as ``interpolated_surface_bucket`` or
``flat_surface_expanded``.

The runtime measure surface now also fills the old spot-risk gap.
``trellis.analytics.measures.Delta`` and ``Gamma`` use explicit finite-
difference spot bumps against one selected runtime spot binding, while
``Theta`` rolls ``as_of`` and ``settlement`` forward by the requested calendar
step and reprices. The support boundary is explicit: delta/gamma now fail with
an actionable error when the runtime state has no unambiguous spot binding
instead of quietly disappearing from the output.

``trellis.analytics.measures.KeyRateDurations`` now consumes that same
substrate by materializing a bucketed view of the base curve on the requested
tenor grid before it reprices bucket-up and bucket-down scenarios. That keeps
KRD output conventions aligned with the shared risk substrate instead of mixing
exact-knot autodiff output with user-requested off-grid buckets. The direct
pricing adapters now reuse that same path as well, so ``Session.price(...)``,
``Session.greeks(...)``, and ``risk_report()`` all project numeric tenor-keyed
KRDs through the same runtime substrate as ``Session.analyze(...)``.

Bootstrap-backed discount curves now also expose a second rates-risk path.
When the active ``MarketState`` carries ``bootstrap_runs`` provenance for the
selected discount curve, ``KeyRateDurations`` and ``ScenarioPnL`` can switch to
``methodology="curve_rebuild"``. That path reconstructs the typed bootstrap
bundle, bumps the quoted deposit/future/swap instruments, rebuilds the curve,
and then reprices on the rebuilt state instead of shocking the final zero curve
directly.

Those risk results are now returned as dict-like objects with attached
``.metadata`` describing the resolved methodology, bucket convention, selected
curve name, scenario templates, and any explicit fallback reason. The runtime
contract is therefore self-describing even when two methodologies coexist.

Named rate scenario packs now sit on top of the same substrate as well.
``trellis.curves.scenario_packs.build_rate_curve_scenario_pack(...)`` produces
desk-style twist and butterfly templates with explicit bucket assumptions and
warning projections, while both ``trellis.analytics.measures.ScenarioPnL`` and
``Pipeline`` expand those templates into concrete ``tenor_bumps`` runs instead
of hand-assembling parallel-only shifts in each caller. Pack-only requests no
longer silently include the default parallel ladder; callers now opt back into
that mixed output with ``include_parallel_shifts=True``.

``Pipeline.run()`` now projects those per-scenario ``BookResult`` objects
through ``trellis.book.ScenarioResultCube`` instead of returning a bare
``dict``. The cube stays mapping-compatible for existing callers, but it also
keeps a stable ``scenario_specs`` map, preserves pipeline/scenario provenance
for each scenario, and exposes reusable ``book_ladder(...)`` /
``position_ladder(...)`` aggregation hooks for later attribution and summary
work. ``Pipeline.compile_compute_plan()`` now makes the scenario-batch plan
explicit before execution, and the returned cube keeps that serialized
``compute_plan`` payload so later saved-template and attribution flows can
reuse the same batch contract instead of reconstructing scenario loops ad hoc.
Named scenario templates can now be sourced from
``market_snapshot.metadata["scenario_templates"]`` and expanded through
``{"scenario_template": ...}`` entries in ``Pipeline.scenarios(...)``, while
``ScenarioResultCube.to_batch_output()`` projects a stable book-review payload
for the saved-template and attribution follow-on slices. The cube now also
exposes ``pnl_attribution()`` so pod-review callers can rank top position
contributors per scenario without reconstructing deltas outside the cube.
Concrete saved templates emitted from rebuild-based rates-risk results now
carry the preserved methodology and quote-bucket shock map needed to replay the
same quote-space curve rebuild through the pipeline. The batch-review payload
also now publishes ``book_pnl`` and ``position_pnl`` as delta surfaces in
``values`` while retaining the underlying scenario levels in
``metadata["levels"]``.

The runtime layer now also exposes the first reusable factor-state simulation
surface for future-value workflows. ``trellis.agent.family_lowering_ir``
contains the typed ``FactorStateSimulationIR`` boundary, and
``trellis.models.monte_carlo.simulation_substrate`` provides the current
runtime companions:

- ``simulate_factor_state_observations(...)`` for the observed factor grid
- explicit market projection ``M_t = Phi_t(X_t)``
- ``evaluate_conditional_valuation_paths(...)`` for intermediate-date value
  maps
- ``FutureValueCube`` for trade/date/path outputs

The first landed consumer is intentionally narrow: a vanilla interest-rate
swap future-value workflow under one-factor Hull-White. That substrate should
be treated as the new reusable base for later portfolio, netting, collateral,
and xVA work, not as evidence that those downstream analytics already exist.

Pod-risk throughput now has a checked benchmark surface as well.
``trellis.analytics.benchmarking`` measures the supported scenario-cube,
rebuild-based rates-risk, bucketed-vega, and spot-risk workflows through the
same public/runtime entrypoints that desk code uses, and persists the current
baseline under ``docs/benchmarks/pod_risk_workflows.{json,md}``. The same
folder-level contract applies there: checked benchmark payloads are repo
artifacts, not ad hoc local output dumps.

The first supported model-parameter calibration workflow now sits on top of the
same substrate. ``calibrate_hull_white(...)`` packages a supported strip of
swaption-style quotes as a typed least-squares request, reuses the selected
curve/bootstrap provenance from ``MarketState``, and emits a serializable
Hull-White parameter payload alongside the normalized solver artifacts. The
callable-bond, Bermudan-swaption, and ZCB-option helpers now resolve
``mean_reversion`` and ``sigma`` from explicit arguments first, then from
``market_state.model_parameters`` or ``model_parameter_sets`` before falling
back to the older heuristic defaults.

Callable-bond analytics now sit on top of that same callable-tree boundary.
``oas_duration`` and ``callable_scenario_explain`` are registered runtime
measures rather than pricing-service one-offs, so both ``Session.analyze(...)``
and the governed ``trellis.price.trade`` surface reuse the same callable
duration and scenario-explain contract.

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
- ``SnapshotService`` owns both reproducibility-bundle persistence and the
  file-based explicit market-snapshot import path used by the exotic-desk
  workflow
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

For the exotic-desk roadmap, the first checked trade-entry extension now spans
three desk-oriented rates slices: ``range_accrual``, ``callable_bond``, and
``bermudan_swaption``. ``TradeService`` can normalize both structured payloads
and trader-style prose onto the supported semantic contracts, carrying explicit
term fields such as reference index, coupon/range data, callable-bond terms,
and Bermudan exercise schedules in ``contract.product.term_fields`` while
returning deterministic ``missing_fields`` when the request is incomplete.

That same transport-neutral boundary now has a generic one-position import
wrapper in ``trellis.book_schema``. ``PositionImportContract`` carries the
stable imported-position shape, and ``TradeService.parse_position(...)`` can
normalize either flat row-style payloads or nested ``trade`` payloads onto that
contract while preserving the applied field map for later CSV/JSON loaders.
``ImportedBook`` and ``ImportedBookLoadResult`` extend that boundary to
mixed-book workflows, and ``TradeService.load_positions_csv(...)`` /
``load_positions_json(...)`` now provide the checked flat-file ingestion path
with per-row validation summaries and deterministic partial-load behavior.

The next desk-facing runtime slice is explicit file import for market
snapshots. ``SnapshotService`` now accepts a local manifest for named curves,
surfaces, FX, credit, spots, and fixing histories, persists the normalized
contract as a governed snapshot resource, and can activate that snapshot on a
session so ``PricingService`` reuses it directly instead of forcing a live
provider resolution hop.

Fixing histories are now a first-class part of that canonical market-data
surface. ``MarketSnapshot`` and ``MarketState`` carry named
``fixing_histories`` plus an optional ``default_fixing_history`` selection, and
the governed provider identity surface advertises ``fixing_history`` alongside
the other canonical market-data capabilities.

That imported-snapshot boundary now also has a request-driven selection layer.
``MarketSnapshot.resolve_request(...)`` and
``SnapshotService.resolve_market_state(...)`` accept explicit named component
requests plus optional named scenario-template ids, return a stable
``SnapshotSelectionResult``, and surface missing-component or stale-snapshot
diagnostics without silently falling back to another component.

That imported-snapshot path now feeds three checked desk routes as well.
``PricingService`` recognizes the approved ``range_accrual_discounted``,
``callable_bond_tree``, and ``bermudan_swaption_tree`` adapters. It resolves
the needed snapshot components, delegates pricing to the checked helper-backed
route modules, and persists route-specific validation bundles plus the
desk-review schedule projection on the canonical run record.

For rates optionality, that runtime boundary now also carries reusable model
parameters. A calibration workflow can enrich ``MarketState`` with calibrated
Hull-White parameters once, and later helper-backed pricing routes can consume
those parameters without every caller threading ``mean_reversion`` or ``sigma``
through ad hoc route-local arguments.

The same model-parameter surface now supports an explicit Heston runtime
binding. ``trellis.models.processes.heston.resolve_heston_runtime_binding(...)``
can resolve one runtime ``Heston`` process from explicit inputs or from
``MarketState.model_parameters`` / ``model_parameter_sets``, while recording
parameter-source provenance and warnings when ``mu`` has to be defaulted from
the discount curve under a zero-dividend assumption.

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
     - ``task_runtime.py``, ``scripts/*.py``, the split task manifests, and ``FRAMEWORK_TASKS.yaml``
     - Batch execution, reruns, benchmark/negative/canary workflows, and separate pricing-vs-framework task inventories

Read Next
---------

- :doc:`hosting_and_configuration`
- :doc:`audit_and_observability`
- :doc:`task_and_eval_loops`
- :doc:`../quant/index`
