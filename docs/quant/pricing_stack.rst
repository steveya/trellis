Pricing Stack
=============

This page records the deterministic pricing stack that now sits underneath
``trellis.ask(...)`` and the semantic DSL boundary.

Layering
--------

.. list-table::
   :header-rows: 1
   :widths: 20 28 24 28

   * - Layer
     - Primary objects
     - Role
     - Main code paths
   * - Public workflow
     - ``trellis.ask``, ``Session``, ``Pipeline``
     - User-facing entry points for ask, direct pricing, and batch flows
     - ``trellis/__init__.py``, ``trellis/session.py``, ``trellis/pipeline.py``
   * - Semantic contract
     - ``SemanticContract`` and typed product semantics
     - Contract meaning independent of solver or market snapshot
     - ``trellis/agent/semantic_contracts.py``
   * - Valuation context
     - ``ValuationContext``, ``EngineModelSpec``, ``RequiredDataSpec``, ``MarketBindingSpec``
     - Market binding, model/measure policy, reporting, and requested outputs
     - ``trellis/agent/valuation_context.py``, ``trellis/agent/market_binding.py``
   * - Product summary and routing
     - ``ProductIR``, ``PricingPlan``, ``RouteSpec``, ``BuildGateDecision``
     - Checked summary, method choice, typed admissibility, and gatekeeping
     - ``trellis/agent/semantic_contract_compiler.py``, ``trellis/agent/route_registry.py``, ``trellis/agent/build_gate.py``
   * - Family lowering
     - family-specific lowering IRs + DSL lowering
     - Narrow typed lowering onto checked-in helper-backed routes
     - ``trellis/agent/family_lowering_ir.py``, ``trellis/agent/dsl_lowering.py``
   * - Numerical engines
     - analytical, lattice, PDE, Monte Carlo, transforms, copulas
     - Deterministic pricing math used by hand-written and agent-built routes
     - ``trellis/models/``
   * - Market and payoff runtime
     - ``MarketState``, ``Payoff``, ``PresentValue``
     - Immutable market inputs and common execution interfaces
     - ``trellis/core/market_state.py``, ``trellis/core/payoff.py``

Deterministic Flow
------------------

The current semantic pricing path is:

1. Normalize a request into a ``SemanticContract``.
2. Validate typed semantics, including phase order, obligations, and controller semantics.
3. Build a ``ValuationContext`` and compile ``RequiredDataSpec`` plus ``MarketBindingSpec``.
4. Build ``ProductIR`` and select a pricing method and candidate route.
5. Apply typed route admissibility through ``BuildGateDecision``.
6. Lower onto a family-specific IR and then onto a checked helper or kernel.
7. Execute the existing deterministic numerical code.

The LLM is not in the pricing hot path. It participates in parsing, planning,
generation, review, and validation around the deterministic library.

Shipped Lowering Boundary
-------------------------

The current compiler boundary is:

.. code-block:: text

   SemanticContract
     + ValuationContext
     -> ProductIR
     -> family lowering IR
     -> helper-backed numerical route

The shipped family IRs are:

- ``AnalyticalBlack76IR``
- ``VanillaEquityPDEIR``
- ``ExerciseLatticeIR``
- ``CorrelatedBasketMonteCarloIR``
- ``CreditDefaultSwapIR``
- ``NthToDefaultIR``

This is intentionally not a flat universal IR. The current stack uses
``ProductIR`` as the shared checked summary and then narrows into family IRs
for the proven route families.

Within the valuation layer, migrated calibration workflows now carry a bounded
``EngineModelSpec`` surface instead of relying only on a free-form
``model_spec`` string. The structured model spec captures model family/name,
potential/source semantics, backend hints, calibration requirements, and
explicit rates discount/forecast curve roles where applicable.

Current Proven Families
-----------------------

The end-to-end typed boundary is currently proven for:

- ``analytical_black76`` on vanilla options
- ``vanilla_equity_theta_pde`` on vanilla options
- ``exercise_lattice`` on callable bonds and Bermudan swaptions
- ``callable_bond_tree_v1`` on the first supported issuer-call bond slice
- ``bermudan_swaption_tree_v1`` on the first supported Bermudan swaption desk slice
- ``correlated_basket_monte_carlo`` on ranked-observation baskets
- ``range_accrual_discounted_cashflow_v1`` on the first single-index range-accrual note slice
- ``credit_default_swap_analytical`` and ``credit_default_swap_monte_carlo`` on single-name CDS
- ``nth_to_default_monte_carlo`` on nth-to-default basket credit

These route IDs and helper-backed numerical kernels are preserved. The new work
changes validation, binding, admissibility, and lowering, not the pricing math.
For single-name CDS comparison builds, the typed boundary now also carries a
comparison-quality ``n_paths`` control on the Monte Carlo spec so the helper
route can tighten internal agreement without changing the checked pricing
kernel.

The range-accrual route is intentionally narrow: it is a deterministic
discounted-cashflow adapter that prices coupon periods off explicit range
checks, imported fixing histories, and a forecast-curve proxy rather than a
generic exotics engine. The goal of this slice is a reviewable desk workflow,
not universal structured-note coverage.

The callable-rates desk slices follow the same philosophy. The callable-bond
and Bermudan-swaption adapters are thin checked wrappers over the stable
exercise-lattice helpers, with typed exercise schedules and trader-facing event
projection layered on top of the preserved numerical kernels rather than a new
generic exotics runtime. The callable-bond slice now also projects callable
analytics directly off that tree boundary: effective ``oas_duration`` plus a
callable-specific scenario ladder that compares callable price, straight-bond
reference price, and embedded call option value under parallel rate shocks.

Calibration Surface
-------------------

The deterministic pricing stack now has a sibling calibration surface rather
than a collection of route-local helper solves.

The calibration boundary is:

1. assemble a typed calibration target or smile/grid surface
2. lower onto ``SolveRequest`` plus ``ObjectiveBundle``
3. execute through the backend registry
4. persist ``solver_provenance`` and ``solver_replay_artifact``
5. hand the calibrated parameter or surface payload back onto ``MarketState``
6. validate the workflow against replay/tolerance fixtures and benchmark baselines

The currently supported calibration workflows are:

- flat Black rates-vol helpers
- Hull-White swaption-strip calibration
- SABR single-smile calibration
- Heston single-smile calibration
- hardened Dupire local-vol workflow

The calibration stack now also carries a checked validation and benchmark
surface. ``tests/test_verification/test_calibration_replay.py`` locks replay
contracts and fit tolerances for the supported synthetic fixtures, while
``docs/benchmarks/calibration_workflows.{json,md}`` records the cold-start and
warm-start throughput baseline for the supported workflows.

For multi-curve runtime projections, the pricing stack now treats selected
curve-role names as first-class replay metadata. The resolved
``selected_curve_names`` contract is carried from ``MarketState`` into the
runtime contract, copied onto task results and persisted run records, and
recovered by replay summaries from either direct trace context fields or the
nested ``runtime_contract.snapshot_reference`` payload.

That calibration provenance is now reused by the supported rates-risk stack as
well. Zero-curve bucket shocks remain the default lightweight KRD and
scenario-P&L path, but bootstrap-backed sessions can now request a
rebuild-based methodology that bumps quoted market instruments, rebuilds the
curve, and reprices on the rebuilt surface. Risk outputs disclose which
methodology actually ran through attached metadata rather than forcing callers
to infer it from context.

The volatility side now has the first matching substrate layer as well.
``trellis.models.vol_surface_shocks`` defines the reusable expiry/strike
bucket grid, support metadata, warning contract, and bumped-surface
materialization that bucketed-vega and later volatility-scenario routes reuse.
``trellis.analytics.measures.Vega`` now exposes the first runtime consumer of
that surface by returning expiry/strike bucket outputs when callers provide an
explicit bucket grid, while preserving the older scalar vega request when they
do not.

The broader runtime-measure layer now also covers spot delta, spot gamma, and
roll-down theta. These are intentionally finite-difference implementations with
explicit support boundaries rather than a full AAD platform: delta/gamma need a
selected spot binding, while theta is defined as one calendar-step repricing on
the existing runtime contract.

For portfolio workflows, the stack now has an explicit scenario-result
aggregation layer as well. ``Pipeline.run()`` returns a mapping-compatible
``ScenarioResultCube`` that stores both the per-scenario ``BookResult`` values
and the stable scenario/provenance metadata needed for downstream book explain.
The same workflow now has an explicit compiled batch plan through
``Pipeline.compile_compute_plan()``, and the resulting cube carries that
serialized ``compute_plan`` so later saved-template and attribution layers can
reuse the same scenario-batch contract. Named scenario-template ids can be
resolved from snapshot metadata during pipeline expansion, and the cube can now
project a stable ``to_batch_output()`` payload alongside reusable book-level
or position-level ladders. ``ScenarioResultCube.pnl_attribution()`` adds a
book-level explain layer on top of those ladders by ranking top position
contributors per scenario without losing which concrete shift template,
scenario pack, or pipeline settings produced each scenario result.

Those pod-risk workflows now also have a checked throughput baseline.
``trellis.analytics.benchmarking`` records scenario-cube execution,
rebuild-based rates sensitivities/scenarios, bucketed vega, and spot-risk
measure bundles in ``docs/benchmarks/pod_risk_workflows.{json,md}``, so later
runtime changes can be compared against an explicit desk-risk benchmark rather
than anecdotal timing claims.

Warning And Error Policy
------------------------

The current stack distinguishes three classes of outcomes:

- semantic validation errors
- route admissibility failures
- successful compilation with warnings

Warnings are used when legacy semantic mirrors are normalized or ignored for
migrated route families. Errors are used for invalid typed semantics, missing
required bindings, unsupported outputs, unsupported control styles, or
unsupported state tags.

Authority Rules
---------------

For the migrated families listed above:

- typed ``SemanticTimeline`` and ``ObligationSpec`` are authoritative for settlement semantics
- typed ``EventMachine`` is authoritative for automatic event semantics
- ``requested_outputs`` is the canonical output field
- ``requested_measures`` is a shim surface only

Legacy fields such as ``settlement_rule`` and ``event_transitions`` remain on
the semantic contract for non-migrated code paths, tracing, and compatibility,
but they are mirrors rather than the truth source for migrated routes.

Deferred Scope
--------------

The current stack does not yet include:

- a full desk-task DSL
- ordered sequential multi-controller protocols
- nonlinear funding or XVA semantics inside ``ValuationContext``
- a universal IR covering every solver family

Those remain future extensions after the typed semantic boundary is stable
across more route families.
