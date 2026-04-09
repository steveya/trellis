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

The first step is now registry-backed instead of branch-order-driven. Semantic
drafting runs through ordered draft rules, then resolves a registered semantic
family plus a registered method surface. The request layer and semantic
compiler both reuse the same specialization authority when a different
preferred method is selected, which keeps admissible-method truth in one place
instead of repeating family-local branching in multiple lower layers.

Below that semantic boundary, the runtime now also treats family identity as an
authority contract. Once the request or compiled product summary knows a
specific family such as ``zcb_option`` or ``basket_option``, lower layers such
as static spec selection, cached-wrapper reuse, and helper binding are not
allowed to widen it back to a generic family like ``european_option`` unless
that widening is an explicitly declared refinement. This is what keeps the
lower stack aligned with the semantic/compiler boundary instead of letting
description-level heuristics silently override it.

The LLM is not in the pricing hot path. It participates in parsing, planning,
generation, review, and validation around the deterministic library.

Shipped Lowering Boundary
-------------------------

The current compiler boundary is:

.. code-block:: text

   SemanticContract
     + ValuationContext
     -> ProductIR
     -> EventProgramIR / ControlProgramIR
     -> family lowering IR
     -> helper-backed numerical route

The shipped family IRs are:

- ``AnalyticalBlack76IR``
- ``EventAwareMonteCarloIR`` as the new bounded single-state Monte Carlo family
  surface
- ``VanillaEquityPDEIR``
- ``ExerciseLatticeIR``
- ``CorrelatedBasketMonteCarloIR``
- ``CreditDefaultSwapIR``
- ``NthToDefaultIR``

This is intentionally not a flat universal IR. The current stack uses
``ProductIR`` as the shared checked summary, then emits one universal semantic
event/control program before narrowing into family IRs for the proven route
families. The shared compiler program consists of:

- ``EventProgramIR``
- ``ControlProgramIR``

Those objects are the canonical semantic authority for scheduled events,
exercise/call control, and same-day phase ordering. Family IRs then project
that shared program into their bounded numerical forms:

- lattice keeps the semantic schedule/control surface on ``ExerciseLatticeIR``
- PDE projects it into ``PDEEventTimeSpec`` / ``PDEEventTransformSpec``
- Monte Carlo projects it into ``MCEventTimeSpec`` plus reduced replay
  requirements

For Monte Carlo, the new event-aware family is now the typed
compiler/admissibility surface for bounded single-state schedule semantics as
well. European rate-style swaptions can already lower into
``EventAwareMonteCarloIR`` with explicit event buckets, replay requirements,
and payoff-reducer contracts. The runtime now also ships the matching bounded
problem-assembly layer in ``trellis.models.monte_carlo.event_aware``:

- ``EventAwareMonteCarloProcessSpec``
- ``EventAwareMonteCarloEvent``
- ``EventAwareMonteCarloProblemSpec``
- ``EventAwareMonteCarloProblem``

That runtime layer resolves process-family plugins, compiles deterministic
event buckets into reduced-state replay requirements, and assembles a
``StateAwarePayoff`` on top of the existing Monte Carlo path-state and
path-event substrate. The generic vanilla MC migration and the final
schedule-driven proof routes are still separate follow-on slices, but the
compiler no longer points at an event-aware family without a checked runtime
problem-spec boundary underneath it.

The first migrated vanilla cases now use that boundary directly:

- vanilla European Monte Carlo lowers onto ``EventAwareMonteCarloIR`` as a
  terminal-only ``gbm_1d`` family instance rather than a synthetic event-replay
  instance
- the vanilla Monte Carlo and transform wrappers now both compose a shared
  single-state diffusion resolver/GBM-support layer under
  ``trellis.models.resolution`` for settlement, maturity, spot, dividend,
  discount, vol, and characteristic-function binding
- the local-vol vanilla helper remains a checked route-level wrapper, but it
  now assembles and prices through ``trellis.models.monte_carlo.event_aware``
  instead of maintaining a separate Monte Carlo engine/payoff loop
- the copula basket-credit slice now also exposes a semantic-facing helper
  layer in ``trellis.models.credit_basket_copula`` so tranche-style CDO and
  nth-to-default requests can bind discount/credit inputs, tranche bounds, and
  dependence-family controls without exposing the raw scalar copula kernels as
  the public route helper

For rate-style swaption comparison builds, the semantic compiler now also keeps
the contract-level convention surface attached to each method-specific plan.
Fixed-leg and floating-leg day-count terms, rate-index bindings, and the
bounded Hull-White calibration/model contract now survive into the
method-specific ``ValuationContext`` and ``MarketBindingSpec`` instead of being
dropped when a multi-method comparison request is compiled.

For the helper-backed analytical, tree, and Monte Carlo swaption routes, the
runtime now also preserves those comparison-regime bindings when it materializes
deterministic exact wrappers. That means the exact helper calls carry the same
explicit Hull-White comparison parameters, and the Monte Carlo wrapper adds a
stable comparison-quality sampling control instead of drifting on an unseeded
default path.

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
- ``copula_loss_distribution`` on tranche-style basket-credit comparison tasks
  through the semantic-facing basket-credit helper surface

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

Below those public callable wrappers, the reusable coupon/event/control layer
now lives in ``trellis.models.short_rate_fixed_income``. Coupon schedule
compilation, embedded issuer/holder exercise semantics, straight-bond
reference PV, and generic lattice/PDE event assembly no longer live only in
``callable_bond_tree`` or ``callable_bond_pde``. That keeps the wrappers
stable while moving the reusable short-rate claim logic into a broader helper
surface for later fixed-income families.

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

Calibration quote maps now carry a broader quote-semantics authority as well.
``QuoteMapSpec`` still exposes the bounded top-level ``quote_family`` and
``convention`` fields for compatibility, but the authoritative contract now
includes a typed quote-semantics payload with:

- quote subject
- axis semantics
- unit semantics
- settlement / numeraire semantics

That means the runtime can distinguish not only that something is, for
example, an implied vol, but also whether it is a swaption or equity-option
quote, which axes identify one point on the quote surface, which unit the
quote uses, and which curve-role / settlement assumptions govern the quote
space. The shipped calibration families now use that same surface for rates,
credit, equity-vol, local-vol, and short-rate comparison regimes instead of
relying on ad hoc metadata keys.

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
