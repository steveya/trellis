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
   * - Backend binding catalog
     - ``BackendBindingSpec``, ``ResolvedBackendBindingSpec``, binding catalogs
     - Canonical exact helper/kernel/schedule/cashflow binding facts used by routing and validation
     - ``trellis/agent/backend_bindings.py``
   * - Family lowering
     - family-specific lowering IRs + DSL lowering
     - Narrow typed lowering onto checked-in helper-backed routes
     - ``trellis/agent/family_lowering_ir.py``, ``trellis/agent/dsl_lowering.py``
   * - Numerical engines
     - analytical, lattice, PDE, Monte Carlo, transforms, copulas
     - Deterministic pricing math used by hand-written and agent-built routes
     - ``trellis/models/``
   * - Market and payoff runtime
     - ``MarketState``, ``Payoff``, present-value scalar contract
     - Immutable market inputs and common execution interfaces
     - ``trellis/core/market_state.py``, ``trellis/core/payoff.py``

Deterministic Flow
------------------

The current semantic pricing path is:

1. Normalize a request into a ``SemanticContract``.
2. Validate typed semantics, including phase order, obligations, and controller semantics.
3. Build a ``ValuationContext`` and compile ``RequiredDataSpec`` plus ``MarketBindingSpec``.
4. Build ``ProductIR`` and select a pricing method plus candidate backend bindings.
5. Apply typed admissibility through ``BuildGateDecision`` and resolve the exact binding surface.
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
- ``TransformPricingIR`` as the bounded transform family surface
- ``VanillaEquityPDEIR``
- ``ExerciseLatticeIR``
- ``CorrelatedBasketMonteCarloIR``
- ``EventTriggeredTwoLeggedContractIR`` as the structural helper-backed family
  surface for event-triggered two-legged contracts, currently proven on
  single-name CDS
- ``NthToDefaultIR``

This is intentionally not a flat universal IR. The current stack uses
``ProductIR`` as the shared checked summary, then emits one universal semantic
event/control program before narrowing into family IRs for the proven route
families. The shared compiler program consists of:

- ``EventProgramIR``
- ``ControlProgramIR``

Family lowering is now binding-first as well: the compiler resolves the exact
binding surface before it emits these family IRs, and the family dispatch
logic keys off binding roles and exact helper/kernel symbols rather than
direct route-id branches.

DSL lowering now follows the same contract. Once a family IR or fallback
binding is selected, helper, pricing-kernel, schedule-builder, market-binding,
and control atoms are resolved from binding roles first. The lowering result
still keeps ``route_id`` as a transitional compatibility alias, but the
executable atom ids and missing-primitive diagnostics are now rooted in
``binding_id`` rather than route-local wording.

Those objects are the canonical semantic authority for scheduled events,
exercise/call control, and same-day phase ordering. Family IRs then project
that shared program into their bounded numerical forms:

- lattice keeps the semantic schedule/control surface on ``ExerciseLatticeIR``
- transforms project it into a terminal-only characteristic-function contract
  on ``TransformPricingIR``
- PDE projects it into ``PDEEventTimeSpec`` / ``PDEEventTransformSpec``
- Monte Carlo projects it into ``MCEventTimeSpec`` plus reduced replay
  requirements

For transforms, that now means route admissibility is family-first rather than
product-first. ``TransformPricingIR`` carries the transform-specific facts that
matter numerically:

- characteristic-function family
- terminal payoff kind
- strike semantics
- quote semantics
- backend capability (helper-backed vanilla diffusion versus raw-kernel
  stochastic volatility)

So a vanilla European payoff can still have holder-side semantic exercise
meaning upstream while the transform lane itself lowers onto an ``identity``
control contract with only terminal-state tags.

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
- vanilla American/Bermudan equity Monte Carlo now resolves the
  ``exercise_monte_carlo`` route to the exact helper
  ``price_american_equity_option_lsm_monte_carlo(...)``. Thin adapters may
  delegate to that helper for Longstaff-Schwartz path simulation and exercise
  control instead of reimplementing GBM path construction in generated code.
- equity variance-swap comparison targets now have a checked Monte Carlo
  realised-variance route in ``trellis.models.variance_swap``. The MC helper
  simulates annualised log-return variance under the market state's GBM
  surface binding and reports both price and fair strike variance, while the
  analytical route remains the log-contract style replication helper in
  ``trellis.models.analytical.equity_exotics``. Variance swaps deliberately
  skip the generic embedded-option flat-vega invariant; their proof contract is
  price sanity plus cross-method fair-strike/price comparison, not option vega.
- CEV European-vanilla proof targets have bounded helper surfaces for the
  retained legacy comparison task: ``price_cev_option_pde(...)`` composes the
  existing ``CEVOperator`` with the theta PDE solver, while
  ``price_cev_option_tree(...)`` provides the matching spot-lattice comparison
  route. These helpers are task-runner proof surfaces for explicit CEV
  comparison targets, not a replacement for the generic agent assembly path.
- Variance Gamma, CGMY, and Kou proof targets keep the same European vanilla
  option product shape but select ``model_family=variance_gamma``,
  ``model_family=cgmy``, or ``model_family=kou``. Transform/reference/MC
  comparison targets bind through ``trellis.models.levy_option`` and require
  explicit model parameters rather than inherited ``black_vol_surface``
  inputs. The CGMY MC target is a bounded terminal-distribution comparator
  from the characteristic function, and the Kou MC target is direct terminal
  double-exponential jump sampling; neither is a pathwise Levy simulator.
- the transform route uses that thin vanilla helper only for true
  ``equity_diffusion`` contracts; stochastic-volatility transform tasks such
  as Heston smile extraction now lower onto a checked Heston transform helper
  that resolves underlier spot plus explicit model parameters into the FFT/COS
  kernels instead of being forced through the single-state Black-vol helper
- stochastic-volatility Monte Carlo for European Heston vanilla options lowers
  onto a checked ``heston`` two-state helper, with explicit ``euler`` versus
  ``heston_qe`` scheme selection and the ``heston:monte_carlo`` validation
  bundle, instead of reusing the vanilla-equity GBM helper
- stochastic-volatility PDE for European Heston vanilla options now has
  bounded ADI binding and diagnostic scaffolding under
  ``trellis.models.pde.heston_adi``. The route binding is
  ``resolve_heston_adi_pde_inputs(...)`` plus
  ``price_heston_option_adi_pde_result(...)``. The scaffold consumes canonical
  ``kappa`` / ``theta`` / ``xi`` / ``rho`` / ``v0`` model parameters through
  the Heston runtime binding and keeps Black vol surfaces as
  market/calibration evidence rather than live Heston inputs. Optional
  transform references are recorded as diagnostics and do not replace the PDE
  scalar or the ADI input resolver. The variance grid uses a CIR
  moment-dispersion upper bound, so high-vol-of-vol Heston fixtures keep useful
  resolution around ``v0`` instead of spreading the grid out to an artificial
  ``v0 + xi * sqrt(T)`` scale.
- Heston calibration now has a bounded problem-IR adapter for single-expiry
  implied-vol smiles. Pricing routes consume explicit Heston model parameters
  from task specs, market state, synthetic fixtures, or recorded calibration
  results; a Black vol surface bump is not treated as a model-parameter bump
  unless a calibration problem records that bridge.
- Unsupported Heston Gauss-Laguerre transform targets now lower to a typed
  quadrature-transform blocker contract. The contract names the Heston
  characteristic-function binding, required model parameters, nodes/weights,
  damping or contour policy, stabilization requirements, diagnostics, and the
  missing quadrature kernel plus validation bundle.
- Bates-style affine jump stochastic-volatility European vanilla tasks now
  lower to explicit ``model_family=bates`` route bindings backed by
  ``trellis.models.bates_option``. The contract names the Heston base
  parameters, compound-Poisson lognormal jump parameters, checked transform and
  terminal Monte Carlo capabilities, and the jump-parameter validation
  requirements. Bates calibration, path-dependent payoffs, early exercise, and
  PDE/PIDE routes remain outside the checked boundary.
- SLV/LSV targets now lower to an explicit leverage-function contract. The
  contract names the local-vol and Black-vol surface authority, Heston model
  parameters, leverage-function surface, recorded leverage calibration
  provenance, diagnostics, and target-specific PDE or Monte Carlo solver
  requirements. Route binding remains fail-closed until those contracts and
  solvers exist.
- Path-dependent early-exercise Heston composites now lower to an explicit
  control blocker contract instead of a generic implementation gap. The
  contract names the missing path-state simulation, event monitor, payoff
  summary, early-exercise control policy, Heston path-state coupling, and the
  target-specific PDE/Monte Carlo/transform blocker. These tasks remain
  expected honest blocks until those abstractions and solvers exist.
- Single-barrier proof routes use
  ``trellis.models.single_barrier_option`` for zero-rebate Black-Scholes
  comparison targets. ``price_single_barrier_option_pde_result`` owns the
  bounded one-dimensional grid with an absorbing barrier boundary and far
  vanilla boundary; ``price_single_barrier_option_monte_carlo_result`` owns the
  GBM path simulation, one ``BarrierMonitor``, notional scaling, and
  deterministic discounting. Knock-in targets are derived by vanilla-minus-out
  parity.
- Double-barrier proof routes share
  ``trellis.models.analytical.support.barriers`` for lower/upper barrier specs,
  terminal payoff semantics, hit masks, and reduced-storage state payoffs.
  ``trellis.models.double_barrier_option`` now provides the checked
  ``price_double_barrier_option_pde_result`` and
  ``price_double_barrier_option_monte_carlo_result`` surfaces. The PDE helper
  owns the bounded Black-Scholes grid on ``[lower_barrier, upper_barrier]``,
  absorbing boundaries, and knock-in/out parity; the Monte Carlo helper owns
  the GBM engine binding, two barrier monitors, and deterministic discounting.
- Digital proof routes now include a bounded one-dimensional PDE helper,
  ``trellis.models.equity_option_pde.price_equity_digital_option_pde``. It
  prices cash-or-nothing and asset-or-nothing equity digitals with the shared
  Black-Scholes theta solver and optional Rannacher startup smoothing, so
  Crank-Nicolson/Rannacher comparison targets can delegate to a checked helper
  instead of regenerating discontinuous-terminal grid code.
- Arithmetic-Asian proof routes keep the numerical method split explicit:
  ``trellis.models.asian_option.price_arithmetic_asian_option_monte_carlo``
  owns the path-sampling MC comparison lane, while
  ``price_arithmetic_asian_option_analytical`` owns the Turnbull-Wakeman
  approximation lane. The analytical approximation is not treated as a Monte
  Carlo primitive just because it appears in an Asian multi-method task.
- Fixed-lookback MC proof routes use
  ``trellis.models.lookback_option.price_equity_fixed_lookback_option_monte_carlo``
  for the checked equity-diffusion comparison surface. The helper applies a
  Brownian-bridge extrema correction so the MC path summary is aligned with the
  continuous-monitoring fixed-strike analytical reference.
- Single-underlier autocallable proof routes now use
  ``trellis.models.autocallable.price_autocallable_monte_carlo_result`` as the
  checked MC/QMC event helper. It owns exact GBM path simulation, fixed
  observation-step mapping, first-trigger redemption, linear coupon accrual,
  terminal protection, and deterministic discounting. The same helper handles
  pseudo-MC and Sobol-QMC via the ``sampling`` argument, so Sobol is required
  only for QMC comparison targets.
- local-vol vanilla comparisons now use ``trellis.models.local_vol_option``
  over one ``LocalVolVanillaOptionSpec``.  The Dupire PDE side assembles the
  shared event-aware PDE substrate with ``operator_family=local_vol_1d`` and a
  supplied local-vol surface; the MC side delegates to
  ``trellis.models.monte_carlo.local_vol``.  This is a bounded European
  vanilla local-vol route and intentionally rejects nonzero dividend yield on
  the PDE side until the operator separates carry from discounting.
- FX vanilla and quanto routes now expose semantic-facing helper kits in
  ``trellis.models.fx_vanilla`` and ``trellis.models.quanto_option`` so the
  checked analytical and Monte Carlo adapters can stay as thin shells over
  shared market-binding and execution helpers rather than separate product
  implementations
- the copula basket-credit slice now also exposes a semantic-facing helper
  layer in ``trellis.models.credit_basket_copula`` so tranche-style CDO,
  nth-to-default, and portfolio loss-distribution requests can bind
  discount/credit inputs, tranche bounds or portfolio horizon, and
  dependence-family controls without exposing the raw scalar copula kernels as
  the public route helper
- bounded credit-index spread-option comparisons use
  ``trellis.models.credit_index_option``. The Black-on-spread helper and the
  antithetic lognormal MC helper share one ``CreditIndexOptionSpec`` carrying
  forward spread, strike spread, spread volatility, index annuity, discounting,
  and loss convention. This is a spread-option task helper, not an index-loss
  curve, tranche, or base-correlation model.
- capped/floored equity cliquet comparisons now have bounded analytical and
  Monte Carlo helper surfaces: the analytical path integrates reset returns
  with local/global caps and floors, while the Monte Carlo path delegates
  reset-date GBM increments, clipping, antithetic sampling, and discounting to
  ``trellis.models.monte_carlo.event_aware.price_equity_cliquet_option_monte_carlo``.
  These helpers support the explicit capped/floored comparison task surface;
  they do not imply arbitrary cliquet-book or path-contract coverage.

The developer-facing notation and task-triage lifecycle for these
stochastic-volatility buckets is maintained in
:doc:`../developer/stochastic_vol_computational_ir`. Use that guide when
writing new task fixtures, repair packets, or generated-adapter instructions.

Route selection now follows that same minimization rule. The deterministic
scorer no longer emits route-id or route-family one-hot authority; it ranks
routes from family capability, blocker state, and backend-binding facts such as
``route_helper`` / ``pricing_kernel`` / ``cashflow_engine`` surfaces instead.
Those exact helper/kernel facts are now materialized through
``trellis.agent.backend_bindings`` as a separate canonical binding catalog, and
the route registry derives its backend-binding authority summary from that
catalog rather than acting as the only source of exact backend identity.
The runtime plans now also carry that identity directly: ``PrimitivePlan`` and
``GenerationPlan`` persist ``backend_binding_id`` plus exact helper/kernel and
schedule-builder refs, so later validation, traces, and replay do not need to
reconstruct the binding contract from a route alias.
The validation surface now follows the same split: ``CompiledValidationContract``
keeps the generic validation pack id in ``bundle_id`` but also emits an
exact binding-scoped validation identity for exact-fit requests, so route ids
are no longer required as the primary exact-fit key in validation summaries,
route-binding authority packets, or downstream trace consumers.
Operator-facing binding wording now follows the same rule: display names,
short descriptions, and diagnostic labels are resolved from a dedicated
binding metadata catalog rather than route-card prose, and the compiled
route-binding authority packet carries that metadata for downstream
diagnostic surfaces. Platform traces, persisted task-run telemetry, and
task-diagnosis dossiers now render those binding-first labels directly
instead of reconstructing operator wording from route ids.
That exact-surface contract now also drives live plan construction, DSL
lowering, and semantic helper review: those paths resolve helpers, kernels,
and schedule builders from ``trellis.agent.backend_bindings`` first and only
fall back to route-card primitives when no binding surface exists.
The generated prompt-skill layer now follows the same contract: exact helper
and schedule constraints still surface when needed, but route-card notes are
kept as historical metadata rather than live ``route_hint`` authority, and
prompt ranking no longer gives first-class priority to exact ``route:<id>``
tag matches over broader family / method / instrument fit.
The experimental offline route-learning scaffold now reuses that same
minimized feature surface instead of emitting its own route-id or
route-family one-hots, so future retraining work cannot silently regress the
live scorer contract.

That architectural migration should still not be overstated. The checked proof
closeout in ``doc/plan/done__binding-first-exotic-proof-closeout.md`` and
``docs/benchmarks/binding_first_exotic_proof_closeout.json`` now certifies the
agreed ``11``-task binding-first exotic proof cohort end to end, including the
honest-block sentinel. That is a real support-contract step up, but it is
still a bounded cohort claim, not a blanket statement of arbitrary
constructable-exotic support.

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
- ``callable_range_accrual_deterministic_v1`` on the bounded issuer-callable
  single-index range-accrual proof slice
- ``credit_default_swap`` on single-name CDS across analytical and Monte Carlo
  bindings, routed through the structural
  ``event_triggered_two_legged_contract`` family
- ``nth_to_default_monte_carlo`` on nth-to-default basket credit
- ``copula_loss_distribution`` on tranche-style basket-credit comparison tasks
  through the semantic-facing basket-credit helper surface

For those credit and copula routes, the route cards are now intentionally thin.
They preserve backend binding, admissibility, validation ownership, and canary
provenance, but no longer carry procedural guidance about schedule-step
survival updates, copula initialization, or tranche-loss assembly when the
checked helper surface already owns that construction.

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

The callable range-accrual proof route keeps that same static conditional
accrual base and adds a deterministic issuer-call wrapper over projected
cashflows. It does not claim stochastic callable range-accrual valuation or
interrupted/barrier-state range-accrual execution.

The callable-rates desk slices follow the same philosophy. The callable-bond
and Bermudan-swaption adapters are thin checked wrappers over the stable
exercise-lattice helpers, with typed exercise schedules and trader-facing event
projection layered on top of the preserved numerical kernels rather than a new
generic exotics runtime. The callable-bond slice now also projects callable
analytics directly off that tree boundary: effective ``oas_duration`` plus a
callable-specific scenario ladder that compares callable price, straight-bond
reference price, and embedded call option value under parallel rate shocks.
The same route-thinning rule now also applies to the short-rate ZCB-option
cohort. QUA-915 collapsed the Jamshidian analytical and Hull-White tree
routes into a single pattern-keyed route ``short_rate_bond_option`` whose
``conditional_primitives`` dispatch on method selects the Jamshidian or
lattice helper. The route card still carries no lattice-construction or
short-rate-input assembly instructions because the checked helper surface
already owns that work.

Plain zero-coupon bond comparison tasks under Vasicek or CIR use a separate
``short_rate_zero_coupon_bond`` route. That route narrows the product to
``instrument=short_rate_bond`` and ``payoff_family=discount_bond`` so it does
not reuse ZCB-option or generic rate-tree artifacts. Analytical targets call
``price_short_rate_zero_coupon_bond_analytical(...)`` and rate-tree targets
call ``price_short_rate_zero_coupon_bond_tree(...)`` from
``trellis.models.short_rate_bond``. The exact task binding may opt into
benchmark defaults for sparse proof manifests, but the reusable helper itself
does not treat equity/Heston model payloads or generic Black-vol surfaces as
short-rate model parameters.

The analytical / PDE / FFT support cohort now follows the same rule. The
helper-backed Black76 swaption routes, the vanilla-equity PDE helper, bounded
CEV PDE/tree proof helpers, bounded event-aware PDE helper branches, Heston ADI
diagnostic scaffold, double-barrier payoff primitives, and vanilla-equity
transform helper keep backend binding, admissibility, and validation ownership
explicit. Exact-helper validation enforces the thin ``(market_state, spec,
...)`` call surface only for true checked route helpers; primitive-only
supports remain available for agent-written route assembly rather than
bypassing it.

For schedule-driven cap/floor strips lowered onto ``analytical_black76``, the
typed schedule state now carries admissibility directly. Structural
caplet/floorlet strips no longer get rejected as generic ``automatic`` event
routes when the lowered family IR is already the checked analytical strip
surface.

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
- bounded rates + equity/FX quanto-correlation calibration

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

The Monte Carlo side now also has the first broader factor-state future-value
substrate. ``trellis.agent.family_lowering_ir.FactorStateSimulationIR`` names
the typed contract for latent state, projected market views, observation
programs, and conditional valuation, while
``trellis.models.monte_carlo.simulation_substrate`` provides the runtime
companions ``simulate_factor_state_observations(...)``,
``evaluate_conditional_valuation_paths(...)``, and the emitted
``FutureValueCube`` / ``FutureValueCubeMetadata`` surface.

The first checked proof path is intentionally narrow: vanilla interest-rate
swap positions and shared-path swap portfolios under one-factor Hull-White.
Those workflows now emit

.. math::

   C_{a,i,n} = V_a^{clean}(t_i^+, X_{t_i}^{(n)})

on either the per-trade floating-boundary grid or the shared portfolio union
of floating-boundary dates, always with explicit ``post_event`` phase
semantics. The cube remains upstream of institutional counterparty analytics:
supported values are pre-netting, pre-collateral, and
pre-``CVA``/``DVA``/``FVA``.

The execution layer now has its first bridge onto that substrate as well.
For the admitted fixed-float IRS cohort,
``trellis.execution.visitors.simulation_bridge`` can:

- compile the route-free execution artifact back onto ``SwapSpec``
- project the same execution artifact onto ``FactorStateSimulationIR``
- emit the same checked ``FutureValueCube`` through
  ``build_future_value_cube_from_execution_ir(...)``

That closes the first repricing/future-value reuse loop at the execution seam
without claiming generic static-leg or xVA-style exposure coverage.

The execution seam now also has the first aggregation-oriented precursor passes
on top of that bridge:

- ``summarize_discounted_execution_ir(...)`` for deterministic present-value
  rolls and schedule-aware summary output
- ``summarize_future_value_execution_ir(...)`` for expected-value and
  exposure-shape summaries backed by the same execution-fed
  ``FutureValueCube``

These are explicit reporting precursors, not a claim that Trellis now has a
full netting, collateral, or xVA engine.

The institutional counterparty layer now has its first semantic representation
for later consumers. ``trellis.analytics.counterparty`` defines frozen
``CollateralAgreement``, ``NettingSet``, and
``CounterpartySemanticContract`` value types, plus
``validate_counterparty_semantic_contract(...)`` for explicit missing-field and
warning behavior. This is a governed representation of collateral agreements,
netting-set membership, closeout convention, and downstream runtime axes. It
also provides ``project_collateral_state(...)`` to produce a bounded
``CollateralStateProjection`` from a ``FutureValueCube`` for one netting set.
Collateral balance is based on valuation-lagged netted values, while closeout
values are read from the first observation date on or after the margin-period
horizon. ``aggregate_netting_set_exposures(...)`` then assembles a
``NettingSetExposureCube`` with one netting-set/date/path tensor, collateral
balances when supplied, and closeout-ready input packets for later exposure and
xVA consumers. ``compute_exposure_metrics(...)`` now produces the first stable
``EE`` curve, trapezoidal ``EPE``, and ``PFE`` quantile curves at portfolio and
per-netting-set levels. ``price_counterparty_xva(...)`` consumes the same
semantic contract and exposure stack to compute bounded flat-hazard
``CVA``/``DVA``/``FVA`` outputs under an explicit ``XVAAssumptionSet``.
Task-facing IRS helpers now bind the supported vanilla swap future-value cube
directly into that stack:
``price_interest_rate_swap_cva_monte_carlo(...)`` and
``price_interest_rate_swap_cva_analytical_approx(...)`` share the same
flat-hazard exposure integration contract, while
``price_interest_rate_swap_wrong_way_cva(...)`` applies a bounded pathwise
default-intensity tilt against ``price_interest_rate_swap_independent_cva(...)``
for wrong-way-risk comparisons. This is not a full enterprise counterparty-risk
platform: ``MVA``/``KVA``, stochastic credit curves, capital models, legal
enforceability workflows, and funding desk integration remain outside the
checked contract.

Those pod-risk workflows now also have a checked throughput baseline.
``trellis.analytics.benchmarking`` records scenario-cube execution,
rebuild-based rates sensitivities/scenarios, bucketed vega, and spot-risk
measure bundles in ``docs/benchmarks/pod_risk_workflows.{json,md}``, so later
runtime changes can be compared against an explicit desk-risk benchmark rather
than anecdotal timing claims.

The institutional exposure path has a separate checked benchmark pack under
``docs/benchmarks/counterparty_exposure_workflows.{json,md}``. It measures the
supported shared-path IRS future-value cube plus the warm-started netting /
collateral / ``EE``-``EPE``-``PFE`` reduction flow, with fixture metadata for
path count, step count, and warm-start assumptions.

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
