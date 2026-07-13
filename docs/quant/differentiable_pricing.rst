Differentiable Pricing
======================

Trellis promotes autograd only where it has a clear payoff:

- closed-form pricing kernels that currently drive real Greeks and calibration
- cap/floor strips and FX/quanto analytics built on top of those kernels
- flat-vol Vega extraction in ``Session.analyze()``
- bounded book-level reverse-mode risk for supported bond books and flat-vol
  vanilla equity option books
- graph-backed scalar hybrid derivative lanes for the bounded single-name
  quanto scalar-coordinate vector
- a bounded graph-backed smooth path-summary VJP lane for arithmetic-average
  Asian options over one flat-vol coordinate
- a bounded graph-backed smooth-interior early-exercise VJP lane for vanilla
  American/Bermudan options over one flat-vol coordinate
- curve bootstrap calibration, where the repricing Jacobian is now traced from
  the public repricing map instead of approximated inside the solver
- SABR calibration, where a gradient is more useful than repeated finite-difference sweeps
- binomial/trinomial tree pricing for smooth payoffs when the tree state is built
  from autograd-aware inputs
- Monte Carlo pathwise pricing when shocks are supplied explicitly and the
  payoff stays on the smooth side of the kink

FX vanilla under Garman-Kohlhagen is the clearest example of the basis
assembly engine in action: spot FX and domestic/foreign discount factors are
bridged to a forward, the option is valued from Black76 asset-or-nothing and
cash-or-nothing basis claims, and the terminal vanilla payoff is assembled with
``terminal_vanilla_from_basis(...)``. The checked helper-facing entrypoint for
that route is now ``trellis.models.analytical.fx.garman_kohlhagen_price_raw(...)``,
so the decomposition stays explicit without asking every adapter to rebuild the
same payoff algebra inline.

The reusable closed-form support surface now lives under
``trellis.models.analytical.support``. Raw analytical helpers such as
``trellis.models.analytical.fx.garman_kohlhagen_price_raw``,
``trellis.models.analytical.quanto.price_quanto_option_raw``,
``trellis.models.analytical.jamshidian.zcb_option_hw_raw``,
``trellis.models.rate_style_swaption.price_swaption_black76_raw``, and the
barrier route kernels compose that support layer rather than re-implementing
the same discounting, forward, and terminal-payoff glue in each route.

For quanto, the raw kernel is intentionally paired with a stricter runtime
resolver boundary: ``resolve_quanto_inputs(...)`` now records per-input
provenance for the underlier spot, FX spot, domestic curve, foreign curve,
volatility lookups, and correlation, and it only accepts noncanonical foreign
carry reuse when an explicit ``quanto_foreign_curve_policy`` bridge is present.

The first bounded hybrid calibration slice sits on top of that resolver
boundary. ``calibrate_quanto_correlation_workflow(...)`` calibrates one scalar
underlier/FX correlation for the checked ``bounded_quanto_correlation`` route,
but the shipped solve provenance is intentionally finite-difference today:
``resolved_derivative_method="scipy_2point_residual_jacobian"``,
``derivative_method_category="finite_difference_bump"``,
``derivative_method_support="fallback"``, and no ``backend_operator``. That is
a governed bounded hybrid calibration contract, not a claim of universal
hybrid AD, hybrid ``jvp``, or broad ``portfolio_aad`` support.

A separate bounded hybrid derivative prototype now sits on the same resolver
boundary. ``resolve_quanto_inputs(..., include_hybrid_factor_graph=True)`` can
attach a ``HybridFactorGraph`` describing the resolved quanto dependencies, and
``differentiate_quanto_scalar_correlation(...)`` differentiates the graph-owned
scalar underlier/FX correlation coordinate with a VJP. The scalar chart supports
both constrained ``rho`` and unconstrained ``x`` coordinates under
``rho = tanh(x)``. ``differentiate_quanto_scalar_inputs(...)`` widens that
same bounded route to a sparse VJP vector over supported graph-owned scalar
coordinates: underlier spot, FX spot, domestic/foreign curve zero-rate nodes,
flat/grid vol nodes, and scalar correlation. The same helper now accepts
``HybridDerivativeRequest(derivative_method="hvp", hvp_direction=...)`` for a
bounded directional HVP, ``H @ v``, over that scalar-coordinate chart.
Correlation matrix requests can now carry a checked PSD chart-policy payload
with deterministic off-diagonal factor coordinates and validation diagnostics,
and ``differentiate_quanto_correlation_matrix(...)`` provides a bounded VJP
and directional HVP lane for the active off-diagonal underlier/FX matrix
coordinate in the terminal quanto route, provided the matrix is
well-conditioned and away from the PSD boundary.
``differentiate_arithmetic_asian_path_summary(...)`` is the first executable
smooth path-summary lane: it reuses the bounded lognormal moment-matching
arithmetic-Asian policy, returns
``hybrid_path_summary_vjp`` metadata, and differentiates one graph-owned
``FlatVol`` coordinate. Correlation surfaces, matrix projection or repair,
grid-vol path summaries, discontinuous event monitors, dynamic state,
path-summary HVP, and hybrid ``jvp`` requests remain fail-closed rather than
approximated. ``differentiate_vanilla_early_exercise(...)`` adds the bounded
smooth-interior early-exercise lane for vanilla American/Bermudan options over
one graph-owned ``FlatVol`` coordinate; it reports
``hybrid_early_exercise_vjp`` metadata and fails closed for grid-vol
early-exercise, exercise-boundary ties, HVP, and JVP.
``fail_closed_dynamic_state_derivative(...)`` materializes the corresponding
``DynamicContractIR`` dynamic-state boundary as a first-class unsupported
``HybridDerivativeResult``. It builds a graph-owned discovery-only
``dynamic_state_policy`` chart, preserves ``semantic_state_policy`` metadata,
keeps selected-factor diagnostics deterministic, and still reports empty risk
through ``unsupported_hybrid_structure`` or ``unsupported_hybrid_jvp`` rather
than attempting state/control replay AD.
The semantic admission layer now carries a ``HybridADStatePolicy`` for
path-dependent, discontinuous-event, early-exercise, and dynamic-state hybrid
shapes. Supported arithmetic-average VJP admissions carry that payload as a
supported state policy, and supported vanilla early-exercise VJP admissions
carry it as a supported hard-exercise-projection policy. Blocked shapes copy
the payload into runtime fail-closed metadata as ``semantic_state_policy``.
Admission also accepts a market-parameterization label. Grid-vol path-summary
and grid-vol early-exercise state/control requests receive deterministic
``grid_node_vols`` factor requirements, planned state-policy payloads, and no
runtime helper, so the metadata is a support-boundary record rather than an
execution claim.
The matching coordinate contract is
``MarketObjectCoordinateChart.grid_vol_state_control_policy(...)``. It records
the active grid-vol node ``RiskFactorId`` keys, interpolation basis, locality
policy, selected-factor policy, and fail-closed reasons for missing surfaces,
unsupported interpolation, unsupported selected factors, event monitors, and
exercise-boundary kinks.
When the arithmetic path-summary runtime sees a grid-vol surface, it now
returns a first-class unsupported ``HybridDerivativeResult`` with that policy
chart and an ``unsupported_grid_vol_interpolation`` dependency instead of
falling back to a flat-vol-like graph. Known selected grid-vol nodes and
missing selected nodes are reported deterministically, but no node VJP is
executed. Grid-vol path-summary HVP and JVP requests use the same fail-closed
policy surface, so unsupported second-order or forward-mode requests still
carry the grid node coordinate chart instead of degrading to a generic flat-vol
diagnostic. Grid-vol early-exercise runtime requests use the same chart family
with ``lane_family="early_exercise_control"`` and fail closed through the
planned hard-exercise-projection control policy; HVP and JVP requests preserve
that policy metadata as well. This remains distinct from flat-vol
exercise-boundary kink diagnostics.
These are bounded state-summary/control lanes, not broad pathwise or dynamic
hybrid AD execution.

The bounded hybrid composition surface is now represented by
``HybridADMultiProductRequest``, ``HybridADMultiProductLaneResult``,
``HybridADMultiProductResult``, and
``aggregate_hybrid_ad_lane_results(...)``. This helper composes already-run
lane-local ``HybridDerivativeResult`` objects: it sums supported sparse VJP
risk vectors by ``RiskFactorId``, scales lane value and risk by explicit
quantities, preserves lane-level semantic admission and derivative-method
metadata, and records unsupported lanes as structured diagnostics. It is not a
cross-lane tape. Unsupported lanes can either coexist with supported lanes
under ``unsupported_lane_policy="collect_supported"`` or suppress aggregate
risk/value under ``unsupported_lane_policy="fail_closed"``.

The goal is not to make every numerical routine differentiable. It is to remove
unnecessary bump/reprice loops, stabilize calibration, and keep the exact same
pricing logic available to both value and sensitivity workflows.

For the mathematical methodology behind these support boundaries, including
the VJP/HVP notation, sparse portfolio-AAD aggregation model, and hybrid
factor-graph coordinate policy, see :doc:`ad_aad_methodology`.

Where Autograd Helps
--------------------

The checked support contract is intentionally explicit:

.. list-table::
   :header-rows: 1

   * - Surface
     - Supported derivative lane
     - Boundary
   * - Public payoff valuation
     - ``PricingValue`` scalar preserved through ``evaluate(...)`` and
       ``price_payoff(...)``
     - smooth payoffs only; reporting/serialization may still coerce to
       ``float``
   * - Public curves
     - ``YieldCurve`` and ``CreditCurve`` node sensitivities through
       ``autodiff_public_curve``
     - node-value AD; query-location AD is only piecewise away from knots
   * - Public vol surfaces
     - ``GridVolSurface`` node sensitivities and bucketed runtime vega through
       ``surface_bucket_bump``
     - node-value AD for the object, explicit bucket bumps for runtime surface
       risk
   * - Book-level reverse mode
     - reverse-mode curve risk for supported bond books through
       ``trellis.book.portfolio_aad_curve_risk(...)`` and
       ``Session.risk_report(...)``; reverse-mode vol risk for bounded
       vanilla equity option books through
       ``trellis.book.portfolio_aad_equity_option_vol_risk(...)``
     - bounded to supported bond positions on a shared ``YieldCurve`` and
       European call/put option specs on one shared ``FlatVol`` or
       ``GridVolSurface`` node grid; factorized risk uses stable
       ``RiskFactorId`` coordinates and ``RiskAggregationMap`` bucket totals,
       while unsupported positions are listed in metadata and excluded from
       the reverse-mode aggregate
   * - Hybrid factor graph
     - scalar quanto VJP through
       ``trellis.analytics.differentiate_quanto_scalar_correlation(...)`` and
       ``trellis.analytics.differentiate_quanto_scalar_inputs(...)``; bounded
       matrix-coordinate VJP/HVP through
       ``trellis.analytics.differentiate_quanto_correlation_matrix(...)``;
       bounded arithmetic-average path-summary VJP through
       ``trellis.analytics.differentiate_arithmetic_asian_path_summary(...)``;
       bounded vanilla early-exercise VJP through
       ``trellis.analytics.differentiate_vanilla_early_exercise(...)``; bounded
       dynamic-state fail-closed diagnostics through
       ``trellis.analytics.fail_closed_dynamic_state_derivative(...)``; bounded
       multi-product composition through
       ``trellis.analytics.aggregate_hybrid_ad_lane_results(...)``
     - bounded to the single-name quanto graph-owned scalar/matrix coordinates
       and one arithmetic-Asian flat-vol path-summary coordinate plus one
       vanilla American/Bermudan flat-vol early-exercise coordinate; the
       multi-product helper aggregates only completed lane-local results and
       keeps unsupported lanes explicit. Broad hybrid product graphs,
       executable correlation surfaces, matrix projection/repair,
       PSD-boundary behavior,
       grid-vol path summaries, discontinuous event monitors, dynamic state
       execution beyond the typed fail-closed policy result, grid-vol or
       boundary-kink early-exercise derivatives,
       path-summary/early-exercise HVP or JVP, and hybrid ``jvp`` remain
       fail-closed unless a future lane explicitly supports them.
   * - Flat volatility risk
     - scalar vega through ``autodiff_flat_vol``
     - flat surfaces only
   * - Calibration
     - rates bootstrap ``autodiff_vector_jacobian``, bounded quanto-correlation
       calibration ``scipy_2point_residual_jacobian``, SABR
       ``autodiff_scalar_gradient``, and Heston smile / full-surface
       ``finite_difference_vector_jacobian``
     - solver provenance records the derivative method that actually ran;
       bounded hybrid calibration is governed fallback finite differences, not
       hybrid AD or AAD
   * - Monte Carlo
     - pathwise gradients through ``simulate_with_shocks(...)`` and
       ``price_event_aware_monte_carlo(...)``
     - explicit shocks plus smooth terminal/snapshot/event-replay contracts

The backend capability surface lives in ``trellis.core.differentiable``.
``get_backend_capabilities()`` currently reports ``backend_id="autograd"`` and
the executable operator truth table ``grad=True``, ``jacobian=True``,
``hessian=True``, ``vjp=True``, ``hessian_vector_product=True``,
``jvp=False``, and ``portfolio_aad=False``. The ``vjp`` wrapper returns the
primal value plus a pullback closure for vector-valued smooth functions, and
the bounded book-level reverse-mode lanes now build on that surface for
supported curve-linked bond books, flat/grid-vol vanilla option books,
smooth-interior early-exercise options, arithmetic-Asian path summaries,
scalar quanto-correlation books, and explicitly configured mixed supported
books. The capability flag stays ``portfolio_aad=False`` because the supported
contract is still book-specific rather than a claim of universal portfolio-AAD
coverage.

The same capability payload now exposes an auditable ``support_matrix`` keyed
by backend operator name. Each entry is available through
``operator_support(...)`` and records ``supported``, ``backend_id``,
``array_namespace``, and ``unsupported_reason``. The payload also carries
``unsupported_reasons`` for disabled hooks. Today the ``jvp`` record is
unsupported because stock ``autograd.make_jvp`` has no checked
pricing-primitive coverage for routes that depend on ``norm.cdf``; the
``portfolio_aad`` record is unsupported because book-level AAD remains a set
of bounded lanes rather than one universal backend operator.

The graph-backed scalar quanto derivative lanes also use executable ``vjp``
and scalar-objective ``hessian_vector_product`` operators, but they report
through ``hybrid_scalar_vjp``, ``hybrid_scalar_vector_vjp``, or
``hybrid_scalar_vector_hvp`` metadata rather than widening the backend
capability table. The bounded matrix-coordinate lane reports through
``hybrid_matrix_vector_vjp`` and ``hybrid_matrix_vector_hvp`` metadata for the
terminal quanto active off-diagonal coordinate. These are bounded single-name
quanto lanes, not a general ``hybrid_ad=True`` backend claim. Correlation
matrix chart policy validation remains the governance boundary: valid PSD
matrix payloads can be represented with deterministic off-diagonal factor
coordinates, but matrix requests near the PSD boundary, projected/repaired
matrix charts, and surface correlation AD still fail closed.
The smooth path-summary lane reports through ``hybrid_path_summary_vjp`` and
is similarly bounded: only arithmetic-average European call/put contracts over
one ``FlatVol`` coordinate are executable. Other path-dependent and dynamic
hybrid requests have a separate semantic state policy boundary:
discontinuous event monitors, grid-vol path summaries, and DynamicContractIR
state/control requests are classified before runtime AD executes and remain
planned or unsupported with typed fail-closed metadata. The bounded
early-exercise lane reports through ``hybrid_early_exercise_vjp`` and is
limited to vanilla American/Bermudan call/put contracts over one ``FlatVol``
coordinate under the hard exercise-projection smooth-interior policy.
Grid-vol early-exercise, exercise-boundary ties, HVP, and JVP fail closed.

``hessian_vector_product`` returns an exact reverse-over-reverse HVP for
scalar-objective functions on smooth-interior regions. It is not a claim about
branch singularities, discontinuous payoffs, or vector-valued objectives. For
``vjp`` and ``hessian_vector_product``, tuple-valued unary primals are preserved
by default; callers must opt into n-ary positional dispatch with
``unpack_primals=True``.

``jvp`` stays fail-closed for now. Although stock ``autograd`` exposes
``make_jvp``, it does not define a JVP rule for pricing primitives Trellis
depends on, including ``autograd.scipy.stats.norm.cdf`` (``norm.cdf``). Trellis
therefore reports ``jvp=False`` until it either owns the missing normal-CDF
pricing primitive rule or adopts a backend with complete forward-mode coverage.
Call sites should use ``require_capability(...)`` rather than assuming those
operators exist.

The important contract is that the capability payload is executable truth, not
roadmap language. If an operator is reported as supported, its public wrapper
must compute checked values. If it is not supported, the wrapper fails closed
with ``NotImplementedError`` before a caller can accidentally rely on an
unstable derivative path. That gives future backend work a stable target while
keeping today's ``autograd`` boundary honest.

- Black76 and Garman-Kohlhagen calls/puts
- FX vanilla pricing can be assembled explicitly from Black76 terminal basis
  claims via ``terminal_vanilla_from_basis(...)`` after mapping spot FX and
  domestic/foreign discount factors to a forward.
- cap and floor strip valuation
- quanto adjustment and resolved quanto pricing inputs
- the T09 barrier kernel pack for the down-and-out / down-and-in call route
- raw resolved-input analytical helpers such as
  ``trellis.models.analytical.fx.garman_kohlhagen_price_raw``,
  ``trellis.models.analytical.jamshidian.zcb_option_hw_raw``,
  ``trellis.models.rate_style_swaption.price_swaption_black76_raw``,
  ``trellis.models.analytical.quanto.price_quanto_option_raw``, and
  ``trellis.models.analytical.barrier.down_and_out_call_raw`` /
  ``trellis.models.analytical.barrier.down_and_in_call_raw``
- public ``YieldCurve`` / ``CreditCurve`` node-value sensitivities and
  ``GridVolSurface`` node-value sensitivities
- runtime rate-risk extraction on public ``YieldCurve`` node grids, with
  ``resolved_derivative_method="autodiff_public_curve"`` recorded on the
  resulting analytics outputs
- bounded book-level reverse mode for supported bond books on a shared
  ``YieldCurve``, with factorized sparse risk metadata and unsupported
  positions excluded explicitly in metadata
- bounded book-level reverse mode for European vanilla equity call/put books on
  a shared ``FlatVol`` or ``GridVolSurface`` through
  ``trellis.book.portfolio_aad_equity_option_vol_risk(...)``, verified against
  independent central finite-difference bump/reprice for scalar flat-vol and
  grid-node sensitivities
- bounded book-level reverse mode for American/Bermudan vanilla equity
  call/put books over a shared ``FlatVol`` under a hard exercise-projection,
  smooth-interior policy; positions close to continuation/intrinsic ties fail
  closed instead of reporting unstable AAD Greeks
- bounded book-level reverse mode for arithmetic-Asian European call/put books
  over a shared ``FlatVol`` through
  ``trellis.book.portfolio_aad_arithmetic_asian_vol_risk(...)`` using a
  lognormal moment-matching smooth path-summary derivative policy
- bounded book-level reverse mode for single-name quanto option books over one
  scalar underlier/FX correlation through
  ``trellis.book.portfolio_aad_quanto_correlation_risk(...)``, with canonical
  ``model_parameter`` / ``correlation`` risk-factor coordinates
- graph-backed scalar quanto-correlation VJP through
  ``trellis.analytics.differentiate_quanto_scalar_correlation(...)``, with a
  ``HybridFactorGraph`` payload, constrained or unconstrained scalar
  correlation coordinates, unsupported dependency diagnostics, and
  fail-closed behavior for unsupported hybrid derivative requests
- graph-backed bounded quanto scalar-vector VJP through
  ``trellis.analytics.differentiate_quanto_scalar_inputs(...)``, with sparse
  graph-owned sensitivities for supported spot, FX spot, curve-node, vol-node,
  and scalar-correlation coordinates plus selected-factor and unsupported
  dependency diagnostics
- graph-backed bounded quanto scalar-vector HVP through
  ``trellis.analytics.differentiate_quanto_scalar_inputs(...)`` when requested
  with an explicit sparse ``hvp_direction`` over supported graph-owned
  coordinates, with fail-closed diagnostics for missing or empty directions
- graph-backed bounded terminal quanto matrix-coordinate VJP/HVP through
  ``trellis.analytics.differentiate_quanto_correlation_matrix(...)`` for a
  checked, well-conditioned direct correlation matrix payload; the sparse risk
  vector is keyed by off-diagonal ``correlation_matrix`` ``RiskFactorId``
  coordinates and HVP directions are explicit sparse vectors over those
  coordinates
- graph-backed bounded vanilla early-exercise VJP through
  ``trellis.analytics.differentiate_vanilla_early_exercise(...)`` for
  American/Bermudan call/put specs over ``FlatVol`` under a hard
  exercise-projection smooth-interior policy
- bounded mixed supported-book reverse mode through
  ``trellis.book.portfolio_aad_supported_book_risk(...)`` for explicitly
  configured combinations of the supported AAD lanes
- rates bootstrap calibration through an ``autodiff_vector_jacobian`` repricing
  matrix
- flat-vol Vega extraction in the analytics layer
- SABR calibration through an ``autodiff_scalar_gradient`` objective
- supported Heston smile and full-surface calibration through an explicit
  bounded ``finite_difference_vector_jacobian`` when the FFT/implied-vol stack
  itself is not autograd-safe; the full-surface entrypoint is
  ``fit_heston_surface``
- simple binomial/trinomial tree rollback through ``backward_induction(..., differentiable=True)``
- pathwise Monte Carlo pricing through ``simulate_with_shocks(..., differentiable=True)``
- smooth terminal-only and event-replay state-aware Monte Carlo payoffs through
  ``MonteCarloEngine.price(..., shocks=..., differentiable=True)`` and
  ``price_event_aware_monte_carlo(..., shocks=..., differentiable=True)``
- barrier-monitor and event-aware Monte Carlo pricing now carries
  ``derivative_metadata`` with the discontinuous derivative policy when a
  traced pathwise Greek is unsupported

These paths now use autograd-friendly primitives and avoid scalarization inside
the traced region.

Product-Family Gradient Matrix
------------------------------

``tests/test_verification/test_autograd_gradient_matrix.py`` is the checked
product-family gradient matrix for the public support contract. It is
representative, not exhaustive: each row pins one product family, derivative
method, fallback boundary, or unsupported lane that future generated routes and
runtime reporting must not overstate.

.. list-table::
   :header-rows: 1

   * - Family id
     - Product family
     - Checked derivative method
     - Support status and boundary
   * - ``analytical_black76``
     - Black76 closed-form route
     - ``autodiff_scalar_gradient``
     - supported on smooth interior inputs; compared against finite
       differences
   * - ``public_curve_nodes``
     - public ``YieldCurve`` and ``CreditCurve`` node routes
     - ``autodiff_public_curve``
     - supported for node-value sensitivities; query-location derivatives are
       only piecewise away from knots
   * - ``grid_vol_surface_bucketed``
     - ``GridVolSurface`` node sensitivities and runtime bucketed vega
     - ``surface_bucket_bump``
     - partial support: node values are traceable, while runtime surface risk is
       explicit bucket bumping rather than scalar surface-native AD
   * - ``smooth_monte_carlo_pathwise``
     - smooth Monte Carlo route through ``simulate_with_shocks(...)`` /
       ``MonteCarloEngine.price(..., shocks=..., differentiable=True)``
     - ``autodiff_pathwise``
     - supported only for deterministic explicit-shock paths and smooth payoff
       contracts
   * - ``rates_bootstrap_calibration``
     - rates bootstrap calibration route
     - ``autodiff_vector_jacobian``
     - supported through the repricing Jacobian recorded in
       ``solver_provenance``
   * - ``bounded_quanto_calibration``
     - bounded hybrid quanto-correlation calibration route
     - ``scipy_2point_residual_jacobian``
     - partial support: the checked ``bounded_quanto_correlation`` slice is a
       real shipped hybrid route, but its least-squares solve currently
       reports finite-difference residual Jacobian provenance with no
       ``backend_operator``; this is not a claim of hybrid AD, ``jvp``, or
       broad ``portfolio_aad``
   * - ``quanto_primitive_composition``
     - generated quanto analytical primitive-composition route
     - ``autodiff_scalar_gradient``
     - supported through the reusable adjusted-forward and Black pricing
       primitives; the retained raw quanto kernel is a verification reference,
       not generated-route authority
   * - ``barrier_mc_discontinuous_policy``
     - barrier Monte Carlo discontinuous policy route
     - ``unsupported_discontinuous_pathwise``
     - unsupported for pathwise AD; the governed fallback is
       ``finite_difference_bump_reprice`` with fail-closed policy metadata
   * - ``portfolio_aad_vjp``
     - bounded bond-book, option-book, and mixed supported-book portfolio AAD routes
     - ``portfolio_aad_vjp``
     - partial support: shared-curve bond books, European option books over
       shared flat/grid vol, smooth-interior early-exercise option books over
       shared flat vol, arithmetic-Asian smooth path-summary books over shared
       flat vol, bounded quanto books over one scalar correlation, and mixed
       books composed from those explicit lanes use VJP-backed reverse-mode
       aggregates over canonical risk-factor IDs, while unsupported positions
       are excluded and reported in metadata
   * - ``hybrid_scalar_quanto_vjp``
     - bounded graph-backed scalar quanto derivative route
     - ``hybrid_scalar_vjp`` / ``hybrid_scalar_vector_vjp`` /
       ``hybrid_scalar_vector_hvp`` / ``hybrid_matrix_vector_vjp`` /
       ``hybrid_matrix_vector_hvp`` / ``hybrid_path_summary_vjp`` /
       ``hybrid_early_exercise_vjp``
     - partial support: the scalar underlier/FX correlation coordinate and the
       bounded single-name quanto scalar-coordinate vector are differentiated
       with VJP, and the scalar-coordinate vector supports checked directional
       HVP requests through a typed ``HybridFactorGraph``; a checked terminal
       quanto correlation matrix payload supports direct off-diagonal
       matrix-coordinate VJP/HVP away from the PSD boundary; bounded
       arithmetic-average flat-vol path summaries and vanilla flat-vol
       early-exercise controls support VJP only; hybrid ``jvp``, correlation
       surfaces, matrix projection/repair, grid-vol or discontinuous
       path-state execution, dynamic state execution, early-exercise boundary
       kinks, early-exercise HVP/JVP, and broader hybrid graphs fail closed

Bounded Portfolio-AAD Factor Payload
------------------------------------

The bounded book-level lanes expose typed factorized payloads.
``portfolio_aad_curve_risk(...)`` still returns a ``RiskMeasureOutput`` whose
values are key-rate-duration-style tenor entries, but its metadata now also
includes:

- ``risk_factor_coordinates``: the discovered coordinate table for the shared
  market object
- ``sparse_risk_vector``: sensitivities keyed by canonical ``RiskFactorId``
  payloads
- ``portfolio_aad_result``: the serialized ``PortfolioAADResult`` containing
  portfolio value, sparse risk, coordinates, unsupported positions, method
  metadata, and diagnostics
- ``risk_aggregation_map`` and ``risk_bucket_totals``: the linear map from
  low-level factors to reporting buckets and the resulting bucket totals

``RiskFactorId`` is the stable identity for a differentiable market or model
coordinate. It records object type, object name, coordinate type, optional
currency or issuer, sorted axes such as ``tenor_years``, and an optional
provenance namespace. ``RiskFactorRegistry`` discovers supported
``YieldCurve`` zero-rate nodes for the executable bond-book lane and supported
``FlatVol`` scalar-vol and ``GridVolSurface`` node-vol coordinates for the
bounded vanilla option lane. It can also describe credit-curve hazard nodes and
scalar model parameters as ``discovery_only`` coordinates for future adapters.

``trellis.analytics.admit_portfolio_aad_lane(...)`` is the semantic admission
gate used before widening those adapters. It classifies ``ContractIR`` and
``DynamicContractIR`` shapes into supported, planned, or unsupported
portfolio-AAD lanes, and records the required market-coordinate family before a
pricing tape is built. Today terminal European vanilla option ``ContractIR``
shapes over scalar flat vol and grid-vol node coordinates are admitted as
supported. Early-exercise/control shapes are admitted for the bounded flat-vol
vanilla option lane under the smooth-interior hard-projection policy.
Arithmetic-average path summaries are admitted for the bounded flat-vol
arithmetic-Asian lane under the lognormal moment-matching policy. Grid-vol
early-exercise and grid-vol path-dependent shapes remain planned fail-closed,
and discontinuous event monitors are reported as unsupported. The bounded
quanto family is admitted only for the scalar underlier/FX correlation
parameterization. Admission metadata is a support decision only; it is not
itself a pricing implementation.

``portfolio_aad_equity_option_vol_risk(...)`` returns the typed
``PortfolioAADResult`` directly. It supports smooth European call/put specs
that expose ``spot``, ``strike``, ``expiry_date``, ``option_type``, optional
``notional``, and optional ``exercise_style="european"``. It also supports
bounded American/Bermudan vanilla call/put specs over ``FlatVol`` when the CRR
exercise policy is smooth enough for the configured
``early_exercise_boundary_tolerance``. Flat-vol books aggregate onto one
canonical ``vol_surface`` / ``flat_vol`` factor, while grid-vol European books
aggregate onto sparse ``vol_surface`` / ``black_vol`` expiry/strike node
factors. Grid-vol early-exercise, arithmetic Asians, barrier, broader
path-dependent, and local-vol option AAD remain unsupported or planned in this
lane and are reported as unsupported positions rather than silently bumped.

``portfolio_aad_arithmetic_asian_vol_risk(...)`` is the bounded smooth
path-summary lane. It supports European arithmetic-average call/put specs that
expose ``spot``, ``strike``, ``expiry_date``, ``observation_dates`` or
``n_observations``, ``option_type``, optional ``notional``, and optional
``dividend_yield``. It prices with the same moment-matched lognormal
approximation as the bounded arithmetic-Asian helper and differentiates the
shared ``FlatVol`` scalar with VJP. Barrier, knock, first-hit, grid-vol path,
early-exercise path, geometric-average, and broader event-monitor shapes fail
closed with explicit support reasons.

``portfolio_aad_quanto_correlation_risk(...)`` is the bounded hybrid lane. It
accepts a ``QuantoCorrelationAADMarketContext`` with already-resolved quanto
inputs and one scalar ``corr`` value, then differentiates that scalar through
the analytical quanto kernel. The resulting factor is a canonical
``RiskFactorId(object_type="model_parameter", coordinate_type="correlation")``
with optional ``factor_a`` / ``factor_b`` axes and a ``tanh`` transform label.
This is not universal hybrid AAD: curves, spots, FX, vols, and any broader
hybrid factor graph are held fixed outside this lane.

``portfolio_aad_supported_book_risk(...)`` is the mixed supported-book
dispatcher. Callers pass the explicit market context for each lane they want
enabled, such as a shared bond curve plus one vanilla option vol surface. The
dispatcher routes each position to the first supporting adapter, aggregates the
resulting sparse ``RiskFactorId`` vector across risk classes, preserves
per-lane support attempts in metadata, and keeps unsupported positions
fail-closed. This is still bounded runtime aggregation over known adapters, not
a generic portfolio compiler or an industrial-scale tape.

The local benchmark gate for these bounded lanes is
``scripts/benchmark_portfolio_aad.py``. It reports book size, lane mix, factor
count, AAD elapsed time, deterministic bump/reprice baseline elapsed time, and
relative speedup for the supported bond, flat-vol option, grid-vol option, and
mixed supported-book fixtures. The gate is evidence for this explicit support
contract only; it does not change the backend ``portfolio_aad=False`` capability
or imply broad tape coverage.

``PortfolioAADRequest`` is the request-side support contract. A request may
select a subset of factors, set the unsupported-position policy, and preserve
whether unsupported values should be included when they can still be priced.
``PortfolioAADResult.apply_request(...)`` filters the sparse risk vector and
coordinate table without mutating the full result, while
``missing_selected_factors(...)`` reports requested factors that were absent
from the produced result. This keeps factor filtering explicit rather than
silently inventing missing sensitivities.

Unsupported products remain fail-closed for AAD risk. The default policy is to
report ``UnsupportedAADPosition`` records with the position name, instrument
type, reason, requested factors, value-inclusion flag, risk-inclusion flag, and
fallback method. The current executable routes exclude unsupported positions
from AAD risk and does not hide a finite-difference fallback inside the AAD
result.

Bounded Hybrid Factor-Graph Derivatives
---------------------------------------

The graph-backed hybrid derivative prototype is intentionally separate from
the book-level portfolio-AAD lanes. ``HybridFactorGraph`` records the market
objects and model parameters visible to a bounded hybrid route, while
``MarketObjectCoordinateChart`` owns the coordinate transform and executable
chart context for any supported factor. The first constrained chart is scalar
correlation:

.. code-block:: text

   rho = tanh(x)

``differentiate_quanto_scalar_correlation(...)`` accepts a quanto spec plus
resolved inputs. If the inputs were resolved with
``include_hybrid_factor_graph=True``, the helper uses that graph; otherwise it
builds a scalar-correlation fallback graph so legacy callers can still receive
typed metadata. The returned ``HybridDerivativeResult`` contains the base
value, sparse risk vector, graph payload, method metadata, unsupported
dependencies, and diagnostics.

``differentiate_quanto_scalar_inputs(...)`` uses the same resolved-input
boundary but requires graph-owned executable scalar chart context. By default it
returns a sparse VJP vector for supported bounded quanto scalar coordinates:
underlier spot, FX spot, domestic and foreign curve zero-rate nodes, flat or
grid volatility nodes, and scalar correlation. With
``derivative_method="hvp"`` and a non-empty sparse ``hvp_direction``, it returns
the directional second derivative ``H @ v`` over the same coordinate chart.
Sparse zero sensitivities are omitted from the vector, so a selected
graph-owned factor such as FX spot can be available and still return no entry
when the raw quanto kernel is insensitive to that coordinate. Missing selected
factors, missing HVP direction factors, unsupported graph dependencies, and
fail-closed policy decisions are reported in diagnostics.

The supported derivative methods are VJP and scalar-objective HVP for this
bounded route. ``HybridDerivativeRequest`` may ask for constrained ``rho``
sensitivity or unconstrained ``x`` sensitivity, may select a subset of graph
factors, and may provide a sparse HVP direction keyed by ``RiskFactorId``.
Unsupported selections can return an empty sparse vector or fail closed
according to the request policy. Empty or unavailable HVP directions always
fail closed.

``differentiate_quanto_correlation_matrix(...)`` uses the same terminal quanto
pricing kernel with an explicit correlation-matrix request. The request must
provide factor labels, a finite symmetric unit-diagonal matrix, bounded
entries, and a positive-semidefinite payload whose minimum eigenvalue is above
the executable context floor. Trellis then promotes the checked policy chart
to a supported matrix-coordinate context without projection, smoothing, or
repair. The active quanto pair defaults to ``(spec.underlier_currency,
spec.fx_pair)`` and maps to one off-diagonal matrix ``RiskFactorId``; other
off-diagonal coordinates remain graph-owned coordinates and can be selected or
used in sparse HVP directions. VJP returns ``hybrid_matrix_vector_vjp``
metadata. HVP requests with ``derivative_method="hvp"`` return
``hybrid_matrix_vector_hvp`` metadata and directional ``H @ v`` values. Empty
or unavailable HVP directions, invalid matrices, matrices too close to the PSD
boundary, surface requests, and ``jvp`` all fail closed with empty risk and
typed diagnostics.

``admit_hybrid_ad_lane(...)`` is the semantic admission boundary for these
graph-owned lanes. Given a ``ContractIR`` and requested derivative method, it
returns a JSON-friendly ``HybridADLaneAdmission`` payload that admits bounded
terminal quanto VJP/HVP requests over scalar graph coordinates or the direct
matrix-coordinate lane, plus the bounded arithmetic-average path-summary VJP
lane and the bounded vanilla early-exercise flat-vol VJP lane. The scalar
quanto payload records requirements for underlier spot, FX
spot, domestic and foreign curve nodes, vol nodes, and scalar correlation; the
matrix payload replaces the scalar-correlation requirement with a
``correlation_matrix`` requirement using the
``correlation_matrix_psd_policy`` parameterization. The path-summary payload
records one graph-owned ``FlatVol`` requirement, and the early-exercise
payload records the same one graph-owned ``FlatVol`` requirement under the
hard exercise-projection smooth-interior policy. When
``market_parameterization="grid_vol"`` is requested for arithmetic path
summaries or vanilla early-exercise controls, admission records a planned
``grid_node_vols`` volatility requirement and a fail-closed state/control
policy; it deliberately omits a runtime helper until an executable lane is
validated. The corresponding grid-vol state/control coordinate chart is
discovery-only: it preserves node identity and selected-factor behavior for
runtime diagnostics, but does not authorize VJP execution by itself. The
arithmetic path-summary runtime uses that chart to fail closed with
``unsupported_grid_vol_interpolation`` until a true node-local path-summary
surface derivative is mathematically defined and verified. The vanilla
early-exercise runtime also uses that chart for grid-vol inputs, but reports
the state/control policy as ``grid_vol_hard_exercise_projection_pending`` and
does not evaluate value or risk. Grid-vol HVP and JVP runtime requests also
preserve the chart payload while failing closed; JVP remains a backend-level
unsupported operator and HVP remains planned for these state/control lanes.
Correlation surfaces, composite underliers,
grid-vol path summaries, discontinuous event monitors, non-arithmetic path
summaries, path-summary HVP, grid-vol early-exercise, early-exercise HVP, JVP,
and boundary-kink
early-exercise shapes are classified as unsupported or planned before runtime
AD is invoked. ``HybridDerivativeRequest`` can carry that admission object, or
its payload, as ``semantic_admission``. Supported
same-lane admissions are copied into
``HybridDerivativeResult.method_metadata["semantic_admission"]``; wrong-lane,
planned, or unsupported admissions return an empty risk vector with the
admission reason in diagnostics and ``fallback_reason``.
When the blocked admission concerns path state or event policy, the runtime
metadata also includes ``semantic_state_policy`` plus searchable fields such
as ``semantic_state_kind``, ``semantic_state_event_policy``, and
``semantic_state_control_policy``. Arithmetic-average path summaries are
supported only in the flat-vol VJP lane; vanilla early-exercise controls are
supported only in the flat-vol VJP lane under the smooth-interior hard
projection policy. Dynamic controls remain planned, while discontinuous event
monitors remain unsupported. None of these state-policy payloads imply broad
pathwise hybrid AD.

``fail_closed_dynamic_state_derivative(...)`` is the runtime surface for
``DynamicContractIR`` state/control requests that reach Hybrid AD diagnostics.
It does not price or differentiate the dynamic program. Instead it returns an
unsupported ``HybridDerivativeResult`` whose ``HybridFactorGraph`` owns a
single ``dynamic_state_policy`` coordinate, maps admission-level
``planned`` status to a graph-level ``discovery_only`` chart, and carries the
dynamic state/event/control policy into both graph metadata and
``method_metadata["semantic_state_policy"]``. Selected ``RiskFactorId`` probes
are checked against that graph-owned policy coordinate; unavailable selections
produce ``selected_factors_unavailable`` diagnostics and keep risk empty. JVP
requests use the backend-level ``unsupported_hybrid_jvp`` payload and do not
advertise an executable ``backend_operator``.

Forward mode remains executable-truth governed. A ``derivative_method="jvp"``
request returns an unsupported result with
``resolved_derivative_method="unsupported_hybrid_jvp"`` and
``fallback_reason.code="hybrid_jvp_backend_unsupported"`` because the active
``autograd`` backend still reports ``jvp=False`` for pricing primitives.
Those fail-closed payloads carry ``requested_backend_operator="jvp"`` plus the
``backend_support`` record from the capability ``support_matrix``. They do not
carry an executable ``backend_operator`` field for JVP, because no JVP backend
hook ran.
``fail_closed_correlation_structure_derivative(...)`` provides the same
fail-closed envelope for correlation matrix and correlation surface requests.
For matrix requests it can validate factor labels, square shape, finite
entries, symmetry, unit diagonal, bounds, and positive semidefiniteness, then
attach chart metadata while still returning
``correlation_matrix_derivative_not_implemented`` for callers using that
explicit fail-closed helper. Executable matrix sensitivities are exposed only
through ``differentiate_quanto_correlation_matrix(...)`` under the bounded
terminal quanto contract described above. For surface requests,
``MarketObjectCoordinateChart.correlation_surface_policy(...)`` now defines a
discovery-only ``correlation_surface_policy`` chart over deterministic
surface-node ``RiskFactorId`` coordinates, including factor pairs, surface
axes, interpolation/locality policy, selected-factor policy, and
no-projection/no-smoothing/no-repair constraints. Valid surface requests still
return ``correlation_surface_derivative_not_implemented`` with an empty sparse
vector, while invalid surface-axis payloads return
``invalid_correlation_surface_axes``. No projection, smoothing, surface AD, or
universal matrix AD execution is implied.

Runtime Derivative-Method Taxonomy
----------------------------------

Runtime analytics, book-level risk, Monte Carlo derivative policy, and the
checked matrix use the shared registry in
``trellis.analytics.derivative_methods`` rather than route-local strings. The
metadata keeps the historical ``resolved_derivative_method`` field and adds a
normalized reporting envelope:

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``resolved_derivative_method``
     - canonical method id, such as ``autodiff_public_curve``,
       ``portfolio_aad_vjp``, ``parallel_curve_bump``, or
       ``unsupported_discontinuous_pathwise``
   * - ``derivative_method_category``
     - family-level category such as ``analytical_autograd``, ``autograd``,
       ``portfolio_aad``, ``hybrid_ad``, ``finite_difference_bump``,
       ``unsupported``, ``forward``, ``unavailable``, ``not_applicable``, or
       ``provided``
   * - ``derivative_method_support``
     - runtime support status: ``supported``, ``partial``, ``fallback``,
       ``unsupported``, or ``not_applicable``
   * - ``backend_operator``
     - executable backend hook when applicable, for example ``grad``,
       ``jacobian``, ``vjp``, or ``hessian_vector_product``; unsupported
       hooks such as ``jvp`` must not be advertised as executed methods
   * - ``requested_backend_operator``
     - backend hook requested by a fail-closed call, such as ``jvp``; this is
       separate from ``backend_operator`` because no hook executed
   * - ``backend_support``
     - the ``operator_support(...)`` record from the backend
       ``support_matrix`` for the requested unsupported hook
   * - ``fallback_derivative_method``
     - declared fallback lane when the requested derivative is unsupported or
       intentionally bump-based, for example ``finite_difference_bump_reprice``
   * - ``fallback_reason`` and ``warnings``
     - structured reason payloads explaining why a fallback or unsupported lane
       was selected
   * - ``parameterization`` plus bump fields
     - route-specific basis, node, bucket, or bump size metadata such as
       ``parallel_zero_rate_shift``, ``curve_node_zero_rates``, ``bump_bps``,
       or ``bump_vol_bps``

The registry deliberately includes both AD-backed and non-AD lanes:

.. list-table::
   :header-rows: 1

   * - Method id
     - Category
     - Support
     - Backend operator / boundary
   * - ``autodiff_scalar_gradient``
     - ``analytical_autograd``
     - ``supported``
     - ``grad`` for smooth scalar analytical or calibration objectives
   * - ``autodiff_vector_jacobian``
     - ``autograd``
     - ``supported``
     - ``jacobian`` for smooth vector repricing maps
   * - ``finite_difference_vector_jacobian``
     - ``finite_difference_bump``
     - ``supported``
     - explicit vector finite differences when the repricing stack is not
       autograd-safe
   * - ``autodiff_public_curve``
     - ``autograd``
     - ``supported``
     - ``grad`` through public curve node values
   * - ``autodiff_flat_vol``
     - ``autograd``
     - ``supported``
     - ``grad`` through scalar flat-vol inputs
   * - ``surface_bucket_bump``
     - ``finite_difference_bump``
     - ``supported``
     - explicit expiry/strike bucket bump for grid-vol runtime risk
   * - ``surface_parallel_bucket_bump``
     - ``finite_difference_bump``
     - ``supported``
     - explicit parallel grid-node bump for scalar grid-vol vega
   * - ``flat_surface_expanded_bucket_bump``
     - ``finite_difference_bump``
     - ``supported``
     - bucketed vega expanded from one flat-vol value
   * - ``representative_flat_vol_bump``
     - ``finite_difference_bump``
     - ``fallback``
     - representative-flat-vol fallback for unsupported smile surfaces
   * - ``parallel_curve_bump``
     - ``finite_difference_bump``
     - ``fallback``
     - parallel curve bump/reprice when public curve AD is unavailable
   * - ``curve_bucket_bump``
     - ``finite_difference_bump``
     - ``supported``
     - explicit curve-tenor bucket bump for key-rate and scenario risk
   * - ``bootstrap_quote_bump_rebuild``
     - ``finite_difference_bump``
     - ``supported``
     - quote bump followed by curve rebuild
   * - ``spot_central_bump``
     - ``finite_difference_bump``
     - ``supported``
     - central spot bump/reprice for delta and gamma
   * - ``calendar_roll_down_bump``
     - ``finite_difference_bump``
     - ``supported``
     - calendar roll-down repricing for theta
   * - ``portfolio_aad_vjp``
     - ``portfolio_aad``
     - ``partial``
     - ``vjp`` for the bounded shared-curve bond-book, option-book,
       arithmetic-Asian, scalar quanto-correlation, and mixed supported-book
       lanes
   * - ``hybrid_scalar_vjp``
     - ``hybrid_ad``
     - ``partial``
     - ``vjp`` for one graph-owned scalar hybrid coordinate, currently the
       bounded quanto underlier/FX correlation
   * - ``hybrid_scalar_vector_vjp``
     - ``hybrid_ad``
     - ``partial``
     - ``vjp`` for the bounded single-name quanto graph-owned scalar-coordinate
       vector over supported spot, curve-node, vol-node, FX spot, and scalar
       correlation coordinates
   * - ``hybrid_scalar_vector_hvp``
     - ``hybrid_ad``
     - ``partial``
     - ``hessian_vector_product`` for bounded single-name quanto directional
       second derivatives over the same graph-owned scalar-coordinate vector
   * - ``hybrid_matrix_vector_vjp``
     - ``hybrid_ad``
     - ``partial``
     - ``vjp`` for bounded terminal quanto direct matrix-coordinate risk over
       checked off-diagonal correlation-matrix coordinates
   * - ``hybrid_matrix_vector_hvp``
     - ``hybrid_ad``
     - ``partial``
     - ``hessian_vector_product`` for bounded terminal quanto directional
       second derivatives over the checked off-diagonal matrix-coordinate
       vector
   * - ``hybrid_path_summary_vjp``
     - ``hybrid_ad``
     - ``partial``
     - ``vjp`` for bounded arithmetic-average smooth path-summary flat-vol
       risk
   * - ``hybrid_early_exercise_vjp``
     - ``hybrid_ad``
     - ``partial``
     - ``vjp`` for bounded vanilla American/Bermudan smooth-interior
       early-exercise flat-vol risk
   * - ``unsupported_hybrid_jvp``
     - ``unsupported``
     - ``unsupported``
     - fail-closed hybrid forward-mode request; payloads include
       ``requested_backend_operator="jvp"`` and ``backend_support`` instead
       of an executable ``backend_operator``
   * - ``unsupported_hybrid_structure``
     - ``unsupported``
     - ``unsupported``
     - fail-closed hybrid request when matrix context validation fails, when a
       matrix request is outside the executable bounded lane, or when a surface
       chart is not implemented
   * - ``autodiff_pathwise``
     - ``autograd``
     - ``supported``
     - ``grad`` through explicit-shock smooth Monte Carlo paths
   * - ``forward_price_only``
     - ``forward``
     - ``unsupported``
     - pricing executed without a derivative lane; declared fallback is
       ``finite_difference_bump_reprice``
   * - ``unsupported_discontinuous_pathwise``
     - ``unsupported``
     - ``unsupported``
     - fail-closed pathwise AD lane for discontinuous Monte Carlo payoffs or
       event logic
   * - ``finite_difference_bump_reprice``
     - ``finite_difference_bump``
     - ``fallback``
     - declared bump/reprice fallback for unsupported runtime derivatives
   * - ``vol_surface_unavailable``
     - ``unavailable``
     - ``unsupported``
     - no volatility surface is available for requested vol risk
   * - ``not_applicable_root_scalar``
     - ``not_applicable``
     - ``not_applicable``
     - scalar root solve with no derivative method
   * - ``provided_scalar_gradient``
     - ``provided``
     - ``supported``
     - caller-supplied scalar gradient
   * - ``provided_vector_jacobian``
     - ``provided``
     - ``supported``
     - caller-supplied vector residual Jacobian
   * - ``scipy_internal_finite_difference_gradient``
     - ``finite_difference_bump``
     - ``fallback``
     - SciPy internal finite-difference scalar gradient
   * - ``scipy_2point_residual_jacobian``
     - ``finite_difference_bump``
     - ``fallback``
     - SciPy two-point finite-difference residual Jacobian

Where Trellis Still Stays Forward-Only
--------------------------------------

- generic lattice calibration and true streaming reduced-storage Monte Carlo
  path-state accumulation
- Numba-accelerated tree, Monte Carlo, and PDE kernels
- discontinuous payoffs that would need smoothing or a custom adjoint
- broader European barrier families beyond the T09 route. Double-barrier PDE
  and Monte Carlo routes now have checked pricing-facing helpers, but
  discontinuous barrier pathwise derivatives still fail closed under the Monte
  Carlo derivative policy.
- scalar vega on unsupported smile surfaces, which now reports an explicit
  representative-flat-vol fallback instead of silently pretending to be a
  surface-native Greek
- hybrid derivative claims outside the graph-backed bounded single-name quanto
  scalar-coordinate and matrix-coordinate VJP/HVP bridges, the bounded
  arithmetic-average flat-vol path-summary VJP lane, and the bounded vanilla
  flat-vol early-exercise VJP lane; the shipped calibration route still
  records ``scipy_2point_residual_jacobian`` solve provenance, and the shipped
  derivative routes do not widen into universal hybrid AD, hybrid ``jvp``,
  correlation-surface AD, matrix projection/repair or PSD-boundary AD,
  grid-vol or event-monitor path-state execution, dynamic state execution,
  grid-vol/boundary-kink early-exercise derivatives, path-summary or
  early-exercise HVP, or broad ``portfolio_aad`` support
- state-aware Monte Carlo contracts with barrier monitors or other
  discontinuous event semantics, which still stay off the traced lane even
  when explicit shocks are supplied; the bounded checked policy is
  ``discontinuous_derivative_policy="fail_closed"`` with
  ``resolved_derivative_method="unsupported_discontinuous_pathwise"`` on the
  rejected AD lane, ``resolved_derivative_method="forward_price_only"`` on
  ordinary forward pricing, and ``fallback_derivative_method="finite_difference_bump_reprice"``
  as the declared derivative fallback rather than silent smoothing
- custom discretization schemes that are not autograd-aware themselves

That split is deliberate: the compiled engines stay fast for production pricing,
while the closed-form layer exposes gradients where they are genuinely useful.

Phase 2 Follow-On Program
-------------------------

The completed public-contract work is now tracked separately from the next
autograd phase. ``QUA-966`` is the follow-on umbrella for portfolio AAD and
gradient governance; ``QUA-1011`` delivered the first risk-factor identity,
request/result, adapter-protocol, and verification substrate under that
program. The repo mirrors are
``doc/plan/draft__autograd-phase-2-aad-and-gradient-governance.md`` and
``doc/plan/draft__broad-portfolio-aad-prerequisites.md``.

That program is deliberately narrower than "differentiate everything". It is
organized around five concrete follow-on slices:

.. list-table::
   :header-rows: 1

   * - Slice
     - Goal
     - Boundary
   * - Backend operators
     - ``vjp`` and scalar-objective ``hessian_vector_product`` are checked;
       ``jvp`` remains a documented fail-closed backend decision
     - do not report ``jvp`` or ``portfolio_aad`` support until the wrappers
       compute checked values
   * - Portfolio AAD
     - widen the checked factorized substrate beyond the current shared-curve
       bond-book and shared-flat-vol vanilla option lanes
     - bounded supported books only; unsupported routes must be excluded or
       reported explicitly, and ``portfolio_aad=False`` remains the backend
       truth until broad support is executable
   * - Discontinuous Greeks
     - define smoothing, custom-adjoint, finite-difference, or unsupported
       policy for barriers, digitals, and exercise/event logic; the first
       checked slice covers Monte Carlo barrier monitors plus barrier/exercise
       event replay as an explicit fail-closed pathwise AD policy
     - no silent smoothing of production prices; finite-difference
       bump/reprice is the declared fallback method for this bounded slice
   * - Gradient matrix
     - expand the support-contract tests into a product-family derivative
       matrix
     - representative coverage, not exhaustive generated-route coverage
   * - Runtime reporting
     - normalize derivative-method metadata across analytical, AD, AAD,
       smoothed/custom-adjoint, bump, and unsupported lanes
     - reporting only; pricing formulas remain owned by their route families

The broader curve, surface, and cube calibration plants are tracked under the
separate calibration sleeve program ``QUA-946``. The autograd Phase 2 work
should consume those stronger calibrated market objects, not duplicate the
calibration industrialization backlog.

Implementation Rules
--------------------

- use ``autograd.numpy`` via ``trellis.core.differentiable.get_numpy()``
- query ``trellis.core.differentiable.get_backend_capabilities()`` before
  relying on backend-specific derivative operators
- avoid ``float(...)`` or other scalarization inside traced pricing code
- for Monte Carlo gradients, pass explicit shocks so the path is deterministic
- keep the public pricing adapter trace-safe and reserve ``float(...)`` for
  explicit reporting or solver boundaries; expose raw resolved-input kernels as
  ``*_raw`` helpers when they improve reuse
- keep generator prompts, cookbook templates, and payoff skeletons aligned to
  that same public contract so learned routes preserve traced PVs by default
- for piecewise-linear curves and surfaces, target node-value derivatives as
  the supported contract and treat query-location derivatives as piecewise only
  away from knot boundaries
- keep Numba kernels as forward engines, not gradient engines
- preserve the full ``MarketState`` when cloning a traced pricing state
- prefer autodiff when it replaces bump/reprice loops, parallel DV01s, or
  optimizer Jacobians
- when a runtime measure falls back to bumps, record that derivative-method
  choice in the public result metadata instead of hiding it behind a plain
  scalar
- for calibration workflows, record the derivative method that actually ran in
  ``solve_result.metadata`` and ``solver_provenance.backend`` so audit/replay
  consumers can distinguish autograd, explicit finite differences, and solver
  fallbacks
- for state-aware Monte Carlo gradients, keep terminal/snapshot adapters
  trace-safe and reject non-smooth barrier or exercise state contracts
  explicitly instead of silently scalarizing them; inspect
  ``derivative_metadata`` or
  ``describe_monte_carlo_derivative_policy(...)`` for the fail-closed policy,
  discontinuous feature labels, and declared fallback method

Implementation References
-------------------------

.. autofunction:: trellis.models.black.black76_call
.. autofunction:: trellis.core.differentiable.get_backend_capabilities
.. autofunction:: trellis.models.black.garman_kohlhagen_call
.. autofunction:: trellis.models.analytical.fx.garman_kohlhagen_price_raw
.. autofunction:: trellis.models.analytical.terminal_vanilla_from_basis
.. autofunction:: trellis.models.analytical.jamshidian.zcb_option_hw_raw
.. autofunction:: trellis.models.rate_style_swaption.price_swaption_black76_raw
.. autofunction:: trellis.models.analytical.quanto.price_quanto_option_raw
.. autofunction:: trellis.analytics.differentiate_quanto_scalar_correlation
.. autofunction:: trellis.analytics.differentiate_quanto_scalar_inputs
.. autofunction:: trellis.analytics.differentiate_quanto_correlation_matrix
.. autofunction:: trellis.analytics.admit_hybrid_ad_lane
.. autofunction:: trellis.analytics.fail_closed_correlation_structure_derivative
.. autofunction:: trellis.models.analytical.barrier.down_and_out_call_raw
.. autofunction:: trellis.models.analytical.barrier.down_and_in_call_raw
.. automethod:: trellis.instruments.cap.CapPayoff.evaluate
.. automethod:: trellis.analytics.measures.Vega.compute
.. autofunction:: trellis.models.calibration.sabr_fit.calibrate_sabr
.. autofunction:: trellis.models.calibration.heston_fit.fit_heston_surface
.. autofunction:: trellis.models.trees.backward_induction.backward_induction
.. automethod:: trellis.models.monte_carlo.engine.MonteCarloEngine.simulate_with_shocks
.. automethod:: trellis.models.monte_carlo.engine.MonteCarloEngine.price

Related Reading
---------------

- :doc:`../mathematical/black76`
- :doc:`../mathematical/calibration`
- :doc:`extending_trellis`
- :doc:`pricing_stack`
