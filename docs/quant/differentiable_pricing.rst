Differentiable Pricing
======================

Trellis promotes autograd only where it has a clear payoff:

- closed-form pricing kernels that currently drive real Greeks and calibration
- cap/floor strips and FX/quanto analytics built on top of those kernels
- flat-vol Vega extraction in ``Session.analyze()``
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

The goal is not to make every numerical routine differentiable. It is to remove
unnecessary bump/reprice loops, stabilize calibration, and keep the exact same
pricing logic available to both value and sensitivity workflows.

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
       ``Session.risk_report(...)``
     - bounded to supported bond positions on a shared ``YieldCurve``;
       unsupported positions are listed in metadata and excluded from the
       reverse-mode aggregate
   * - Flat volatility risk
     - scalar vega through ``autodiff_flat_vol``
     - flat surfaces only
   * - Calibration
     - rates bootstrap ``autodiff_vector_jacobian``, SABR
       ``autodiff_scalar_gradient``, and Heston smile / full-surface
       ``finite_difference_vector_jacobian``
     - solver provenance records the derivative method that actually ran
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
the first bounded book-level reverse-mode lane now builds on that surface for
supported curve-linked bond books. The capability flag stays
``portfolio_aad=False`` because the supported contract is still book-specific
rather than a claim of universal portfolio-AAD coverage.
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
  ``YieldCurve``, with unsupported positions excluded explicitly in metadata
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
   * - ``quanto_generated_helper``
     - route-generated quanto analytical helper route
     - ``autodiff_scalar_gradient``
     - supported through the semantic helper-facing raw pricing path, not a
       claim that every generated adapter preserves traced values
   * - ``barrier_mc_discontinuous_policy``
     - barrier Monte Carlo discontinuous policy route
     - ``unsupported_discontinuous_pathwise``
     - unsupported for pathwise AD; the governed fallback is
       ``finite_difference_bump_reprice`` with fail-closed policy metadata
   * - ``portfolio_aad_vjp``
     - bounded bond-book portfolio AAD route
     - ``portfolio_aad_vjp``
     - partial support: shared-curve bond books use a VJP-backed reverse-mode
       aggregate, while unsupported positions are excluded and reported in
       metadata

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
       ``portfolio_aad``, ``finite_difference_bump``, ``unsupported``,
       ``forward``, ``unavailable``, ``not_applicable``, or ``provided``
   * - ``derivative_method_support``
     - runtime support status: ``supported``, ``partial``, ``fallback``,
       ``unsupported``, or ``not_applicable``
   * - ``backend_operator``
     - executable backend hook when applicable, for example ``grad``,
       ``jacobian``, ``vjp``, or ``hessian_vector_product``; unsupported
       hooks such as ``jvp`` must not be advertised as executed methods
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
     - ``vjp`` for the bounded shared-curve bond-book lane
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
- broader European barrier families beyond the T09 route, which remain
  forward-only until a second consumer justifies shared barrier support
- scalar vega on unsupported smile surfaces, which now reports an explicit
  representative-flat-vol fallback instead of silently pretending to be a
  surface-native Greek
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
gradient governance; its repo mirror is
``doc/plan/draft__autograd-phase-2-aad-and-gradient-governance.md``.

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
     - add the first book-level reverse-mode sensitivity substrate
     - bounded supported books only; unsupported routes must be excluded or
       reported explicitly
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
