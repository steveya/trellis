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

Scheduled Weighted Observations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``trellis.models.observation_aggregation`` provides one product-neutral linear
functional of scheduled scalar process state.  A
``WeightedObservationContract`` declares ordered non-negative observation
times and one explicit finite weight per observation.  Trellis does not
normalize those weights: an arithmetic mean uses ``1 / n`` for each of ``n``
observations, while sums, spreads, and signed linear summaries keep their own
declared coefficients.

``weighted_observation_sum(...)`` evaluates the same contract from an existing
observation matrix. ``build_weighted_observation_reducer(...)`` accumulates it
as a ``PathReducer``, and ``weighted_observation_payoff(...)`` combines that
reduced state with a caller-supplied ``settlement_fn``.  This keeps the
responsibility boundary explicit:

- the library maps contractual times to simulation steps and owns the weighted
  state reduction;
- generated pricing code owns strike, option direction, notional, final payoff
  shape, and discounting;
- ``MonteCarloEngine`` can run the resulting ``StateAwarePayoff`` with
  ``return_paths=False``.

Time zero is an admissible observation.  Every observation, including the
final one, must map exactly and distinctly onto the selected uniform simulation
grid. ``resolve_uniform_grid_steps(...)`` validates a caller-selected grid
against exact alignment and the explicit ``min_steps`` / ``max_steps`` bounds,
or finds the smallest exact grid inside those bounds. Off-grid times, aliased
times, exhausted search bounds, vector process
state, non-finite values, and settlement callbacks that do not return one value
per path fail closed. Negative weights are valid because the abstraction is a
linear functional, not an implicit average or a positive-price contract.

This surface does not resolve market data and does not price a named
derivative.  Product route authority moves only in a separate migration after
fresh generated source demonstrates the complete process, payoff, and
discounting composition.

Weighted Lognormal-Sum Moments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``trellis.models.analytical.support.lognormal_moments`` provides exact first
and second moments for one explicitly weighted sum of jointly lognormal
observations.  For

.. math::

   A = \sum_i w_i X_i,

the contract stores each horizon :math:`t_i`, initial level :math:`S_{0,i}`,
expected-growth rate :math:`b_i`, and the covariance
:math:`C_{ij}=\operatorname{Cov}(\log X_i,\log X_j)` of log returns.  Therefore

.. math::

   m_i = \mathbb{E}[X_i] = S_{0,i}e^{b_i t_i},

.. math::

   \mathbb{E}[A] = \sum_i w_i m_i,
   \qquad
   \mathbb{E}[A^2] = \sum_{i,j} w_i w_j m_i m_j e^{C_{ij}}.

``WeightedLognormalSumContract`` requires finite dimensions, positive initial
levels, ordered non-negative horizons, and a finite symmetric positive-
semidefinite log-covariance matrix.  It accepts signed weights for raw moment
calculation.  ``match_lognormal_moments(...)`` fails closed for signed weights
because the resulting sum is not guaranteed to have positive support.

For one constant-parameter GBM,
``single_factor_lognormal_sum_contract(...)`` constructs

.. math::

   C_{ij} = \sigma^2\min(t_i,t_j).

The fitted ``LognormalMomentMatch`` reports total log variance

.. math::

   v_{\log} = \log\left(\frac{\mathbb{E}[A^2]}{\mathbb{E}[A]^2}\right),

and ``effective_volatility(maturity=T)`` returns
:math:`\sqrt{v_{\log}/T}`.  A generated analytical adapter can pass the fitted
mean as the forward and that volatility to ``black76_call`` or
``black76_put``.  Those kernels are undiscounted, so the adapter must still
apply the contractual discount factor and notional explicitly.

Only the two declared moments are exact.  Replacing the distribution of the
sum by a lognormal distribution is an approximation and should be validated
against a direct Monte Carlo construction such as the scheduled weighted-
observation reducer above.  The compatibility arithmetic-average wrapper now
uses this kernel, but product route authority remains unchanged until its
separate migration ticket proves fresh generated composition.

Scheduled Observation Returns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``trellis.models.observation_returns`` provides the product-neutral algebra for
a bounded sum of consecutive simple returns. ``ObservationReturnContract``
declares ordered positive observation times, up/down return direction, local
bounds, global bounds, and payoff scale. It does not resolve a product or market
state and it does not price anything by itself.

The same contract has two reusable execution paths:

- ``bounded_observation_return_sum(...)`` evaluates gross interval returns
  directly. An analytical adapter can map independent standard-normal nodes to
  lognormal gross returns and evaluate that function inside
  ``gauss_hermite_product_expectation(...)``.
- ``observation_return_payoff(...)`` produces a ``StateAwarePayoff`` whose
  ``PathReducer`` stores only the previous observed level and the accumulated
  locally bounded return. ``MonteCarloEngine`` can therefore run with
  ``return_paths=False`` and apply the global bounds at settlement.

Observation times must map exactly to distinct positive simulation steps. The
contract fails closed when the chosen uniform grid aliases two contractual
observations, cannot represent an observation time, or ends before the final
observation. The Gaussian tensor expectation also fails closed when ``order **
dimension`` exceeds its explicit node budget. The Monte Carlo reducer accepts a
scalar positive-level process; the analytical expectation separately assumes
independent normal interval drivers. Vector state and nonpositive observables
need a different admitted substrate.

This layer is construction infrastructure, not a replacement product helper.
Existing product-route authority is migrated separately only after fresh task
artifacts prove that they can compose these primitives.

Discrete Path Statistics
~~~~~~~~~~~~~~~~~~~~~~~~

``trellis.models.monte_carlo.path_statistics`` provides two public,
product-neutral summaries for finite positive scalar paths.  For explicitly
selected simulation steps :math:`i_0 < \cdots < i_m`, the extrema contract
computes either

.. math::

   \min_j S_{i_j}
   \quad\text{or}\quad
   \max_j S_{i_j},

while the squared-log-return contract computes

.. math::

   a\sum_{j=1}^{m}
   \left(\log\frac{S_{i_j}}{S_{i_{j-1}}}\right)^2.

``RunningExtremumContract`` and ``SquaredLogReturnContract`` carry the exact
engine ``n_steps`` value and the complete ordered ``observation_steps`` tuple.
There is no implicit time-zero observation.  The first squared-return
observation is the baseline and contributes no return.  The scalar
``annualization_factor`` :math:`a` multiplies the complete sum; Trellis does
not infer it from maturity, calendar, or observation count.  A prior positive
extremum may be supplied through ``initial_extremum`` without turning the
reducer into a named derivative.

Each statistic has matching full-path and reduced-state surfaces:

- ``discrete_path_extremum(...)`` and
  ``annualized_squared_log_return_sum(...)`` provide direct path-array
  evidence;
- ``build_running_extremum_reducer(...)`` and
  ``build_squared_log_return_reducer(...)`` produce bounded
  ``PathReducer`` state for ``MonteCarloEngine``;
- ``PathReducer.finalize_fn`` separates private accumulator state from the
  public value stored under ``MonteCarloPathState.reduced_value(name)``.  For
  example, squared returns retain the previous level internally but publish
  only the annualized sum.

Generated pricing code owns settlement.  A discrete fixed-strike maximum
claim can therefore be assembled without a product pricer:

.. code-block:: python

   from trellis.core.differentiable import get_numpy
   from trellis.models.monte_carlo.engine import MonteCarloEngine
   from trellis.models.monte_carlo.path_state import (
       MonteCarloPathRequirement,
       StateAwarePayoff,
   )
   from trellis.models.monte_carlo.path_statistics import (
       RunningExtremumContract,
       build_running_extremum_reducer,
       discrete_path_extremum,
   )
   from trellis.models.processes.gbm import GBM

   np = get_numpy()
   contract = RunningExtremumContract(
       n_steps=64,
       observation_steps=tuple(range(65)),
       direction="maximum",
   )
   reducer = build_running_extremum_reducer(contract, name="running_maximum")
   settlement = lambda maximum: notional * np.maximum(maximum - strike, 0.0)
   payoff = StateAwarePayoff(
       path_requirement=MonteCarloPathRequirement(reducers=(reducer,)),
       evaluate_paths_fn=lambda paths: settlement(
           discrete_path_extremum(paths, contract)
       ),
       evaluate_state_fn=lambda state: settlement(
           state.reduced_value("running_maximum")
       ),
   )
   result = MonteCarloEngine(
       GBM(mu=rate - dividend_yield, sigma=volatility),
       n_paths=100_000,
       n_steps=contract.n_steps,
       seed=42,
       method="exact",
   ).price(
       spot,
       maturity,
       payoff,
       discount_rate=rate,
       return_paths=False,
   )

The admitted variance-swap Monte Carlo route uses this pattern directly.  It
first calls
``resolve_scalar_diffusion_market_inputs(..., volatility_coordinate=spec.spot)``
to bind spot, maturity, deterministic rate and carry, one scalar Black
volatility, and the matching discount factor without importing option strike
or call/put semantics.  For the supported ``annualization_convention`` value
``per_year``, generated code declares observations at every engine step and
sets :math:`a=1/T`.  It then adds any historical realized variance, subtracts
the contractual variance strike, applies notional, and discounts the result.
Those settlement terms are deliberately absent from both the resolver and the
path-state module.

This is a constant-parameter GBM comparison lane, not a general variance-model
claim.  Smile integration, local or stochastic volatility, irregular or
calendar-weighted observations, alternative annualization conventions, and
continuous-sampling corrections require separate semantic and numerical
contracts.  The retained product-level variance-swap functions are
compatibility and independent-comparison references, not generated-route
authority.

The admitted analytical comparison is a separate bounded composition.  It
parses a finite positive volatility quote at each finite positive, strictly
increasing strike; uses ``linear_interp(...)`` for the at-the-money volatility;
and computes the FinancePy-compatible smile-slope approximation

.. math::

   K_{\mathrm{var,fair}}
   = \sigma_{\mathrm{ATM}}^2 \sqrt{1 + 3 T s^2},
   \qquad
   s = S_0\frac{\sigma_{\mathrm{high}}-\sigma_{\mathrm{low}}}
                    {K_{\mathrm{high}}-K_{\mathrm{low}}}.

After strict monotonicity validation, a grid whose endpoint strike span has
absolute value at most ``1e-12`` uses ``s = 0``.  This preserves the retained
FinancePy-compatible flat-smile edge behavior instead of amplifying quote noise
through division by a near-zero span.

Generated adapter code then reports ``fair_strike_variance`` and settles

.. math::

   PV = N D(0,T)
        \left(K_{\mathrm{var,fair}}-K_{\mathrm{var,strike}}\right).

When explicit volatility quotes are absent, the adapter samples the admitted
Black surface at the strike grid.  Explicit quote grids are contractual inputs
and therefore do not move under a separate surface bump.  This approximation
is not model-free log-contract replication: Trellis does not yet expose the
option-strip integration, truncation, interpolation, and error contract needed
for that claim.  The retained analytical wrapper is comparison evidence only.

Extrema are exact only over the listed discrete observations.  The existing
``brownian_bridge(...)`` utility constructs Brownian paths and bridge-ordered
increments; it does not sample conditional transition extrema.  Continuous
extrema use the separate transition-state contract below.  The deterministic
``PathReducer.update(...)`` interface remains unchanged and must not be
presented as a continuous-monitoring correction.

Full-path operations preserve the differentiable backend away from extrema
ties.  Extrema remain nonsmooth at ties, and true streaming reduced-state AD
is outside the current engine support contract recorded in ``LIMITATIONS.md``.

Conditional Transition Extrema
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``trellis.models.monte_carlo.transition_state`` represents statistics defined
between two simulated endpoints.  ``ScalarTransitionObservation`` carries the
previous and current scalar cross-sections, interval times, the process-owned
bridge coordinate and integrated variance, and an independent uniform draw.
``ScalarTransitionReducer`` consumes that observation without widening or
changing deterministic ``PathReducer`` callbacks.

The first admitted kernel is the exact conditional extremum of a scalar log
diffusion.  If :math:`x_0=\log S_{t_0}`, :math:`x_1=\log S_{t_1}` and
:math:`v` is integrated log variance on the interval, then for
:math:`U\sim\mathcal U(0,1)` Trellis samples

.. math::

   X_{\max/\min}
   = \frac{x_0+x_1 \pm
      \sqrt{(x_0-x_1)^2-2v\log(1-U)}}{2},
   \qquad S_{\max/\min}=e^{X_{\max/\min}}.

``GBM`` supplies :math:`v=\sigma^2(t_1-t_0)`.  The engine admits this state
only with exact constant-parameter scalar GBM transitions. Euler, Milstein,
custom schemes, piecewise parameter regimes, vector processes, local or
stochastic volatility, jumps, and processes without an explicit
conditional-bridge capability fail closed. A piecewise log diffusion cannot in
general be reduced to endpoint values and total variance because its
conditional mean may be curved across the interval.

Generated code composes the primitive as follows:

.. code-block:: python

   from trellis.models.monte_carlo.engine import MonteCarloEngine
   from trellis.models.monte_carlo.path_state import (
       MonteCarloPathRequirement,
       StateAwarePayoff,
   )
   from trellis.models.monte_carlo.transition_state import (
       ConditionalBridgeExtremumContract,
       build_conditional_bridge_extremum_reducer,
       replay_scalar_transition_reducers,
   )
   from trellis.models.monte_carlo.variance_reduction import (
       sobol_transition_inputs,
   )
   from trellis.models.processes.gbm import GBM

   contract = ConditionalBridgeExtremumContract(
       n_steps=64,
       transition_steps=tuple(range(1, 65)),
       direction="maximum",
   )
   reducer = build_conditional_bridge_extremum_reducer(
       contract,
       name="conditional_maximum",
   )
   process = GBM(mu=carry, sigma=volatility)
   random_inputs = sobol_transition_inputs(
       n_paths=65_536,
       n_steps=contract.n_steps,
       seed=17,
   )
   payoff = StateAwarePayoff(
       path_requirement=MonteCarloPathRequirement(
           transition_reducers=(reducer,),
       ),
       evaluate_paths_fn=lambda paths: settle(
           replay_scalar_transition_reducers(
               paths,
               process=process,
               maturity=maturity,
               reducers=(reducer,),
               transition_uniforms=random_inputs.transition_uniforms,
           )["conditional_maximum"]
       ),
       evaluate_state_fn=lambda state: settle(
           state.reduced_value("conditional_maximum")
       ),
   )
   result = MonteCarloEngine(
       process,
       n_paths=65_536,
       n_steps=contract.n_steps,
       method="exact",
   ).price(
       spot,
       maturity,
       payoff,
       discount_rate=rate,
       random_inputs=random_inputs,
       return_paths=False,
   )

Pseudo-random execution draws bridge uniforms from a private seeded transition
stream, so requesting the reducer does not perturb process paths.  Explicit and
QMC execution uses ``MonteCarloRandomInputs``.  ``sobol_transition_inputs(...)``
allocates process-normal and transition-uniform coordinates from one joint
scrambled Sobol point and transforms only the process coordinates to normals.
Reusing a process shock as a bridge uniform, or silently mixing pseudo uniforms
into QMC, is invalid.

The current engine accepts one stochastic transition reducer per requirement.
Sampling minimum and maximum independently would not provide their joint law,
so simultaneous stochastic reducers fail closed.  Strike, option direction,
notional, discounting, prior extrema, and output construction remain
caller-owned settlement.  ``replay_scalar_transition_reducers(...)`` provides
full-path evidence when given the same process, maturity, reducer, and uniform
matrix.

The first migrated vanilla cases now use that boundary directly:

- vanilla European Monte Carlo lowers onto ``EventAwareMonteCarloIR`` as a
  terminal-only ``gbm_1d`` family instance rather than a synthetic event-replay
  instance
- vanilla European Monte Carlo route authority is the product-neutral
  ``price_single_state_terminal_claim_monte_carlo_result(...)`` estimator plus
  an explicit terminal-payoff callback. Generated call/put adapters bind
  ``terminal_intrinsic_from_resolved(...)`` and choose the requested scheme and
  variance-reduction controls; the product-level wrapper is retained only for
  compatibility and reference use
- the vanilla Monte Carlo estimator and transform wrappers share the
  single-state diffusion resolver/GBM-support layer under
  ``trellis.models.resolution`` for settlement, maturity, spot, dividend,
  discount, vol, and characteristic-function binding
- vanilla American/Bermudan equity Monte Carlo now resolves the
  ``exercise_monte_carlo`` route to a primitive composition. Generated adapters
  use ``resolve_single_state_monte_carlo_inputs(...)`` for market and numerical
  controls, build ``GBM`` and ``MonteCarloEngine`` paths, evaluate
  ``terminal_intrinsic_from_resolved(...)``, and pass the paths and exercise
  schedule to ``longstaff_schwartz(...)``. The product-level American LSM helper
  remains a compatibility/reference surface, not route authority.
- FX vanilla analytical routes now compose
  ``resolve_fx_vanilla_inputs(...)`` with
  ``garman_kohlhagen_price_raw(...)`` and apply notional at the adapter
  boundary. Monte Carlo routes compose the same resolved contract with
  ``GBM``, ``MonteCarloEngine``, ``terminal_value_payoff(...)``, and
  ``terminal_intrinsic(...)``. The terminal-only payoff contract avoids full
  path storage, uses domestic-minus-foreign drift, and discounts with the
  domestic rate. Product-level FX vanilla wrappers remain
  compatibility/reference surfaces, not route authority.
- FX single-barrier analytical routes now compose
  ``resolve_fx_barrier_inputs(...)`` with the scalar
  ``barrier_option_price(...)`` kernel. Monte Carlo routes compose the same
  resolved contract with ``GBM``, ``MonteCarloEngine``, ``BarrierMonitor``,
  ``MonteCarloPathRequirement``, ``StateAwarePayoff``, and
  ``terminal_intrinsic(...)``. The MC path contract stores terminal state and a
  barrier-hit flag rather than retaining full paths. Product-level FX barrier
  wrappers remain compatibility/reference surfaces, not route authority.
  When an observation frequency is not supplied, resolution derives it from
  the Monte Carlo grid so analytical and MC monitoring contracts remain
  aligned for task-level comparison.
- equity variance-swap comparison targets now have two primitive-composed task
  lanes. The Monte Carlo lane assembles a scalar market resolver, squared-log-
  return reducer, GBM, and generic engine. The analytical lane assembles exact
  time, quote-grid interpolation, discounting, and adapter-owned
  FinancePy-compatible smile-slope settlement. The retained product wrappers
  remain independent comparison evidence. Variance swaps deliberately skip
  the generic embedded-option flat-vega invariant; their proof contract is
  price sanity plus cross-method fair-strike/price comparison, not option vega
  or full log-contract replication.
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
- European analytical digital routes are primitive-composed. They bind the
  contract day count, rate, Black volatility, dividend carry, and option type
  through ``resolve_single_state_diffusion_inputs(...)``. They build the
  forward through ``forward_from_dividend_yield(...)`` and then select one of
  ``black76_cash_or_nothing_call``, ``black76_cash_or_nothing_put``,
  ``black76_asset_or_nothing_call``, or
  ``black76_asset_or_nothing_put`` from the payoff semantics. These kernels
  return undiscounted basis values; the adapter applies discount and notional
  once through ``discounted_value(...)`` and applies the cash amount only to
  cash settlement. Shared cash/asset intrinsic primitives own strict expiry
  semantics. The retained
  ``price_equity_digital_option_analytical`` function is a cash-digital
  compatibility/reference surface, not route construction authority.
- Digital proof routes now include a bounded one-dimensional PDE helper,
  ``trellis.models.equity_option_pde.price_equity_digital_option_pde``. It
  prices cash-or-nothing and asset-or-nothing equity digitals with the shared
  Black-Scholes theta solver and optional Rannacher startup smoothing, so
  Crank-Nicolson/Rannacher comparison targets can delegate to a checked helper
  instead of regenerating discontinuous-terminal grid code.
- Arithmetic-Asian proof routes now compose from product-neutral primitives.
  Both lanes start with ``resolve_single_state_diffusion_inputs(...)`` and the
  same explicit observation schedule. The analytical lane combines
  ``single_factor_lognormal_sum_contract(...)``, exact weighted moments,
  ``match_lognormal_moments(...)``, and a Black-76 call or put kernel. The
  Monte Carlo lane combines ``WeightedObservationContract``,
  ``weighted_observation_payoff(...)``, ``GBM``, and ``MonteCarloEngine`` with
  reduced-state execution. Product-level Asian functions remain independent
  comparison references and are not generated-route authority.
- Fixed-lookback MC proof routes use
  the scalar-diffusion resolver, normalized call/put semantics,
  ``ConditionalBridgeExtremumContract``, exact ``GBM``, ``StateAwarePayoff``,
  and ``MonteCarloEngine``. The route sets every exact transition, initializes
  the reducer with the contractual running maximum or minimum, and rejects
  discrete monitoring rather than silently using endpoint extrema. Generated
  adapter code owns expiry settlement, strike, notional, discounting, and
  estimator diagnostics. Integer seeds make the independent process and bridge
  streams reproducible; ``seed=None`` preserves nondeterministic execution.
  The adapter validates the generic engine's population-dispersion standard
  error internally, while the scalar ``Payoff.evaluate()`` result remains the
  final PV. Comparison with the retained structured sample-dispersion helper is
  statistical, not pathwise. The product-level lookback function remains a
  compatibility and independent-comparison reference, not route authority.
  Sparse strike/monitoring semantics, floating strike, discrete monitoring,
  and unsupported dynamics produce structured contract or numerical blockers
  before generic Monte Carlo generation.
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
- FX vanilla and quanto market resolution remain product-semantic boundaries,
  but generated pricing adapters bind resolved inputs directly to reusable
  numerical primitives. For quanto analytics, the route composes
  ``resolve_quanto_inputs``, ``quanto_adjusted_forward``, the Black-76 call/put
  kernels, explicit expiry handling, and domestic discounting. The simulation
  route composes the same resolver with ``CorrelatedGBM``,
  ``MonteCarloEngine``, and a terminal-only payoff. QMC supplies seeded
  two-factor Sobol shocks to that same engine and rounds the path count to a
  power of two. Product-level FX vanilla and quanto pricing wrappers remain
  compatibility/reference APIs; they are not generated-route construction
  authority. The quanto market boundary is no longer single-surface:
  ``MarketState.vol_surfaces`` preserves
  named implied-volatility objects, and the quanto spec can bind an exact
  underlier id plus independent underlier/FX surface keys. The resolver records
  those object names in provenance and refuses missing exact keys. Explicit
  quanto correlation descriptors also take precedence over legacy ``rho`` so
  an ambient Heston pack cannot silently define cross-asset dependence
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
- scheduled equity reset-return comparisons now assemble from
  ``ObservationReturnContract`` and product-neutral numerical primitives. An
  unbounded analytical lane prices each reset interval with ``black76_call`` or
  ``black76_put`` and explicit carry and discounting. A locally or globally
  bounded analytical lane evaluates ``bounded_observation_return_sum`` inside
  ``gauss_hermite_product_expectation``. The Monte Carlo lane combines
  ``observation_return_payoff``, ``PiecewiseConstantGBM``, and
  ``MonteCarloEngine`` with one drift/volatility pair per reset interval and
  ``return_paths=False`` so only the previous level and accumulated return are
  retained. The product-level cliquet pricing functions remain independent
  comparison and compatibility references; calling one does not satisfy a
  generated route's construction contract.

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
helper-backed Black76 swaption routes, the primitive-composed vanilla-equity
PDE route, bounded CEV PDE/tree proof helpers, bounded event-aware PDE helper
branches, Heston ADI diagnostic scaffold, double-barrier payoff primitives,
and vanilla-equity transform helper keep backend binding, admissibility, and
validation ownership explicit. Exact-helper validation enforces the thin
``(market_state, spec, ...)`` call surface only for true checked route helpers;
primitive-only supports, including vanilla theta-method PDE, the analytical,
Monte Carlo, and QMC quanto lanes, and scheduled observation-return lanes,
require agent-written route assembly rather than a product/method wrapper.
Their lowering records primitive targets separately from true ``route_helper``
bindings so generic resolvers and kernels are not mislabeled as helper authority
in prompts or traces.

The admitted European analytical digital lane is also primitive-only. Its
compact hot start is ``digital_option_composition`` in
``canonical/api_map.yaml``. Canonical decomposition and route evidence state
the resolver, basis selection, and scaling obligations, while exact symbol
validation resolves through the import registry. Cash/asset settlement and
call/put orientation remain contract choices; a product wrapper cannot
substitute for that semantic selection even when one numeric case is close.
API-map rendering selects this card from typed digital product/payoff cues
rather than from a separate global family list. Other composition cards remain
reachable through the complete bounded catalog without being injected into
unrelated builder prompts.

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

Analytical Probability And Critical-State Composition
-----------------------------------------------------

Analytical adapters can compose Gaussian probability terms from the public,
product-neutral kernels in ``trellis.models.analytical.support.probability``.
``standard_normal_cdf(x)`` supplies :math:`\Phi(x)`, while
``bivariate_standard_normal_cdf(x, y, rho)`` supplies
:math:`\Phi_2(x, y; \rho)`. Inputs must be finite and correlation is admitted
only on the closed interval :math:`[-1, 1]`; the exact singular boundaries are
handled explicitly rather than perturbed onto a nearby model.

These are scalar SciPy-backed functions, not automatic-differentiation
primitives. They provide probability integration only. A derivative adapter
still owns the contractual formula, state transformation, discounting,
notional, and call/put or exercise branching.

When an analytical formula requires an implicit critical state, the adapter
should define its scalar residual and lower it onto the existing typed solve
surface:

1. put the residual in a ``root_scalar`` ``ObjectiveBundle``
2. provide finite bracketing bounds with ``SolveBounds``
3. construct ``SolveRequest`` with ``solver_hint="brentq"``
4. call ``execute_solve_request`` and consume the checked ``SolveResult``

This keeps the numerical policy bounded and auditable. Compound and other
critical-state formulas should not create route-local Newton loops or promote
a product pricing helper as construction authority. The semantic API map card
``analytical_gaussian_composition`` is the builder's general hot start for
these probability and root-solving pieces.

Fixed lookback analytical composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The admitted European fixed-strike continuous-lookback lane uses the dedicated
``fixed_lookback_analytical_composition`` card. It exposes
``resolve_scalar_diffusion_market_inputs``, ``year_fraction``,
``normalized_option_type``, ``discount_factor_from_zero_rate``, and
``standard_normal_cdf``. A scalar root and bivariate Gaussian probability are
not part of this formula, so the lookback alias does not select the general
critical-state card.

Let :math:`S_0` be runtime spot, :math:`K` the fixed strike, :math:`T` time to
expiry, :math:`r` the continuously compounded rate, :math:`q` dividend carry,
and :math:`\sigma` the Black volatility resolved at :math:`(T,K)`. Let
:math:`H_{\max}` be the observed running maximum for a call and
:math:`H_{\min}` the observed running minimum for a put. Runtime spot remains
part of the observed path state, so define

.. math::

   M = \max(H_{\max},S_0), \qquad
   m = \min(H_{\min},S_0), \qquad
   B_C = \max(K,M), \qquad
   B_P = \min(K,m).

For either boundary :math:`B`, write :math:`b=r-q` and

.. math::

   d_1(B) =
   \frac{\log(S_0/B)+(b+\tfrac12\sigma^2)T}{\sigma\sqrt{T}},
   \qquad
   d_2(B)=d_1(B)-\sigma\sqrt{T}.

For nonzero carry, the correction terms are

.. math::

   \begin{aligned}
   A_C(B;b) ={}& \frac{\sigma^2}{2b}
      \left[-\left(\frac{S_0}{B}\right)^{-2b/\sigma^2}
      \Phi\!\left(d_1(B)-\frac{2b\sqrt{T}}{\sigma}\right)
      +e^{bT}\Phi(d_1(B))\right], \\
   A_P(B;b) ={}& \frac{\sigma^2}{2b}
      \left[\left(\frac{S_0}{B}\right)^{-2b/\sigma^2}
      \Phi\!\left(-d_1(B)+\frac{2b\sqrt{T}}{\sigma}\right)
      -e^{bT}\Phi(-d_1(B))\right].
   \end{aligned}

The per-unit-notional prices are

.. math::

   \begin{aligned}
   V_C ={}& e^{-rT}(M-K)^+
      +S_0e^{-qT}\Phi(d_1(B_C))
      -B_Ce^{-rT}\Phi(d_2(B_C))
      +S_0e^{-rT}A_C(B_C;b), \\
   V_P ={}& e^{-rT}(K-m)^+
      -S_0e^{-qT}\Phi(-d_1(B_P))
      +B_Pe^{-rT}\Phi(-d_2(B_P))
      +S_0e^{-rT}A_P(B_P;b).
   \end{aligned}

The apparent :math:`1/b` singularity is removable. With

.. math::

   d_1^0(B)=
   \frac{\log(S_0/B)+\tfrac12\sigma^2T}{\sigma\sqrt{T}},

the checked adapter uses the analytic limits

.. math::

   \begin{aligned}
   A_C(B;0) ={}&
      \left(\log(S_0/B)+\tfrac12\sigma^2T\right)\Phi(d_1^0(B))
      +\sigma\sqrt{T}\,\phi(d_1^0(B)), \\
   A_P(B;0) ={}&
      \left(-\log(S_0/B)-\tfrac12\sigma^2T\right)\Phi(-d_1^0(B))
      +\sigma\sqrt{T}\,\phi(d_1^0(B)).
   \end{aligned}

The historical maximum must be at least contract spot and the historical
minimum must be at most contract spot. At expiry the adapter settles
:math:`(M-K)^+` or :math:`(K-m)^+` directly. It rejects floating strike,
discrete monitoring, non-European exercise, non-positive spot, strike, or
volatility, unsupported dynamics, and non-finite formula results. Notional is
applied once after the selected formula. The retained product wrapper is
independent comparison evidence, not construction authority.

The adapter prefers ``MarketState.spot`` over the contract fallback and updates
the effective path extreme with that spot. The F011 FinancePy binding uses the
same product-neutral central ``bump_and_reprice`` Delta policy; no
lookback-specific Greek helper is required.

Simple chooser composition
~~~~~~~~~~~~~~~~~~~~~~~~~~

The admitted European simple-chooser lane uses the more complete
``chooser_option_composition`` card. It joins the scalar-diffusion market
resolver, contractual time fractions, Black-76 call/put kernels, discount and
forward support, bivariate Gaussian probabilities, and the typed scalar-root
surface. The checked adapter owns the derivative formula; the retained
``price_equity_chooser_option_analytical`` wrapper is comparison evidence and
does not appear in promoted route or backend construction authority.

Let :math:`t_c` be the choice time, :math:`T_C` and :math:`T_P` the call and
put expiries, and :math:`K_C` and :math:`K_P` their strikes. The admitted
market projection resolves one constant :math:`r`, :math:`q`, and
:math:`\sigma` at the longest expiry, using :math:`K_C` as the explicit
volatility coordinate. At the choice date, the critical stock state
:math:`S^*` solves

.. math::

   C_{BS}(S^*, K_C, T_C-t_c)
   - P_{BS}(S^*, K_P, T_P-t_c) = 0.

The adapter builds each Black-Scholes value from an equity forward, a discount
factor, and the public Black-76 kernel. It submits the residual through one
``root_scalar`` ``SolveRequest``. The checked F012 adapter uses the finite
positive bracket

.. math::

   [10^{-8}M, 10^6M], \qquad
   M = \max(S_0, K_C, K_P, 1),

and verifies a sign change before solving. This is a deterministic admission
policy, not a claim that every chooser contract has a root on that bracket.

For the solved state, define

.. math::

   \begin{aligned}
   d_1 &= \frac{\log(S_0/S^*) + (r-q+\tfrac12\sigma^2)t_c}
                 {\sigma\sqrt{t_c}},
   & d_2 &= d_1-\sigma\sqrt{t_c}, \\
   y_C &= \frac{\log(S_0/K_C) + (r-q+\tfrac12\sigma^2)T_C}
                 {\sigma\sqrt{T_C}},
   & y_P &= \frac{\log(S_0/K_P) + (r-q+\tfrac12\sigma^2)T_P}
                 {\sigma\sqrt{T_P}}, \\
   \rho_C &= \sqrt{t_c/T_C},
   & \rho_P &= \sqrt{t_c/T_P}.
   \end{aligned}

With :math:`\Phi_2` denoting the bivariate standard-normal CDF, the price per
unit notional is

.. math::

   \begin{aligned}
   V ={}& S_0 e^{-qT_C}\Phi_2(d_1,y_C;\rho_C)
      - K_C e^{-rT_C}\Phi_2(d_2,y_C-\sigma\sqrt{T_C};\rho_C) \\
      &- S_0 e^{-qT_P}\Phi_2(-d_1,-y_P;\rho_P)
      + K_P e^{-rT_P}\Phi_2(-d_2,-y_P+\sigma\sqrt{T_P};\rho_P).
   \end{aligned}

The admitted lane fails closed unless settlement is strictly before the choice
date and the choice date is strictly before both expiries. Spot, strikes, and
volatility must be positive and finite. Immediate-choice, choice-at-expiry,
zero-volatility, time-varying coefficient, local-volatility, and
stochastic-volatility chooser claims are outside this formula rather than
silently narrowed onto it.

For risk, the adapter first uses the selected runtime ``MarketState.spot`` and
falls back to the contract spot only when no runtime spot is bound. That keeps
the pricing formula compatible with the product-neutral central spot-bump
``Delta`` measure. The F012 FinancePy binding requests the same generic
``bump_and_reprice`` policy; no chooser-specific Greek helper is required.

European compound composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The admitted European compound lane uses the complete
``compound_option_composition`` card. Like the chooser lane, it joins one
scalar-diffusion market projection, contractual time fractions, Black-76
call/put kernels, discount and forward support, Gaussian probabilities, and
the typed scalar-root surface. The checked adapter owns the four
call/put-on-call/put formulas. The retained
``price_equity_compound_option_analytical`` wrapper is comparison evidence and
does not appear in route, backend, or deterministic generation authority.

Let :math:`t_o` be the outer expiry, :math:`T_i` the inner expiry,
:math:`K_o` the outer strike, and :math:`K_i` the inner strike, with

.. math::

   0 < t_o < T_i.

The admitted market projection resolves one constant :math:`r`, :math:`q`,
and :math:`\sigma` at :math:`T_i`, using :math:`K_i` as the explicit
volatility coordinate. At :math:`t_o`, the positive critical stock state
:math:`S^*` solves

.. math::

   V_i(S^*, K_i, T_i-t_o) - K_o = 0,

where :math:`V_i` is the Black-Scholes value of the inner call or put. The
adapter constructs :math:`V_i` from
``forward_from_dividend_yield``, ``discount_factor_from_zero_rate``,
``discounted_value``, and the matching public Black-76 kernel. It verifies a
sign change over the finite positive bracket

.. math::

   [10^{-8}M, 10^6M], \qquad
   M = \max(S_0, K_o, K_i, 1),

before submitting one ``root_scalar`` ``SolveRequest`` with the bounded
Brent solver. A missing sign change is an honest admission failure; it is not
permission to switch to an unbounded route-local Newton loop.

For the solved state, define

.. math::

   \begin{aligned}
   a_1 &= \frac{\log(S_0/S^*) + (r-q+\tfrac12\sigma^2)t_o}
                 {\sigma\sqrt{t_o}},
   & a_2 &= a_1-\sigma\sqrt{t_o}, \\
   b_1 &= \frac{\log(S_0/K_i) + (r-q+\tfrac12\sigma^2)T_i}
                 {\sigma\sqrt{T_i}},
   & b_2 &= b_1-\sigma\sqrt{T_i}, \\
   \rho &= \sqrt{t_o/T_i}.
   \end{aligned}

With :math:`\Phi` and :math:`\Phi_2` denoting the univariate and bivariate
standard-normal CDFs, the four per-unit-notional values are

.. math::

   \begin{aligned}
   V_{CC} ={}& S_0e^{-qT_i}\Phi_2(a_1,b_1;\rho)
      - K_ie^{-rT_i}\Phi_2(a_2,b_2;\rho)
      - K_oe^{-rt_o}\Phi(a_2), \\
   V_{PC} ={}& K_ie^{-rT_i}\Phi_2(-a_2,b_2;-\rho)
      - S_0e^{-qT_i}\Phi_2(-a_1,b_1;-\rho)
      + K_oe^{-rt_o}\Phi(-a_2), \\
   V_{CP} ={}& K_ie^{-rT_i}\Phi_2(-a_2,-b_2;\rho)
      - S_0e^{-qT_i}\Phi_2(-a_1,-b_1;\rho)
      - K_oe^{-rt_o}\Phi(-a_2), \\
   V_{PP} ={}& S_0e^{-qT_i}\Phi_2(a_1,-b_1;-\rho)
      - K_ie^{-rT_i}\Phi_2(a_2,-b_2;-\rho)
      + K_oe^{-rt_o}\Phi(a_2).
   \end{aligned}

The first subscript is the outer option type and the second is the inner
option type. Notional is applied once after selecting the contractual subtype.
The lane fails closed for invalid option types, non-finite contract scalars,
non-positive spot, strikes, or volatility, and any date ordering outside
settlement :math:`< t_o < T_i`. Time-varying coefficient, local-volatility,
stochastic-volatility, American, and Bermudan compound claims are outside this
constant-input analytical formula.

The adapter prefers ``MarketState.spot`` over the contract fallback spot, so
the product-neutral central spot-bump ``Delta`` measure reprices the actual
runtime state. The F013 FinancePy binding declares the same
``bump_and_reprice`` fallback; no compound-specific Greek helper is required.

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
