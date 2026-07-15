Monte Carlo Methods
===================

.. math::

   V_0 = e^{-rT} \cdot \frac{1}{N} \sum_{k=1}^{N} f(S_T^{(k)})

When you need gradients, Trellis can run the same path generator in a
pathwise/autograd mode by supplying explicit shocks to
``MonteCarloEngine.simulate_with_shocks(..., differentiable=True)`` and using
an autograd-aware payoff callable.

SDE Discretization
------------------

**Euler-Maruyama** (order 0.5):

.. math::

   X_{t+\Delta t} = X_t + \mu(X_t, t)\Delta t + \sigma(X_t, t)\sqrt{\Delta t} \, Z

**Milstein** (order 1.0):

.. math::

   X_{t+\Delta t} = X_t + \mu\Delta t + \sigma\sqrt{\Delta t} \, Z
   + \tfrac{1}{2}\sigma\sigma'(Z^2 - 1)\Delta t

**Exact simulation** for GBM:

.. math::

   S_{t+\Delta t} = S_t \exp\!\left[(\mu - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t}\,Z\right]

**Heston quadratic-exponential** for stochastic volatility:

Trellis exposes a checked two-state Heston Monte Carlo helper for European
vanilla options through
``trellis.models.monte_carlo.stochastic_vol.price_heston_option_monte_carlo``.
The helper resolves spot, discounting, and explicit Heston model parameters
into a ``(S_t, V_t)`` simulation problem, then selects either the vector-state
Euler scheme or the Andersen-style quadratic-exponential variance scheme.

Use ``scheme="heston_qe"`` for ``qe_heston`` targets and ``scheme="euler"`` for
``euler_heston`` / ``heston_mc`` targets. The helper treats ``v0`` and
``theta`` as variances. It does not infer Heston parameters from a Black
volatility surface; that bridge belongs to an explicit calibration problem.

**Bates Heston-plus-jump terminal Monte Carlo**:

``trellis.models.bates_option.price_bates_option_monte_carlo`` prices
European vanilla Bates options by simulating the Heston spot/variance state
with an explicit Heston parameter pack and then applying independent
compound-Poisson lognormal terminal jumps. The helper resolves
``jump_parameters`` / ``jump_parameter_sets`` with canonical keys
``jump_intensity`` (or ``lam`` / ``lambda``), ``jump_mean``, and ``jump_vol``
(or ``jump_variance``). It is a terminal European comparator for the checked
Bates transform route, not a path-state contract for barrier, Asian, or early
exercise claims under Bates.

**Levy terminal sampling** for European vanilla proof comparisons:

``trellis.models.levy_option`` exposes bounded Monte Carlo helpers for
Variance Gamma, CGMY, and Kou European vanilla options. Variance Gamma uses
direct gamma-subordination terminal sampling. CGMY uses a deterministic
terminal distribution built from its characteristic function and samples that
distribution for comparison against the transform/reference routes. Kou uses
direct compound-Poisson terminal sampling of asymmetric double-exponential
jump sums. These helpers are terminal-payoff comparators; they are not path
simulators and should not be reused for barrier, Asian, or event-monitored
Levy claims.

Brownian Bridge
---------------

.. math::

   W(t_m) \mid W(t_1), W(t_2) \sim N\!\left(\frac{(t_2 - t_m)W_1 + (t_m - t_1)W_2}{t_2 - t_1},\;
   \frac{(t_m - t_1)(t_2 - t_m)}{t_2 - t_1}\right)

This midpoint construction generates a Brownian path conditional on endpoint
values. It is different from sampling the exact interval extremum. For a scalar
log diffusion with endpoint logs :math:`x_0,x_1`, integrated log variance
:math:`v`, and an independent :math:`U\sim\mathcal U(0,1)`, the conditional
maximum and minimum are sampled by

.. math::

   X_{\max/\min}
   = \frac{x_0+x_1 \pm
     \sqrt{(x_0-x_1)^2 - 2v\log(1-U)}}{2}.

``ScalarTransitionObservation`` carries those inputs and
``ConditionalBridgeExtremumContract`` selects the monitored transitions.
Constant-parameter ``GBM`` currently provides the admitted exact log bridge
variance. Piecewise parameter regimes can leave a curved conditional mean and
are not admitted merely from total variance. Other coordinates, approximate
schemes, vector state, and non-diffusion processes require a separately derived
kernel.

The fixed-strike continuous-lookback task route uses one independent bridge
uniform per monitored transition in addition to the process shock. A supplied
integer seed makes both channels reproducible without making them identical;
``seed=None`` preserves the Monte Carlo engine's nondeterministic mode. Route
comparison with the retained product helper is statistical rather than
pathwise because the transition-state engine owns a separate auxiliary stream.
The route computes and validates the generic engine convention
:math:`\operatorname{std}_{\mathrm{ddof}=0}(PV_i)/\sqrt{N}` internally. The
scalar ``Payoff.evaluate()`` contract returns only the final PV; it does not
expose that diagnostic. The retained compatibility helper returns a structured
result and uses ``ddof=1``, a difference that must not be mistaken for a price
disagreement.

Variance Reduction
------------------

**Antithetic**: average :math:`f(S^+)` and :math:`f(S^-)` with mirrored paths.

**Control variate**: :math:`\hat{V}_{\text{cv}} = \hat{V} - \beta^*(\hat{C} - \mathbb{E}[C])` with :math:`\beta^* = \text{Cov}(V,C)/\text{Var}(C)`.

**Quasi-random (Sobol)**: low-discrepancy sequences give nearly :math:`O(N^{-1})` convergence.

Canonical package surface: ``trellis.models.qmc`` re-exports Sobol normals,
joint Sobol process-normal/transition-uniform inputs, and Brownian-bridge path
helpers while estimator and transition-state logic remain in
``trellis.models.monte_carlo``. Auxiliary transition uniforms use distinct
Sobol coordinates; they are not recovered from process shocks.

Longstaff-Schwartz (LSM)
-------------------------

For American/Bermudan options: regress continuation value on basis functions at each exercise date, exercise if intrinsic > continuation estimate.

Standard error: :math:`\text{SE} = \hat\sigma / \sqrt{N}`, convergence :math:`O(N^{-1/2})`.

Implementation
--------------

.. autoclass:: trellis.models.monte_carlo.engine.MonteCarloEngine
   :members:

.. autofunction:: trellis.models.monte_carlo.lsm.longstaff_schwartz
   :no-index:
.. autoclass:: trellis.models.monte_carlo.schemes.HestonQuadraticExponential
   :members:
   :no-index:
.. autofunction:: trellis.models.monte_carlo.stochastic_vol.price_heston_option_monte_carlo
   :no-index:
.. autofunction:: trellis.models.monte_carlo.stochastic_vol.price_heston_option_monte_carlo_result
   :no-index:
.. autofunction:: trellis.models.bates_option.price_bates_option_monte_carlo
   :no-index:
.. autofunction:: trellis.models.bates_option.price_bates_option_monte_carlo_result
   :no-index:
.. autofunction:: trellis.models.qmc.sobol_normals
   :no-index:
.. autofunction:: trellis.models.qmc.brownian_bridge
   :no-index:

References
----------

- Glasserman (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Longstaff & Schwartz (2001). *Review of Financial Studies*, 14(1), 113-147.
