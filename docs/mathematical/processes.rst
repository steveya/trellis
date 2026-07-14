Stochastic Processes
====================

Each process provides :math:`\mu(x,t)` and :math:`\sigma(x,t)` for the SDE
:math:`dX = \mu\,dt + \sigma\,dW`.

GBM
---

.. math::

   dS/S = \mu\,dt + \sigma\,dW

Exact: :math:`S_T = S_0\exp[(\mu-\sigma^2/2)T + \sigma\sqrt{T}Z]`.
Moments: :math:`\mathbb{E}[S_T] = S_0 e^{\mu T}`.

Piecewise-Constant GBM
----------------------

``PiecewiseConstantGBM`` keeps deterministic drift and volatility pairs on an
ordered time grid. Its exact transition integrates drift and variance across
every interval crossed by a simulation step, so the numerical grid need not
coincide with parameter boundaries. This is useful for scheduled-return
composition when each observation interval binds a different market quote; it
does not infer a local-volatility process from an implied-volatility surface.

Vasicek
-------

.. math::

   dr = a(b-r)\,dt + \sigma\,dW

Mean-reverting to :math:`b`. :math:`\mathbb{E}[r_T] = r_0 e^{-aT} + b(1-e^{-aT})`.
Rates can go negative.

CIR
---

.. math::

   dr = a(b-r)\,dt + \sigma\sqrt{r}\,dW

Positive rates when Feller condition :math:`2ab > \sigma^2` holds.

Plain Short-Rate Zero-Coupon Bonds
----------------------------------

The bounded proof route for plain zero-coupon bonds under one-factor affine
short-rate models lives in ``trellis.models.short_rate_bond``. It exposes
Vasicek and CIR analytical prices plus a bounded trinomial tree comparator for
the same ``(market_state, spec)`` contract. The resolver consumes explicit
``market_state.model_parameters`` / ``model_parameter_sets`` entries keyed by
``vasicek`` or ``cir`` and ignores Heston-shaped stochastic-volatility payloads
such as ``kappa/theta/xi/rho/v0``.

Sparse legacy proof tasks may opt into benchmark defaults through the internal
task exact-binding path. Ordinary helper use remains fail-closed unless the
short-rate volatility is supplied by model parameters or by an explicit
short-rate comparison regime.

Hull-White
----------

.. math::

   dr = [\theta(t) - ar]\,dt + \sigma\,dW

Time-dependent :math:`\theta(t)` calibrates exactly to the yield curve.

Heston
------

.. math::

   dS = \mu S\,dt + \sqrt{V}S\,dW_1, \quad
   dV = \kappa(\theta-V)\,dt + \xi\sqrt{V}\,dW_2, \quad
   \text{corr} = \rho

Closed-form characteristic function enables FFT/COS pricing.

Path-Dependent Heston Control Boundary
--------------------------------------

American Asian barrier-style claims under Heston combine spot/variance state,
a path summary, a barrier event monitor, and an early-exercise control policy.
Current task diagnostics recognize this as a composite control problem and
emit a ``path_dependent_control_contract`` with the missing path-state
simulation, event-monitor, payoff-summary, control-policy, stochastic-vol
coupling, and target-specific solver components. Trellis does not yet admit a
checked PDE, Monte Carlo LSM, or transform route for this composite class.

Bates Boundary
--------------

Bates-style affine jump stochastic volatility extends the Heston variance
process with compound-Poisson lognormal spot jumps:

.. math::

   \frac{dS}{S_-} = (\mu - \lambda k)\,dt + \sqrt{V}\,dW_1
      + (e^J - 1)\,dN,
   \quad
   dV = \kappa(\theta - V)\,dt + \xi\sqrt{V}\,dW_2.

The computational contract is Heston model parameters
``kappa``, ``theta``, ``xi``, ``rho``, and ``v0`` plus jump parameters
``jump_intensity``, ``jump_mean``, and ``jump_vol`` (or
``jump_variance``). The checked European vanilla support lives in
``trellis.models.bates_option``. That helper composes the existing Heston
runtime binding with compound-Poisson lognormal jumps, exposes Bates
FFT/COS transform pricing through the shared transform kernels, and exposes a
terminal Monte Carlo comparator using Heston paths plus independent jump
aggregation. It consumes explicit model and jump parameters; it does not infer
Bates parameters from a Black volatility surface.

The support boundary is still narrow: Bates calibration, path-dependent Bates
payoffs, early exercise under Bates, and Bates PIDE/PDE solvers are not
checked routes today.

SABR
----

.. math::

   dF = \hat\sigma F^\beta\,dW_1, \quad
   d\hat\sigma = \nu\hat\sigma\,dW_2, \quad
   \text{corr} = \rho

Hagan approximation gives closed-form implied vol :math:`\sigma_{\text{impl}}(K)`.
The checked European forward-style option support lives in
``trellis.models.sabr_option``. That helper resolves SABR parameters from
``spec.sabr``, named ``market_state.model_parameter_sets``, default
``market_state.model_parameters``, or synthetic-market provenance, using
canonical keys ``alpha``, ``beta``, ``rho``, and ``nu``. It exposes a Hagan
Black76 price and an Euler Monte Carlo comparator under the same runtime
contract. The helper consumes model parameters directly; it does not infer a
SABR process from a Black vol surface unless an upstream calibration workflow
has explicitly produced the SABR parameter pack.

Local Volatility
----------------

:math:`dS/S = \mu\,dt + \sigma_{\text{loc}}(S,t)\,dW`. Dupire's formula extracts
:math:`\sigma_{\text{loc}}` from the implied vol surface.

SLV/LSV Boundary
----------------

Stochastic-local-volatility and local-stochastic-volatility models couple a
local-vol surface with a stochastic-vol process through a leverage function
:math:`L(t,S)`:

.. math::

   \frac{dS}{S} = \mu\,dt + L(t,S)\sqrt{V}\,dW_1.

The diagnostic contract requires local-vol and Black-vol surface authority,
Heston model parameters, a recorded leverage-function calibration problem,
the leverage-function surface, interpolation on the ``(time, spot)`` domain,
and solver-specific PDE or Monte Carlo requirements. Trellis currently records
this contract and blocks honestly; it does not yet admit checked SLV/LSV PDE or
Monte Carlo pricing routes.

Merton Jump-Diffusion
---------------------

.. math::

   dS/S = (\mu - \lambda k)\,dt + \sigma\,dW + J\,dN

Poisson jumps :math:`N` with intensity :math:`\lambda`, log-normal jump size :math:`J`.
The checked European vanilla support lives in
``trellis.models.merton_jump_diffusion_option``. That helper resolves
``jump_parameters`` / ``jump_parameter_sets`` from the runtime ``MarketState``
using canonical keys ``sigma``, ``lam`` or ``jump_intensity``,
``jump_mean``, and ``jump_vol``. It exposes a Poisson-mixture Black reference,
FFT/COS transform pricing, and direct terminal Monte Carlo sampling. The
transform route is a model-family-specific binding; it should not be replaced
with a vanilla Black-vol adapter when the product contract says
``model_family=jump_diffusion``.

Variance Gamma, CGMY, And Kou Levy Models
-----------------------------------------

Variance Gamma, CGMY, and Kou targets keep the product shape as a European
vanilla option while narrowing the runtime model family to an explicit Levy
process. They consume model parameters from ``market_state.model_parameters``
or named ``market_state.model_parameter_sets`` entries; they do not infer Levy
parameters from a Black volatility surface.

For Variance Gamma, canonical payloads use ``sigma``, ``theta``, and ``nu``.
The checked helper in ``trellis.models.levy_option`` exposes FFT/COS transform
pricing, a Madan-Carr-Chang-style reference wrapper, and direct terminal Monte
Carlo sampling via gamma subordination.

For CGMY, canonical payloads use ``C``, ``G``, ``M``, and ``Y``. The checked
helper exposes FFT/COS transform pricing and a bounded terminal-distribution
Monte Carlo comparator obtained from the characteristic function. That
comparator is useful for proof and cross-method validation of terminal
European payoffs, but it is not a path simulator for barrier, Asian, or other
path-dependent Levy claims.

For Kou double-exponential jump diffusion, canonical payloads use ``sigma``,
``jump_intensity``, ``up_probability``, ``eta_up``, and ``eta_down``. The
checked helper exposes FFT/COS transform pricing, a high-resolution COS
reference, and direct terminal Monte Carlo sampling of the asymmetric
double-exponential jump sum. The support boundary is the same as for the other
checked Levy proof routes: European vanilla terminal claims only, with no
path-dependent monitors, early exercise, calibration bridge, or broad Levy
model selection.

Implementation
--------------

.. autoclass:: trellis.models.processes.gbm.GBM
   :members:
.. autoclass:: trellis.models.processes.gbm.PiecewiseConstantGBM
   :members:
.. autoclass:: trellis.models.processes.heston.Heston
   :members:
.. autoclass:: trellis.models.processes.sabr.SABRProcess
   :members:
.. autofunction:: trellis.models.sabr_option.price_sabr_forward_option_hagan
.. autofunction:: trellis.models.sabr_option.price_sabr_forward_option_monte_carlo
.. autofunction:: trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_transform
.. autofunction:: trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_monte_carlo
.. autofunction:: trellis.models.bates_option.price_bates_option_transform
.. autofunction:: trellis.models.bates_option.price_bates_option_monte_carlo
.. autofunction:: trellis.models.short_rate_bond.price_short_rate_zero_coupon_bond_analytical
.. autofunction:: trellis.models.short_rate_bond.price_short_rate_zero_coupon_bond_tree
.. autofunction:: trellis.models.levy_option.price_variance_gamma_option_transform
.. autofunction:: trellis.models.levy_option.price_variance_gamma_option_monte_carlo
.. autofunction:: trellis.models.levy_option.price_cgmy_option_transform
.. autofunction:: trellis.models.levy_option.price_cgmy_option_monte_carlo

References
----------

- Shreve (2004). *Stochastic Calculus for Finance II*. Springer.
- Heston (1993). *Review of Financial Studies*, 6(2), 327-343.
- Hagan et al. (2002). *Wilmott Magazine*, Sep 2002.
- Merton (1976). *Journal of Financial Economics*, 3, 125-144.
- Madan, Carr, and Chang (1998). *European Economic Review*, 42, 79-105.
- Carr, Geman, Madan, and Yor (2002). *Journal of Business*, 75(2), 305-332.
