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
``jump_intensity``, ``jump_mean``, and ``jump_variance``. Current diagnostics
recognize this contract and emit the missing
``bates_affine_jump_stochastic_vol_kernel`` primitive for Bates transform or
Monte Carlo targets. Trellis does not yet admit a checked Bates characteristic
function or simulation route.

SABR
----

.. math::

   dF = \hat\sigma F^\beta\,dW_1, \quad
   d\hat\sigma = \nu\hat\sigma\,dW_2, \quad
   \text{corr} = \rho

Hagan approximation gives closed-form implied vol :math:`\sigma_{\text{impl}}(K)`.

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

Implementation
--------------

.. autoclass:: trellis.models.processes.gbm.GBM
   :members:
.. autoclass:: trellis.models.processes.heston.Heston
   :members:
.. autoclass:: trellis.models.processes.sabr.SABRProcess
   :members:
.. autofunction:: trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_transform
.. autofunction:: trellis.models.merton_jump_diffusion_option.price_merton_jump_diffusion_option_monte_carlo

References
----------

- Shreve (2004). *Stochastic Calculus for Finance II*. Springer.
- Heston (1993). *Review of Financial Studies*, 6(2), 327-343.
- Hagan et al. (2002). *Wilmott Magazine*, Sep 2002.
- Merton (1976). *Journal of Financial Economics*, 3, 125-144.
