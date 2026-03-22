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

Merton Jump-Diffusion
---------------------

.. math::

   dS/S = (\mu - \lambda k)\,dt + \sigma\,dW + J\,dN

Poisson jumps :math:`N` with intensity :math:`\lambda`, log-normal jump size :math:`J`.

Implementation
--------------

.. autoclass:: trellis.models.processes.gbm.GBM
   :members:
.. autoclass:: trellis.models.processes.heston.Heston
   :members:
.. autoclass:: trellis.models.processes.sabr.SABRProcess
   :members:

References
----------

- Shreve (2004). *Stochastic Calculus for Finance II*. Springer.
- Heston (1993). *Review of Financial Studies*, 6(2), 327-343.
- Hagan et al. (2002). *Wilmott Magazine*, Sep 2002.
- Merton (1976). *Journal of Financial Economics*, 3, 125-144.
