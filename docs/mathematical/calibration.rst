Calibration Methods
===================

Calibration maps market-observed prices to model parameters.

Implied Volatility
------------------

Given an observed option price :math:`C_{\text{mkt}}`, find :math:`\sigma` such that:

.. math::

   \text{BS}(S, K, T, r, \sigma) = C_{\text{mkt}}

Newton-Raphson Method
~~~~~~~~~~~~~~~~~~~~~

.. math::

   \sigma_{n+1} = \sigma_n - \frac{\text{BS}(\sigma_n) - C_{\text{mkt}}}{\mathcal{V}(\sigma_n)}

where :math:`\mathcal{V} = \partial C / \partial\sigma` is the **vega**:

.. math::

   \mathcal{V} = S\sqrt{T}\,\phi(d_1)

**Initial guess** (Brenner-Subrahmanyam):

.. math::

   \sigma_0 \approx \sqrt{\frac{2\pi}{T}} \cdot \frac{C_{\text{mkt}}}{S}

Falls back to Brent's method for edge cases (deep ITM/OTM).

Jaeckel Rational Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Uses a rational polynomial approximation for the initial guess, then
Newton refinement. More robust than pure Newton for extreme strikes.

SABR Calibration
----------------

Given market implied vols :math:`\sigma_{\text{mkt}}(K_i)` at strikes
:math:`K_i`, calibrate SABR parameters :math:`(\alpha, \rho, \nu)` with
:math:`\beta` typically fixed:

.. math::

   \min_{\alpha, \rho, \nu} \sum_i \left[\sigma_{\text{SABR}}(K_i; \alpha, \beta, \rho, \nu) - \sigma_{\text{mkt}}(K_i)\right]^2

subject to :math:`\alpha > 0`, :math:`|\rho| < 1`, :math:`\nu > 0`.

Trellis solves this with gradient-assisted L-BFGS-B optimization. The Hagan
approximation is differentiable in the calibrated parameters, so the optimizer
can use exact gradients instead of repeated finite-difference sweeps.
The raw implied-vol kernel stays separate from the optimizer adapter; the
calibration loop consumes the gradient rather than duplicating the pricing math.

The ATM vol provides a good initial guess for :math:`\alpha`:

.. math::

   \alpha_0 \approx \sigma_{\text{ATM}} \cdot F^{1-\beta}

Dupire Local Volatility
------------------------

Dupire's formula extracts the local volatility surface from the implied
vol surface :math:`\sigma_{\text{impl}}(K, T)`:

.. math::

   \sigma_{\text{loc}}^2(K, T) = \frac{\sigma^2 + 2\sigma T\!\left(\frac{\partial\sigma}{\partial T} + rK\frac{\partial\sigma}{\partial K}\right)}
   {\left(1 + Kd_1\sqrt{T}\frac{\partial\sigma}{\partial K}\right)^2 + K^2 T\sigma\!\left(\frac{\partial^2\sigma}{\partial K^2} - d_1\sqrt{T}\!\left(\frac{\partial\sigma}{\partial K}\right)^2\right)}

where :math:`d_1 = \frac{\ln(S/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}`.

The derivatives are computed from a smooth interpolation of the implied
vol surface (bicubic spline).

**Consistency check**: for a flat implied vol surface, :math:`\sigma_{\text{loc}} = \sigma` everywhere.

Curve Bootstrapping
-------------------

Calibrate a zero-rate curve from market instruments (deposits, futures, swaps).

Our implementation uses a differentiable Newton solver (see :doc:`/user_guide/market_data`):

.. math::

   \mathbf{r}^{(k+1)} = \mathbf{r}^{(k)} - J^{-1} \left[\text{model}(\mathbf{r}^{(k)}) - \text{quotes}\right]

where :math:`J` is the Jacobian of instrument repricing w.r.t. zero rates.

Written entirely in autograd numpy — :math:`\nabla_{\text{quotes}} V(\text{portfolio})` flows
through the bootstrap.

Rates Option Calibration
------------------------

Cap/floor and European swaption quotes are often calibrated as implied Black
volatilities under a multi-curve environment. Trellis keeps the calibration
surface explicit: the result preserves the selected curve names from
``MarketState`` and records any caller-supplied volatility or correlation
source labels.

Given an observed price :math:`P_{\text{mkt}}`, the rates calibration helpers
solve:

.. math::

   P_{\text{cap/floor}}(\sigma) = P_{\text{mkt}}
   \qquad\text{or}\qquad
   P_{\text{swaption}}(\sigma) = P_{\text{mkt}}

where the pricing side uses the same OIS discount and forecast-curve selection
as the underlying route. For cap/floor workflows, the price is the discounted
sum of Black76 caplets or floorlets. For European swaptions, the price is:

.. math::

   PV = N \cdot A \cdot \text{Black76}(S, K, \sigma, T)

with :math:`S` the forward swap rate and :math:`A` the annuity assembled from
the selected discount and forecast curves.

Calibration regression checks should treat the recovered volatility and the
reported residual as the primary success criteria. Small PV drift at the level
of the root-finding tolerance is expected, so route-appropriate tolerances are
preferred over machine-epsilon equality on the repriced present value.

The swaption payoff route and the swaption calibration helper share the same
term builder for expiry, annuity, forward swap rate, and payment count. That
shared algebra keeps route pricing and calibration aligned while still letting
the calibration result report a small numerical residual.

Implementation
--------------

.. autofunction:: trellis.models.calibration.implied_vol.implied_vol
.. autofunction:: trellis.models.calibration.sabr_fit.calibrate_sabr
.. autofunction:: trellis.models.calibration.local_vol.dupire_local_vol
.. autofunction:: trellis.models.calibration.rates.calibrate_cap_floor_black_vol
.. autofunction:: trellis.models.calibration.rates.calibrate_swaption_black_vol
.. autofunction:: trellis.curves.bootstrap.bootstrap_yield_curve
   :no-index:

References
----------

- Jaeckel, P. (2015). "Let's be rational." *Wilmott*, 2015(75), 40-53.
- Hagan, P. et al. (2002). "Managing smile risk." *Wilmott Magazine*, Sep 2002.
- Dupire, B. (1994). "Pricing with a smile." *Risk*, 7(1), 18-20.
