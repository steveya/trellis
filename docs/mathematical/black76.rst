Black76 Model
=============

The Black76 model prices European options on forward rates or futures.
It is the standard model for caps, floors, and European swaptions.

Call Price (Undiscounted)
-------------------------

.. math::

   C = F \cdot N(d_1) - K \cdot N(d_2)

where:

.. math::

   d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma^2 T}{\sigma\sqrt{T}}, \quad
   d_2 = d_1 - \sigma\sqrt{T}

- :math:`F` — forward rate
- :math:`K` — strike rate
- :math:`\sigma` — Black (lognormal) volatility
- :math:`T` — time to option expiry
- :math:`N(\cdot)` — standard normal CDF

Put Price (Undiscounted)
------------------------

.. math::

   P = K \cdot N(-d_2) - F \cdot N(-d_1)

Put-Call Parity
~~~~~~~~~~~~~~~

.. math::

   C - P = F - K

This holds exactly for undiscounted prices.

FX Vanilla as Basis Assembly
----------------------------

The Garman-Kohlhagen FX European formula is the same Black76 basis
decomposition after a forward bridge.

Let:

.. math::

   F = S \cdot \frac{D_f}{D_d}

where :math:`S` is the spot FX quote, :math:`D_d` is the domestic discount
factor, and :math:`D_f` is the foreign discount factor. The domestic-currency
call PV is:

.. math::

   V_{\text{call}} = D_d \cdot \left(F N(d_1) - K N(d_2)\right)

and the put is:

.. math::

   V_{\text{put}} = D_d \cdot \left(K N(-d_2) - F N(-d_1)\right)

Equivalently, the route can be assembled from the terminal basis claims:

.. math::

   V_{\text{call}} = D_d \cdot \text{terminal\_vanilla\_from\_basis}
   (\text{call}, \text{asset}, \text{cash}, K)

where :math:`\text{asset}` and :math:`\text{cash}` are the Black76
asset-or-nothing and cash-or-nothing basis claims evaluated on the forward.

Special Cases
-------------

**Zero volatility:**

.. math::

   C(\sigma \to 0) = \max(F - K, 0), \quad
   P(\sigma \to 0) = \max(K - F, 0)

**At-the-money** (:math:`F = K`):

.. math::

   d_1 = \frac{\sigma\sqrt{T}}{2}, \quad d_2 = -\frac{\sigma\sqrt{T}}{2}

   C_{\text{ATM}} = F \left[ N\!\left(\frac{\sigma\sqrt{T}}{2}\right) - N\!\left(-\frac{\sigma\sqrt{T}}{2}\right) \right]

For small :math:`\sigma\sqrt{T}`: :math:`C_{\text{ATM}} \approx F \cdot \sigma\sqrt{T} \cdot \phi(0) \approx 0.3989 \cdot F \cdot \sigma\sqrt{T}`

Cap Pricing
-----------

An interest rate cap is a portfolio of caplets. Each caplet covers an
accrual period :math:`[t_i, t_{i+1}]` and pays:

.. math::

   \text{Caplet payoff} = N \cdot \tau_i \cdot \max(L_i - K, 0)

where :math:`L_i` is the floating rate fixing at :math:`t_i` and
:math:`\tau_i = \text{year\_fraction}(t_i, t_{i+1})`.

The caplet price (discounted to settlement) is:

.. math::

   \text{Caplet PV} = N \cdot \tau_i \cdot D(t_{i+1}) \cdot \text{Black76\_Call}(F_i, K, \sigma_i, t_i)

where :math:`F_i = F(t_i, t_{i+1})` is the forward rate and
:math:`\sigma_i` is the caplet volatility.

The cap price is the sum of all caplet prices:

.. math::

   \text{Cap PV} = \sum_{i} \text{Caplet PV}_i

Swaption Pricing
----------------

A European payer swaption gives the right to enter a payer swap at expiry.
Under the annuity measure:

.. math::

   \text{Swaption PV} = N \cdot A \cdot \text{Black76\_Call}(S, K, \sigma, T)

where:

- :math:`S` — forward swap rate
- :math:`A = \sum_j \tau_j \cdot D(T_j)` — swap annuity (PV01)
- :math:`T` — option expiry
- :math:`K` — strike (fixed rate of the underlying swap)

The forward swap rate is:

.. math::

   S = \frac{\sum_j F_j \cdot \tau_j \cdot D(T_j)}{A}

Implementation
--------------

The implementation uses autograd-compatible primitives, so the same closed-form
expressions can supply both prices and Greeks. Zero-volatility and zero-expiry
cases fall back to intrinsic value without introducing scalarized trace breaks.
These functions are the raw analytical kernels used by higher-level adapters;
the adapters resolve market inputs, then call into the kernel without adding
their own differentiation logic.

.. autofunction:: trellis.models.black.black76_call
.. autofunction:: trellis.models.black.black76_put

Numerical Example
-----------------

ATM caplet: F = K = 5%, σ = 20%, T = 1Y:

.. math::

   d_1 = \frac{0 + 0.02}{0.20} = 0.1, \quad d_2 = -0.1

   C = 0.05 \times N(0.1) - 0.05 \times N(-0.1) = 0.05 \times 2 \times 0.03983 = 0.003983

References
----------

- Black, F. (1976). "The pricing of commodity contracts."
  *Journal of Financial Economics*, 3(1-2), 167-179.
- Hull, J. C. (2022). *Options, Futures, and Other Derivatives*, 11th ed. Pearson.
