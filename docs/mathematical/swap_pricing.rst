Interest Rate Swap Pricing
==========================

A fixed-for-floating interest rate swap exchanges fixed coupon payments
for floating rate payments based on a reference index.

Swap Valuation
--------------

The value of a payer swap (pay fixed, receive floating) is:

.. math::

   V_{\text{payer}} = V_{\text{float}} - V_{\text{fixed}}

Fixed Leg
~~~~~~~~~

.. math::

   V_{\text{fixed}} = N \cdot R \sum_{j=1}^{M} \tau_j^{\text{fix}} \cdot D(T_j)

where :math:`R` is the fixed rate, :math:`\tau_j^{\text{fix}}` is the
fixed-leg day count fraction, and :math:`M` is the number of fixed periods.

Floating Leg
~~~~~~~~~~~~

.. math::

   V_{\text{float}} = N \sum_{i=1}^{m} F_i \cdot \tau_i^{\text{flt}} \cdot D(T_i)

where :math:`F_i = F(t_i, T_i)` is the forward rate for the floating period
and :math:`\tau_i^{\text{flt}}` is the floating-leg day count fraction.

Par Swap Rate
~~~~~~~~~~~~~

The par swap rate :math:`S` makes the swap value zero:

.. math::

   S = \frac{\sum_{i} F_i \cdot \tau_i^{\text{flt}} \cdot D(T_i)}
            {\sum_{j} \tau_j^{\text{fix}} \cdot D(T_j)}

   = \frac{V_{\text{float}} / N}{A}

where :math:`A = \sum_j \tau_j^{\text{fix}} \cdot D(T_j)` is the **annuity**
(also called PV01 or DV01 of the fixed leg).

Multi-Curve Framework
---------------------

Post-2008, discounting and forecasting use different curves:

- **OIS curve** (e.g., SOFR) for discounting: :math:`D(t)`
- **Forecast curve** (e.g., EURIBOR-6M) for forward rates: :math:`F_i`

The par swap rate in multi-curve:

.. math::

   S = \frac{\sum_{i} F_i^{\text{forecast}} \cdot \tau_i \cdot D^{\text{OIS}}(T_i)}
            {\sum_{j} \tau_j \cdot D^{\text{OIS}}(T_j)}

In Trellis, this is handled by ``MarketState.forecast_forward_curve(rate_index)``
which returns a ``ForwardCurve`` built from the forecast curve, while
``MarketState.discount`` provides the OIS discounting.

Forward Rate Extraction
-----------------------

Simply compounded forward rate:

.. math::

   F(t_1, t_2) = \frac{1}{\tau} \left( \frac{D(t_1)}{D(t_2)} - 1 \right)

Continuously compounded forward rate:

.. math::

   F_{\text{cc}}(t_1, t_2) = \frac{1}{\tau} \ln \frac{D(t_1)}{D(t_2)}

**No-arbitrage identity** (simple compounding):

.. math::

   \frac{D(t_1)}{D(t_2)} = 1 + F(t_1, t_2) \cdot \tau

Swap DV01
---------

The DV01 of a par swap with notional :math:`N`:

.. math::

   \text{DV01} = N \cdot A \cdot 0.0001

For a 5-year :math:`10M swap, DV01 :math:`\approx` \$4,500 per bp.

Implementation
--------------

.. autoclass:: trellis.instruments.swap.SwapPayoff
   :members:
   :no-index:

.. autofunction:: trellis.instruments.swap.par_swap_rate
   :no-index:

References
----------

- Hull, J. C. (2022). *Options, Futures, and Other Derivatives*, 11th ed. Ch. 7.
- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models - Theory and Practice*, 2nd ed. Springer.
