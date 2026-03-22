FX Pricing
==========

Covered Interest Rate Parity
-----------------------------

The FX forward rate is determined by the no-arbitrage condition:

.. math::

   F(T) = S_0 \cdot \frac{D_{\text{for}}(T)}{D_{\text{dom}}(T)}

where:

- :math:`S_0` — spot FX rate (units of domestic per unit of foreign)
- :math:`D_{\text{for}}(T)` — foreign currency discount factor
- :math:`D_{\text{dom}}(T)` — domestic currency discount factor

Under continuous compounding:

.. math::

   F(T) = S_0 \cdot e^{(r_{\text{dom}} - r_{\text{for}}) \cdot T}

Forward Points
~~~~~~~~~~~~~~

.. math::

   \text{Forward points} = F(T) - S_0

When :math:`r_{\text{dom}} > r_{\text{for}}`, forward points are positive
(forward premium).

Cross-Currency Payoff Conversion
---------------------------------

A foreign-currency payoff :math:`\pi_{\text{for}}` at time :math:`T`
has domestic present value:

.. math::

   PV_{\text{dom}} = \pi_{\text{for}} \cdot F(T) \cdot D_{\text{dom}}(T)

This simplifies to:

.. math::

   PV_{\text{dom}} = \pi_{\text{for}} \cdot S_0 \cdot D_{\text{for}}(T)

Implementation
--------------

.. autoclass:: trellis.instruments.fx.FXForward
   :members:

.. autoclass:: trellis.instruments.fx.FXForwardPayoff
   :members:

Numerical Example
-----------------

EURUSD spot = 1.10, USD rate = 5%, EUR rate = 3%, T = 1Y:

.. math::

   F(1) = 1.10 \times \frac{e^{-0.03}}{e^{-0.05}} = 1.10 \times e^{0.02} = 1.1222

Forward points = 0.0222 (222 pips).

References
----------

- Hull, J. C. (2022). *Options, Futures, and Other Derivatives*, 11th ed. Ch. 5.
- Wystup, U. (2017). *FX Options and Structured Products*, 2nd ed. Wiley.
