FX and Credit Inputs
====================

The package-level API exposes the market inputs you need for FX-aware and
credit-aware workflows, even when the final payoff construction happens at
another layer.

FX Forwards
-----------

.. code-block:: python

   from trellis import FXForward, FXRate, YieldCurve

   eurusd = FXRate(spot=1.08, domestic="USD", foreign="EUR")
   usd_curve = YieldCurve.flat(0.045)
   eur_curve = YieldCurve.flat(0.030)

   forward = FXForward(eurusd, domestic_curve=usd_curve, foreign_curve=eur_curve)
   print(forward.forward(1.0))
   print(forward.forward_points(1.0))

Credit Curves in a Session
--------------------------

.. code-block:: python

   from datetime import date
   from trellis import CreditCurve, Session, YieldCurve

   s = Session(
       curve=YieldCurve.flat(0.045),
       settlement=date(2024, 11, 15),
       credit_curve=CreditCurve.flat(0.02),
   )

   print(s.credit_curve.survival_probability(5.0))

How to Use These Inputs
-----------------------

These package-level objects are the stable way to provide:

- FX spots and covered-interest-parity forwards
- credit and survival assumptions
- market-state components for agent or payoff workflows

Treat deeper product-specific credit structures as advanced or experimental
until they are promoted into the stable package-level surface.

Related Reading
---------------

- :doc:`../user_guide/market_data`
- :doc:`../api/instruments`
- :doc:`../api/curves`
