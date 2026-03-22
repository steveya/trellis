Pricing
=======

Three Ways to Price
-------------------

**1. Direct** (quant writing code):

.. code-block:: python

   from trellis import price, Bond, YieldCurve
   result = price(bond, curve, settlement)

**2. Session** (trader in a notebook):

.. code-block:: python

   s = Session(curve=curve)
   result = s.price(bond)

**3. Natural language** (anyone):

.. code-block:: python

   result = trellis.ask("Price a 5Y cap at 4%")

The Payoff Framework
--------------------

For instruments beyond bonds, use the ``Payoff`` protocol:

.. code-block:: python

   from trellis import CapPayoff, CapFloorSpec, price_payoff, MarketState
   from trellis.core.types import Frequency

   spec = CapFloorSpec(notional=1e6, strike=0.04,
                        start_date=start, end_date=end,
                        frequency=Frequency.QUARTERLY)
   cap = CapPayoff(spec)

   ms = MarketState(as_of=settle, settlement=settle,
                     discount=curve, vol_surface=FlatVol(0.20))
   pv = price_payoff(cap, ms)

Return Types
~~~~~~~~~~~~

``evaluate()`` returns either:

- **Cashflows** — undiscounted dated cashflows (bonds, caps, swaps)
- **PresentValue** — already-discounted PV (tree, MC, PDE methods)

``price_payoff()`` handles both automatically.

Greeks
------

Greeks are computed via automatic differentiation (autograd):

.. code-block:: python

   result = s.price(bond, greeks="all")
   print(result.greeks["dv01"])
   print(result.greeks["duration"])
   print(result.greeks["convexity"])
   print(result.greeks["key_rate_durations"])

   # Selective:
   result = s.price(bond, greeks=["dv01", "duration"])
   # or skip Greeks entirely:
   result = s.price(bond, greeks=None)

Missing Data Errors
-------------------

If required market data is missing, you get a helpful error:

.. code-block:: text

   MissingCapabilityError: Missing market data: ['black_vol']
     Missing market data: 'black_vol' — Black (lognormal) implied volatility surface.
       How to provide: Session(vol_surface=FlatVol(0.20))
