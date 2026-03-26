Rates and Volatility Workflows
==============================

Use this path when you want to supply your own curve, volatility, and
rate-index conventions while staying on the public package-level surface.

Build a Custom Session
----------------------

.. code-block:: python

   from datetime import date
   from trellis import FlatVol, Session, YieldCurve

   s = Session(
       curve=YieldCurve.flat(0.045),
       settlement=date(2024, 11, 15),
       vol_surface=FlatVol(0.20),
   )

Price a Cap
-----------

.. code-block:: python

   from datetime import date
   from trellis import CapFloorSpec, CapPayoff, MarketState, price_payoff
   from trellis.core.types import Frequency

   cap = CapPayoff(
       CapFloorSpec(
           notional=1_000_000,
           strike=0.04,
           start_date=date(2025, 2, 15),
           end_date=date(2030, 2, 15),
           frequency=Frequency.QUARTERLY,
       )
   )

   market_state = MarketState(
       as_of=s.settlement,
       settlement=s.settlement,
       discount=s.curve,
       vol_surface=s.vol_surface,
   )

   print(price_payoff(cap, market_state))

Compute a Par Swap Rate
-----------------------

.. code-block:: python

   from datetime import date
   from trellis import MarketState, SwapSpec, par_swap_rate

   spec = SwapSpec(
       notional=10_000_000,
       fixed_rate=0.045,
       start_date=date(2024, 11, 15),
       end_date=date(2029, 11, 15),
   )

   market_state = MarketState(
       as_of=s.settlement,
       settlement=s.settlement,
       discount=s.curve,
   )

   print(par_swap_rate(spec, market_state))

Current Limitations
-------------------

Some public analytics are intentionally documented with caveats:

- tenor bumps are currently exact-tenor only, so key-rate style analyses
  on sparse curves need care
- vega analytics assume a flat volatility surface

Related Reading
---------------

- :doc:`../user_guide/market_data`
- :doc:`../user_guide/conventions`
- :doc:`../api/curves`
