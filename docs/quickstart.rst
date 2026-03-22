Quickstart
==========

Installation
------------

.. code-block:: bash

   pip install trellis

   # With optional dependencies:
   pip install trellis[agent]      # LLM agent (OpenAI/Anthropic)
   pip install trellis[data]       # Live market data (FRED, Treasury.gov)
   pip install trellis[crossval]   # Cross-validation (QuantLib, FinancePy)

Your First Price
----------------

.. code-block:: python

   import trellis

   # Quickstart session with mock market data (no API keys needed)
   s = trellis.quickstart()

   # Price a bond
   bond = trellis.sample_bond_10y()
   result = s.price(bond)
   print(f"Clean price: {result.clean_price:.4f}")
   print(f"DV01: {result.greeks['dv01']:.6f}")

Natural Language Pricing
------------------------

.. code-block:: python

   # Requires OPENAI_API_KEY or ANTHROPIC_API_KEY in .env
   result = trellis.ask("Price a 5-year interest rate cap at 4% on $10M")
   print(f"Price: ${result.price:,.2f}")
   print(f"Instrument: {result.payoff_class}")

Working with Sessions
---------------------

.. code-block:: python

   from trellis import Session, YieldCurve, FlatVol
   from datetime import date

   # Create a session with your own market data
   s = Session(
       curve=YieldCurve.flat(0.045),
       settlement=date(2024, 11, 15),
       vol_surface=FlatVol(0.20),
   )

   # Price a bond
   from trellis import Bond
   bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
               maturity=10, frequency=2)
   result = s.price(bond)

   # Scenario analysis
   s_up = s.with_curve_shift(+100)  # parallel shift +100bp
   print(f"Base: {result.clean_price:.4f}")
   print(f"+100bp: {s_up.price(bond).clean_price:.4f}")

Using the Payoff Framework
--------------------------

.. code-block:: python

   from trellis import (
       Session, YieldCurve, FlatVol, MarketState,
       CapPayoff, CapFloorSpec, price_payoff, Cashflows, PresentValue,
   )
   from trellis.core.types import Frequency
   from datetime import date

   # Cap pricing via the Payoff protocol
   spec = CapFloorSpec(
       notional=1_000_000, strike=0.04,
       start_date=date(2025, 2, 15), end_date=date(2030, 2, 15),
       frequency=Frequency.QUARTERLY,
   )
   cap = CapPayoff(spec)
   print(f"Requirements: {cap.requirements}")
   # {'discount', 'forward_rate', 'black_vol'}

   ms = MarketState(
       as_of=date(2024, 11, 15), settlement=date(2024, 11, 15),
       discount=YieldCurve.flat(0.05), vol_surface=FlatVol(0.20),
   )
   pv = price_payoff(cap, ms)
   print(f"Cap PV: ${pv:,.2f}")
