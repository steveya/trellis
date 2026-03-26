Quickstart
==========

Trellis supports two primary ways to get started:

- ask for a price in natural language
- work programmatically from a reproducible session

The examples below default to package-level APIs and mock data unless a
section is explicitly marked otherwise.

Installation
------------

.. code-block:: bash

   pip install trellis

   # Optional runtime dependencies are installed separately today:
   pip install openai              # or: pip install anthropic
   pip install requests fredapi

Natural Language First
----------------------

.. important::

   ``trellis.ask(...)`` requires an installed provider client and a
   configured provider API key such as ``OPENAI_API_KEY`` or
   ``ANTHROPIC_API_KEY``.
   Built payoffs for unsupported products are experimental and should be
   validated before production use.

.. code-block:: python

   import trellis

   result = trellis.ask("Price a 5-year SOFR cap at 4% on $10M")
   print(f"Price: ${result.price:,.2f}")
   print(f"Instrument: {result.payoff_class}")
   print(f"Matched existing payoff: {result.matched_existing}")

Offline First Price
-------------------

.. code-block:: python

   import trellis

   # Quickstart session with mock market data (no network, no API keys)
   s = trellis.quickstart()

   bond = trellis.sample_bond_10y()
   result = s.price(bond)
   print(f"Clean price: {result.clean_price:.4f}")
   print(f"DV01: {result.greeks['dv01']:.6f}")

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

Ask Against Your Own Session
----------------------------

.. code-block:: python

   from trellis import Session, YieldCurve, FlatVol
   from datetime import date

   s = Session(
       curve=YieldCurve.flat(0.045),
       settlement=date(2024, 11, 15),
       vol_surface=FlatVol(0.20),
   )

   result = s.ask(
       "Price a 5-year payer swap at 4.5% on $25M SOFR",
       measures=["price", "dv01"],
   )
   print(result.price)
   print(result.analytics.dv01)

Using the Payoff Framework
--------------------------

.. code-block:: python

   from trellis import (
       Session, YieldCurve, FlatVol, MarketState,
       CapPayoff, CapFloorSpec, price_payoff,
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

Next Steps
----------

- :doc:`tutorials/index` for end-to-end workflows
- :doc:`user_guide/session` for immutable sessions and scenario analysis
- :doc:`user_guide/pricing` for pricing modes and analytics
- :doc:`quant/index` for pricing constructs, extensions, and knowledge maintenance
- :doc:`developer/index` for the platform and agent runtime behind ``ask()``
