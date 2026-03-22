Market Data
===========

Yield Curves
------------

.. code-block:: python

   from trellis import YieldCurve

   # Flat curve
   curve = YieldCurve.flat(0.045)

   # From Treasury yields (BEY → continuous)
   curve = YieldCurve.from_treasury_yields({
       0.25: 0.045, 1.0: 0.047, 2.0: 0.048,
       5.0: 0.045, 10.0: 0.044, 30.0: 0.046,
   })

   # From bootstrap
   from trellis import BootstrapInstrument, bootstrap_yield_curve
   instruments = [
       BootstrapInstrument(0.25, 0.04, "deposit"),
       BootstrapInstrument(2.0, 0.045, "swap"),
       BootstrapInstrument(5.0, 0.048, "swap"),
   ]
   curve = bootstrap_yield_curve(instruments)

Data Providers
--------------

.. code-block:: python

   # Mock (offline, no API key)
   from trellis.data.mock import MockDataProvider
   yields = MockDataProvider().fetch_yields()

   # FRED (requires FRED_API_KEY)
   from trellis.data.fred import FredDataProvider
   yields = FredDataProvider().fetch_yields()

   # Treasury.gov (no key needed, but requires internet)
   from trellis.data.treasury_gov import TreasuryGovDataProvider
   yields = TreasuryGovDataProvider().fetch_yields()

Mock Data
~~~~~~~~~

The mock provider ships 4 historical snapshots (no network needed):

- 2019-09-15: Pre-COVID normal curve (~1.6-2.1%)
- 2020-03-15: COVID crisis, near-zero front end
- 2023-10-15: Peak rates, inverted curve (~4.5-5.3%)
- 2024-11-15: Easing cycle (~4.2-4.6%)

Multi-Curve
-----------

Post-2008, use separate discount and forecast curves:

.. code-block:: python

   s = Session(
       curve=ois_curve,                          # OIS for discounting
       forecast_curves={"USD-SOFR-3M": sofr_curve},  # SOFR for forwards
   )

Credit Curves
-------------

.. code-block:: python

   from trellis import CreditCurve

   cc = CreditCurve.flat(0.02)  # 200bp hazard rate
   cc = CreditCurve.from_spreads({5.0: 0.012}, recovery=0.4)
