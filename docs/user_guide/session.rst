Session & Pipeline
==================

Session
-------

A ``Session`` is an immutable market snapshot for pricing. It holds a yield curve,
optional vol surface, credit curve, and other market data.

.. code-block:: python

   from trellis import Session, YieldCurve, FlatVol
   from datetime import date

   s = Session(
       curve=YieldCurve.flat(0.05),
       settlement=date(2024, 11, 15),
       vol_surface=FlatVol(0.20),
   )

**Auto-resolution**: if no curve is provided, the session fetches from a data provider:

.. code-block:: python

   s = Session(data_source="mock")  # uses built-in mock data

Scenario Analysis
~~~~~~~~~~~~~~~~~

Sessions are immutable — scenario methods return new sessions:

.. code-block:: python

   s_up = s.with_curve_shift(+100)       # +100bp parallel shift
   s_bumped = s.with_tenor_bumps({10.0: +50})  # +50bp at 10Y
   s_off_grid = s.with_tenor_bumps({7.0: +25})  # inserts a local 7Y shock node
   s_new = s.with_curve(other_curve)     # replace curve entirely
   s_vol = s.with_vol_surface(FlatVol(0.30))

``with_tenor_bumps(...)`` now routes through the shared curve-shock substrate.
Exact tenor requests still move the matching knot directly, while off-grid
tenors insert a bumped node into the shocked curve so repricing follows the
same interpolation logic as the base curve. The resulting curve keeps optional
``curve_shock_surface`` and ``curve_shock_warnings`` metadata for callers that
need to inspect sparse-support or endpoint-extension warnings.

Pipeline
--------

For batch pricing with scenarios:

.. code-block:: python

   from trellis import Pipeline

   results = (
       Pipeline()
       .instruments(book)
       .market_data(curve=curve)
       .scenarios([
           {"name": "base", "shift_bps": 0},
           {"name": "up100", "shift_bps": 100},
       ])
       .output_csv("output/{scenario}.csv")
       .run()
   )

Named rate scenario packs are also available for desk-style curve stresses:

.. code-block:: python

   results = (
       Pipeline()
       .instruments(book)
       .market_data(curve=curve)
       .scenarios([
           {
               "scenario_pack": "twist",
               "bucket_tenors": (2.0, 5.0, 10.0, 30.0),
               "amplitude_bps": 25.0,
           }
       ])
       .run()
   )

The current built-in packs are ``"twist"`` (steepener + flattener) and
``"butterfly"`` (belly-up + belly-down). Each pack expands into named
``tenor_bumps`` scenarios on the shared bucket-shock substrate, so the same
assumptions flow through both pipeline and one-trade analytics surfaces.

Book & BookResult
-----------------

A ``Book`` holds a collection of instruments with notionals:

.. code-block:: python

   from trellis import Book, Bond
   book = Book(
       {"10Y": bond_10y, "5Y": bond_5y},
       notionals={"10Y": 25_000_000, "5Y": 10_000_000},
   )
   result = s.price(book)
   print(f"Total MV: {result.total_mv:,.0f}")
   print(f"Book DV01: {result.book_dv01:,.0f}")
