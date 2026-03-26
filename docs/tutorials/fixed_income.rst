Fixed-Income Workflow
=====================

This tutorial covers the deterministic, offline path for bonds, books,
and batch pricing. It is the best place to start when you want fully
reproducible examples with no API keys or live data.

Start from the Sample Session
-----------------------------

.. code-block:: python

   import trellis

   s = trellis.quickstart()
   bond = trellis.sample_bond_10y()
   result = s.price(bond)

   print(result.clean_price)
   print(result.greeks["dv01"])

Work with a Book
----------------

.. code-block:: python

   import trellis

   s = trellis.quickstart()
   book = trellis.sample_book()
   result = s.price(book)

   print(result.total_mv)
   print(result.book_dv01)

Run Scenario Analysis
---------------------

.. code-block:: python

   s_up = s.with_curve_shift(+100)
   shocked = s_up.price(bond)
   print(shocked.clean_price)

Batch Pricing with Pipeline
---------------------------

.. code-block:: python

   import trellis

   results = (
       trellis.Pipeline()
       .instruments(trellis.sample_book())
       .market_data(curve=trellis.sample_curve())
       .compute(["price", "dv01", "duration"])
       .scenarios([
           {"name": "base", "shift_bps": 0},
           {"name": "up100", "shift_bps": 100},
       ])
       .run()
   )

   print(results["base"].total_mv)
   print(results["up100"].book_dv01)

Why This Path Matters
---------------------

This package-level workflow gives you:

- deterministic mock market data
- explicit control over session state
- stable entry points for scripts and notebooks
- a clean fallback when you do not want agent behavior

Related Reading
---------------

- :doc:`../user_guide/session`
- :doc:`../user_guide/pricing`
- :doc:`../api/engine`
