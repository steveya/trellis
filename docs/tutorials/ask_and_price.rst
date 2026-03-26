Ask and Price
=============

This is the highest-level Trellis workflow: describe the instrument, let
the agent parse and route it, and inspect the result before deciding
whether to stay at the agent layer or drop into Python objects.

.. important::

   This tutorial requires an installed provider client (``openai`` or
   ``anthropic``) plus a configured provider API key. Requests that
   require Trellis to build a new payoff are experimental and should be
   validated before production use.

Ask for a Price
---------------

.. code-block:: python

   import trellis

   result = trellis.ask("Price a 5-year SOFR cap at 4% on $10M")
   print(result.price)
   print(result.payoff_class)
   print(result.matched_existing)

What You Get Back
-----------------

``ask()`` returns an ``AskResult`` with:

- ``price``: the computed price
- ``term_sheet``: the parsed term-sheet representation
- ``payoff_class``: the payoff implementation actually used
- ``matched_existing``: whether Trellis used an existing payoff or built one
- ``analytics``: optional analytics if you requested measures

Ask Against Your Own Market Snapshot
------------------------------------

.. code-block:: python

   from datetime import date
   import trellis
   from trellis import FlatVol, Session, YieldCurve

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

When Trellis Builds a New Payoff
--------------------------------

If a request does not match an existing payoff, Trellis can route into the
agent build path and return an ``AskResult`` with ``matched_existing=False``.

Treat that path as experimental:

- inspect the parsed term sheet
- confirm the payoff class and analytics make sense
- validate against known benchmarks before relying on the result
- prefer package-level workflows for production-critical paths

Related Reading
---------------

- :doc:`../quickstart`
- :doc:`../agent/architecture`
- :doc:`../user_guide/pricing`
