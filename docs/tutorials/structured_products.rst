Structured and Callable Products
================================

Trellis can route unsupported or advanced requests through its agent build
path, which makes this the natural entry point for callable, exotic, or
otherwise less-standard products.

.. important::

   This workflow is experimental. Use it for exploration, prototyping, and
   validation-backed research rather than as an unquestioned production path.

Start with a Natural-Language Request
-------------------------------------

.. code-block:: python

   import trellis

   result = trellis.ask(
       "Price a callable bond with a 5% coupon, 10-year maturity, "
       "and annual call dates after year 3"
   )

   print(result.price)
   print(result.payoff_class)
   print(result.matched_existing)

How to Read the Result
----------------------

``matched_existing=True`` means Trellis found an existing payoff and priced
it directly. ``matched_existing=False`` means Trellis went through the build
path to create a payoff class for the request.

Validation Checklist
--------------------

For structured-product workflows, validate more aggressively than you would
for plain bonds:

- inspect the parsed term sheet
- inspect the payoff class returned by the build
- compare against a simpler bound or benchmark when available
- confirm the market inputs in the session are appropriate
- record whether the output is exploratory or validated

Blocked Requests
----------------

Some requests can be blocked when foundational machinery is missing. In
that case, Trellis raises a runtime error with the blocker report instead
of pretending the product is supported.

Related Reading
---------------

- :doc:`../agent/architecture`
- :doc:`../mathematical/trees`
- :doc:`../mathematical/pde`
- :doc:`../mathematical/monte_carlo`
