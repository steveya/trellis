Quant Agent
===========

The quant agent makes the **financial** decision: given an instrument,
which computational method is appropriate and what market data does it need?
It does NOT write code.

Method Selection
----------------

Static rules for 13 instrument types:

.. list-table::
   :header-rows: 1
   :widths: 25 20 30

   * - Instrument
     - Method
     - Data Requirements
   * - Bond, Swap
     - ``analytical``
     - discount, forward_rate
   * - Cap, Floor, Swaption
     - ``analytical``
     - discount, forward_rate, black_vol
   * - Callable Bond, Puttable Bond
     - ``rate_tree``
     - discount, black_vol
   * - Bermudan Swaption
     - ``rate_tree``
     - discount, forward_rate, black_vol
   * - Barrier Option, Asian Option
     - ``monte_carlo``
     - discount, black_vol
   * - CDO, Nth-to-Default
     - ``copula``
     - discount, credit
   * - MBS
     - ``monte_carlo``
     - discount

For unknown instruments, the quant agent falls back to an LLM call.

PricingPlan
-----------

.. autoclass:: trellis.agent.quant.PricingPlan
   :members:

Early Data Check
----------------

Before any code generation, ``check_data_availability()`` verifies the
required market data is in ``MarketState``. If missing, a helpful error
is raised:

.. code-block:: text

   Missing market data: 'black_vol' — Black (lognormal) implied volatility surface.
     How to provide: Session(vol_surface=FlatVol(0.20))

Implementation
--------------

.. autofunction:: trellis.agent.quant.select_pricing_method
.. autofunction:: trellis.agent.quant.check_data_availability
