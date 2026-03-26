Trellis: AI-Augmented Pricing Platform
======================================

**Trellis** is an AI-augmented pricing platform for quantitative finance.
Start by asking a pricing question in natural language, move into Python
when you need control, and drop down into sessions, payoffs, and
numerical methods when you need full transparency.

.. code-block:: python

   import trellis

   result = trellis.ask("Price a 5Y SOFR cap at 4% on $10M")
   print(result.price, result.payoff_class, result.matched_existing)

   s = trellis.quickstart()
   bond = trellis.sample_bond_10y()
   print(s.price(bond).clean_price)

.. note::

   The public documentation defaults to reproducible mock data and the
   package-level API. Live market data, agent-built payoffs, and
   ``trellis-ui`` are documented, but explicitly marked when they remain
   experimental.

Start Here
----------

- :doc:`quickstart` for the shortest path from install to first result
- :doc:`tutorials/index` for end-to-end workflows
- :doc:`user_guide/pricing` for the package-level programming model
- :doc:`quant/index` for mathematical constructs, extension patterns, and knowledge maintenance
- :doc:`developer/index` for hosting, agents, audit systems, and runtime operations

Documentation Layers
--------------------

Trellis maintains three public documentation layers:

- **User guide** for high-level, package-level workflows such as ``trellis.ask(...)``,
  ``Session``, ``Pipeline``, and reproducible notebook usage
- **Quant documentation** for existing mathematical and computational constructs,
  extending pricing logic, linking deterministic code to the agent path, and
  maintaining lessons, memory, and cookbooks
- **Developer documentation** for the full platform surface, including request
  compilation, hosting/configuration, audit traces, issue sync, validation, and
  task/eval loops

Key Features
------------

- **Agent-first workflow**: describe an instrument and price it through
  ``trellis.ask(...)`` or ``Session.ask(...)``
- **Package-level library APIs** for sessions, books, curves, vols,
  payoffs, analytics, and scenario analysis
- **Multiple numerical families**: trees, Monte Carlo, PDE, transforms,
  copulas, analytical methods, and calibration helpers
- **Knowledge-backed build pipeline** for unsupported payoffs and task-driven
  extension
- **Cross-validation hooks** against QuantLib, FinancePy, and TF Quant Finance
- **Experimental companion UI** in ``trellis-ui``

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/session
   user_guide/pricing
   user_guide/market_data
   user_guide/conventions

.. toctree::
   :maxdepth: 2
   :caption: Quant Documentation

   quant/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   developer/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/instruments
   api/curves
   api/models
   api/engine
   api/conventions

.. toctree::
   :maxdepth: 1
   :caption: Reference

   migration_notes
   validation/numba_cache

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
