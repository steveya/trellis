Trellis: AI-Augmented Quantitative Pricing Library
===================================================

**Trellis** is a self-evolving quantitative pricing library that combines
classical financial mathematics with AI-powered code generation. Describe
an instrument in natural language, and Trellis either prices it from its
existing engine or builds a new pricer automatically.

.. code-block:: python

   import trellis

   # Natural language pricing
   result = trellis.ask("Price a 5Y cap at 4% on $10M")
   print(result.price)

   # Programmatic pricing
   s = trellis.quickstart()
   print(s.price(trellis.sample_bond_10y()).clean_price)

Key Features
------------

- **590+ tests** verified against QuantLib, FinancePy, and TF Quant Finance
- **Complete mathematical toolkit**: trees, Monte Carlo, PDE, FFT, copulas, LSM
- **Multi-agent pipeline**: quant agent → builder → critic → arbiter
- **35 rate indices**, 9 calendars, 9 day count conventions
- **Autograd-compatible**: Greeks via automatic differentiation

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
   :caption: Mathematical Specifications

   mathematical/bond_pricing
   mathematical/black76
   mathematical/swap_pricing
   mathematical/credit
   mathematical/fx
   mathematical/trees
   mathematical/monte_carlo
   mathematical/pde
   mathematical/transforms
   mathematical/processes
   mathematical/copulas
   mathematical/calibration
   mathematical/cashflow_engine

.. toctree::
   :maxdepth: 2
   :caption: Agent Architecture

   agent/architecture
   agent/quant_agent
   agent/builder_agent
   agent/critic_agent

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/core
   api/instruments
   api/curves
   api/models
   api/engine
   api/conventions

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
