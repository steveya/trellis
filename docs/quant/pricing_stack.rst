Pricing Stack
=============

This section is the map of the deterministic pricing stack that sits underneath
``trellis.ask(...)`` and the rest of the public API.

Layering
--------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Layer
     - Primary objects
     - Role
     - Main code paths
   * - Public workflow
     - ``trellis.ask``, ``Session``, ``Pipeline``
     - User-facing entry points for ask, direct pricing, and batch flows
     - ``trellis/__init__.py``, ``trellis/session.py``, ``trellis/pipeline.py``
   * - Market contract
     - ``MarketState``, ``MarketSnapshot``, capability checks
     - Immutable market inputs and data-availability checks
     - ``trellis/core/market_state.py``, ``trellis/data/schema.py``, ``trellis/core/capabilities.py``
   * - Payoff contract
     - ``Payoff``, ``Cashflows``, ``PresentValue``
     - Common evaluation interface for deterministic engines
     - ``trellis/core/payoff.py``, ``trellis/engine/payoff_pricer.py``
   * - Instrument library
     - Bonds, swaps, caps, callable bonds, FX, credit
     - Hand-written reference implementations and public product surfaces
     - ``trellis/instruments/``
   * - Numerical engines
     - Trees, MC, PDE, transforms, copulas, analytical methods
     - Computational machinery used by hand-written and agent-built payoffs
     - ``trellis/models/``
   * - Agent interface
     - Term sheet, quant, planner, builder, critic, validator
     - Compiles natural-language or build requests into deterministic execution
     - ``trellis/agent/``

Deterministic Flow
------------------

The stable pricing path is:

1. Resolve or construct market data into a ``Session`` or ``MarketState``.
2. Instantiate a hand-written instrument/payoff or let ``ask()`` parse a term sheet.
3. Check required capabilities against the market state.
4. Evaluate through deterministic pricing code.

The LLM is not in the pricing hot path. It participates only in parsing,
routing, code generation, review, and model validation around the deterministic
library.

Method Families
---------------

Trellis groups computational methods by directory and by the agent method labels
it uses during routing and code generation:

- ``analytical`` for closed-form or direct cashflow-style valuation
- ``rate_tree`` and related lattice routines for callable or early-exercise rate products
- ``monte_carlo`` for path-dependent or simulation-heavy products
- ``pde_solver`` for finite-difference grids and operator-based valuation
- ``fft_pricing`` and transform methods for characteristic-function pricing
- ``copula`` for portfolio credit and tranche/default aggregation
- calibration and cashflow-engine utilities for supporting model inputs and structured cashflows

The detailed formulas and implementation notes live in the mathematical reference
pages linked from :doc:`index`.

Agent Boundary
--------------

The agent system reuses this stack rather than replacing it.

- The quant agent selects a method and data requirements: :doc:`../agent/quant_agent`
- The builder agent writes an ``evaluate()`` body around deterministic interfaces: :doc:`../agent/builder_agent`
- The critic, arbiter, and model validator test the generated artifact before use: :doc:`../agent/critic_agent`

That design matters for quant work: if a method family or market-data contract is
wrong in the deterministic library, the agent will reproduce that mistake.

Related Reading
---------------

- :doc:`extending_trellis`
- :doc:`knowledge_maintenance`
- :doc:`../developer/overview`

