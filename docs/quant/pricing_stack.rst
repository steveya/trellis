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

When multiple named curves are present, the compiled ``MarketState`` also
retains the selected curve names alongside the curve objects. That preserves
curve provenance for multi-curve routing, trace export, and replay/debugging
without changing the underlying pricing math.

The market-data resolver can also assemble named rate-curve sources from
bootstrap instrument sets before compilation. That keeps the source-selection
step explicit: a bootstrapped curve is still just a named curve in the
snapshot, and the runtime state records which curve name was chosen.

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
- ``calibration`` for implied-parameter solves such as cap/floor and swaption Black-vol fits
- calibration and cashflow-engine utilities for supporting model inputs and structured cashflows

The detailed formulas and implementation notes live in the mathematical reference
pages linked from :doc:`index`.

Analytical routes follow a resolver → support-helper → raw-kernel →
thin-adapter pattern. The reusable helper surface lives under
``trellis.models.analytical.support``, while the public analytical modules keep
the float-returning adapter boundary explicit. See
:doc:`analytical_route_cookbook` for the step-by-step pattern and shared helper
reference, and :doc:`basis_claim_patterns` for the extraction policy that
governs when route-local code is promoted to the support layer.

Agent Boundary
--------------

The agent system reuses this stack rather than replacing it.

- The quant agent selects a method and data requirements: :doc:`../agent/quant_agent`
- The builder agent writes an ``evaluate()`` body around deterministic interfaces: :doc:`../agent/builder_agent`
- The critic, arbiter, and model validator test the generated artifact before use: :doc:`../agent/critic_agent`

Analytical trace export
-----------------------

When the agent build loop assembles an analytical route, Trellis now persists both a machine-readable JSON trace and a Markdown rendering derived from the same trace object. The trace is emitted before cached or reused route short-circuits, so the task result and task-run records keep the trace paths even when the builder does not regenerate the route body. Downstream tools can inspect the build steps, validation outcomes, reuse decisions, and resolved curve provenance without re-running the build.

That design matters for quant work: if a method family or market-data contract is
wrong in the deterministic library, the agent will reproduce that mistake.

Related Reading
---------------

- :doc:`analytical_route_cookbook`
- :doc:`basis_claim_patterns`
- :doc:`differentiable_pricing`
- :doc:`extending_trellis`
- :doc:`knowledge_maintenance`
- :doc:`../developer/overview`
