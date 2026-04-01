Pricing Stack
=============

This page records the deterministic pricing stack that now sits underneath
``trellis.ask(...)`` and the semantic DSL boundary.

Layering
--------

.. list-table::
   :header-rows: 1
   :widths: 20 28 24 28

   * - Layer
     - Primary objects
     - Role
     - Main code paths
   * - Public workflow
     - ``trellis.ask``, ``Session``, ``Pipeline``
     - User-facing entry points for ask, direct pricing, and batch flows
     - ``trellis/__init__.py``, ``trellis/session.py``, ``trellis/pipeline.py``
   * - Semantic contract
     - ``SemanticContract`` and typed product semantics
     - Contract meaning independent of solver or market snapshot
     - ``trellis/agent/semantic_contracts.py``
   * - Valuation context
     - ``ValuationContext``, ``RequiredDataSpec``, ``MarketBindingSpec``
     - Market binding, model/measure policy, reporting, and requested outputs
     - ``trellis/agent/valuation_context.py``, ``trellis/agent/market_binding.py``
   * - Product summary and routing
     - ``ProductIR``, ``PricingPlan``, ``RouteSpec``, ``BuildGateDecision``
     - Checked summary, method choice, typed admissibility, and gatekeeping
     - ``trellis/agent/semantic_contract_compiler.py``, ``trellis/agent/route_registry.py``, ``trellis/agent/build_gate.py``
   * - Family lowering
     - family-specific lowering IRs + DSL lowering
     - Narrow typed lowering onto checked-in helper-backed routes
     - ``trellis/agent/family_lowering_ir.py``, ``trellis/agent/dsl_lowering.py``
   * - Numerical engines
     - analytical, lattice, PDE, Monte Carlo, transforms, copulas
     - Deterministic pricing math used by hand-written and agent-built routes
     - ``trellis/models/``
   * - Market and payoff runtime
     - ``MarketState``, ``Payoff``, ``PresentValue``
     - Immutable market inputs and common execution interfaces
     - ``trellis/core/market_state.py``, ``trellis/core/payoff.py``

Deterministic Flow
------------------

The current semantic pricing path is:

1. Normalize a request into a ``SemanticContract``.
2. Validate typed semantics, including phase order, obligations, and controller semantics.
3. Build a ``ValuationContext`` and compile ``RequiredDataSpec`` plus ``MarketBindingSpec``.
4. Build ``ProductIR`` and select a pricing method and candidate route.
5. Apply typed route admissibility through ``BuildGateDecision``.
6. Lower onto a family-specific IR and then onto a checked helper or kernel.
7. Execute the existing deterministic numerical code.

The LLM is not in the pricing hot path. It participates in parsing, planning,
generation, review, and validation around the deterministic library.

Shipped Lowering Boundary
-------------------------

The current compiler boundary is:

.. code-block:: text

   SemanticContract
     + ValuationContext
     -> ProductIR
     -> family lowering IR
     -> helper-backed numerical route

The shipped family IRs are:

- ``AnalyticalBlack76IR``
- ``VanillaEquityPDEIR``
- ``ExerciseLatticeIR``
- ``CorrelatedBasketMonteCarloIR``

This is intentionally not a flat universal IR. The current stack uses
``ProductIR`` as the shared checked summary and then narrows into family IRs
for the proven route families.

Current Proven Families
-----------------------

The end-to-end typed boundary is currently proven for:

- ``analytical_black76`` on vanilla options
- ``vanilla_equity_theta_pde`` on vanilla options
- ``exercise_lattice`` on callable bonds and Bermudan swaptions
- ``correlated_basket_monte_carlo`` on ranked-observation baskets

These route IDs and helper-backed numerical kernels are preserved. The new work
changes validation, binding, admissibility, and lowering, not the pricing math.

Warning And Error Policy
------------------------

The current stack distinguishes three classes of outcomes:

- semantic validation errors
- route admissibility failures
- successful compilation with warnings

Warnings are used when legacy semantic mirrors are normalized or ignored for
migrated route families. Errors are used for invalid typed semantics, missing
required bindings, unsupported outputs, unsupported control styles, or
unsupported state tags.

Authority Rules
---------------

For the migrated families listed above:

- typed ``SemanticTimeline`` and ``ObligationSpec`` are authoritative for settlement semantics
- typed ``EventMachine`` is authoritative for automatic event semantics
- ``requested_outputs`` is the canonical output field
- ``requested_measures`` is a shim surface only

Legacy fields such as ``settlement_rule`` and ``event_transitions`` remain on
the semantic contract for non-migrated code paths, tracing, and compatibility,
but they are mirrors rather than the truth source for migrated routes.

Deferred Scope
--------------

The current stack does not yet include:

- a full desk-task DSL
- ordered sequential multi-controller protocols
- nonlinear funding or XVA semantics inside ``ValuationContext``
- a universal IR covering every solver family

Those remain future extensions after the typed semantic boundary is stable
across more route families.
