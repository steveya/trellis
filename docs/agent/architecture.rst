Agent Architecture
==================

Trellis uses a multi-agent pipeline where each agent has a distinct role.
The LLM is **never in the pricing hot path** — it only participates in
code generation and review. All pricing is deterministic.

Pipeline Overview
-----------------

.. code-block:: text

   User: "Price a callable bond"
     │
     ├─ Term Sheet Parser ──→ TermSheet (structured data)
     │     (LLM call #1: extract fields from English)
     │
     ├─ Quant Agent ──→ PricingPlan (method + data requirements)
     │     (static rules or LLM call #2: select method)
     │
     ├─ Data Check ──→ error if market data missing
     │     (deterministic: check MarketState fields)
     │
     ├─ Planner ──→ BuildPlan (spec schema + module path)
     │     (static specs for 10 instruments, LLM fallback)
     │
     ├─ Builder Agent ──→ Python module with evaluate()
     │     (LLM call #3: generate code from skeleton + cookbook)
     │
     ├─ Critic Agent ──→ list of (concern, test_case) pairs
     │     (LLM call #4: adversarial code review)
     │
     ├─ Arbiter ──→ pass / fail
     │     (deterministic: run invariants + critic tests)
     │
     └─ Price ──→ result
           (deterministic: price_payoff)

Agent Roles
-----------

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Agent
     - Role
     - Input
     - Output
   * - **Term Sheet Parser**
     - Extract structured data from English
     - Natural language description
     - ``TermSheet`` (type + parameters)
   * - **Quant Agent**
     - Select pricing method
     - Instrument description
     - ``PricingPlan`` (method, modules, data needs)
   * - **Builder Agent**
     - Write ``evaluate()`` code
     - Skeleton + cookbook + references
     - Python module
   * - **Critic Agent**
     - Find errors in generated code
     - Generated source code
     - ``(concern, test_code)`` pairs
   * - **Arbiter**
     - Run tests deterministically
     - Invariants + critic tests
     - Pass/fail with failure messages

Design Principles
-----------------

1. **LLM never in hot path**: ``agent=False`` by default everywhere.
   Deterministic pricing is the foundation; the agent is opt-in.

2. **Market data vs computational methods**: ``MarketState`` holds observed data.
   Methods (trees, MC, PDE) are library code the agent imports.

3. **Deterministic interface, LLM implementation**: the spec dataclass and
   class skeleton are generated deterministically. The LLM only fills in
   ``evaluate()``.

4. **Adversarial validation**: the critic sees code only (not the builder's
   reasoning). It outputs executable assertions, not opinions.

5. **Retry on failure**: if invariants or critic tests fail, the builder
   retries with the failure feedback (up to 3 attempts).

Return Type Convention
----------------------

``evaluate()`` returns one of two types:

- **Cashflows**: undiscounted dated cashflows. ``price_payoff()`` discounts each one.
  Used for analytical/cashflow-based pricing (bonds, caps, swaps).

- **PresentValue**: already-discounted PV. ``price_payoff()`` returns it directly.
  Used for tree/MC/PDE methods that handle their own discounting.

.. code-block:: python

   # Analytical pattern
   return Cashflows([(date1, amount1), (date2, amount2)])

   # Tree/MC/PDE pattern
   price = backward_induction(tree, payoff_fn, r)
   return PresentValue(price)

Validation Levels
-----------------

.. list-table::
   :header-rows: 1

   * - Level
     - Checks
     - When to use
   * - ``"fast"``
     - Syntax only
     - Development / iteration
   * - ``"standard"``
     - Syntax + invariants + critic
     - Default for production
   * - ``"thorough"``
     - Standard + cross-model
     - High-value instruments
