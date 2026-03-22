Builder Agent
=============

The builder agent generates Python code for new payoff classes. It uses a
two-step structured pipeline with deterministic interfaces.

Two-Step Pipeline
-----------------

**Step 1: Spec Design** (deterministic or LLM)

For known instruments (10 static specs), the spec dataclass is deterministic:

.. code-block:: python

   @dataclass(frozen=True)
   class SwaptionSpec:
       notional: float
       strike: float
       expiry_date: date
       swap_start: date
       swap_end: date
       swap_frequency: Frequency = Frequency.SEMI_ANNUAL
       day_count: DayCountConvention = DayCountConvention.ACT_360
       rate_index: str | None = None
       is_payer: bool = True

For unknown instruments, an LLM call designs the spec via structured JSON output.

**Step 2: Code Generation** (LLM)

The system generates a complete skeleton (imports, spec, class, requirements)
and asks the LLM to fill in ``evaluate()``. The prompt includes:

- The skeleton code
- The quant agent's method selection
- The relevant **cookbook** pattern
- Reference implementations for the chosen method

Cookbooks
---------

Five evaluate() body templates, one per method:

.. list-table::
   :header-rows: 1

   * - Method
     - Return Type
     - Pattern
   * - ``analytical``
     - ``Cashflows``
     - Forward rates → Black76 → dated payoffs
   * - ``rate_tree``
     - ``PresentValue``
     - Build tree → backward induction with exercise
   * - ``monte_carlo``
     - ``PresentValue``
     - Simulate paths → compute path-dependent payoff
   * - ``copula``
     - ``PresentValue``
     - Simulate correlated defaults → tranche loss
   * - ``waterfall``
     - ``Cashflows``
     - Amortize → prepay → run through tranches

Each cookbook marks ``>>> INSTRUMENT-SPECIFIC <<<`` where the builder fills in
instrument logic.

Static Specs
------------

10 instrument types with deterministic field names:

swaption, cap, floor, callable_bond, puttable_bond, barrier_option,
asian_option, cdo, nth_to_default, bermudan_swaption.

Caching
-------

Built payoffs persist in ``trellis/instruments/_agent/``. On subsequent calls,
``build_payoff(..., force_rebuild=False)`` reuses the existing module without
calling the LLM.

Implementation
--------------

.. autofunction:: trellis.agent.executor.build_payoff
