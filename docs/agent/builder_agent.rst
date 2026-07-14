Builder Agent
=============

The builder agent generates Python code for new payoff classes. It uses a
two-step structured pipeline with deterministic interfaces.

Two-Step Pipeline
-----------------

**Step 1: Spec Design** (deterministic or LLM)

For known instruments with a registered static spec, the spec dataclass is
deterministic:

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

- the compiled semantic contract, method selection, and route obligations
- task-relevant API-map cards selected from typed product and route cues
- exact public symbols verified against the import registry
- bounded validated lessons and read-only cookbook evidence
- the skeleton code and deterministic acceptance requirements

Navigation And Authority
------------------------

The builder owns code and import selection. A no-query ``inspect_api_map``
call returns a bounded complete catalog; semantic fields or explicit family
names return the relevant full construction cards. The builder confirms exact
symbols with ``find_symbol`` or ``list_exports`` before reading modules
or writing code.

Cookbooks are validated method-pattern evidence, not product implementation
authority. Runtime tasks cannot write or promote cookbook entries, and a
product- or method-specific pricing helper is not a substitute for assembling
an admitted route from reusable market, process, payoff, control, numerical,
and validation primitives. The quant and model-validator role packets remain
code-free and do not receive the builder API-map surface.

Static Specs
------------

Registered static specs provide deterministic field names and defaults for
known instrument families. Product meaning still comes from the semantic
contract; a static spec cannot broaden exercise style, payoff family, model
family, or market binding.

Caching
-------

Built payoffs persist in ``trellis/instruments/_agent/``. On subsequent calls,
``build_payoff(..., force_rebuild=False)`` reuses the existing module without
calling the LLM.

Implementation
--------------

.. autofunction:: trellis.agent.executor.build_payoff

See Also
--------

- :doc:`../developer/runtime_agent_orientation` for the separation between
  worktree instructions, runtime role packets, and builder API navigation
- :doc:`../quant/knowledge_maintenance` for primitive-first knowledge
  maintenance and cookbook governance
