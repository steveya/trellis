Quant Agent
===========

The runtime quant agent makes the financial decision: given typed product
semantics, which computational method is appropriate, what market data is
required, and which residual model risks must be handed to validation? It does
not write code, choose imports, run deterministic acceptance checks, or promote
cookbook entries.

Runtime Orientation
-------------------

The quant agent receives the versioned ``quant-runtime-navigation`` card at
its LLM boundary. The card is rendered from
``trellis/agent/knowledge/canonical/agent_orientations.yaml`` and points to the
smallest useful navigation surface in this order:

#. the runtime semantic contract, ``ProductIR``, and quant challenger packet
#. canonical product decompositions and admitted route families
#. the model grammar and method requirements
#. the cookbook catalog as read-only pattern evidence
#. the quant documentation index and model-grammar design reference

The card deliberately omits exact symbol and import lookup. Existing routing
context may still include the family-level API map, but exact symbol selection
remains the builder's responsibility through the API map and import registry.
This keeps method selection independent from whichever convenience helper
happens to exist.

Method Selection
----------------

Known products are decomposed through canonical typed knowledge, not a second
table in ``quant.py``. Product features, method candidates, required market
data, exercise style, state dependence, and model family become ``ProductIR``.
The quant layer ranks the admitted candidates and emits a ``PricingPlan`` plus
a ``QuantChallengerPacket``. Genuinely novel products may use bounded LLM
decomposition, and that prompt receives the same orientation card.
The selected method family is mapped to deterministic default construction
modules in runtime code; the LLM is not asked to emit module or import paths.

PricingPlan
-----------

.. autoclass:: trellis.agent.quant.PricingPlan
   :members:

Quant Challenger Packet
------------------------

The packet records the selected method, rejected alternatives, assumption
basis, market-data obligations, expected executable checks, requested
measures, residual-risk handoff, and quant orientation identity. Downstream
validation consumes this packet instead of reconstructing method-selection
reasoning from prose.

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

See Also
--------

- :doc:`../quant/index` for the quantitative documentation index
- :doc:`../developer/runtime_agent_orientation` for contract loading, role
  separation, and trace behavior
- :doc:`../developer/task_and_eval_loops` for task execution and bounded
  learning behavior
