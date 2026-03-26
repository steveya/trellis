Extending Trellis
=================

This guide is for quant developers adding new pricing constructs rather than
operating the platform itself.

Choose The Smallest Extension Point
-----------------------------------

Common extension targets map cleanly to the package structure:

- new payoff or product type: ``trellis/instruments/``
- new model, process, or numerical engine: ``trellis/models/``
- new market-data component or capability: ``trellis/core/`` and ``trellis/data/``
- new public convenience surface: ``trellis/__init__.py`` and package-level wrappers

For agent-generated artifacts, ``trellis/instruments/_agent/`` is a cache of
built modules, not the canonical source tree.

Adding A New Payoff
-------------------

The core checklist is:

1. Define the frozen spec/value object that describes the product.
2. Implement a payoff class with explicit ``requirements`` and ``evaluate()`` semantics.
3. Return either ``Cashflows`` or ``PresentValue`` so ``price_payoff()`` can handle it consistently.
4. Add targeted pricing and capability tests.
5. Export the surface from ``trellis.__init__`` only if it is intended to be public.

If the payoff is hand-written but should also be discoverable through ``ask()``,
you also need to update the agent-facing knowledge described below.

Adding A New Method Family
--------------------------

When a new numerical family becomes part of Trellis, keep the deterministic and
agent layers aligned:

- implement the engine in ``trellis/models/``
- define the market-data requirements and capability expectations
- add tests for basic pricing behavior, numerical sanity, and any known edge cases
- document the mathematical assumptions and numerical limitations
- add cookbook and method-requirement knowledge if the builder should be able to use it

Linking Deterministic Code To Agents
------------------------------------

The agent build path depends on canonical knowledge files in
``trellis/agent/knowledge/``. For new quant features, the usual touch points are:

- ``canonical/features.yaml`` for reusable feature atoms and implication chains
- ``canonical/decompositions.yaml`` for product-to-feature and product-to-method mappings
- ``canonical/cookbooks.yaml`` for method-family code templates
- ``canonical/data_contracts.yaml`` for market-data conventions and conversions
- ``canonical/method_requirements.yaml`` for non-optional modeling constraints

The import registry in ``trellis.agent.knowledge.import_registry`` is the
authoritative source for prompt-safe imports. It is built from introspection with
a static fallback, so changes to public module surfaces should be verified there
as part of agent-facing work.

Extension Strategy
------------------

Prefer extending the deterministic library first, then teaching the agent about it.

- If a construct is useful without the agent, implement and test it in the library.
- If the agent should route or build against it, add or update the matching knowledge assets.
- If a change affects operational behavior, traces, or issue sync, continue in :doc:`../developer/index`.

Related Reading
---------------

- :doc:`knowledge_maintenance`
- :doc:`../agent/architecture`
- :doc:`../developer/task_and_eval_loops`

