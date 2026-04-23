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
   If the payoff has a resolved-input pricing kernel, expose it as ``*_raw``
   or ``evaluate_raw(...)`` and keep ``evaluate()`` as the public trace-safe
   adapter boundary. Generated payoff skeletons should annotate the public
   return as ``PricingValue`` and preserve the final present-value scalar
   through the adapter.
3. Return the present-value scalar directly. Legacy ``Cashflows`` and
   ``PresentValue`` wrappers remain for backward compatibility only.
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

Prefer Stable Schedule And Event Objects
----------------------------------------

If a route depends on accrual periods, event windows, or payment timing, do not
reconstruct those contracts from raw ``list[date]`` schedule output inside the
payoff implementation. Use the periodized schedule substrate instead.

The stable pattern is:

1. build an :class:`EventSchedule <trellis.core.types.EventSchedule>` with
   ``build_period_schedule(...)``
2. consume explicit :class:`SchedulePeriod <trellis.core.types.SchedulePeriod>`
   objects inside ``evaluate()``
3. keep product-specific logic focused on payoff math rather than date
   reconstruction

This matters most for credit, insurance-linked, callable, and other event-style
products where routes otherwise drift on:

- period boundaries
- end-date inclusion
- payment timing
- day-count handling
- model time origin

The DSL and agent surfaces should map to these stable objects, not to raw
``generate_schedule(...)`` plus hand-built ``prev_date`` logic.

When a route family has enough repeated structure, go one step further and
expose a checked-in helper surface in ``trellis/models/``. For example,
single-name CDS routes should prefer shared schedule and leg-pricing helpers
over open-coded premium/protection loops inside generated adapters. The same
pattern now applies to:

- vanilla equity tree routes via ``trellis.models.equity_option_tree``
- vanilla European equity PDE routes via ``trellis.models.equity_option_pde``
- rate-style swaption analytics via ``trellis.models.rate_style_swaption``
- zero-coupon bond option tree routes via ``trellis.models.zcb_option_tree``
- Jamshidian analytical ZCB option routes via ``trellis.models.zcb_option``

For schedule-dependent swaptions, split the stable surfaces by solver role:

- Bermudan tree routes should delegate to
  ``trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree(...)``
- analytical comparison lanes should delegate to
  ``trellis.models.rate_style_swaption.price_bermudan_swaption_black76_lower_bound(...)``

That analytical helper is intentionally a lower-bound surface. It prices the
European swaption exercisable only on the final Bermudan date. It is not a
closed-form Bermudan solver and should not be replaced with an inline loop that
sums multiple exercise-date European values.

The same rule now applies to schedule-dependent lattice exercise. Prefer the
checked-in control helpers in ``trellis.models.trees.control`` over hand-coded
``exercise_type`` / ``exercise_steps`` / ``exercise_fn`` combinations:

1. build an exercise timeline
2. convert it with ``lattice_steps_from_timeline(...)``
3. resolve the product semantics with ``resolve_lattice_exercise_policy(...)``
4. pass the resulting ``exercise_policy`` into
   ``lattice_backward_induction(...)``

That keeps issuer-call, holder-put, and Bermudan semantics stable across
products and agent-generated routes.

Promote Autograd With Purpose
-----------------------------

Autograd should be promoted when it replaces a real cost center:

- repeated bump/reprice loops for Greeks
- calibration Jacobians or optimizer gradients
- sensitivity extraction on closed-form kernels

Keep Numba and other compiled kernels as forward engines. Use a separate
differentiable path only when the gradient meaningfully improves accuracy,
stability, or runtime. The current differentiable pricing notes live in
:doc:`differentiable_pricing`.

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

Semantic concepts are tracked separately from product wrappers in
``trellis.agent.semantic_concepts``. That registry is the source of truth for
concept versioning, compatibility wrappers, extension policy, and the
distinction between product contracts, supporting atoms, and market-input
concepts. If a request matches a thin wrapper name such as ``basket_option``
or ``swaption``, the agent should still resolve it to the canonical semantic
concept before deciding whether the change is a new attribute, a wrapper, or a
genuinely new concept. Supporting atoms are only surfaced when the request is
clearly about that semantic layer instead of a broader product change that just
mentions schedule or payoff details.

The semantic-extension loop also records an explicit role matrix in request
metadata and traces. Quant owns bounded route and primitive assembly proposals,
model_validator owns payoff/model validation only, and the knowledge agent
owns the trace handoff. None of those roles should invent new semantic grammar;
if no safe owner exists, the extension should fail closed instead of being
forced through a guess.

Extension Strategy
------------------

Prefer extending the deterministic library first, then teaching the agent about it.

- If a construct is useful without the agent, implement and test it in the library.
- If the agent should route or build against it, add or update the matching knowledge assets.
- If a change affects operational behavior, traces, or issue sync, continue in :doc:`../developer/index`.
- If a numerical family has a natural raw kernel, prefer that kernel for AD and
  keep the user-facing adapter trace-safe, with float conversion only at
  explicit reporting or solver boundaries.

Related Reading
---------------

- :doc:`knowledge_maintenance`
- :doc:`../agent/architecture`
- :doc:`../developer/task_and_eval_loops`
