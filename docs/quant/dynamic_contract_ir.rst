Dynamic Contract IR
===================

``DynamicContractIR`` is the bounded event/state/control sibling semantic
surface for products whose contractual meaning depends on ordered events,
running state, or explicit holder/issuer decisions.

Why It Exists
-------------

Some products are not faithfully described by either:

- one payoff-expression tree, or
- one static set of coupon/cashflow legs

Examples include callable coupon structures, range accruals with interruption,
target-redemption products, swing-style contracts, and insurance-style control
overlays. Those products need a semantic wrapper that is explicit about:

- event ordering
- state variables and update rules
- decision dates
- controller role
- admissible action sets

Current Surface
---------------

The shipped first slice lives in ``trellis.agent.dynamic_contract_ir`` and
provides the bounded semantic foundation:

- ``DynamicContractIR(base_contract, state_schema, event_program, control_program)``
- ``StateSchema`` / ``StateFieldSpec`` / ``StateUpdateSpec``
- ``EventProgram`` / ``EventTimeBucket`` / ``TerminationRule``
- event nodes:

  - ``ObservationEvent``
  - ``CouponEvent``
  - ``PaymentEvent``
  - ``DecisionEvent``
  - ``AutomaticTerminationEvent``
  - ``StateResetEvent``

- ``ControlProgram`` / ``ActionSpec``

The key design rule is compositional:

- ``base_contract`` is a static semantic contract, currently either
  ``ContractIR`` or ``StaticLegContractIR``
- dynamic semantics live in the wrapper, not in product-name leaves

Validation Contract
-------------------

The constructors enforce the local dynamic invariants for the first slice:

- event buckets are ordered by event date
- bucket phases must be drawn from the declared event ordering
- event labels must be unique
- state updates must reference declared state fields
- decision events require a ``ControlProgram``
- control-program decision labels must point at decision events
- decision-event controller roles must agree with the control program
- decision-event actions must be declared in the control program

Bounded Decomposition
---------------------

``trellis.agent.knowledge.decompose.decompose_to_dynamic_contract_ir(...)``
now lands one bounded decomposition example for the foundation slice:

- issuer-callable fixed coupon bonds with explicit call dates

Representative description:

.. code-block:: text

   Issuer callable fixed coupon bond USD face 1000000 coupon 5% issue 2025-01-15
   maturity 2030-01-15 semiannual day count ACT/ACT
   call dates 2027-01-15, 2028-01-15, 2029-01-15

That decomposition emits:

- a ``StaticLegContractIR`` base coupon-bond substrate
- one ``DecisionEvent`` per explicit call date
- an issuer ``ControlProgram`` with ``redeem`` and ``continue`` actions
- a termination rule keyed to the redeem action

This slice is intentionally representational. It proves that Trellis can emit a
dynamic semantic authority packet without product-name nodes, but it does not
yet claim a checked numerical lowering lane for dynamic products.

Classifier Boundary
-------------------

``trellis.agent.semantic_track_classifier.classify_semantic_track(...)`` is the
current deterministic boundary helper between the sibling tracks:

- ``payoff_expression``
- ``quoted_observable``
- ``static_leg``
- ``dynamic_wrapper``

For dynamic requests the classifier also emits a ``base_track`` hint so later
lanes know whether the dynamic wrapper sits over payoff-expression, quoted, or
static-leg semantics.

Current Non-Goals
-----------------

This first dynamic slice does not yet provide:

- automatic event/state lowerings for autocallables or TARN/TARF
- checked discrete-control lowerings for callable exotics
- continuous or singular control lowerings for GMWB / GMxB

Those are the next lanes after the semantic foundation. The current support
contract is: dynamic semantics are now representable and classifiable, but not
yet executable through an admitted pricing compiler.
