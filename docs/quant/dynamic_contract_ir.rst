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

- ``DynamicContractIR(base_contract, semantic_family, base_track, state_schema, event_program, control_program)``
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

The control layer is now explicit about more than finite action labels:

- ``ActionSpec`` can carry ``action_domain`` together with an optional
  ``quantity_source`` and ``bounds_expression`` for continuous or singular
  control
- action-level ``state_updates`` record inventory or account-state effects
- ``ControlProgram.inventory_fields`` names the state variables that govern
  discrete inventory-style feasibility

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
- action-level state updates must reference declared state fields
- decision events require a ``ControlProgram``
- control-program decision labels must point at decision events
- decision-event controller roles must agree with the control program
- decision-event actions must be declared in the control program
- control-program inventory fields must point at declared state fields
- continuous or singular actions must declare quantity semantics

Bounded Decomposition
---------------------

``trellis.agent.knowledge.decompose.decompose_to_dynamic_contract_ir(...)``
now lands one bounded proving cohort for each admitted dynamic lane:

- automatic event/state: autocallable or phoenix-style notes and TARN/TARF
  structures
- discrete control: issuer-callable fixed coupon bonds and bounded
  swing-style inventory contracts
- continuous control: a financial-control-only GMWB fixture

Representative callable-bond description:

.. code-block:: text

   Issuer callable fixed coupon bond USD face 1000000 coupon 5% issue 2025-01-15
   maturity 2030-01-15 semiannual day count ACT/ACT
   call dates 2027-01-15, 2028-01-15, 2029-01-15

That decomposition emits:

- a ``StaticLegContractIR`` base coupon-bond substrate
- one ``DecisionEvent`` per explicit call date
- an issuer ``ControlProgram`` with ``redeem`` and ``continue`` actions
- a termination rule keyed to the redeem action

The automatic and financial-control fixtures stay intentionally bounded:

- the autocallable cohort carries automatic termination and coupon-memory state
- the TARN/TARF cohort carries running accrued state and target-triggered
  stopping
- the GMWB cohort carries explicit account-value and guarantee-base state plus
  a continuous withdrawal action

These decompositions are still fixture-grade and deterministic. They exist to
prove that Trellis can emit one dynamic semantic authority packet per lane
without route-local product authority, not to claim broad free-form product
coverage.

Lane Admission
--------------

``trellis.agent.dynamic_lane_admission.compile_dynamic_lane_admission(...)``
is the current bounded lowering-admission companion for the dynamic surface.
It emits three lane-specific typed contracts:

- ``AutomaticEventStateLaneAdmission``
- ``DiscreteControlLaneAdmission``
- ``ContinuousControlLaneAdmission``

Each admission object preserves the semantics that later numerical work must
not erase:

- automatic lane: event ordering, state-update fields, and stopping-rule labels
- discrete-control lane: controller role, decision timing, action set, and
  inventory fields
- continuous-control lane: controlled state, action domain, and control-magnitude
  semantics

The admission compiler is deliberately fail-closed. In particular,
quoted-observable hybrids such as callable CMS-spread range-accrual notes still
raise an explicit admission error rather than being silently widened into the
current bounded cohorts.

Current Support Contract
------------------------

The dynamic track is no longer merely representational: Trellis now has
admitted lane contracts and benchmark-plan packets for one proving cohort in
each of the automatic, discrete, and continuous control classes.

It is still not an executable fresh-build pricing path. The current support
contract is:

- dynamic semantics are representable and classifiable
- one bounded proving cohort per lane can be admitted structurally
- benchmark or parity plans are attached at admission time
- route-free executable pricing for those lanes remains future work

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

This bounded dynamic slice still does not provide:

- authoritative fresh-build pricing compiler integration for the admitted lanes
- quoted-observable dynamic hybrids such as callable CMS-spread range-accrual
  structures
- insurance overlays such as mortality, lapse, or fee behavior on top of the
  financial-control core

The current financial-control slice is intentionally overlay-free. Bounded
``gmwb`` decomposition now fails closed when the request introduces mortality,
lapse, or fee-bearing insurance terms, and the dynamic-lane admission surface
rejects policy-state-tagged fixtures with an explicit deferred-overlay blocker
instead of silently reclassifying them as ordinary financial-control contracts.
