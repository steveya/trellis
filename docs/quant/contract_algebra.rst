Contract Algebra For General Derivative Pricing
=============================================

This note records the contract boundary Trellis now ships. The goal is not a
single universal solver. The goal is one semantic contract surface that can be
validated, bound to a valuation policy, and lowered onto stable helper-backed
numerical routes.

Shipped Boundary
----------------

Trellis now separates four concerns:

1. semantic contract meaning
2. valuation context and market binding
3. numerical lowering and route admissibility
4. requested desk outputs

The canonical semantic object is still ``SemanticContract`` in
``trellis.agent.semantic_contracts``. It now carries typed sub-objects rather
than relying only on flat semantic strings.

Semantic Contract
-----------------

The shipped semantic layer is centered on:

- ``ConventionEnv``
- ``SemanticTimeline``
- ``ObservableSpec``
- ``StateField``
- ``ObligationSpec``
- ``ControllerProtocol``
- ``EventMachine``

The semantic reading is:

.. math::

   \mathfrak{C} = (\Xi, \mathbb{T}, \Phi, \mathcal{O}, E, Y, U, G, \mathcal{A}, \Pi)

where:

- :math:`\Xi` is the convention environment
- :math:`\mathbb{T}` is the role-labelled timeline
- :math:`\Phi` is the same-day phase order
- :math:`\mathcal{O}` is the typed observable surface
- :math:`E` is event state
- :math:`Y` is contract memory
- :math:`U` is the state update logic
- :math:`G` is obligation emission
- :math:`\mathcal{A}` is the admissible action set
- :math:`\Pi` is the controller protocol

The tranche-1 default same-day phase order is:

- ``EVENT``
- ``OBSERVATION``
- ``DECISION``
- ``DETERMINATION``
- ``SETTLEMENT``
- ``STATE_UPDATE``

This is represented concretely by ``SemanticTimeline.phase_order`` and is
validated before lowering.

Meaning Rules
-------------

The semantic layer now follows these rules:

- automatic triggers stay in event/state machinery
- strategic rights stay in ``ControllerProtocol``
- contracts emit typed obligations, not discounted cashflows
- event state and contract memory are distinct ``StateField.kind`` values
- solver-facing state tags live on ``StateField.tags``
- schedule-bearing routes compile onto explicit timeline carriers such as
  ``ContractTimeline`` rather than raw comma-separated date strings

Legacy mirrors such as ``settlement_rule`` and ``event_transitions`` still
exist on ``SemanticProductSemantics``, but they are no longer the authority for
migrated route families.

Valuation Context
-----------------

Valuation policy is now separate from contract meaning through
``trellis.agent.valuation_context.ValuationContext``.

The shipped tranche-1 surface contains:

- normalized market source or snapshot handle
- compatibility ``model_spec`` string
- structured ``engine_model_spec``
  - ``model_family`` and ``model_name``
  - ``PotentialSpec`` and ``SourceSpec``
  - explicit rates ``discount_curve_role`` and ``forecast_curve_role`` when applicable
- ``measure_spec``
- ``discounting_policy``
- optional ``collateral_policy``
- ``reporting_policy``
- canonical ``requested_outputs``

The valuation reading is:

.. math::

   \mathfrak{V} = (M, Q, B, \Gamma, \rho, \mathcal{R})

where:

- :math:`M` is the model specification
- :math:`Q` is the measure specification
- :math:`B` is the numeraire or discounting policy
- :math:`\Gamma` is collateral, funding, and FX reporting policy
- :math:`\rho` is the reporting policy
- :math:`\mathcal{R}` is the requested output set

For migrated calibration workflows, ``engine_model_spec`` is the authoritative
model-binding surface and ``model_spec`` remains as a compatibility shim for
legacy callers.

The compiler also emits:

- ``RequiredDataSpec``
- ``MarketBindingSpec``
- ``LaneConstructionPlan``

These are compiled before route code generation. Raw contract hint dicts are no
longer the valuation truth.

``LaneConstructionPlan`` is the constructive bridge from contract algebra and
DSL lowering onto the computational lanes. It records:

- the lane family
- required timeline roles
- market-binding obligations
- state and control obligations
- lane construction steps
- exact checked backend targets when they already satisfy those obligations

After the lane plan is emitted, the runtime may also attach a
``RouteBindingAuthority`` packet. That packet is intentionally narrower than
the contract algebra itself. It does not tell the agent what should be built;
the lane obligations already do that. It only records:

- the exact checked backend fit, if one was found
- the approved modules, primitive refs, and helper refs for that fit
- the validation bundle and canary IDs that cover the fit
- the typed route admissibility contract and any request-local failures

In other words, contract algebra remains the constructive source of truth while
route authority is reduced to backend binding, validation ownership, and
provenance.

This separates "what must be built" from "which existing backend already
matches it", which is the key tranche-2 shift away from treating route IDs as
the whole compiler output.

Runtime Contract
----------------

The compiled contract now also has a shared helper-facing runtime substrate:

- ``ContractState``
- ``ResolvedInputs``
- ``RuntimeContext``

This keeps automatic event state separate from contract memory and gives helper
routes one stable place to read resolved market inputs and runtime metadata.
The first checked consumers are the ranked-observation basket Monte Carlo path,
single-name CDS helper routes, and nth-to-default basket-credit lowering, but
the surface is intentionally generic.

For schedule-dependent helper routes, tranche 1 now treats explicit timelines
as the practical runtime boundary. Agent-facing specs may still be normalized
from legacy date tuples, but raw string schedule fields are rejected before the
generated adapter reaches execution.

Checked Summaries And Lowering
------------------------------

The current compilation path is:

.. code-block:: text

   SemanticContract
     -> semantic validation
     -> ValuationContext
     -> RequiredDataSpec / MarketBindingSpec
     -> ProductIR
     -> typed route admissibility
     -> family lowering IR
     -> existing checked-in helper or kernel

``ProductIR`` remains the shared checked summary used by route selection.
Trellis does not currently use one flat universal numerical IR. The shipped
lowering boundary is ``ProductIR`` plus family-specific lowering IRs.

Shipped family IRs:

- ``AnalyticalBlack76IR``
- ``VanillaEquityPDEIR``
- ``ExerciseLatticeIR``
- ``CorrelatedBasketMonteCarloIR``
- ``CreditDefaultSwapIR``
- ``NthToDefaultIR``

Current Proven Families
-----------------------

The typed semantic boundary is proven end-to-end for:

- ``analytical_black76`` on vanilla options
- ``vanilla_equity_theta_pde`` on vanilla options
- ``exercise_lattice`` on callable bonds and Bermudan swaptions
- ``correlated_basket_monte_carlo`` on ranked-observation baskets
- ``credit_default_swap_analytical`` and ``credit_default_swap_monte_carlo`` on single-name CDS
- ``nth_to_default_monte_carlo`` on nth-to-default basket credit

These routes preserve the existing helper-backed pricing math. The work in this
slice changes the contract, validation, binding, admissibility, and lowering
boundaries, not the numerical kernels.

Admissibility And Authority Rules
---------------------------------

Route capability checks are now typed through ``RouteSpec.admissibility`` and
enforced through ``BuildGateDecision``.

The main tranche-1 checks cover:

- control style
- automatic-event support
- phase sensitivity
- supported outputs
- supported state tags
- multicurrency and reporting support

For the migrated families above:

- typed ``obligations`` and ``SemanticTimeline`` are authoritative for settlement
- typed ``EventMachine`` is authoritative for automatic event semantics
- legacy ``settlement_rule`` and ``event_transitions`` are mirrors only

Validation now distinguishes:

- hard errors for invalid typed semantics
- warnings when legacy mirrors are normalized or ignored for migrated routes

The semantic validator also treats compiled primitive obligations as binding
contract. If a resolved lowering plan requires a checked helper or primitive,
importing related modules is not enough; the generated code must actually call
the required symbol.

For migrated routes, the compiled blueprint metadata now also records:

- the selected family IR type and YAML-safe payload
- the normalized DSL expression kind
- the helper target bindings
- structured lowering errors with stable codes

That metadata is persisted into platform-request traces so replay tools can see
the algebra objects that drove helper selection, not just the final module
list.

Deferred Scope
--------------

The current contract algebra does not yet claim:

- ordered sequential multi-controller game semantics
- a nonlinear funding or XVA layer inside ``ValuationContext``
- a universal solver IR for every numerical backend
- portfolio-level netting or exposure algebra

Those are explicitly deferred until the typed semantic boundary is stable
across more route families.
