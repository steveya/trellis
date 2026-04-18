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
- family drafting now runs through an ordered draft-rule registry rather than
  one giant branch-ordered parser
- admissible methods now come from registry-backed
  ``SemanticFamilyDefinition`` plus ``SemanticMethodSurfaceDefinition`` entries
  instead of being re-encoded independently in the request/compiler layers

Legacy mirrors such as ``settlement_rule`` and ``event_transitions`` still
exist on ``SemanticProductSemantics``, but they are no longer the authority for
migrated route families.

That registry-backed structure also means lower layers now share one
specialization authority when a contract needs to be rebuilt for a different
preferred method. Request compilation, semantic compilation, and runtime
metadata all consume the same family/method surface instead of each carrying
its own family-local branching.

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

- the thin compatibility route alias, if one still exists
- the nested backend-binding record for the exact checked fit, if one was found
- the approved modules, primitive refs, helper refs, and admissibility facts
  for that binding
- the validation bundle and canary IDs that cover the fit
- the typed route admissibility contract and any request-local failures

In other words, contract algebra remains the constructive source of truth while
route authority is reduced to backend binding, validation ownership, and
provenance.

This separates "what must be built" from "which existing backend already
matches it", which is the key tranche-2 shift away from treating route IDs as
the whole compiler output.

The trace and checkpoint layer now reflects that same split through a
``construction_identity`` summary. That summary is family-first:

- exact backend bindings surface the stable binding id as the primary identity
- otherwise the surfaced identity falls back to the lowered family IR or lane
  family
- the route alias is retained only as secondary backend/provenance context

So operators see the constructive meaning first and the compatibility alias
second.

Compatibility aliases are now governed explicitly. Migrated exact-helper
families may mark the route alias as internal-only, which means:

- the raw route id still survives in validation/replay/canary metadata
- operator-facing traces and prompts omit the alias when the backend binding id
  already carries the full meaning
- retained aliases remain visible only where replay or compatibility still
  depends on them

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
     -> EventProgramIR / ControlProgramIR
     -> typed route admissibility
     -> family lowering IR
     -> existing checked-in helper or kernel

``ProductIR`` remains the shared checked summary used by route selection.
Trellis does not currently use one flat universal numerical IR. The shipped
lowering boundary is ``ProductIR`` plus one shared event/control compiler
program plus family-specific lowering IRs.

The shared semantic compiler program is:

- ``EventProgramIR``
- ``ControlProgramIR``

That program is now the universal authority for:

- schedule-aware event meaning
- control/exercise meaning
- same-day phase ordering carried into numerical lowering

Family IRs do not invent their own event semantics anymore. They project the
shared program into bounded numerical surfaces that each family can support.

Shipped family IRs:

- ``AnalyticalBlack76IR``
- ``EventAwareMonteCarloIR`` as the new bounded single-state Monte Carlo family
  surface
- ``TransformPricingIR`` as the bounded terminal-only transform family surface
- ``EventAwarePDEIR``
- ``VanillaEquityPDEIR`` as the current compatibility wrapper for the vanilla
  theta-method PDE route
- ``ExerciseLatticeIR``
- ``CorrelatedBasketMonteCarloIR``
- ``EventTriggeredTwoLeggedContractIR`` as the structural helper-backed family
  surface for event-triggered two-legged contracts, currently proven on
  single-name CDS
- ``NthToDefaultIR``

For transform routes, the compiler now also carries an explicit bounded family
surface on ``TransformPricingIR``. That surface keeps transform admissibility
restricted to:

- terminal-only payoff semantics
- numerical-lane ``identity`` control
- one typed characteristic-function family such as ``gbm_log_spot`` or
  ``heston_log_spot``
- explicit quote and strike semantics
- a backend capability split between helper-backed diffusion execution and
  raw-kernel stochastic-volatility execution

That means the transform route no longer admits on the full upstream
``vanilla_option`` contract. Admissibility reads the lowered transform family
contract first, so upstream semantic tags such as ``holder_max`` or
``recombining_safe`` stop leaking into transform-only route checks.

For PDE routes, the compiler now carries typed sub-specifications inside
``EventAwarePDEIR``:

- ``PDEStateSpec``
- ``PDEOperatorSpec``
- ``PDEEventTimeSpec``
- ``PDEEventTransformSpec``
- ``PDEControlSpec``
- ``PDEBoundarySpec``

Those PDE-specific objects are now projections of the shared
``EventProgramIR`` / ``ControlProgramIR`` boundary rather than a separate
family-local event language.

The runtime now has a matching bounded rollback surface under
``trellis.models.pde.event_aware``:

- ``EventAwarePDEGridSpec``
- ``EventAwarePDEOperatorSpec``
- ``EventAwarePDEBoundarySpec``
- ``EventAwarePDEEventBucket``
- ``EventAwarePDETransform``
- ``EventAwarePDEProblemSpec``
- ``EventAwarePDEProblem``

For Monte Carlo routes, the compiler now also carries a bounded event-aware
family surface:

- ``MCStateSpec``
- ``MCProcessSpec``
- ``MCEventTimeSpec``
- ``MCEventSpec``
- ``MCPathRequirementSpec``
- ``MCPayoffReducerSpec``
- ``MCControlSpec``
- ``MCMeasureSpec``
- ``MCCalibrationBindingSpec``
- ``EventAwareMonteCarloIR``

Those Monte Carlo-specific objects are likewise projections of the shared
``EventProgramIR`` / ``ControlProgramIR`` boundary. This is why a European
rate-style swaption can now carry semantic holder-exercise control at the
universal layer while still lowering onto an identity-control Monte Carlo
numerical contract where the exercise right is absorbed into the payoff
reducer.

The runtime now has the matching bounded problem-assembly surface under
``trellis.models.monte_carlo.event_aware``:

- ``EventAwareMonteCarloProcessSpec``
- ``EventAwareMonteCarloEvent``
- ``EventAwareMonteCarloProblemSpec``
- ``EventAwareMonteCarloProblem``

That layer resolves bounded process families, compiles deterministic
event schedules into ``PathEventTimeline`` replay contracts, and assembles
``StateAwarePayoff`` objects over reduced Monte Carlo state. Trellis still
does not claim that all vanilla or schedule-driven Monte Carlo routes have
migrated onto that family. The intended end state is to express the existing
simple Monte Carlo routes plus bounded one-factor event-driven cases through
``EventAwareMonteCarloIR`` rather than through generic route-local synthesis.

The first vanilla migration slice is now checked in as well:

- vanilla European Monte Carlo lowers as a terminal-only ``gbm_1d`` family
  instance with no synthetic event-replay timeline
- the existing local-vol vanilla helper remains a compatibility surface, but
  it now delegates into the generic event-aware Monte Carlo runtime rather than
  owning a separate engine/payoff implementation

In the current tranche this is still a bounded 1D family surface. It makes the
operator family and event-transform contract explicit before the generic
rollback assembly is introduced. That rollback layer is now checked in and
supports deterministic event buckets plus bounded ``identity``,
``holder_max``, ``issuer_min``, and ``state_remap`` transforms without a
product-specific rollback loop. ``VanillaEquityPDEIR`` now inherits from that
surface so the existing vanilla route remains stable while future PDE families
can migrate onto the same compiler boundary. It is transitional-only: the end
state is for the vanilla route to emit a plain ``EventAwarePDEIR`` once
downstream traces and review surfaces no longer rely on the legacy wrapper
type.

The first migrated runtime consumer is the checked vanilla-equity PDE helper in
``trellis.models.equity_option_pde``. It now assembles an
``EventAwarePDEProblem`` with an empty event timeline and ``identity`` control
instead of maintaining a separate vanilla-only rollback implementation.

For schedule-driven PDE requests, lowering can now preserve explicit event-time
payloads even when no checked rollback helper exists yet. In that case the
compiler emits typed event buckets and transform kinds on ``EventAwarePDEIR``
so lane obligations, admissibility, and traces see normalized semantics such as
cashflow additions or exercise projections instead of raw ``schedule_state``
tags leaking past lowering.

Current Proven Families
-----------------------

The typed semantic boundary is proven end-to-end for:

- ``analytical_black76`` on vanilla options
- ``vanilla_equity_theta_pde`` on vanilla options
- ``pde_theta_1d`` on bounded event-aware 1D rollback, including holder-max
  equity exercise and issuer-min Hull-White callable bonds
- ``exercise_lattice`` on callable bonds and Bermudan swaptions
- ``correlated_basket_monte_carlo`` on ranked-observation baskets
- ``credit_default_swap`` on single-name CDS across analytical and Monte Carlo
  bindings, routed through the structural
  ``event_triggered_two_legged_contract`` family
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
- helper-backed analytical / PDE / FFT routes admit on lowered family IR and
  typed schedule/state semantics, not on route-card instructions
- structural cap/floor strips lowered onto ``AnalyticalBlack76IR`` use typed
  ``schedule_state`` authority and no longer fail the generic
  ``unsupported_event_support:automatic_triggers`` gate

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
list. The trace boundary now also projects
``generation_boundary.lowering.family_ir_summary`` as the review-friendly PDE
surface: operator family, control style, event-transform kinds, event dates,
and the transitional wrapper state when a route still uses
``VanillaEquityPDEIR``.

Deferred Scope
--------------

The current contract algebra does not yet claim:

- ordered sequential multi-controller game semantics
- a nonlinear funding or XVA layer inside ``ValuationContext``
- a universal solver IR for every numerical backend
- portfolio-level netting or exposure algebra

Those are explicitly deferred until the typed semantic boundary is stable
across more route families.
