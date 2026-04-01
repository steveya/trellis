Lattice Algebra
===============

This note records the lattice-specific slice that Trellis currently ships. It
is the lattice counterpart to :doc:`contract_algebra`.

The goal is not one universal tree implementation. The goal is one checked
semantic-to-lattice boundary for the low-dimensional strategic-rights products
that Trellis already prices through stable helper-backed routes.

Shipped Lattice Slice
---------------------

The current lattice boundary is:

.. code-block:: text

   SemanticContract
     -> semantic validation
     -> ValuationContext
     -> ProductIR
     -> ExerciseLatticeIR
     -> helper-backed lattice route

The shipped ``ExerciseLatticeIR`` covers tranche-1 strategic-rights products:

- callable bonds
- Bermudan swaptions

It is intentionally narrow. It does not try to cover every lattice-style
product family yet.

Bellman Reading
---------------

The lattice recursion remains the standard controlled backward induction:

.. math::

   V_n(z)=
   \operatorname{Opt}_{a \in A_n(z)}
   \left[
       \sum_{z'} K_n(z,a;z')\left(C_n(z,a,z') + D_n(z,a,z')V_{n+1}(z')\right)
   \right]

where ``Opt`` is:

- ``max`` for holder optimality
- ``min`` for issuer optimality
- identity when there is no strategic choice

In the current code this is represented by:

- ``ControllerProtocol`` on the semantic contract
- ``ExerciseLatticeIR.control_style``
- ``trellis.models.trees.control.LatticeExercisePolicy``
- ``resolve_lattice_exercise_policy_from_control_style(...)``

Timing And Phase Order
----------------------

Same-day ordering is first-class in the shipped lattice slice.

The default tranche-1 phase order is:

- ``EVENT``
- ``OBSERVATION``
- ``DECISION``
- ``DETERMINATION``
- ``SETTLEMENT``
- ``STATE_UPDATE``

The lattice lowering validates, at minimum:

- observation before decision
- decision before settlement
- settlement dates not earlier than decision dates

This is enforced before helper code runs. Route helpers do not reconstruct
timing semantics from free-form settlement strings for migrated lattice paths.

Typed Contract Inputs
---------------------

The current lattice slice relies on typed semantic inputs:

- ``SemanticTimeline``
- ``ObservableSpec``
- ``StateField``
- ``ObligationSpec``
- ``ControllerProtocol``

For callable bonds, the required typed observables include:

- ``discount_curve``
- ``cashflow_schedule``

For Bermudan swaptions, the required typed observables include:

- ``forward_rate``
- ``discount_curve``

Derived schedule/rate quantities are recorded explicitly on
``ExerciseLatticeIR`` rather than being reconstructed ad hoc in every lowering
path.

Admissibility
-------------

Lattice admissibility is now typed through ``RouteSpec.admissibility`` and
checked through ``BuildGateDecision``.

The tranche-1 lattice checks cover:

- supported control style
- supported outputs
- supported state tags
- event support
- reporting and multicurrency limits

For the migrated lattice routes:

- ``holder_max`` and ``issuer_min`` are explicit
- automatic triggers remain in event/state machinery
- ordered sequential controllers are out of scope

Current Helper-Backed Routes
----------------------------

The checked helper-backed lattice routes remain:

- ``trellis.models.callable_bond_tree.price_callable_bond_tree``
- ``trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree``

This documentation change does not imply new pricing math. The current work
changes the semantic contract, admissibility, and lowering boundary only.

Deferred Scope
--------------

The lattice algebra does not yet claim:

- high-dimensional basket trees
- ordered sequential multi-controller games
- general non-Markov hybrid lattices
- universal event-overlay trees for all exotic products

Those remain future extensions. The current shipped slice is the shared
exercise-lattice boundary for callable bonds and Bermudan swaptions.
