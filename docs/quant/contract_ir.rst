Contract IR
===========

``ContractIR`` is the additive structural payoff tree introduced for the
Phase 2 contract-compiler work. It does not replace ``ProductIR`` yet.
Instead, it sits beside the flat routing record and captures the part that
Phase 3 and Phase 4 actually need for route-free fresh builds:

- the payoff expression tree
- the exercise surface
- the observation surface
- the underlier specification

Why It Exists
-------------

``ProductIR`` is intentionally coarse. It is good for broad routing and
knowledge retrieval, but it collapses structurally different contracts onto
shared string families such as ``vanilla_option`` or ``swaption``.

``ContractIR`` keeps the structural information that a kernel matcher needs.
For example, these two products are no longer separated only by instrument
name:

- ``max(Spot - Strike, 0)``
- ``Annuity * max(SwapRate - Strike, 0)``

The second still carries swaption-specific structure, but the shared ramp
shape is now explicit and machine-readable.

Phase 2 keeps the rollout additive:

- existing routing still reads ``ProductIR``
- ``SemanticImplementationBlueprint`` now also carries ``contract_ir``
- unsupported or out-of-scope products attach ``None`` instead of failing

Current Surface
---------------

The shipped Phase 2 AST lives in ``trellis.agent.contract_ir`` and is built
from frozen dataclasses.

Top-level contract:

.. code-block:: python

   ContractIR(
       payoff=...,
       exercise=Exercise(...),
       observation=Observation(...),
       underlying=Underlying(...),
   )

Schedules:

- ``Singleton(date)``
- ``FiniteSchedule(tuple[date, ...])``
- ``ContinuousInterval(start, end)``

Underliers:

- ``EquitySpot(name, dynamics)``
- ``ForwardRate(name, dynamics)``
- ``RateCurve(name, dynamics)``
- ``CompositeUnderlying(parts=...)``

Payoff nodes:

- leaves: ``Constant``, ``Strike``, ``Spot``, ``Forward``, ``SwapRate``,
  ``Annuity``, ``VarianceObservable``
- structural nodes: ``LinearBasket``, ``ArithmeticMean``, ``Max``, ``Min``,
  ``Add``, ``Sub``, ``Mul``, ``Scaled``, ``Indicator``

Naming follows the semantic ``Observable`` versus quote-map ``Quote``
discipline documented in :doc:`contract_algebra`: nodes are named after the
contractual quantity they denote, not after a downstream pricing method.

Predicates:

- comparisons: ``Gt``, ``Ge``, ``Lt``, ``Le``
- boolean combinators: ``And``, ``Or``, ``Not``

Well-Formedness
---------------

The constructors enforce the local Phase 2 invariants:

- underlier names must be unique
- every payoff underlier reference must resolve against the root underlier set
- schedule-bearing leaves must use concrete schedule objects, not string refs
- ``FiniteSchedule`` must be non-empty and strictly increasing
- ``SwapRate`` and ``Annuity`` require ``FiniteSchedule``
- ``VarianceObservable`` requires ``ContinuousInterval``
- European exercise uses ``Singleton``
- terminal observation uses ``Singleton``

Phase 2 intentionally does not introduce top-level heterogeneous composites.
If multiple legs can share one root surface, they stay inside ``payoff`` via
``Add``, ``Mul``, or ``LinearBasket``. A richer leg-based root is tracked
separately.

Canonicalization
----------------

``canonicalize(expr)`` puts payoff expressions into a stable structural form
for matching.

The important current rules are:

- flatten and sort commutative nodes
- fuse constants in ``Add`` and ``Mul``
- drop additive and multiplicative identities
- normalize ``LinearBasket`` zero-weight and singleton cases
- keep option side in ``Sub`` operand order
- preserve factorized positive outer scales instead of distributing them
  across ramps when the factor is already shared structurally

The last rule matters:

- call ramp: ``Max(Sub(X, Strike(K)), Constant(0))``
- put ramp: ``Max(Sub(Strike(K), X), Constant(0))``

A put is not a negatively scaled call. The canonicalizer does not rewrite
short-call exposure into put orientation.

Phase 2 also prefers factored positive outer scales. When a positive scalar is
common across a ramp, the canonical form keeps:

.. code-block:: python

   Scaled(weight, Max(Sub(lhs, rhs), Constant(0)))

instead of eagerly expanding the weight across the ``Max`` arguments. This is
the structural form that downstream pattern matching consumes.

That normalization contract is defended by property-based tests covering:

- idempotence of ``canonicalize``
- ordering confluence for commutative nodes
- numerical semantic preservation under synthetic payoff environments

Phase 2 Families
----------------

The bounded family set implemented today is:

1. European terminal linear payoffs
2. variance-settled payoffs
3. digital payoffs
4. arithmetic Asians

Examples:

.. code-block:: python

   # Vanilla call
   Max((Sub(Spot("AAPL"), Strike(150.0)), Constant(0.0)))

   # European payer swaption
   Scaled(
       Annuity("USD-IRS-5Y", schedule),
       Max((Sub(SwapRate("USD-IRS-5Y", schedule), Strike(0.05)), Constant(0.0))),
   )

   # Variance swap
   Scaled(
       Constant(10000.0),
       Sub(VarianceObservable("SPX", interval), Strike(0.04)),
   )

   # Cash-or-nothing digital
   Mul((Constant(2.0), Indicator(Gt(Spot("AAPL"), Strike(150.0)))))

   # Arithmetic Asian
   Max((Sub(ArithmeticMean(Spot("SPX"), avg_schedule), Strike(4500.0)), Constant(0.0)))

Decomposition And Compilation
-----------------------------

``trellis.agent.knowledge.decompose.decompose_to_contract_ir(...)`` provides a
bounded, fixture-driven natural-language bridge for the four Phase 2 families.
It returns:

- a well-formed ``ContractIR`` for supported descriptions
- ``None`` for out-of-scope families such as barriers, lookbacks, callable
  bonds, Bermudan exercise, or leg-based products

``trellis.agent.semantic_contract_compiler.compile_semantic_contract(...)``
threads that result onto ``SemanticImplementationBlueprint.contract_ir``.

That field is attached before route-specific lowering choices. In other words,
``contract_ir`` is intended to be route-free compiler input, not a derived
summary of the selected route.

Pattern Matching
----------------

Phase 2 also extends ``ContractPattern`` evaluation so patterns can match
directly against ``ContractIR`` trees.

Regular ``FiniteSchedule`` cadences can now participate in
``schedule.frequency`` matching for the common discrete cases
(``weekly``, ``monthly``, ``quarterly``, ``semiannual``, ``annual``).
Irregular or cadence-free schedules fail closed instead of guessing.

That matters for the next phase:

- Phase 3 kernel declarations match structural payoff templates against
  ``blueprint.contract_ir``
- Phase 4 can retire direct dependence on hard-coded instrument routing for
  fresh builds, because the structural contract tree becomes the primary match
  surface

The target state is explicit: even a simple rebuilt vanilla option should be
able to go through ``ContractIR -> pattern match -> lowering obligations``
without needing a direct hard-coded route by instrument name.
