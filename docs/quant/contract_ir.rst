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
Concrete cadence matches fail closed for irregular or cadence-free
schedules instead of guessing. Frequency wildcards remain unconstrained:
an anonymous wildcard matches without inference, and a named wildcard
binds the inferred cadence when one exists or ``None`` otherwise.

That matters for the next phase:

- Phase 3 kernel declarations match structural payoff templates against
  ``blueprint.contract_ir``
- Phase 4 can retire direct dependence on hard-coded instrument routing for
  fresh builds, because the structural contract tree becomes the primary match
  surface

The target state is explicit: even a simple rebuilt vanilla option should be
able to go through ``ContractIR -> pattern match -> lowering obligations``
without needing a direct hard-coded route by instrument name.

Phase 3 Structural Solver Compiler
----------------------------------

Phase 3 introduces the first bounded structural compiler on top of
``ContractIR``:

- ``trellis.agent.contract_ir_solver_compiler.compile_contract_ir_solver(...)``
- ``trellis.agent.contract_ir_solver_compiler.execute_contract_ir_solver_decision(...)``

This compiler is still additive in the current shipped state. It does not
replace the legacy route path yet. Instead it proves that a fresh build can
bind a checked solver call directly from:

- ``contract_ir``
- a generic non-structural term environment
- the requested method / output surface
- a bound ``MarketState``

and do so without consulting ``ProductIR.instrument`` or any route id during
selection.

Normalized Term Environment
---------------------------

Some checked helpers need reusable contract conventions that are intentionally
not baked into the structural payoff tree. Phase 3 therefore uses a generic
term environment rather than product-specific payloads.

The current groups are:

- ``CashSettlementTerms`` for notional, payout currency, and settlement kind
- ``AccrualConventionTerms`` for day-count and payment-frequency conventions
- ``FloatingRateReferenceTerms`` for index / curve-name references
- ``QuoteGridTerms`` for bounded observable quote grids such as variance
  replication strike-vol pairs

The important boundary rule is unchanged:

- if a field changes structural payoff family or pattern match, it belongs in
  ``ContractIR``
- if it preserves the structural family but changes helper materialization, it
  may belong in the generic term environment

Phase 3 Families
----------------

The current bounded structural-solver wave covers:

1. European call / put terminal ramps through the Black76 basis kernels in
   ``trellis.models.black``
2. Cash-or-nothing and asset-or-nothing digitals through the same Black76
   basis family
3. European payer / receiver swaptions through
   ``trellis.models.rate_style_swaption.price_swaption_black76``
4. Two-asset analytical basket / spread call / put payoffs through
   ``trellis.models.basket_option.price_basket_option_analytical``
5. Equity variance swaps through
   ``trellis.models.analytical.equity_exotics.price_equity_variance_swap_analytical``

The equity vanilla / digital lane intentionally mirrors the current checked
zero-carry parity contract:

.. code-block:: python

   forward = spot / discount_factor
   price = discount_factor * black76_basis_kernel(forward, strike, sigma, T)

That is the contract Phase 3 shadow-mode parity defends today. A future
carry-aware lane will need an explicit market capability and a documented
carry-source policy before it can replace this basis.

Swaptions also keep the current checked helper conventions. When the semantic
term surface does not provide a more specific payment frequency, the structural
adapter defaults the fixed leg to semiannual rather than inheriting a coarse
annualized decomposition schedule.

Variance swaps reuse the existing bounded smile-based analytical helper. The
``VarianceObservable`` node names the contractual observed quantity. Optional
quote-grid inputs remain helper materialization detail carried through
``QuoteGridTerms``.

Shadow Mode And Provenance
--------------------------

When ``compile_semantic_contract(...)`` receives a bound ``MarketState`` and
the structural compiler can bind one of the admitted families, the blueprint
now also carries ``contract_ir_solver_shadow``.

That compact record includes:

- the structural declaration id
- the checked callable ref
- requested method
- market identity and overlay identity
- adapter-resolved market-coordinate provenance
- the legacy route id / family / module cohort used only for comparison

This is intentionally a shadow surface in Phase 3:

- structural compilation is authoritative for the comparison record
- legacy route metadata is carried only for observability
- failure to bind the structural compiler does not perturb the live route path

``compile_build_request(...)`` threads the same summary onto request metadata
under ``request.metadata["semantic_blueprint"]["contract_ir_solver_shadow"]``.

For request paths that do not compile through a semantic blueprint but still
decompose route-freely, ``compile_build_request(...)`` now also emits a
top-level additive summary under ``request.metadata["contract_ir_compiler"]``.

That packet records:

- whether the structural view came from ``semantic_blueprint`` or direct
  request decomposition
- the YAML-safe ``contract_ir`` summary
- the compact structural shadow record when binding succeeds
- ``shadow_status``:

  - ``bound`` when the structural compiler selected a declaration
  - ``contract_ir_only`` when decomposition succeeded but no bound market was
    available
  - ``no_match`` when the family is representable but intentionally outside
    the admitted Phase 3 solver wave

- an explicit ``shadow_error`` payload for fail-closed no-match outcomes

This makes the structural boundary visible even for generic free-form request
paths such as digitals, terminal baskets, and variance swaps that are not all
wrapped by dedicated semantic contracts yet.

Parity Ledger
-------------

The checked Phase 3 parity / closure ledger lives in:

- ``docs/benchmarks/contract_ir_solver_parity.json``
- ``docs/benchmarks/contract_ir_solver_parity.md``

That ledger is the current promotion gate for Phase 4. It distinguishes
families that are merely representable from families that have enough
selection, parity, and provenance evidence to be considered phase-4-ready.

Explicit Phase 3 Non-goals
--------------------------

Arithmetic Asians remain representable in ``ContractIR`` but are still an
explicit no-match for the structural solver. The current checked repository
does not expose a dedicated analytical arithmetic-Asian helper that meets the
Phase 3 migration contract, so those products remain on the legacy route path.

That distinction is deliberate:

- ``ContractIR`` representation coverage is broader than the current migrated
  solver wave
- "IR exists" must not be read as "the family is already route-free"
- explicit ``shadow_status = "no_match"`` is part of the governed blocker
  surface, not an incidental omission

Phase 4 is the retirement phase. Phase 3 only proves that the admitted
families can already price through the structural compiler with parity and
provenance recorded in shadow mode.
