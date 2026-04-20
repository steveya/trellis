Contract IR Solver Compiler
===========================

``trellis.agent.contract_ir_solver_compiler`` is the Phase 3 bridge between
the additive ``ContractIR`` tree and a bounded checked-solver catalog.

It is deliberately narrower than ``ContractIR`` itself. The module exists to
prove that Trellis can bind solver calls from structural contract semantics
without using per-instrument route authority for the admitted families, while
still keeping the current runtime route path intact during rollout.

Role In The Build Path
----------------------

The current Phase 3 flow is:

1. semantic validation and normalization
2. ``ContractIR`` decomposition
3. legacy route / DSL lowering as before
4. additive structural shadow compilation when a bound ``MarketState`` exists

The shadow compiler is therefore observational today:

- it may attach a structural decision to the semantic blueprint
- it may attach structural provenance to request metadata
- it must not change the legacy route selected for live execution

Selection Inputs
----------------

The compiler consumes:

- ``contract_ir``: the structural payoff tree
- ``term_environment``: generic, reusable non-structural contract terms
- ``valuation_context``: requested method / output surface plus market identity
- ``market_state``: resolved capabilities and market observables

It does not consume ``ProductIR.instrument``, route ids, or backend binding
metadata during structural selection. Those values may be copied onto the
result only as shadow comparison data.

Declaration Shape
-----------------

Each declaration remains factored into four concerns:

1. selection authority
2. output support
3. market requirements
4. callable materialization

The checked registry substrate lives in
``trellis.agent.contract_ir_solver_registry``. The Phase 3 compiler builds a
bounded default registry on top of that substrate.

Current Generic Term Groups
---------------------------

The shipped structural compiler uses reusable term groups instead of
instrument-keyed economic-term blobs:

- ``CashSettlementTerms``
- ``AccrualConventionTerms``
- ``FloatingRateReferenceTerms``
- ``QuoteGridTerms``

The design rule is strict:

- do not introduce ``VanillaEconomicTerms`` / ``SwaptionEconomicTerms`` style
  payloads
- if a helper cannot be bound from ``ContractIR`` plus reusable term groups,
  that family is not structurally ready yet

Admitted Families
-----------------

The default Phase 3 registry admits:

1. European Black76 call / put ramps
2. Cash-or-nothing and asset-or-nothing digitals
3. European payer / receiver swaptions via ``price_swaption_black76``
4. Two-asset analytical basket / spread call / put helpers
5. Equity variance swaps via ``price_equity_variance_swap_analytical``

Arithmetic Asians are intentionally excluded. The compiler fails closed on
that family until a checked solver surface exists.

Failure And Ambiguity Policy
----------------------------

The compiler is intentionally fail-closed:

- no admissible declaration -> ``ContractIRSolverNoMatchError``
- multiple admissible top-precedence declarations ->
  ``ContractIRSolverAmbiguityError``

Registration order is not a semantic tiebreak. Precedence must make any
intentional overlap explicit.

Observability Surface
---------------------

``SemanticImplementationBlueprint`` now carries an additive
``contract_ir_solver_shadow`` field. The compact shadow record stores:

- declaration id
- callable ref
- requested method
- market identity / overlay identity
- resolved market coordinates
- legacy route id / family / module cohort for comparison only

``trellis.agent.platform_requests._semantic_blueprint_summary(...)`` projects
that record into request metadata so runtime traces can compare route and
structural authority during the Phase 3 rollout.

Extension Guidance
------------------

When adding a new structural family:

1. prove the family is representable in ``ContractIR`` without helper-shaped
   escape hatches
2. bind any extra helper terms through reusable generic groups, not a
   product-family blob
3. add a declaration with explicit precedence and capability requirements
4. add parity tests against the checked helper or basis kernel
5. document any still-blocked neighboring family explicitly instead of
   widening the declaration past the validated helper contract
