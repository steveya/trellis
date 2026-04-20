Contract IR Solver Compiler
===========================

``trellis.agent.contract_ir_solver_compiler`` is the bounded bridge between the
additive ``ContractIR`` tree and a checked solver catalog.

It is deliberately narrower than ``ContractIR`` itself. The module exists to
prove that Trellis can bind solver calls from structural contract semantics
without using per-instrument route authority for the admitted families.

Role In The Build Path
----------------------

The current shipped flow is:

1. semantic validation and normalization
2. ``ContractIR`` decomposition
3. structural solver selection for the admitted exact cohort
4. route-free exact binding / lowering when structural selection succeeds
5. additive legacy comparison metadata when a bound ``MarketState`` exists

Legacy routing still exists, but only as compatibility fallback or comparison
context for requests that do not admit a structural selection.

The structural compiler is authoritative only when the request has a real
``ContractIR`` match. Everything else fails closed and leaves the fallback path
explicit.

Selection Inputs
----------------

The compiler consumes:

- ``contract_ir``: the structural payoff tree
- ``term_environment``: generic, reusable non-structural contract terms
- ``valuation_context``: requested method / output surface plus market identity
- ``market_state``: resolved capabilities and market observables

It does not consume ``ProductIR.instrument``, route ids, or backend binding
metadata during structural selection. Legacy route ids may still be copied onto
comparison records for audit-only parity tracking.

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

``SemanticImplementationBlueprint`` now carries two structural provenance
surfaces:

- ``contract_ir_solver_selection`` for the authoritative route-free declaration
  selected from ``ContractIR``
- ``contract_ir_solver_shadow`` for the compact legacy-comparison packet when a
  bound ``MarketState`` exists

The compact shadow record stores:

- declaration id
- callable ref
- requested method
- market identity / overlay identity
- resolved market coordinates
- legacy route id / family / module cohort for comparison only

``trellis.agent.platform_requests._semantic_blueprint_summary(...)`` projects
both surfaces into request metadata so runtime traces can distinguish:

- the authoritative structural selection
- the route-free exact backend binding identity
- any legacy comparison alias retained for replay or parity evidence

Generic request paths now expose the same structural boundary through the
top-level ``request.metadata["contract_ir_compiler"]`` packet. That packet is
intentionally small and stable:

- ``source`` identifies whether the structural surface came from a semantic
  blueprint or direct request decomposition
- ``contract_ir`` carries the YAML-safe structural tree
- ``contract_ir_solver_selection`` carries the authoritative structural
  declaration when one exists
- ``contract_ir_solver_shadow`` carries the compact bound comparison record
  when market-bound shadow execution succeeds
- ``shadow_status`` distinguishes ``bound``, ``contract_ir_only``, and
  ``no_match``
- ``shadow_error`` makes fail-closed no-match outcomes explicit instead of
  relying on absence as a signal

This is important because not every migrated payoff family currently has a
dedicated semantic-contract wrapper. Structural provenance therefore still
needs to remain observable on generic request paths too.

Parity Evidence
---------------

The checked parity / closure artifacts for the current Phase 3 wave live in:

- ``docs/benchmarks/contract_ir_solver_parity.json``
- ``docs/benchmarks/contract_ir_solver_parity.md``

Those artifacts remain the promotion gate for any additional Phase 4 cutover.
A family should not be treated as route-retirement-ready merely because it
binds structurally; the ledger must also show sufficient comparison evidence
and an explicit non-blocked status.

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
