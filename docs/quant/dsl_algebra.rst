DSL Algebra
===========

This note records the algebraic reading of the internal semantic DSL as it is
currently shipped.

The short version is:

- the control-free fragment is a typed linear composition core
- strategic rights are explicit Bellman-style choice operators
- lowering happens after semantic validation and valuation-context binding

This is semiring-style for compiler intuition, but the concrete implementation
is a typed interface algebra over checked helper-backed routes.

Shipped Algebraic Split
-----------------------

Trellis keeps two distinct layers:

1. control-free composition
2. explicit control or choice

The control-free fragment is represented by:

- ``ContractSignature``
- ``ContractAtom``
- ``ContractZero``
- ``ContractUnit``
- ``ScaleExpr``
- ``AddExpr``
- ``ThenExpr``

The strategic-rights fragment is represented by:

- ``ChoiceExpr``
- ``choose_holder(...)``
- ``choose_issuer(...)``

This split matters because linear rewrites are valid only in the control-free
fragment.

Typed Linear Core
-----------------

The linear fragment is typed by:

- input and output ports
- required timeline roles
- required market-data capabilities

For compatible interfaces, Trellis normalizes:

- additive portfolio superposition
- sequential composition
- identity and zero elimination
- canonical ordering of compatible additive terms

This gives deterministic normalization without pretending that optionality is
ordinary addition.

Choice Layer
------------

Strategic rights stay explicit. In the current shipped slice the relevant
control styles are:

- ``identity``
- ``holder_max``
- ``issuer_min``

These map onto Bellman-style operators and then onto checked-in route helpers.
Automatic triggers do not become choice operators. They remain in typed
event/state machinery.

Executable Compiler Boundary
----------------------------

The current lowering pipeline is:

.. code-block:: text

   SemanticContract
     -> validate_semantic_contract(...)
     -> ValuationContext
     -> ProductIR
     -> family lowering IR
     -> helper-backed DSL lowering

The concrete bridge lives in:

- ``trellis.agent.semantic_contract_compiler.compile_semantic_contract(...)``
- ``trellis.agent.dsl_lowering.lower_semantic_blueprint(...)``

The DSL is not the same thing as a route helper. The DSL lowers onto helpers.

Shipped family-specific lowering IRs:

- ``AnalyticalBlack76IR``
- ``VanillaEquityPDEIR``
- ``ExerciseLatticeIR``
- ``CorrelatedBasketMonteCarloIR``

Current Supported Lowerings
---------------------------

The executable lowering slice currently proves:

- vanilla analytical Black76 kernel lowering
- vanilla theta-method PDE helper lowering
- callable-bond and Bermudan-swaption exercise-lattice lowering
- ranked-observation basket Monte Carlo lowering

For these migrated paths:

- typed semantic fields are authoritative
- legacy settlement and event mirrors are not the truth source
- typed route admissibility is checked before helper code runs

Safe And Unsafe Rewrites
------------------------

Safe rewrites in the shipped control-free fragment:

- flatten nested additive portfolios
- flatten sequential chains
- remove ``unit`` from sequential chains
- remove ``zero`` from additive portfolios
- canonicalize equivalent linear fragments with matching signatures

Unsafe rewrites:

- distributing choice over portfolio addition without proof
- collapsing issuer and holder control into one untyped operator
- treating automatic triggers as strategic exercise
- rewriting control branches as if they were ordinary additive legs

Warning And Error Policy
------------------------

The compiler now draws a hard line between:

- semantic validation errors
- admissibility failures
- successful lowerings with warnings

Warnings are used when legacy mirrors are normalized or ignored for migrated
routes. Errors are used when typed semantics are inconsistent or when a route
cannot support the requested control, outputs, or state tags.

Deferred Scope
--------------

The DSL algebra does not yet attempt:

- a universal numerical IR for all route families
- ordered sequential multi-controller protocols
- symbolic proof search over analytical rewrites
- nonlinear portfolio funding or XVA composition

Those remain future extensions. The current design goal is a narrow, typed, and
deterministic lowering boundary for proven helper-backed routes.
