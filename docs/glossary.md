# Trellis Glossary

This glossary defines terms that carry specific technical meaning in Trellis.
When a term appears in code, docs, or prompts, it should be read with the
definition given here.

---

## Product semantics

The typed mathematical identity of a financial product, independent of the
numerical method used to price it.

Product semantics are expressed through `ProductIR` fields and, where
applicable, the richer `ProductSemantics` / `FamilyContract` layer:

- payoff family and payoff traits
- exercise style (none, european, american, bermudan, issuer_call, holder_put)
- state dependence (terminal_markov, path_dependent, schedule_dependent)
- schedule dependence and schedule semantics
- model family (generic, black_scholes, hull_white, heston, ...)
- candidate engine families
- required market data

Product semantics answer the question *what is this product*, not *how do we
price it*.  Two products that happen to use the same numerical method may have
different product semantics (a European call and a European digital both admit
analytical pricing but carry different payoff traits).  Two products with the
same product semantics may be priced by different methods (an American put can
be priced by lattice, PDE, or Monte Carlo with LSM).

Preferred prose forms: "product semantics", "the product's semantics".

---

## Semantic validation

The static Trellis-aware typechecker that runs after code generation (Phase 2
in the build pipeline) and before module write.

Semantic validation checks whether generated code respects the product contract
established by `ProductIR` and the selected `GenerationPlan` / `PrimitivePlan`:

- engine families in the generated code match candidate engine families
- required primitives from the route appear in resolved calls
- early-exercise products use approved control primitives
- lattice exercise uses the correct exercise_fn direction
- transform pricers use vector-safe characteristic functions

Semantic validation does **not** cover:

- syntax or import validity (handled by earlier gates)
- runtime convergence or numerical accuracy (handled by post-build validation)
- module layout or file structure
- tuning parameters (step counts, path counts, grid sizes)

Preferred prose forms: "semantic validation", "the semantic validator".

---

## Solution contract

An analytical pricing formula together with the specific modeling assumptions
and payoff definition that make it valid.

Names like Black-76, Garman-Kohlhagen, and Jamshidian are shorthand for
distinct solution contracts.  They are not interchangeable just because they
all belong to the `analytical` method family.  Each solution contract implies:

- a payoff definition (e.g. European call/put on a forward vs. on an FX spot)
- modeling assumptions (e.g. lognormal forward, GBM spot, covered interest
  parity, deterministic rates)
- market data requirements (e.g. Black vol + discount vs. domestic/foreign
  discounting + FX spot)

A lattice or PDE formulation is an alternate numerical realization of the same
solution contract only when both the payoff definition and the assumption set
match.

The `analytical` label in `methods.py` remains the canonical method-family
identifier.  Solution contracts are expressed within the cookbook entries and
method requirements, not as separate method families.

---

## Method family

One of the canonical coarse-grained solver categories defined in
`trellis/agent/knowledge/methods.py`:

- `analytical` — closed-form or semi-closed-form pricing
- `rate_tree` — backward induction on calibrated short-rate lattices
- `monte_carlo` — path simulation with optional early-exercise control
- `qmc` — low-discrepancy Monte Carlo accelerators
- `pde_solver` — finite-difference theta-method pricing
- `fft_pricing` — characteristic-function transform pricing
- `copula` — correlated-default portfolio credit pricing
- `waterfall` — structured-cashflow distribution

Method families are stable identifiers used across retrieval, planning,
validation, and prompt assembly.  They are deliberately coarse: within a
method family, the cookbook and method requirements distinguish solution
contracts (for analytical) or route variants (for other families).
