# Contract IR — Phase 3: Structural Solver Compiler

## Status

Draft. Pre-queue design document. Not yet the live execution mirror for
Phase 3.

## Linked Linear

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-904 — Phase 2 umbrella (completed additive IR substrate)
- QUA-905 — Phase 3 umbrella (this plan expands)
- QUA-906 — Phase 4 umbrella (consumes Phase 3 parity and provenance)
- QUA-903 — Phase 1 (Done; pattern-keyed registry)
- QUA-916 — Phase 1.5 (Done; `ContractPattern` AST + evaluator)

## Companion Docs

- `doc/plan/draft__contract-ir-phase-2-ast-foundation.md`
- `doc/plan/draft__contract-ir-phase-4-route-retirement.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__contract-ir-compiler-retiring-route-registry.md`
- `docs/quant/contract_ir.rst`
- `docs/quant/contract_algebra.rst`

## Purpose

Specify the first compiler that consumes `blueprint.contract_ir` as
selection authority for fresh builds.

Phase 2 proved that Trellis can carry a route-free structural contract
tree alongside `ProductIR`. Phase 3 is where that tree becomes load
bearing: the compiler must match structural solver declarations against
the Contract IR, resolve the required market observables, and produce a
bound checked solver call without consulting hard-coded per-instrument
routes.

The target is not "IR exists." The target is:

- a fresh build for a migrated family can choose a checked solver from
  `(contract_ir, preferred method, valuation context, market
  capabilities)`
- masking `ProductIR.instrument`, `route_id`, and `route_family` does
  not change the selected solver for that migrated family
- the output remains parity-equivalent to the current route-based path
  on all benchmark and fixture surfaces in scope

## Framing

### Why this is its own phase

Phase 2 was intentionally additive. It made the semantic structure
available but did not change who makes the dispatch decision.

Phase 3 is the first phase that changes authority:

- from route registry clauses keyed on `ProductIR` summaries
- to structural solver declarations keyed on `ContractIR`

That is a qualitatively different step from "attach a new field to the
blueprint." It needs its own mathematical contract, ambiguity policy,
parity harness, and rollout plan.

### What stays additive

Phase 3 still does NOT:

- delete `routes.yaml`
- delete `backend_bindings.yaml`
- delete `ProductIR.instrument`
- remove route-based fallback for unmigrated surface
- claim universal coverage for every Phase 2 family

Those are Phase 4 actions. Phase 3 proves solver selection and parity in
shadow mode first.

### Scope discipline

Phase 2's IR family set is larger than Phase 3's first migration wave.
That is intentional.

Current checked solver surfaces in the repo support the following
Phase-3-first-wave families cleanly:

1. European terminal vanilla ramp via the Black76 basis kernels in
   `trellis.models.black`
2. Cash-or-nothing and asset-or-nothing digital payoffs via the same
   Black76 basis family
3. European payer / receiver rate-style swaptions via
   `trellis.models.rate_style_swaption.price_swaption_black76`
4. Two-asset analytical basket / spread payoffs via
   `trellis.models.basket_option.price_basket_option_analytical`
5. Equity variance swaps via
   `trellis.models.analytical.equity_exotics.price_equity_variance_swap_analytical`

Arithmetic Asians remain in Contract IR scope, but they are NOT a
first-wave Phase 3 migration target because the current checked model
surface does not expose a dedicated analytical arithmetic-Asian helper.
They stay on the legacy route path until a real solver declaration
surface exists.

That distinction matters. "IR can represent it" is not the same as
"Phase 3 can migrate it safely."

## Mathematical Contract

### Input surfaces

For Phase 3 the fresh-build selection problem is parameterized by:

- `c`: a well-formed `ContractIR`
- `v`: valuation context and requested method surface
- `m`: resolved market state / market binding surface
- `h`: optional legacy metadata (`ProductIR`, route aliases, old route
  ids) carried only for parity and observability during rollout

The structural compiler must be a function of `(c, v, m)` for migrated
surface. Legacy metadata `h` may be logged, compared, or replayed during
shadow mode, but it is not allowed to change the chosen solver for a
migrated contract.

### Solver declaration

Each solver declaration is a tuple

$$d = (p, \mathcal{M}, \mathcal{Q}, A, \kappa, \pi, \rho)$$

where:

- `p` is a `ContractPattern`
- `\mathcal{M}` is the admissible requested-method set
- `\mathcal{Q}` is the required market capability set
- `A` is an adapter from `(c, v, m, \theta)` to the callable's native
  argument surface
- `\kappa` is the checked callable or helper reference
- `\pi` is an explicit integer precedence for overlapping declarations
- `\rho` is provenance metadata: declaration id, helper refs,
  validation bundle ids, compatibility alias policy

The adapter output may be one of two forms:

1. raw-kernel kwargs, e.g. `dict(F=..., K=..., sigma=..., T=...)`
2. helper-call materialization, e.g. `dict(market_state=m,
   spec=Resolved...Spec(...))`

That distinction is deliberate. The current repo exposes both raw
analytical kernels (`black76_call`, `black76_put`,
`black76_cash_or_nothing_call`, ...) and exact helper wrappers
(`price_swaption_black76`, `price_basket_option_analytical`,
`price_equity_variance_swap_analytical`). Phase 3 must support both so
it can migrate real checked surfaces rather than force them through an
artificial one-style-only abstraction.

### Structural match relation

Let `canon(c.payoff)` denote Phase 2 canonicalization of the payoff
expression. Let

$$\operatorname{Match}(p, c) = \theta$$

mean `evaluate_pattern(p, c)` succeeds and returns bindings `\theta`.

The denotation of a pattern is:

$$\llbracket p \rrbracket = \{ c \mid \operatorname{Match}(p, c) \text{ succeeds} \}$$

A solver declaration `d` is admissible for `(c, v, m)` when all of the
following hold:

1. `c` is well-formed
2. `\operatorname{Match}(p_d, c)` succeeds with bindings `\theta_d`
3. `\text{method}(v) \in \mathcal{M}_d`
4. `\mathcal{Q}_d \subseteq \text{Capabilities}(m)`
5. `A_d(c, v, m, \theta_d)` is defined

The bound solver call is then:

$$\operatorname{Call}_d(c, v, m) = \kappa_d\big(A_d(c, v, m, \theta_d)\big)$$

### Selection semantics

Let `D(c, v, m)` be the set of admissible declarations.

The ideal semantic order is set inclusion on pattern denotations:

$$d_1 \succ d_2 \quad \text{if} \quad \llbracket p_{d_1} \rrbracket \subset \llbracket p_{d_2} \rrbracket$$

but exact subset checking is not an implementable runtime rule. Phase 3
therefore uses:

1. explicit declaration precedence `\pi`
2. deterministic stable registration order as the last tiebreak

with a review invariant:

- if two declarations overlap and neither is intentionally subordinate,
  equal precedence is a build-time error
- if `d_1` is the strictly narrower declaration in the design, it must
  carry strictly higher precedence than `d_2`

So the runtime selection is:

$$d^\* = \arg\max_{d \in D(c,v,m)} (\pi_d, -\operatorname{registration\_index}(d))$$

and the compiler must fail closed on ambiguous equal-precedence overlaps.
It must not silently "pick whichever registered first" for two intended
peers.

### Fresh-build invariance

For any two legacy-metadata packets `h_1, h_2` that differ only in
`ProductIR.instrument`, `route_id`, `route_family`, or compatibility
alias fields, migrated fresh-build selection must satisfy:

$$\operatorname{Select}(c, v, m, h_1) = \operatorname{Select}(c, v, m, h_2)$$

This is the Phase 3 contract that Phase 4 later promotes from shadow
mode to primary authority.

## Family Math For The First Migration Wave

This section is not a textbook of all pricing theory. It states the
actual solver math Phase 3 will target in the current repo.

### 1. Vanilla terminal ramp under Black76 basis

The raw helper surfaces are:

- `trellis.models.black.black76_call`
- `trellis.models.black.black76_put`

These are **undiscounted** Black76 values:

$$C_{u}(F,K,\sigma,T) = F \Phi(d_1) - K \Phi(d_2)$$

$$P_{u}(F,K,\sigma,T) = K \Phi(-d_2) - F \Phi(-d_1)$$

with

$$d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma^2 T}{\sigma \sqrt{T}}, \qquad d_2 = d_1 - \sigma \sqrt{T}.$$

For equity-style spot underliers the adapter must resolve:

$$F_0 = S_0 e^{(r-q)T}, \qquad D(0,T) = e^{-rT}$$

from the market state's spot, discount curve, and carry / dividend
surface, then assemble

$$PV = N \, D(0,T) \, C_u(F_0, K, \sigma, T)$$

or the put analogue.

This explicit discounting requirement is why the Phase 3 declaration
model must permit raw-kernel adapters rather than only helper wrappers.

### 2. Cash-or-nothing and asset-or-nothing digitals

The raw helper surfaces are:

- `black76_cash_or_nothing_call`
- `black76_cash_or_nothing_put`
- `black76_asset_or_nothing_call`
- `black76_asset_or_nothing_put`

These are also **undiscounted** basis claims.

Cash-or-nothing:

$$\mathbf{1}_{F_T > K} \mapsto \Phi(d_2), \qquad \mathbf{1}_{F_T < K} \mapsto \Phi(-d_2)$$

Asset-or-nothing:

$$F_T \mathbf{1}_{F_T > K} \mapsto F_0 \Phi(d_1), \qquad F_T \mathbf{1}_{F_T < K} \mapsto F_0 \Phi(-d_1).$$

The adapter must distinguish:

- `Mul(Constant(c), Indicator(...))` for cash-or-nothing scale `c`
- `Mul(Spot(u), Indicator(...))` for asset-or-nothing exposure

Phase 3 should not collapse both onto
`price_equity_digital_option_analytical`, because the current exact
helper only prices the cash-payoff case. Asset-or-nothing must bind to
the dedicated Black76 asset basis kernels.

### 3. European payer / receiver swaptions

The checked helper surface is
`trellis.models.rate_style_swaption.price_swaption_black76`, which
internally resolves `ResolvedSwaptionBlack76Inputs`.

The pricing identity used by the helper is:

$$PV = N \, A(T_{\mathrm{pay}}) \, B_u(F_{\mathrm{swap}}, K, \sigma, T_{\mathrm{exp}})$$

where:

- `N` is notional
- `A` is the swap annuity over the payment schedule
- `F_swap` is the forward par swap rate
- `B_u` is the undiscounted Black76 call or put

Concretely, `price_swaption_black76_raw(...)` returns:

$$N \cdot A \cdot \texttt{black76\_call}(F_{\mathrm{swap}}, K, \sigma, T)$$

for payer swaptions and the corresponding put for receiver swaptions.

The solver declaration therefore belongs on the helper surface, not on
`black76_call` directly. The helper already owns:

- schedule construction
- annuity calculation
- forward swap-rate resolution
- volatility resolution from `market_state.vol_surface`

Phase 3 must reuse that exact checked assembly.

### 4. Two-asset basket / spread analytics

The checked helper surface is
`trellis.models.basket_option.price_basket_option_analytical`.

Its current exact contract is narrower than the generic Contract IR
family:

- exactly two underliers
- analytical support for basket / spread styles only
- typed semantics resolved through
  `resolve_basket_option_inputs(...)`

So the first-wave declaration must not be "all `basket_payoff`."
It must be constrained to the two-asset analytical cohort already
supported by the helper.

This is an example of why Phase 3 needs explicit declaration
precedence/admissibility rather than only top-level family tags.

### 5. Equity variance swap

The checked helper surfaces are:

- `price_equity_variance_swap_analytical`
- `equity_variance_swap_outputs_analytical`

The current repo implementation is a bounded smile-based approximation,
not a full log-contract integral over a continuum of options. Its
resolved fair variance strike is:

$$K_{\mathrm{var,fair}} = \sigma_{\mathrm{ATM}}^2 \sqrt{1 + 3 T s^2}$$

where `s` is the current helper's linear smile-slope proxy:

$$s = S_0 \frac{\sigma_{\max} - \sigma_{\min}}{K_{\max} - K_{\min}}.$$

The helper then prices

$$PV = N \, D(0,T) \, \big(K_{\mathrm{var,fair}} - K_{\mathrm{var,strike}}\big).$$

Phase 3 must document and preserve **this** semantics when migrating
the current helper. It must not silently upgrade the meaning of
`VarianceObservable` to a different replication formula inside the
compiler.

### 6. Arithmetic Asians are explicit blockers

`ArithmeticMean(...)` is a valid Phase 2 IR node, but there is no
dedicated analytical arithmetic-Asian helper currently exposed in
`trellis/models/`.

So the Phase 3 plan must carry an explicit blocker:

- Asian IR family is representable and matchable
- Asian fresh-build structural dispatch is deferred until a checked
  solver declaration surface exists

That is not a design failure. It is disciplined scope control.

## Compiler Surface

### Candidate dataclasses

Phase 3 should introduce a dedicated compiler surface under
`trellis/agent/`, for example:

- `ContractIRSolverDeclaration`
- `BoundContractIRSolverCall`
- `ContractIRCompilerDecision`
- `ContractIRCompilerAmbiguityError`

The output should carry:

- selected declaration id
- primitive / helper refs
- adapter output payload
- provenance and validation bundle metadata
- shadow-mode comparison fields when a legacy route decision also exists

### Declaration placement

Solver declarations should live next to the checked callable they wrap,
not in one giant central registry file.

Reason:

- signature drift becomes local
- helper-specific normalization stays local
- reviewers can audit declaration + callable together

The central registry should collect declarations, not define their
adapter logic.

### Shadow mode first

Phase 3 should start in shadow mode:

1. current route selection still determines the actual fresh build
2. structural compiler runs alongside it for migrated families
3. the system records `(legacy decision, structural decision, parity
   result)`
4. divergence is surfaced as a regression, not silently ignored

Only after that shadow-mode parity is clean for a family does Phase 4
promote structural dispatch to primary.

## Ordered Sub-Ticket Queue

### P3.1 — Structural declaration substrate

**Objective.** Land the declaration and registry dataclasses plus
registration / overlap validation.

**Artifacts.**

- new `trellis/agent/contract_ir_solver_registry.py`
- declaration overlap checker
- tests for precedence, ambiguity, and registration order

### P3.2 — Structural compiler and bound-call output

**Objective.** Compile `(contract_ir, valuation_context, market_state)`
into a deterministic bound solver call.

**Artifacts.**

- `compile_contract_ir_solver(...)`
- ambiguity failure contract
- route-metadata masking tests

### P3.3 — Vanilla + digital basis declarations

**Objective.** Declare and validate the Black76 vanilla / digital basis
family.

**Artifacts.**

- declarations on the raw Black76 basis kernels
- adapters resolving forward, discounting, and volatility
- separate declarations for cash-or-nothing and asset-or-nothing

### P3.4 — Swaption / basket / variance helper declarations

**Objective.** Migrate the checked helper-backed structural families.

**Artifacts.**

- declaration on `price_swaption_black76`
- declaration on `price_basket_option_analytical`
- declaration on `price_equity_variance_swap_analytical`
- helper-specific adapter tests

### P3.5 — Shadow-mode integration in the fresh-build path

**Objective.** Run structural compilation alongside the current route
path and record the comparison.

**Artifacts.**

- shadow-mode hook in semantic / route selection path
- scorecard output comparing structural vs legacy decisions
- route-masked regression tests for migrated families

### P3.6 — Family-by-family parity harness

**Objective.** Prove valuation parity on benchmark and fixture surfaces.

**Artifacts.**

- parity harness script
- per-family benchmark cohorts
- explicit tolerance policy per family

### P3.7 — Asian blocker ticket and solver follow-on

**Objective.** File and track the missing solver surface rather than
pretending Asian migration is complete.

**Artifacts.**

- explicit blocker / follow-on issue
- mirrored note in the plan table
- no route retirement yet for Asian

### P3.8 — Docs and compiler governance

**Objective.** Document declaration rules, ambiguity policy, and shadow
mode.

**Artifacts.**

- updates to `docs/quant/contract_ir.rst`
- updates to `docs/developer/` compiler docs

## Acceptance Criteria

- The compiler can select a bound solver call from `contract_ir` for the
  first-wave families without reading `ProductIR.instrument`,
  `route_id`, or `route_family`.
- Overlapping equal-precedence declarations fail closed.
- Route-masked tests prove the same structural decision is recovered for
  migrated families.
- Shadow-mode parity is green on fixture and benchmark cohorts for the
  first-wave families.
- Asian remains explicitly deferred until a real solver surface exists;
  the plan does not blur that blocker.

## Validation

- unit tests for declaration registration, overlap, and ambiguity
- unit tests for adapter payloads on canonical Contract IR fixtures
- route-masked compiler tests
- dual-path parity tests on benchmark fixtures
- benchmark replay showing valuation parity within policy tolerances

Family-specific tolerance guidance:

- exact deterministic helper pairs should be effectively exact, up to
  floating-point noise
- Monte Carlo-backed future declarations need separate RNG / confidence
  interval policy and should not reuse the deterministic tolerance
  threshold

## Failure Modes To Watch

- **Pattern over-match.** A top-level `vanilla_payoff` tag is not enough
  to distinguish all structural cases safely. Use narrower declarations
  when the callable's true domain is smaller.
- **Adapter centralization.** If a single central adapter module starts
  learning every family's quirks, it becomes the new hidden registry.
- **Helper/domain mismatch.** Do not declare a helper against a broader
  IR family than the helper actually supports.
- **Silent metadata dependence.** If masking legacy metadata changes the
  selected solver, the compiler is not route-free yet.
- **Asian scope leakage.** "IR exists" must not be used as a shortcut to
  claim Asian migration is done.

## Relationship To Phase 4

Phase 3 is successful only if it makes Phase 4 boring.

That means Phase 3 must already produce:

- deterministic structural decisions
- explicit provenance
- parity evidence
- route-masked invariance

If any of those are missing, Phase 4 becomes speculative deletion
instead of governed retirement.

## Next Steps

1. Land this document as the dedicated Phase 3 draft.
2. Promote the first-wave family list from prose to filed Linear
   children under QUA-905.
3. Open the explicit Asian solver blocker before coding starts.
4. Start Phase 3 only after the current Phase 2 draft is accepted as the
   semantic contract for `ContractIR`.
