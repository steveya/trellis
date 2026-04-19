# Contract IR — Phase 2: Algebraic AST Foundation

## Status

Draft. Pre-queue design document. Not yet the live execution mirror for
Phase 2.

## Linked Linear

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-904 — Phase 2 umbrella (this plan expands)
- QUA-916 — Phase 1.5 (Done; ContractPattern AST + evaluator landed)
- QUA-903 — Phase 1 (Done; pattern-keyed registry)
- QUA-905 — Phase 3 (Backlog; consumes Phase 2's IR)
- QUA-906 — Phase 4 (Backlog; retires ProductIR.instrument after Phase 3)

## Companion Drafts

- `doc/plan/draft__external-prior-art-adoption-map.md`
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`
- `doc/plan/draft__contract-ir-phase-4-route-retirement.md`
- `doc/plan/draft__quoted-observable-contract-ir-foundation.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__contract-ir-normalization-and-rewrite-discipline.md`
- `doc/plan/draft__contract-ir-compiler-retiring-route-registry.md`

## Purpose

Specify Contract IR — a compositional algebraic AST for derivative
payoffs — in enough detail that a reviewer can audit semantics and an
implementer can turn each section into a dataclass or test without
re-litigating design questions.

Phase 2 is purely additive. Contract IR lands alongside the existing
flat `ProductIR`; no dispatch path changes. Phase 3 (QUA-905) consumes
this IR through `@solves_pattern` kernel declarations; Phase 4 (QUA-906)
eventually retires `ProductIR.instrument` as the dispatch key.

The document is the mathematical companion to QUA-904's umbrella body.
Linear sub-tickets P2.1 through P2.6 reference specific sections here.

## Framing

### Why now

Phase 1 (QUA-903) retired 16 instrument-keyed routes in favor of
pattern-keyed dispatch on `payoff_family` + `required_market_data`.
Phase 1.5 (QUA-916) introduced `ContractPattern` — a pattern AST that
matches against `ProductIR` (a flat record). These let Phase 1 collapse
the registry without a new IR, but they don't give kernels a structural
representation of what they price.

A kernel like `black76_call(F, K, σ, T)` solves the payoff shape
`max(F - K, 0)` at a terminal date under lognormal forward dynamics. A
kernel like `price_swaption_black76(swap_rate, K, …)` solves the same
payoff shape — `max(swap_rate - K, 0)` — but on a forward swap rate,
scaled by an annuity. The two kernels look distinct under instrument-name
dispatch (`vanilla_option` vs `swaption`) but identical under
payoff-shape dispatch.

Phase 2's Contract IR makes that payoff-shape explicit. Phase 3's
`@solves_pattern` decorator then matches kernels to structural payoff
templates, not to instrument strings. The route registry's remaining
job — picking which kernel serves which request — becomes pattern
matching on a tree rather than string lookup on a record.

### What stays additive

Phase 2 does NOT:

- Change dispatch. `ProductIR` remains the primary record; `ContractIR`
  is a new field on `SemanticImplementationBlueprint` alongside
  existing ones. No consumers read it yet.
- Extend coverage beyond four payoff families: European terminal
  linear, variance-settled payoff, digital (cash/asset-or-nothing),
  arithmetic Asian. Barrier, lookback, chooser, compound, cliquet,
  credit, rates, path-dependent exercise, quoted-observable spreads,
  and leg-based cashflow products — all explicit Phase 2 follow-on
  tickets.
- Delete or rename `ProductIR`. `ContractIR` and `ProductIR` coexist.
- Touch kernel signatures. Phase 3 annotates kernels; Phase 2 does
  not.

These guardrails come from the same rationale that killed prior
registry-retirement attempts (see
`doc/plan/draft__contract-ir-compiler-retiring-route-registry.md`
"Why prior attempts stalled"): boil-the-ocean scope, dual-track drift,
unclear per-instrument "done" signal. Phase 2 is scope-bounded
specifically to avoid those traps.

### Closure role of Phase 2

The semantic-contract program has three distinct closure questions:
representation, decomposition, and lowering. The cross-phase definition
lives in `doc/plan/draft__semantic-contract-closure-program.md`.

Phase 2 owns:

- representation closure for the bounded payoff-expression cohort in
  this document
- bounded decomposition closure from supported request surfaces into the
  same canonical `ContractIR`

Phase 2 does **not** own lowering closure. That is Phase 3 work.

This boundary matters. If a family cannot be represented honestly inside
the payoff-expression AST without opaque product nodes or route-local
blobs, the right action is not to contort the Phase 2 AST until the
helper happens to fit. The right action is to move that family onto the
quoted-observable or leg-based follow-on tracks.

## Mathematical Substrate

### Contract IR as an algebraic data type

A `ContractIR` is a finite tree whose root carries four projections:
payoff expression, exercise specification, observation schedule, and
underlying-state descriptor. Formally (with pseudo-Python for
type-annotation clarity):

```
ContractIR = 
    { payoff: PayoffExpr
    ; exercise: Exercise
    ; observation: Observation
    ; underlying: Underlying
    }
```

Each field is independent and must be well-formed on its own. The AST
is implemented as frozen dataclasses with `__post_init__` invariant
checks.

Top-level composite contracts are intentionally OUT of Phase 2. If a
multi-leg structure can be represented under one shared exercise,
observation, and underlying surface, encode it inside `PayoffExpr`
using `Add`, `Mul`, `Scaled`, or `LinearBasket`. Heterogeneous
multi-leg structures that need multiple root-level surfaces are a
follow-on after Phase 2.

#### Sub-types

```
PayoffExpr = 
    | Constant(value: float)
    | Spot(underlier_id: str)
    | Forward(underlier_id: str, schedule: Schedule)  # forward price over a concrete delivery / accrual schedule
    | SwapRate(underlier_id: str, schedule: FiniteSchedule) # par swap rate for a concrete underlying swap schedule
    | Annuity(underlier_id: str, schedule: FiniteSchedule)  # positive swap annuity for the same schedule
    | Strike(value: float)
    | LinearBasket(terms: tuple[tuple[float, PayoffExpr], ...])  # k >= 1
    | ArithmeticMean(expr: PayoffExpr, schedule: FiniteSchedule)
    | VarianceObservable(underlier_id: str, interval: ContinuousInterval)
    | Max(args: tuple[PayoffExpr, ...])       # k-ary, k >= 1
    | Min(args: tuple[PayoffExpr, ...])       # k-ary, k >= 1
    | Add(args: tuple[PayoffExpr, ...])       # k-ary, k >= 2
    | Sub(lhs: PayoffExpr, rhs: PayoffExpr)   # binary
    | Mul(args: tuple[PayoffExpr, ...])       # k-ary, k >= 2
    | Scaled(scalar: PayoffExpr, body: PayoffExpr)  # body × scalar (e.g. annuity-scaled)
    | Indicator(predicate: Predicate)
```

```
Predicate = 
    | Gt(lhs: PayoffExpr, rhs: PayoffExpr)
    | Ge(lhs: PayoffExpr, rhs: PayoffExpr)
    | Lt(lhs: PayoffExpr, rhs: PayoffExpr)
    | Le(lhs: PayoffExpr, rhs: PayoffExpr)
    | And(args: tuple[Predicate, ...])
    | Or(args: tuple[Predicate, ...])
    | Not(arg: Predicate)
```

```
Exercise = 
    { style: ExerciseStyle   # european | bermudan | american (bermudan/american out of Phase 2 scope but typed-admissible)
    ; schedule: Schedule     # exercise dates
    }
```

```
Observation = 
    { kind: ObservationKind  # terminal | schedule | path_dependent
    ; schedule: Schedule
    }
```

```
Underlying = 
    { spec: UnderlyingSpec   # named processes + dynamics
    }
```

```
UnderlyingSpec = 
    | EquitySpot(name: str, dynamics: str)        # e.g. name="AAPL", dynamics="gbm"
    | RateCurve(name: str, dynamics: str)         # e.g. Hull-White 1F
    | ForwardRate(name: str, dynamics: str)       # lognormal forward
    | CompositeUnderlying(parts: tuple[UnderlyingSpec, ...])   # for baskets; leaf names must be unique
```

```
Schedule = 
    | Singleton(t: Date)
    | FiniteSchedule(dates: tuple[Date, ...])
    | ContinuousInterval(t_start: Date, t_end: Date)
```

#### Naming Discipline

Contract node names should describe contractual semantics or observed
market quantities, not a pricing method, numerical scheme, or lowering
strategy.

Permanent documentation for this naming rule lives in
`docs/quant/contract_algebra.rst` under "Observable Naming Discipline".

Naming audit for the current Phase 2 surface:

- `VarianceReplication` was renamed to `VarianceObservable` because
  option replication is one admissible lowering strategy, not part of
  the contract-level meaning.
- The remaining `PayoffExpr` names (`Spot`, `Forward`, `SwapRate`,
  `Annuity`, `ArithmeticMean`, `LinearBasket`, `Strike`) are retained
  because they denote semantic observables or algebraic constructors,
  not pricing methods.
- No other current `PayoffExpr`, `Predicate`, `Schedule`, or candidate
  leg-track node names in this draft were found to be method-leaky.

#### Constructor budget discipline

The Phase 2 AST should stay deliberately small. The payoff-expression
track is closer in spirit to the small combinator traditions behind
Peyton Jones' contract algebra than to a product-name catalog.

Practical admission rule for new Phase 2 constructors:

- add a new node when it denotes a semantic observable or algebraic
  constructor that cannot be expressed honestly from the existing node
  set
- do not add a node only because a desk or route currently has a named
  product bucket
- do not add a node whose primary meaning is a pricing method, helper
  convention, or route-local decomposition shortcut

This is why the current surface prefers:

- `VarianceObservable` over a method-leaky replication node
- `SwapRate` and `Annuity` as semantic observables
- algebraic composition (`Add`, `Scaled`, `Max`, `LinearBasket`) over
  instrument-named payoff nodes

If a proposed new node fails this constructor-budget test, it probably
belongs in:

- a later quoted-observable extension
- a leg-based extension
- or a lowering adapter, not the Phase 2 AST

#### Well-formedness

A `ContractIR c` is well-formed when all of the following hold:

1. The leaf `name` values inside `c.underlying.spec` are unique. They
   define the admissible namespace for all payoff-level `underlier_id`
   references.
2. Every `underlier_id` appearing in `c.payoff` (via `Spot`, `Forward`,
   `SwapRate`, `Annuity`, `LinearBasket` leaves, etc.) resolves to a
   named entry in `c.underlying.spec`.
3. Every schedule embedded in `c.payoff` (via `Forward`, `SwapRate`,
   `Annuity`, `ArithmeticMean`, `VarianceObservable`) is a concrete
   `Schedule` value. Phase 2 has no symbolic `schedule_ref` lookup.
   When a payoff schedule is intended to line up with exercise or
   observation, that alignment is by structural schedule equality, not
   by name indirection.
4. `FiniteSchedule` dates are strictly increasing, duplicate-free, and
   non-empty;
   `ContinuousInterval` requires `t_start <= t_end`.
5. Node-local schedule discipline is respected:
   `SwapRate` / `Annuity` use non-empty `FiniteSchedule`s;
   `ArithmeticMean` uses a non-empty `FiniteSchedule`;
   `VarianceObservable` uses a `ContinuousInterval`.
6. `c.exercise.style = european` iff `c.exercise.schedule` is a
   `Singleton`. For future typed-admissible surfaces, `bermudan`
   requires a non-empty `FiniteSchedule` and `american` requires a
   `ContinuousInterval`.
7. `c.observation.kind = terminal` iff `c.observation.schedule` is a
   `Singleton`.
   `Observation(kind=schedule, …)` requires a non-empty
   `FiniteSchedule`. `Observation(kind=path_dependent, …)` is reserved
   for follow-ons and may carry a `FiniteSchedule` or
   `ContinuousInterval`.
8. `PayoffExpr` subtree arities match their constructor declarations
   (e.g. `Max` has `k ≥ 1` args; `Sub` has exactly 2).
9. In `LinearBasket`, the tuple of terms is non-empty, the sum of
   weights may be any real (no normalization required), and each
   `PayoffExpr` in the tuple must well-form against the same underlying
   spec.

Well-formedness is enforced in the constructor (`__post_init__`) for
each dataclass and at the top-level `ContractIR` dataclass.

### PayoffExpr algebra

#### Denotational semantics

Let `(Ω, F, F_t, P)` be a filtered probability space and let
`M|_D = (M_t)_{t \in D}` be the market path restricted to the
observation domain `D` induced by the expression. Here `D` may be a
singleton date, a finite schedule, or a continuous interval. Each
`PayoffExpr e` denotes a measurable function:

$$\llbracket e \rrbracket : \text{PathState}_{D} \to \mathbb{R}$$

where `PathState_D` is a time-indexed market-path restriction on `D`.
For terminal leaves, `D = {t}` and we abbreviate the input by `m_t`.
For schedule-dependent leaves, the input is the restricted path on the
relevant schedule or interval.

Denotation rules (omitting the trivial `D = {t}` case for brevity):

$$\llbracket \texttt{Constant}(c) \rrbracket(m) = c$$

$$\llbracket \texttt{Spot}(u) \rrbracket(m_t) = m_t.\text{spot}(u)$$

$$\llbracket \texttt{Forward}(u, s) \rrbracket(m_t) = m_t.\text{forward}(u, s)$$

$$\llbracket \texttt{SwapRate}(u, s) \rrbracket(m_t) = m_t.\text{swap\_rate}(u, s)$$

$$\llbracket \texttt{Annuity}(u, s) \rrbracket(m_t) = m_t.\text{annuity}(u, s)$$

$$\llbracket \texttt{Strike}(K) \rrbracket(m) = K$$

$$\llbracket \texttt{LinearBasket}([(w_i, e_i)]) \rrbracket(m) = \sum_i w_i \cdot \llbracket e_i \rrbracket(m)$$

For `s = (t_1, \ldots, t_n)` with `n \ge 1`:

$$\llbracket \texttt{ArithmeticMean}(e, s) \rrbracket(m_s) = \frac{1}{n} \sum_{i=1}^{n} \llbracket e \rrbracket(m_{t_i})$$

$$\llbracket \texttt{VarianceObservable}(u, I) \rrbracket(m_I) = \operatorname{VarObs}(u, I; m_I)$$

`VarObs(u, I; m_I)` is the annualized variance observable attached to
underlier `u` and interval `I`. At the Contract IR layer this remains an
abstract scalar observable; the AST does NOT hard-code a particular
pricing identity. Analytical lowerings may later price its expectation
through static log-contract replication / Black vol-surface inputs,
matching the current Trellis / FinancePy-facing variance-swap route.

$$\llbracket \texttt{Max}(e_1, \ldots, e_k) \rrbracket(m) = \max_i \llbracket e_i \rrbracket(m)$$

$$\llbracket \texttt{Min}(\ldots) \rrbracket = \min_i \ldots$$

$$\llbracket \texttt{Add}(e_1, \ldots, e_k) \rrbracket(m) = \sum_i \llbracket e_i \rrbracket(m)$$

$$\llbracket \texttt{Sub}(l, r) \rrbracket(m) = \llbracket l \rrbracket(m) - \llbracket r \rrbracket(m)$$

$$\llbracket \texttt{Mul}(e_1, \ldots, e_k) \rrbracket(m) = \prod_i \llbracket e_i \rrbracket(m)$$

$$\llbracket \texttt{Scaled}(s, b) \rrbracket(m) = \llbracket s \rrbracket(m) \cdot \llbracket b \rrbracket(m)$$

$$\llbracket \texttt{Indicator}(p) \rrbracket(m) = \mathbb{1}_{p(m)}$$

Predicate denotation is standard:
$\llbracket \texttt{Gt}(l, r) \rrbracket(m) = (\llbracket l \rrbracket(m) > \llbracket r \rrbracket(m))$
and so on for the other comparators.

#### Equivalence relation

Two `PayoffExpr` `e` and `e'` are semantically equivalent (written
`e ≡ e'`) iff:

$$\forall m \in \text{PathState}_{D} \; : \; \llbracket e \rrbracket(m) = \llbracket e' \rrbracket(m)$$

This is the rewriting system's target invariant. Every simplification
rewrite must preserve `≡`.

### Simplification rewrites

The rewriter is a confluent rewrite system `→` whose canonical forms
are used for pattern matching in Phase 3. Every rule is
semantics-preserving; `e → e'` implies `e ≡ e'`.

Rules are listed in groups. Where needed, side conditions are explicit.

#### Algebraic (commutative monoid laws)

The sort / flatten / singleton rules below apply to each commutative
k-ary constructor (`Max`, `Min`, `Add`, `Mul`) unless stated otherwise.
Idempotence is specific to `Max` and `Min`; it does NOT apply to `Add`
or `Mul`.

```
Max(args_1, ..., args_k) → Max(sort(args_1, ..., args_k))
```
Sorting by a deterministic total order gives a canonical operand form.

```
Max(Max(a, b), c) → Max(a, b, c)
```
Flattening.

```
Max(a) → a
Min(a) → a
Add(a) → a        # as sum of one term
Mul(a) → a        # as product of one term
```
Singletons degenerate.

```
Max(a, a) → a
Min(a, a) → a
```
Idempotence.

#### Identity and absorbing elements

```
Add(a, Constant(0)) → a
Sub(a, Constant(0)) → a
Sub(a, a) → Constant(0)
Mul(a, Constant(1)) → a
Mul(a, Constant(0)) → Constant(0)
Scaled(Constant(1), a) → a
Scaled(Constant(0), a) → Constant(0)
Scaled(Constant(c_1), Scaled(Constant(c_2), a)) → Scaled(Constant(c_1 * c_2), a)
Scaled(Constant(c), Constant(d)) → Constant(c * d)
```

#### Distribution

```
Scaled(c, Add(a_1, ..., a_k)) → Add(Scaled(c, a_1), ..., Scaled(c, a_k))
```

```
Scaled(c, Max(a, b)) → Max(Scaled(c, a), Scaled(c, b))     [side cond: c ≥ 0]
Scaled(c, Min(a, b)) → Min(Scaled(c, a), Scaled(c, b))     [side cond: c ≥ 0]
```

For `c < 0` (technically `c` is a `PayoffExpr` that denotes a
non-positive scalar) the distribution swaps: `Scaled(c, Max(a, b)) =
Min(Scaled(c, a), Scaled(c, b))`. This is deliberately omitted from
auto-rewrites; callers canonicalize the sign manually.

The side condition `c ≥ 0` is syntactic, not symbolic. P2.2 only fires
the distribution rules when non-negativity is certified from node form
without solving inequalities globally, e.g. `Constant(c)` with `c ≥ 0`
or `Annuity(...)`. If the scalar's sign is not locally provable, leave
`Scaled(c, …)` in place.

```
Add(args) with Constant terms → fuse constants
```
E.g. `Add(a, Constant(2), Constant(3)) → Add(a, Constant(5))`.

```
Mul(args) with Constant factors → fuse constants
```
E.g. `Mul(a, Constant(2), Constant(3)) → Mul(a, Constant(6))`.

#### Shape canonicalization

The canonical ramp forms are:

```
Max(Sub(X, Strike(K)), Constant(0))   # call orientation
Max(Sub(Strike(K), X), Constant(0))   # put orientation
```

When a rewrite produces `Max(Constant(0), Sub(a, b))`, the sort rule
above puts the `Sub` term first (under a lexicographic `Max` ordering
that ranks `Sub` before `Constant`). This makes every ramp match one of
the two canonical templates above.

Negative outer scaling is NOT an orientation rewrite. For example,
`Scaled(Constant(-1), Max(Sub(X, Strike(K)), Constant(0)))` remains a
short-call expression; it does not canonicalize to the put template.

#### LinearBasket normalization

```
LinearBasket([(w, e)]) → Scaled(Constant(w), e)               [single term]
LinearBasket([(0, e_1), ..., (0, e_n)]) → Constant(0)         [all-zero basket]
LinearBasket([(0, e), ...]) → LinearBasket([...])             [drop zero-weight terms if terms remain]
LinearBasket([(w_1, e), (w_2, e), ...]) is not simplified     [no automatic fusion of duplicate expressions]
```

#### Confluence requirement

The rewrite system must be confluent: for any `e`, any maximal
reduction sequence ends at the same canonical form `e*`. This is
what P2.2's property-based tests will verify using Hypothesis-style
fuzzing:

- Generate random PayoffExpr trees.
- Apply the rewriter.
- Apply the rewriter again to the result; assert fixed-point.
- Apply the rewriter starting from different initial orderings; assert
  identical canonical form.

Confluence is non-trivial when distribution rules interact with
ordering rules. The canonical-form specification above is designed to
be confluent; if the property tests surface counter-examples, the
spec gets adjusted (not the tests weakened).

#### Strategy discipline

The rewrite layer should follow an explicit strategy contract rather
than a pile of ad hoc recursive simplifiers. The cross-cutting guidance
lives in `doc/plan/draft__contract-ir-normalization-and-rewrite-discipline.md`
and is inspired by the useful parts of SymPy's strategy vocabulary.

For Phase 2 the intended shape is:

1. local node rewrites with explicit side conditions
2. deliberate traversal order, typically bottom-up for algebraic
   simplification
3. first-success or chained rule groups where order matters
4. exhaustive fixed-point normalization until no rule changes the tree

That strategy boundary matters because Phase 3 declaration matching will
rely on canonical forms remaining stable across refactors.

## The Four Phase 2 Payoff Families

Each family is an equivalence class under semantic equivalence `≡`.
Every product in the family has an IR that canonicalizes to the same
template (up to underlying identity and schedule details).

### Family 1 — European terminal linear payoff

The family of payoffs whose form at a single terminal time is a
non-negative scalar times a single ramp:

$$A \cdot \max(L - R, 0)$$

where `A` is a non-negative scalar payoff expression (`Constant(1)` for
plain calls / puts, `Annuity(...)` for swaptions), and `L` / `R` are
linear functionals of market state. Call / put orientation is encoded in
the ordering of `Sub`:

- call: `L = X`, `R = K`
- put: `L = K`, `R = X`

Canonical IR template:

```
ContractIR(
    payoff = Scaled(weight_expr, Max(Sub(lhs, rhs), Constant(0))),
    exercise = Exercise(style=european, schedule=Singleton(T)),
    observation = Observation(kind=terminal, schedule=Singleton(T)),
    underlying = Underlying(spec = …)
)
```

For unscaled calls / puts, `weight_expr = Constant(1)` and the identity
rule collapses `Scaled(Constant(1), …)` away. For scaled families such as
swaptions, the positive outer factor stays explicit.

Important distinction: a put is NOT a negatively scaled call. The
canonical put form is `Max(Sub(Strike(K), X), Constant(0))`. By
contrast, `Scaled(Constant(-1), Max(Sub(X, Strike(K)), Constant(0)))`
denotes a short call, not a put, and the rewriter must preserve that
distinction.

#### Members

**European equity call on spot.**

$$\text{payoff}_T = \max(S_T - K, 0)$$

Contract IR:
```
payoff = Max(Sub(Spot("AAPL"), Strike(150)), Constant(0))
underlying = Underlying(spec=EquitySpot("AAPL", "gbm"))
```

**European payer swaption on forward swap rate.**

$$\text{payoff}_T = A(T) \cdot \max(s_T - K, 0)$$

where `A(T)` is the annuity of the forward swap at expiry and `s_T` is
the par swap rate. Contract IR:

```
payoff = Scaled(Annuity("USD-IRS-5Y", FiniteSchedule((...,))),
                Max(Sub(SwapRate("USD-IRS-5Y", FiniteSchedule((...,))), Strike(0.05)),
                    Constant(0)))
underlying = Underlying(spec=ForwardRate("USD-IRS-5Y", "lognormal_forward"))
```

`Annuity` is a dedicated `PayoffExpr` constructor in Phase 2. It denotes
the positive swap annuity associated with the same underlying swap
schedule used by `SwapRate`.

**European basket option.**

$$\text{payoff}_T = \max\Big(\sum_i w_i S_T^{(i)} - K,\ 0\Big)$$

Contract IR:
```
payoff = Max(
    Sub(LinearBasket([(0.5, Spot("SPX")), (0.5, Spot("NDX"))]),
        Strike(4500)),
    Constant(0)
)
underlying = Underlying(spec=CompositeUnderlying((EquitySpot("SPX", "gbm"),
                                                  EquitySpot("NDX", "gbm"))))
```

#### Unification observation

All three products share the same positive-weighted single-ramp core:
an optional non-negative outer scale multiplying
`Max(Sub(_, Strike(_)), Constant(0))`. They differ only in the linear
observable `X`, the optional positive weight, and the `Underlying`
specification. Under Phase 3's `@solves_pattern`, three
kernels (`black76_call`, `price_swaption_black76`,
`price_basket_option_analytical`) declare the same outer pattern and
differentiate via their `Underlying` constraint. This is the
operational payoff of unifying `vanilla_option`, `swaption`, and
`basket_option` into one Contract IR family.

### Family 2 — Variance-settled payoff

Payoffs whose terminal value is a linear function of a variance
observable over an interval:

$$\text{payoff}_T = N \cdot (V^{\mathrm{obs}}_{[t_0, T]} - K_{\text{var}})$$

where `V^{\mathrm{obs}}_{[t_0,T]}` is the annualized variance observable
named by the contract and `K_var` is the variance strike. For
realized-variance-settled contracts, pricing later identifies the fair
value of that observable under the chosen model / quote convention.
Phase 2 does not encode the discrete sampling, annualization, or
pricing method inside the AST.

Canonical IR template:

```
ContractIR(
    payoff = Scaled(Constant(notional),
                    Sub(VarianceObservable(u, ContinuousInterval(t_0, T)),
                        Strike(K_var))),
    exercise = Exercise(style=european, schedule=Singleton(T)),
    observation = Observation(kind=terminal, schedule=Singleton(T)),
    underlying = Underlying(spec=EquitySpot(u, dynamics))
)
```

Implementation note: `VarianceObservable(u, I)` denotes the variance
observable that the lowering will price. One admissible lowering is
static option replication. The current Trellis helper
`price_equity_variance_swap_analytical` implements this using the Black
vol surface and a FinancePy-compatible fair-strike approximation. That
pricing identity lives below the IR boundary; the Contract IR node is
kept at the observable level so later phases can swap lowerings without
changing the AST contract. Phase 3's `@solves_pattern` on that helper
will match this template.

#### Members

**Variance swap** (sole Phase 2 member). Future members (variance
dispersion, vol-of-vol) extend this family by varying `underlying.spec`
to a basket / vol-process spec — not Phase 2 scope.

### Family 3 — Digital (cash-or-nothing / asset-or-nothing)

Indicator-weighted payoffs at a single terminal time:

$$\text{payoff}_T = Q \cdot \mathbb{1}_{X > K}$$

where `Q ∈ {\text{constant amount},\ X\ \text{itself}}`.

Canonical IR template:

```
ContractIR(
    payoff = Mul(Q_expr,                                 # Q_expr: Constant for cash-or-nothing; Spot for asset-or-nothing
                 Indicator(Gt(X, Strike(K)))),
    exercise = Exercise(style=european, schedule=Singleton(T)),
    observation = Observation(kind=terminal, schedule=Singleton(T)),
    underlying = Underlying(spec = …)
)
```

#### Members

**Cash-or-nothing digital call**: `Q_expr = Constant(c)`,
`X = Spot(u)`.

$$\text{payoff}_T = c \cdot \mathbb{1}_{S_T > K}$$

**Asset-or-nothing digital call**: `Q_expr = Spot(u)`,
`X = Spot(u)`.

$$\text{payoff}_T = S_T \cdot \mathbb{1}_{S_T > K}$$

Put variants swap the predicate to `Lt`. Positive cash-or-nothing and
asset-or-nothing puts remain positive payouts; a negative outer scale
still denotes a short position rather than put orientation. The
rewriter normalizes put digitals to a canonical `Lt` form; details in
P2.2. Phase 2 treats strict comparators (`Gt` / `Lt`) as canonical.
Inclusive boundaries (`Ge` / `Le`) remain available in the AST but are
not canonicalized into or out of the strict digital family.

#### Non-Phase-2 future members

Rate digitals (on `SwapRate`), FX digitals, basket digitals — all
expressible as Family 3 with different `X` and `Underlying`. Not Phase 2
scope; explicit Phase 2 follow-on work.

### Family 4 — Arithmetic Asian

Payoffs whose observable is the arithmetic mean of an underlying over
an averaging schedule:

$$\text{payoff}_T = \max\Big(\bar{X}_{\text{avg}} - K,\ 0\Big) \quad\text{where}\quad \bar{X}_{\text{avg}} = \frac{1}{|s|} \sum_{t \in s} X_t$$

Canonical IR template:

```
ContractIR(
    payoff = Max(Sub(ArithmeticMean(X_expr, s_avg),
                     Strike(K)),
                 Constant(0)),
    exercise = Exercise(style=european, schedule=Singleton(T)),
    observation = Observation(kind=schedule, schedule=s_avg),
    underlying = Underlying(spec = …)
)
```

#### Members

**Arithmetic Asian call on equity spot**: `X_expr = Spot(u)`,
`s_avg = FiniteSchedule(...)`.

#### Non-Phase-2 future members

Geometric Asian (different aggregation operator), floating-strike
Asian (`Max(Sub(X_T, ArithmeticMean(Spot, s)), 0)`), Asian swaption,
multi-asset Asian baskets — all structurally related but outside
Phase 2 scope.

## Follow-On Boundary: Quoted Observables vs Leg-Based Products

Phase 2 keeps `VarianceObservable` as a dedicated observable leaf. It
should NOT be widened into a generic "market surface query" node. That
would blur two different future tracks that need different semantics.

### Track A — Quoted-observable products

These are contracts whose payoff is a function of one or more quoted
market observables at one observation surface. The contract can still
fit naturally inside the current `PayoffExpr` algebra if later phases
add explicit quote-observable leaves with quote conventions carried in
the node.

Representative examples:

- a vol-skew product settling on `σ(T, K_1) - σ(T, K_2)` or another
  explicit function of two implied-vol surface points
- a terminal curve-spread product settling on `S_{10Y}(T) - S_{2Y}(T)`
  where both coordinates are quoted par rates observed at one time

The key property is snapshot semantics: the payoff depends on quoted
points, not on a schedule of coupon accrual and payment obligations.

### Track B — Leg-based cashflow products

These are contracts whose economic meaning is a schedule of dated
contingent cashflows assembled from legs, accrual rules, fixing rules,
payment rules, and notional exchanges. They do not fit cleanly into the
Phase 2 payoff-only AST without either hiding cashflow logic inside
opaque leaf nodes or rebuilding a mini cashflow engine inside
`PayoffExpr`.

Representative examples:

- SOFR-FF basis swaps
- vanilla fixed-float and float-float swaps
- coupon-bearing notes and bonds
- callable / putable coupon products once exercise is reintroduced

The key property is leg semantics: the product is defined by contractual
cashflow assembly, not by one terminal quoted-observable formula.

### Boundary rule

Classify by contract semantics, not by desk label. A trade described as
"10Y-2Y basis" may land in either track:

- if it settles once on a terminal quote spread, it is a
  quoted-observable product
- if it exchanges scheduled coupons or floating legs, it is a
  leg-based cashflow product

The leg-based future track is recorded in
`doc/plan/draft__leg-based-contract-ir-foundation.md`.
The quoted-observable future track is recorded in
`doc/plan/draft__quoted-observable-contract-ir-foundation.md`.

## Relationship to ProductIR

`ContractIR` is additive: `SemanticImplementationBlueprint` gains a
new field `contract_ir: ContractIR | None`. No existing code reads
it. Phase 3 is where consumption happens.

There is a forward projection `π : ContractIR → ProductIR` that
extracts the flat string tags from a Contract IR:

```python
def project_to_product_ir(c: ContractIR) -> ProductIR:
    return ProductIR(
        instrument = _derive_instrument_string(c),        # legacy string tag
        payoff_family = _derive_payoff_family(c),         # e.g. "vanilla_option"
        payoff_traits = _derive_payoff_traits(c),         # e.g. ("call",)
        exercise_style = c.exercise.style,
        model_family = _derive_model_family(c.underlying.spec),
        …
    )
```

`π` is lossy (the `ContractIR` tree is richer than the flat record).
Phase 2 does not implement `π`; `ProductIR` continues to be produced
by the existing decomposer. Phase 4 considers using `π` as a
compatibility shim while retiring `ProductIR.instrument`.

## Sub-Ticket Specifications

### P2.1 — `contract_ir.py`: AST types and frozen dataclasses

**Objective.** Land `trellis/agent/contract_ir.py` implementing
`ContractIR`, `PayoffExpr`, `Predicate`, `Exercise`, `Observation`,
`Underlying`, `UnderlyingSpec`, and `Schedule` as frozen dataclasses
with `__post_init__` well-formedness checks.

**Scope.**

- New module `trellis/agent/contract_ir.py` with the ADT types above.
- Top-level `ContractIR.Composite` is explicitly OUT of Phase 2; any
  multi-leg example in this phase must fit inside one root contract via
  `PayoffExpr` composition.
- No consumer-side changes. No evaluator extension (that's P2.5).
- Tests in `tests/test_agent/test_contract_ir_types.py`: construction,
  well-formedness rejection (each invariant independently), structural
  equality (dataclass `__eq__` + tuple ordering for k-ary constructors).

**Files.**

- `trellis/agent/contract_ir.py` (new, ~400–600 lines)
- `tests/test_agent/test_contract_ir_types.py` (new, ~200–300 lines)

**Acceptance criteria.**

- Every constructor listed in the "Sub-types" block above exists as a
  frozen dataclass with matching field names and types.
- `__post_init__` raises `ContractIRWellFormednessError` (new exception
  class) when any of well-formedness rules 1–9 is violated. One
  targeted test per rule.
- Dataclass equality: structurally equal IRs compare equal; the test
  suite includes a fixture for each of the four Phase 2 families'
  canonical IRs.
- No evaluator, no pattern-matcher, no decomposer extension.
- Full agent suite green (purely additive).

**Validation.**

- `pytest tests/test_agent/test_contract_ir_types.py -q` — all green.
- `pytest tests/test_agent -q` — no regressions.

**Dependencies.** None.

**Forward-compat.** Reuse the existing ContractPattern head-tag names
where the vocabularies already overlap (`"max"`, `"sub"`, `"scaled"`,
`"indicator"`, `"spot"`, `"strike"`, `"constant"`). New IR-only heads
introduced in Phase 2 (`"forward"`, `"swap_rate"`, `"annuity"`,
`"linear_basket"`, `"arithmetic_mean"`, `"variance_observable"`) are
part of the Contract IR surface immediately, but ContractPattern parser
support for them lands in P2.5. P2.5's evaluator extension relies on
that eventual head-tag correspondence to match patterns against IRs.

---

### P2.2 — Simplification rewrites and property-based tests

**Objective.** Implement the rewrite rules from "Simplification
rewrites" as a canonicalization function
`canonicalize(e: PayoffExpr) -> PayoffExpr` and verify
confluence via property-based tests.

**Scope.**

- Function `canonicalize(expr)` in `trellis/agent/contract_ir.py` (or a
  sibling module `contract_ir_simplify.py`).
- Implementations for each rule listed in "Simplification rewrites"
  above: commutative-monoid normalization (sort + flatten + singleton
  + idempotence), identity/absorbing, distribution (with side
  conditions), shape canonicalization, LinearBasket normalization.
- Add an explicit anti-regression fixture for the semantic distinction
  between a put and a short call:
  `Max(Sub(Strike(K), X), Constant(0))` must never canonicalize to
  `Scaled(Constant(-1), Max(Sub(X, Strike(K)), Constant(0)))`, and vice
  versa.
- Property-based tests using Hypothesis:
  - **Idempotence**: `canonicalize(canonicalize(e)) = canonicalize(e)`
    for any randomly generated `e`.
  - **Confluence**: for any two equivalent-by-construction trees
    (e.g. `Max(a, b)` and `Max(b, a)`), their canonicalizations are
    identical.
  - **Semantic preservation**: evaluate both `e` and
    `canonicalize(e)` against a synthetic leaf-valuation environment
    and assert numerical equality within `1e-12`. (Requires a simple
    evaluator — not the full pattern evaluator, just a direct
    `PayoffExpr → float` interpreter under fixed numeric assignments for
    leaves such as `Spot`, `SwapRate`, `Annuity`, and
    `VarianceObservable`.)
  - **Fixture tests**: each of the four family templates canonicalizes
    to the shape listed above.

**Files.**

- `trellis/agent/contract_ir.py` or `trellis/agent/contract_ir_simplify.py`
- `tests/test_agent/test_contract_ir_simplify.py` (new, ~300–500 lines)

**Acceptance criteria.**

- `canonicalize` is a total function (no exceptions for well-formed
  input).
- Family-1 call / put orientation is preserved: operand order inside
  `Sub` carries the option side; negative outer scaling is preserved as
  short exposure and is not rewritten into a put.
- Hypothesis idempotence property passes on 1000+ randomly generated
  trees.
- Hypothesis confluence property passes on 500+ random equivalent
  pairs.
- Numerical semantic-preservation property passes within `1e-12` on
  500+ random (tree, environment) pairs.
- Canonical form for each of the four Phase 2 family templates matches
  the spec in the "Four Phase 2 Payoff Families" section above.

**Validation.**

- `pytest tests/test_agent/test_contract_ir_simplify.py -q` green.
- Full agent suite green.

**Dependencies.** `blockedBy` P2.1 (needs the AST types).

**Forward-compat.** Pattern matching in P2.5 and Phase 3 assumes
canonical form. If future rules extend the rewriter, property tests
re-run to confirm confluence still holds.

---

### P2.3 — Decomposer extension: natural language → Contract IR

**Objective.** Extend `trellis/agent/knowledge/decompose.py` so
`decompose_to_ir(description, instrument_type)` also emits a
`ContractIR | None` alongside the existing `ProductIR`. The return
type becomes a tuple or a structured record; existing callers keep
reading `ProductIR` unchanged.

**Scope.**

- New function `decompose_to_contract_ir(description, instrument_type, product_ir) -> ContractIR | None`.
- Returns `None` when the description is outside the four Phase 2
  families.
- Returns a well-formed `ContractIR` for 20+ canonical fixtures
  spanning the four families (see "Fixtures" below).
- Produces `ContractIR` from the semantic description and parsed product
  semantics only; it must not consult `route_id`, `route_family`,
  backend-binding ids, or other route-selected metadata.

**Fixtures (canonical descriptions → expected Contract IR).**

| Description | Family | Expected shape sketch |
|---|---|---|
| European call on AAPL strike 150 expiring 2025-11-15 | 1 | `Max(Sub(Spot("AAPL"), Strike(150.0)), Constant(0))` |
| European put on SPX strike 4500 expiring 2025-11-15 | 1 | `Max(Sub(Strike(4500.0), Spot("SPX")), Constant(0))` |
| European payer swaption on 5Y USD IRS strike 5% expiring 2025-11-15 | 1 | `Scaled(Annuity(…), Max(Sub(SwapRate(…), Strike(0.05)), Constant(0)))` |
| European receiver swaption on 5Y USD IRS strike 5% expiring 2025-11-15 | 1 | `Scaled(Annuity(…), Max(Sub(Strike(0.05), SwapRate(…)), Constant(0)))` |
| European basket call on {SPX 50%, NDX 50%} strike 4500 | 1 | `Max(Sub(LinearBasket([(0.5, Spot("SPX")), (0.5, Spot("NDX"))]), Strike(4500)), Constant(0))` |
| Equity variance swap on SPX, variance strike 0.04, notional 10000, expiry 2025-11-15 | 2 | `Scaled(Constant(10000), Sub(VarianceObservable("SPX", ContinuousInterval(t_0, T)), Strike(0.04)))` |
| Cash-or-nothing digital call on AAPL paying $1 if spot > 150 at expiry | 3 | `Mul(Constant(1), Indicator(Gt(Spot("AAPL"), Strike(150))))` |
| Asset-or-nothing digital put on AAPL if spot < 150 at expiry | 3 | `Mul(Spot("AAPL"), Indicator(Lt(Spot("AAPL"), Strike(150))))` |
| Arithmetic Asian call on SPX monthly average over 2025 strike 4500 | 4 | `Max(Sub(ArithmeticMean(Spot("SPX"), FiniteSchedule((…monthly dates…))), Strike(4500)), Constant(0))` |
| Arithmetic Asian put on SPX weekly average strike 4500 | 4 | analogous, put orientation |

Plus 10 more varying underlier, schedule, strike shape, and boundary
cases (zero strike, negative strike, extreme maturities) to catch
off-by-one and parsing edge cases.

**Out-of-family descriptions that must return `None`.**

- "American put on AAPL" (exercise style outside Phase 2 scope)
- "Barrier option with 200 knock-out" (barrier — Phase 2 follow-on)
- "Lookback option" (lookback — Phase 2 follow-on)
- "Chooser option" (chooser — Phase 2 follow-on)
- "Callable bond" (exercise-style + embedded coupons)
- "CDS on Ford" (credit — different kernel substrate)
- "Caplet" (single-period rate option — rate family, Phase 2 follow-on)

**Files.**

- `trellis/agent/knowledge/decompose.py` (extension, ~100–200 new lines)
- `tests/test_agent/test_decompose_contract_ir.py` (new, ~400–600 lines
  for the 30+ fixtures)

**Acceptance criteria.**

- Every in-family fixture above produces a well-formed `ContractIR`
  matching the expected shape (tested via dataclass equality).
- Every out-of-family fixture returns `None`.
- Every returned IR canonicalizes cleanly (P2.2's `canonicalize` leaves
  the decomposer's output unchanged).
- `decompose_to_ir` return type stays backward-compatible (existing
  callers reading `ProductIR` still work).
- The IR result is route-independent: masking or omitting legacy route
  metadata does not change the emitted `ContractIR`.
- Full agent suite green.

**Validation.**

- `pytest tests/test_agent/test_decompose_contract_ir.py -q` green.
- `pytest tests/test_agent -q` no regressions.

**Dependencies.** `blockedBy` P2.1, P2.2.

**Forward-compat.** Phase 3 `@solves_pattern` compiler reads
`contract_ir` from `SemanticImplementationBlueprint`. P2.3 is the
upstream of that field; P2.4 wires the field onto the blueprint.

---

### P2.4 — `SemanticImplementationBlueprint.contract_ir` wiring

**Objective.** Thread the Contract IR from the decomposer through the
compile-build-request pipeline so
`SemanticImplementationBlueprint.contract_ir: ContractIR | None` is
populated on every blueprint.

**Scope.**

- Add `contract_ir: ContractIR | None = None` to
  `SemanticImplementationBlueprint` (in
  `trellis/agent/semantic_contract_compiler.py`).
- Modify `compile_build_request` to call `decompose_to_contract_ir` and
  attach the result.
- Failure-mode handling: if decomposition raises, log a warning and
  attach `None`; do NOT fail the build (additive discipline).
- Failure-mode handling: if decomposition succeeds with `None` (out of
  Phase 2 family), attach `None` silently.
- Preserve route independence: `contract_ir` must be attached before any
  downstream route-specific codegen hinting, and its contents must not
  depend on which route was later selected.

**Files.**

- `trellis/agent/semantic_contract_compiler.py` (extension)
- `tests/test_agent/test_semantic_contract_compiler_contract_ir.py`
  (new, ~150 lines)

**Acceptance criteria.**

- `SemanticImplementationBlueprint` has a `contract_ir` field that
  defaults to `None`.
- For every in-family fixture from P2.3, building a blueprint produces
  a well-formed `contract_ir`.
- For every out-of-family description, `contract_ir` is `None`.
- Decomposer exceptions produce `None` + warning log; build does not
  fail.
- A regression test masks route metadata on the compiled request /
  blueprint path and confirms the same `contract_ir` still attaches for
  in-family fixtures.
- Full agent suite green.

**Validation.**

- `pytest tests/test_agent/test_semantic_contract_compiler_contract_ir.py -q`
  green.
- `pytest tests/test_agent -q` no regressions.

**Dependencies.** `blockedBy` P2.1, P2.3.

**Forward-compat.** Phase 3 reads `blueprint.contract_ir` in the
compiler. P2.4 is the blueprint-side entry point.

---

### P2.5 — Extend `ContractPattern` evaluator to match against `ContractIR`

**Objective.** Generalize Phase 1.5.B's
`evaluate_pattern(pattern: ContractPattern, target) -> MatchResult`
to accept `target: ContractIR` in addition to `target: ProductIR`.
Pattern types (P1.5.A) stay unchanged; the evaluator gains a second
target type.

**Scope.**

- Extend `trellis/agent/contract_pattern.py` parser / dumper allowlists
  with the IR-only payoff heads required by Phase 2 pattern matching:
  `forward`, `swap_rate`, `annuity`, `linear_basket`,
  `arithmetic_mean`, and `variance_observable`.
- Extend `trellis/agent/contract_pattern_eval.py::evaluate_pattern`
  to dispatch on `target` type.
- When `target` is a `ContractIR`:
  - `PayoffPattern` walks `target.payoff` (the `PayoffExpr` tree).
    Head-tag correspondence: `PayoffPattern(kind="max", args=[...])`
    matches `Max(args)` if the pattern args recursively match.
  - `ExercisePattern` matches `target.exercise.style` and
    `target.exercise.schedule`.
  - `UnderlyingPattern` matches `target.underlying.spec` (name,
    dynamics, kind).
  - `ObservationPattern` matches `target.observation.kind` and
    `target.observation.schedule`.
  - `Wildcard` matches any sub-tree; named wildcards bind the matched
    sub-tree.
  - `AndPattern` / `OrPattern` / `NotPattern` compose recursively.
- Round-trip test: the existing `ProductIR` parity matrix (Phase 1.5.B's
  `TestAnalyticalBlack76Parity::test_full_parity_matrix`) extends to
  assert the same patterns evaluate to the same `ok: bool` against the
  corresponding `ContractIR` fixtures.

**Files.**

- `trellis/agent/contract_pattern.py` (extension for parser vocabulary)
- `trellis/agent/contract_pattern_eval.py` (extension, ~200 new lines)
- `tests/test_agent/test_contract_pattern_eval_contract_ir.py` (new,
  ~400 lines)

**Acceptance criteria.**

- Structured ContractPattern parse / dump round-trips succeed for the
  newly admitted IR-only head tags above.
- `evaluate_pattern(pattern, target: ContractIR)` works for every
  pattern kind.
- For every (pattern, IR) pair derived from Phase 1.5.B's canonical
  `analytical_black76` patterns paired with the P2.3 in-family
  `ContractIR` fixtures: `evaluate_pattern(pattern, contract_ir).ok`
  equals `evaluate_pattern(pattern, product_ir).ok`.
- Named wildcards bind correctly against IR sub-trees (e.g. a pattern
  `Max(Sub(Spot(_u), Strike(_k)), Constant(0))` matching an IR
  populates bindings `{_u: "AAPL", _k: 150.0}`).
- ContractIR matching is self-sufficient: the evaluator reads only the
  IR tree and pattern, not legacy route ids / route families.
- Full agent suite green.

**Validation.**

- `pytest tests/test_agent/test_contract_pattern_eval_contract_ir.py -q`
  green.
- Full agent suite green.

**Dependencies.** `blockedBy` P2.1, P2.3.

**Forward-compat.** Phase 3's `@solves_pattern` decorator calls
`evaluate_pattern` with the kernel's pattern and the request's
`ContractIR`. P2.5 is the substrate Phase 3 consumes.

---

### P2.6 — `docs/quant/contract_ir.rst`

**Objective.** Document Contract IR for the quant-developer audience.

**Scope.**

- New file `docs/quant/contract_ir.rst` covering:
  1. Motivation: why Contract IR, position vs `ProductIR`.
  2. Formal ADT specification (the constructors).
  3. Denotational semantics.
  4. The four Phase 2 payoff families with worked examples.
  5. Simplification rewrites + confluence property.
  6. Decomposer contract (natural language → IR).
  7. Relationship to `ContractPattern` (Phase 1.5 patterns match
     against Contract IR in Phase 2).
  8. Forward roadmap: Phase 3 `@solves_pattern`, Phase 4 retirement of
     `ProductIR.instrument`.
- Integration: link from existing `docs/quant/contract_algebra.rst`
  index; add entry to `docs/quant/index.rst` if present.

**Files.**

- `docs/quant/contract_ir.rst` (new)
- `docs/quant/index.rst` (edit if needed)

**Acceptance criteria.**

- Document lands; cross-references from `contract_algebra.rst` work.
- All worked examples in the document correspond to real fixtures in
  `tests/test_agent/test_decompose_contract_ir.py`.
- Reviewer (quant lead) sign-off on mathematical correctness.

**Validation.**

- Document renders via Sphinx without warnings (if the docs build
  pipeline is active).
- Code examples cross-check against test fixtures.

**Dependencies.** `blockedBy` P2.1, P2.2 (so the documented rewrites
match shipped behavior).

**Forward-compat.** Phase 3 docs will extend this with
`@solves_pattern` examples; Phase 4 will add the retirement plan for
`ProductIR.instrument`.

## Cross-Cutting Principles

These apply to every Phase 2 sub-ticket. Relaxing any of them would
reopen the traps that killed prior registry-retirement attempts (see
`doc/plan/draft__contract-ir-compiler-retiring-route-registry.md`
"Why prior attempts stalled").

1. **Additive only.** Phase 2 adds `ContractIR`, `canonicalize`, the
   decomposer extension, the blueprint field, and the evaluator
   extension. Nothing existing is modified in a way that changes
   behavior. Every sub-ticket's acceptance criteria includes "full
   agent suite green".
2. **Scope bounded to four families.** Barrier, lookback, chooser,
   compound, cliquet, credit products, rate exotics, path-dependent
   exercise — all explicit Phase 2 follow-on tickets. Do not
   expand.
3. **Property-based tests for semantic invariants.** `canonicalize`'s
   idempotence + confluence + semantic-preservation properties are
   non-negotiable. Hypothesis (or equivalent) is the tool.
4. **Decomposer fixture-driven.** Do not free-form the natural-language
   parsing. Every in-family description has a hand-written fixture
   with its expected IR. If a new description shape is needed, add a
   fixture; don't widen the parser.
5. **Canonical-form correspondence.** The decomposer emits canonical
   IRs directly (or `canonicalize`-output-identical IRs). P2.5's
   pattern evaluator assumes canonical form.
6. **Route-free fresh build is the target surface.** Contract IR is not
   permitted to become a documentation-only sidecar. Every Phase 2
   artifact must be shaped so that Phase 3 can select kernels from
   `(contract_ir, pattern matches, lowering obligations)` without
   consulting `route_id`, `route_family`, backend-binding ids, or
   hard-coded per-instrument route switches. Legacy route data may
   coexist as observability-only metadata until Phase 4, but it is not
   allowed to be a required compiler input for fresh builds.
7. **Pattern-vocabulary parity before Phase 3.** If a Phase 2 family
   cannot be expressed in the ContractPattern surface, that is a Phase 2
   bug, not a Phase 3 TODO. P2.5 closes this gap by extending parser and
   evaluator vocabulary to cover every Phase 2 payoff head.
8. **Shadow-mode retirement proof.** Before QUA-904 closes, the plan
   must identify at least one Phase 3 shadow-mode harness that masks
   route hints and proves kernel selection can be reconstructed from
   `ContractIR` for the four Phase 2 families. Phase 4 then promotes
   that proof from shadow mode to the live fresh-build path.
9. **Representation before lowering.** A Phase 2 family is only
   considered ready for Phase 3 if it has both representation closure
   (an honest canonical IR shape) and bounded decomposition closure
   (fixture-driven route-independent emission of that shape). Lowering
   closure remains a Phase 3 obligation and must not be faked by
   backporting product-specific helper payloads into the Phase 2 IR.

## Locked Decisions For This Draft

The following ambiguities are resolved in this draft so P2.1 can be
filed without conflicting AST guidance:

1. `Annuity` is a dedicated `PayoffExpr` variant in Phase 2.
2. Payoff nodes embed concrete `Schedule` values directly; Phase 2 has
   no symbolic `schedule_ref` lookup environment.
3. `ContractIR.payoff` is a raw `PayoffExpr`; there is no extra
   `Payoff(...)` wrapper node.
4. Top-level `ContractIR.Composite` is out of Phase 2. Shared-surface
   multi-leg examples use `Add(...)` or other `PayoffExpr`
   composition instead.

## Open Questions

These are places where the author is least certain and outside judgment
would help most. Answer before P2.1 lands; the answer shapes the
dataclass design.

1. **Measure convention.** The denotational semantics above is agnostic
   about measure. Does `ContractIR` carry a measure annotation, or is
   it always "price under the numeraire implied by `Underlying`"? For
   Phase 2, "always under numeraire implied by Underlying" is fine
   (every family has an unambiguous pricing measure). Phase 3 may
   need to be more explicit when kernels have measure-specific
   assumptions (e.g. Black76 assumes forward martingale under
   `T`-forward measure).

2. **Schedule date representation.** `Schedule` uses `Date` — is that
   `datetime.date`, `numpy.datetime64`, a custom `Date` newtype? Phase 2
   should pick one and stick with it. Weak preference: `datetime.date`
   (standard library, matches existing `trellis.core.date_utils`).

3. **`UnderlyingSpec` dynamics strings.** `EquitySpot("AAPL", "gbm")` —
   where does the list of valid dynamics strings live? Options:
   (a) free-form strings validated against a registry;
   (b) `enum DynamicsKind`. Weak preference: (a) with a registry in
   `trellis/agent/knowledge/canonical/dynamics.yaml` (not in Phase 2
   scope to create; just use free-form strings for now).

4. **Predicate scope.** `Indicator(Gt(...))` is in scope. Do we also
   need compound predicates — e.g. `And`, `Or`, `Not` — for barrier
   payoffs in future families? They're included in the grammar above
   for compositional completeness but no Phase 2 fixture exercises
   them. Implement the dataclasses; skip the evaluator branch (P2.5)
   until a Phase 2-follow-on family uses them.

## Next Steps

1. Collect reviewer feedback on the four remaining open questions.
2. Promote this file from `draft__contract-ir-phase-2-ast-foundation.md`
   to `active__contract-ir-phase-2-ast-foundation.md` when QUA-904 is
   moved to In Progress.
3. File P2.1 as the first Linear sub-ticket; it has no Phase-2-internal
   dependencies.
4. Open follow-on sub-tickets (P2.2–P2.6) in order; each links back to
   its corresponding section in this document.
5. Before Phase 3 coding starts, open a shadow-mode compiler ticket that
   runs kernel selection from `contract_ir` with route metadata masked
   and compares the result against the current route-driven path for the
   four Phase 2 families.
6. Review the dedicated Phase 3 and Phase 4 companion drafts together
   with this file; they now carry the structural selection semantics and
   route-retirement invariants that this Phase 2 document only points
   toward.

Once P2.1–P2.6 all land and QUA-904 marks Done, Phase 3 (QUA-905)
starts. Phase 3's first sub-ticket consumes `blueprint.contract_ir` in
a compiler that matches kernel `@solves_pattern` declarations against
it. The Phase 3 success bar is not just "IR exists"; it is "a fresh
build can select and lower the kernel from IR without a hard-coded
route." Phase 4 (QUA-906) then removes the remaining legacy route /
instrument dependencies from that fresh-build path.
