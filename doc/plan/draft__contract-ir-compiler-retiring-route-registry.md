# Contract IR Compiler — Retiring the Route Registry

## Status

Draft. Pre-queue design document. Not yet the live execution mirror.

## Linked Linear

- QUA-887 — Semantic contract: contract-IR compiler (replace route registry)
- QUA-792 — Binding-first exotic assembly (active successor epic; this plan
  completes its dispatch-replacement slice)
- QUA-727 / QUA-778–791 — Route-registry minimization (Done; groundwork)

## Purpose

This plan expands the four phases originally sketched in QUA-887 into a
reviewable architecture proposal. The goal is to retire `routes.yaml` as the
instrument-dispatch key and replace it with a pattern-based compiler that
matches incoming Contract IRs against kernel declarations.

The document is written to be read in isolation by a reviewer who has not
followed the prior discussion.

Dedicated companion drafts now exist for the two consuming phases:

- `doc/plan/draft__contract-ir-phase-3-solver-compiler.md`
- `doc/plan/draft__contract-ir-phase-4-route-retirement.md`

## Framing

This plan is a continuation of work already in flight, not a green-field
proposal. The repo shows two prior pushes at this target:

- `doc/plan/done__route-registry-minimization.md` — QUA-727 + QUA-778–791
  (completed). Focus: retiring route-card prose (notes, adapter wording,
  promoted-route synthesis authority). Outcome: route cards are now
  metadata-first, but the dispatch key is still `ProductIR.instrument →
  route_id`.
- `doc/plan/active__binding-first-exotic-assembly.md` — QUA-792 (backlog).
  Five-epic program: backend-binding architecture, lowering/assembly
  decoupling, validation/replay migration, operator surface separation,
  exotic composition proof. This is the live strand. It establishes the
  substrate (binding catalogs, family IRs, DSL lowering) but does not
  specify the replacement dispatch mechanism.

What is already in the code (so the plan is not starting from zero):

- `trellis/agent/dsl_algebra.py` has `ContractSignature`, `ContractAtom`,
  `ControlStyle` — a minimal algebraic substrate, intentionally
  structure-preserving.
- `trellis/agent/dsl_lowering.py::SemanticDslLowering` produces typed
  `DslTargetBinding` records with `module.symbol` primitives and typed
  roles, plus explicit `admissibility_errors` instead of guessing.
- `trellis/agent/semantic_contract_compiler.py::SemanticImplementationBlueprint`
  already emits dual views: legacy route/module hints and a conservative
  `dsl_lowering` companion. This is the natural splice point for a Contract
  IR dispatch.
- `routes.yaml` has 30 routes, of which 16 are instrument-dispatch
  (`match.instruments: [X]`), 14 use `match.methods` only, and 6 use
  `conditional_primitives` — a pattern-style when-clause dispatch that
  already works.

## Why Prior Attempts Stalled

Honest assessment based on the pattern of work in the plan docs and the
shape of the code today. The next attempt has to account for each of
these explicitly, or it will stall the same way.

1. Boil-the-ocean scope. Prior attempts (and the instinct everyone has
   when thinking "retire the registry") reach for a general IR that can
   express every instrument — callable / Bermudan, PDE-style path
   dependence, copula baskets, credit. The IR balloons into something as
   complex as the registry it replaces and never lands.
2. Dual-track drift. A new IR is introduced alongside the old system.
   Both need maintenance. The new one never reaches feature parity; the
   old one keeps growing because it is the one that actually prices.
   QUA-727 partially avoided this by minimizing rather than replacing —
   that is why it landed. Full replacement attempts have not.
3. Unclear "done" signal per instrument. Without a concrete parity gate
   per migration slice, "we have migrated variance swaps" is ambiguous,
   so the registry never shrinks even as the new system grows.
4. Kernel signature heterogeneity. `black76_call(F, K, σ, T)` is pure
   scalars. `price_rate_cap_floor_strip_analytical(market_state, spec,
   **11 kwargs)` is market-state plus spec plus eleven keyword args.
   `price_equity_variance_swap_analytical(market_state, spec)` is
   market-state plus untyped spec. Any pattern-matching compiler needs to
   adapt to each kernel's native calling convention. If that adaptation
   layer is built separately from the kernels, it accumulates special
   cases and breaks.
5. Implicit dependencies on `ProductIR.instrument`. Logging, tracing,
   diagnostic text, task-runtime bookkeeping, benchmark scorecards — many
   things read `instrument` as a string tag. "Delete the field" surfaces
   hundreds of call sites.
6. Registry is not a single thing. `routes.yaml` carries at least five
   orthogonal concerns: instrument dispatch, primitive resolution,
   admissibility envelopes, market-data access hints, and scoring
   bonuses. Only instrument dispatch is the retirement target. Trying to
   retire the whole file blocks on unrelated concerns.

## What Is Different This Time

Four principles baked into every phase:

- Parity gate is non-negotiable. No migration slice lands without a
  side-by-side test: old path and new path produce equal output within ε
  tolerance on the full set of benchmark tasks that touch the slice. No
  parity, no migration.
- Additive before subtractive. Phases 1–3 add infrastructure without
  changing existing paths. Only Phase 4 deletes. This prevents "we broke
  everything while building the new thing."
- Scope gate, not time gate. A phase is done when its migration checklist
  is ticked. No date pressure. Time pressure against scope is what killed
  prior attempts.
- Kernel signature normalization is inside the pattern declaration, not
  in a separate layer. Each `@solves_pattern` declaration carries its own
  mapping from Contract IR field values to the kernel's native kwargs.
  This collapses the combinatorial explosion of a central adapter into N
  independent, testable, small adapters.

## Phase 1 — Pattern-keyed kernel declarations inside `routes.yaml`

### Why

The file can shrink dramatically without introducing a new IR. 16 routes
use `match.instruments: [X]`, but 6 already use `conditional_primitives`
(when-clause dispatch on `payoff_family` + `exercise_style` +
`model_family` → primitives). Extending that pattern to the remaining 16
retires instrument-name dispatch as the matching key while keeping
`routes.yaml` as the storage.

### What we do

- For each of the 16 instrument-keyed routes, rewrite its match clause
  from `instruments: [X]` to a pattern declaration on `(payoff_family,
  exercise_style, model_family, payoff_traits)`. Example:
  `equity_variance_swap_analytical` changes from `match.instruments:
  [variance_swap]` to `match.payoff_family: [variance_replication],
  match.model_family: [equity_diffusion]`.
- Collapse groups of instrument-keyed routes that resolve to the same
  kernel family into one pattern-keyed route with `conditional_primitives`
  dispatch (mirroring how `analytical_black76` already does this for
  basket / swaption / vanilla).
- `ProductIR.instrument` stops being consulted by the route matcher.
  Other callers (traces, diagnostics) keep reading it until Phase 4.

### Deliverables

- 16 route rewrites. Each is its own Linear ticket.
- Per-rewrite parity test: `rank_primitive_routes` returns the same
  `PrimitivePlan.route` and `PrimitivePlan.primitives` under the new
  pattern declaration as under the old `instruments:` declaration, for
  every `ProductIR` fixture that hits the route.
- A handful of collapsed routes (projection: 16 narrow routes → ~7
  pattern families).

### Done criteria per rewrite

- New pattern declaration landed.
- Old `match.instruments` entry deleted.
- `rank_primitive_routes` returns identical output for every fixture
  that exercised the old entry (parity test in
  `tests/test_agent/test_route_registry.py`).
- Full agent test suite green.
- Live benchmark task that was routed through the old entry produces
  identical price within ε.

### Failure modes to watch

- Scoring bonuses tied to route id. `trellis/agent/route_scorer.py`
  scores routes with `ScoringContext`. Some bonuses may be keyed on
  specific route ids. Audit before rewriting.
- Admissibility drift. Each route carries admissibility metadata
  (`control_styles`, `event_support`, `supported_state_tags`). When two
  routes collapse into one pattern, their admissibility envelopes must be
  the intersection, not the union — otherwise you silently expand what
  the route accepts.
- Conditional_primitives resolution order. The order of `when:` clauses
  matters (first match wins). When collapsing routes, the combined
  ordering must be explicit and tested.

### Why this will land where prior attempts did not

- No new infrastructure. `conditional_primitives` is 20% of the registry
  already — this is migration, not invention.
- Per-route tickets with per-route parity gates. Each slice is small
  enough to review in a single PR.
- `routes.yaml` gets lighter each time, observably.

## Phase 2 — Contract IR as an algebraic AST alongside ProductIR

### Why

Phase 1 moves dispatch from instrument names to structural tags on a
flat record (`ProductIR`). Real dispatch-by-structure needs a recursive,
compositional AST — so that `barrier_variance_swap` is
`indicator(hit_barrier) × variance_payoff` rather than a new leaf. This
phase introduces that AST without touching dispatch yet.

### What we do

- Extend `dsl_algebra.ContractAtom` into a proper sum type. Sketch:

      ContractIR = Payoff(expr: PayoffExpr)
                 | Exercise(style: ExerciseStyle, schedule: Schedule)
                 | Observation(kind: ObservationKind, schedule: Schedule)
                 | Underlying(process: ProcessRef)
                 | Composite(parts: tuple[ContractIR, ...])

      PayoffExpr = Max(args: tuple[PayoffExpr, ...])
                 | Sub(lhs, rhs)
                 | Indicator(pred: Predicate)
                 | Constant(value: float)
                 | Spot(underlier: str)
                 | Strike(value: float)
                 | Integral(integrand: PayoffExpr, over: Schedule)
                 | ...

- Define simplification rewrites: associativity / commutativity of
  `Max`, distribution of `Indicator` over sums, collapse of `Sub(x, 0)
  → x`. Property-based tests prove the rewrites preserve semantics.
- Decomposer learns to emit Contract IR for a bounded set: vanilla
  call / put, variance swap, digital (cash-or-nothing and
  asset-or-nothing), arithmetic asian. Four instruments only. Not five.
  Not barrier yet.
- Contract IR is purely additive. It lives as a new field on
  `SemanticImplementationBlueprint.contract_ir: ContractIR | None`. No
  production code consumes it yet. No dispatch path reads it. The
  existing `route_id → primitives` pipeline continues to work unchanged.

### Deliverables

- `trellis/agent/contract_ir.py` with the AST and simplification
  rewrites.
- Decomposer extension: `decompose_to_ir()` emits Contract IR for the
  four target payoffs, returns `None` otherwise.
- Property-based test suite (`tests/test_agent/test_contract_ir.py`)
  using Hypothesis or similar to fuzz simplification invariants.
- Fixture-level test that the decomposer produces the expected Contract
  IR for 20+ canonical descriptions spanning the four payoffs.

### Done criteria

- AST types defined with frozen dataclasses.
- Simplification rewrites pass property-based tests for associativity,
  commutativity, idempotence.
- Decomposer emits Contract IR for all 20+ fixtures and matches the
  hand-written expected AST.
- `SemanticImplementationBlueprint.contract_ir` populated end-to-end.
- Full agent test suite green. Contract IR is purely additive, so
  regressions are impossible if the additive discipline holds.

### Failure modes to watch

- Premature generalization. The temptation to "while we are here, let
  us also do path-dependent exercise and stochastic vol" is the kill
  shot. If barrier-under-Heston does not fit, leave it as
  `decompose_to_ir()` returning `None` for that case. Phase 2 covers the
  trivial ground.
- Non-compositional design. If `Payoff` carries instrument-specific
  fields, it is not a real AST — it is just a renamed record. Keep the
  expression types structurally uniform.
- Simplification divergence. If a rewrite is not confluent (different
  simplification orders produce different canonical forms), pattern
  matching in Phase 3 breaks. Property tests must include confluence.

### Why this will land

- Scope explicitly bounded to four instruments. No one is tempted to
  expand it because the phase is done the moment those four work.
- Additive — no existing path changes. Regression risk is zero if the
  additive discipline holds.
- Property-based tests validate the algebra before any compiler
  consumes it.

## Phase 3 — Kernels declare the patterns they solve

See the dedicated companion draft
`doc/plan/draft__contract-ir-phase-3-solver-compiler.md` for the
current mathematical contract, first-wave family scope, and the explicit
Asian blocker treatment.

### Why

This is where Contract IR actually dispatches. After Phase 2 we have an
AST. After Phase 3 kernels in `trellis/models/` declare which ASTs they
solve, and a compiler walks incoming Contract IRs to find the satisfying
kernel.

### What we do

- Introduce a `@solves_pattern(ir_pattern, adapter_fn)` decorator for
  kernels. Example:

      @solves_pattern(
          Payoff(Max(Sub(Spot("S"), Strike(K)), Constant(0)))
            * Exercise("european", T)
            * Underlying("gbm"),
          adapter=lambda ir, ms: dict(
              F=ms.forward(ir.underlier),
              K=ir.strike,
              sigma=ms.vol_surface.black_vol(ir.expiry, ir.strike),
              T=year_fraction(ms.as_of, ir.expiry, ir.day_count),
          ),
      )
      def black76_call(F, K, sigma, T) -> float:
          ...

  The decorator declares both the IR pattern the kernel solves and the
  adapter that translates Contract IR into the kernel's native
  signature. Normalization lives per-kernel, not centrally.
- Compiler: `compile_contract_ir(ir: ContractIR, market_state:
  MarketState) -> KernelCall | None`. Simplifies the IR, pattern-matches
  against the registered declarations, returns a bound `KernelCall(fn,
  kwargs)`. Returns `None` on no match — the legacy
  `rank_primitive_routes` path still runs as fallback.
- Parity harness. For every benchmark task that hits one of the four
  migrated instruments, run both the old route-based pipeline and the
  new Contract IR compiler. Assert output equality within ε.
- Parity tickets, one per instrument: `[vanilla, variance_swap,
  digital, asian]`. Each is a Linear ticket with its own parity test.

### Deliverables

- `trellis/agent/contract_ir_compiler.py` with the decorator, registry,
  pattern matcher, and simplifier feed.
- `@solves_pattern` annotations on `black76_call`, `black76_put`,
  `price_equity_variance_swap_analytical`,
  `price_equity_digital_option_analytical`, and the asian analytical
  helper (whichever subset of the four instruments' kernels lands
  cleanly).
- Parity harness in `scripts/` that runs dual-pipeline pricing on the
  full benchmark corpus and reports any divergence above ε.
- Per-instrument parity test in
  `tests/test_agent/test_contract_ir_compiler.py`.

### Done criteria per instrument

- `@solves_pattern` declaration on the target kernel.
- Adapter translates Contract IR to native kernel kwargs.
- Compiler resolves this kernel for the expected IR patterns.
- Parity harness confirms old path and new path produce equal output
  within ε on all benchmark tasks for this instrument.
- Live benchmark run (`scripts/run_financepy_benchmark.py`) green under
  both pipelines.

### Failure modes to watch

- Pattern matching is not exact equality. `Payoff(Max(Sub(Spot("S"),
  Strike(K)), Constant(0)))` needs to match even when the IR is
  `Payoff(Max(Constant(0), Sub(Spot("S"), Strike(K))))` after
  simplification. The simplifier must normalize before matching. If
  confluence is not proven (Phase 2 output), matching fails silently.
- Kernel signature drift. If `black76_call`'s signature changes, the
  adapter in `@solves_pattern` must change. Keep adapters close to the
  kernel they wrap — same file — so refactoring is local.
- Parity harness lies. The harness must compare on the actual benchmark
  (live FinancePy runs, not just test fixtures), otherwise it passes by
  construction.
- Dual-path maintenance cost. Running both pipelines on every task
  forever is expensive. Make the dual-pipeline mode a flag that stays on
  through migration and flips off instrument-by-instrument after parity
  is proven.

### Why this will land

- Pattern matching is a well-understood technique, not novel compiler
  research.
- Scope bounded to four instruments. Each has its own ticket, parity
  test, and done checklist.
- Dual-path with parity gate makes regressions impossible — if parity
  fails, you do not migrate.
- Normalization lives per-kernel, so there is no central adapter that
  accumulates special cases.

## Phase 4 — Delete migrated routes and eventually retire `ProductIR.instrument`

See the dedicated companion draft
`doc/plan/draft__contract-ir-phase-4-route-retirement.md` for the
fresh-build invariance contract, deletion order, and provenance / replay
separation.

### Why

Phase 4 is a consequence of Phases 1–3, not a separate effort. If the
earlier phases landed with parity gates, Phase 4 is deletion of redundant
code paths. This phase is intentionally boring.

### What we do

- For each migrated instrument (from Phase 3), delete its `routes.yaml`
  entry (which by Phase 1 is a pattern declaration, not an
  instrument-keyed one).
- Flip `rank_primitive_routes` to consult `compile_contract_ir` first
  and fall back to the legacy `match_candidate_routes` for unmigrated
  surface.
- Over time, as more instruments migrate through Phases 2–3, more
  `routes.yaml` entries are deleted.
- `ProductIR.instrument` is retired last, field-by-field. The
  diagnostic / trace consumers of `instrument` get migrated to read from
  Contract IR first. Only when no production code reads
  `ProductIR.instrument` does it get deleted.

### Deliverables

- Deleted route entries in `routes.yaml` for each migrated instrument
  (per-instrument tickets).
- Rewritten `rank_primitive_routes` with Contract IR compiler as the
  primary path and legacy registry as fallback.
- Per-consumer ticket for each reader of `ProductIR.instrument` that
  gets migrated to Contract IR.

### Done criteria per slice

- Route entry deleted.
- Live benchmark tasks for the instrument still green.
- No test or prod path references the deleted route.
- Trace output still identifies the instrument (via Contract IR, not
  via the deleted route id).

### Failure modes to watch

- Trace / diagnostic breakage. Traces and `LIMITATIONS.md` entries
  reference route ids and instrument names. Deleting the route entry
  without updating trace renderers produces broken telemetry.
- Compatibility aliases. Routes have `compatibility_alias_policy` that
  governs deprecated naming. Deleted routes need their aliases
  explicitly retired or redirected.
- `ProductIR.instrument` read sites are not all in `trellis/agent/`.
  Benchmark scorecards, task runtime, and arbiter all read it. Audit
  before deletion.

### Why this will land

- Nothing in Phase 4 is novel. It is deletion gated on passing parity
  tests.
- Deletion is per-instrument, so the unit of risk is one instrument at
  a time.
- The legacy registry persists as fallback until all instruments
  migrate, so partial progress is safe.

## Cross-cutting Principles

These apply to every phase. Dropping any one of them is how prior
attempts stalled.

1. No migration without a parity test. The gate is not "the new path
   works" — it is "the new path produces equal output within ε to the
   old path on all fixtures and all benchmark tasks."
2. No generalization to un-migrated surface. If Phase 2 covers
   vanilla / variance / digital / asian, Phase 3 can only migrate those
   four. Do not try to "cover barrier while we are here." Each
   instrument is its own Phase 3 ticket.
3. Per-kernel adapters. Normalization lives inside `@solves_pattern`,
   not in a central adapter layer. This prevents the central layer from
   becoming the new registry.
4. The registry does not disappear in Phase 1. It becomes declarative
   pattern metadata. It shrinks incrementally through Phase 4.
5. Traces and diagnostics migrate last. Do not touch observability
   until the dispatch change lands. This keeps debugging possible
   throughout the migration.
6. Reviewer gets a parity report per PR. Every migration PR must
   include the parity test output (old vs. new path for all affected
   fixtures and benchmarks) in the PR description.

## Open Questions For The Reviewer

These are the places where the author is least certain and where
outside judgment would help most.

1. Should Phase 1 and Phase 2 overlap? Phase 1 uses pattern-keyed
   `conditional_primitives` inside `routes.yaml`. Phase 2 introduces a
   proper Contract IR. Is there a path where Phase 1's patterns are
   Contract IR patterns from the start, rather than two separate layers?
   This could collapse Phases 1 and 2 but might overconstrain Phase 1.
2. What is the smallest possible Phase 2? The current pick is vanilla +
   variance + digital + asian (four). Could it be two (vanilla +
   variance)? The fewer the scope, the more likely the phase lands, but
   below some threshold the Contract IR does not prove its generality.
3. Kernel normalization — decorator vs. subtyping. The proposal uses
   `@solves_pattern(ir, adapter_fn)`. Alternative: kernels implement a
   `Pricer[IR]` protocol with an `apply(ir, market_state) -> float`
   method. Protocol-based is more discoverable; decorator-based is more
   additive. Which does the codebase's conventions prefer?
4. Relationship to QUA-792. This plan assumes Contract IR completes
   what QUA-792 started. Is that the right framing, or should the
   Contract IR compiler be a parallel epic that feeds into QUA-792
   epic 5 (exotic composition proof)? The reviewer may see this
   differently.
5. Parity tolerance ε. FinancePy parity runs use 1–3% tolerance today.
   For dual-path parity (old vs. new Trellis paths) ε should probably be
   much lower — around 1e-10 relative, since both paths should produce
   bit-identical outputs when the math is equivalent. Kernels with Monte
   Carlo components have RNG state, so ε has to be strategy-dependent.
   Worth deciding up front.

## Next Steps

- Land this document as a draft plan doc for review.
- Collect reviewer feedback on the five open questions.
- Promote to `active__contract-ir-compiler-retiring-route-registry.md`
  once the framing is endorsed and the first implementation ticket is
  queued.
- The first implementation ticket is the smallest Phase 1 slice: one
  instrument-keyed route converted to a pattern declaration, with its
  parity test.
