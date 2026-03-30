# Autonomous Library Development Workstream

This note is the working roadmap for Trellis' next major objective:

`repeat task reruns -> accumulate shared knowledge -> close substrate gaps -> add missing pricing infrastructure under guardrails -> increase task pass rate`

It is meant to be a living document. We should refine it as the failure buckets
change and as new proving grounds succeed or fail.

## North Star

The end goal is not just "more tasks pass." The end goal is:

- repeated pricing-task reruns become informative instead of noisy
- agents stop repeating the same mistakes because they share memory
- missing market-data and substrate gaps are classified honestly
- missing pricing infrastructure is added in bounded, testable slices
- successful reruns promote reusable knowledge back into the platform
- over time, more failed tasks turn into passing tasks without bespoke manual
  rescue prompts

Longer term, this roadmap is also the bridge to a guarded multi-agent "Library
Developer" loop, where LLMs can propose and implement bounded infrastructure
changes, but only under deterministic validation and explicit promotion gates.

## Operating Rules

These rules apply to every phase below:

- deterministic substrate changes use TDD
- agent-behavior changes use eval-driven development
- every phase starts with failure review and ends with reruns
- no new primitive is considered "supported" without deterministic tests
- no knowledge promotion is accepted on prompt quality alone
- task results, traces, lessons, and external issues must remain auditable

## Current Baseline

As of March 25, 2026, the latest-run task store shows:

- many historical failures have already become stale and need reruns under the
  current substrate
- provider/config noise is still large enough to distort the signal if we do
  not classify it separately
- real current market-data gaps still exist, especially around `fx`,
  `forecast_rate` / forecast-curve bridging, and `state_space`
- the UI and trace surfaces are now good enough to inspect why a task passed or
  failed, but they still depend on the underlying run quality

The current proving-ground task groups are:

- stale-failure refresh tranche:
  - `E09`, `T18`, `T23`, `T25`, `T27`, `T33`, `T42`, `T62`
- comparison / stress tranche:
  - `E21` through `E28`
- immediate market-data targets:
  - `T94`, `T105`, `T108`, `E25`
- reusable regression guards:
  - `T74`, `T104`, `E28`

## Immediate Top Priority

Before more family-specific primitives, the next highest-priority tranche is:

- `C0` AI contract synthesis for known product families

This is the bridge from "agents generate payoff code" to "agents define the
guardrailed contract layers that let the library grow safely."

The concrete handoff note is:

- [phase_contract_synthesis_quanto_himalaya_handoff.md](phase_contract_synthesis_quanto_himalaya_handoff.md)

Checked-in implementation plans for this tranche:

- [phase_c0_contract_synthesis_implementation_plan.md](phase_c0_contract_synthesis_implementation_plan.md)
- [phase_c0_quanto_contract_plan.md](phase_c0_quanto_contract_plan.md)
- [phase_c0_himalaya_contract_plan.md](phase_c0_himalaya_contract_plan.md)
- [phase_c0_glue_surface_sweep.md](phase_c0_glue_surface_sweep.md)
- [phase_c1_analytical_support_plan.md](phase_c1_analytical_support_plan.md)
- [phase_c2_family_name_free_semantic_synthesis_plan.md](phase_c2_family_name_free_semantic_synthesis_plan.md)

Recommended execution order:

- `C0.1` shared contract schema, validator, templates, and compiler
- `C0.2` quanto family path tied to `T105`
- `C0.3` future Himalaya runtime-request path for term-sheet requests

Current checked-in family template:

- `quanto`

Reserved next family request case:

- `Himalaya`

Current checked-in progress:

- `C0.1` shared family-contract substrate is implemented.
- `C0.2` core quanto routing is implemented through task/runtime detection,
  planner specialization, platform-request compilation, quant blueprint
  selection, and family-aware validation-bundle selection.
- `C0.2` now also respects `preferred_method` on the quanto family path, so
  `T105`-style analytical and Monte Carlo comparison targets compile into
  distinct routes instead of collapsing to the default analytical branch.
- `C0.2` now also promotes the repeated `T105` quanto failures into checked-in
  deterministic analytical and Monte Carlo route adapters, so the executor can
  reuse known-good family-specific code instead of regenerating the same route.
- `T105` now succeeds end-to-end against the checked-in quanto contract path,
  with reused analytical and Monte Carlo adapters and comparison passing inside
  tolerance.
- a `fresh_build` proving mode now exists on the task/build path, so `T105`
  can exercise live code generation without deterministic route reuse.
- shared quanto input resolution now lives in
  `trellis.models.resolution.quanto`, so generated routes can call a stable
  helper instead of reimplementing market binding from scratch.
- the public payoff surface now includes a small resolved-input adapter base
  plus a Monte Carlo path-adapter base in `trellis.core.payoff`, and the
  checked-in quanto analytical / MC routes now use those scaffolds instead of
  duplicating expiry, simulation, and aggregation glue.
- deterministic validation now preserves structured failure diagnostics through
  invariant execution and bundle execution, so repair prompts can include the
  failing check name, actual value, exception text, and compact input context
  instead of only flat failure strings.
- `_generate_module` now validates expected module structure before accepting
  builder output and can recover import-plus-`def evaluate(...)` repair
  fragments back into the deterministic skeleton, which restored fresh-build
  `T105` success after MC repair snippets had started failing with syntax and
  fragment-shape errors.
- fresh-build candidates now write to isolated scratch modules under
  `_agent/_fresh`, so proving runs no longer overwrite checked-in deterministic
  routes.
- successful fresh-build comparison runs now also snapshot method-level
  promotion candidates under `trellis/agent/knowledge/traces/promotion_candidates`,
  preserving the generated code plus cross-validation evidence without
  automatically overwriting the checked-in deterministic route.
- promotion candidates can now be reviewed through an explicit deterministic
  gate, which writes decision artifacts under
  `trellis/agent/knowledge/traces/promotion_reviews` and computes the
  recommended checked-in route path without auto-adopting the generated code.
- approved reviews can now flow through a separate adoption gate, including a
  dry-run mode and persisted adoption artifacts under
  `trellis/agent/knowledge/traces/promotion_adoptions`, so route replacement is
  explicit and auditable rather than folded into candidate review.
- generated payoffs now face an actual-market smoke gate before they count as a
  successful build, which is meant to catch task-market-state failures earlier
  than comparison pricing.
- Phase 1 of the glue-surface follow-on plan is now implemented: the shared
  quanto resolver contract in `trellis.models.resolution.quanto` is now the
  authoritative analytical market-input surface, including valuation-date and
  correlation aliases that generated routes can rely on instead of probing
  fallback market-state fields.
- Phase 2 of the glue-surface follow-on plan is now implemented: the quanto
  analytical pricing body now lives in `trellis.models.analytical.quanto`, and
  the checked-in analytical adapter delegates to that shared helper rather than
  open-coding the quanto-adjusted Black route.
- Phase 3 of the glue-surface follow-on plan is now implemented: the quanto
  Monte Carlo route glue now lives in `trellis.models.monte_carlo.quanto`,
  including the shared joint-process builder, initial-state builder, engine
  defaults, terminal payoff mapping, and route-level pricing helper.
- Phase 4 of the glue-surface follow-on plan is now implemented in the builder
  path: `_generate_skeleton()` now emits family-aware quanto route scaffolds,
  `_reference_modules()` includes the checked-in Monte Carlo helper surface, and
  prompt guidance now tells the model to call shared route helpers instead of
  rebuilding process / engine / payoff / discount glue ad hoc.
- the latest `T105 --fresh-build` benchmark confirms the Phase 3/4 effect:
  `mc_quanto` now succeeds on the first live-generation attempt, while the
  remaining overall failure was narrowed to an analytical syntax-emission issue
  rather than the old Monte Carlo route-glue problem.
- the analytical fresh-build syntax hardening step is now implemented:
  malformed full-module analytical output can be recovered by extracting a valid
  `evaluate()` body back into the deterministic family scaffold instead of only
  attempting generic fragment recovery.
- Phase 5 reliability benchmarking for the quanto proving ground is now also
  complete enough for the current tranche:
  - `T105 --fresh-build --model gpt-5.4-mini` now succeeds in
    `task_runs/history/T105/20260326T211522397021.json`
  - `T105 --fresh-build --model gpt-5-mini` now succeeds in
    `task_runs/history/T105/20260326T211610323555.json`
  - both successful fresh-build runs emitted promotion candidates, and the
    latest candidate set cleared deterministic review and dry-run adoption
    gates
- remaining follow-on work is no longer the narrow `T105` autonomous loop.
  It is broader analytical substrate work, later runtime market-binding
  behavior beyond mock proving-ground fixtures, and future runtime-authored
  family contracts for requests such as Himalaya.
- the next checked-in follow-on plan is now `C1` analytical support substrate,
  which turns analytical implementation into composition over resolved inputs
  plus reusable subproblem kernels rather than freehand Trellis-specific
  formula glue.
- the first `C1` implementation slice is now checked in:
  - `trellis.models.analytical.support` now exposes foundational discounting,
    forward, payoff-transform, and cross-asset helpers
  - `trellis.models.analytical.quanto` now reuses that support surface more
    explicitly
  - analytical `quanto` prompt guidance now steers fresh-build routes toward
    `trellis.models.analytical.support`
  - deterministic regressions cover the support layer in
    `tests/test_models/test_analytical_support.py`
- the next `C1` support slice is now also implemented:
  - time/rate normalization helpers are checked in
  - cross-asset covariance and forward-bridge helpers are checked in
  - `quanto_adjusted_forward` now composes those smaller helpers
  - the deterministic `T105` route still succeeds after the change
- unsupported semantic requests now emit a structured `semantic_gap` report
  on fallback compilation paths, which is the first gate in the novel-request
  extension loop (`QUA-376`)
- the next analytical TODO slice is now:
  - add explicit autodiff regression tests for `trellis.models.black` and the
    quanto analytical route
  - refactor `trellis.models.analytical.jamshidian` into a pure resolved-input
    kernel plus a thin adapter
  - formalize a traced analytical entrypoint such as `evaluate_raw(...)` or
    `price_raw(...)` on resolved-input adapters, while keeping `evaluate()`
    float-returning for the public payoff boundary
  - update the analytical docs to distinguish traced kernels from adapter
    boundaries
  - defer `barrier.py` until there is a concrete gradient consumer
- the earlier non-blocking critic-path warning
  `name 'get_model_for_stage' is not defined` is now resolved in
  `trellis/agent/executor.py`; the remaining critic concern is stage latency,
  not missing stage-helper wiring.
- the next architectural branch beyond named-family proving work is now also
  documented: `C2` family-name-free semantic product synthesis, whose goal is
  to let an agent synthesize mountain-range-style products from reusable
  semantics without explicit runtime product-name branches such as
  `himalaya_option`.
- the first C2 slice is now checked in: the semantic ranked-observation
  basket contract schema, deterministic validator/compiler, and request
  drafting hooks now exist, so Himalaya-style requests can compile into
  family-name-free semantic metadata instead of a named-family branch.
- the next C2 tranche is also checked in: the generic basket-state resolver,
  ranked-observation state helpers, state-aware Monte Carlo payoff helpers,
  and correlated-basket route selection now exist without a mountain-range
  product-id branch in the runtime path.
- the representative-derivative onboarding and regression slices are now also
  checked in: quanto, callable bond, vanilla option, and swaption all compile
  through the semantic path, while the regression matrix keeps basket-specific
  modules and named-family ids from leaking into those routes.
- the roadmap now keeps three layers separate:
  - semantic understanding: draft the product into a generic contract
  - method arbitration: choose and explain the pricing route plus assumptions
  - numerical pricing: execute the reusable analytical/tree/MC substrate
- `QUA-334` is the docs/knowledge hardening step that keeps the roadmap,
  knowledge store, and import registry aligned with the checked-in substrate.
- runtime task runs now preserve a replayable runtime contract payload with
  snapshot references, evaluation tags, and trace identifiers so eval traces
  remain tied to the semantic request and mock snapshot that produced them.
- the first arbitrary-derivative proving run is documented in
  `docs/qua-284-arbitrary-derivative-proving-run.md`, including the stored task
  record, the semantic trace, and the deterministic mock pricing output

## Milestone Overview

The work is organized into five milestones:

1. `M1` Trustworthy Reruns and Clean Failure Signals
2. `M2` Mock Market-Data Completion
3. `M3` First Missing-Primitive Proving Ground
4. `M4` Second Missing-Primitive Proving Ground
5. `M5` Continuous Knowledge Promotion and Guarded Autonomy

Each milestone is split into smaller concrete phases below.

## M1: Trustworthy Reruns and Clean Failure Signals

Goal:
- make reruns meaningful enough that "what failed?" maps to a real current gap
  instead of stale history, provider noise, or UI/backend drift
- make token usage visible and bounded enough that reruns remain economically
  meaningful

### Phase M1.1: Refresh the Stale Failure Baseline

Goal:
- rerun stale failures under the current substrate and replace outdated
  historical signatures with current traces

Primary task set:
- `E09`, `T18`, `T23`, `T25`, `T27`, `T33`, `T42`, `T62`

Concrete work:
- rerun the stale tranche using the canonical task-run store
- compare old vs new results using the shared-memory evaluation report
- classify each rerun into one of:
  - provider/config noise
  - real market-data gap
  - missing primitive
  - semantic/codegen failure
  - comparison disagreement
- record the refreshed failure bucket summary

Key artifacts:
- canonical task-run records in `task_runs/latest` and `task_runs/history`
- shared-memory comparison reports
- updated issue links for tasks that now escalate into tracked work

Validation:
- rerun tranche completes without relying on stale `task_results_*.json`
- failure-bucket summary is generated from latest canonical runs only

Exit criteria:
- the stale tranche no longer shows old `Available: []`-style capability
  failures as the main diagnosis
- we have a current bucketization for every task in the tranche

### Phase M1.2: Eliminate Provider and Config Noise

Goal:
- ensure that a failed task reflects a pricing-system problem, not a bad model
  default, quota surprise, hanging call, or opaque invalid-JSON crash

Concrete work:
- unify model defaults across CLI, backend, and UI-triggered runs
- keep explicit timeout / retry / invalid-response classification
- separate provider failures from pricing failures in run summaries
- treat quota failures as infra noise, not as missing pricing support
- add preflight checks for active provider/model selection when practical

Key files:
- `trellis/agent/config.py`
- `scripts/run_tasks.py`
- `trellis-ui/backend/routers/tasks.py`
- `trellis-ui/backend/task_runner.py`

Validation:
- deterministic tests for default-model resolution and timeout / invalid-response
  handling
- rerun at least one small task batch from the UI and one from the CLI

Exit criteria:
- new reruns do not silently fall back to stale model defaults
- provider noise is clearly isolated in the failure-bucket summary

### Phase M1.3: Stabilize Audit Trails and Workflow Status

Goal:
- make every rerun inspectable enough that we can tell what the agents did, why
  they failed, and what follow-up work was triggered

Concrete work:
- keep task-run persistence as the canonical source of truth
- ensure latest task-run records include:
  - comparison prices and deviations
  - method-level results
  - market context
  - trace summaries
  - linked Linear and GitHub issues
  - workflow status and next action
- keep UI surfaces aligned to canonical run records

Validation:
- inspect `T104`, `T105`, `E23`, and `T74` in the Task Monitor
- verify passed tasks explain why they passed
- verify failed tasks show what the agents are doing about them

Exit criteria:
- for the latest run of a task, the answer to "why did it pass or fail?" is
  recoverable from persisted artifacts without terminal archaeology

### Phase M1.4: Token Telemetry and Budget Visibility

Goal:
- make LLM usage measurable at the stage, task, and batch level before further
  large rerun campaigns

Concrete work:
- persist prompt, completion, and total token usage per LLM stage where the
  provider exposes it
- attach token usage to:
  - platform traces
  - task-run records
  - batch summaries
- distinguish usage by stage:
  - decomposition
  - spec design
  - code generation
  - critic
  - model validator
  - reflection
- expose when token data is unavailable so missing telemetry is explicit

Validation:
- deterministic tests for token-usage persistence and aggregation
- one small rerun batch showing stage-level token accounting

Exit criteria:
- for a task run, we can answer "where did the tokens go?" without reading raw
  provider logs

### Phase M1.5: Token Budgets, Model Tiering, and Prompt Compression

Goal:
- reduce token burn without weakening the validation and learning loop

Concrete work:
- define default model tiers by stage:
  - cheaper/smaller model for decomposition and reflection
  - stronger model reserved for code generation or hard review stages
- lower overly loose completion caps unless a task explicitly needs more
- cap or summarize shared knowledge payloads instead of always sending the full
  retrieval surface
- avoid running the full critic / validator stack on obviously invalid early
  attempts when deterministic gates already failed
- define per-task and per-batch token budgets with explicit stop reasons when a
  budget is exceeded

Validation:
- deterministic tests for budget enforcement and stage-model selection
- before/after comparison on a small rerun batch using the shared-memory
  evaluation report plus token totals

Exit criteria:
- task reruns use materially fewer tokens on average without reducing the
  quality of the failure classification

### Phase M1.6: Prompt Surface Minimization

Goal:
- reduce prompt size further by sending the smallest context that still changes
  outcomes

Concrete work:
- move from "full retrieved knowledge" to compact route / lesson summaries first
- expand to fuller context only on retries or hard failures
- replace repeated reference code blocks with smaller route cards or primitive
  summaries
- define stage-aware OpenAI defaults the same way Anthropic stages are tiered,
  with stronger OpenAI models reserved for code generation and model validation
- keep prompt-size telemetry in task and trace outputs

Validation:
- before/after token comparison on the same rerun tranche
- deterministic prompt-shape tests for the compact-first path

Exit criteria:
- prompt size drops materially for repeated tasks without degrading the rerun
  outcome buckets

### Phase M1.7: Deterministic-First Routing and Review

Goal:
- move low-ambiguity reasoning out of the LLM path entirely

Concrete work:
- make more first-pass routing and review checks deterministic
- use the LLM only when deterministic checks leave ambiguity
- avoid running the full review stack when import/semantic gates already fail
- use compact route-card builder prompts on first attempt and escalate to fuller
  generation-plan context only on retries
- keep retry prompts focused on failure-specific repair guidance instead of
  replaying broad context:
  - import failures -> import-repair card with approved modules only
  - semantic failures -> semantic-repair card with primitive/route guidance and
    a minimal reference surface
  - post-validation retries -> expanded builder knowledge and full generation
    plan
- add a deterministic lite-reviewer for obvious generated-code mistakes so the
  platform can block cheap, high-confidence failures before critic or
  model-validator token spend

Validation:
- deterministic tests for the non-LLM fast path
- before/after token comparison showing fewer review-stage calls

Exit criteria:
- common routing and early review work no longer consumes LLM tokens by default

### Phase M1.8: Toolization of Primitive Assembly and Validation

Goal:
- shift work from free-form prompt generation into structured library tools

Concrete work:
- expose structured operations for:
  - primitive lookup
  - thin-adapter assembly
  - invariant packs
  - comparison harness execution
  - cookbook-candidate capture
- make the builder orchestrate these tools instead of re-synthesizing large code
  blocks where possible
- feed deterministic primitive lookup, thin-adapter plans, and invariant packs
  directly into builder prompts before falling back to broader reference context

Validation:
- deterministic tests for the new tool surfaces
- rerun evidence showing less code-generation volume on supported products

Exit criteria:
- supported-product builds become thinner orchestrations over structured tools

### Phase M1.9: Memory Distillation and Caching

Goal:
- make experience reuse cheaper than re-reading raw traces and verbose lessons

Status:
- implemented on March 25, 2026 via distilled builder/reviewer/routing prompt
  views plus warm runtime caches for decomposition, knowledge retrieval, and
  generation planning

Concrete work:
- distill verbose lessons, traces, and cookbook patterns into compact reusable
  forms
- cache successful decompositions, route plans, primitive plans, and known
  blocker outcomes
- retrieve distilled memory first, raw artifacts only when drill-down is needed

Validation:
- deterministic tests for cache hits and compact-memory retrieval
- token comparison on repeated tasks from the same family

Exit criteria:
- repeated reruns of similar tasks consume materially less context and fewer
  calls

## M2: Mock Market-Data Completion

Goal:
- close the remaining market-data gaps that the mock connector can reasonably
  satisfy so task failures shift from "missing input" to "real substrate gap"

### Phase M2.1: Close the FX Market-Data Gap

Goal:
- make `fx_rates` a clean first-class path in tasks, execution, and comparison
  workflows

Status:
- implemented on March 25, 2026 via task-selected `fx_rate` -> runtime
  `fx_rates` + `spot` bridging, plus explicit FX market specs for `T105`,
  `T108`, and `E25`

Primary task set:
- `T105`, `T108`, `E25`

Concrete work:
- verify mock snapshot selection and task assertions for FX tasks
- ensure runtime capability checks accept canonical `fx_rates`
- make task results and market context show selected FX inputs clearly
- rerun the FX tranche and verify failures are no longer generic missing-`fx`
  failures

Validation:
- deterministic market-data tests around FX selection
- rerun `T105`, `T108`, and `E25`

Exit criteria:
- the FX tranche no longer fails for avoidable mock-data plumbing reasons

### Phase M2.2: Forecast-Rate and State-Space Bridging

Goal:
- close the remaining mock-data gaps for tasks that need forecast-rate or
  bounded `state_space` support

Status:
- implemented on March 25, 2026 via selected `forecast_curve` -> runtime
  `forward_curve` bridging and a bounded mock `macro_regime` state space used
  by scenario-aware mock/runtime tests and future framework work

Primary task set:
- any refreshed stale failures that now point here

Concrete work:
- bridge `forecast_rate` requirements to canonical forecast-curve availability
- define the smallest simulated `state_space` support needed by current tasks
- keep this bounded to what the current mock regimes can honestly provide

Validation:
- deterministic capability tests
- rerun directly related tasks and scenario-aware stress coverage

Exit criteria:
- tasks that only needed mock forecast/state-space support stop failing for
  avoidable data-plumbing reasons

### Phase M2.3: Connector Stress Tranche as a Standing Regression Slice

Goal:
- use the stress tranche to continuously verify that tasks are genuinely
  exercising the mock connector rather than a broad default market state

Status:
- implemented on March 25, 2026 via a dedicated stress-tranche runner,
  canonical preflight/live stress summaries, and a named batch report path in
  `scripts/run_stress_tranche.py`

Primary task set:
- `E21` through `E28`

Concrete work:
- keep `market:` and `market_assertions:` populated for the stress tranche
- verify compare-ready tasks fail, if at all, for route/code reasons rather
  than missing mock data
- verify honest-block tasks fail for substrate reasons rather than connector
  gaps
- persist one canonical stress-batch report that summarizes:
  - compare-ready vs honest-block outcomes
  - market-context provenance
  - forbidden-failure-pattern hits
  - per-task failure bucket
- treat the tranche as a precondition before deeper proving-ground work:
  - if the connector stress slice regresses, stop and fix that first

Validation:
- deterministic preflight grading
- live stress-tranche batch runs

Exit criteria:
- the stress tranche remains a reliable regression gate for connector-aware task
  execution

## Cross-Cutting Enablers

These are not broad refactors for their own sake. They are the minimum shared
substrates that more than one later tranche depends on.

### Phase X1: Deterministic Validation Bundles

Goal:
- turn route/product family validation into an explicit executable policy layer
  instead of a partly hardcoded, partly prompt-described convention

Status:
- implemented on March 25, 2026 via `trellis.agent.validation_bundles` and
  executor integration that records selected/executed bundle details in
  platform traces

Why now:
- this supports `M3`, `M4`, and the early-exercise workstream
- it is a high-leverage way to improve correctness without increasing agent
  freedom

Concrete work:
- add a validation-bundle registry keyed by route/product family
- separate:
  - universal checks
  - no-arbitrage checks
  - route-contract checks
  - product-family checks
  - comparison-task checks
- make execution/validation choose checks from the selected bundle
- persist selected bundle ids in traces and task runs

Validation:
- deterministic tests for bundle selection and check execution
- one supported-route rerun showing explicit selected-bundle provenance

Exit criteria:
- supported routes no longer depend on loosely coupled prompt guidance for core
  validation behavior

### Phase X2: Shared Pricing/Framework Run Contract

Goal:
- keep pricing-task runs and future framework-task runs on one persisted result
  schema so audit/UI/issue plumbing does not fork

Why now:
- `F1` needs this soon
- this prevents a second incompatible run-history path

Concrete work:
- define a common run-record shape for:
  - pricing runs
  - framework/meta runs
- identify which fields are shared, optional, or pricing-specific
- update persistence helpers and API normalization to accept the shared shape

Validation:
- deterministic schema tests
- one pricing record and one synthetic framework record both normalize cleanly

Exit criteria:
- the framework-task harness can ship without inventing a parallel audit model

Status:
- implemented on March 26, 2026 via the shared task-run contract in
  `trellis.agent.task_run_store`
- pricing and framework/meta runs now persist through the same canonical latest
  and history stores, with `task_kind`, framework payloads, shared learning
  summaries, and common workflow/issue/audit surfaces

### Phase X3: Non-Canonical Knowledge Precision Cleanup

Goal:
- bring older lessons and legacy experience surfaces up to the same precision
  standard as the canonical knowledge layer

Why now:
- this affects `M3`, `M4`, `M5.1`, and later guarded autonomy
- inaccurate legacy guidance can still leak into retrieval and reflection

Concrete work:
- audit and tighten:
  - `trellis/agent/experience.yaml`
  - older lesson entries under `trellis/agent/knowledge/lessons`
- normalize overly specific or inaccurate contracts to the right abstraction
  level
- add regression tests for any historically misleading entries we correct

Validation:
- knowledge-store tests
- retrieval spot checks on representative task families

Exit criteria:
- legacy knowledge no longer undercuts the precision of the canonical layer

## M3: First Missing-Primitive Proving Ground

Goal:
- take one real blocker from the refreshed failure buckets and close it end to
  end as reusable library infrastructure

Status:
- core FX vanilla substrate and deterministic planning support were implemented
  on March 26, 2026:
  - reusable Garman-Kohlhagen analytical kernels in `trellis.models.black`
  - FX-aware analytical route selection in `trellis.agent.quant`
  - deterministic FX primitive planning/lite review in
    `trellis.agent.codegen_guardrails` and `trellis.agent.lite_review`
- canonical knowledge-corpus registration and live reruns remain a separate
  follow-up under `M3.4`

Recommended first target:
- FX vanilla option support

Trigger tasks:
- `E25`
- `T105`
- `T108`

### Phase M3.1: FX Primitive Contract and Scope Note

Goal:
- define the exact mathematical and product scope before implementation

Concrete work:
- write a short contract note covering:
  - supported product class
  - required market inputs
  - supported numerical methods
  - explicit non-goals
- decide the canonical module placement and route naming

Validation:
- review note checked into `docs/`

Exit criteria:
- there is one unambiguous contract for what "FX vanilla support" means

### Phase M3.2: Red Tests for FX Support

Goal:
- define the deterministic substrate gates before writing the implementation

Concrete work:
- add tests for:
  - analytical FX pricing formula
  - runtime market-data requirements
  - route selection and primitive planning
  - comparison-task behavior for `E25`

Validation:
- targeted red tests fail for the expected missing support

Exit criteria:
- we have a deterministic proof that the current substrate is insufficient

### Phase M3.3: Implement the FX Primitive and Route

Goal:
- add the reusable library primitive, not a bespoke generated artifact

Concrete work:
- implement the canonical FX analytical route
- wire market-data extraction and runtime use
- register method/route selection in the planning stack
- keep the implementation reusable for direct library use, not only tasks

Likely file areas:
- `trellis/models/`
- `trellis/instruments/`
- `trellis/agent/quant.py`
- `trellis/agent/codegen_guardrails.py`

Validation:
- targeted deterministic tests
- comparison task reruns

Exit criteria:
- FX vanilla pricing is reusable substrate

### Phase M3.4: Knowledge Registration and Reruns

Goal:
- teach the rest of the system to use the new primitive correctly

Concrete work:
- update cookbooks, method requirements, contracts, and route metadata
- rerun `E25`, `T105`, and `T108`
- compare before/after task outcomes and failure buckets

Validation:
- knowledge tests
- rerun tranche report

Exit criteria:
- the FX tasks pass, or fail only for non-FX reasons

Status:
- partially implemented on March 26, 2026:
  - FX tranche rerun/report tooling added via `scripts/run_fx_tranche.py`
  - before/after tranche comparison and promotion-discipline summaries now use
    the shared run contract
  - canonical cookbook / method-requirement / contract registration under
    `trellis/agent/knowledge/` is now complete
  - targeted knowledge tests and focused FX regression checks passed
  - live reruns were attempted and showed the FX knowledge in trace summaries
- remaining work:
  - rerun `E25`, `T105`, and `T108` after provider/runtime noise is reduced
  - classify whether any FX task still has a real substrate blocker once the
    provider path stops failing in `spec_design` / invalid-JSON handling

## M4: Second Missing-Primitive Proving Ground

Goal:
- prove the same loop on a harder blocker family

Recommended second target:
- local-vol support centered on `E23`

### Phase M4.1: Local-Vol Preflight Review

Goal:
- verify the blocker is real and bounded before committing to implementation

Concrete work:
- collect the latest blocker reports and traces for `E23`
- identify what is missing:
  - model process
  - PDE wiring
  - MC wiring
  - market-data extraction
  - route registration
- explicitly reject scope creep into unrelated exotic support

Validation:
- checked-in phase note with bounded scope

Exit criteria:
- one minimal local-vol slice is defined

Status:
- implemented on March 26, 2026 via
  `docs/refactor_local_vol_preflight_review.md`
- bounded outcome:
  - market-data plumbing is already present
  - the real gap is route/adapter execution
  - `M4.2` should start with the narrower Monte Carlo local-vol slice before
    deciding whether a separate PDE operator tranche is still required

### Phase M4.2: Local-Vol Primitive Slice

Goal:
- implement only the smallest real primitive family needed for the target tasks

Concrete work:
- add the missing local-vol execution primitive(s)
- keep the slice narrow enough that it can be cross-validated cleanly

Validation:
- deterministic tests for the new primitive path

Exit criteria:
- the local-vol target tasks reach execution or honest narrower blockers

Status:
- implemented on March 26, 2026 via
  `docs/refactor_local_vol_monte_carlo_slice_review.md`
- bounded outcome:
  - local-vol Monte Carlo for vanilla equity now has a reusable primitive plus
    deterministic route support
  - the remaining local-vol proving-ground question is now the PDE side of
    `E23`, not the MC branch

### Phase M4.3: Route, Knowledge, and Comparison Integration

Goal:
- make the system able to plan and compare the local-vol methods correctly

Concrete work:
- register the route and cookbook support
- rerun `E23`
- inspect method-level comparison results and blocker deltas

Validation:
- knowledge tests
- task reruns

Exit criteria:
- `E23` either passes or now fails for a much narrower and more honest reason

### Phase M4.4: Early-Exercise Monte Carlo Policy Family

Goal:
- generalize `exercise_monte_carlo` from a Longstaff-Schwartz-only mental model
  into a small approved family of optimal-stopping policy classes

Approved policy classes:
- `longstaff_schwartz` (implemented)
- `tsitsiklis_van_roy` continuation regression (planned)
- `primal_dual_mc` (implemented)
- `stochastic_mesh` (implemented)

Concrete work:
- keep the semantic / prompt / planning layer aligned around those four classes
- add common library constructs for early-exercise Monte Carlo:
  - stopping-policy contract
  - continuation estimator contract
  - lower-bound / upper-bound result bundle where applicable
- then implement the missing classes in bounded slices instead of as one large
  rewrite

Implementation note:
- see `docs/early_exercise_monte_carlo_policy_family.md`

Validation:
- deterministic tests for policy-family planning and semantic contracts
- route-aware reruns for American / Bermudan task families

Exit criteria:
- Trellis no longer treats Monte Carlo early exercise as synonymous with one
  algorithm, while staying honest about which classes are implemented versus
  planned

Status:
- completed at the policy-family level by March 26, 2026:
  - `EEMC.1` shared contracts implemented
  - `EEMC.2` `longstaff_schwartz` refactored onto shared contracts
  - `EEMC.3` `tsitsiklis_van_roy` implemented
  - `EEMC.4` `primal_dual_mc` implemented
  - `EEMC.5` `stochastic_mesh` implemented
  - `EEMC.6` route/task integration implemented

## M5: Continuous Knowledge Promotion and Guarded Autonomy

Goal:
- make successful reruns compound, and make the eventual autonomous
  Library-Developer loop safe enough to prove in bounded slices

### Phase M5.1: Promotion Discipline for Successful Reruns

Goal:
- ensure each successful rerun leaves reusable knowledge behind

Concrete work:
- promote lessons and cookbook candidates from successful reruns
- compare before/after retrieval for similar later tasks
- keep shared-memory evaluation reports for each batch

Validation:
- rerun reports show lesson and shared-knowledge deltas

Exit criteria:
- successful fixes become available to later tasks without bespoke prompt edits

Status:
- implemented on March 26, 2026 from the run/reporting side:
  - tranche summaries now report captured lessons, attribution, cookbook
    candidates, knowledge traces, and successful runs that left no reusable
    artifacts
  - FX tranche reruns can now emit a shared-memory report plus promotion
    discipline report in one batch
- remaining deeper promotion-policy tuning still depends on Knowledge-Agent
  work inside `trellis/agent/knowledge/`

### Phase M5.2: Semi-Autonomous Library-Developer Workflow

Goal:
- structure the eventual multi-agent developer loop before turning it on for
  real infrastructure changes

Concrete work:
- define the bounded roles in a primitive-implementation loop:
  - triage
  - architect/planner
  - library developer
  - critic / arbiter / validator
  - test / verification
  - knowledge promotion
- define the hard gates for a change to be accepted

Validation:
- checked-in workflow note or extension to this document

Exit criteria:
- the autonomous developer loop is specified as a guarded process, not an
  aspiration

### Phase M5.3: First Guarded Autonomous Developer Proving Ground

Goal:
- let the agent group propose and implement a bounded primitive slice under hard
  review and validation gates

Candidate target:
- whichever of the `M3` or `M4` families still has one bounded unresolved gap

Concrete work:
- use blocker reports to generate a bounded implementation plan
- require:
  - deterministic tests
  - critic / validator acceptance
  - knowledge registration
  - rerun evidence
- promote only if all gates pass

Validation:
- proving-ground report

Exit criteria:
- one primitive slice is added through a guarded agent-development loop without
  bypassing deterministic validation

## Parallel Workstream: Framework and Meta Task Harness

Goal:
- keep non-priceable framework, infrastructure, and experience tasks executable
  after moving them out of the pricing-task runner

Why:
- `FRAMEWORK_TASKS.yaml` now holds the tasks that should not run through the
  pricing-task build path
- those tasks still matter as evolution and extraction work, but they need a
  different harness with different success criteria

Task set:
- `T91`-`T93`
- `E01`-`E20`

### Phase F1: Framework Task Contract and Runner

Goal:
- define what a framework/meta task is allowed to do and how it reports success

Concrete work:
- add a dedicated framework-task runner that does not assume a pricing build
- define structured result classes for:
  - extraction candidate
  - consolidation candidate
  - infrastructure review
  - explicit blocked / does-not-yet-apply outcome
- keep audit traces and issue creation aligned with the pricing-task runner

Validation:
- deterministic tests for manifest loading and contract enforcement
- one small framework-task slice run through the new harness

Exit criteria:
- moved tasks are executable through a dedicated path instead of being silently
  ignored or rejected as pricing tasks

Status:
- implemented on March 26, 2026 via `trellis.agent.framework_runtime` and
  `scripts/run_framework_tasks.py`
- framework/meta tasks now persist as:
  - `extraction_candidate`
  - `consolidation_candidate`
  - `infrastructure_review`
  - `does_not_yet_apply`
  on the shared run contract

### Phase F2: Framework Task UI and Audit Surfacing

Goal:
- make framework/meta runs visible in the same operational surfaces as pricing
  runs

Concrete work:
- add UI/API support for framework-task latest runs and history
- preserve trace links, issue refs, and next-action summaries
- keep pricing and framework inventories clearly separated in the UI

Validation:
- run one extraction task and inspect the persisted result plus UI audit trail

Exit criteria:
- framework/meta task execution is inspectable without terminal archaeology

### Phase F3: Framework Task Promotion Loop

Goal:
- let successful framework tasks feed back into the library and knowledge
  system the same way pricing-task reruns do

Concrete work:
- wire extraction / consolidation outputs into:
  - issue creation
  - cookbook / lesson candidates where relevant
  - explicit follow-up work for Library Developer
- define when a framework task counts as complete versus merely proposed

Validation:
- one or two extraction tasks produce actionable persisted outputs and follow-up
  traces

Exit criteria:
- framework tasks participate in the same learning and development loop instead
  of sitting outside it

## Maintenance Cadence

This roadmap should be revisited after:

- every stale-tranche rerun batch
- every market-data gap closure
- every missing-primitive proving ground
- every substantial shift in the latest failure buckets

The minimum update on each revisit should be:

- what changed in the failure buckets
- which phase is now current
- which phase is blocked
- what the next rerun tranche should be

## Current Recommended Execution Order

Completed in version 1:

1. `M1.1` through `M1.9`
2. `M2.1` through `M2.3`
3. `X1` Deterministic validation bundles
4. `EEMC.1`, `EEMC.2`, and `EEMC.4` through `EEMC.6`
5. `X2` Shared pricing/framework run contract
6. `F1` Framework task contract and runner
7. `M5.1` Promotion-discipline reporting/persistence slice

Recommended next order in the current version:

1. remaining Knowledge-Agent-owned `M3.4` work:
   canonical FX knowledge registration plus live reruns
2. `X3` Non-canonical knowledge precision cleanup
3. remaining `M4.3` local-vol proving ground
4. `F2` and `F3` framework/meta task UI and promotion

Deferred to the next version:

1. `M5.2` Semi-autonomous Library-Developer workflow
2. `M5.3` First guarded autonomous developer proving ground

## Open Questions

These are intentionally left open until the earlier phases tighten the signal:

- which provider/model combination should be the default for long task batches
- whether `state_space` should remain a market-data capability or become a more
  explicit execution primitive dependency
- how aggressive cookbook auto-promotion should be when reflection quality is
  weak
- when the first guarded autonomous-developer proving ground should move from
  supervised to semi-automatic execution

## Revision Log

- `2026-03-25`: initial roadmap created from the current task-failure, shared
  memory, market-data, and primitive-proving-ground workstreams
- `2026-03-25`: added parallel framework/meta task harness workstream after
  moving non-priceable tasks into `FRAMEWORK_TASKS.yaml`
- `2026-03-25`: reordered pending work to prioritize connector regression,
  validation bundles, FX/local-vol proving grounds, and framework-task
  execution before guarded autonomous developer work
- `2026-03-26`: completed the shared pricing/framework run contract (`X2`),
  the first framework/meta task runner (`F1`), and the promotion-discipline
  reporting slice of `M5.1`; marked `M3.4` as partially complete pending
  Knowledge-Agent-owned canonical registration and live FX reruns
- `2026-03-26`: completed `M4.1` local-vol preflight review and `EEMC.3`
  `tsitsiklis_van_roy`
- `2026-03-26`: completed `M4.2` local-vol Monte Carlo slice; the remaining
  local-vol proving-ground work is now the PDE/comparison side of `M4.3`
