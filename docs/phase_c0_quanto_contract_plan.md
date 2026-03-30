# Phase C0 Detailed Plan: Quanto Contract Path

## Goal

Define the first deterministic family-contract path for `quanto` so `T105`
stops degrading into generic `european_option` routing.

Current checked-in progress:

- the checked-in `quanto_option` contract template, validator, and compiler are
  in place
- `task_runtime`, `planner`, `platform_requests`, `quant`, and
  `codegen_guardrails` now preserve the `quanto_option` family path
- family-aware quanto validation-bundle selection is wired for deterministic
  validation
- `preferred_method` now selects distinct analytical vs Monte Carlo routes on
  the quanto family path, which is required for `T105` comparison targets such
  as `quanto_bs` and `mc_quanto`
- repeated `T105` failures are now promoted into checked-in deterministic
  adapters for the analytical quanto path and the correlated-GBM Monte Carlo
  path, so the build loop can reuse them directly
- `T105` now succeeds end-to-end on the checked-in quanto contract path, with
  analytical and Monte Carlo comparison both passing and no code-generation
  attempts required
- `fresh_build` proving mode now exists on the task/build path, so `T105` can
  bypass deterministic supported-route reuse and measure live code generation
- shared quanto market-input resolution now lives in
  `trellis.models.resolution.quanto`, and the checked-in adapters import that
  helper instead of open-coding market binding
- `trellis.core.payoff` now exposes a small resolved-input adapter scaffold and
  a Monte Carlo path-adapter scaffold, and the checked-in quanto routes use
  those bases to centralize expiry handling, engine defaults, path
  normalization, and discounted aggregation
- deterministic validation now returns structured failure diagnostics through
  the bundle path, so fresh-build repair prompts can cite the failing check,
  actual PV, exception text, and compact market/input context
- the builder now rejects compilable-but-structurally-incomplete repair output
  and reconstructs import-plus-`def evaluate(...)` fragments against the
  deterministic skeleton, which is now part of the fresh-build `T105` proving
  path
- fresh-build candidate modules now write to `_agent/_fresh`, preserving the
  checked-in deterministic adapters while proving runs are in progress
- successful fresh-build comparison runs now emit method-level promotion
  candidate snapshots under `trellis/agent/knowledge/traces/promotion_candidates`,
  so generated code and cross-validation evidence are preserved for review
  before any deterministic-route promotion decision
- those snapshots now have an explicit deterministic review gate, producing
  approval or rejection artifacts under
  `trellis/agent/knowledge/traces/promotion_reviews` before any future route
  adoption step
- approved reviews can now be passed through a separate adoption gate with
  `dry_run` support, producing adoption artifacts under
  `trellis/agent/knowledge/traces/promotion_adoptions` before any checked-in
  route file is overwritten
- generated candidates now face an actual-market smoke gate against the task
  market state before they are treated as successful builds
- Phase 1 of the glue-surface follow-on plan is now implemented:
  `ResolvedQuantoInputs` is now the authoritative analytical resolver contract,
  including valuation-date and correlation aliases that generated routes can
  rely on instead of re-deriving market-state fields ad hoc
- Phase 2 of the glue-surface follow-on plan is now implemented:
  `trellis.models.analytical.quanto.price_quanto_option_analytical` is now the
  shared quanto analytical kernel, and the checked-in analytical adapter
  delegates to it directly
- Phase 3 of the glue-surface follow-on plan is now implemented:
  `trellis.models.monte_carlo.quanto` is now the shared quanto Monte Carlo
  helper surface, and the checked-in MC adapter delegates process construction,
  initial-state binding, engine defaults, and terminal payoff mapping to that
  module instead of open-coding those steps
- Phase 4 of the glue-surface follow-on plan is now implemented:
  the builder now emits family-aware quanto route scaffolds and prompt guidance
  that explicitly prefer shared analytical and Monte Carlo route helpers over
  ad hoc route glue
- the latest `T105 --fresh-build` proving run confirms the Phase 3/4 outcome:
  the Monte Carlo `mc_quanto` branch now succeeds on the first live-generation
  attempt, while the remaining overall failure was narrowed to an analytical
  syntax-emission issue rather than the old Monte Carlo route-glue problem
- the analytical fresh-build syntax hardening step is now implemented:
  malformed full-module analytical output can be recovered by extracting a valid
  `evaluate()` body back into the deterministic family scaffold instead of only
  attempting generic fragment recovery
- Phase 5 reliability benchmarking is now complete enough for the current
  quanto tranche:
  - `T105 --fresh-build --model gpt-5.4-mini` succeeded in
    `task_runs/history/T105/20260326T211522397021.json`
  - `T105 --fresh-build --model gpt-5-mini` succeeded in
    `task_runs/history/T105/20260326T211610323555.json`
  - the latest successful fresh-build candidate set cleared deterministic
    review and dry-run adoption gates:
    - `trellis/agent/knowledge/traces/promotion_reviews/20260326_211710_t105_quanto_bs_approved.yaml`
    - `trellis/agent/knowledge/traces/promotion_reviews/20260326_211710_t105_mc_quanto_approved.yaml`
    - `trellis/agent/knowledge/traces/promotion_adoptions/20260326_211721_t105_quanto_bs_ready.yaml`
    - `trellis/agent/knowledge/traces/promotion_adoptions/20260326_211721_t105_mc_quanto_ready.yaml`
- remaining near-term work is no longer the narrow quanto proving loop itself.
  It is broader analytical substrate expansion, later connector-backed runtime
  market binding, and broader promotion from task-specific adapters into more
  general library surfaces where appropriate
- the concrete next checked-in tranche for that analytical substrate work is
  now documented in `docs/phase_c1_analytical_support_plan.md`
- the earlier non-blocking critic warning
  `name 'get_model_for_stage' is not defined` is now fixed in the build path;
  fresh-build follow-on work should treat critic latency and cost as the
  remaining issue rather than stale missing-helper wiring

Initial scope is intentionally narrow:

- single-underlier European quanto option
- domestic payout currency and foreign underlier currency are explicit
- analytical quanto adjustment and correlated Monte Carlo are the only
  candidate methods
- sensitivities are honest and likely `bump_only`

Out of scope for this phase:

- American or Bermudan quanto structures
- path-dependent quanto exotics
- multi-asset quanto baskets
- generalized quanto pricing support beyond the narrow checked-in single-name
  proving route

## Current Repo Surfaces

Inspect first:

- `trellis/agent/task_runtime.py`
  - now maps `quanto` titles to `quanto_option`
- `trellis/agent/planner.py`
  - now has contract-backed quanto specialized schemas
- `trellis/agent/quant.py`
  - now exposes a blueprint-backed pricing-plan path for checked-in family
    contracts
- `trellis/agent/codegen_guardrails.py`
  - now expresses explicit quanto analytical and correlated-GBM MC route
    obligations
- `trellis/agent/validation_bundles.py`
  - now adds family-specific quanto checks when the family blueprint or family
    instrument is known
- `trellis/agent/platform_requests.py`
  - now compiles known quanto requests through the checked-in family-contract
    path before generic `ProductIR` fallback
- `trellis/models/resolution/quanto.py`
  - shared stable helper for quanto spot / FX / curve / correlation binding
- `trellis/models/analytical/quanto.py`
  - shared analytical kernel for quanto-adjusted Black pricing
- `trellis/models/monte_carlo/quanto.py`
  - shared Monte Carlo helper surface for quanto process wiring, terminal
    payoff mapping, and route-level pricing
- `trellis/models/processes/correlated_gbm.py`
  - usable substrate for underlier/FX joint MC simulation
- `TASKS.yaml`
  - `T105` defines the proving-ground task
- `scripts/run_tasks.py`
  - now supports `--fresh-build` to benchmark live code generation on a task
- `task_runs/latest/T105.json`
  - latest normal run succeeds via deterministic reuse on the checked-in
    `quanto_option` family path
- `task_runs/history/T105/20260326T172314788036.json`
  - first fresh-build proving run; MC succeeded, analytical failed

Useful comparison surface:

- `task_runs/latest/T108.json`
  - shows the deterministic FX vanilla route that C0 should learn from but not
    blindly reuse

## Proposed Contract Scope

Recommended `family_id`:

- `quanto_option`

Recommended minimum semantics:

- `underlier_asset`
- `option_type`
- `strike`
- `expiry_date`
- `notional`
- `domestic_currency`
- `underlier_currency`
- `fx_pair`
- `settlement_style`
- `quanto_fixing_rule`

Required market-data inputs:

- domestic discount curve
- foreign discount or carry curve
- underlier spot
- FX spot
- underlier vol
- FX vol
- underlier/FX correlation

Optional market-data inputs:

- dividend or convenience yield
- local vol or smile data for later extensions

Derived-input policy:

- allow explicit bridges from connector fields into:
  - domestic discounting
  - foreign discounting or carry
  - forward-style drift inputs
- do not allow the runtime to fabricate missing `fx_vol` or
  `underlier_fx_correlation` silently
- if estimation is allowed later, it must be recorded with provenance

Method contract:

- reference method:
  - `analytical`
- production/proving methods:
  - `analytical`
  - `monte_carlo`
- explicit unsupported variants:
  - early exercise
  - path dependence
  - basket quanto
  - stochastic rates

Sensitivity contract:

- initial level:
  - `bump_only`
- measures to allow initially:
  - `dv01`
  - `duration`
  - `convexity`
  - `key_rate_durations`
  - `vega`
- note:
  - validator should downgrade or warn if a route claims `native`
  - `delta` and `rho` should remain deferred until the shared runtime
    sensitivity surface expands beyond the current method-level contract

## Deterministic Validation Scope

Structural checks:

- currencies and `fx_pair` are both present
- domestic and underlier currencies are distinct
- method labels normalize to known Trellis method families
- analytical route is only allowed for European single-underlier semantics

Coherence checks:

- analytical path requires `fx_vol` and `underlier_fx_correlation`
- MC path requires a joint underlier/FX state description
- payout semantics use domestic discounting
- foreign carry/discount semantics are explicit instead of inferred from a vague
  `forecast_curve` label

Unsupported-claim checks:

- reject contracts that mark path-dependent quanto as `analytical`
- reject contracts that mark sensitivities as `native`
- reject contracts that omit correlation but still claim a quanto adjustment

Runtime-binding checks:

- the contract must distinguish observed vs derived vs estimated inputs
- the contract must state which required inputs must come from connectors
- missing required inputs must map to family-specific error messages rather than
  generic capability failures

## Blueprint / Compiler Scope

The compiler output should preserve enough detail for the next implementation
tranche.

Required blueprint contents:

- `family_id = "quanto_option"`
- normalized `product_ir`
- candidate methods: `analytical`, `monte_carlo`
- required market data with domestic/foreign aliases preserved
- connector-binding hints for market-data resolution
- derivation/provenance rules for any non-observed inputs
- target spec schema hint for:
  - `QuantoOptionAnalyticalPayoff`
  - `QuantoOptionMonteCarloPayoff`
- primitive route hints
- adapter obligations
- target modules
- proving tasks

Recommended route hints:

- analytical route:
  - reuse existing Black/GK-style analytical kernel patterns
  - add a quanto-adjustment adapter layer rather than hiding semantics in a
    prompt
- Monte Carlo route:
  - use `CorrelatedGBM` or a thin joint-process adapter around it
  - simulate underlier and FX together
  - discount in domestic currency

## Runtime Market Binding Expectations

`TASKS.yaml` mock market specs are only proving-ground fixtures for `T105`.
They should not be treated as the target runtime interface.

For real quanto requests, the embedded AI should be able to:

- read the full contract or term sheet
- identify the underlier, currencies, and payout semantics
- resolve the appropriate connector fields for:
  - underlier spot
  - FX spot
  - domestic discounting
  - foreign discounting or carry
  - underlier vol
  - FX vol
  - underlier/FX correlation
- raise a meaningful error if a required input cannot be fetched or derived

Best-effort derivation is acceptable only when policy is explicit. For example:

- foreign discounting may come from a forecast/discount bridge when the
  connector surface makes that mapping explicit
- correlation may later be estimated from historical data if the runtime policy
  allows it, but the result must be labeled as estimated and should not be
  treated the same as directly sourced market data

The runtime should never silently downgrade a missing required quanto input into
generic `european_option` behavior.

## Cross-Check Expectations

Quanto is the best initial family for a true independent cross-check.

Expected hierarchy:

- first pricing route:
  - analytical quanto adjustment
- first cross-check route:
  - correlated Monte Carlo using `CorrelatedGBM`
- later follow-up once knowledge accumulates:
  - additional MC variance reduction or control-variate routes

This means C0 should preserve enough contract detail that the embedded AI can
eventually:

- select the analytical route when data quality supports it
- fall back to MC when the analytical assumptions are not met
- compare both when both are available

Likely target modules for the later implementation tranche:

- `trellis/models/analytical/quanto.py`
  - preferred new home for the closed-form adjustment
- `trellis/models/processes/correlated_gbm.py`
  - inspect first before adding a new process type
- `trellis/agent/planner.py`
  - contract-backed spec schema hook
- `trellis/agent/codegen_guardrails.py`
  - new quanto route and primitive/adaptor notes

## Likely Files To Change

Phase-one contract tranche:

- `trellis/agent/family_contracts.py`
- `trellis/agent/family_contract_validation.py`
- `trellis/agent/family_contract_templates.py`
- `trellis/agent/family_contract_compiler.py`
- `trellis/agent/task_runtime.py`
- `trellis/agent/planner.py`
- `trellis/agent/quant.py`
- `trellis/agent/codegen_guardrails.py`
- `trellis/agent/validation_bundles.py`
- `trellis/agent/platform_requests.py`
- `trellis/agent/term_sheet.py`

Likely later implementation tranche:

- `trellis/models/analytical/quanto.py`
- `trellis/models/processes/correlated_gbm.py`

## Exact Test Files To Add Or Extend

Add:

- `tests/test_agent/test_family_contracts.py`
  - `test_quanto_contract_template_validates`
  - `test_quanto_contract_rejects_missing_correlation`
  - `test_quanto_contract_compiles_to_expected_blueprint`

Add:

- `tests/test_agent/test_platform_requests.py`
  - `test_compile_term_sheet_request_uses_quanto_family_contract`
  - `test_quanto_compiled_request_preserves_connector_binding_hints`

Extend:

- `tests/test_agent/test_task_runtime.py`
  - assert `T105`-style titles map to `quanto_option`, not `european_option`
- `tests/test_agent/test_planner.py`
  - assert a quanto request chooses a contract-backed specialized schema
- `tests/test_agent/test_quant.py`
  - assert quanto contract routing preserves FX vol and correlation requirements
- `tests/test_agent/test_codegen_guardrails.py`
  - assert the blueprint yields a quanto-specific route and points at
    `correlated_gbm`
- `tests/test_agent/test_validation_bundles.py`
  - assert quanto bundles include family-specific comparison or reduction checks
- `tests/test_agent/test_term_sheet.py`
  - parse fixture for a simple quanto term sheet
- later integration/eval surface:
  - verify connector fetch vs derive vs missing-data error behavior on
    non-mock requests

## Phased Implementation Plan

### Q0: Review and boundary confirmation

Method:

- review only

Checklist:

1. review prior FX work and `T105` / `T108` persisted runs
2. review the main C0 plan and the quanto plan
3. confirm the exact initial scope:
   - single-underlier
   - European
   - analytical + MC
   - no early exercise
4. record any scope correction in docs before code begins

Expected summary:

- exact quanto slice frozen
- known unsupported variants listed explicitly

### Q1: Quanto contract template and validator

Method:

- TDD

Write tests first:

- `test_quanto_contract_template_validates`
- `test_quanto_contract_rejects_missing_correlation`
- `test_quanto_contract_rejects_same_currency_domestic_and_underlier`
- `test_quanto_contract_marks_estimated_inputs_explicitly`

Implement:

- quanto template in `trellis/agent/family_contract_templates.py`
- quanto validation rules in `trellis/agent/family_contract_validation.py`

Run:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_family_contracts.py -q
```

Docs:

- update this file if any contract field names change

Expected summary:

- validated quanto contract exists
- runtime fetch/derive/error expectations are explicit

### Q2: Quanto compiler and routing integration

Method:

- TDD for deterministic surfaces
- EDD for request compilation behavior

Write tests first:

- `test_quanto_contract_compiles_to_expected_blueprint`
- `test_t105_title_maps_to_quanto_option`
- `test_quanto_request_uses_contract_backed_schema`
- `test_quanto_quant_route_preserves_fx_vol_and_correlation_requirements`
- `test_quanto_compiled_request_preserves_connector_binding_hints`

Implement:

- compile validated contract into blueprint
- map requests to `quanto_option`
- integrate planner/quant/codegen with the blueprint

Run:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_family_contracts.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_planner.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_requests.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_codegen_guardrails.py -q
```

Docs:

- update route/cross-check notes if the blueprint shape changes

Expected summary:

- quanto no longer degrades to generic vanilla routing
- analytical-vs-MC cross-check path remains available in the blueprint

### Q3: Quanto validation bundles and runtime-binding behavior

Method:

- TDD for bundle logic
- EDD for honest runtime behavior

Write tests first:

- quanto family validation-bundle selection
- meaningful missing-data error messaging
- honest blocking when required connector data is unavailable

Implement:

- quanto family bundle hints
- connector-binding hint propagation into compiled requests
- family-specific failure wording

Run:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_validation_bundles.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_requests.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_term_sheet.py -q
```

Docs:

- update this file’s runtime-binding section if error policy changes

Expected summary:

- what a real quanto request can now compile and bind
- what later connector-backed pricing work still needs to be implemented

## Acceptance Criteria

The quanto contract path is complete when:

- a checked-in `quanto_option` contract template exists and validates
- `T105`-style requests compile with `family_id = quanto_option`
- the compiled blueprint preserves:
  - domestic-vs-foreign payout semantics
  - FX vol
  - underlier/FX correlation
- the compiled blueprint tells the runtime AI which inputs must be fetched,
  which may be derived, and how missing inputs should fail
- planner/quant/generation planning no longer treat the request as plain
  `european_option`
- unsupported quanto variants are explicitly blocked or downgraded to MC-only
- the route definition preserves analytical-vs-MC cross-checking as a runtime
  capability, not just a mock-task fixture

Current status against those criteria:

- satisfied for the narrow proving slice used by `T105`
- not yet satisfied for richer connector-backed market binding or broader
  quanto families beyond the current single-underlier European route

## Risks / Open Questions

- The mock market layer currently uses `forecast_curve: EUR-DISC`; the contract
  layer needs a clear rule for when that stands in for foreign discounting.
- The exact analytical formula variant must be fixed up front. The initial
  contract should target one canonical single-underlier European quanto
  definition and reject other variants.
- `MarketState` may need future expansion for FX vol and correlation storage;
  C0 should describe this honestly rather than pretending the adapter already
  exists.
- If correlation estimation is allowed in later runtime work, the policy must
  specify data window, regularization, and provenance instead of leaving the
  choice implicit.

## Recommended Implementation Order

1. Add red tests for the quanto contract template and compiler.
2. Implement the shared schema and validator.
3. Implement the checked-in quanto template and compiler output.
4. Wire `task_runtime`, `quant`, `planner`, and `codegen_guardrails` to the
   compiled contract path.
5. Add targeted `T105` validation fixtures and update docs for the next pricing
   tranche.
