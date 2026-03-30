# Phase C0 Detailed Plan: Future Himalaya Runtime-Request Path

## Goal

Define the first deterministic family-contract path for `Himalaya` products so
future live requests can compile a runtime-authored contract into a bounded
MC-first blueprint instead of falling back to a generic basket option build.

Important scope boundary:

- do not add a checked-in `himalaya_option` template in the current tranche
- reserve `Himalaya` as a future request case that exercises the embedded AI's
  ability to draft the family contract from provided details

Initial scope is intentionally narrow:

- classic observation-schedule Himalaya payoff semantics
- selection and lock/remove state transitions are explicit
- Monte Carlo is the only supported initial method family
- the first outcome may be `compile_only` or honest block for unsupported note
  wrappers

Out of scope for this phase:

- autocallable wrappers
- coupon-bearing structured notes unless explicitly reduced to a core payoff
- principal guarantees
- callable redemption state machines
- final pricing implementation

## Current Repo Surfaces

Inspect first:

- `trellis/agent/task_runtime.py`
  - recognizes `basket`, `best-of`, `worst-of`, and `rainbow`
  - does not recognize `Himalaya`
- `trellis/agent/platform_requests.py`
  - future term-sheet requests will enter here
- `trellis/agent/term_sheet.py`
  - current prompt does not ask for Himalaya-specific state semantics
- `trellis/agent/knowledge/decompose.py`
  - current `ProductIR` derivation has no notion of observation-state machines
- `trellis/agent/planner.py`
  - generic `basket_option` spec exists but is too weak for Himalaya semantics
- `trellis/agent/quant.py`
  - no family-level MC-only override for Himalaya
- `trellis/agent/codegen_guardrails.py`
  - has generic `monte_carlo_paths` only
- `trellis/agent/validation_bundles.py`
  - no family-specific path-state checks
- `trellis/models/processes/correlated_gbm.py`
  - reusable substrate for multi-asset correlated MC
- `TASKS.yaml`
  - no direct Himalaya task yet
  - `T102` rainbow is the closest reduction target
- `FRAMEWORK_TASKS.yaml`
  - `E10` multi-asset MC framework is the nearest downstream extraction task

## Proposed Contract Scope

Recommended `family_id`:

- `himalaya_option`

Recommended initial canonical slice:

- basket of risky assets
- fixed ordered observation schedule
- at each observation, choose the best performer among remaining assets
- selected asset is locked or removed from the remaining basket
- final payoff aggregates locked observations under a single maturity payment

Required semantics:

- `constituents`
- `observation_schedule`
- `selection_rule`
- `lock_rule`
- `payoff_aggregation`
- `maturity_redemption_rule`
- `state_variables`
- `event_transitions`

Required market-data inputs:

- per-name spots
- per-name vols or a volatility surface abstraction
- per-name dividends/carry
- correlation matrix
- discount curve

Optional inputs:

- FX data for later cross-currency extensions
- local vol or stochastic-vol inputs for later variants

Derived-input policy:

- per-name carry inputs may be bridged from connector dividend/borrow fields
  when that mapping is explicit
- correlation may be estimated only under an explicit runtime policy with
  provenance
- the runtime must not silently fabricate a correlation matrix for a true
  multi-asset MC route

Method contract:

- reference methods:
  - none in phase one
- supported methods:
  - `monte_carlo`
- unsupported methods:
  - `analytical`
  - `rate_tree`
  - `pde_solver`

Sensitivity contract:

- initial level:
  - `bump_only`
- note:
  - stability should be marked low/experimental when the eventual pricing path
    lands

## Deterministic Validation Scope

Structural checks:

- observation schedule is non-empty and ordered
- at least two constituents are present
- selection rule references the remaining basket, not the full basket after
  removals
- state variables and event transitions are explicit

Coherence checks:

- `path_dependence` is required
- schedule dependence is required
- method contract is MC-only
- lock semantics and payoff aggregation agree

Unsupported-claim checks:

- reject autocallable or coupon note claims in the initial C0 slice
- reject missing correlation input for a multi-asset MC route
- reject analytical or tree-based claims

Runtime-binding checks:

- the contract must preserve constituent identifiers needed for connector lookup
- the contract must distinguish observed vs derived vs estimated inputs
- missing required inputs must fail with a family-specific message naming the
  missing constituent field

Reduction checks:

- single-observation Himalaya should reduce to a best-of/rainbow-style basket
  semantics
- identical-asset fixtures should not produce contradictory state-machine
  behavior

## Blueprint / Compiler Scope

Required blueprint contents:

- `family_id = "himalaya_option"`
- normalized `product_ir`
- MC-only pricing-plan hint
- required market-data vector semantics
- connector-binding hints for constituent-level data resolution
- derivation/provenance rules for correlation and carry inputs
- state-machine summary
- primitive route hints
- adapter obligations
- proving tasks or reduction targets
- unsupported wrappers

Recommended route hints:

- reuse `CorrelatedGBM` for basket paths
- add explicit adapter steps for:
  - constituent vector assembly
  - observation-date slicing
  - winner-selection state machine
  - locked-payoff accumulation

## Runtime Market Binding Expectations

`TASKS.yaml` does not yet contain a direct Himalaya proving task, and any future
mock task should still be treated as a regression fixture rather than the real
runtime interface.

For real Himalaya requests, the embedded AI should be able to:

- read the full contract or term sheet, including basket names and observation
  schedule
- resolve connector data for each constituent:
  - spot
  - vol
  - carry/dividend inputs
  - identifier normalization
- resolve discounting and correlation inputs
- use `CorrelatedGBM` when the correlation-aware MC route is available
- fail honestly with a useful message if any required constituent data is
  unavailable

If correlation is not directly supplied, the runtime may later support a
best-effort estimation policy, but that policy must be explicit. At minimum it
should define:

- the source data window
- matrix regularization / PSD repair
- provenance labeling as estimated rather than observed

The initial C0 docs should not imply that a production-quality Himalaya pricer
can avoid correlation entirely.

## Cross-Check Expectations

Himalaya should use a staged cross-check hierarchy.

Near-term:

- reduction checks:
  - one observation should reduce toward best-of/rainbow semantics
  - degenerate equal-name cases should behave predictably
- invariant checks:
  - locked-state progression
  - schedule monotonicity

Later, once knowledge accumulates and additional routes exist:

- independent MC variants
- QMC / variance-reduced MC
- family-specific control variates or reductions

Likely target modules for the later implementation tranche:

- `trellis/agent/codegen_guardrails.py`
  - new multi-asset schedule/state-machine route hints
- `trellis/agent/planner.py`
  - contract-backed `HimalayaOptionSpec`
- `trellis/models/processes/correlated_gbm.py`
  - inspect before adding any new joint process type
- future likely new module:
  - `trellis/models/monte_carlo/himalaya.py`

## Likely Files To Change

Phase-one contract tranche:

- `trellis/agent/family_contracts.py`
- `trellis/agent/family_contract_validation.py`
- `trellis/agent/family_contract_compiler.py`
- `trellis/agent/task_runtime.py`
- `trellis/agent/platform_requests.py`
- `trellis/agent/term_sheet.py`
- `trellis/agent/planner.py`
- `trellis/agent/quant.py`
- `trellis/agent/codegen_guardrails.py`
- `trellis/agent/validation_bundles.py`

Likely later implementation tranche:

- `trellis/models/monte_carlo/himalaya.py`
- `trellis/models/processes/correlated_gbm.py`

## Exact Test Files To Add Or Extend

Add:

- `tests/test_agent/test_family_contracts.py`
  - `test_runtime_drafted_himalaya_contract_validates`
  - `test_runtime_drafted_himalaya_contract_rejects_autocall_wrapper`
  - `test_runtime_drafted_himalaya_contract_compiles_to_mc_only_blueprint`

Add:

- `tests/test_agent/test_platform_requests.py`
  - `test_himalaya_term_sheet_compiles_to_contract_blueprint`
  - `test_unsupported_himalaya_note_wrapper_returns_block_or_compile_only`
  - `test_himalaya_compiled_request_preserves_connector_binding_hints`

Extend:

- `tests/test_agent/test_task_runtime.py`
  - assert Himalaya titles map to `himalaya_option`
- `tests/test_agent/test_term_sheet.py`
  - parse a simple Himalaya term-sheet fixture into structured fields
- `tests/test_agent/test_quant.py`
  - assert Himalaya contract selects MC-only routing and vector market data
- `tests/test_agent/test_planner.py`
  - assert a Himalaya request chooses a contract-backed schema
- `tests/test_agent/test_codegen_guardrails.py`
  - assert the route references `CorrelatedGBM` and state-machine adapters
- `tests/test_agent/test_validation_bundles.py`
  - assert Himalaya family checks include reduction/schedule sanity hooks
- later integration/eval surface:
  - verify constituent lookup, correlation estimation policy, and meaningful
    missing-data errors on non-mock requests

Optional follow-up once a framework task exists:

- `tests/test_agent/test_framework_runtime.py`
  - tie future Himalaya extraction work to `T102` plus new proving tasks

## Phased Implementation Plan

### H0: Review and canonical-slice freeze

Method:

- review only

Checklist:

1. review the main C0 plan and this Himalaya plan
2. review current basket/rainbow surfaces and `T102`
3. freeze the exact initial canonical slice:
   - observation schedule
   - winner-selection rule
   - lock/remove semantics
   - maturity aggregation
4. list excluded wrappers before coding

Expected summary:

- one canonical Himalaya variant chosen
- unsupported wrappers called out explicitly

### H1: Himalaya validator rules and inline draft fixtures

Method:

- TDD

Write tests first:

- `test_runtime_drafted_himalaya_contract_validates`
- `test_runtime_drafted_himalaya_contract_rejects_autocall_wrapper`
- `test_runtime_drafted_himalaya_contract_rejects_missing_correlation_for_mc_route`
- `test_runtime_drafted_himalaya_contract_requires_ordered_observation_schedule`

Implement:

- Himalaya validation rules in `trellis/agent/family_contract_validation.py`
- inline or fixture-backed Himalaya draft construction in tests and request
  fixtures, not in `trellis/agent/family_contract_templates.py`

Run:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_family_contracts.py -q
```

Docs:

- update this file if the canonical slice changes

Expected summary:

- validated runtime-authored Himalaya contract exists
- state-machine and data-binding requirements are explicit

### H2: Himalaya compiler and request-routing integration

Method:

- TDD for deterministic route/state-machine outputs
- EDD for term-sheet and request behavior

Write tests first:

- `test_himalaya_contract_compiles_to_mc_only_blueprint`
- `test_himalaya_title_maps_to_himalaya_option`
- `test_himalaya_term_sheet_compiles_to_contract_blueprint`
- `test_himalaya_request_uses_contract_backed_schema`
- `test_himalaya_compiled_request_preserves_connector_binding_hints`

Implement:

- compile validated Himalaya contract into MC-first blueprint
- recognize Himalaya requests before generic basket fallback
- propagate constituent-level connector-binding hints

Run:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_family_contracts.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_term_sheet.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_planner.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_requests.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_codegen_guardrails.py -q
```

Docs:

- update unsupported-wrapper guidance if compile-only vs block behavior changes

Expected summary:

- Himalaya no longer routes as a plain basket option
- MC/state-machine route is explicit without claiming final pricing support

### H3: Himalaya validation bundles and reduction-based cross-checks

Method:

- TDD for family checks
- EDD for honest runtime behavior

Write tests first:

- reduction to best-of/rainbow semantics for one-observation cases
- locked-state progression checks
- meaningful missing-data errors for constituent-level binding

Implement:

- Himalaya family validation-bundle selection
- reduction-case hints in the compiled blueprint
- honest block or compile-only behavior for unsupported wrappers

Run:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_validation_bundles.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_requests.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_term_sheet.py -q
```

Docs:

- update the cross-check section if new reduction rules are added

Expected summary:

- what can now be validated deterministically for Himalaya
- what later pricing and connector work still remains

## Acceptance Criteria

The Himalaya contract path is complete when:

- a runtime-authored `himalaya_option` contract can validate and compile
- a Himalaya-style request compiles into a deterministic MC-first blueprint
- the blueprint preserves:
  - observation schedule semantics
  - selection rule
  - lock/remove semantics
  - state variables and transitions
- the blueprint tells the runtime AI how to bind constituent-level market data
  and how estimated inputs must be labeled
- the platform no longer routes Himalaya as a plain basket option
- unsupported structured-note wrappers are explicitly blocked or compile-only
- the route definition is explicit enough that a later runtime layer can fetch
  basket data, use `CorrelatedGBM`, and add richer cross-checks as new methods
  become available

## Risks / Open Questions

- "Himalaya" is not a single market-standard product; the first runtime
  contract slice must name one canonical payoff slice and reject the rest.
- The term-sheet parser may not reliably extract state-machine semantics in the
  first pass. C0 should allow a compile-only path from partial structured data
  rather than forcing full build readiness.
- The current market substrate may not yet have a clean first-class home for
  per-name vectors and correlation matrices in every execution path.
- If best-effort correlation estimation is later allowed, the policy needs to
  be deterministic enough for audit and rerun reproducibility.

## Recommended Implementation Order

1. Choose and document the canonical initial Himalaya variant.
2. Add red tests for the contract template, validator, and compiler.
3. Implement the shared schema/validator/template/compiler path.
4. Wire `term_sheet` and `platform_requests` so future Himalaya requests hit
   the family-contract path first.
5. Add MC route and validation-bundle hooks, then document the follow-on
   implementation tranche.
