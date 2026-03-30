# Phase C0 Implementation Plan: Contract Synthesis for Known Product Families

## Objective

Build the first checked-in contract-synthesis layer that sits between loose
task or term-sheet descriptions and the existing deterministic
`ProductIR -> quant -> generation_plan -> validation_bundle` path.

This tranche is successful if Trellis can:

- represent a known product family as a typed machine-readable contract
- validate that contract deterministically
- compile the validated contract into a bounded implementation blueprint
- do that first for a checked-in `quanto` proving family while keeping the same
  substrate ready for a later runtime-authored `Himalaya` request case

This tranche does **not** need to implement the final pricers.

## Why This Tranche Exists

Current FX vanilla work proved the platform can support a family once the
contract surface is explicit:

- `E25` and `T108` now succeed through a deterministic FX vanilla route
- `T105` still collapses to generic `european_option` handling
- the persisted `T105` run loses quanto-specific semantics such as FX vol,
  underlier/FX correlation, and domestic-vs-foreign payout structure

The same architectural gap will block future structured requests such as
Himalaya term sheets if Trellis keeps routing them as generic basket or vanilla
options.

## Current Relevant Surfaces

These files define the starting point and should be reviewed before coding:

- `trellis/agent/planner.py`
  - deterministic `SpecSchema` selection exists
  - specialization today is route-level (`FX vanilla`, `local vol`), not
    family-contract-level
- `trellis/agent/quant.py`
  - pricing plans are derived from canonical decompositions or `ProductIR`
  - there is no family-specific contract override layer
- `trellis/agent/codegen_guardrails.py`
  - route planning already compiles into primitives, adapters, and blockers
  - this is the natural downstream consumer of a contract blueprint
- `trellis/agent/validation_bundles.py`
  - validation bundles are explicit, but mostly method/instrument driven
- `trellis/agent/framework_runtime.py`
  - framework tasks already aggregate evidence from proving-ground tasks
  - this should later consume C0 outcomes when pricing tranches land
- `trellis/agent/task_runtime.py`
  - task titles currently map `quanto` to `european_option`
  - there is no `Himalaya` family recognition
- `trellis/agent/platform_requests.py`
  - term-sheet compilation currently goes straight from `TermSheet` to
    `ProductIR` or build routing
- `trellis/agent/term_sheet.py`
  - current term-sheet output is semi-structured and family-agnostic
- `trellis/agent/user_defined_products.py`
  - existing structured compilation path is a strong template for the C0
    compiler
- `trellis/agent/knowledge/decompose.py`
  - `ProductIR` is useful, but intentionally coarser than what quanto/Himalaya
    need
- `trellis/models/processes/correlated_gbm.py`
  - existing multi-asset path substrate for multi-factor or basket-style MC
- `TASKS.yaml`
  - `T105` is the immediate regression target
  - `T102` is the nearest existing reduction target for Himalaya semantics
- `FRAMEWORK_TASKS.yaml`
  - `E10`, `E16`, and `E17` are the most relevant downstream extraction tasks

## Proposed Architecture

### 1. Add a dedicated contract layer

Recommended new modules:

- `trellis/agent/family_contracts.py`
  - frozen dataclasses and enums for typed family contracts
- `trellis/agent/family_contract_validation.py`
  - deterministic structural and coherence validation
- `trellis/agent/family_contract_templates.py`
  - checked-in contract templates for the initial proving family set, starting
    with `quanto`
- `trellis/agent/family_contract_compiler.py`
  - validated contract -> `ProductIR` + pricing-plan hints + blueprint output

Keep the first implementation flat under `trellis/agent/` rather than adding a
new package tree. That keeps imports, tests, and future prompt guidance simple.

### 2. Keep `ProductIR` but do not overload it

`ProductIR` should remain the coarse routing abstraction.

The new contract layer should produce two outputs:

- a normalized `ProductIR` for compatibility with quant/retrieval paths
- a richer `FamilyImplementationBlueprint` for downstream planning

That prevents quanto and Himalaya semantics from being flattened away before
route selection and validation.

### 3. Make checked-in templates the first input source

Although the long-term goal is AI-drafted contracts, the first tranche should
use deterministic checked-in templates and fixtures. AI-authored drafts can be
added as an alternate input source once the validator and compiler are stable.

Important interpretation:

- checked-in templates are reference templates and regression fixtures
- they are not meant to replace runtime AI reasoning for real term sheets
- on real requests, the embedded AI should still:
  - infer or draft the family contract from the provided details
  - determine required vs optional inputs
  - resolve connectors and fetch data
  - estimate explicitly-derivable inputs when policy allows
  - select the best method supported by the available data
  - build the payoff, price it, and compute supported Greeks
  - cross-check with another method when a real second route exists

## Proposed Contract Schema Scope

The shared schema should cover six areas.

### Product semantics

Required fields:

- `family_id`
- `family_version`
- `instrument_aliases`
- `payoff_family`
- `exercise_style`
- `path_dependence`
- `schedule_semantics`
- `state_variables`
- `event_transitions`

### Market-data contract

Required fields:

- `required_inputs`
- `optional_inputs`
- `input_aliases`
- `bridging_notes`
- `derivable_inputs`
- `estimation_policy`
- `provenance_requirements`
- `missing_data_error_policy`

### Method contract

Required fields:

- `candidate_methods`
- `reference_methods`
- `production_methods`
- `unsupported_variants`
- `method_limitations`

### Sensitivity contract

Required fields:

- `support_level`
- `supported_measures`
- `stability_notes`

Validation must normalize this against current method-level support instead of
trusting optimistic claims.

### Validation contract

Required fields:

- `bundle_hints`
- `universal_checks`
- `family_checks`
- `comparison_targets`
- `reduction_cases`

### Blueprint hints

Required fields:

- `target_modules`
- `primitive_families`
- `adapter_obligations`
- `proving_tasks`
- `blocked_by`

## Runtime Market Binding Scope

The contract layer should not stop at static schema validation. It must carry
enough information for later runtime binding against real market-data
connectors.

For C0, the schema and blueprint should support these runtime obligations:

- connector resolution:
  - which contract inputs must be fetched from market-data connectors
  - which aliases are acceptable when connector field names differ
- derived-input policy:
  - which inputs may be estimated or derived from fetched data
  - which inputs are mandatory and must never be fabricated
- provenance:
  - whether an input is observed, derived, estimated, or user-supplied
- error handling:
  - meaningful, family-specific failures when a required input is missing
  - explicit messaging when a route is blocked because only weak estimated
    inputs are available

Mock task market specs remain useful, but only as proving-ground fixtures.
They are not the intended long-term source of contract detail for real runs.

## Implementation Discipline

Every C0 coding phase should follow the same execution discipline.

### Deterministic substrate changes: TDD

Use TDD for:

- schema dataclasses
- validators
- compiler outputs
- planner/quant/codegen deterministic routing
- validation-bundle selection

Required loop:

1. review previous work and current failing surface
2. review the phase plan and acceptance criteria
3. write or extend failing deterministic tests
4. implement the minimal code needed to make the tests pass
5. run targeted tests
6. update docs/plan notes if behavior or scope changed
7. summarize work, residual risks, and next phase

### Agent-facing behavior changes: EDD

Use eval-driven development for:

- term-sheet compilation behavior
- platform request routing
- connector-binding hint propagation
- honest block vs compile-only behavior
- future real-connector fetch/derive/error behavior

Required loop:

1. review prior traces, persisted task runs, and relevant eval fixtures
2. review the phase plan and expected runtime behavior
3. add or tighten eval-style fixtures and assertions first
4. implement the smallest routing/binding change that satisfies the eval
5. run targeted deterministic tests plus the new eval surface
6. update docs/roadmap if runtime expectations changed
7. summarize observed behavior, known gaps, and follow-up evals

## Standard Per-Phase Checklist

Every implementation phase should explicitly record:

- previous work reviewed
- plan section reviewed
- tests added first
- implementation files changed
- tests run and results
- docs updated
- concise summary of what is now supported vs still unsupported

## Deterministic Validation Scope

The validator should reject contracts that are syntactically valid but
architecturally dishonest.

Minimum checks:

- required sections and fields are present
- family ids and method labels normalize to known values
- schedule/state semantics agree with `path_dependence`
- unsupported variants are explicit instead of silently implied
- sensitivity support does not overstate current runtime guarantees
- blueprint hints reference real method families and plausible target modules
- family-specific required market data is present before a route is marked
  available
- any derivable or estimated inputs are explicitly marked as such
- error-policy rules exist for required inputs that cannot be fetched or
  derived safely

Minimum outputs:

- `ok`
- `errors`
- `warnings`
- `normalized_contract`

## Blueprint / Compiler Scope

The compiler should turn a validated family contract into a deterministic
implementation artifact that future agents can act on without rethinking the
architecture.

Recommended blueprint shape:

- `family_id`
- `product_ir`
- `preferred_method`
- `candidate_methods`
- `required_market_data`
- `derivable_market_data`
- `connector_binding_hints`
- `estimation_hints`
- `spec_schema_hint`
- `primitive_routes`
- `adapter_steps`
- `validation_bundle_hint`
- `target_modules`
- `proving_tasks`
- `unsupported_paths`

Compiler responsibilities:

- map contract semantics onto a normalized `ProductIR`
- preserve family-specific market-data requirements
- preserve runtime market-binding requirements and data provenance rules
- select bounded route candidates for quant and generation planning
- emit a deterministic spec/schema hint for planner integration
- emit explicit unsupported or blocked paths rather than degrading to generic
  fallback
- preserve enough route detail that the embedded AI still has to perform data
  binding, method selection, build, pricing, and Greek calculation at runtime

## Likely Module Ownership

- Library Developer:
  - `trellis/agent/family_contracts.py`
  - `trellis/agent/family_contract_validation.py`
  - `trellis/agent/family_contract_templates.py`
  - `trellis/agent/family_contract_compiler.py`
  - `trellis/agent/planner.py`
  - `trellis/agent/quant.py`
  - `trellis/agent/codegen_guardrails.py`
  - `trellis/agent/platform_requests.py`
  - `trellis/agent/task_runtime.py`
  - `trellis/agent/term_sheet.py`
- Test Agent:
  - `tests/test_agent/...`
- Knowledge Agent:
  - no required phase-one changes
  - optional follow-up only after family ids and routes stabilize

## Exact Test Plan

Add:

- `tests/test_agent/test_family_contracts.py`
  - schema construction
  - validator normalization
  - quanto template validation
  - blueprint compilation
  - explicit assertion that the checked-in template set is limited to `quanto`

Add:

- `tests/test_agent/test_platform_requests.py`
  - term-sheet request compilation through the family-contract path
  - honest block/compile-only behavior for unsupported variants
  - runtime connector-binding hints preserved in the compiled request

Extend:

- `tests/test_agent/test_planner.py`
  - contract-backed specialized schema selection
- `tests/test_agent/test_quant.py`
  - family contract -> pricing plan behavior
- `tests/test_agent/test_codegen_guardrails.py`
  - blueprint-driven route selection
- `tests/test_agent/test_validation_bundles.py`
  - family-specific validation bundle selection
- `tests/test_agent/test_task_runtime.py`
  - title-to-family mapping for `quanto`
- `tests/test_agent/test_term_sheet.py`
  - family-aware term-sheet parse fixtures
- future integration/eval surface:
  - real-connector tests should verify fetch/derive/error behavior separately
    from mock `TASKS.yaml` proving-ground fixtures

## Phased Execution Plan

### Phase C0.0: Baseline review and boundary freeze

Purpose:

- confirm what already exists
- freeze the exact initial checked-in slice for quanto
- freeze the future runtime-request boundary for Himalaya
- avoid drifting into premature pricing implementation

Review first:

- `AGENTS.md`
- `CLAUDE.md`
- `docs/autonomous_library_development_workstream.md`
- `docs/phase_contract_synthesis_quanto_himalaya_handoff.md`
- `docs/phase_c0_contract_synthesis_implementation_plan.md`
- `docs/phase_c0_quanto_contract_plan.md`
- `docs/phase_c0_himalaya_contract_plan.md`
- `task_runs/latest/T105.json`
- `task_runs/latest/T108.json`

Deliverables:

- exact initial family boundary for `quanto_option`
- exact future runtime-request boundary for `himalaya_option`
- explicit list of non-goals to keep out of C0

Validation:

- no code changes required
- update docs if boundaries changed

### Phase C0.1: Shared contract substrate

Method:

- TDD

Review first:

- `trellis/agent/user_defined_products.py`
- `trellis/agent/knowledge/decompose.py`
- `trellis/agent/planner.py`
- `trellis/agent/quant.py`

Write tests first:

- create `tests/test_agent/test_family_contracts.py`
- add schema-construction tests
- add validator normalization tests
- add invalid-contract rejection tests
- add compiler-output shape tests

Implement:

- `trellis/agent/family_contracts.py`
- `trellis/agent/family_contract_validation.py`
- `trellis/agent/family_contract_templates.py`
- `trellis/agent/family_contract_compiler.py`

Targeted validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_family_contracts.py -q
```

Documentation step:

- update the main C0 plan if schema fields or blueprint shape changed

Summary expected at phase end:

- what the shared schema can now express
- what contracts now validate
- what runtime AI still needs to infer/bind later

### Phase C0.2: Quanto family compilation path

Method:

- TDD for contract/route logic
- EDD for request-routing behavior

Review first:

- `trellis/agent/task_runtime.py`
- `trellis/agent/planner.py`
- `trellis/agent/quant.py`
- `trellis/agent/codegen_guardrails.py`
- `trellis/agent/validation_bundles.py`
- `task_runs/latest/T105.json`
- `task_runs/latest/T108.json`

Write tests first:

- extend `tests/test_agent/test_task_runtime.py`
- extend `tests/test_agent/test_planner.py`
- extend `tests/test_agent/test_quant.py`
- extend `tests/test_agent/test_codegen_guardrails.py`
- extend `tests/test_agent/test_validation_bundles.py`
- create or extend `tests/test_agent/test_platform_requests.py`
- extend `tests/test_agent/test_term_sheet.py`

Implement:

- task-title or request-family recognition for `quanto_option`
- contract-backed spec-schema selection
- quant/compiler route integration
- connector-binding hint propagation
- family-specific validation-bundle selection

Targeted validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_family_contracts.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_planner.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_codegen_guardrails.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_validation_bundles.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_term_sheet.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_requests.py -q
```

Documentation step:

- update the quanto plan with any changed connector-binding or cross-check
  expectations

Summary expected at phase end:

- what `T105`-style requests now preserve
- what data must still be connector-bound at runtime
- what remains unsupported

### Phase C0.3: Future Himalaya runtime-request path

Method:

- TDD for contract/state-machine/route logic
- EDD for term-sheet/request behavior

Review first:

- `trellis/agent/task_runtime.py`
- `trellis/agent/platform_requests.py`
- `trellis/agent/term_sheet.py`
- `trellis/agent/planner.py`
- `trellis/agent/quant.py`
- `trellis/agent/codegen_guardrails.py`
- `trellis/models/processes/correlated_gbm.py`
- `TASKS.yaml` entry for `T102`

Write tests first:

- extend `tests/test_agent/test_task_runtime.py`
- extend `tests/test_agent/test_term_sheet.py`
- extend `tests/test_agent/test_planner.py`
- extend `tests/test_agent/test_quant.py`
- extend `tests/test_agent/test_codegen_guardrails.py`
- extend `tests/test_agent/test_validation_bundles.py`
- create or extend `tests/test_agent/test_platform_requests.py`

Implement:

- family recognition for `himalaya_option`
- contract-backed schema selection
- MC-first route compilation with state-machine hints
- connector-binding hints for constituent-level market data
- honest block/compile-only behavior for unsupported wrappers

Targeted validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_family_contracts.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_term_sheet.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_planner.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_codegen_guardrails.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_validation_bundles.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_requests.py -q
```

Documentation step:

- update the Himalaya plan with the exact canonical slice and unsupported
  wrappers

Summary expected at phase end:

- what Himalaya semantics now compile correctly
- what runtime market binding still needs real connector work
- which cross-checks are reductions only vs genuine second routes

### Phase C0.4: Runtime binding and eval preparation

Method:

- EDD

Purpose:

- prepare the runtime layer for real connector-backed requests without claiming
  full production readiness

Review first:

- connector access surfaces already in repo
- platform request compilation path
- mock-task limitations documented in the family plans

Write tests/evals first:

- add eval-style fixtures for:
  - connector-binding hints preserved
  - observed vs derived vs estimated provenance
  - meaningful missing-data failures
  - honest block vs compile-only behavior

Implement:

- no full pricing-engine implementation required here
- only the request/contract/runtime-binding surface needed for future real
  connector runs

Targeted validation:

- run the targeted agent tests above
- add any new eval command once the eval surface exists

Documentation step:

- update roadmap and family plans with what real-runtime work remains after C0

Summary expected at phase end:

- what is ready for later connector-backed execution
- what still depends on later pricing-engine work
- `tests/test_agent/test_user_defined_products.py`
  - reuse or convergence between structured-user-product compilation and family
    contract compilation

## Recommended Implementation Order

### C0.1 Shared contract substrate

Implement first:

- schema dataclasses
- validator
- checked-in family templates
- compiler output dataclass
- red tests in `test_family_contracts.py`

### C0.2 Quanto path

Implement next because it has the clearest proving task and immediate roadmap
value:

- task/runtime family detection for `T105`
- contract compiler integration into quant/planner/generation
- validation bundle hints for quanto

### C0.3 Himalaya path

Implement after the shared substrate and quanto path:

- runtime-authored family recognition from term-sheet and build requests
- MC-only blueprint compilation from a draft family contract
- honest blocking of unsupported note wrappers or autocall variants

### C0.4 Platform integration and docs

Finish by wiring:

- `platform_requests`
- `term_sheet`
- roadmap references
- acceptance-test documentation for the next implementation tranche

## Acceptance Criteria

This tranche is complete when:

- Trellis has a checked-in typed family contract for `quanto`
- contract validation rejects incoherent or overclaimed family drafts
- a validated contract compiles into a deterministic blueprint
- `T105`-style requests preserve quanto semantics instead of degrading to
  generic `european_option`
- the shared substrate is explicit enough that a later runtime-authored
  Himalaya contract can compile into a bounded MC-first blueprint or an honest
  blocked state
- the blueprint explicitly preserves what the runtime AI still needs to do:
  connector resolution, allowable derivations, method selection, build, price,
  and supported Greeks
- the resulting artifacts tell a future agent exactly what modules, tests, and
  validation bundles come next

## Explicit Non-Goals

- shipping final quanto pricing kernels in this tranche
- shipping final Himalaya pricing kernels in this tranche
- solving full free-form term-sheet extraction for all structured notes
- updating knowledge YAML before the family ids and compiler surface stabilize

## Risks and Open Questions

- `MarketState` does not yet expose every likely quanto or Himalaya input as a
  clean first-class field; the contract layer must be honest about aliases and
  unsupported inputs.
- `ProductIR` may need a small compatibility extension, but the first choice
  should be keeping family detail in the blueprint instead.
- Himalaya has many commercial variants; the first runtime-request path must
  define a narrow canonical slice and reject the rest.
- The platform needs a decision on whether unsupported family requests should
  return `block` or `compile_only`; the docs below recommend this per family.

## Recommended Next Phase After These Docs

Once the docs are checked in, the next coding tranche should start with:

1. `tests/test_agent/test_family_contracts.py`
2. shared contract substrate modules
3. the quanto path tied to `T105`
4. the future Himalaya runtime-request path
