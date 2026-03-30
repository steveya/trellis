# Phase C2 Implementation Plan: Family-Name-Free Semantic Product Synthesis

## Goal

Enable an agent to synthesize a structured path-dependent basket product from
reusable semantics, without requiring an explicit named family such as
`himalaya_option` in the codebase.

The proving case is still a basket-shaped payoff, but the architecture goal is
larger: the same semantic path should stay family-name-free as new derivative
shapes are added.

The first full proving-run writeup for the basket case lives in
`docs/qua-284-arbitrary-derivative-proving-run.md`.

This tranche is successful if Trellis can take a request like:

- "Himalaya option on these equities"
- or a term-sheet-style description of a mountain-range payoff

and translate it into:

- a generic semantic contract
- a deterministic implementation blueprint
- a bounded Monte Carlo route over shared primitives
- a thin generated payoff/state-machine layer

without introducing a product-name-specific branch in runtime code.

## Why This Tranche Exists

The long-term Trellis claim is stronger than "we can add another named exotic
family." The real claim is:

- the codebase contains reusable pricing pieces
- an agent can compose those pieces into a novel product
- product names are only request language, not architecture

For mountain-range products, that means the codebase should know:

- multi-asset correlated path simulation
- observation schedules
- ranking/selection over remaining assets
- state transitions such as remove/lock
- payoff accumulation and maturity settlement

It should not need to know the literal word `Himalaya`.

## Non-Goals

This tranche does not attempt to:

- support every mountain-range commercial wrapper
- support autocallables, coupon notes, principal protection, or callable
  redemption state machines
- create a full symbolic payoff algebra
- guarantee that arbitrary free-form requests compile successfully

The first goal is one canonical mountain-range-style request path that is
expressed entirely through reusable semantics.

## Hard Architecture Boundary

Do not add new runtime branches keyed on product-family names such as:

- `himalaya_option`
- `atlas_option`
- `everest_option`
- `altiplano_option`
- `annapurna_option`

Permitted usage of those names:

- docs
- natural-language task fixtures
- blog/demo traces

Not permitted as the architectural boundary:

- planner specialization keys
- runtime contract `family_id`
- validator branches
- compiler branches
- codegen guardrail branches

Instead, the runtime boundary should be semantic and compositional.

## Target Semantic Representation

Add a richer intermediate layer above coarse `ProductIR`.

Recommended name:

- `SemanticProductContract`

Recommended sections:

### Product semantics

- `instrument_class`
  - e.g. `basket_path_payoff`
- `exercise_style`
  - likely `european`
- `state_dependence`
  - `path_dependent`
- `schedule_dependence`
  - `True`
- `underlier_kind`
  - `equity_basket`

### Observation semantics

- `observation_schedule`
- `observation_basis`
- `observation_operator`
  - e.g. `simple_return_from_start`

### Selection semantics

- `selection_operator`
  - e.g. `argmax_remaining`
- `selection_scope`
  - `remaining_constituents`
- `selection_count`
  - usually `1`

### State-machine semantics

- `state_variables`
  - e.g. `remaining_names`, `locked_returns`
- `transition_rules`
  - e.g. `remove_selected_name`
- `transition_order`

### Payoff semantics

- `locked_value_transform`
- `aggregation_operator`
  - e.g. `sum`, `average`
- `maturity_settlement_rule`

### Market-data semantics

- `constituents`
- `spots`
- `vols`
- `carry_inputs`
- `discount_curve`
- `correlation_matrix`
- provenance / estimation policy

### Method semantics

- `candidate_methods`
  - initially `("monte_carlo",)`
- `required_primitives`
- `blocked_variants`

## Current Repo Surfaces To Review First

- `trellis/agent/knowledge/decompose.py`
  - current `ProductIR` is intentionally coarse and does not encode
    observation-state machines
- `trellis/agent/platform_requests.py`
  - likely entry point for runtime-authored semantic requests
- `trellis/agent/term_sheet.py`
  - current request extraction is too family-agnostic and too shallow
- `trellis/agent/planner.py`
  - current `basket_option` schema is too weak for ranked path state machines
- `trellis/agent/quant.py`
  - current method routing is family/method oriented, not semantic-contract
    oriented
- `trellis/agent/codegen_guardrails.py`
  - current guardrails know about generic MC path generation, not ranked basket
    state machines
- `trellis/models/processes/correlated_gbm.py`
  - current shared correlated path substrate
- `trellis/models/monte_carlo/engine.py`
  - current shared MC engine substrate
- `trellis/core/payoff.py`
  - current adapter bases
- `trellis/models/resolution/`
  - current family-specific resolver direction; this tranche should move toward
    semantic resolvers

## Proposed Generic Runtime Substrate

The codebase should contain reusable semantic primitives, not product-name
handlers.

### 1. Basket path state substrate

Recommended new module:

- `trellis/models/monte_carlo/basket_state.py`

Responsibilities:

- constituent vector assembly
- observation-date slicing
- ranking/selecting constituents
- remaining/locked index bookkeeping
- transition helpers

Example primitive responsibilities:

- `extract_observation_states(paths, observation_indices)`
- `rank_remaining_constituents(observation_slice, remaining_mask, rule)`
- `apply_selection_transition(state, selected_idx, transition_rule)`

### 2. Generic ranked-observation payoff substrate

Recommended new module:

- `trellis/models/monte_carlo/ranked_observation_payoffs.py`

Responsibilities:

- compute observation return metric
- apply selection operator
- lock/remove selected constituents
- accumulate payoff state
- finalize maturity payoff

This module should be parameterized by semantics, not by named product family.

### 3. Semantic market binding

Recommended new module:

- `trellis/models/resolution/basket_semantics.py`

Responsibilities:

- resolve constituent spots
- resolve vol vectors
- resolve carry inputs
- resolve discounting
- resolve correlation matrix
- label observed vs derived vs estimated inputs

This is the semantic analogue of the current `quanto` resolver.

### 4. Semantic MC route helper

Recommended new module:

- `trellis/models/monte_carlo/semantic_basket.py`

Responsibilities:

- bind semantic contract + resolved inputs + shared MC engine
- construct correlated path process
- pass paths through ranked-observation primitives
- return PV

This is the key module that lets the agent generate only a thin adapter or only
the novel payoff logic.

## Implementation Phases

### Phase C2.0: Freeze One Canonical Mountain-Range Variant

Define one canonical request semantics for the proving task:

- basket of equities
- fixed ordered observation dates
- at each date, choose the best performer among remaining constituents
- remove selected name from remaining set
- lock selected simple return
- final payoff is average locked return times notional, paid at maturity

Important:

- the docs may call this "Himalaya-style" for explanation
- the runtime contract must not

Files to change:

- this plan
- roadmap note
- future task fixture in `TASKS.yaml`

Tests first:

- semantic request fixture with explicit expected semantic contract

Acceptance:

- one frozen canonical semantic target exists

Status:

- checked in as the ranked-observation basket semantic contract in `trellis/agent/semantic_contracts.py`

### Phase C2.1: Rich Semantic Contract Layer

Add a generic semantic contract representation and validator.

Recommended modules:

- `trellis/agent/semantic_contracts.py`
- `trellis/agent/semantic_contract_validation.py`
- `trellis/agent/semantic_contract_compiler.py`

Do not branch on named families. Validate by semantic shape:

- path dependence present
- ordered observation schedule
- multi-asset constituents
- selection operator coherent
- transition rules coherent
- MC-only route required when ranked path state is present

Tests first:

- `tests/test_agent/test_semantic_contracts.py`

Suggested tests:

- `test_ranked_observation_basket_contract_validates`
- `test_contract_rejects_missing_observation_schedule`
- `test_contract_rejects_selection_without_remaining_scope`
- `test_contract_requires_correlation_for_multi_asset_mc`

Acceptance:

- runtime-authored semantic contract validates without any named-family key

Status:

- checked in with deterministic validation and blueprint compilation in
  `trellis/agent/semantic_contract_validation.py` and
  `trellis/agent/semantic_contract_compiler.py`

### Phase C2.2: Semantic Request Drafting

Teach the request path to draft the semantic contract from natural language or
term-sheet-like input.

Files to change:

- `trellis/agent/platform_requests.py`
- `trellis/agent/term_sheet.py`
- `trellis/agent/task_runtime.py`

Behavior:

- detect semantic cues:
  - basket
  - observation dates
  - best/worst remaining
  - remove/lock
  - maturity aggregation
- produce semantic contract fields, not `family_id`

Tests first:

- `tests/test_agent/test_platform_requests.py`
- `tests/test_agent/test_task_runtime.py`

Suggested tests:

- `test_mountain_range_request_drafts_ranked_observation_contract`
- `test_himalaya_named_request_does_not_require_named_family_branch`
- `test_request_missing_schedule_returns_semantic_error`

Acceptance:

- a natural-language Himalaya-style request produces a semantic contract
- no new named family mapping is added

Status:

- checked in through semantic request drafting in
  `trellis/agent/platform_requests.py`, `trellis/agent/term_sheet.py`, and
  `trellis/agent/task_runtime.py`

### Phase C2.3: Generic Basket-State MC Primitives

Implement the reusable MC state-machine substrate.

Files to add:

- `trellis/models/monte_carlo/basket_state.py`
- `trellis/models/monte_carlo/ranked_observation_payoffs.py`
- `trellis/models/resolution/basket_semantics.py`
- `trellis/models/monte_carlo/semantic_basket.py`

Tests:

- `tests/test_models/test_monte_carlo/test_basket_substrate.py`
- `tests/test_models/test_resolution.py`
- `tests/test_models/test_monte_carlo/test_himalaya.py`

Suggested tests:

- `test_ranked_observation_basket_state_replays_locked_returns_from_snapshots`
- `test_ranked_observation_basket_price_helper_uses_snapshot_state_requirement`
- `test_resolve_basket_semantics_matches_expected_market_binding`

Acceptance:

- one canonical ranked-observation payoff can be priced from semantics with
  shared MC primitives and no named product module

Status:

- checked in through `trellis/models/resolution/basket_semantics.py`,
  `trellis/models/monte_carlo/basket_state.py`,
  `trellis/models/monte_carlo/ranked_observation_payoffs.py`, and
  `trellis/models/monte_carlo/semantic_basket.py`
- task runs now carry a runtime contract payload with snapshot references,
  evaluation tags, and trace identifiers so replay artifacts stay aligned with
  the semantic request that produced them

### Phase C2.4: Planner / Quant / Guardrail Integration

Compile the semantic contract into a bounded implementation blueprint.

Files to change:

- `trellis/agent/planner.py`
- `trellis/agent/quant.py`
- `trellis/agent/codegen_guardrails.py`
- `trellis/agent/prompts.py`
- `trellis/agent/executor.py`

Behavior:

- planner emits a semantic ranked-basket spec/schema rather than generic
  `basket_option`
- quant chooses MC-only
- guardrails require:
  - correlated basket paths
  - semantic basket resolver
  - ranked observation state-machine helpers
- prompts instruct the model to generate only the novel layer

Tests first:

- `tests/test_agent/test_planner.py`
- `tests/test_agent/test_quant.py`
- `tests/test_agent/test_codegen_guardrails.py`
- `tests/test_agent/test_prompts.py`

Suggested tests:

- `test_semantic_ranked_basket_request_uses_mc_only_route`
- `test_generation_plan_reuses_semantic_basket_helper_surface`
- `test_prompt_for_ranked_basket_forbids_open_coding_mc_infrastructure`

Acceptance:

- generated code is bounded to thin semantic adapter logic

Status:

- checked in through `trellis/agent/planner.py`, `trellis/agent/quant.py`,
  `trellis/agent/codegen_guardrails.py`, `trellis/agent/prompts.py`, and
  `trellis/agent/executor.py`
- the representative-derivative regression matrix is now checked in as
  `QUA-333`, covering basket, quanto, callable bond, vanilla option, and
  rate-style swaption controls without leaking into the basket-specific route

### Phase C2.5: Proving Task and Full Trace

Add one proving task whose title may say "Himalaya" but whose runtime path is
purely semantic.

Files to change:

- `TASKS.yaml`
- possibly `FRAMEWORK_TASKS.yaml`

Run:

- `python scripts/run_tasks.py --fresh-build --validation standard <task> <task>`

Required artifacts:

- task result JSON
- platform build traces
- generated module under `_fresh`
- deterministic validation output
- promotion candidate if successful

Acceptance:

- one full traced run exists showing semantic synthesis from request to price
- the trace does not rely on a `himalaya_option` branch in runtime code

Status:

- this is now the docs/knowledge hardening step in `QUA-334`
- the roadmap and knowledge surfaces should keep the distinction clear between
  semantic understanding, method arbitration, and numerical pricing
- novel unsupported requests now emit structured `semantic_gap` metadata on
  fallback compilation paths (`QUA-376`)

## Exact Test Files To Add or Extend

Add:

- `tests/test_agent/test_semantic_contracts.py`
- `tests/test_models/test_basket_state.py`
- `tests/test_models/test_ranked_observation_payoffs.py`
- `tests/test_models/test_basket_semantics_resolution.py`

Extend:

- `tests/test_agent/test_platform_requests.py`
- `tests/test_agent/test_task_runtime.py`
- `tests/test_agent/test_planner.py`
- `tests/test_agent/test_quant.py`
- `tests/test_agent/test_codegen_guardrails.py`
- `tests/test_agent/test_prompts.py`

## Risks

### Risk: Semantic contract becomes a hidden family system

If the semantic contract ends up hardcoding one exact state-machine pattern, it
will just be `himalaya_option` under another name.

Mitigation:

- name and validate by semantic primitives
- keep operators generic

### Risk: Too much freedom for the agent

If the semantic scaffold is too loose, the agent will re-invent the MC
infrastructure again.

Mitigation:

- keep the shared semantic-basket helper large enough that generated code stays
  thin

### Risk: Over-generalizing before first proof

Trying to support all mountain-range variants first will stall the proving
task.

Mitigation:

- one canonical variant first
- prove the trace
- then widen operators and wrappers

## Recommended Implementation Order

1. freeze the canonical semantic variant
2. add semantic contract + validator + compiler
3. teach request drafting to emit semantics instead of family names
4. add generic basket-state MC primitives
5. integrate planner/quant/guardrails/prompts
6. add one proving task and produce a full fresh-build trace
7. harden docs, knowledge, and roadmap notes so the next agent can continue
   without re-deriving the plan

## Next-Agent Summary

The next agent should aim for this explicit success condition:

- a request may contain the word "Himalaya"
- the runtime should not
- the runtime should synthesize a semantic ranked-observation basket contract
- shared correlated-path and state-machine primitives should carry almost all of
  the implementation burden
- generated code should be only the thin semantic payoff layer

That is the shortest path from the current `quanto` proof to the blog’s
stronger claim about decomposability.
