# Phase C1 Implementation Plan: Analytical Support Substrate

## Goal

Build the next reusable substrate tranche for analytical methods so the agent
stops open-coding Trellis-specific analytical plumbing and instead assembles
pricing routes from resolved inputs plus checked-in support kernels.

This tranche is the follow-on to the `quanto` proving work in `C0`.

The concrete objective is:

- make analytical implementation look more like assembly than translation
- let analytical engines call reusable subproblem kernels, not only top-level
  closed forms
- reduce model sensitivity on fresh-build routes by shrinking the amount of
  adapter and formula glue the agent must invent

Current checked-in progress:

- the first analytical support package now exists under
  `trellis/models/analytical/support/`
- the initial stable helper slice is implemented and exported:
  - discounting and rate transforms:
    - `implied_zero_rate`
    - `discount_factor_from_zero_rate`
    - `discounted_value`
    - `safe_time_fraction`
    - `continuous_rate_from_simple_rate`
    - `simple_rate_from_discount_factor`
    - `forward_discount_ratio`
  - forward builders:
    - `forward_from_discount_factors`
    - `forward_from_carry_rate`
    - `forward_from_dividend_yield`
  - payoff transforms:
    - `normalized_option_type`
    - `terminal_intrinsic`
    - `cash_or_nothing_intrinsic`
    - `asset_or_nothing_intrinsic`
    - `call_put_parity_gap`
  - cross-asset transforms:
    - `quanto_adjusted_forward`
    - `effective_covariance_term`
    - `exchange_option_effective_vol`
    - `foreign_to_domestic_forward_bridge`
- `trellis.models.analytical.quanto` now consumes the support layer directly
  instead of normalizing option type locally, and `quanto_adjusted_forward`
  now composes the smaller cross-asset bridge helpers rather than open-coding
  the covariance term
- prompt guidance for analytical `quanto` fresh-build routes now explicitly
  steers the model toward `trellis.models.analytical.support`
- deterministic regression coverage now exists in:
  - `tests/test_models/test_analytical_support.py`
  - `tests/test_agent/test_prompts.py`
  - `tests/test_agent/test_build_loop.py`
  - `tests/test_core/test_payoff_adapters.py`
- the deterministic `T105` path still succeeds after the tranche with
  `attempts=0`
- the latest deterministic rerun after the new rate/cross-asset helper slice
  also succeeded:
  - `task_results_t105_t105.json`
  - `task_results_t105_t105_summary.json`

## Why This Tranche Exists

`quanto` proved that Trellis can now close a fresh-build autonomous loop for a
guardrailed product family. The remaining gap is not "the model does not know
the formula." It is:

- analytical routes still require too much Trellis-specific wiring
- subproblems such as discounting, forwards, intrinsic transforms, and
  cross-asset adjustments are still too ad hoc
- larger future products, especially runtime-authored requests, will need to
  call analytical kernels inside broader decompositions rather than always use
  analytical methods as the top-level route

So the next library goal is not another product family first. It is a reusable
analytical support layer.

## Current Relevant Surfaces

Inspect first:

- `trellis/models/analytical/quanto.py`
  - current shared route kernel for the single-name quanto analytical path
- `trellis/models/analytical/support/__init__.py`
  - exported analytical support surface
- `trellis/models/analytical/support/discounting.py`
  - rate and discount helpers
- `trellis/models/analytical/support/forwards.py`
  - forward-construction helper
- `trellis/models/analytical/support/payoffs.py`
  - terminal intrinsic helper
- `trellis/models/analytical/support/cross_asset.py`
  - quanto adjustment helper
- `trellis/models/resolution/quanto.py`
  - resolved market-input contract already separating `MarketState` binding from
    analytical kernels
- `trellis/agent/prompts.py`
  - builder guidance for analytical fresh-build routes
- `trellis/agent/executor.py`
  - builder path, fragment recovery, validation loop, and critic/model stage
    orchestration
- `tests/test_models/test_analytical_support.py`
  - first deterministic regression suite for the support layer
- `tests/test_agent/test_build_loop.py`
  - fresh-build benchmark regressions
- `task_runs/history/T105/20260326T211522397021.json`
- `task_runs/history/T105/20260326T211610323555.json`
  - successful fresh-build evidence showing the current `quanto` proving loop
    works with the smaller analytical/MC glue surface

## Scope

### In Scope

- reusable analytical support functions that are `MarketState`-free
- support kernels usable both by:
  - top-level analytical routes
  - subproblems inside larger pricing engines
- prompt and scaffold guidance that forces fresh-build analytical routes to use
  the checked-in support layer where available
- deterministic tests and reruns proving the support layer is actually reused

### Out of Scope

- claiming broad closed-form support for products Trellis does not price yet
- building a giant universal symbolic system
- replacing all product-specific analytical kernels with one generic formula API
- runtime connector-backed market binding beyond the existing `quanto` proving
  path

## Design Principles

- analytical kernels do not read `MarketState`
- family resolvers own market binding
- support helpers are pure and typed
- route kernels compose support helpers
- generated code should prefer:
  - resolver
  - support helper
  - route kernel
  - thin adapter
- support helpers should be granular enough to reuse in decompositions, but not
  so tiny that they become formula-confetti

## Support Taxonomy

The support layer should grow in this order.

### 1. Discounting and Rate Transforms

Target module:

- `trellis/models/analytical/support/discounting.py`

Current checked-in functions:

- `implied_zero_rate`
- `discount_factor_from_zero_rate`
- `discounted_value`

Likely next additions:

- `continuous_rate_from_simple_rate`
- `simple_rate_from_discount_factor`
- `forward_discount_ratio`
- `safe_time_fraction`

### 2. Forward Builders

Target module:

- `trellis/models/analytical/support/forwards.py`

Current checked-in functions:

- `forward_from_discount_factors`

Likely next additions:

- `forward_from_carry_rate`
- `forward_from_dividend_yield`
- `forward_from_repo_style_carry`

### 3. Payoff Transforms

Target module:

- `trellis/models/analytical/support/payoffs.py`

Current checked-in functions:

- `terminal_intrinsic`

Likely next additions:

- `cash_or_nothing_intrinsic`
- `asset_or_nothing_intrinsic`
- `call_put_parity_gap`
- `normalized_option_type`

### 4. Cross-Asset Transforms

Target module:

- `trellis/models/analytical/support/cross_asset.py`

Current checked-in functions:

- `quanto_adjusted_forward`

Likely next additions:

- `effective_covariance_term`
- `exchange_option_effective_vol`
- `foreign_to_domestic_forward_bridge`

### 5. Distribution and Approximation Helpers

Recommended future modules:

- `trellis/models/analytical/support/distributions.py`
- `trellis/models/analytical/support/approximations.py`

Initial target helpers:

- stable normal CDF/PDF wrappers already aligned with Trellis differentiability
- bivariate-normal wrapper if/when needed
- Kirk-style effective strike/vol pieces
- moment-matching helpers for approximate Asian/basket routes

### 6. Inversion and Quoting Helpers

Recommended future module:

- `trellis/models/analytical/support/inversion.py`

Initial target helpers:

- implied-vol inversion wrappers
- delta/strike conversion helpers once Trellis exposes a stable runtime contract

## Implementation Phases

### Phase C1.1: Stabilize the Foundational Support Layer

Purpose:

- lock the first tranche into a stable public analytical support surface

Implementation:

- keep the existing helpers in:
  - `discounting.py`
  - `forwards.py`
  - `payoffs.py`
  - `cross_asset.py`
- ensure `trellis/models/analytical/__init__.py` exports only the helpers that
  are stable enough for generator reuse
- keep `trellis.models.analytical.quanto` routed entirely through those helpers

Tests first:

- extend `tests/test_models/test_analytical_support.py`
- keep route-level coverage in `tests/test_core/test_payoff_adapters.py`

Acceptance criteria:

- support helpers are pure, deterministic, and individually unit-tested
- `price_quanto_option_analytical(...)` uses support helpers instead of
  open-coded transforms

Status:

- implemented for the first foundational helper slice

### Phase C1.2: Add Reusable Analytical Subproblem Kernels

Purpose:

- make support helpers useful inside non-analytical engines and decompositions

Implementation:

- add the next kernel slice to the support package:
  - parity helpers
  - digital payoff transforms
  - normalized carry/forward transforms
- keep these helpers `MarketState`-free and composable

Likely files to change:

- `trellis/models/analytical/support/payoffs.py`
- `trellis/models/analytical/support/forwards.py`
- `trellis/models/analytical/support/discounting.py`
- `trellis/models/analytical/support/__init__.py`

Tests first:

- `tests/test_models/test_analytical_support.py`

Acceptance criteria:

- at least one support helper is explicitly reusable as a subproblem kernel
  outside the top-level quanto route
- tests cover reduction identities, not only spot numerical examples

Status:

- partially implemented via:
  - digital and asset-or-nothing intrinsic helpers
  - parity-gap helper
  - carry/dividend forward builders
  - time/rate normalization helpers
  - cross-asset covariance and forward-bridge helpers

### Phase C1.3: Refactor Analytical Route Kernels Onto the Support Layer

Purpose:

- prove the support layer reduces route-specific glue in checked-in code

Implementation:

- keep `quanto` as the first proving route
- move any remaining direct discounting/forward/intrinsic logic out of route
  kernels where stable support helpers exist
- identify the next analytical route to refactor once a second product warrants
  it

Likely files to change:

- `trellis/models/analytical/quanto.py`
- future route kernels under `trellis/models/analytical/`

Tests first:

- `tests/test_models/test_analytical_support.py`
- route-specific regression tests where applicable

Acceptance criteria:

- checked-in route kernels become thin compositions over support helpers
- new support helpers have a concrete checked-in consumer

### Phase C1.4: Tighten Builder Guidance Around Analytical Assembly

Purpose:

- make fresh-build analytical routes consume the checked-in support layer by
  default

Implementation:

- update `trellis/agent/prompts.py` so known analytical routes explicitly prefer:
  - family resolver
  - analytical support helpers
  - route kernels
  - thin adapters
- update `trellis/agent/executor.py` family-aware scaffold guidance so the model
  does not restate stable transforms

Likely files to change:

- `trellis/agent/prompts.py`
- `trellis/agent/executor.py`

Tests first:

- `tests/test_agent/test_prompts.py`
- `tests/test_agent/test_build_loop.py`

Acceptance criteria:

- prompt/regression tests assert known analytical routes reference the support
  layer explicitly
- fresh-build candidates stop re-deriving resolved quanto inputs that the
  resolver already guarantees

### Phase C1.5: Reliability Benchmark and Knowledge Capture

Purpose:

- confirm the analytical support tranche materially reduces fresh-build
  analytical variability

Implementation:

- rerun:
  - `python scripts/run_tasks.py --model gpt-5.4-mini --fresh-build --validation standard T105 T105`
  - `python scripts/run_tasks.py --model gpt-5-mini --fresh-build --validation standard T105 T105`
- capture:
  - task history artifacts
  - promotion candidates
  - review/adoption readiness
- compare fresh-build analytical code before vs after the support tranche

Acceptance criteria:

- fresh-build analytical success remains stable on the `quanto` proving route
- generated analytical code is visibly thinner and more scaffolded than earlier
  candidates
- no regression on the deterministic `T105` path

## Exact Test Files to Add or Extend

Add or extend:

- `tests/test_models/test_analytical_support.py`
- `tests/test_agent/test_prompts.py`
- `tests/test_agent/test_build_loop.py`
- `tests/test_core/test_payoff_adapters.py`

Target assertions:

- helper round-trips and reduction identities
- parity and intrinsic transforms
- cross-asset adjustment reductions
- analytical route kernels use support helpers
- prompt scaffolds mention support helpers explicitly
- fresh-build analytical candidates reuse the support layer rather than probing
  raw market-state fields

## Risks and Open Questions

### Risk: Over-fragmentation

If we split analytical helpers too finely, generated code will have more import
decisions and more assembly burden instead of less.

Mitigation:

- only extract helpers that represent stable reusable subproblems
- prefer route kernels over microscopic helpers where the math is tightly bound

### Risk: Fake Generality

A helper should not imply broader model support than Trellis actually has.

Mitigation:

- keep helper names honest and scope-specific
- avoid pretending that a `quanto` helper generalizes to all cross-asset
  exotics

### Risk: Prompt Drift

The support layer only helps fresh-build generation if prompts and scaffolds
require it.

Mitigation:

- keep prompt regressions alongside support regressions

### Open Question: How Far Should Support Go Before Another Product Family?

Recommendation:

- stop after the first clearly reusable analytical kernel tranche plus prompt
  tightening
- then measure fresh-build behavior again before committing to a wider
  analytical support buildout

## Critic-Path Follow-On Note

The earlier non-blocking warning:

- `Critic validation error (non-blocking): name 'get_model_for_stage' is not defined`

should now be treated as resolved for this tranche. The fix was to import the
stage helpers inside `_validate_build(...)` in `trellis/agent/executor.py`, and
the regression is covered in `tests/test_agent/test_build_loop.py`.

End-to-end note:

- a March 27, 2026 fresh-build rerun reached the critic LLM call itself rather
  than failing on the old missing-name path
- recent platform traces and `T105` task history no longer contain that warning
  signature

The remaining critic issue is not the old missing helper. It is runtime cost
and latency for the critic stage itself.

## Recommended Implementation Order

1. stabilize and extend the first support tranche with deterministic tests
2. add the next reusable subproblem kernels
3. refactor checked-in analytical routes onto the support layer
4. tighten analytical builder scaffolds and prompt obligations
5. rerun fresh-build `T105` under both mini models and compare generated code

## Next-Agent Review Checkpoint

The next agent should start by reviewing:

- `trellis/models/analytical/support/discounting.py`
- `trellis/models/analytical/support/forwards.py`
- `trellis/models/analytical/support/payoffs.py`
- `trellis/models/analytical/support/cross_asset.py`
- `trellis/models/analytical/quanto.py`
- `tests/test_models/test_analytical_support.py`

Recommended next concrete step:

- continue `C1.3` by identifying the next checked-in analytical route that can
  be thinned onto the support layer
- likely review targets are `trellis/models/analytical/barrier.py` and
  `trellis/models/analytical/jamshidian.py` for repeated discounting,
  normalization, or distribution glue that should move into support helpers
- only add new support helpers if a second checked-in route can consume them or
  if the helper is clearly a reusable analytical subproblem for decomposition

## Summary

`C1` is the analytical equivalent of the earlier `quanto` glue-surface work:
move repeated analytical plumbing into stable Trellis surfaces until the agent
is mostly assembling checked-in parts rather than rewriting formula glue from
scratch.
