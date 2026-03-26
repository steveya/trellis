# M1.7 Extension: Deterministic Lite Reviewer

Date: 2026-03-25

## Goal

Reject obviously wrong generated code before spending critic and
model-validator tokens on it.

## What Changed

### 1. Added a deterministic lite-reviewer stage

`trellis.agent.lite_review` now performs a cheap AST pass between semantic
validation and file write/import.

The first implemented checks are intentionally conservative and high-confidence:

- hardcoded discount-rate literals when `discount_curve` is required
- hardcoded volatility literals when `black_vol_surface` is required
- hardcoded FX literals when `fx_rates` is required
- hardcoded spot literals when `spot` is required
- route-specific analytical checks for the vanilla `analytical_black76` adapter:
  - missing `market_state.discount` access
  - missing `market_state.vol_surface` access
- route-specific Monte Carlo and rate-tree checks:
  - missing `market_state.discount` access
  - missing `market_state.vol_surface` access

These only block when the code appears to hardcode the relevant market input
instead of reading it from `market_state`.

### 2. Lite-review failures now retry through the semantic-repair surface

If the lite reviewer fails, the builder does not escalate to a broad retry.
Instead it:

- records a deterministic failure
- appends a `lite_reviewer` observation
- retries with the semantic-repair prompt surface
- keeps builder knowledge compact

### 3. Deterministic failures now short-circuit LLM reviewer stages

Inside `_validate_build`, if deterministic validation has already produced
failures, Trellis now skips:

- the critic
- LLM conceptual review in the model validator

That means low-value token spend is avoided when the build is already blocked by
deterministic evidence.

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_lite_review.py \
  tests/test_agent/test_prompts.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_llm_guards.py -q
```

Broader slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_autonomous.py \
  tests/test_agent/test_task_runtime.py \
  tests/test_agent/test_critic.py \
  tests/test_agent/test_model_validator.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_llm_guards.py \
  tests/test_agent/test_knowledge_store.py \
  tests/test_agent/test_prompts.py \
  tests/test_agent/test_lite_review.py \
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

## Residual Limits

- The lite reviewer is currently focused on hardcoded market-input anti-patterns.
- It does not yet classify a richer family of route-specific deterministic
  mistakes.
- Critic/model-validator prompts are still less failure-type aware than the
  builder retry path.

## Next Step

Extend the lite reviewer with a few more cheap route-specific checks, then keep
moving toward `M1.8` toolization so supported routes rely less on free-form
generation at all.
