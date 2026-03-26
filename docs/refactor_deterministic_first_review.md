# M1.7 Review: Deterministic-First Routing and Review

Date: 2026-03-25

## Goal

Reduce LLM usage by keeping common, low-ambiguity validation work on a
deterministic path and escalating to critic/model-validator LLM review only for
higher-risk routes.

## What Changed

### 1. Deterministic review policy

A new deterministic review policy now classifies build-validation risk and
decides whether Trellis should run:

- the critic LLM stage
- the model-validator LLM conceptual-review stage

Current low-risk fast path:

- supported `european_option`
- `analytical`
- `exercise_style` in `{european, none}`
- `state_dependence == terminal_markov`
- no schedule dependence
- no unresolved primitives
- no high-risk payoff traits

These routes now skip LLM reviewer stages by default.

### 2. Automated-only model validation for low-risk builds

`validate_model(...)` now supports deterministic-only execution. Automated
checks still run:

- sensitivity
- benchmark checks where available

But the LLM conceptual review is skipped unless the deterministic review policy
requires escalation.

### 3. Executor skip reasons and audit trail

The build executor now emits explicit audit events when it skips reviewer LLM
stages:

- `critic_skipped`
- `model_validator_llm_review_skipped`

These events include risk level and skip reason, so traces show why tokens were
not spent.

`model_validator_completed` now also records whether LLM review ran and, when it
did not, the deterministic skip reason.

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_model_validator.py \
  tests/test_agent/test_critic.py \
  tests/test_agent/test_platform_loop.py -q
```

Result:

- `33 passed`

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
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

## Residual Limits

- The deterministic fast path is intentionally conservative; it currently only
  covers the simplest supported analytical vanilla route.
- The critic itself is still fully LLM-based when it does run.
- Routing is already mostly deterministic for canonical products, but we have
  not yet added a richer compile-time ambiguity score to decide when
  decomposition should escalate.

## Next Step

The next extension of `M1.7` should widen the deterministic fast path from
"analytical vanilla only" into a broader risk-based classifier and add more
non-LLM first-pass review heuristics for common generated-code mistakes.
