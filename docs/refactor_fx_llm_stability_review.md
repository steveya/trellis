# FX LLM Stability Follow-Up

Date: 2026-03-26

## Scope

This tranche addressed the LLM invocation failures that were blocking live FX reruns for `E25`.

## Changes

- Added deterministic spec-schema fast paths for common vanilla option families in `/Users/steveyang/Projects/steveya/trellis/trellis/agent/planner.py`.
  - `FXVanillaOptionSpec` now resolves without an LLM `spec_design` call for vanilla FX analytical and Monte Carlo routes.
- Reduced `spec_design` prompt size in `/Users/steveyang/Projects/steveya/trellis/trellis/agent/prompts.py` and `/Users/steveyang/Projects/steveya/trellis/trellis/core/capabilities.py`.
  - Only required capabilities are included.
  - The full computational-method catalog is no longer sent for schema-design prompts.
- Replaced the stale Anthropic stage defaults in `/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py`.
- Added robust JSON extraction for Anthropic/OpenAI JSON stages in `/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py`.
- Reworked OpenAI hard timeouts to use a real wall-clock worker-thread wrapper in `/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py`.
  - This now applies off the main thread too.
- Switched the default OpenAI `code_generation` stage from `gpt-5` to `gpt-5-mini` in `/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py`.
  - The goal is bounded rerun latency for Trellis stress/proving-ground tasks.
  - Override guidance is documented in `/Users/steveyang/Projects/steveya/trellis/docs/openai_stage_model_selection.md`.
- Recorded `spec_design_failed` and `builder_attempt_failed(reason=code_generation)` more explicitly in `/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py`.
- Skipped reflection when provider/config failures happened before any code was produced in `/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/autonomous.py`.

## Validation

- Targeted slice:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_llm_guards.py /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_planner.py /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_autonomous.py -q`
  - Result: `34 passed`
- Executor/runtime slice:
  - `/Users/steveyang/miniforge3/bin/python3 -m pytest /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_loop.py /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_build_loop.py -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q`
  - Result: `48 passed, 1 deselected`

## Live FX Findings

- `E25` no longer stalls in `spec_design`.
- The FX traces now show:
  - `planner_completed` with `FXVanillaOptionSpec`
  - `spec_design_skipped`
  - `builder_attempt_failed` with `reason: code_generation` when the OpenAI codegen call times out
- Reflection is now skipped on those provider/codegen failures, avoiding extra token spend.

## Remaining Issue

The remaining live blocker is no longer FX market data or FX knowledge. It is code-generation latency / provider responsiveness for the `gpt-5` codegen stage under the current timeout budget.

The next sensible follow-up is:

1. inspect and shrink the FX codegen prompt surface further
2. consider a shorter compact builder prompt for vanilla FX routes
3. decide whether the `gpt-5` codegen timeout budget should be increased slightly for valid builds or whether a smaller/faster codegen model should be preferred for this route
