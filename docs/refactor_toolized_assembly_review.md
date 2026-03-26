# M1.8 Review: Toolized Primitive Assembly and Validation

Date: 2026-03-25

## Goal

Shift more supported-route work from free-form prompt reasoning into structured
library tools and deterministic plan objects.

## What Changed

### 1. Added structured assembly helpers

`trellis.agent.assembly_tools` now exposes deterministic helpers for:

- primitive lookup
- thin-adapter plans
- invariant-pack selection
- comparison-harness planning
- cookbook-candidate payload capture

These are reusable from prompts, task/runtime code, reflection, and interactive
tool calls.

### 2. Builder prompts now consume assembly tools directly

`evaluate_prompt(...)` now injects three structured sections ahead of the
reference implementations:

- `Primitive Lookup`
- `Thin Adapter Plan`
- `Invariant Pack`

This makes supported-route builds thinner by default. The builder is no longer
expected to infer all adapter obligations from broad context alone.

### 3. Repo-aware interactive tools now expose the same deterministic surfaces

The interactive tool layer now exposes:

- `lookup_primitive_route`
- `build_thin_adapter_plan`
- `select_invariant_pack`
- `build_comparison_harness`
- `capture_cookbook_candidate`

That keeps the interactive agent aligned with the deterministic planning logic
already used by the build/runtime path.

### 4. Comparison planning and cookbook capture now reuse the shared tools

- `task_runtime` now builds comparison-target plans via `assembly_tools`
- `reflect.py` now uses a deterministic cookbook-candidate payload helper

That reduces duplicated logic across the platform.

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_assembly_tools.py \
  tests/test_agent/test_tools.py \
  tests/test_agent/test_prompts.py \
  tests/test_agent/test_task_runtime.py \
  tests/test_agent/test_reflect_loop.py -q
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
  tests/test_agent/test_tools.py \
  tests/test_agent/test_assembly_tools.py \
  tests/test_agent/test_reflect_loop.py \
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

## Residual Limits

- The builder still produces Python via free-form generation; the new assembly
  tools constrain that generation rather than replacing it completely.
- Invariant packs are selected deterministically but `_validate_build(...)`
  still runs a more fixed validation sequence.
- Comparison-harness planning is now centralized, but runtime comparison
  execution remains in `task_runtime`.

## Next Step

The next step after this tranche should be to push invariant-pack selection and
adapter obligations deeper into execution/validation so supported routes become
even thinner orchestrations over deterministic tools.
