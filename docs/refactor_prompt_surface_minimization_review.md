# M1.6 Review: Prompt Surface Minimization

Date: 2026-03-25

## Goal

Reduce prompt size for repeated task reruns without weakening the build and
validation loop, and extend stage-aware model selection to the OpenAI provider.

## What Changed

### 1. OpenAI stage-aware model hierarchy

The OpenAI provider no longer uses `gpt-5-mini` for every stage.

Default stage mapping is now:

- `decomposition` -> `gpt-5-mini`
- `spec_design` -> `gpt-5-mini`
- `code_generation` -> `gpt-5`
- `critic` -> `gpt-5-mini`
- `model_validator` -> `gpt-5`
- `reflection` -> `gpt-5-mini`

This keeps cheaper/faster reasoning on low-leverage stages while reserving a
stronger model for the two stages most sensitive to raw model capability:
code generation and deep model review.

Stage env overrides still take precedence through:

- `TRELLIS_MODEL_<STAGE>`
- `TRELLIS_OPENAI_MODEL_<STAGE>`

### 2. Compact-first shared knowledge

Shared knowledge payloads now materialize both:

- compact prompt views
- expanded prompt views

and store prompt-size telemetry in the shared knowledge summary.

This means traces and compiled requests can carry:

- compact builder/reviewer/routing text for first-pass calls
- expanded variants for retry escalation
- compact vs expanded character counts for prompt-surface inspection

### 3. Retry-time prompt escalation

The executor now uses:

- compact builder knowledge on the first generation attempt
- expanded builder knowledge on retries
- compact reviewer knowledge on the first critic/model-validator pass
- expanded reviewer knowledge on retries

The prompt surface and knowledge-context size are recorded in platform events
for builder, critic, and model-validator stages.

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_llm_guards.py \
  tests/test_agent/test_knowledge_store.py \
  tests/test_agent/test_prompts.py \
  tests/test_agent/test_platform_loop.py -q
```

Result:

- `69 passed`

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

Result:

- `107 passed, 1 deselected`

## Residual Limits

- Prompt surface is smaller, but reference-source payloads are still relatively
  heavy for some routes.
- Retry escalation is attempt-count based, not yet failure-type aware.
- Direct non-build flows still benefit less from the compact/expanded split than
  the knowledge-aware build loop.

## Next Phase

`M1.7` should move more routing and first-pass review decisions out of the LLM
path entirely so prompt minimization is paired with fewer LLM calls overall.
