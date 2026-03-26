# M1.5 Review: Token Budgets, Model Tiering, and Prompt Compression

This note records the `M1.5` implementation slice from
[autonomous_library_development_workstream.md](/Users/steveyang/Projects/steveya/trellis/docs/autonomous_library_development_workstream.md).

## Goal

Reduce token burn without weakening the validation and learning loop.

This phase focuses on three levers:

- stage-aware model tiering
- explicit task and batch token budgets
- prompt compression for shared knowledge

## Implemented

### 1. Stage-aware model tiering

Implemented in:

- [config.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py)
- [autonomous.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/autonomous.py)
- [executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)

Adds:

- `get_model_for_stage(stage, requested_model=...)`

Current policy:

- if the caller passes a non-default model, Trellis respects it
- if the caller is using the provider default, Trellis may tier down cheaper
  stages automatically

For Anthropic-backed runs, the current default tiering is:

- `decomposition` -> `claude-3-5-haiku-latest`
- `spec_design` -> `claude-3-5-haiku-latest`
- `critic` -> `claude-3-5-haiku-latest`
- `reflection` -> `claude-3-5-haiku-latest`
- `code_generation` -> `claude-sonnet-4-6`
- `model_validator` -> `claude-sonnet-4-6`

These can be overridden via environment variables such as:

- `TRELLIS_MODEL_DECOMPOSITION`
- `TRELLIS_MODEL_REFLECTION`
- `TRELLIS_ANTHROPIC_MODEL_CRITIC`

### 2. Explicit token budgets

Implemented in:

- [config.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py)
- [autonomous.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/autonomous.py)
- [executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)
- [run_tasks.py](/Users/steveyang/Projects/steveya/trellis/scripts/run_tasks.py)

Adds:

- `TokenBudgetExceeded`
- `get_task_token_budget()`
- `get_batch_token_budget()`
- `enforce_llm_token_budget(...)`

Behavior:

- per-task budgets are enforced after expensive LLM stages
- reflection is skipped with an explicit note if the budget is already spent
- batch runs can stop early once cumulative token spend exceeds the configured
  batch budget

Environment controls:

- `TRELLIS_TASK_TOKEN_BUDGET`
- `TRELLIS_BATCH_TOKEN_BUDGET`

### 3. Prompt compression for shared knowledge

Implemented in:

- [retrieval.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/retrieval.py)
- [autonomous.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/autonomous.py)
- [executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)
- [prompts.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/prompts.py)

Shared knowledge payloads now use compact prompt views by default.

Current compression behavior:

- cap principles, lessons, requirements, data contracts, and unresolved
  primitives
- truncate large cookbook templates
- truncate the import-registry block in prompt views
- make omissions explicit with markers like:
  - `[omitted N additional lessons]`
  - `[truncated cookbook template]`

This reduces prompt size while preserving the main route and failure-memory
signals.

### 4. Roadmap extension beyond M1.5

Updated in:

- [autonomous_library_development_workstream.md](/Users/steveyang/Projects/steveya/trellis/docs/autonomous_library_development_workstream.md)

Added next planned efficiency phases:

- `M1.6` Prompt surface minimization
- `M1.7` Deterministic-first routing and review
- `M1.8` Toolization of primitive assembly and validation
- `M1.9` Memory distillation and caching

## Validation

Focused `M1.5` slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_llm_guards.py \
  tests/test_agent/test_knowledge_store.py \
  tests/test_agent/test_prompts.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_task_run_store.py \
  tests/test_agent/test_evals.py -q
```

Result:

- `80 passed`

Broader surrounding slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_autonomous.py \
  tests/test_agent/test_task_runtime.py \
  tests/test_agent/test_critic.py \
  tests/test_agent/test_model_validator.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_llm_guards.py \
  tests/test_agent/test_task_run_store.py \
  tests/test_agent/test_evals.py \
  tests/test_agent/test_knowledge_store.py \
  tests/test_agent/test_prompts.py \
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

Result:

- `118 passed, 1 deselected`

## Remaining Limitation

This phase reduces waste, but it does not yet fundamentally change how much of
the workflow is still prompt-driven.

The next high-leverage work is:

- `M1.6` prompt surface minimization
- `M1.7` deterministic-first routing and review
- `M1.8` toolization of primitive assembly and validation
- `M1.9` memory distillation and caching

That is where Trellis starts shifting work out of tokens instead of only
spending tokens more carefully.
