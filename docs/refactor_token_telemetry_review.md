# M1.4 Review: Token Telemetry and Budget Visibility

This note records the first implementation slice for `M1.4` from
[autonomous_library_development_workstream.md](/Users/steveyang/Projects/steveya/trellis/docs/autonomous_library_development_workstream.md).

## Goal

Make LLM usage visible at the stage, task, and batch level before trying to
optimize it.

The immediate question this phase answers is:

- where did the tokens go for a given rerun?

## Implemented

### 1. LLM usage collection in the provider wrapper

Implemented in:

- [config.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py)

Adds:

- `llm_usage_session()`
- `llm_usage_stage(...)`
- `summarize_llm_usage(...)`
- provider-specific token extraction for OpenAI and Anthropic responses

The wrapper now normalizes:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`

When a provider does not expose usage, the summary records that explicitly via
`calls_without_usage`.

### 2. Stage labeling across the build loop

Implemented across:

- [autonomous.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/autonomous.py)
- [executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)

The main stages now tracked are:

- `decomposition`
- `spec_design`
- `code_generation`
- `critic`
- `model_validator`
- `reflection`

This is enough to explain most of the current task token burn.

### 3. Platform trace persistence

Implemented in:

- [platform_traces.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/platform_traces.py)

Platform traces can now persist an aggregated `token_usage` block. The current
build path attaches the final per-request summary back to the trace file after
the knowledge-aware build completes.

### 4. Task-run persistence and aggregation

Implemented in:

- [task_runtime.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/task_runtime.py)
- [task_run_store.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/task_run_store.py)

Task results and canonical task-run records now carry:

- task-level `token_usage_summary`
- method-level token usage for comparison tasks
- trace-level token usage in trace summaries

This means a persisted task record can now answer:

- how many LLM calls happened
- how many had token telemetry
- how many tokens were spent overall
- which stages dominated usage

### 5. Batch summaries

Implemented in:

- [evals.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/evals.py)
- [run_tasks.py](/Users/steveyang/Projects/steveya/trellis/scripts/run_tasks.py)

Batch summaries now include a `token_usage` section, so rerun campaigns can be
compared on both outcome quality and token spend.

## Validation

Focused telemetry slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_llm_guards.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_task_run_store.py \
  tests/test_agent/test_evals.py -q
```

Result:

- `34 passed`

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
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

Result:

- `72 passed, 1 deselected`

## Remaining Limitation

This phase adds visibility, not control.

The main remaining limitation is:

- Trellis can now tell us where tokens went, but it does not yet enforce token
  budgets or aggressively compress prompts

That is the next phase:

- `M1.5` Token budgets, model tiering, and prompt compression
