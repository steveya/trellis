# Structured Task and Learning Loop Hardening

This tranche finished the remaining platform-loop seams that still sat between
older task-runner plumbing and the newer request/compiler stack.

## Scope

The focus was:

1. make `TASKS.yaml` structured fields actually drive execution
2. thread `preferred_method` all the way through the knowledge-aware build path
3. let multi-method tasks fan out into per-method builds instead of being
   forced through one route
4. make empty or invalid model responses fail explicitly
5. leave deterministic cookbook-candidate artifacts when reflection is weak
6. add deterministic closed-loop coverage for lesson capture and retrieval

## What Changed

### Task execution

In [task_runtime.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/task_runtime.py):

- `construct` is now normalized to canonical methods
- a single known construct becomes `preferred_method`
- multiple known constructs become a comparison task with one build per method
- `cross_validate` and `new_component` are now carried through the task result

### Preferred-method routing

In [autonomous.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/autonomous.py)
and [executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py):

- `preferred_method` now reaches gap checking, IR-native retrieval, and the
  canonical build compiler
- the build path can deliberately target PDE, tree, Monte Carlo, or transform
  routes for the same product semantics

### LLM response hardening

In [config.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/config.py)
and [executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py):

- blank text responses now raise an explicit runtime error
- blank JSON responses now raise an explicit runtime error
- invalid JSON responses now include line/column diagnostics
- module generation retries on empty module bodies instead of failing later with
  opaque parser errors

### Reflection fallback

In [reflect.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/reflect.py):

- successful builds without a canonical cookbook now record a deterministic
  cookbook candidate under the trace store when LLM reflection does not produce
  a safe cookbook extract
- the canonical cookbook is still only updated through the stricter enrichment
  path

## Validation

Focused slice:

- [test_task_runtime.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py)
  with the known unrelated generic cached-transform benchmark excluded
- [test_autonomous.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_autonomous.py)
- [test_llm_guards.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_llm_guards.py)
- [test_reflect_loop.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_reflect_loop.py)

Result:

- `16 passed, 1 deselected`

Broader slice:

- [test_platform_loop.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_loop.py)
- [test_quant.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py)
- [test_knowledge_store.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_knowledge_store.py)
- [test_term_sheet.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_term_sheet.py)
- [test_capabilities.py](/Users/steveyang/Projects/steveya/trellis/tests/test_core/test_capabilities.py)

Result:

- `99 passed`

## Notes

- The excluded test is the already-known unrelated generic cached-transform
  offline benchmark collision. It was not changed in this tranche.
- This tranche is still deterministic-library work. Agent-quality assessment
  for comparison-task build quality should continue to use evals rather than
  brittle unit assertions.
