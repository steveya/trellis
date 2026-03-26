# M1.9 Review: Memory Distillation and Caching

Date: March 25, 2026

## Goal

Finish the last `M1` tranche by making repeated task-family runs reuse compact
knowledge and deterministic planning artifacts instead of re-reading verbose
knowledge surfaces every time.

## What Changed

### 1. Distilled shared-memory views

The shared knowledge payload in
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/retrieval.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/retrieval.py)
now emits three compact-first views:

- `builder_text_distilled`
- `review_text_distilled`
- `routing_text_distilled`

These are smaller than the prior compact prompt surfaces and are now the
default first-pass knowledge context for:

- compiled platform requests
- user-defined product compilation
- autonomous build orchestration
- executor fallback retrieval paths

The shared knowledge summary now also records prompt-size telemetry for
distilled, compact, and expanded views.

### 2. Runtime retrieval cache

The knowledge store in
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/store.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/store.py)
now keeps a warm retrieval cache keyed by retrieval spec. That avoids
recomputing feature expansion, lesson ranking, cookbook selection, and failure
signature matching for repeated tasks from the same family inside one process.

It also exposes:

- `retrieval_cache_stats()`
- `clear_runtime_caches()`

### 3. Runtime decomposition cache

The decomposition layer in
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/decompose.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/decompose.py)
now caches repeated `decompose(...)` calls by description, explicit instrument
hint, model, and store identity.

That means repeated task reruns can reuse the same decomposition result before
even hitting the retrieval cache.

The cache exposes:

- `decomposition_cache_stats()`
- `clear_decomposition_cache()`

Knowledge reload now clears this warm cache via
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/__init__.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/__init__.py)
so cache state cannot silently outlive a knowledge reload.

### 4. Generation-plan cache

Primitive planning in
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py)
now caches deterministic generation plans. That includes the primitive plan,
blocker report, and new-primitive workflow, so repeated supported or blocked
routes do not need to rebuild the same route plan in-process.

It exposes:

- `generation_plan_cache_stats()`
- `clear_generation_plan_cache()`

### 5. Executor/autonomous integration

The builder loop in
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)
and
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/autonomous.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/autonomous.py)
now consistently uses distilled knowledge on first pass and expanded knowledge
only when it explicitly escalates. The autonomous builder monkeypatch was also
updated to match the newer `compact=` retrieval signature.

## Why This Matters

This completes the `M1` token-efficiency arc at the runtime reuse layer:

- `M1.4` made token usage visible
- `M1.5` added budgets and stage-aware model tiering
- `M1.6` reduced prompt surface area
- `M1.7` removed avoidable LLM review calls
- `M1.8` moved more planning into structured tools
- `M1.9` now makes repeated work cheaper than recomputing it

Repeated reruns of the same task family now reuse:

- decomposition
- retrieval
- generation planning
- distilled knowledge views

instead of rebuilding the whole reasoning surface from scratch.

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_knowledge_store.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_decomposition_ir.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_primitive_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_autonomous.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_loop.py \
  -q
```

Result:

- `78 passed`

Broader nearby slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_knowledge_store.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_critic.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_model_validator.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_reflect_loop.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_autonomous.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_loop.py \
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

Result:

- see closeout summary for final count

## Remaining Limits

This tranche is intentionally warm-cache only:

- caches are process-local, not persisted across restarts
- raw traces are not yet distilled into a separate durable memory tier
- negative-result caching is currently indirect through cached generation plans,
  not yet a dedicated blocker-memory store

Those are acceptable remaining items for `M1`. Persistent or cross-run memory
compression belongs in later autonomy work, not this signal-cleanup milestone.
