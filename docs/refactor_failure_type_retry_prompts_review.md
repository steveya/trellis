# M1.7 Extension: Failure-Type-Aware Retry Prompts

Date: 2026-03-25

## Goal

Reduce retry token waste by matching the retry prompt surface to the actual
failure class instead of expanding all builder context after every failed
attempt.

## What Changed

### 1. Import-repair retries now use an import-only repair surface

If a build fails import validation, the next retry now uses:

- compact builder knowledge
- an `Import Repair Card`
- no reference implementation excerpts

This keeps the retry focused on approved modules and symbols instead of
replaying route context that is irrelevant to an import-only fix.

### 2. Semantic retries now use a semantic repair surface

If a build fails semantic validation, the next retry now uses:

- compact builder knowledge
- a `Semantic Repair Card`
- only a very small truncated reference surface

The semantic repair card emphasizes:

- selected route
- required primitives
- adapters
- uncertainty flags

This gives the builder the route-specific repair context it needs without
promoting the entire expanded prompt path.

### 3. Full expansion is now reserved for post-validation retries

Only retries caused by later validation failures now escalate to:

- expanded builder knowledge
- full structured generation plan
- fuller reference context

That means the heavier retry path is now reserved for failures where broader
reasoning context is actually more likely to help.

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  tests/test_agent/test_prompts.py \
  tests/test_agent/test_platform_loop.py \
  tests/test_agent/test_llm_guards.py -q
```

Result:

- `31 passed`

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

- `116 passed, 1 deselected`

Compile check:

```bash
/Users/steveyang/miniforge3/bin/python3 -m py_compile \
  trellis/agent/codegen_guardrails.py \
  trellis/agent/prompts.py \
  trellis/agent/executor.py
```

## Residual Limits

- Retry shaping is based on coarse failure classes, not a richer failure
  taxonomy yet.
- Post-validation retries still expand the full generation-plan context rather
  than a more surgical validation-repair surface.
- Critic/model-validator retry prompts are still less failure-type aware than
  the builder path.

## Next Step

The next `M1.7` extension should add a small deterministic lite-reviewer for
common generated-code mistakes, so obviously wrong code gets rejected before
spending tokens on the critic or deeper conceptual review.
