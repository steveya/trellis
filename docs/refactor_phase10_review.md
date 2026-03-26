# Phase 10 Review: New-Primitive Build Workflow

## Review

After Phase 9, Trellis could explain blockers structurally, but it still did
not tell the developer or future agent what to do next in concrete engineering
terms.

There was still a missing step between:

- "this product is blocked by a missing primitive"

and

- "here is the package, contract, test surface, docs, and knowledge update path
  required to add that primitive correctly"

That is the operational gap Phase 10 closes.

## Plan

Phase 10 should turn a `BlockerReport` into a concrete implementation workflow:

1. classify the action kind
   - new foundational primitive
   - library repair
   - taxonomy extension
2. map the blocker onto a target package / module surface
3. define the mathematical contract in plain terms
4. carry over the required tests, docs, and knowledge files to update

## Tests First

Red tests were added for:

- numerical substrate gaps produce `new_foundational_primitive` workflow items
- missing-symbol gaps produce `library_repair` workflow items
- rendered workflow text includes mathematical contract, tests, docs, and
  knowledge updates

## Implementation

New module:

- [/Users/steveyang/Projects/steveya/trellis/trellis/agent/new_primitive_workflow.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/new_primitive_workflow.py)

It introduces:

- `NewPrimitiveWorkItem`
- `NewPrimitiveWorkflow`
- `plan_new_primitive_workflow(...)`
- `render_new_primitive_workflow(...)`

The workflow planner uses the structured blocker report from Phase 9 and maps
blockers into concrete action kinds:

- `numerical_substrate_gap` -> `new_foundational_primitive`
- `implementation_gap` / `export_or_registry_gap` -> `library_repair`
- fallback -> `taxonomy_extension`

`GenerationPlan` in
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py)
now carries `new_primitive_workflow`, and prompt rendering includes that
workflow when blockers are present.

The early blocker failure in
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)
also now includes the rendered workflow so unsupported composites fail with
actual engineering guidance instead of only a blocker summary.

## Validation

Focused Phase 10 slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_new_primitive_workflow.py
```

Broader regression:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_new_primitive_workflow.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_blocker_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_learning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_scoring.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_primitive_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_semantic_validation.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_ir_retrieval.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py
```

Result:

- focused slice: `3 passed`
- broader regression: green

## Outcome

Phase 10 turns unsupported user-defined products into explicit engineering work
items. Trellis can now say not only:

- which primitive is missing

but also:

- what kind of work it is
- where it belongs
- what mathematical contract it must satisfy
- what tests, docs, and knowledge files need updating

That is the minimum workflow needed before agents can safely attempt to add new
foundational pricing machinery instead of only assembling from existing pieces.
