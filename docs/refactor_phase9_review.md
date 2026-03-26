# Phase 9 Review: Missing-Primitive Planner and Blocker Taxonomy

## Review

Before Phase 9, Trellis already knew when a route was blocked, but it only
communicated that through raw blocker strings such as:

- `path_dependent_early_exercise_under_stochastic_vol`
- `missing_module:...`
- `missing_symbol:...`

Those tokens were enough for internal gating, but not enough for planning new
machinery. They did not answer the operational questions:

- what kind of primitive is missing?
- where should it live?
- what tests should be added first?
- what docs and knowledge files need updating?

## Plan

Phase 9 should turn blocker strings into structured missing-primitive reports:

1. introduce a blocker taxonomy for known substrate gaps
2. interpret dynamic `missing_module:` and `missing_symbol:` failures
3. attach a structured blocker report to the generation plan
4. surface blocker reports in prompt/context/error rendering

## Tests First

Red tests were added for:

- blocked unsupported composites attach a structured blocker report
- the known stochastic-volatility early-exercise blocker resolves to the right
  category and target package
- `missing_module:` and `missing_symbol:` blockers are classified correctly
- rendered generation plans include structured blocker actions, not only raw IDs

## Implementation

New canonical taxonomy:

- [/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/canonical/blockers.yaml](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/canonical/blockers.yaml)

New planning module:

- [/Users/steveyang/Projects/steveya/trellis/trellis/agent/blocker_planning.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/blocker_planning.py)

The new blocker layer introduces:

- `PrimitiveBlocker`
- `BlockerReport`
- `plan_blockers(...)`
- `render_blocker_report(...)`

`GenerationPlan` in
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py)
now carries `blocker_report`, and prompt rendering now includes a structured
blocker section with:

- blocker category
- primitive kind
- recommended package
- suggested modules
- required tests
- docs and knowledge files to update

The early build rejection path in
[/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)
now uses the structured blocker report instead of concatenating raw tokens.

## Validation

Focused Phase 9 slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_blocker_planning.py
```

Broader regression:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
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

Phase 9 turns unsupported composites into actionable engineering work instead of
opaque blocker strings. That is the missing bridge into Phase 10, where those
blocker reports become concrete new-primitive workflows with package placement,
test requirements, docs, and knowledge updates.
