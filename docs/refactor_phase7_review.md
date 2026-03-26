# Phase 7 Review: Deterministic Route Scoring

## Review

After Phase 6, Trellis had an explicit `PrimitivePlan` and the right route
families, but route choice was still effectively hard-coded:

- `build_primitive_plan(...)` selected a single route branch
- `score` existed only as metadata on the chosen route
- there was no way to inspect how alternative valid routes compared

That made the architecture less testable than it should be. The design intent
for the pre-ML stage is:

- enumerate legal candidate routes
- score them deterministically
- select the highest-scored route
- keep blockers as hard negative evidence

## Plan

Phase 7 should make the deterministic heuristic explicit and inspectable without
introducing any learned ranking yet.

1. Add a ranked candidate-route API.
2. Make `build_primitive_plan(...)` choose the top-ranked candidate.
3. Penalize blocked unsupported composites strongly enough that they are
   clearly non-buildable.
4. Add regression tests for relative route ordering on known products.

## Tests First

Red tests were added for:

- callable bond ranks `exercise_lattice` above generic tree rollback
- American put ranks `exercise_monte_carlo` above plain Monte Carlo
- blocked unsupported composites get a negative route score
- `build_generation_plan(...)` selects the highest-scored route

## Implementation

`/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py`

The route-selection logic now has three explicit stages:

1. `_candidate_routes(...)`
   - enumerates the plausible route set for the current method + `ProductIR`
2. `_route_score(...)`
   - computes a deterministic heuristic score for each route
3. `rank_primitive_routes(...)`
   - builds and sorts the candidate `PrimitivePlan` list

`build_primitive_plan(...)` now delegates to `rank_primitive_routes(...)` and
returns the top-ranked plan.

The current scoring heuristic stays intentionally simple:

- reward route existence
- reward engine-family compatibility with `ProductIR`
- reward explicit exercise-route matches
- reward schedule-aware lattice routes for schedule-dependent exercise products
- penalize plain MC or generic tree routes for early-exercise products
- heavily penalize unresolved blockers
- add an extra penalty when the product is unsupported and blockers remain

This is still hand-coded, but it is now testable and inspectable.

## Validation

Focused Phase 7 slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_scoring.py
```

Broader deterministic regression:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_scoring.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_primitive_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_semantic_validation.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_rate_tree_generation.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_ir_retrieval.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_build_loop.py::TestBuildLoop::test_rebuilt_bermudan_rate_tree_payoff_prices_plausibly \
  /Users/steveyang/Projects/steveya/trellis/tests/test_tasks/test_t04_bermudan_swaption.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_tasks/test_t02_bdt_callable.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_cookbooks.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_knowledge_store.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py
```

Result:

- focused slice: `4 passed`
- broader regression: `133 passed`

## Outcome

Phase 7 does not add ML. It does something more important first: it makes the
current hardcoded ranking logic explicit enough to evaluate and trust.

That gives Trellis a stable precondition for a later learned-ranking phase:

- we now have deterministic candidate-route enumeration
- we now have inspectable heuristic scores
- we can log route comparisons and outcomes cleanly
- and we still keep semantic and primitive-availability checks as hard gates
