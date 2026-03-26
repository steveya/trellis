# Phase 8 Review: Learned Ranking Baseline

## Review

By the end of Phase 7, Trellis had:

- deterministic candidate-route enumeration
- deterministic heuristic scores
- hard blocker semantics
- a stable `ProductIR -> candidate routes -> selected route` pipeline

What it did **not** have was any concrete offline training loop for a future
learned ranker. The design discussion for Phase 8 converged on a specific
requirement:

1. generate many supported and unsupported products
2. run `ProductIR` construction on them
3. enumerate candidate routes
4. capture blocker / semantic decision state
5. fit a learned scorer on top of that data

That means Phase 8 should start with a deterministic offline pipeline, not with
runtime ML integration.

## Plan

Phase 8 should build a minimal but real learned-ranking baseline:

1. define a deterministic corpus of synthetic product cases
2. convert those cases into supervised route-ranking rows
3. fit a simple linear scorer offline
4. keep blocker semantics as hard gates around the learned ranking

This is intentionally conservative. The point is to establish a clean dataset
and evaluation loop before any learned model is allowed to influence the live
build path.

## Tests First

Red tests were added for:

- the synthetic corpus includes both supported and blocked products
- training rows contain both `proceed` and `block` decisions
- a fitted learned ranker prefers `exercise_lattice` for callable bonds and
  `exercise_monte_carlo` for American puts
- blocked unsupported composites remain blocked even after learned reranking

## Implementation

`/Users/steveyang/Projects/steveya/trellis/trellis/agent/route_learning.py`

The new module provides:

- `SyntheticProductCase`
- `RouteTrainingRow`
- `LearnedRouteRanker`
- `LearnedRouteDecision`
- `default_synthetic_product_cases()`
- `build_route_training_rows(...)`
- `fit_linear_route_ranker(...)`
- `rank_routes_with_learned_model(...)`
- `learned_route_decision(...)`

The training data is built from the exact stack we discussed:

- synthetic product description
- `decompose_to_ir(...)`
- `rank_primitive_routes(...)`
- blocker-aware decision labeling

The learned model is deliberately simple:

- ridge-regularized linear fit
- no extra dependency like scikit-learn
- no runtime override of hard blocker rules

The current labels are a bootstrap signal:

- they combine the deterministic route score from Phase 7
- plus the explicit `proceed` / `block` decision implied by the blocker logic

This means the learned model is still trained against Trellis policy, not loose
prompt outcomes.

## Validation

Focused Phase 8 slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_learning.py
```

Broader regression:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_learning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_route_scoring.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_primitive_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_ir_retrieval.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py
```

Result:

- focused slice: `4 passed`
- broader regression: `36 passed`

## Outcome

Phase 8 does not replace the heuristic route scorer. It gives Trellis a
concrete offline learning substrate:

- synthetic products
- route-level supervised rows
- learned ranking experiments
- hard blocker semantics preserved

That is enough to support the next phases:

- Phase 9: structured blocker taxonomy instead of raw blocker strings
- Phase 10: concrete workflow planning for adding new primitives when blockers
  reveal real substrate gaps
