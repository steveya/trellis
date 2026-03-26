# Phase 6 Review: Generalized Exercise / Control Route

## Review

Phase 5 proved that Trellis could build and validate two schedule-dependent
exercise products:

- callable bond via lattice backward induction
- Bermudan swaption via lattice backward induction

The remaining architectural gap was that the agent stack still treated those
products as a generic `rate_tree_backward_induction` route. That was sufficient
to reuse `build_rate_lattice(...)` and `lattice_backward_induction(...)`, but it
did not encode the product-side semantics that matter for exercise products:

- schedule-dependent exercise must use `exercise_type="bermudan"`
- schedule-dependent exercise must pass explicit `exercise_steps`
- issuer-callable products minimize liability with `exercise_fn=min`
- holder-exercised products maximize value with `exercise_fn=max`

The semantic validator also had no visibility into those lattice-specific
contracts, so a generated callable bond or Bermudan swaption could still pass as
long as it imported the right tree modules.

## Plan

Phase 6 should generalize the exercise/control route across lattice products
without introducing new numerical substrate:

1. Add a distinct `exercise_lattice` primitive route for callable, puttable,
   Bermudan, and other rate-tree exercise products.
2. Keep the primitives unchanged, but make the route metadata explicit about
   schedule mapping and objective selection.
3. Extend semantic extraction to read `exercise_type`, `exercise_steps`, and
   `exercise_fn` from `lattice_backward_induction(...)`.
4. Reject semantically invalid lattice exercise code even when imports and
   syntax are valid.

## Tests First

Red tests were added for:

- primitive planning chooses `exercise_lattice` for callable bond
- primitive planning chooses `exercise_lattice` for Bermudan swaption
- semantic extraction sees the callable artifact as a Bermudan-style lattice
  exercise with `exercise_fn=min`
- callable lattice code using `exercise_fn=max` is rejected
- Bermudan lattice code using `exercise_fn=min` is rejected
- schedule-dependent lattice code without `exercise_steps` is rejected

## Implementation

The implementation changed two layers:

### Planning

`/Users/steveyang/Projects/steveya/trellis/trellis/agent/codegen_guardrails.py`

- added `exercise_lattice` route selection for rate-tree exercise products
- added route adapters:
  - `map_cashflows_and_exercise_dates_to_tree_steps`
  - `select_exercise_fn_for_issuer_or_holder`
- kept the same reusable primitives:
  - `build_rate_lattice`
  - `lattice_backward_induction`
- slightly refined the deterministic score so explicit lattice exercise routes
  score above generic tree routing for schedule-dependent exercise products

### Semantic validation

`/Users/steveyang/Projects/steveya/trellis/trellis/agent/semantic_validation.py`

- added lattice semantic signals:
  - `lattice_exercise_types`
  - `lattice_has_exercise_steps`
  - `lattice_exercise_functions`
- added deterministic checks:
  - `lattice.exercise_type_mismatch`
  - `lattice.exercise_schedule_missing`
  - `lattice.exercise_objective_mismatch`

## Validation

Focused Phase 6 slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_primitive_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_semantic_validation.py
```

Broader Phase 6 regression:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -x -q \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_primitive_planning.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_semantic_validation.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_rate_tree_generation.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_build_loop.py::TestBuildLoop::test_rebuilt_bermudan_rate_tree_payoff_prices_plausibly \
  /Users/steveyang/Projects/steveya/trellis/tests/test_tasks/test_t04_bermudan_swaption.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_tasks/test_t02_bdt_callable.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_cookbooks.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_knowledge_store.py
```

Result:

- focused slice: green
- broader regression: `107 passed`

## Outcome

Phase 6 turns exercise-on-lattice products into a first-class route family
instead of a generic tree special case. That gives Phase 7 a cleaner target:
review whether the current deterministic route score is choosing sensible
routes, now that exercise products are represented more honestly.
