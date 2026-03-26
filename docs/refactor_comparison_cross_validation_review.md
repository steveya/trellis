# Comparison Task Runtime Cross-Validation

This tranche resolves the remaining caveat in the task runner for benchmark and
comparison tasks.

Before this change, multi-method tasks could build one payoff per method family,
but the task runner still treated that as sufficient success. It did not:

- build the concrete `cross_validate.internal` targets from `TASKS.yaml`
- add the analytical reference target when present
- instantiate the built payoffs and compare prices
- fail the task when the numeric comparison itself failed

## What Changed

In [task_runtime.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/task_runtime.py):

- `cross_validate.internal` now defines the concrete comparison build targets
- `cross_validate.analytical` now becomes an optional analytical reference build
- comparison targets are mapped back to canonical method families through
  deterministic target-to-method routing
- each successful build is instantiated and priced through the same test-payoff
  path used by the offline benchmark helpers
- comparison tasks now report:
  - concrete target prices
  - selected reference target and price
  - deviations in percent
  - tolerance
  - pass/fail status

Comparison-task success now requires:

- all requested builds succeed
- runtime cross-validation status is `passed`

## Additional Builder Hint

The task runner now passes a `comparison_target` hint into the knowledge-aware
build wrapper. The wrapper folds that into the effective build description so
the agent can distinguish targets like `fft` and `cos` even though they share
the broader `fft_pricing` method family.

## Validation

Focused runtime slice:

- [test_task_runtime.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py)
  with the known unrelated generic cached-transform benchmark excluded

Result:

- `11 passed, 1 deselected`

Broader deterministic slice:

- [test_task_runtime.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py)
- [test_autonomous.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_autonomous.py)
- [test_llm_guards.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_llm_guards.py)
- [test_reflect_loop.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_reflect_loop.py)
- [test_platform_loop.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_platform_loop.py)
- [test_quant.py](/Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py)

Result:

- `44 passed, 1 deselected`

## Remaining Limitation

This makes comparison tasks numerically meaningful at the task-runner level,
but it does not yet add agent evals for comparison-task build quality. That
should remain eval-driven rather than over-specified with brittle unit tests.
