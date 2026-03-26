# X1 Review: Deterministic Validation Bundles

Date: March 25, 2026

## Goal

Make route/product-family validation an explicit executable policy layer rather
than a mix of hardcoded checks and prompt guidance.

## What Changed

### 1. Validation-bundle selection now exists as a real module

[/Users/steveyang/Projects/steveya/trellis/trellis/agent/validation_bundles.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/validation_bundles.py)
now provides:

- `select_validation_bundle(...)`
- `execute_validation_bundle(...)`
- `ValidationBundle`
- `ValidationBundleExecution`

The current bundle categories are:

- `universal`
- `no_arbitrage`
- `product_family`

The selected checks are derived from the existing invariant-pack selector, but
they are now executable runtime policy instead of only builder guidance.

### 2. Executor validation now runs through the selected bundle

[/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/executor.py)
no longer hardcodes:

- `check_non_negativity`
- `check_price_sanity`
- optional callable-bond bounding

as separate ad hoc logic.

Instead it now:

- selects a validation bundle
- records `validation_bundle_selected`
- executes the bundle
- records `validation_bundle_executed`

### 3. Bundle execution is fail-fast after universal deterministic failures

If a universal deterministic check already fails, the executor now skips later
bundle checks instead of piling on secondary scenario/no-arbitrage failures.
That keeps the failure surface cleaner and preserves the deterministic fast
path.

## Why This Matters

This is the first real step toward “validation bundles” being a runtime
contract rather than a prompt convention.

That supports:

- FX proving-ground validation
- local-vol proving-ground validation
- later early-exercise policy-family validation

without making the LLM more powerful or freer.

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_evals.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_stress_task_preflight.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_lite_review.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_validation_bundles.py \
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

Result:

- `49 passed, 1 deselected`

## Remaining Limits

- `check_zero_vol_intrinsic` is selectable but still skipped unless the runtime
  has an explicit intrinsic-value contract for the product family
- comparison-task validation bundles are not yet a first-class separate
  category
- route-contract checks still live mostly in the lite reviewer rather than the
  validation-bundle executor
