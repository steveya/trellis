# M2.1 / M2.2 Review: FX, Forecast-Curve, and State-Space Mock Bridging

Date: March 25, 2026

## Goal

Close the remaining mock-market gaps that were still surfacing as avoidable
task failures:

- `fx_rates` should be usable as an explicit task-selected market input
- a selected `forecast_curve` should become the runtime `forward_curve`
- the mock connector should expose a bounded `state_space` for scenario-aware
  tasks and framework experiments

## What Changed

### 1. MarketSnapshot now bridges selected FX and forecast inputs into runtime state

[/Users/steveyang/Projects/steveya/trellis/trellis/data/schema.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/schema.py)
now supports three important runtime bridges in `to_market_state(...)`:

- `forecast_curve=...`
  - narrows the snapshot to the selected named forecast curve
  - sets runtime `forward_curve` from that selected curve instead of leaving the
    default discount-derived forward curve in place
- `fx_rate=...`
  - narrows `fx_rates` to the selected pair
  - bridges the selected FX spot into runtime `spot`
  - records the pair in `underlier_spots` so FX tasks can reuse the same
    spot-like access path as equity tasks
- `state_space=...`
  - resolves either a concrete state-space object or a state-space factory
  - attaches the resulting `StateSpace` to the runtime `MarketState`

This makes task-selected market components materially affect the runtime market
state instead of only trimming metadata around it.

### 2. Resolver supports state-space overlays

[/Users/steveyang/Projects/steveya/trellis/trellis/data/resolver.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/resolver.py)
now accepts:

- `state_space=...`
- `state_spaces=...`
- `default_state_space=...`

and merges them the same way it already handled vol surfaces, FX rates, and
parameter packs.

### 3. Mock provider now supplies bounded state-space scenarios and FX-friendly forecast curves

[/Users/steveyang/Projects/steveya/trellis/trellis/data/mock.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/mock.py)
now adds:

- `default_state_space="macro_regime"`
- `state_spaces={"macro_regime": ...}`
- forecast-curve entries:
  - `USD-DISC`
  - `EUR-DISC`
  - `GBP-DISC`

The mock `macro_regime` state space is intentionally bounded:

- `base`
- `bull_repricing`
- `stress_repricing`

It is built from the compiled base market state using deterministic shifts to:

- discount curves
- forecast curves
- FX spots
- equity underlier spots
- vol surfaces
- credit curves

This is not meant to be production scenario generation. It is a sensible,
deterministic stand-in so scenario-aware tasks stop failing purely because the
mock connector had no `state_space`.

### 4. Task runtime now uses the new bridges directly

[/Users/steveyang/Projects/steveya/trellis/trellis/agent/task_runtime.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/task_runtime.py)
now passes task-selected:

- `forecast_curve`
- `fx_rate`
- `state_space`

directly into snapshot compilation instead of post-hoc patching pieces of the
runtime `MarketState`.

That fixes the earlier forecast-rate problem where selecting a forecast curve
still left `market_state.forward_curve` pointing at the discount-derived
default.

### 5. Trigger tasks now declare the intended mock market context

[/Users/steveyang/Projects/steveya/trellis/TASKS.yaml](/Users/steveyang/Projects/steveya/trellis/TASKS.yaml)
was updated so the current trigger set uses explicit market context:

- `T94`
  - `discount_curve=usd_ois`
  - `forecast_curve=EUR-DISC`
  - `fx_rate=EURUSD`
- `T105`
  - `discount_curve=usd_ois`
  - `forecast_curve=EUR-DISC`
  - `fx_rate=EURUSD`
- `T108`
  - `discount_curve=usd_ois`
  - `forecast_curve=EUR-DISC`
  - `fx_rate=EURUSD`
- `E25`
  - now also selects `forecast_curve=EUR-DISC`
  - asserts `spot` and `forward_curve` in addition to `fx_rates`

The previous “automatic model selection” version of `T94` was removed from the
task corpus because it was not a coherent pricing-build task. That framework
rejection case now lives as a deterministic task-runner stress test instead of
an invalid pricing task definition.

## Why This Matters

This tranche moves the mock connector from “data exists somewhere in the
snapshot” to “task-selected mock data actually becomes the runtime market
substrate the pricing/evaluation path sees.”

The important practical improvements are:

- FX tasks can now select a named FX pair and get both `fx_rates` and `spot`
- forecast-rate tasks now get the selected forecast curve as the runtime
  `forward_curve`
- scenario/state-space tasks now have a deterministic bounded mock input rather
  than a guaranteed missing-capability failure

## Validation

Focused slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_data/test_market_snapshot.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_data/test_mock.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_data/test_resolver.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py \
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

Result:

- `51 passed, 1 deselected`

Broader nearby slice:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_data/test_market_snapshot.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_data/test_mock.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_data/test_resolver.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_task_runtime.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_quant.py \
  -k 'not benchmark_existing_task_supports_generic_cached_transform_task' -q
```

Result:

- `71 passed, 1 deselected`

## Remaining Limits

- This closes the mock-data plumbing gap, not the full FX pricing substrate.
  `M3` still needs to add the first reusable FX primitive slice.
- The mock `state_space` is intentionally simple and deterministic. It is good
  enough for task/runtime stress and framework experiments, not a full scenario
  engine.
