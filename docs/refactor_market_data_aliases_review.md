# Market Data Alias and Synthetic Pack Review

Date: 2026-03-25

## Goal

Finish the high-value market-data fixes needed to improve previously failed task
runs without relying on a live market-data vendor.

The requested changes were:

1. route task runtime market-state creation through `resolve_market_snapshot(source="mock")`
2. bridge legacy capability names onto the modern runtime capability model
3. extend the snapshot/runtime schema with generic underlier spots
4. add synthetic local-vol and jump/model-parameter packs for research tasks

## What Changed

`build_market_state()` in
[task_runtime.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/task_runtime.py)
was already moved onto the simulated snapshot path during the earlier `T74`
substrate repair.

This tranche completed the remaining three items:

- [capabilities.py](/Users/steveyang/Projects/steveya/trellis/trellis/core/capabilities.py)
  now treats `discount_curve`, `yield_curve`, `risk_free_curve`,
  `forward_rate_curve`, `volatility_surface`, and `black_vol_surface` as
  aliases over the modern capability inventory.
- [market_state.py](/Users/steveyang/Projects/steveya/trellis/trellis/core/market_state.py)
  now carries:
  - `spot`
  - `underlier_spots`
  - `local_vol_surface`
  - `local_vol_surfaces`
  - `jump_parameters`
  - `jump_parameter_sets`
  - `model_parameters`
  - `model_parameter_sets`
- [schema.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/schema.py)
  now supports named/default selection for those same market-data families.
- [mock.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/mock.py)
  now provides deterministic stand-in underlier spots, local-vol surfaces,
  Merton jump parameters, and Heston-style model parameters.
- [resolver.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/resolver.py)
  now preserves and merges those richer simulated components.

## Why This Matters

Several failed task reruns were not blocked by missing mathematics; they were
blocked by thin market-data scaffolding and legacy capability names.

This change does not claim full live market-data support. It does make the
offline/simulated market-data path much closer to what the task harness and
agent expect when they ask for:

- discount curves under legacy names
- forward-rate curves
- volatility surfaces
- underlier spots
- local-vol capability
- jump/model parameter packs

## Validation

Focused slice:

- `77 passed, 1 deselected`

Broader dependent regression slice:

- `113 passed, 1 deselected`

The deselected test is the unrelated generic cached-transform benchmark that is
already known to depend on a stale `_agent/buildapayoff.py` collision.
