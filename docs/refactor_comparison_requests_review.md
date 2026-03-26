# Comparison Request and Canonical Capability Review

Date: 2026-03-25

## Goal

Start the next platform tranche by implementing two foundational changes:

1. remove legacy market-data capability names from the core runtime/compiler vocabulary
2. add a comparison-aware request/compiler surface for multi-method tasks

## What Changed

### Canonical capability vocabulary

The core market-data capability vocabulary now uses canonical object-shaped
names:

- `discount_curve`
- `forward_curve`
- `black_vol_surface`
- `local_vol_surface`
- `credit_curve`
- `fx_rates`
- `spot`
- `state_space`
- `jump_parameters`
- `model_parameters`

The core/runtime/compiler surfaces now emit those names through:

- [capabilities.py](/Users/steveyang/Projects/steveya/trellis/trellis/core/capabilities.py)
- [market_state.py](/Users/steveyang/Projects/steveya/trellis/trellis/core/market_state.py)
- [payoff_pricer.py](/Users/steveyang/Projects/steveya/trellis/trellis/engine/payoff_pricer.py)
- [quant.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/quant.py)
- [decompose.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/knowledge/decompose.py)

Older names are still normalized at ingestion boundaries so historical tasks and
lessons do not break immediately.

### Comparison-aware compiler layer

The platform request compiler now has a dedicated request-intent layer for
multi-method tasks in:

- [platform_requests.py](/Users/steveyang/Projects/steveya/trellis/trellis/agent/platform_requests.py)

New request/compiler types:

- `ComparisonSpec`
- `ComparisonMethodPlan`
- `make_comparison_request(...)`

`compile_platform_request(...)` can now compile a single product request into
multiple method-specific plans instead of forcing everything into one route.

## Why This Matters

This is the structural prerequisite for `T74`-style tasks.

`European equity call: 5-way (tree, PDE, MC, FFT, COS)` is not just a product
description. It is a product description plus comparison intent. That intent
does not belong in `ProductIR`; it belongs in the request/compiler layer.

## Validation

Focused slice:

- `46 passed`

Broader regression slice:

- `120 passed, 1 deselected`

The deselected test is the already-known unrelated generic cached-transform
benchmark collision.
