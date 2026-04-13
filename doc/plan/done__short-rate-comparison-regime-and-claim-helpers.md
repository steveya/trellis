# Short-Rate Comparison Regime And Claim Helpers

## Purpose

This plan covers two linked follow-on problems exposed by `T01`:

1. task-level model assumptions are still too often interpreted from prose by
   individual comparator lanes instead of being materialized once as typed
   market/model objects
2. the current zero-coupon-bond option helpers are useful, but they are still
   too instrument-shaped to be the long-term reusable short-rate claim layer

The goal is to fix both at the same time so `T01` and future short-rate
comparison tasks stop failing for assembly reasons.

## Current Status

As of April 10, 2026, the proving canary `T01` is passing on the new runtime
path, and both the validation / exact-binding recovery slice (`QUA-751`) and
the docs / observability cleanup slice (`QUA-750`) are closed in Linear:

- the task comparison assumptions are materialized once into a typed
  short-rate comparison regime on `MarketState`
- the ZCB analytical and tree wrappers resolve the same regime/claim layer
- the lower-layer planner/runtime path now enforces the broader rule that an
  explicit family such as `zcb_option` must not be widened back into a generic
  `european_option` schema by text heuristics

That last rule is intentionally broader than `T01`: it is now also defended in
the lower planner/runtime seams so basket/barrier/Asian families and explicit
credit families do not drift back into generic option specializations when a
concrete family is already known.

## Why Now

The fresh `T01` failure is not a numerical-kernel problem.

The analytical Jamshidian lane succeeds, but the two tree comparators fail
because the build/runtime layer does not carry the comparison regime cleanly:

- the Ho-Lee comparator drifts into an unapproved module import instead of
  staying on the checked tree helper surface
- the Hull-White comparator copies `sigma = 0.01` from the task prose instead
  of resolving the same assumption through a shared market/model authority

That tells us the next work is not another product-local patch. It is:

- typed comparison-regime materialization
- reusable short-rate claim helpers underneath the current ZCB wrappers

## Target Outcome

For bounded short-rate comparison tasks, Trellis should:

1. compile task-level comparison assumptions into typed market/model objects
2. attach those objects to `MarketState` and related runtime metadata
3. resolve method-specific inputs from those objects through shared helper kits
4. keep analytical and tree wrappers as thin consumers of the same shared layer

The proving target is `T01`:

- `jamshidian`
- `hull_white_tree`
- `ho_lee_tree`

All three should price from the same typed comparison regime rather than from
copied literals or route-local glue.

## Design Summary

The intended shape is:

```text
task comparison assumptions
  -> typed comparison regime
  -> materialized market/model objects on MarketState
  -> shared short-rate regime resolver
  -> shared short-rate claim helpers
  -> thin analytical/tree wrappers
  -> comparison-aware validation and exact binding
```

Two principles drive the plan:

- task assumptions should become real market/model objects, not generator prose
- reusable short-rate claim helpers should sit below `zcb_option.py` and
  `zcb_option_tree.py`, not inside them

## Strand A: Comparison Regimes As Market/Model Objects

### Problem

Today a task can state a bounded model assumption such as:

- flat short-rate vol `sigma = 0.01`
- mean reversion `a = 0.1`
- flat yield curve `5%`

but those assumptions are still too easy for individual comparator lanes to
copy into generated code.

### Target

Materialize those assumptions as typed runtime objects using the same broad
market-object path that Trellis already uses for more realistic generated or
calibrated market inputs.

For the first slice, the runtime should support bounded comparison-regime
objects for:

- flat short-rate volatility assumptions
- flat short-rate mean-reversion assumptions where applicable
- flat discount-curve assumptions already required by the task fixture

### Requirements

- the object must carry semantics, not only numbers
- the same object family should be reusable for future synthetic or calibrated
  market/model packs
- helper/resolver code should obtain the right model point from the object
  instead of reading literals from prose

### First proving case

`T01` should materialize a flat short-rate comparison regime and make it
available to all three comparison methods through `MarketState` and runtime
metadata.

## Strand B: Reusable Short-Rate Claim Helper Layer

### Problem

The repo already has useful ZCB helpers:

- `trellis/models/zcb_option.py`
- `trellis/models/zcb_option_tree.py`

but they still bundle multiple concerns:

- regime resolution
- strike normalization
- discount-bond claim normalization
- analytical/tree assembly

That is too instrument-shaped to be the shared abstraction.

### Target

Extract a reusable short-rate claim/helper layer below the current ZCB
wrappers.

The reusable layer should cover:

1. short-rate regime resolution
2. discount-bond claim normalization
3. short-rate tree assembly helpers
4. affine/Jamshidian-style analytical resolved-input helpers where valid

The current ZCB wrappers should remain, but become thin compositions over the
shared layer.

### First proving case

`price_zcb_option_jamshidian(...)` and `price_zcb_option_tree(...)` should both
be refactored onto the shared short-rate regime and claim helpers, with the
tree path able to consume the same typed comparison-regime objects as the
analytical lane.

## Ordered Delivery Queue

### `QUA-747` Comparison regimes: typed short-rate assumption objects

Objective:

Compile bounded short-rate comparison assumptions from task/runtime inputs into
typed market/model objects instead of leaving them as prose-only hints.

Scope:

- runtime/task comparison-regime materialization
- flat short-rate volatility/model assumptions for the first slice
- runtime metadata/provenance for the attached comparison regime

Acceptance:

- `T01` comparison assumptions are materialized as typed runtime objects
- no comparator lane needs to rediscover `sigma = 0.01` from prose

### `QUA-748` Helper layers: shared short-rate regime resolver and bond-claim kit

Objective:

Extract reusable short-rate regime resolution and discount-bond claim
normalization from the current ZCB helpers.

Scope:

- shared helper modules under `trellis/models/`
- regime resolution for short-rate methods
- discount-bond claim normalization

Acceptance:

- reusable short-rate helper layer exists below the ZCB wrappers
- the layer can serve both analytical and tree consumers

### `QUA-749` Short-rate wrappers: migrate ZCB analytical and tree helpers

Objective:

Refactor `zcb_option.py` and `zcb_option_tree.py` into thin wrappers over the
new shared short-rate helper kit.

Acceptance:

- wrappers shrink materially
- tree and analytical helpers consume the same regime/claim normalization layer
- direct route-local literal binding is no longer needed

### `QUA-750` Docs, observability, and canary hardening

Objective:

Document the new comparison-regime path and short-rate helper layer, update
trace and observability surfaces, and sync the canary mirror after `T01`
recovery.

Acceptance:

- official docs updated
- trace/runtime metadata exposes the typed short-rate comparison regime
- canary plan reflects `T01` recovery on the new architecture

### `QUA-751` Validation and exact binding: recover `T01`

Objective:

Move the `T01` comparator routes onto the shared comparison-regime and
short-rate helper path, then close the canary.

Acceptance:

- `ho_lee_tree` stays on the checked ZCB tree helper surface
- `hull_white_tree` resolves volatility from the typed comparison regime rather
  than a local literal
- canary plan reflects `T01` recovery on the new architecture
- `T01` passes live

## Critical Path

The intended order is:

`QUA-747 -> QUA-748 -> QUA-749 -> QUA-751 -> QUA-750`

This order matters:

- the comparison regime should exist before helpers consume it
- the shared helper layer should exist before wrappers are migrated
- validation/exact-binding cleanup should happen after the runtime authority is
  real, not before

## Relationship To Existing Plans

This workstream is a concrete follow-on under:

- `doc/plan/done__semantic-platform-hardening.md`
- `doc/plan/active__general-helper-layer-extraction.md`

It also becomes the durable `T01` recovery path for:

- `doc/plan/done__canary-suite-stabilization.md`

## Acceptance Surface

The workstream is done when:

- `T01` passes live
- all three comparison methods consume the same typed comparison regime
- no method needs to hardcode short-rate volatility literals locally
- the ZCB wrappers are thin compositions over shared short-rate helper layers
