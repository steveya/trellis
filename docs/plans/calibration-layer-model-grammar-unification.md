# Calibration-Layer Model Grammar Unification Plan

## Purpose

This document defines a bounded implementation plan for strengthening the
Trellis calibration layer with a typed model-grammar surface.

The goal is not to replace the shipped semantic-contract stack with a universal
solver grammar. The goal is to close the current gap between:

1. semantic contract and family lowering on one side, and
2. typed calibration and runtime materialization on the other.

This plan is written as an implementation planning document and is already
decomposed into Linear issues.

## Decision Summary

The direction is:

- keep `SemanticContract`, `ProductIR`, and family-specific lowering IRs as the
  product-semantic authority
- keep `CalibrationContract` and `SolveRequest` as the solver-facing
  calibration substrate
- add a bounded calibration-layer model grammar between them
- make multi-curve rates semantics explicit at the model-grammar level rather
  than leaving discount/forecast roles implicit in route-local code
- make quote semantics explicit through typed quote maps
- make calibrated runtime materialization explicit instead of route-local
  payload conventions
- add canonical model-grammar registry entries so the planner can look up
  supported model semantics instead of reconstructing them from scratch
- upgrade mock/proving-market generation so synthetic snapshots are consistent
  with bounded rate, credit, and volatility model assumptions

## Why This Plan Exists

Trellis already has two strong architectural layers:

- a typed semantic-contract, valuation-context, and family-lowering boundary
- a typed solve-request and replay/provenance substrate for calibration

What remains weaker is the middle layer that decides:

- what model semantics are actually in force
- which calibration is required before pricing
- how prices map to market quotes and back
- how calibrated outputs land on `MarketState`

That middle layer is currently split across:

- `trellis/agent/valuation_context.py`
- `trellis/agent/calibration_contract.py`
- `trellis/core/market_state.py`
- `trellis/models/calibration/`
- `trellis/curves/credit_curve.py`
- `trellis/models/vol_surface.py`
- process-specific runtime binding helpers

The result is avoidable planning ambiguity, repeated quote-convention logic,
and calibration-class failures that are larger than ordinary route-local bugs.

## Repo-Grounded Current State

### Shipped boundaries to preserve

Trellis already ships:

- typed contract meaning through `SemanticContract`
- typed valuation policy and market-binding compilation through
  `ValuationContext`, `RequiredDataSpec`, and `MarketBindingSpec`
- `ProductIR` plus family-specific lowering IRs as the executable compiler
  boundary
- typed calibration steps through `CalibrationContract`
- typed solver execution through `SolveRequest`, `ObjectiveBundle`, and replay
  provenance

This plan must build on those surfaces rather than replace them.

### The missing authority surface

The current repo still has these gaps:

- `ValuationContext.model_spec` is too shallow to act as a reliable
  calibration-planning authority
- quote semantics are still spread across route-local helpers and workflow code
- calibrated runtime objects materialize onto `MarketState` through a
  heterogeneous mix of fields and payload dicts
- credit calibration still sits outside the strongest part of the typed
  calibration substrate
- supported model semantics are not yet recorded in canonical knowledge the way
  imports, routes, and decompositions are
- the current synthetic/mock market snapshots are useful, but they are still
  generated from relatively primitive regime heuristics rather than from one
  bounded model-grammar authority for rates, credit, and volatility

## What This Plan Is Not

This plan is intentionally bounded.

It does not attempt to:

- introduce a universal pricing IR for all solver families
- replace `SemanticContract`, `ProductIR`, or family-specific lowering IRs
- make `StochasticProcess` the universal model authority
- widen immediately into HJM/SPDE, rough-vol, dynamic filtering, or Bayesian
  estimation
- redesign existing pricing math unless a test proves an actual bug

## Design Guardrails

The implementation should follow these rules.

### 1. The model grammar is declarative, not a live process object

Do not make the new surface equivalent to
`ModelSpec(state_model: StochasticProcess)`.

That would be too narrow for the shipped Trellis stack, because current routes
span analytical, lattice, PDE, Monte Carlo, transform, and helper-backed
surfaces. The authority object should be a serializable specification, not a
runtime engine instance.

### 2. The new layer is a calibration and binding authority, not a universal solver

Do not treat the model grammar as the new end-to-end pricing IR.

The shipped boundary remains:

`SemanticContract -> ValuationContext -> ProductIR -> family lowering IR -> checked helper/kernel`

The new layer should strengthen `ValuationContext`, calibration planning, and
runtime materialization without reopening the product-semantic boundary.

### 3. Existing calibration substrate is evolved, not replaced

`CalibrationContract` and `SolveRequest` already exist and should remain the
authority for typed calibration execution. The new work should make them easier
to invoke correctly, not create a parallel calibration stack.

### 4. Quote maps must be first-class and two-sided

`QuoteMap` is not just a formatting helper. It must carry:

- price-to-quote semantics
- quote-to-price or inverse-transform semantics where applicable
- residual and warning conventions
- provenance and convention metadata

### 5. The first tranche stays inside current supported workflows

The first migrated set is:

- rates bootstrap and Hull-White under explicit multi-curve discount/forecast
  semantics
- SABR
- Heston
- local vol
- reduced-form single-name credit

### 6. Multi-curve is a requirement, not a separate rewrite target

This plan does not own first-time implementation of the multi-curve rates
foundation. That work already exists in substantial form.

What this plan does own is making multi-curve semantics first-class in the
new model-grammar layer. For migrated rates workflows, the authority surface
must explicitly distinguish:

- discount-curve role
- forecast-curve role
- quote/calibration conventions that depend on both
- calibrated rates-model parameters that depend on the selected curve set

Unfinished multi-curve cleanup or hardening work may proceed in parallel, but
the rates slices in this epic must not regress or bypass the existing
multi-curve foundation.

## Target Codebase Shape

The bounded target shape is:

```text
SemanticContract
  + ValuationContext
    + EngineModelSpec
    + QuoteMapSpec
    + calibration requirements
  -> ProductIR
  -> family lowering IR
  -> checked numerical route

Calibration workflow
  -> CalibrationContract
  -> SolveRequest / ObjectiveBundle
  -> typed result
  -> calibrated-object materialization onto MarketState
```

Conceptually, the new surfaces are:

- `EngineModelSpec`
- `PotentialSpec`
- `SourceSpec`
- `QuoteMapSpec`
- typed calibrated-object materialization/binding
- canonical model-grammar registry entries

The names may change during implementation, but the ownership boundary should
not.

## Initial Implementation Slices

### Slice 1: Typed engine model spec

Add a bounded, serializable engine-model surface for supported calibration
workflows and thread it through `ValuationContext` and binding summaries.

For rates workflows, this slice must treat multi-curve discount and forecast
roles as explicit model-binding inputs rather than one blended "curve" concept.

### Slice 2: Explicit quote maps

Extract `Price`, implied-vol, par-rate, spread, and hazard quote semantics into
typed quote-map surfaces shared across migrated calibration workflows.

For rates workflows, quote maps must preserve multi-curve discounting and
forecasting assumptions explicitly.

### Slice 3: Typed calibrated-object materialization

Normalize how calibrated model parameter packs, vol surfaces, local-vol
surfaces, and credit curves land on `MarketState`.

For rates workflows, this includes preserving selected discount/forecast curve
roles and provenance through the calibrated runtime binding path.

### Slice 4: Typed credit calibration

Bring single-name credit calibration onto the same typed substrate as rates and
vol through explicit potential binding and quote semantics.

### Slice 5: Canonical model-grammar registry

Record the bounded supported model semantics in canonical knowledge and wire the
registry into planning and gap-check surfaces.

### Slice 6: Model-consistent synthetic mock market data

Upgrade the mock/proving-market path so synthetic discount/forecast curves,
credit curves, and volatility objects are generated from bounded model-aware
assumptions rather than only flat or regime-heuristic stand-ins.

### Slice 7: Validation and docs hardening

Lock the new boundary down with replay, benchmark, canary, and documentation
coverage.

## Critical Path

The critical path is:

`QUA-687 -> QUA-688 -> QUA-689 -> QUA-690 -> QUA-691 -> QUA-693 -> QUA-692`

This order is intentional:

- typed model semantics first
- quote semantics second
- runtime materialization third
- credit follows once the shared abstractions exist
- canonical registry comes after the authority surfaces stabilize
- model-consistent synthetic fixtures follow once the authority surfaces they
  should reflect are explicit
- validation and docs hardening close the loop

## Parallel Workstream Contract

This plan is intended to run in parallel with
`docs/plans/multi-curve-rates-completion.md`.

Recommended two-agent split:

- Agent A owns the multi-curve queue beginning at `QUA-356`.
- Agent B owns this queue beginning at `QUA-687`.

Shared checkpoints:

- Agent B can begin `QUA-687` and `QUA-688` immediately.
- Before rates-specific work in `QUA-689`, Agent B should read the latest
  closeout or progress note for `QUA-356`.
- Before finalizing rates registry entries or synthetic rates fixtures in
  `QUA-691` and `QUA-693`, Agent B should read the latest closeout or progress
  note for `QUA-381` and `QUA-358`.
- Before `QUA-692` closeout, both workstreams should reconcile replay, docs,
  and provenance expectations.

This means the calibration-layer model-grammar workstream does not need the
entire multi-curve queue to be done before it starts. It does need the relevant
rates outcomes before it claims the rates-side boundary is complete.

## Agent Intake Bundle

Each local or cloud coding agent assigned to this workstream should start with:

- this plan doc
- `docs/plans/multi-curve-rates-completion.md` for the rates-side dependency
  story
- `AGENTS.md`
- the current Linear ticket body plus any hard blockers
- `docs/mathematical/calibration.rst`
- `docs/developer/composition_calibration_design.md`
- the current code paths under `trellis/agent/`, `trellis/core/market_state.py`,
  `trellis/models/calibration/`, `trellis/curves/`, and `trellis/models/`

## Linear Ticket Mirror

These tables mirror the current Linear tickets for this plan and their intended
implementation order.

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This file is the repo-local mirror for subsequent agents.
- Implement the earliest ticket in the ordered queue below whose status is not
  `Done` and whose blockers are satisfied.
- Do not mark a row `Done` here before the corresponding Linear issue is
  actually closed.
- When a ticket changes behavior, APIs, runtime workflow, or operator
  expectations, update the relevant official docs in the same implementation
  closeout.

Status mirror last synced: `2026-04-06`

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-686` Semantic model grammar: calibration-layer unification for model specs, quote maps, and bindings | Backlog |

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-687` | Typed engine model spec and valuation-context boundary | Done |
| `QUA-688` | Explicit quote maps and calibration-target transforms | Done |
| `QUA-689` | Typed calibrated-object materialization onto `MarketState` | Done |
| `QUA-690` | Typed credit calibration and potential-term binding | Done |
| `QUA-691` | Canonical model-grammar registry and planner lookup | Done |
| `QUA-693` | Model-consistent synthetic mock snapshots for rates, credit, and vol | Backlog |
| `QUA-692` | Replay, benchmark, canary, and docs hardening | Backlog |

## Validation Posture

This plan should be implemented with explicit local, regional, and global
validation.

### Local

- new type validation and serialization tests
- quote-map transform and inverse-transform tests
- calibrated-object materialization tests

### Regional

- valuation-context to calibration-workflow handoff
- calibration result to `MarketState` materialization
- CDS calibration to pricing handoff on the migrated credit slice
- planner and gap-check rendering against the canonical model-grammar registry

### Global

- replay contracts for migrated calibration workflows
- benchmark or throughput baselines where the new boundary changes workflow
  cost materially
- canaries for missing calibration binding, unsupported quote maps, and invalid
  runtime materialization
- synthetic benchmark and mock/proving fixtures that remain consistent with the
  bounded model grammar for rates, credit, and volatility

## Official Docs Expected To Change

If this plan is implemented as intended, the main official docs that should be
updated during closeout are:

- `docs/quant/pricing_stack.rst`
- `docs/quant/contract_algebra.rst`
- `docs/mathematical/calibration.rst`
- `docs/developer/composition_calibration_design.md`
- `docs/developer/overview.rst`
- `docs/market_data_workstream.md`
- `docs/user_guide/market_data.rst`
- `docs/user_guide/pricing.rst`

## Deferred Scope

The following remain explicitly deferred until the bounded calibration-layer
grammar is stable:

- dynamic state estimation and filtering
- function-space and SPDE model grammar entries
- rough-vol and lifted non-Markovian grammar beyond current shipped workflows
- generalized hybrid-model families
- any attempt to replace family-specific lowering with one flat universal IR
