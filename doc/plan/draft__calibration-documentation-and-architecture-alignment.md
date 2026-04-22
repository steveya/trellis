# Calibration Documentation And Architecture Alignment Plan

## Status

Active execution mirror for the filed calibration-documentation alignment
slice.

The architecture-alignment work is tracked primarily by `QUA-947` under the
calibration-sleeve umbrella `QUA-946`. This document remains the repo-local
spec for that ticket and should stay synchronized with both Linear and the main
calibration hardening plan.

Status mirror last synced: `2026-04-21`

## Linked Context

- `QUA-946` Calibration sleeve: Trellis-native industrial hardening program
- `QUA-947` Calibration architecture: align Trellis-native docs, runtime
  vocabulary, and plan mirror
- `doc/plan/draft__calibration-sleeve-industrial-hardening-program.md`

## Linear Ticket Mirror

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This file is the repo-local mirror for the documentation-alignment slice.
- Do not mark the row `Done` here before `QUA-947` is actually closed.
- Keep the vocabulary in this file, the main calibration hardening plan, and
  the calibration docs synchronized in the same closeout.

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-946` Calibration sleeve: Trellis-native industrial hardening program | Backlog |

### Documentation Slice

| Ticket | Status | Scope |
| --- | --- | --- |
| `QUA-947` | Done | Trellis-native architecture framing, market-object-first vocabulary, and calibration-doc alignment |

## Purpose

This document aligns the calibration documentation with the desired end state
of the Trellis calibration sleeve.

The immediate problem is not that the current mathematical framework is weak.
It is that the strongest current ideas are spread across multiple documents and
can easily be summarized in a misleading way as "a general multivariate SDE
framework." That shorthand is too narrow and does not reflect the Trellis
architecture we actually want.

The calibration sleeve is not a separate library. It is part of Trellis and
should leverage Trellis abstractions end to end.

## Reviewed Source Documents

This alignment plan is grounded in the following checked-in docs:

- `docs/mathematical/calibration.rst`
- `docs/unified_pricing_engine_model_grammar.md`
- `docs/developer/composition_calibration_design.md`
- `doc/plan/draft__calibration-sleeve-industrial-hardening-program.md`

## Decision Summary

The documentation should present the calibration sleeve as:

1. a Trellis-native market inference layer
2. organized into market reconstruction, model compression, and hybrid
   composition
3. using quote maps, market binding, solve requests, and runtime
   materialization as first-class Trellis abstractions
4. treating generic multivariate SDEs as one latent-state representation family
   inside the model layer, not as the top-level unifier of the sleeve

## What The Current Docs Already Get Right

### 1. The mathematical note rejects the "one master SDE" framing

`docs/unified_pricing_engine_model_grammar.md` already says the right unifier
is not one master SDE and instead identifies the pricing operator, latent
state, generator, contract family, quote map, and calibration objective as the
reusable abstraction.

This is the strongest current mathematical framing and should be preserved.

### 2. Quote-space semantics are already treated as first-class

Both the mathematical note and the calibration docs correctly emphasize that:

- prices are core
- quotes are transforms
- the calibration layer needs both forward and inverse quote maps

This is an excellent foundation for a desk-grade calibration sleeve.

### 3. The developer design already anchors calibration onto Trellis runtime objects

`docs/developer/composition_calibration_design.md` correctly connects
calibration to:

- `MarketState`
- quote-map semantics
- runtime materialization
- generated and reused Trellis workflows

This is the right architectural direction.

## Main Documentation Gaps

### A. The layered end state is not stated explicitly enough

The docs do not yet clearly separate:

- market reconstruction
- model compression
- hybrid composition

Without that layering, readers can wrongly infer that every calibration should
end in one reduced-form stochastic model.

### B. The market-object-first view is underemphasized

The current calibration docs still read primarily as "fit model parameters to
market quotes." That is true for some workflows, but for liquid products the
authoritative calibrated result is often a Trellis market object:

- curve
- vol surface or cube
- credit curve
- correlation surface

The docs should say that directly.

### C. The mathematical framework is easy to mislabel

The mathematical note is broader than a multivariate SDE note:

- it starts from the pricing operator
- uses quote maps explicitly
- treats calibration as an inverse problem in quote space
- allows generators, SPDEs, lifted states, and non-diffusion structure

The docs should make that broader framing more obvious.

### D. The Trellis-native integration should be more explicit

The docs should more clearly state that the calibration sleeve works through
existing Trellis abstractions:

- `SemanticContract`
- `ValuationContext`
- `MarketBindingSpec`
- `MarketState`
- `quote_maps`
- `solve_request`
- `materialization`

This would prevent the calibration workstream from being interpreted as a
parallel engine.

### E. The current docs do not yet map product families to authoritative outputs

The docs should have one explicit product-family table showing:

- calibration instruments
- authoritative calibrated output
- optional reduced-model output
- downstream Trellis consumers

Without this, the architecture remains too implicit.

## Documentation Target Shape

### 1. `docs/mathematical/calibration.rst`

This should become the practical and user-facing calibration architecture
document.

It should lead with:

- the three-layer stack
- the market-object-first rule
- the relationship between quote maps, solve requests, and materialized runtime
  objects

Then it can describe the current shipped workflows.

### 2. `docs/unified_pricing_engine_model_grammar.md`

This should remain the deep mathematical reference for latent-state and
generator-based models, but the opening should be interpreted as:

- a model-layer and hybrid-layer framework
- not the sole top-level description of the sleeve
- not a separate engine outside Trellis

It should explicitly point readers back to the calibration architecture and
runtime materialization docs.

### 3. `docs/developer/composition_calibration_design.md`

This should stay the Trellis implementation bridge.

It should explicitly connect:

- calibration contracts
- quote maps
- `MarketState` capability materialization
- chained calibrations
- Trellis planner and compiler surfaces

to the three-layer end-state architecture.

## Required Documentation Changes

### Phase 1: Calibration architecture framing

1. Add a front section to `docs/mathematical/calibration.rst` defining:
   - market reconstruction
   - model compression
   - hybrid composition
2. Add an explicit statement that simple liquid products often calibrate
   directly to Trellis market objects rather than to reduced-model parameters.
3. Add a product-family mapping table.

### Phase 2: Mathematical note alignment

1. Add a short framing note near the top of
   `docs/unified_pricing_engine_model_grammar.md` explaining that this is the
   latent-state and generator framework within the broader Trellis calibration
   sleeve.
2. State explicitly that the document is not proposing a separate calibration
   library or one master-SDE abstraction for all workflow layers.
3. Cross-link to `docs/mathematical/calibration.rst` and
   `docs/developer/composition_calibration_design.md`.

### Phase 3: Trellis-native developer integration

1. Expand `docs/developer/composition_calibration_design.md` with a short
   section mapping the three calibration layers onto Trellis abstractions.
2. Show where market reconstruction outputs land on `MarketState`.
3. Show how later model compression and hybrid composition steps depend on
   those materialized outputs.

### Phase 4: Runtime and plan alignment

1. Keep the canonical plan doc
   `doc/plan/draft__calibration-sleeve-industrial-hardening-program.md`
   synchronized with the doc wording.
2. Ensure planner and canonical model-grammar registry language use the same
   distinctions between priced-only, reconstructed-market-object, and
   reduced-model workflows.

## Recommended Product-Family Mapping Table

The docs should include a table close to this form.

| Product family | Liquid inputs | Primary calibrated output | Optional reduced-model output |
| --- | --- | --- | --- |
| OIS / IRS / FX swap | deposits, OIS, IRS, FX swaps, basis swaps | discount and forecast curves, basis structures | short-rate or curve-dynamics factors |
| Vanilla equity / FX options | option prices or quoted vols | implied-vol surface or cube | local vol, Heston, stochastic-vol models |
| Caps/floors / swaptions | option prices or quoted vols | caplet strips, swaption cubes | SABR, Hull-White, G2++, LMM |
| CDS | running and upfront CDS quotes | credit curve | reduced-form factor model |
| Basket credit tranches | tranche spreads | base-correlation or correlation surface | copula or factor correlation model |
| Hybrid liquid sets | linked single-asset objects | cross-asset binding and correlation objects | hybrid state model |

## Acceptance Criteria

The documentation alignment work is complete when:

- the docs no longer encourage the shorthand "general multivariate SDE
  framework" as the top-level description of the sleeve
- `docs/mathematical/calibration.rst` clearly presents the three-layer Trellis
  calibration stack
- the mathematical note is clearly positioned as a latent-state and
  generator-layer reference inside Trellis
- the developer docs explicitly connect calibration to Trellis abstractions and
  runtime materialization
- the plan docs and docs use the same end-state vocabulary

## Non-Goals

This alignment plan does not by itself:

- widen the shipped numerical coverage
- claim that the current workflows already meet the end state
- turn the mathematical note into the only source of calibration truth

The goal is alignment of architecture and documentation so later implementation
work lands on a clear target.
