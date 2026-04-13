# Semantic Platform Hardening Plan

## Purpose

This document defines the next cross-cutting implementation program for the
Trellis pricing stack.

The goal is to close the recurring low-level gaps that still cause canary and
fresh-build failures even after the recent PDE, Monte Carlo, calibration, and
synthetic-market-data workstreams.

This plan is intentionally architectural. It does not introduce one more
product route or one more narrow helper. It strengthens the shared semantic,
assembly, quote, and validation layers that every product-method combination
depends on.

## Why This Plan Exists

The recent workstreams improved important pieces of the stack:

- `EventAwarePDEIR` and the bounded event-aware PDE lane
- `EventAwareMonteCarloIR` and the bounded event-aware Monte Carlo lane
- calibration-layer model grammar and typed quote/materialization support
- model-generated synthetic market snapshots
- route-registry minimization planning

Those were necessary, but they did not fully remove the six failure patterns
that still dominate difficult tasks and canaries:

1. event semantics are still implemented family-by-family instead of being
   universal compiler knowledge
2. generated and checked adapters still own too much glue logic
3. helper surfaces are still too kernel-shaped and not yet comprehensive enough
   to make the correct path the easy path
4. quote normalization is stronger than before, but still too narrow for the
   real variety of quote subjects and conventions
5. the route layer still carries too much effective authority
6. validation still sometimes checks the wrong abstraction layer and therefore
   produces false repair loops or weak acceptance gates

The result is that Trellis can still have:

- valid product semantics
- valid numerical kernels
- and yet fail because the semantic-to-runtime assembly layers are too weak

## Repo-Grounded Current State

### Strong foundations already present

The repo already ships the main boundaries we should preserve:

- `SemanticContract`, `ValuationContext`, and `ProductIR`
- `EventAwarePDEIR` and `EventAwareMonteCarloIR`
- `EngineModelSpec` and typed market-binding compilation
- typed quote-map and calibration materialization surfaces
- typed route admissibility and validation bundles
- replay/provenance infrastructure

Relevant code and docs include:

- `trellis/agent/family_lowering_ir.py`
- `trellis/agent/dsl_lowering.py`
- `trellis/agent/valuation_context.py`
- `trellis/agent/market_binding.py`
- `trellis/models/calibration/quote_maps.py`
- `trellis/models/monte_carlo/event_aware.py`
- `trellis/models/pde/event_aware.py`
- `doc/plan/done__event-aware-pde-lane.md`
- `doc/plan/done__event-aware-monte-carlo-lane.md`
- `doc/plan/done__route-registry-minimization.md`
- `doc/plan/done__calibration-layer-model-grammar-unification.md`

### Remaining cross-cutting gaps

The remaining gaps are not isolated to one method family:

- event awareness is not yet one universal semantic/compiler surface
- helper design is still inconsistent across families
- quote semantics remain too thin for broad comparison and market-binding
  authority
- adapters still drift into hand-written glue even after recent stale-adapter
  cleanup
- route minimization is planned but not yet absorbed into a family-first
  backend-binding model
- validators still mix low-level primitive checks with higher-level helper
  contracts

These are platform problems, not product problems.

## What This Plan Is Not

This plan does not attempt to:

- replace `SemanticContract`, `ProductIR`, or family IRs with one universal IR
- remove all routes immediately
- remove all checked helper surfaces immediately
- solve arbitrary syntactically legal contracts with no admissibility boundary
- convert skills into runtime pricing helpers
- broaden immediately into every future numerical family

## Design Summary

The target shape is:

```text
SemanticContract
  -> universal event/control program
  -> ProductIR
  -> family IR
  -> comprehensive family helper kit
  -> backend binding catalog
  -> validation stack aligned to the same abstraction layer
```

The central idea is:

- move meaning upward into universal semantic/compiler surfaces
- move reusable glue into family helper kits
- move quote semantics into a broader quote authority
- move backend facts sideways into a thinner binding catalog
- keep validators aligned with whichever abstraction layer is authoritative

## Six-Strand Implementation Program

### 1. Universal event/control program

Event awareness should become universal compiler knowledge rather than a
family-local feature.

The target is one shared semantic layer above lattice, PDE, Monte Carlo, and
future families:

- `EventProgramIR`
- `ControlProgramIR`

Each family should then lower from that shared program into its own bounded
runtime representation.

This means every numerical family will "know" what events and control mean,
even if some families still reject some combinations through typed
admissibility.

### 2. Adapter architecture minimization

The stale-adapter tranche improved freshness handling, but it did not fully
solve adapter architecture.

The target is to minimize generated adapter logic by making adapters mostly a
declarative composition of:

- semantic binding
- schedule binding
- market/convention binding
- model/calibration binding
- helper assembly
- result extraction

Generated adapters should stop owning business logic.

### 3. Comprehensive family helper kits

Helpers should not be product-specific whenever a reusable family helper can be
built instead.

But they also should not stop at a raw pricing kernel.

The target helper shape is:

- schedule/timeline translation
- spec/convention hydration
- market-role binding
- model-parameter resolution
- quote/calibration normalization
- event payload construction
- runtime problem assembly
- execution wrapper

The correct path should be easier than the wrong path.

### 4. Quote semantics v2

The current `QuoteMapSpec` surface is useful but too thin.

The next abstraction should separate:

- quote family
- quote subject
- axis semantics
- unit semantics
- transform semantics
- settlement/numeraire assumptions

This must cover not only current rates/credit/vol workflows, but also broader
quote types such as:

- Black vs normal implied vol
- caplet vs forward-rate vs swap-rate vol
- price vs par rate vs spread vs hazard
- probability vs odds

The system should know what is being quoted, not only how to transform it.

### 5. Route/binding decentering

Routes should continue shrinking until they are mostly:

- backend-binding aliases
- admissibility envelopes
- replay/provenance IDs
- canary ownership handles

Meaning should instead come from:

- universal event/control program
- `ProductIR`
- family IR
- family helper assembly

This is the second route-minimization tranche, not a new route program.

### 6. Capability-aware validation stack

Validation should align with the same abstraction layer the compiler and
runtime actually use.

The target split is:

- semantic-contract validation
- family-IR validation
- assembly-contract validation
- market-regime validation
- comparison validation

Higher-level route/family helpers should subsume lower-level primitive checks.
Validation should stop forcing generated code to expose internals that the
public helper surface intentionally hides.

## Ordered Delivery Queue

1. umbrella: semantic platform hardening
2. universal event/control program above family lowering
3. adapter architecture minimization and stale-surface closure
4. comprehensive family helper kits
5. quote semantics v2 and generalized quote authority
6. route/binding decentering tranche 2
7. capability-aware validation stack
8. docs, observability, and compatibility cleanup

## Critical Path

The critical path is:

`event/control program -> helper kits + quote semantics -> adapter minimization -> validation stack -> cleanup`

Route/binding decentering should run in parallel where possible, but it should
remain aligned with the stronger family/helper boundaries rather than racing
ahead of them.

## Relationship to Existing Workstreams

This plan is intentionally a coordination umbrella above several existing
strands:

- event-aware PDE lane
- event-aware Monte Carlo lane
- route-registry minimization
- calibration-layer model grammar unification
- instrument-identity phaseout

It should not replace those plans. It should absorb the remaining cross-cutting
gaps that they exposed.

## Agent Intake Bundle

Each coding agent assigned to this workstream should start with:

- this plan doc
- `AGENTS.md`
- `doc/plan/done__event-aware-pde-lane.md`
- `doc/plan/done__event-aware-monte-carlo-lane.md`
- `doc/plan/done__route-registry-minimization.md`
- `doc/plan/done__calibration-layer-model-grammar-unification.md`
- `doc/plan/done__instrument-identity-phaseout.md`
- `docs/quant/pricing_stack.rst`
- `docs/developer/dsl_system_design_review.md`
- the current Linear ticket body and any upstream blockers

Primary code paths in scope:

- `trellis/agent/family_lowering_ir.py`
- `trellis/agent/dsl_lowering.py`
- `trellis/agent/valuation_context.py`
- `trellis/agent/market_binding.py`
- `trellis/agent/semantic_validation.py`
- `trellis/agent/semantic_validators/`
- `trellis/agent/route_registry.py`
- `trellis/models/calibration/quote_maps.py`
- `trellis/models/monte_carlo/event_aware.py`
- `trellis/models/pde/event_aware.py`

## Linear Ticket Mirror

Status mirror last synced: `2026-04-09`

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-733` | Done |

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-734` | Universal event/control program above family lowering | Done |
| `QUA-735` | Comprehensive family helper kits | Done |
| `QUA-736` | Quote semantics v2 and generalized quote authority | Done |
| `QUA-739` | Adapter architecture minimization and stale-surface closure | Done |
| `QUA-737` | Route/binding decentering tranche 2 | Done |
| `QUA-738` | Capability-aware validation stack | Done |
| `QUA-740` | Docs, observability, and compatibility cleanup | Done |
