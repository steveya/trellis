# Route Registry Minimization Plan

## Purpose

This document defines the next route-cleanup workstream for Trellis.

The goal is not to delete every route identifier immediately. The goal is to
minimize the route mechanism until it is no longer a first-class synthesis
abstraction.

The target end state is:

- product semantics come from `SemanticContract`
- executable problem shape comes from `ProductIR` plus family-specific IRs
- numerical construction comes from bounded family assembly layers
- backend reuse comes from exact helper/kernel bindings
- "routes" survive only as compatibility aliases and backend-binding records

In other words, route IDs should stop deciding what a product *means* and only
help answer:

- is there an exact checked backend fit?
- what bounded capability envelope does that backend support?
- what provenance / validation / canary ownership attaches to that backend?

## Why This Plan Exists

The repo has already completed the first route-decentering umbrella under
`QUA-546`. That work explicitly defined routes as backend-binding and safety
infrastructure rather than the primary synthesis contract.

That architectural shift is correct, but the current implementation is still
transitional:

- `route_registry.py` still mixes several concerns in one table
- live tests still overfit to route inventory size and exact route lists
- discovered routes can still drift into the live registry and break unrelated
  tests
- route matching still carries product-level selection logic that should move
  into typed family lowering and backend capability matching
- route prose and notes still occupy too much conceptual space in some traces
  and diagnostics

The new PDE and Monte Carlo family workstreams make this the right moment to
shrink routes again. Each time a reusable family IR becomes stronger, the route
layer should become thinner.

## Repo-Grounded Current State

### What is already true

Trellis already has:

- compiler-emitted lane obligations
- typed route admissibility
- route-binding authority packets
- `EventAwarePDEIR` as a reusable PDE family boundary
- `EventAwareMonteCarloIR` as the new reusable bounded Monte Carlo family
  boundary
- docs that explicitly describe routes as backend-binding and safety
  infrastructure

### What is still too route-heavy

The route registry still bundles together:

- candidate matching rules
- backend primitive/helper bindings
- admissibility metadata
- market-data access hints
- scoring hints
- prompt-oriented notes
- compatibility aliases
- discovered-route inventory

That is too much responsibility for a layer whose intended role is now only
backend-fit and compatibility.

## Design Summary

The intended progression is:

```text
old:
  method + product -> route registry -> route-specific build logic

target:
  SemanticContract
    -> ProductIR
    -> family IR
    -> family problem assembly
    -> exact backend binding catalog
    -> compatibility route alias (temporary)
```

So route minimization means:

1. move meaning upward into typed family IRs and family assembly
2. move backend facts sideways into a smaller binding catalog
3. leave only a thin compatibility shell where route IDs are still needed

## Design Guardrails

### 1. Do not remove the route layer in one step

The route registry still carries real compatibility and replay value. The
correct move is staged shrinkage, not a flag day.

### 2. Remove route responsibilities, not just route files

If route IDs disappear but the same hidden product-selection logic is copied
into prompt text, route scoring, or ad hoc helper lookups, nothing has been
improved.

### 3. Family IRs must absorb meaning before route logic is deleted

We should only remove a route-local semantic distinction after the relevant
family IR and family assembly layer can express it directly.

### 4. Discovered routes must stop destabilizing the live registry

Exploratory/discovered route entries should not change core registry-size or
coverage assertions for unrelated tickets. Live pricing should be more stable
than that.

## Residual Route Contract

The desired residual route mechanism owns only:

- route / alias identity for backward compatibility
- exact backend-binding identity
- admissibility envelope for a concrete backend binding
- validation-bundle and canary ownership
- provenance and replay identity

The residual route mechanism should no longer own:

- primary product semantics
- primary family selection
- family-level construction logic
- product-local numerical plans
- broad prompt-facing design guidance when no exact backend fit exists

## Ordered Delivery Queue

1. route-inventory stabilization and discovered-route quarantine
2. split route responsibilities into:
   - family capability selection
   - backend binding catalog
   - compatibility aliases
3. migrate scoring and matching away from product-specific route cards toward
   family-IR and backend-capability predicates
4. demote route-facing traces and diagnostics where family IR / lane
   obligations already carry the real meaning
5. remove or compress compatibility aliases once the migrated families no
   longer need them

## Concrete Tranches

### Tranche 1: Inventory and quarantine

- stop discovered routes from changing the live registry by default
- make route-registry tests assert the supported canonical contract, not the
  presence of opportunistic discovered entries
- separate "analysis/discovery inventory" from "live route authority"

### Tranche 2: Backend-binding catalog

- extract exact helper/kernel/primitive binding facts into a smaller backend
  catalog surface
- keep route IDs as aliases to catalog entries where compatibility requires it
- move prompt and trace surfaces to prefer family IR plus backend-binding
  authority over route-card identity

### Tranche 3: Family-first matching

- move candidate matching and scoring toward:
  - family IR type
  - typed capability predicates
  - backend availability
- reduce product-level route matching clauses that duplicate semantic lowering

### Tranche 4: Compatibility collapse

- once migrated families are stable, collapse redundant route aliases
- keep only durable public/replay identifiers that still add value

## Dependency Notes

- this plan is a sequel to `QUA-546`, not a duplicate of it
- it is directly related to:
  - `docs/plans/event-aware-pde-lane.md`
  - `docs/plans/event-aware-monte-carlo-lane.md`
- each new family migration should also remove one corresponding slice of
  route-local authority

## Agent Intake Bundle

Each coding agent assigned to this workstream should begin with:

- this plan doc
- `AGENTS.md`
- `trellis/agent/route_registry.py`
- `trellis/agent/route_scorer.py`
- `trellis/agent/lane_obligations.py`
- `trellis/agent/platform_traces.py`
- `trellis/agent/codegen_guardrails.py`
- `docs/quant/contract_algebra.rst`
- `docs/quant/pricing_stack.rst`
- `docs/developer/dsl_system_design_review.md`
- the completed `QUA-546` umbrella and closeout tickets

## Linear Mirror

Status mirror last synced: `2026-04-07`

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-727` | Route registry minimization: family-first backend binding cleanup | Backlog |

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-728` | Route inventory: quarantine discovered routes from live authority | Backlog |
| `QUA-729` | Route bindings: split backend-binding facts from route compatibility aliases | Backlog |
| `QUA-730` | Route matching: move canonical selection toward family-first capability predicates | Backlog |
| `QUA-731` | Diagnostics: demote route identity behind family IR and lane obligations | Backlog |
| `QUA-732` | Compatibility cleanup: retire redundant route aliases after migrated-family adoption | Backlog |
