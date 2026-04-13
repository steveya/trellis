# Binding-First Exotic Assembly Program

## Purpose

This document defines the next architecture program after route-registry
minimization.

The goal is not another route cleanup pass. The goal is to replace `route` as
an architectural primitive with explicit backend bindings, typed family/lane
assembly, and first-class operator metadata so Trellis can process arbitrary
but constructable exotic derivatives without inventing a new route card for
each structure.

The target end state is:

- product meaning comes from `SemanticContract`
- executable structure comes from `ProductIR`, family IR, and lane obligations
- exact implementation identity comes from a backend-binding catalog
- validation, traces, replay, and diagnostics are keyed by binding identity
- operator-visible names and wording come from first-class operator metadata
- route ids disappear from the runtime-critical path

## Decision Summary

The program direction is:

- treat route minimization as complete enough to start replacement work
- introduce a first-class backend-binding layer
- refactor lowering and execution to bind from typed roles instead of route ids
- migrate validation, replay, traces, and diagnostics onto binding identity
- separate operator-facing metadata from runtime implementation identity
- prove the result on representative constructable exotic cohorts

This is a capability program, not a cleanup program.

## Why Now

The current codebase is already past the most dangerous route-centric phase:

- semantic contracts and product IR are upstream of route selection
- family lowering and lane obligations already carry much of the real meaning
- route scoring, prompt selection, and offline route-learning no longer treat
  route id as dominant authority

But route still remains the implementation identity in several critical places:

- exact backend selection in `trellis/agent/codegen_guardrails.py`
- route-id-special lowering in `trellis/agent/family_lowering_ir.py`
- helper/kernel/schedule lookup in `trellis/agent/dsl_lowering.py`
- validation identity in `trellis/agent/validation_contract.py`
- trace and replay identity in `trellis/agent/platform_traces.py`,
  `trellis/agent/task_run_store.py`, and checkpoint/replay artifacts
- operator-facing wording and diagnostic labels that still piggyback on route
  metadata

That is good enough for bounded current support, but it is not a stable end
state for arbitrary constructable exotics. If route remains the main
implementation identity, Trellis will eventually hit route explosion or will
force novel products into the wrong legacy bucket.

## Repo-Grounded Current Gap

The remaining gap is not semantic understanding. The remaining gap is runtime
identity and assembly authority.

### What is already strong enough

- `SemanticContract` and semantic compilation
- `ProductIR`
- family IRs and lane obligations
- route-score and prompt-authority minimization
- task-backed route-card retirement for the old route surfaces

### What still needs to move

- `PrimitivePlan` / `GenerationPlan` now carry explicit backend-binding
  identity, but downstream lowering/validation surfaces still treat route ids
  as the dominant join key
- family lowering still branches directly on `route_id`
- DSL lowering still resolves helpers/kernels/schedules by route id
- validation and trace contracts still store route identity as primary runtime
  provenance
- semantic gap taxonomy still uses route-shaped helper labels instead of the
  binding-first blocker vocabulary
- operator-facing wording still leaks out of route-bound surfaces instead of a
  dedicated operator metadata layer

## Design Guardrails

### 1. Replace runtime identity first, then delete route

Route should disappear because the binding-first runtime is complete, not
because route strings were renamed.

### 2. Keep semantics and family assembly upstream

This program must not reintroduce product meaning into bindings. Bindings are
exact implementation contracts, not semantic categories.

### 3. Preserve reviewable slices

Every child issue should leave behind a checked-in artifact and a concrete
validation gate. Avoid broad rewrites with ambiguous outcomes.

### 4. Prove with tasks, not only unit tests

This program exists to support arbitrary constructable exotics. Each major
slice should be validated on representative task cohorts, not only module-local
tests.

### 5. Do not preserve route for nostalgia

Replay, validation, and operator visibility are valid needs. Route ids are not
the only way to satisfy them. If a new binding-first surface carries those
needs correctly, the route layer should shrink or disappear.

## Program Structure

The implementation program is organized into five epics:

1. Backend binding architecture
2. Lowering and assembly decoupling
3. Validation, replay, and trace identity migration
4. Operator surface separation
5. Exotic composition proof program

The first four epics replace route as a runtime primitive. The fifth epic
proves the replacement on the kind of constructable exotic cohorts Trellis
ultimately targets.

## Linear Ticket Mirror

Status mirror last synced: `2026-04-13`

### Ordered Epic Queue

| Ticket | Status |
| --- | --- |
| `QUA-792` | Exotic assembly: binding-first runtime and route retirement program | Backlog |
| `QUA-793` | Backend binding: first-class binding catalog and runtime identity | Backlog |
| `QUA-794` | Exotic assembly: bind lowering and construction from typed roles | Backlog |
| `QUA-795` | Validation and replay: migrate runtime identity to backend bindings | Backlog |
| `QUA-796` | Operator surfaces: first-class binding metadata and diagnostics | Backlog |
| `QUA-797` | Exotic composition: prove route-free assembly on constructable cohorts | Backlog |

### Ordered Backend Binding Queue

| Ticket | Status |
| --- | --- |
| `QUA-798` | Backend binding: introduce canonical binding catalog beside route registry | Done |
| `QUA-799` | Backend binding: carry binding identity through primitive and generation plans | Done |
| `QUA-800` | Backend binding: move exact helper and kernel lookup onto binding specs | Done |
| `QUA-810` | Route aliases: collapse route registry to transitional alias and admissibility shell | Done |

### Ordered Lowering And Assembly Queue

| Ticket | Status |
| --- | --- |
| `QUA-801` | Family lowering: replace route-id special cases with binding-role dispatch | Done |
| `QUA-805` | DSL lowering: resolve helpers, kernels, schedules, and controls from binding roles | Done |
| `QUA-811` | Semantic blockers: rename route-shaped helper gaps to binding and primitive taxonomy | Done |
| `QUA-816` | Constructive guidance: retire residual route adapters and notes from fallback lanes | Backlog |

### Ordered Validation / Replay Queue

| Ticket | Status |
| --- | --- |
| `QUA-802` | Validation contract: key exact-fit validation bundles by binding identity | Done |
| `QUA-806` | Platform traces and diagnostics: primary construction provenance is binding-first | Done |
| `QUA-812` | Replay and checkpoints: regenerate binding-first canary and learning artifacts | Done |
| `QUA-813` | Task stores and benchmark reports: retire route-primary health summaries | Done |

### Ordered Operator Surface Queue

| Ticket | Status |
| --- | --- |
| `QUA-803` | Operator metadata: introduce first-class binding display and diagnostic catalog | Done |
| `QUA-807` | Operator views: use binding metadata in MCP, session, and task diagnostics surfaces | Done |
| `QUA-814` | Route YAML cleanup: strip operator-facing wording after binding metadata adoption | Done |

### Ordered Exotic Composition Proof Queue

| Ticket | Description | Status |
| --- | --- | --- |
| `QUA-804` | Exotic benchmark: define constructable proof cohort for binding-first assembly | Done |
| `QUA-808` | Exotic assembly: run the event-control-schedule proof cohort and split residual gaps | Done |
| `QUA-817` | Proof follow-on: callable-bond PDE exact binding or constructive steps (`T17`) | Backlog |
| `QUA-818` | Proof follow-on: swaption analytical/tree/MC parity drift (`T73`) | Backlog |
| `QUA-819` | Proof follow-on: cap/floor fresh-build stability and reference-target evidence (`E22`) | Backlog |
| `QUA-820` | Proof follow-on: structured blocker persistence for honest-block sentinel (`E27`) | Backlog |
| `QUA-821` | Proof telemetry: remove residual `unknown` route ids from proof traces | Backlog |
| `QUA-809` | Exotic assembly: prove basket-credit and loss-distribution structures on binding roles | Backlog |
| `QUA-815` | Exotic program closeout: measure proof-cohort outcomes on the binding-first runtime | Backlog |

The benchmark contract for those proof tickets lives in
`docs/plans/binding-first-exotic-proof-cohort.md`.

## Cross-Epic Sequencing Constraints

- Do not start binding propagation through plans until the binding catalog
  exists.
- Do not start lowering or DSL binding-role migration until plan/runtime
  objects can carry binding identity.
- Do not start validation/replay migration until binding identity exists on the
  runtime plans used to construct those contracts.
- Do not strip route-facing operator wording until a dedicated operator
  metadata surface exists.
- Do not start the proof-of-capability cohort until the first four epics have
  landed enough runtime migration to make the result meaningful.
- Do not collapse the route registry to a thin alias shell until lowering,
  validation, replay, and operator surfaces all have working binding-first
  replacements.

## Success Criteria

The program is successful only when:

- new constructable products can bind onto checked helpers/kernels without
  introducing new route ids
- runtime plans, validation, traces, and replay all expose binding identity as
  the primary implementation contract
- operator-visible names and diagnostics do not depend on route YAML prose
- route ids, if any remain temporarily, are reduced to alias/transition-only
  status rather than core runtime meaning

## Agent Intake Bundle

Each coding agent assigned to this program should begin with:

- `AGENTS.md`
- `ARCHITECTURE.md`
- this plan doc
- `docs/plans/route-registry-minimization.md`
- `trellis/agent/route_registry.py`
- `trellis/agent/codegen_guardrails.py`
- `trellis/agent/family_lowering_ir.py`
- `trellis/agent/dsl_lowering.py`
- `trellis/agent/validation_contract.py`
- `trellis/agent/platform_traces.py`
- `trellis/agent/task_run_store.py`
- `docs/quant/pricing_stack.rst`
- `docs/developer/dsl_system_design_review.md`
