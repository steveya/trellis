# Lower-Layer Cleanup And Canary Verification Plan

## Purpose

This plan closes the remaining lower-layer cleanup after the recent semantic,
route, and short-rate workstreams:

1. retire or explicitly bless the remaining stale checked-in FX/quanto adapters
2. keep extracting reusable family helper kits underneath product wrappers
3. keep moving lower-layer authority out of executor/route/product-local glue
   and into family/helper registries
4. verify every cleanup slice with the corresponding canary or `TASKS.yaml`
   acceptance task so the result remains good agent code instead of a paper
   refactor

## Why This Plan Exists

Recent work materially improved the stack:

- curated canaries are currently green
- semantic family/method authority is registry-backed
- route identity is thinner than before
- reusable short-rate fixed-income helpers exist beneath callable-bond wrappers

But several transitional seams remain:

- four checked-in adapters are still marked stale against `_fresh`
- executor guidance still contains product-shaped exact-binding rules
- several model/helper entry points remain product-shaped wrappers over logic
  that should sit lower in reusable family kits
- the remaining debt is spread across completed umbrellas, open helper-layer
  tickets, and stale-adapter lifecycle machinery, so the next work needs one
  durable queue

## Current Repo-Grounded State

### Remaining stale adapter inventory

These checked-in adapters still differ from their validated `_fresh`
counterparts:

- `trellis/instruments/_agent/fxvanillaanalytical.py`
- `trellis/instruments/_agent/fxvanillamontecarlo.py`
- `trellis/instruments/_agent/quantooptionanalytical.py`
- `trellis/instruments/_agent/quantooptionmontecarlo.py`

The current diffs suggest two distinct possibilities:

- the checked-in shell is the better thin helper-backed adapter and should be
  explicitly blessed
- the `_fresh` output is closer to the desired family helper surface and the
  checked-in shell should be replaced

This workstream resolves that ambiguity explicitly instead of leaving the
inventory stale indefinitely.

### Existing open tickets to reuse

The following open tickets already cover part of the desired direction and
should be reused rather than duplicated:

- `QUA-742` helper layers: event-aware Monte Carlo family composition
- `QUA-743` helper layers: reusable transform assembly surface
- `QUA-745` helper layers: docs, observability, and canary hardening
- `QUA-746` semantic comparison regimes: short-rate market objects and claim
  helper generalization
- `QUA-748` helper layers: shared short-rate regime resolver and discount-bond
  claim kit
- `QUA-749` short-rate wrappers: migrate ZCB analytical and tree helpers onto
  shared claim kits
- `QUA-458` full-task canary replay with diagnosis parity
- `QUA-710` trustworthy canary telemetry and historical baselines

### Acceptance tasks to keep or recover

#### Canary acceptance

- `T01` short-rate comparison regime and helper authority
- `T17` callable-bond PDE/tree event-aware short-rate path
- `T25` and `T26` vanilla Monte Carlo helper surface
- `T39` and `T40` transform-family helper surface
- `T49` basket-credit helper surface
- `T73` swaption comparison and event-aware MC path
- `T105` quanto analytical-vs-MC parity

#### Direct task acceptance

- task `1314` quanto option: quanto-adjusted BS vs MC cross-currency
- task `1370` FX vanilla option: Garman-Kohlhagen vs MC

The rule for this workstream is simple:

- no cleanup slice is considered done until its corresponding canary or task
  rerun still passes on the cleaned surface

## Scope

- stale-adapter closure for the remaining FX/quanto checked-in shells
- reusable helper extraction for FX/quanto family surfaces
- reuse and completion of the existing open MC/transform and short-rate helper
  tickets
- continued removal of product-local exact-binding authority from executor and
  route-era guidance
- canary/task reruns that prove the cleaned lower layer still supports good
  agent-generated code

## Non-Goals

- deleting every product wrapper immediately
- deleting every checked-in adapter immediately
- broad new product-family expansion unrelated to the current debt
- redesigning the canary runner again
- replacing the public pricing API or all compatibility layers in one step

## Design Principles

### 1. Family helper kits should replace product-local glue gradually

Product wrappers may remain temporarily, but they should become thin
compositions over reusable helper layers rather than acting as the true lower
authority.

### 2. Stale-adapter closure must be explicit

If a checked-in adapter remains the preferred thin shell, bless it and tighten
the freshness policy around it. If a fresh-build adapter is better, replace the
checked-in shell. Do not leave a stale warning unresolved just because both
variants happen to work.

### 3. Executor authority should keep shrinking

Executor/binding code should prefer registry-backed helper metadata and family
capability surfaces. Product-local exact-binding prose in the executor should
be treated as transitional debt.

### 4. Canary/task acceptance is mandatory

Every slice must name and rerun the canaries or direct tasks it defends.

## Ordered Delivery Queue

### New umbrella

| Issue | Title | Status |
| --- | --- | --- |
| `QUA-766` | Lower-layer cleanup: stale adapters, family helper convergence, and canary-backed verification | Backlog |

### Reused prerequisite tickets

1. `QUA-742`
2. `QUA-743`
3. `QUA-748`
4. `QUA-749`
5. `QUA-745`

These already exist and should be completed as part of this broader cleanup
program.

### New cleanup tickets

1. `QUA-767` FX/quanto family helper extraction beneath current wrappers
2. `QUA-768` stale-adapter closure for FX/quanto checked-in shells
3. `QUA-769` executor/binding cleanup for remaining product-shaped
   exact-binding rules
4. `QUA-770` final canary/task verification and closeout for the cleanup
   program

### Ordered queue mirror

Status mirror last synced: `2026-04-09`

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-742` | Helper layers: event-aware Monte Carlo family composition | Done |
| `QUA-743` | Helper layers: reusable transform assembly surface | Done |
| `QUA-748` | Helper layers: shared short-rate regime resolver and discount-bond claim kit | Done |
| `QUA-749` | Short-rate wrappers: migrate ZCB analytical and tree helpers onto shared claim kits | Done |
| `QUA-767` | Helper layers: FX and quanto family helper kits beneath current wrappers | Done |
| `QUA-768` | Adapter freshness: close remaining FX and quanto stale checked-in shells | Done |
| `QUA-769` | Backend binding: retire remaining product-shaped exact-binding guidance from executor | Done |
| `QUA-745` | Helper layers: docs, observability, and canary hardening | Done |
| `QUA-770` | Lower-layer cleanup: canary and direct-task verification closeout | Done |

Notes:

- `QUA-768` is blocked by `QUA-767`.
- `QUA-770` is blocked by `QUA-768` and `QUA-769`.
- `QUA-458` and `QUA-710` are not delivery blockers for the core cleanup, but
  they should be reused if replay/telemetry work is needed during closeout.

## Acceptance Matrix

| Slice | Primary acceptance | Secondary acceptance |
| --- | --- | --- |
| Stale FX/quanto adapter closure | `T105` | task `1370`, task `1314` |
| FX/quanto helper extraction | `T105` | task `1370`, task `1314` |
| MC/transform helper reuse (`QUA-742`, `QUA-743`) | `T25`, `T26`, `T39`, `T40` | full curated canary rerun |
| Short-rate helper reuse (`QUA-748`, `QUA-749`) | `T01` | `T17` |
| Executor/binding authority reduction | `T01`, `T17`, `T73`, `T49` | full curated canary rerun |
| Final closeout | full curated canary rerun | task `1314`, task `1370` |

## Validation Posture

### Local

- targeted unit tests for each helper or binding slice
- direct task reruns for `1314` and `1370` on FX/quanto cleanup slices

### Regional

- canary reruns by family:
  - `T105` for quanto
  - `T25`, `T26` for MC helper extraction
  - `T39`, `T40` for transform helper extraction
  - `T01`, `T17` for short-rate helper extraction
  - `T49` for basket-credit/executor authority
  - `T73` for rates comparison authority

### Global

- full curated canary rerun after the queue stabilizes
- broader non-integration suite where touched subsystems justify it

## Done State

This plan is complete when:

- the remaining FX/quanto stale adapters are either replaced or explicitly
  blessed as the intended thin shells
- the open MC/transform and short-rate helper tickets are complete
- executor/binding exact-helper authority is materially thinner and more
  registry-driven than it is today
- the direct FX/quanto tasks and the relevant curated canaries still pass
- the final curated canary rerun remains green and the cleanup is reflected in
  docs and mirrored plan state

## Closeout

Status mirror last validated: `2026-04-09`

Implemented outcome:

- shared single-state Monte Carlo and transform helper kits now sit under the
  vanilla wrappers
- FX vanilla and quanto routes now bind through semantic-facing helper kits and
  the checked-in adapters/_fresh snapshots were refreshed as the intended thin
  shells
- executor exact-binding guidance is narrower and fresh-build runs no longer
  short-circuit through deterministic exact-binding materialization
- shared short-rate claim helpers remain the authority under ZCB/callable
  wrappers

Acceptance evidence:

- direct task reruns passed:
  - ``T105`` in ``task_results_t105_qua766.json``
  - ``T108`` in ``task_results_t108_qua766.json``
- final curated canary rerun passed ``14/14`` in
  ``canary_results_20260409_qua766_closeout_v2.json``

Residual debt kept explicit:

- product wrappers still exist where they are acting as compatibility surfaces
  over broader helper kits
- retry counts on several Monte Carlo / transform / PDE canaries are still
  non-zero, so this umbrella improved correctness and lower-layer authority
  more than first-pass efficiency
- the transform lane still wants a true lowered transform-family IR in the
  future; see
  ``docs/plans/transform-family-ir-and-admissibility-hardening.md`` for the
  follow-on tranche that replaces the bounded route-metadata/model-family fix
  with a real lowered transform family contract
