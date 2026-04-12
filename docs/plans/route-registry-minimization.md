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
- add an explicit alias-retention policy so operator-facing surfaces can hide
  internal-only route ids while replay/canary metadata keeps them

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

## Post-Closeout Audit

The `QUA-727` queue is correctly closed, but a post-closeout audit on
`2026-04-10` found several smaller residual route-heavy surfaces worth
tracking as maintenance work rather than as a reopened umbrella:

- `QUA-417` is now a low-priority cleanup slice, not a core minimization
  tranche. Its role is only to collapse redundant optional utility bindings
  such as `schedule_builder` / `time_measure` in canonical route cards.
- `QUA-778` tracks the remaining route-card / route-spec-first build, trace,
  and validation surfaces. `trellis/agent/codegen_guardrails.py` still turns
  route-card helpers and notes into `InstructionRecord`s, and
  `trellis/agent/platform_traces.py` plus validator surfaces such as
  `trellis/agent/semantic_validators/algorithm_contract.py` still expose or
  consume `route_method` / `RouteSpec` directly more often than the target end
  state allows.
- `QUA-777` tracks the remaining route-scoring gap. `trellis/agent/route_scorer.py`
  still learns and scores on route identity (`route:<id>`,
  `route_family:<family>`) and still falls back to the route heuristic in
  `codegen_guardrails._route_score(...)`.

These gaps do not invalidate the completed `QUA-546` / `QUA-727` work. They do
mean route minimization still has a small maintenance tail after the completed
planned tranches.

## 2026-04-12 Route-Card Retirement Program

A fresh audit after the live `KL01` FX rerun showed that the remaining route
cards are not all equally risky.

Current repo-grounded split:

- 18 route cards still carry constructive authority once
  `conditional_primitives` are included
- 4 route cards already look metadata-first and should be audited before any
  code-changing cleanup is invented

The active route-card retirement queue is now tracked under umbrella
`QUA-780`, with task-backed child slices:

1. `QUA-778` FX and quanto exact-helper routes:
   `analytical_garman_kohlhagen`, `monte_carlo_fx_vanilla`,
   `quanto_adjustment_analytical`, `correlated_gbm_monte_carlo`
   Validation cohort: `KL01`, `T105`, `T108`, `E25`
2. `QUA-782` credit and copula helper routes:
   `credit_default_swap_analytical`, `credit_default_swap_monte_carlo`,
   `nth_to_default_monte_carlo`, `copula_loss_distribution`
   Validation cohort: `T38`, `T49`, `T50`, `T124`, `KL03`
   Carry-forward regressions discovered during closeout:
   `T53` remains a mixed analytical / FFT / generic Monte Carlo failure for
   `QUA-783` and `QUA-784`; `E26` is still misrouted into the generic basket
   path and carries forward to `QUA-784` and `QUA-777`
3. `QUA-781` rate-tree routes:
   `exercise_lattice`, `rate_tree_backward_induction`,
   `zcb_option_rate_tree`, `zcb_option_analytical`
   Validation cohort: `T01`, `T04`, `T05`, `T17`
   Closeout status: route cards thinned and helper-signature enforcement landed;
   `T01` and `T05` recovered, `T17` carries forward to `QUA-783`, and the
   residual `T04` comparison mismatch is tracked in `QUA-786` outside this
   route-surface workstream
4. `QUA-783` analytical / PDE / FFT routes:
   `analytical_black76`, `vanilla_equity_theta_pde`, `pde_theta_1d`,
   `transform_fft`
   Validation cohort: `T13`, `T17`, `T39`, `T53`, `T73`, `T94`, `E21`, `E22`, `T103`
   Closeout status: helper-backed route cards are thin, explicit empty
   adapter/note overrides now survive conditional resolution, exact-helper
   signature enforcement covers the vanilla-equity transform / Monte Carlo /
   PDE helpers, and analytical cap/floor strips now admit on typed schedule
   state instead of failing the generic automatic-event gate. `T13` and `T94`
   still pass; `E22` now reaches a successful analytical comparator and carries
   the remaining Monte Carlo comparator failure to `QUA-784`; `T17` carries
   forward to `QUA-787`; `T39` and the helper-backed comparator side of `E21`
   carry forward to `QUA-788`; `T73` carries forward to `QUA-789`; `T53` and
   `T103` no longer fail on route-surface authority but remain broader
   comparator/build-quality work outside this slice.
5. `QUA-784` generic Monte Carlo and basket routes:
   `monte_carlo_paths`, `correlated_basket_monte_carlo`
   Validation cohort: `T37`, `T53`, `T102`, `T104`, `T126`, `E21`, `E22`, `E24`, `E26`
   Closeout status: both route cards are now metadata-first, with explicit
   empty adapter / note overrides through conditional resolution. Ranked-
   observation baskets stay on `correlated_basket_monte_carlo`, generic
   `basket_option` wrappers now stay on `monte_carlo_paths`, and explicit
   credit-basket cues on generic basket wrappers upgrade through the semantic /
   decomposition layer to `nth_to_default_monte_carlo` instead of widening into
   the basket helper route. `E21` and `E22` still pass; `E26` now reaches the
   correct nth-to-default helper pair and carries only a downstream comparison
   failure outside this slice; `T102` now stays on the generic Monte Carlo path
   and carries only downstream build-quality gaps outside this slice; `T37`,
   `T53`, and `E24` no longer fail because of route-card authority but remain
   broader out-of-scope comparison / semantic gaps.
6. `QUA-785` metadata-first residual audit:
   `exercise_monte_carlo`, `local_vol_monte_carlo`, `qmc_sobol_paths`,
   `waterfall_cashflows`
   Representative cohort: `T14`, `T36`, `E23`, `E27`
7. `QUA-777` route scoring tail after the route-card surfaces themselves are
   retired

This queue is intentionally task-backed. The goal is not to delete route files
in the abstract. The goal is to remove route-card synthesis authority while
proving that the corresponding task cohorts still process on the modern
semantic/family/lane surface.

## Linear Mirror

Status mirror last synced: `2026-04-12`

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-727` | Route registry minimization: family-first backend binding cleanup | Done |
| `QUA-780` | Route surfaces: task-backed retirement of residual route-card synthesis authority | Backlog |

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-728` | Route inventory: quarantine discovered routes from live authority | Done |
| `QUA-729` | Route bindings: split backend-binding facts from route compatibility aliases | Done |
| `QUA-730` | Route matching: move canonical selection toward family-first capability predicates | Done |
| `QUA-731` | Diagnostics: demote route identity behind family IR and lane obligations | Done |
| `QUA-732` | Compatibility cleanup: retire redundant route aliases after migrated-family adoption | Done |
| `QUA-778` | Route surfaces: FX and quanto exact-helper routes stop emitting procedural authority | Done |
| `QUA-782` | Route surfaces: credit and copula routes retire procedural authority behind backend helpers | Done |
| `QUA-781` | Route surfaces: rate-tree routes retire procedural authority and stay task-backed | Done |
| `QUA-783` | Route surfaces: analytical Black76, PDE, and FFT routes retire procedural guidance | Done |
| `QUA-784` | Route surfaces: generic Monte Carlo and basket routes collapse to family-first metadata | Done |
| `QUA-785` | Route inventory: audit metadata-first residual route cards against representative tasks | Backlog |
| `QUA-777` | Route scoring: remove residual route-identity authority after route-card retirement | Backlog |
