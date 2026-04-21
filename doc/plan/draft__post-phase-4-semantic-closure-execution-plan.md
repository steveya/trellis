# Post-Phase-4 Semantic Closure Execution Plan

## Status

Active execution mirror for the filed post-Phase-4 semantic-closure
queue.

The umbrella and `CLX.*` child tickets are now filed in Linear. This
document remains the ordered repo-local mirror for that queue and should
stay aligned with the live issue graph.

`CLX.1` through `CLX.6` are now implemented and merged. Under `CLX.2`,
`QUA-941` and `QUA-942` landed the bounded static-leg semantic home and
route-free decomposition for scheduled period-rate-option strips,
`QUA-943` adds the first executable analytical/Monte-Carlo lowering
lane on the checked cap/floor helpers, and `QUA-944` closes the
selection-only basis-swap gap with a checked floating-vs-floating
lowering lane. Under `CLX.7`, `QUA-936` landed the overlay-boundary
fixtures and `QUA-939` lands the minimal policy-state overlay surface.
`QUA-937` plus `QUA-938` landed the first shared `CLX.8` readiness
infrastructure. `QUA-940` then normalized the scheduled-strip family
onto the clean canonical `period_rate_option_strip` semantic surface,
and `QUA-945` closed the `F003`-`F005` parity slice on the deterministic
exact-binding build path. The main queue pickup now returns to the
`QUA-934` umbrella until a bounded proving family actually requires
overlay execution.

Status mirror last synced: `2026-04-21`

## Linked Context

- QUA-887 — Semantic contract: contract-IR compiler (root umbrella)
- QUA-905 — Phase 3 structural solver compiler
- QUA-906 — Phase 4 route retirement / dispatch phaseout
- QUA-927 — Post-Phase-4 semantic closure umbrella
- QUA-928 — `CLX.1` quoted-observable snapshot closure
- QUA-929 — `CLX.2` static leg-based closure
- QUA-940 — `CLX.2` follow-on: normalize scheduled rate option strips onto the static-leg semantic track
- QUA-930 — `CLX.3` event/state/control semantic foundation
- QUA-931 — `CLX.4` automatic event/state lowering lane
- QUA-932 — `CLX.5` discrete control lowering lane
- QUA-933 — `CLX.6` continuous/singular control lowering lane
- QUA-934 — `CLX.7` insurance-style overlays
- QUA-935 — `CLX.8` later-family route retirement follow-ons
- QUA-936 — overlay boundary fixtures for financial-control vs policy-state overlays
- QUA-939 — minimal policy-state overlay surface
- QUA-937 — readiness ledger for dynamic closure cohorts
- QUA-938 — reusable masked-authority harness for later-family cutovers
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/draft__quoted-observable-contract-ir-foundation.md`
- `doc/plan/draft__leg-based-contract-ir-foundation.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `doc/plan/draft__contract-ir-phase-4-route-retirement.md`
- `doc/plan/draft__automatic-event-state-lowering-plan.md`
- `doc/plan/draft__discrete-control-lowering-plan.md`
- `doc/plan/draft__continuous-singular-control-lowering-plan.md`
- `doc/plan/draft__insurance-contract-overlay-foundation.md`

## Purpose

Turn the closure-program vocabulary into an ordered implementation queue
for the next major strand after the current Phase 4 work.

The core rule is:

- static semantic homes land before dynamic wrappers
- decomposition boundaries land before lowering expansion
- dynamic lowering lanes are split by mathematical problem class, not by
  product label
- route retirement for later families inherits the Phase 3 / Phase 4
  authority model instead of inventing a second dispatch regime

## Why This Needs Its Own Execution Plan

The foundation notes now describe the right long-run semantic basis:

- quoted-observable snapshot contracts
- static leg-based contracts
- event/state/control wrappers over static bases

But those notes are still architectural. Without an execution plan,
post-Phase-4 work risks stalling in familiar ways:

1. trying to solve static quote, static leg, and dynamic products in one
   combined tranche
2. implementing dynamic lowerings before the static semantic bases are
   stable
3. conflating automatic event/state products with holder/issuer control
   products
4. burying insurance-specific overlays inside the same tickets as the
   core financial-control semantics

## Execution Principles

1. **Static before dynamic.**
   Quoted-observable and static leg representations land before
   event/state/control families rely on them.
2. **Boundary first.**
   Each semantic family must have a fixture-backed classification matrix
   before its lowering queue starts.
3. **Problem-class lowering.**
   Dynamic lowerings split into:
   - automatic event/state
   - discrete control
   - continuous or singular control
4. **No product-name escape hatches.**
   If a queue slice cannot be described as semantic structure plus
   generic terms plus valuation context plus market capabilities, it is
   not ready.
5. **Phase 4 reuse.**
   Every later family should ultimately satisfy the same route-masked
   authority rule already proven on the current payoff-expression slice.

## Ordered Execution Queue

### Queue Table

| Queue ID | Linear | Status | Scope | Hard prerequisites |
| --- | --- | --- | --- | --- |
| `CLX.1` | `QUA-928` | Done | quoted-observable snapshot closure | current payoff-expression Phase 4 closeout |
| `CLX.2` | `QUA-929` | Done | static leg-based closure | current payoff-expression Phase 4 closeout |
| `CLX.3` | `QUA-930` | Done | event/state/control semantic foundation | `CLX.1`, `CLX.2` representation boundaries defined |
| `CLX.4` | `QUA-931` | Done | automatic event/state lowering lane | `CLX.3` |
| `CLX.5` | `QUA-932` | Done | discrete control lowering lane | `CLX.3` |
| `CLX.6` | `QUA-933` | Done | continuous/singular control lowering lane | `CLX.3` |
| `CLX.7` | `QUA-934` | Backlog | insurance-style overlays on top of financial control | `CLX.6` |
| `CLX.8` | `QUA-935` | Backlog | route-retirement follow-ons for migrated post-Phase-4 families | family-specific parity and provenance from the relevant earlier queue |

### Pickup Rule

- pick the earliest queue item whose hard prerequisites are satisfied
- do not skip from static closure directly to continuous-control work
- do not promote a later family into lowering until its representation
  and decomposition closure artifacts are explicit
- do not start `CLX.8` for any family until its earlier queue slice has
  a parity ledger and route-masked invariance story comparable to the
  current Phase 3 / Phase 4 program

Current next pickup:

- `QUA-934` — insurance-style overlays on top of financial control

Active out-of-order follow-on:

- none currently

## Queue Details

### CLX.1 — Quoted-observable snapshot closure

Goal:

Close representation and decomposition for terminal quote-point
contracts without dragging in scheduled coupons or dynamic wrappers.

Representative proving families:

- terminal curve-spread payoffs
- terminal vol-skew payoffs
- options on those quoted spreads where a checked lowering already
  exists

Required artifacts:

- admitted quote-node ADT
- boundary fixtures against basis swaps and other leg products
- decomposition fixtures showing route-independent quote-node emission
- first lowering declarations for the bounded admitted cohort

### CLX.2 — Static leg-based closure

Goal:

Close representation and decomposition for static scheduled cashflow
products without callability, interruption, or target state.

Representative proving families:

- vanilla IRS
- float-float basis swaps
- static coupon bonds
- bounded CMS coupon structures if they remain static

Required artifacts:

- static leg ADT
- static coupon-formula surface
- boundary fixtures against quoted snapshot contracts
- first lowering declarations on existing checked cashflow engines

Landed follow-ons:

- `QUA-941` — add the first-class static-leg representation for
  scheduled `period_rate_option_strip` contracts so the canonical family
  has a real leg-semantic home
- `QUA-942` — emit that strip representation route-independently from
  supported cap/floor request surfaces
- `QUA-943` — lower admitted scheduled strip legs onto checked
  analytical and Monte Carlo cap/floor helpers
- `QUA-944` — close the residual basis-swap executable-lowering gap in
  the bounded static-leg cohort
- `QUA-945` — repair deferred `F003`-`F005` parity through the canonical
  static-leg scheduled-strip path
- `QUA-940` — normalize scheduled cap/floor strips onto the canonical
  `period_rate_option_strip` semantic family and remove the transitional
  semantic alias before repairing the deferred `F003`-`F005`
  analytical binding gap

### CLX.3 — Event/state/control semantic foundation

Goal:

Land the shared dynamic semantic surface over static semantic bases.

Required artifacts:

- dynamic root ADT or equivalent sibling semantic surface
- explicit event ordering semantics
- explicit state schema and update semantics
- explicit controller-role and action-set semantics
- static-vs-dynamic classifier matrix

This queue item is representational. It should not yet try to solve all
numerical lanes.

### CLX.4 — Automatic event/state lowering lane

Goal:

Lower dynamic products that carry state and stopping rules but no true
holder/issuer optimization.

Representative proving families:

- autocallables / phoenix / snowball
- TARN / TARF
- bounded automatic coupon-interruption structures

Primary plan:

- see `draft__automatic-event-state-lowering-plan.md`

### CLX.5 — Discrete control lowering lane

Goal:

Lower products with explicit holder/issuer decisions on a discrete
schedule.

Representative proving families:

- swing options
- callable coupon structures
- issuer-call overlays on structured notes

Primary plan:

- see `draft__discrete-control-lowering-plan.md`

### CLX.6 — Continuous or singular control lowering lane

Goal:

Lower the stronger control class where action magnitude itself is part
of the contractual optimization problem.

Representative proving families:

- GMWB / selected GMxB families

Primary plan:

- see `draft__continuous-singular-control-lowering-plan.md`

### CLX.7 — Insurance-style overlays

Goal:

Keep mortality, lapse, fee, or policyholder-behavior overlays separate
enough from the core financial-control semantics that the latter remains
reusable outside insurance.

Primary plan:

- see `draft__insurance-contract-overlay-foundation.md`

Filed follow-ons:

- `QUA-936` — fixture-backed boundary between bounded financial control
  and deferred policy-state overlays

### CLX.8 — Later-family route retirement

Goal:

Apply the current Phase 3 / Phase 4 authority model to later migrated
families once their closure and parity evidence exists.

Required artifacts per migrated family:

- explicit closure record
- family-specific parity ledger
- provenance readiness
- route-masked selector tests

Filed follow-ons:

- `QUA-937` — readiness ledger for dynamic closure cohorts
- `QUA-938` — reusable masked-authority harness for later-family
  cutovers

Implementation note for the current tranche:

- the shared readiness ledger now lives in
  `trellis.agent.route_retirement_readiness.dynamic_route_retirement_readiness_ledger`
- the shared masking harness now lives in
  `trellis.agent.route_retirement_readiness.require_masked_authority_invariant`
- the seeded dynamic cohorts still remain blocked on parity /
  provenance even after the masking harness lands

## Family-to-Queue Map

| Family cluster | Static semantic owner | Dynamic owner | Lowering queue |
| --- | --- | --- | --- |
| terminal curve spread / vol skew | quoted-observable | none | `CLX.1` |
| vanilla IRS / SOFR-FF basis | static leg | none | `CLX.2` |
| CMS spread note without callability/interruption | static leg + quoted leaf reuse if needed | none | `CLX.2` |
| autocallable / phoenix / snowball | payoff-like or leg-like static fragments | event/state | `CLX.4` |
| TARN / TARF | static coupon fragments | event/state | `CLX.4` |
| callable CMS-spread range accrual | static leg + quoted leaves | event/state/control | `CLX.5` unless purely automatic |
| PRDC-style coupon hybrids | static leg + quoted leaves | event/state or event/state/control | `CLX.4` or `CLX.5` depending on control |
| swing | base delivery/payoff fragment | discrete control | `CLX.5` |
| GMWB / GMxB | bounded financial-control core | continuous/singular control + possible insurance overlays | `CLX.6`, then `CLX.7` as needed |

## Validation Contract

Each queue slice should leave behind the following before the next slice
depends on it:

1. one stable semantic note
2. one ordered fixture set for classification and canonicalization
3. one bounded family-admission statement
4. one lowering admission contract for the families in scope
5. one parity or benchmark plan
6. one explicit honest-block policy for out-of-scope relatives

## Risks To Avoid

- **Static/dynamic conflation.** If callable or interruptible products
  are forced into the same tickets as static IRS or basis swaps, the
  static leg slice will sprawl.
- **Product-first prioritization.** Picking products before problem
  classes will recreate route-local design pressure.
- **Insurance bleed-through.** Mortality or lapse semantics should not
  distort the generic financial-control foundation prematurely.
- **Premature route-retirement ambition.** Later route-retirement work
  should remain family-scoped and parity-gated.

## Next Steps

1. Keep this document as the live execution mirror for `QUA-927` and
   its child queue.
2. Work `QUA-934` next as the open `CLX.7` umbrella slice.
3. Treat `QUA-935` as the open `CLX.8` umbrella, with `QUA-937` and
   `QUA-938` already landed as shared readiness infrastructure for later
   family cutovers.
