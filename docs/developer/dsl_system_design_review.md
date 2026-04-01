# Trellis Semantic DSL — System Design Review

**Date:** 2026-04-01  
**Status:** current shipped boundary  
**Supersedes:** earlier design-review notes that described the pre-boundary in-flight state

---

## 1. Architecture Summary (as-built)

```text
Request / term sheet
  -> SemanticContract
  -> semantic validation
  -> ValuationContext
  -> RequiredDataSpec / MarketBindingSpec
  -> ProductIR
  -> PricingPlan + RouteSpec admissibility
  -> family lowering IR
  -> existing helper-backed numerical route
```

The important point is that Trellis now has an explicit boundary between:

1. semantic contract meaning
2. valuation and market binding
3. route admissibility and lowering
4. numerical execution

This boundary is shipped code, not just prompt guidance.

---

## 2. Current Authority Model

### 2.1 Semantic authority

`SemanticContract` is still evolved in place. The authoritative typed fields are:

- `ConventionEnv`
- `SemanticTimeline`
- `ObservableSpec`
- `StateField`
- `ObligationSpec`
- `ControllerProtocol`
- `EventMachine`

Automatic triggers live in event/state machinery. Strategic rights live in
`ControllerProtocol`.

### 2.2 Valuation authority

`ValuationContext` is the authoritative valuation-policy object. It owns:

- market snapshot/source handle
- model spec
- measure spec
- discounting policy
- optional collateral policy
- reporting policy
- canonical `requested_outputs`

`requested_measures` is retained only as a compatibility shim.

### 2.3 Routing authority

`ProductIR` is the shared checked summary used for route selection. It is not
the full semantic contract and not a universal solver IR.

Typed route admissibility is owned by:

- `RouteSpec.admissibility`
- `BuildGateDecision`

### 2.4 Lowering authority

The current lowering boundary is family-specific, not universal:

- `AnalyticalBlack76IR`
- `VanillaEquityPDEIR`
- `ExerciseLatticeIR`
- `CorrelatedBasketMonteCarloIR`

These lower onto existing checked-in helpers and kernels. The pricing math
remains in `trellis/models/`.

---

## 3. Shipped Route Families

The typed semantic boundary is proven end-to-end for these route families:

| Route ID | Family IR | Current contract families |
|---|---|---|
| `analytical_black76` | `AnalyticalBlack76IR` | vanilla European options |
| `vanilla_equity_theta_pde` | `VanillaEquityPDEIR` | vanilla European options |
| `exercise_lattice` | `ExerciseLatticeIR` | callable bonds, Bermudan swaptions |
| `correlated_basket_monte_carlo` | `CorrelatedBasketMonteCarloIR` | ranked-observation baskets |

This boundary preserves current route IDs and helper entry surfaces.

---

## 4. Warning And Error Policy

The current system distinguishes:

- hard semantic validation errors
- typed admissibility failures
- successful compilation with warnings

Warnings are used when:

- legacy mirrors are normalized into typed fields
- a migrated route ignores a stale legacy mirror
- output requests fall back to bump-and-reprice support

Errors are used when:

- typed phase order is invalid
- observables or state fields future-peek
- an automatic trigger is represented as strategic control
- settlement-bearing contracts omit typed obligations
- route capabilities do not support the requested control, outputs, or state tags

---

## 5. Migrated-Family Authority Rules

For the migrated families above:

- typed obligations and timeline are authoritative for settlement semantics
- typed `EventMachine` is authoritative for automatic event semantics
- `settlement_rule` and `event_transitions` are mirrors, not truth sources

This is intentionally narrow. Non-migrated code paths may still consume those
legacy fields.

The practical consequence is that a stale legacy mirror can now warn without
blocking the migrated family path, as long as the typed surface is valid.

---

## 6. What Changed Relative To The Earlier In-Flight Review

The earlier review was directionally useful but stale on several points. The
following are now shipped, not hypothetical:

- typed semantic fields on `SemanticContract`
- structured semantic validation findings
- explicit `ValuationContext`
- compiled `RequiredDataSpec` and `MarketBindingSpec`
- typed route admissibility
- `ProductIR` as the shared checked summary
- family-specific lowering IRs for proven route families
- migrated-family authority flip away from legacy settlement/event mirrors

The following remain intentionally deferred:

- ordered sequential multi-controller protocols
- nonlinear funding/XVA semantics inside `ValuationContext`
- a universal numerical IR
- a full desk-task DSL

---

## 7. Engineering Rules For The Current Slice

These rules are now part of the shipped design:

- do not change pricing math in semantic-boundary PRs unless a test proves an existing bug
- do not re-create schedules, timing order, or settlement semantics inside route-local glue
- do not model automatic triggers as control nodes
- do not treat free-form legacy strings as authoritative on migrated paths
- do not introduce a parallel event DSL; reuse `EventMachine`

---

## 8. Remaining Gaps Worth Tracking

These are real follow-ons, but they are outside the current shipped boundary:

1. broader route-family migration beyond the current proven four
2. richer typed observable algebra beyond the tranche-1 descriptors
3. typed desk-task objects for exposure, explain, and calibration workflows
4. proof-carrying analytical rewrite lemmas beyond current helper-backed kernels
5. portfolio-level nonlinear valuation layers such as funding and XVA

Those should build on the current semantic/valuation/lowering split rather than
reopening it.
