# Discrete Control Lowering Plan

## Status

Draft. Pre-queue lowering plan for dynamic contracts with explicit
holder or issuer choice on a discrete decision schedule.

## Linked Context

- `doc/plan/draft__post-phase-4-semantic-closure-execution-plan.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/done__event-aware-pde-lane.md`
- `doc/plan/done__event-aware-monte-carlo-lane.md`
- existing repo surfaces:
  - `trellis.agent.dsl_algebra`
  - `trellis.models.trees.control`

## Purpose

Define the lowering program for contracts whose semantics include an
explicit discrete action set controlled by the holder or issuer.

Typical families:

- swing options
- callable coupon structures
- issuer-call overlays on structured notes
- other Bermudan-style structured decisions that are not merely
  automatic stopping rules

## Mathematical Scope

The defining contract class is:

- explicit decision dates
- explicit controller role
- explicit admissible action set
- post-decision value determined by a discrete optimization

Informally:

```text
V_i(x, y) = opt_{a in A_i(x, y)} [ immediate(i, x, y, a) + continuation(i, x, y, a) ]
```

where `opt` is `max` for holder control and `min` for issuer control.

This lane is distinct from automatic event/state lowering because the
contract now carries a genuine control problem.

## Non-Goals

- purely automatic stopping products
- continuous or singular control over action magnitude
- mortality/lapse overlays

## Admissible Numerical Families

The first bounded numerical families for this lane should be:

- trees or lattices with explicit backward-induction control
- control-aware PDE lanes where a checked discrete-control formulation
  exists
- LSMC-style Monte Carlo for higher-dimensional families where the
  action set remains discrete

The lane should never blur these into one fake universal "optionality"
surface.

## Ordered Queue

| Queue ID | Status | Scope | Hard prerequisites |
| --- | --- | --- | --- |
| `DCL.1` | Backlog | discrete control lowering contract and controller semantics | dynamic semantic foundation |
| `DCL.2` | Backlog | canonical fixture corpus for issuer-control and holder-control structures | `DCL.1` |
| `DCL.3` | Backlog | projection of dynamic control semantics onto tree/PDE/LSMC decision timelines | `DCL.1`, `DCL.2` |
| `DCL.4` | Backlog | first proving lane on checked discrete-control numerics | `DCL.3` |
| `DCL.5` | Backlog | secondary proving lane for a different numerical family where useful | `DCL.4` optional |
| `DCL.6` | Backlog | parity, provenance, and migration-readiness ledger | admitted proving lane |

## Queue Details

### DCL.1 — Lowering contract

Artifacts:

- typed controller-role semantics
- admissible-action semantics
- explicit post-decision valuation timing contract
- admission rules separating discrete control from automatic stopping

Acceptance:

- the lane can explain whether a family is `holder_max`, `issuer_min`,
  or not discrete-control at all

### DCL.2 — Fixture corpus

Representative fixtures:

- bounded swing with remaining-right inventory
- callable structured coupon note with issuer decision dates
- note with discrete accrual interruption plus issuer call

Acceptance:

- fixtures make action sets and controller roles explicit
- normalization does not depend on product-name shortcuts

### DCL.3 — Decision-timeline projection

Artifacts:

- explicit projection onto tree/PDE/LSMC decision phases
- state-before-decision vs state-after-decision semantics
- explicit action feasibility checks

Acceptance:

- the numerical lane can recover the control problem directly from the
  semantic object

### DCL.4 — First proving lane

Preferred first proving direction:

- whichever checked lane already has the strongest explicit control
  contract in repo-local infrastructure for the family in scope

Candidate path:

- lattice/tree control for issuer-call or Bermudan-style decisions

Acceptance:

- one bounded discrete-control family has a checked semantic-to-numeric
  lowering path

### DCL.5 — Secondary lane

Candidate direction:

- LSMC for higher-dimensional or less tree-friendly discrete-control
  families

Acceptance:

- second-lane support is clearly bounded and does not widen the family
  claim beyond the admitted semantic domain

### DCL.6 — Migration readiness

Artifacts:

- parity or benchmark ledger
- route-masked invariance plan
- provenance contract for controller role, action set, and selected lane

## Validation

- controller-role normalization tests
- action-set and inventory-state regression tests
- numerical parity/benchmark tests for admitted proving families
- honest-block tests for nearby continuous-control families

## Risks To Avoid

- **Automatic/control confusion.** Autocallables and issuer-call notes
  can look similar at a desk-label level; the lane must classify by
  actual controller semantics.
- **Decision-timing ambiguity.** If the semantics do not say whether a
  reported value is pre-decision or post-decision, downstream parity and
  exposure work will drift.
- **Action-set leakage.** Numerical heuristics such as regression basis
  or penalty approximation are not substitutes for an explicit semantic
  action set.

## Next Steps

1. Keep this as the discrete-control companion to the dynamic semantic
   foundation.
2. Promote `DCL.1` only after the dynamic semantic root and the static
   semantic bases are stable.
3. Use this lane instead of the automatic event/state lane whenever the
   contract truly contains holder or issuer choice.
