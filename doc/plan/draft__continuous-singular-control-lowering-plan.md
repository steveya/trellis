# Continuous / Singular Control Lowering Plan

## Status

Draft. Pre-queue lowering plan for the control class where action
magnitude itself is part of the contractual optimization problem.

## Linked Context

- `doc/plan/draft__post-phase-4-semantic-closure-execution-plan.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/draft__insurance-contract-overlay-foundation.md`

## Purpose

Define the bounded execution plan for the strongest control class in the
post-Phase-4 closure program.

Representative families:

- GMWB
- selected GMxB products where the financial core is a withdrawal or
  surrender optimization problem

## Mathematical Scope

The defining contract class is stronger than Bermudan-style discrete
choice. The controller may choose an action magnitude, not just a label
from a small finite set.

Typical semantics include:

- account-value state
- guarantee-base or benefit-base state
- withdrawal amount or rate as a decision variable
- penalty regions
- surrender or termination features

In a bounded notation:

```text
V(t, x, y) = sup_{a in A(t, x, y)} [ immediate(t, x, y, a) + continuation(t, x', y') ]
```

and in some formulations the limiting object is a quasi-variational
inequality or singular-control problem rather than a finite-grid Bellman
recursion.

## Why This Needs Its Own Plan

This lane must stay separate from discrete control because:

- discretizing the action space is an approximation choice, not the core
  semantic truth
- numerical methods, admissibility, and validation policy are materially
  different
- insurance overlays may later sit on top of the financial-control core,
  but should not be used to define it

## Non-Goals

- pretending all continuous-control products can be reduced to small
  Bermudan action sets without recording that as an approximation
- blending mortality/lapse overlays into the first financial-control
  tickets
- claiming universal GMxB support from one GMWB slice

## Ordered Queue

| Queue ID | Status | Scope | Hard prerequisites |
| --- | --- | --- | --- |
| `CSC.1` | Backlog | continuous/singular control admission contract | dynamic semantic foundation |
| `CSC.2` | Backlog | financial-control fixture corpus for bounded GMWB-style semantics | `CSC.1` |
| `CSC.3` | Backlog | state and action normalization contract for account-value / guarantee-base products | `CSC.1`, `CSC.2` |
| `CSC.4` | Backlog | first admitted numerical lane for bounded financial-control slice | `CSC.3` |
| `CSC.5` | Backlog | parity / benchmark / approximation policy ledger | `CSC.4` |
| `CSC.6` | Backlog | hand-off to insurance overlays where applicable | `CSC.5` |

## Queue Details

### CSC.1 — Admission contract

Artifacts:

- explicit distinction between discrete control and continuous/singular
  control
- typed action-space contract
- explicit approximation-disclosure rule when a continuous control is
  discretized for numerics

Acceptance:

- the lane can say exactly which control class a product belongs to

### CSC.2 — Fixture corpus

Representative fixtures:

- bounded GMWB without mortality overlay
- withdrawal-penalty variant
- surrender feature if included in the first slice

Acceptance:

- fixtures encode account-value and benefit-base state explicitly
- controller semantics are not smuggled through product labels

### CSC.3 — State and action normalization

Artifacts:

- normalized financial state variables
- normalized action semantics
- explicit distinction between contractual action space and numerical
  discretization used by the proving lane

Acceptance:

- later reviewers can tell which pieces are semantics and which pieces
  are solver approximation policy

### CSC.4 — First numerical lane

The first admitted lane should be one bounded family with explicit
truth-in-approximation. Acceptable starting points may include:

- a bounded PDE/QVI/penalty-style lane
- a bounded dynamic-programming lane with an explicit discretized action
  grid, provided the approximation is documented as such

The first lane should optimize for semantic honesty over breadth.

### CSC.5 — Validation ledger

Artifacts:

- parity or benchmark comparison policy
- approximation-disclosure policy
- provenance requirements for selected action discretization or control
  solver family

Acceptance:

- no family enters later migration planning without an explicit record of
  what was actually validated

### CSC.6 — Overlay hand-off

Goal:

Once the financial-control core is stable, hand insurance-specific
biometric or behavioral overlays off to the dedicated overlay plan.

## Validation

- normalization tests for account-value and benefit-base state
- controller/action-space regression tests
- explicit approximation-policy tests
- benchmark or literature-grounded validation on the admitted bounded
  family

## Risks To Avoid

- **Discrete approximation amnesia.** If a continuous-control problem is
  solved on a coarse action grid, that approximation must remain visible
  in the plan and provenance.
- **Family overclaim.** One bounded GMWB slice does not imply generic
  GMxB support.
- **Overlay confusion.** Mortality, lapse, or fees should not be allowed
  to obscure the core financial-control semantics.

## Next Steps

1. Keep this as the strongest-control companion to the dynamic semantic
   foundation.
2. Promote `CSC.1` only after the discrete-control lane is well defined,
   so the class boundary remains sharp.
3. Use the insurance overlay companion doc for any post-financial-control
   extensions rather than widening the first bounded control slice.
