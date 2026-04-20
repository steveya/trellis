# Automatic Event/State Lowering Plan

## Status

Draft. Pre-queue lowering plan for dynamic contracts with explicit
events and running state but no true holder/issuer optimization.

## Linked Context

- `doc/plan/draft__post-phase-4-semantic-closure-execution-plan.md`
- `doc/plan/draft__event-state-control-contract-foundation.md`
- `doc/plan/draft__semantic-contract-closure-program.md`
- `doc/plan/done__event-aware-pde-lane.md`
- `doc/plan/done__event-aware-monte-carlo-lane.md`
- `doc/plan/active__semantic-simulation-substrate.md`
- existing repo surfaces:
  - `trellis.agent.event_machine`
  - `trellis.agent.semantic_contract_compiler`

## Purpose

Define the first dynamic lowering lane after the static semantic bases
are in place: contracts whose semantics include event ordering and
running state, but whose evolution does not require holder or issuer
optimization.

Typical families:

- autocallables / phoenix / snowball with automatic early redemption
- TARN / TARF with target-triggered stopping
- bounded coupon-interruption structures with deterministic stopping
  rules

## Mathematical Scope

The defining contract class is:

- explicit event schedule
- explicit state vector `Y_i`
- deterministic or observable-driven transition map
- automatic stopping rule
- no controller optimization over an action set

Informally:

```text
Y_{i+1} = T_i(Y_i, O_i)
termination_i = G_i(Y_i, O_i)
```

where `O_i` are observables at event time `t_i`.

This class is materially easier than discrete control because there is
no continuation-versus-exercise optimization problem. The lane should
preserve that distinction.

## Non-Goals

- holder or issuer choice
- Bermudan or swing-style dynamic programming
- continuous or singular control
- insurance-specific mortality or lapse overlays

## Lowering Targets

Admissible numerical families for this lane include:

- event-aware Monte Carlo
- event-aware PDE where the event/state dimension remains bounded and
  numerically tractable
- event-aware tree or lattice where a checked lane already exists

The lane should not force all families through one numerical method.
Admission should depend on event/state semantics plus the method’s
stated capability envelope.

## Ordered Queue

| Queue ID | Status | Scope | Hard prerequisites |
| --- | --- | --- | --- |
| `AES.1` | Backlog | automatic event/state lowering contract and admissibility rules | dynamic semantic foundation |
| `AES.2` | Backlog | canonical fixture corpus for autocallable and TARN/TARF state programs | `AES.1` |
| `AES.3` | Backlog | automatic event/state compiler projection onto numerical timelines | `AES.1`, `AES.2` |
| `AES.4` | Backlog | Monte Carlo proving lane for automatic event/state families | `AES.3` |
| `AES.5` | Backlog | optional PDE proving lane where state dimension stays bounded | `AES.3` |
| `AES.6` | Backlog | parity ledger, provenance, and later-family migration readiness | `AES.4` and any admitted alternative lane |

## Queue Details

### AES.1 — Lowering contract

Artifacts:

- typed automatic event/state lowering input surface
- admission rules distinguishing automatic stopping from control
- required provenance fields for event order, state schema, and
  stopping rule identity

Acceptance:

- the lane can say exactly which dynamic products are in scope without
  naming products as the primary abstraction

### AES.2 — Fixture corpus

Representative fixtures:

- single-underlier autocallable with fixed coupon
- phoenix variant with coupon memory state
- TARN with running accumulated coupon state
- TARF-style running gain state and target cap

Acceptance:

- fixtures encode explicit event ordering and state update semantics
- semantically equivalent requests normalize to the same dynamic
  structure

### AES.3 — Compiler projection

Artifacts:

- lowering from dynamic semantic surface onto numerical event timelines
- explicit state-update ordering
- explicit stopping-event semantics

Acceptance:

- no family-specific local branching remains necessary to recover the
  event order

### AES.4 — Monte Carlo proving lane

Why first:

- the repo already has event-aware Monte Carlo substrate work
- many autocallable/TARF structures are naturally checked against a
  pathwise event-aware simulation surface

Acceptance:

- one bounded MC declaration family exists for admitted automatic
  event/state contracts
- parity/benchmark plan is explicit

### AES.5 — PDE follow-on where admissible

Why not universal:

- some automatic event/state structures may be PDE-friendly
- others will not remain tractable once state dimension or hybrid factor
  structure grows

Acceptance:

- PDE admission rules are explicit and do not overclaim universal
  coverage for all automatic state families

### AES.6 — Migration readiness

Artifacts:

- family-specific parity ledger
- route-masked invariance plan
- provenance packet requirements

Acceptance:

- one admitted automatic event/state family is ready to enter the same
  later Phase-4-style authority program as the current payoff-expression
  families

## Validation

- fixture-level normalization tests
- event-order and state-update regression tests
- numerical-lane parity or benchmark tests per admitted family
- explicit honest-block tests for nearby out-of-scope control families

## Risks To Avoid

- **Hidden control creep.** A product with issuer call or holder choice
  must not slip into this lane just because its first examples look
  similar to an autocallable.
- **State under-modeling.** Coupon memory, running targets, or
  interruption counters must not be erased into one opaque payoff
  callback.
- **Method overclaim.** Event-aware Monte Carlo support does not imply
  PDE coverage, and vice versa.

## Next Steps

1. Keep this as the lowering-plan companion to the dynamic semantic
   foundation.
2. Use it only after the static semantic bases and dynamic foundation
   exist.
3. Promote `AES.1` first when the post-Phase-4 closure queue becomes
   active.
