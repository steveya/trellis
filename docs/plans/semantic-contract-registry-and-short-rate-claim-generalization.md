# Semantic Contract Registry And Short-Rate Claim Generalization

## Purpose

This plan formalizes two linked follow-on directions:

1. `trellis/agent/semantic_contracts.py` should stop relying on one giant
   ordered branch chain for family drafting and admissible-method authority.
2. callable-bond PDE/tree wrappers should stop owning reusable short-rate
   event, cashflow, and claim assembly logic that belongs in a broader
   short-rate helper layer.

The goal is to make lower-layer family/method authority explicit and
declarative, then use that cleaner authority boundary to keep migrating
product-shaped short-rate wrappers onto reusable claim/helper kits.

## Summary Of The Two Directions

### A. Semantic contracts need a structural refactor

The current `semantic_contracts.py` still mixes three concerns in one large
module:

- request-family drafting
- family construction
- method specialization

Those concerns are encoded partly in a giant ordered `if/elif` chain and
partly in duplicated family-local branching across:

- `semantic_contracts.py`
- `platform_requests.py`
- `semantic_contract_compiler.py`

That duplication is exactly why Trellis kept reintroducing admissible-method
bugs for otherwise supported family/method pairs.

The target is:

- ordered draft-rule registry for family drafting
- registry-backed family definitions and method surfaces
- one shared specialization authority reused by the lower layers
- registration-time invariants that make admissible-method drift loud

### B. Callable-bond numerics should not remain the reusable authority

The callable-bond PDE/tree work was the right proving slice, but it should not
be the long-term location of reusable short-rate assembly logic.

The reusable layer should sit below the current public callable-bond wrappers
and own:

- fixed-income cashflow/event timeline compilation
- short-rate claim assembly
- short-rate event-aware PDE preparation
- short-rate lattice/tree claim preparation

The current callable-bond modules should remain only as thin compatibility and
public entry surfaces until downstream users are migrated.

## Why Now

The current stack is at a point where both problems are now easy to see:

- the semantic layer already carries enough meaning that lower layers should
  not keep rediscovering family and method authority through branch order
- the short-rate helper direction is already started in
  `docs/plans/short-rate-comparison-regime-and-claim-helpers.md`, but that
  workstream currently proves the shared layer through ZCB helpers rather than
  through callable-bond PDE/tree wrappers

This plan turns those observations into a reviewable implementation queue.

## Relationship To Existing Plans

This plan depends on the already-created short-rate foundation work in
`docs/plans/short-rate-comparison-regime-and-claim-helpers.md`:

- `QUA-746`
- `QUA-747`
- `QUA-748`
- `QUA-749`
- `QUA-751`

It is also a semantic-layer follow-on to:

- `docs/plans/semantic-platform-hardening.md`
- `docs/plans/route-registry-minimization.md`

This plan does not replace those plans. It narrows the next concrete work.

## Design Summary

The intended shape is:

```text
request text / structured task
  -> ordered draft-rule registry
  -> registry-backed family definition
  -> registry-backed method surface
  -> one specialization authority
  -> ProductIR / family IR / helper assembly

and, for short-rate event claims:

typed short-rate regime + claim semantics
  -> fixed-income event timeline helpers
  -> shared short-rate claim assembly
  -> thin callable-bond PDE/tree wrappers
```

Two principles drive the plan:

- once a family is known, lower layers should not rediscover family/method
  authority through ad hoc branching
- reusable short-rate assembly logic should live below product wrappers, not
  inside them

## Track A: Semantic Contract Registry Refactor

### Problem

`semantic_contracts.py` is still too branch-heavy and too duplicated.

It remains possible for:

- family recognition to depend on incidental branch order
- candidate methods to drift away from actual method surfaces
- downstream specialization to encode its own family-local method logic

### Target

Create a registry-backed semantic-contract layer with:

- explicit ordered draft rules
- explicit family definitions
- explicit method surfaces
- one specialization authority reused by the lower layers
- invariant tests that enumerate the supported family/method matrix

### Acceptance boundary

This track is successful when admissible-method bugs stop being a runtime
surprise and become either:

- impossible through registration structure, or
- loud at registration/test time

## Track B: Short-Rate Claim Generalization Beneath Callable-Bond Wrappers

### Problem

The callable-bond PDE/tree modules still bundle reusable short-rate logic that
should sit in a generic short-rate claim/helper layer.

### Target

Use the short-rate comparison-regime and ZCB-helper work as the foundation for
the next extraction step:

- reusable fixed-income event timeline helpers
- reusable short-rate claim assembly surfaces
- callable-bond PDE/tree wrappers reduced to thin public shells

### Acceptance boundary

This track is successful when the callable-bond wrappers are materially
smaller and the shared short-rate layer clearly owns the reusable event,
cashflow, and claim assembly logic.

## Ordered Delivery Queue

### Existing prerequisite foundation

These issues already exist and remain the prerequisite base for the
short-rate follow-on:

| Issue | Title | Status |
| --- | --- | --- |
| `QUA-746` | Semantic comparison regimes: short-rate market objects and claim helper generalization | Done |
| `QUA-747` | Task runtime: typed short-rate comparison regime objects | Done |
| `QUA-748` | Helper layers: shared short-rate regime resolver and discount-bond claim kit | Done |
| `QUA-749` | Short-rate wrappers: migrate ZCB analytical and tree helpers onto shared claim kits | Done |
| `QUA-751` | Validation and exact binding: recover T01 on typed short-rate regimes | Done |

### New umbrella

| Issue | Title | Status |
| --- | --- | --- |
| `QUA-758` | Semantic contracts: registry-driven family drafting and short-rate claim generalization | Done |

### Track A: semantic-contract registry

| Issue | Title | Status |
| --- | --- | --- |
| `QUA-759` | Semantic contracts: ordered draft-rule registry instead of giant family branching | Done |
| `QUA-760` | Semantic contracts: family-definition and method-surface registries | Done |
| `QUA-761` | Semantic compiler: one method-specialization authority across contracts, requests, and compiler | Done |
| `QUA-762` | Semantic contracts: registry invariants and admissible-method coverage | Done |

### Track B: callable-bond short-rate helper generalization

| Issue | Title | Status |
| --- | --- | --- |
| `QUA-763` | Short-rate helpers: fixed-income event timelines and claim assembly below callable-bond wrappers | Done |
| `QUA-764` | Callable rates wrappers: migrate callable-bond PDE and tree helpers onto shared short-rate claim kits | Done |
| `QUA-765` | Semantic contracts and short-rate helpers: docs, observability, and compatibility cleanup | Done |

## Critical Path

The intended order is:

```text
QUA-759 -> QUA-760 -> QUA-761 -> QUA-762
```

for the semantic-contract refactor, and:

```text
QUA-747 -> QUA-748 -> QUA-749 -> QUA-763 -> QUA-764 -> QUA-765
```

for the short-rate helper follow-on.

These two tracks are linked but not identical:

- Track A is the lower-layer semantic-authority cleanup.
- Track B is the next reusable short-rate helper extraction.

They can overlap in time, but the callable-bond short-rate extraction should
reuse the already-landed `QUA-746` foundation rather than bypass it.

## Validation Expectations

### Track A

- `tests/test_agent/test_semantic_contracts.py`
- `tests/test_agent/test_platform_requests.py`
- `tests/test_agent/test_semantic_contract_compiler.py`
- explicit registry-coverage tests for supported family/method pairs

### Track B

- targeted callable-bond PDE/tree tests
- targeted short-rate helper-layer tests
- `T17` regression coverage where the wrapper boundary changes
- any additional short-rate canary or workflow checks required by the migrated
  helper path

## Done State

This plan is complete when:

- semantic family drafting and method specialization are registry-backed
- lower-layer specialization authority is shared instead of duplicated
- admissible-method drift is guarded by registry invariants and tests
- callable-bond wrappers are compatibility/public shells over shared
  short-rate event/cashflow/claim helpers
- the official docs and plan mirrors explain the new authority boundaries
