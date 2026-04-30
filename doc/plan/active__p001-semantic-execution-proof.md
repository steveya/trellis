# P001 Semantic Execution Proof

## Status

Active execution mirror for `QUA-989`.

Status mirror last synced: `2026-04-30`.

## Linked Linear

- `QUA-975` - Semantic execution: contract execution IR and visitor framework
- `QUA-989` - Semantic execution proof: P001 route-free Bermudan rainbow

## Linear Ticket Mirror

| Order | Ticket | Status | Objective | Hard blocker |
| --- | --- | --- | --- | --- |
| 0 | `QUA-989` | Backlog | umbrella proof for route-free `P001` semantic execution | `QUA-975` |
| 1 | `QUA-990` | Done | deterministic underlier binding and fail-closed guardrails | none |
| 2 | `QUA-991` | Done | operator IR for Bermudan best-of contract | `QUA-990` |
| 3 | `QUA-992` | Done | capability admission for MC and lattice | `QUA-991` |
| 4 | `QUA-993` | In Progress | generic multi-asset Bermudan MC visitor | `QUA-992` |
| 5 | `QUA-994` | Backlog | lattice state-grid admission or generic executor | `QUA-992` |
| 6 | `QUA-995` | Backlog | demote `_agent` adapter to execution shim and close proof | `QUA-993`, `QUA-994` |

## Objective

Use `P001` as the first route-free exotic proof that Trellis can move
from generated `_agent/{product}.py` implementation logic to semantic
contract execution.

The target workflow is:

1. normalize the task contract into named market and semantic inputs
2. represent the product as semantic operators rather than a product route
3. admit or reject engines by capability
4. lower through reusable execution visitors
5. leave `_agent` as a compatibility shell only

## Proof Contract

`P001` is a Bermudan best-of-two rainbow call on the `equity_rainbow_two_asset`
scenario. The scenario provides the authoritative underlier universe:
`AAPL`, `MSFT`.

The proof must not introduce a product-specific Bermudan rainbow route. It
must either price through generic execution visitors or fail closed with a
typed missing-primitive blocker.

## Acceptance Gates

- `P001` binds vector contract inputs to `AAPL` and `MSFT`.
- no generated code path guesses `Asset1`, `Asset2`, or `state_space`
- no generated code path calls short-rate lattice APIs for `P001`
- Monte Carlo execution consumes semantic/execution IR
- lattice execution either uses a compatible generic state-grid executor or
  reports a structured missing primitive
- `_agent/rainbow_option.py`, if present, contains no product-local pricing
  formula or market binding authority

## Current Slice

`QUA-993` adds the generic Monte Carlo visitor for the admitted `P001`
execution IR. It should consume named observables, correlation, carry,
exercise dates, and best-of payoff semantics from the route-free artifact,
then price through correlated multi-asset GBM plus reusable multi-state
Longstaff-Schwartz exercise logic without using `_agent/rainbow_option.py`
as product authority.
