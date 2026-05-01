# P001 Semantic Execution Proof

## Status

Completed execution mirror for `QUA-989`.

Status mirror last synced: `2026-05-01`.

## Linked Linear

- `QUA-975` - Semantic execution: contract execution IR and visitor framework
- `QUA-989` - Semantic execution proof: P001 route-free Bermudan rainbow

## Linear Ticket Mirror

| Order | Ticket | Status | Objective | Hard blocker |
| --- | --- | --- | --- | --- |
| 0 | `QUA-989` | Done | umbrella proof for route-free `P001` semantic execution | `QUA-975` |
| 1 | `QUA-990` | Done | deterministic underlier binding and fail-closed guardrails | none |
| 2 | `QUA-991` | Done | operator IR for Bermudan best-of contract | `QUA-990` |
| 3 | `QUA-992` | Done | capability admission for MC and lattice | `QUA-991` |
| 4 | `QUA-993` | Done | generic multi-asset Bermudan MC visitor | `QUA-992` |
| 5 | `QUA-994` | Done | lattice state-grid admission or generic executor | `QUA-992` |
| 6 | `QUA-995` | Done | demote `_agent` adapter to execution shim and close proof | `QUA-993`, `QUA-994` |

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

## Closeout

`QUA-995` closes the bounded `P001` proof by demoting the generated
`_agent/rainbowoption.py` artifact to a compatibility shim. The shim delegates
to `price_bermudan_best_of_basket_from_compat_spec(...)`, which recompiles the
legacy adapter spec into execution IR and then dispatches to the checked
Monte Carlo or lattice execution visitor. The focused task replay now passes
both comparison targets: Monte Carlo `16.351660882611306`, lattice
`16.597284916802067`, both within the 5% internal comparison tolerance.

`QUA-989` is closed in Linear after all child slices completed and PR #701
merged. The broader execution-IR umbrella `QUA-975` remains open for reusable
framework work beyond the bounded `P001` proof.
