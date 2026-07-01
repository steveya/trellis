# Contract-Backed Task Learning Repair

## Purpose

This plan covers the repair program for the failed assisted task rerun pack.
The retry mechanism now runs, but the learning content is not yet strong enough:
it mostly nudges generation with prose instead of carrying exact callable,
primitive, binding, market, and validation contracts.

The goal is to make assisted task learning useful without weakening production
semantics:

- production-like `strict` mode remains deterministic and fail-closed
- assisted mode may retry once per target only with contract-backed evidence
- failed runs produce machine-readable repair packets instead of vague lessons
- successful retries can teach future agents how to compose Trellis components
- cookbook growth is not the primary fix; durable library contracts are
  preferred

## Current Failure Evidence

The latest live rerun of `T20 T22 T105 T107 E27` showed:

- `T105` remained green with no generation attempts.
- `E27` was correctly handled as a passed honest block.
- `T107` QMC recovered after retry, proving the retry path can work.
- `T20` still failed from Heston binding/transform-contract drift.
- `T22` still failed from double-barrier PDE/MC assembly-contract drift.
- `T107` still had a pseudo-MC/QMC primitive-obligation mismatch.

The right fix is not to add broad cookbook text. The right fix is to make the
retry payload precise enough that a pricing-function agent can see the same
contracts a human implementer would use: accepted function signatures, required
imports, supported primitives, selected bindings, validation bundle identity,
and the semantic target.

## Design Direction

### Evidence Before Retry

Repair candidates should be built from deterministic evidence:

- import availability and exact symbol identity
- `inspect.signature(...)` for failed helpers or constructors
- structured primitive obligations for missing numerical support
- route or binding identity, payoff module, validation bundle, and market
  binding
- method prices, tolerances, and comparison target when cross-validation fails

### Gate Weak Learning

Assisted retry should not run just because a model produced plausible prose.
The runtime should classify a candidate as retryable only when it includes
actionable obligations. Otherwise it should persist the repair packet and fail
closed.

### Keep Library Abstractions Small

The implementation should avoid solving every task by adding derivative-specific
checked helpers. Add helpers only when they are real reusable computational
primitives. The preferred outcome is that Trellis exposes enough stable
contracts for the agent to write derivative-specific code correctly.

## Linear Ticket Mirror

Status mirror last synced: `2026-07-01`

### Parent

| Ticket | Title | Status |
| --- | --- | --- |
| `QUA-1131` | Agent learning: contract-backed intra-run repair | Todo |

### Ordered Implementation Queue

| Ticket | Title | Status | Depends on |
| --- | --- | --- | --- |
| `QUA-1132` | Agent learning: deterministic repair evidence packets | Todo | - |
| `QUA-1133` | Agent learning: overlay gates and retry contract enforcement | Todo | `QUA-1132` |
| `QUA-1134` | Heston ADI: exact runtime binding and transform contract | Todo | `QUA-1132`, `QUA-1133` |
| `QUA-1135` | Double barrier: PDE and MC assembly contracts | Todo | `QUA-1132`, `QUA-1133` |
| `QUA-1136` | Autocallable MC: event engine and QMC primitive obligations | Todo | `QUA-1132`, `QUA-1133` |
| `QUA-1137` | Task learning: promotion-grade evidence and docs closeout | Todo | `QUA-1134`, `QUA-1135`, `QUA-1136` |

## Validation Plan

Each implementation ticket should land with targeted tests first. The closeout
ticket then reruns the failed pack and remediation analysis:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/rerun_ids.py T20 T22 T105 T107 E27 --recovery-mode assisted
/Users/steveyang/miniforge3/bin/python3 scripts/remediate.py --analyze-only --results <result-file> --skip-platform-traces
```

Expected closeout:

- `T105` stays green
- `E27` remains a certified honest block with `passed_expectation=true`
- `T20`, `T22`, and `T107` either pass or emit concrete implementation-gap
  repair packets instead of signature/primitive/prose-learning drift
- `strict` mode never invokes LLM/codegen recovery

## Implementation Notes

Start with `QUA-1132`. Do not jump directly to the pricing tasks until the
repair evidence packet is structured; otherwise the retry loop will keep
spending attempts on weak instructions.

The current branch contains useful retry-loop work plus generated task-run
artifacts from live reruns. Before starting the first ticket branch, isolate the
source changes from generated result files and lesson traces so review remains
clean.
