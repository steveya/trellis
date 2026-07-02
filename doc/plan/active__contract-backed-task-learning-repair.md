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

Status mirror last synced: `2026-07-02`

### Parent

| Ticket | Title | Status |
| --- | --- | --- |
| `QUA-1131` | Agent learning: contract-backed intra-run repair | Todo |
| `QUA-1138` | Agent learning: deterministic promotion loop | In Progress |

### Ordered Implementation Queue

| Ticket | Title | Status | Depends on |
| --- | --- | --- | --- |
| `QUA-1132` | Agent learning: deterministic repair evidence packets | In Progress | - |
| `QUA-1133` | Agent learning: overlay gates and retry contract enforcement | In Progress | `QUA-1132` |
| `QUA-1134` | Heston ADI: exact runtime binding and transform contract | In Progress | `QUA-1132`, `QUA-1133` |
| `QUA-1135` | Double barrier: PDE and MC assembly contracts | In Progress | `QUA-1132`, `QUA-1133` |
| `QUA-1136` | Autocallable MC: event engine and QMC primitive obligations | In Progress | `QUA-1132`, `QUA-1133` |
| `QUA-1137` | Task learning: promotion-grade evidence and docs closeout | Todo | `QUA-1134`, `QUA-1135`, `QUA-1136` |
| `QUA-1139` | Agent learning: retry attribution contract | In Progress | `QUA-1131` |
| `QUA-1140` | Agent learning: deterministic overlay consumption | In Progress | `QUA-1132`, `QUA-1133` |
| `QUA-1141` | Semantic validation: helper-backed primitive closure | In Progress | `QUA-1135` |
| `QUA-1142` | Semantic contract: static exotic spec catalog | In Progress | `QUA-1134`, `QUA-1136` |
| `QUA-1143` | Task learning: no-LLM scorecard and docs closeout | In Progress | `QUA-1139`, `QUA-1140`, `QUA-1141`, `QUA-1142` |

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

## Progress Log

### 2026-07-02 offline local-agent closeout pass

Created `QUA-1138` as the deterministic-promotion-loop parent and child
tickets `QUA-1139` through `QUA-1143` for attribution, overlay consumption,
helper-backed semantic validation, static exotic specs, and no-LLM scorecard
closeout.

Implemented a no-LLM offline local-agent guard for `scripts/run_tasks.py` and
`scripts/rerun_ids.py`. Offline runs set post-build learning skips, disable
LLM-backed critic/model-validator review through deterministic review policy,
and keep the LLM override guard as a hard backstop.

The final no-LLM rerun:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id T20 --task-id T22 --task-id T105 --task-id T107 --task-id E27 \
  --status all --offline-local-agents --recovery-mode assisted \
  --validation standard \
  --output task_results_qua1138_offline_subset_20260702_final.json
```

reported `5/5` passed expectations in `85s`, with `4` pricing successes,
`1` certified honest block, `0` actionable failures, and zero LLM token usage.
`T20`, `T22`, `T105`, and `T107` passed. `E27` printed as `HONEST_BLOCK` and
remained fail-closed with `passed_expectation=true`.

The bounded remediation check:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/remediate.py \
  --analyze-only \
  --results task_results_qua1138_offline_subset_20260702_final.json \
  --skip-platform-traces
```

loaded `5` results, reported `4` success, `1` fail-closed, `5` passed
expectation, and `0` total failures.

### 2026-07-02 QUA-1139 retry attribution slice

Started `QUA-1139` and added the retry-attribution contract to task runtime
payloads.  Recovery attempts now record whether a candidate was merely
constructed, skipped, or actually consumed structured contract evidence that
changed deterministic build inputs.  The raw attempt record and
`intra_run_learning` summary expose `attribution_kind`,
`contract_evidence_consumed`, `deterministic_input_changed`,
`changed_input_fields`, repair-obligation counts, and observed retry outcome
changes.

Targeted mocked task-runtime tests cover recovered retries and skipped
prose-only candidates without live LLM calls.

### 2026-07-02 QUA-1140 deterministic overlay-consumption slice

Started `QUA-1140` and threaded retry overlays from
`build_with_knowledge(...)` through `build_payoff(...)` into
`compile_build_request(...)`.  The request compiler now consumes available
`required_primitive` and `callable_signature` obligations into
`GenerationPlan` module, symbol, reusable-primitive, and helper-ref fields
before validation and route-binding metadata are finalized.  Compiled request
metadata records `intra_run_learning_overlay_consumption` with candidate ids,
target ids, obligation kinds, applied deterministic inputs, and unapplied
obligations.

Validation so far: the new compiler/autonomous tests pass, the autonomous
suite reports `11 passed`, the platform request suite reports `69 passed`, and
the touched modules compile cleanly.

The no-LLM offline replay:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id T20 --task-id T22 --task-id T105 --task-id T107 --task-id E27 \
  --status all --offline-local-agents --recovery-mode assisted \
  --validation standard \
  --output task_results_qua1140_offline_subset_20260702.json
```

reported `5/5` passed expectations in `91s`, with `4` pricing successes,
`1` certified honest block, `0` actionable failures, and zero LLM token usage.
The bounded remediation analysis for that result file reported `0` failures.

### 2026-07-02 QUA-1141 helper-backed primitive-closure slice

Started `QUA-1141` and tightened semantic algorithm-contract validation for
helper-owned exact routes. Double-barrier PDE and Monte Carlo route helpers now
have explicit callable-surface contracts, and the validator treats a successful
call to those helpers as satisfying the lower-level grid/operator/payoff,
barrier-monitor, and discounting obligations owned inside the helper.

The closure remains fail-closed. Generated code that omits the required helper,
calls only a low-level terminal payoff, or invents raw helper keywords such as
`spot` still emits a semantic route-helper finding.

Validation so far:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q tests/test_agent/test_semantic_validators.py
/Users/steveyang/miniforge3/bin/python3 -m pytest -q tests/test_agent/test_semantic_validation.py::test_accepts_helper_backed_double_barrier_route_with_internal_primitives_subsumed
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id T22 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1141_t22_20260702.json
/Users/steveyang/miniforge3/bin/python3 scripts/remediate.py \
  --analyze-only \
  --results task_results_qua1141_t22_20260702.json \
  --skip-platform-traces
```

The semantic-validator suite reports `59 passed`; the helper-backed
double-barrier semantic-validation test reports `2 passed`; the offline T22
replay reports `1/1` passed expectation with zero LLM calls; and bounded
remediation reports `0` failures.

### 2026-07-02 QUA-1142 static exotic spec audit

Audited `QUA-1142` and found it already satisfied by the existing base repair
stack rather than needing a new code branch. Heston ADI now has exact helper
binding materialization and primitive planning. Autocallable targets now use a
static `AutocallableSpec` under the offline guard and deterministic helper
materialization for pseudo-MC and Sobol QMC targets.

Validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_primitive_planning.py::test_plan_build_uses_static_autocallable_spec_under_offline_guard \
  tests/test_agent/test_primitive_planning.py::test_builds_heston_adi_plan_for_pde_method \
  tests/test_agent/test_executor.py::test_deterministic_exact_binding_module_materializes_heston_adi_helper \
  tests/test_agent/test_executor.py::test_deterministic_exact_binding_module_materializes_autocallable_helper
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id T20 --task-id T107 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1142_t20_t107_20260702.json
/Users/steveyang/miniforge3/bin/python3 scripts/remediate.py \
  --analyze-only \
  --results task_results_qua1142_t20_t107_20260702.json \
  --skip-platform-traces
```

The targeted unit checks report `5 passed`; the offline T20/T107 replay reports
`2/2` passed expectations with zero LLM calls; and bounded remediation reports
`0` failures. Linear `QUA-1142` remains `In Progress`, not `Done`, until the
base PR stack lands.

### 2026-07-02 QUA-1143 no-LLM scorecard closeout

Started `QUA-1143` and reran the full failed-pack closeout with offline local
agents:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id T20 --task-id T22 --task-id T105 --task-id T107 --task-id E27 \
  --status all --offline-local-agents --recovery-mode assisted \
  --validation standard \
  --output task_results_qua1143_offline_closeout_20260702.json
/Users/steveyang/miniforge3/bin/python3 scripts/remediate.py \
  --analyze-only \
  --results task_results_qua1143_offline_closeout_20260702.json \
  --skip-platform-traces
```

The run reported `5/5` passed expectations in `95s`: `T20`, `T22`, `T105`, and
`T107` were `compare_ready` pricing successes; `E27` was an `honest_block`;
token usage was zero; and bounded remediation reported `0` failures. The
important interpretation is that the pack is now first-pass deterministic reuse
of checked contracts rather than retry rescue: `successful_after_retry=0`,
`first_attempt_successes=4`, and all pricing successes had shared context.
