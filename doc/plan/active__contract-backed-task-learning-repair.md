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

Status mirror last synced: `2026-07-03`

### Parent

| Ticket | Title | Status |
| --- | --- | --- |
| `QUA-1131` | Agent learning: contract-backed intra-run repair | In Progress |
| `QUA-1138` | Agent learning: deterministic promotion loop | In Progress |
| `QUA-1151` | Task learning: replay-safe self-learning closure | In Progress |

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
| `QUA-1152` | Canary replay: deterministic exact-binding contract lane | Done | `QUA-1138` |
| `QUA-1153` | Task learning: failure-seeded retry benchmark | In Progress | `QUA-1152` |
| `QUA-1154` | Semantic route materialization: pack-2 offline closure | In Progress | `QUA-1131` |

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

### 2026-07-02 QUA-1152 deterministic canary replay lane

Started `QUA-1152` after the stacked PRs failed `pr-gate-tier2-contracts`.
The failing CI path was `T13` full-task canary replay.  Pricing had become
deterministic, but the old full-task cassette still recorded generation and
critic calls, so the replay failed on stale prompt hashes or unconsumed calls.

The fix added a deterministic exact-binding replay lane to
`scripts/run_canary.py`.  Canary entries with
`replay_mode: deterministic_exact_binding` run the normal `run_task(...)`
surface under the offline-local LLM guard, persist the same diagnosis packet
and dossier artifacts, and mark the result with
`execution_mode=deterministic_replay` plus `llm_cassette.used=false`.
T13 now uses that lane.  Cassette replay remains strict for canaries that
actually need recorded LLM calls.

The same slice added deterministic exact-binding materialization for
`price_vanilla_equity_option_pde(...)` comparison targets.  The generated
wrapper maps `theta_0.5` to `theta=0.5` and `theta_1.0` to `theta=1.0`, then
delegates to the checked helper instead of regenerating PDE code.

Validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_executor.py::test_deterministic_exact_binding_module_materializes_vanilla_equity_pde_helper_wrapper \
  tests/test_agent/test_canary_runner.py::TestRunCanaries::test_run_canaries_deterministic_replay_uses_offline_local_scope_without_cassette \
  tests/test_agent/test_canary_runner.py::TestCanaryFileValidity::test_real_canary_file_marks_t38_as_live_only_during_route_refactor \
  tests/test_agent/test_canary_runner.py::TestCanaryFileValidity::test_real_canary_file_covers_canary_kinds \
  tests/test_contracts/test_cassette_contract_helpers.py
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_contracts/test_canary_replay_contracts.py \
  -m "tier2 and not freshness"
make gate-tier2-contracts PYTHON=/Users/steveyang/miniforge3/bin/python3
```

The focused unit/manifest/contract-helper set reports `10 passed`; full
canary replay reports `1 passed, 1 skipped`; and the local tier-2 contract
shard reports `32 passed, 15 skipped, 7 deselected`.

### 2026-07-03 QUA-1153 failure-seeded retry benchmark

Started `QUA-1153` after `QUA-1152` landed. The goal of this slice is to
prove actual intra-run learned recovery, not just first-pass deterministic
reuse from checked exact bindings.

The benchmark runner now supports a local seeded retry fixture:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py \
  --seeded-retry-fixture --passes 1 --knowledge-light \
  --report-name qua1153_seeded_retry_20260703
```

The fixture bypasses manifest task selection and uses a local fake builder. Its
first build fails with a concrete callable-signature contract error against
`trellis.models.equity_option_pde.price_vanilla_equity_option_pde`; the
assisted retry receives the structured `KnowledgePatchCandidate` and succeeds
only after `knowledge_overlays` is present. The saved scorecard reports
`retry_learned_recoveries=["L001"]`, `first_pass_deterministic_reuse=[]`,
`retry_taxonomy.by_stage.contract_evidence_consumed.count=1`, and token usage
`0`.

Validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_task_learning_benchmark.py \
  tests/test_agent/test_task_learning_benchmark_runner.py \
  tests/test_agent/test_evals.py
/Users/steveyang/miniforge3/bin/python3 scripts/run_task_learning_benchmark.py \
  --seeded-retry-fixture --passes 1 --knowledge-light \
  --report-name qua1153_seeded_retry_20260703
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id T20 --task-id T22 --task-id T105 --task-id T107 --task-id E27 \
  --status all --offline-local-agents --recovery-mode assisted \
  --validation standard \
  --output task_results_qua1153_offline_closeout_20260703.json
/Users/steveyang/miniforge3/bin/python3 scripts/remediate.py \
  --analyze-only \
  --results task_results_qua1153_offline_closeout_20260703.json \
  --skip-platform-traces
```

The focused unit files report `28 passed`; the seeded fixture reports `1/1`
success, attempts-to-success `2.0`, and zero token usage; the failed-pack
closeout remains `5/5` passed expectations with `4` pricing successes, `1`
honest block, `0` actionable failures, and zero token usage; bounded
remediation reports `0` failures. PR creation for this slice is intentionally
deferred under the current commits-only goal constraint.

### 2026-07-03 F007 CDS exact-binding offline closure

The next pending offline pack found one actionable failure:
`F007` selected the single-name CDS analytical route and exact CDS helper
surface, but offline execution still fell through to live LLM generation
instead of materializing a thin deterministic helper-backed wrapper.

The fix adds deterministic exact-binding materialization for CDS analytical and
Monte Carlo route helpers. The generated adapter builds the CDS schedule from
`CDSSpec`, requires `market_state.credit_curve` and `market_state.discount`,
and delegates to `price_cds_analytical(...)` or `price_cds_monte_carlo(...)`.

Validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_executor.py::test_deterministic_exact_binding_module_materializes_cds_analytical_wrapper \
  tests/test_agent/test_executor.py::test_deterministic_exact_binding_module_materializes_cds_monte_carlo_wrapper
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_executor.py \
  -k "deterministic_exact_binding_module or generate_skeleton_prefills_cds"
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id F007 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1131_f007_exact_binding_20260703.json
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id F004 --task-id F005 --task-id F007 --task-id F009 \
  --task-id F010 --task-id F011 --task-id F012 --task-id F013 \
  --task-id F014 --task-id F015 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1131_pending_pack1_fixed_20260703.json
/Users/steveyang/miniforge3/bin/python3 scripts/remediate.py \
  --analyze-only \
  --results task_results_qua1131_pending_pack1_fixed_20260703.json \
  --skip-platform-traces
```

The focused CDS wrapper tests report `2 passed`; the exact-binding executor
slice reports `57 passed`; isolated `F007` reports `1/1` pricing success with
zero LLM calls; the full pending pack reports `10/10` passed expectations,
all first-attempt offline successes, zero actionable failures, and zero token
usage; bounded remediation reports `0` failures.

### 2026-07-03 QUA-1154 pack-2 route-materialization slice

Started `QUA-1154` after the next offline local-agent pack reported actionable
failures for `P002`, `P004`, `P006`, `P007`, `T14`, `T15`, and `T16`; `T18`
remained an honest manifest block and is excluded from pricing remediation.

The first implementation slice closes `P002` and `P006` without LLM calls.
`P006` now materializes a deterministic thin wrapper over
`price_nth_to_default_basket(...)` when the route compiler selects that exact
binding. `P002` now materializes a deterministic ranked-observation basket
wrapper over `price_ranked_observation_basket_monte_carlo(...)` and the shared
basket event-state substrate supports locked selected price levels for
level-based average-best-of contracts.

This slice also fixed a reduced-state MC bug: the ranked-observation
state-aware payoff now applies the option payoff transform to the replayed
basket aggregate instead of returning the raw aggregate level. That restores
the expected volatility sensitivity for `P002`.

Validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_models/test_monte_carlo/test_basket_substrate.py \
  tests/test_models/test_monte_carlo/test_event_state.py
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_executor.py \
  -k "ranked_basket_wrapper or nth_to_default_wrapper or deterministic_exact_binding_module"
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id P002 --task-id P006 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1154_p002_p006_payoff_20260703.json
```

The basket/event substrate tests report `13 passed`; the exact-binding
executor slice reports `58 passed`; the offline `P002/P006` replay reports
`2/2` passed expectations, first-attempt successes, zero actionable failures,
and zero token usage. Remaining `QUA-1154` targets are `P004`, `P007`, `T14`,
`T15`, and `T16`.

The second implementation slice closes `P004` without LLM calls. The executor
now reads exact backend/helper refs from both object-shaped and mapping-shaped
generation plans, including nested `route_binding_authority` payloads emitted by
the compiler. The planner now has a deterministic
`period_rate_option_strip` spec schema that exposes cap/floor collar aliases,
call dates, and schedule tweaks, so offline-local runs do not need LLM spec
design before exact helper materialization.

Validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_planner.py tests/test_agent/test_executor.py \
  -k "period_rate_option_strip or cap_strip or exact_binding_refs_collect_backend_helper_refs or deterministic_exact_binding_module"
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id P004 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1154_p004_static_spec_20260703.json
```

The planner/executor regression slice reports `64 passed`; the offline `P004`
replay reports `1/1` passed expectations, first-attempt success, zero actionable
failures, and zero token usage. Remaining `QUA-1154` targets are `P007`, `T14`,
`T15`, and `T16`.

The third implementation slice closes `P007` without LLM calls. The static
`cliquet_option` spec now carries local/global cap and floor fields, reset-time
day-count control, quadrature order, path count, and seed. Analytical cliquet
pricing preserves the existing uncapped FinancePy-parity path and adds a
bounded Gauss-Hermite reset-return integrator for capped/floored cliquets. The
Monte Carlo layer now exposes a checked reset-date GBM cliquet helper, and the
deterministic exact-binding materializer can emit a thin adapter over that
helper for the MC comparison target.

The validation gates were tightened to match cliquet semantics: volatility
sensitivity remains active, but generic volatility monotonicity is not enforced
for cliquet options because local/global caps and floors can make the capped
return value non-monotone in Black volatility. The semantic and lite-review
route-helper gates now recognize the checked cliquet MC helper as satisfying
the lower-level `monte_carlo_paths` route obligation for `cliquet_option`,
without relaxing ordinary missing-helper checks.

Validation:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache \
  /Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_assembly_tools.py \
  tests/test_agent/test_validation_bundles.py \
  tests/test_agent/test_semantic_validation.py \
  tests/test_agent/test_semantic_validators.py \
  tests/test_agent/test_lite_review.py \
  tests/test_agent/test_planner.py \
  tests/test_agent/test_executor.py \
  tests/test_models/test_equity_exotics_analytical.py
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id P007 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1154_p007_cliquet_final_20260703.json
```

The local regression pass reports `293 passed`; the offline `P007` replay
reports `1/1` passed expectations, first-attempt success, zero actionable
failures, and zero token usage. Remaining `QUA-1154` targets are `T14`, `T15`,
and `T16`.

The fourth implementation slice closes `T14` without LLM calls. Sparse legacy
American-put text now compiles to the American-option route contract, and the
deterministic adapter delegates the LSM comparison target to
`price_american_equity_option_lsm_monte_carlo(...)` instead of trying to
assemble raw GBM, regression, and exercise logic inside generated code.

Validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_agent/test_primitive_planning.py \
  tests/test_agent/test_platform_requests.py \
  tests/test_agent/test_task_runtime.py \
  tests/test_agent/test_executor.py
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id T14 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1154_t14_american_lsm_v4_20260703.json
```

The focused regression pass and offline replay were green, with first-attempt
task success and zero token usage. Remaining `QUA-1154` targets are `T15` and
`T16`.

The fifth implementation slice closes `T15` without LLM calls. Sparse CEV proof
text now bridges to a `vanilla_option` contract with `model_family=cev_diffusion`
and a `cev_process` trait. The PDE and tree comparison lanes bind to
`price_cev_option_pde(...)` and `price_cev_option_tree(...)`; validation uses
the `*:cev_option` bundle so Black-vol-surface monotonicity checks do not
misclassify explicit CEV-parameter helpers.

Validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_models/test_equity_option_pde.py \
  tests/test_models/test_equity_option_tree.py \
  tests/test_agent/test_backend_bindings.py \
  tests/test_agent/test_route_registry.py \
  tests/test_agent/test_primitive_planning.py \
  tests/test_agent/test_validation_bundles.py \
  tests/test_agent/test_validation_contract.py \
  tests/test_agent/test_platform_requests.py \
  tests/test_agent/test_task_runtime.py \
  tests/test_agent/test_planner.py \
  tests/test_agent/test_executor.py \
  tests/test_agent/test_codegen_guardrails.py \
  tests/test_agent/test_import_registry.py \
  tests/test_agent/test_knowledge_store.py -x
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  T14 T15 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1154_t14_t15_v1_20260703.json
```

The broad touched-suite pass reports `724 passed`; the offline `T14/T15` replay
reports `2/2` passed expectations, first-attempt successes, zero actionable
failures, and zero token usage. Remaining `QUA-1154` target is `T16`.

The sixth implementation slice closes `T16` without LLM calls. Ordinary
barrier-option text now receives a `single_barrier` payoff trait, distinct from
`double_barrier`. The PDE and MC comparison lanes bind to
`price_single_barrier_option_pde_result(...)` and
`price_single_barrier_option_monte_carlo_result(...)`; the helper owns the
absorbing barrier boundary, far vanilla boundary, single `BarrierMonitor`,
notional convention, and deterministic discounting.

Validation:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest -q \
  tests/test_models/test_single_barrier_option.py \
  tests/test_agent/test_decomposition_ir.py::TestProductIR::test_ir_for_barrier_option_includes_promoted_analytical_support \
  tests/test_agent/test_primitive_planning.py::test_builds_pde_plan_for_barrier_option_uses_grid_and_operator \
  tests/test_agent/test_primitive_planning.py::test_builds_mc_plan_for_barrier_option_uses_single_barrier_helper \
  tests/test_agent/test_backend_bindings.py::test_resolve_backend_binding_spec_uses_single_barrier_exact_helpers \
  tests/test_agent/test_executor.py::test_deterministic_exact_binding_module_materializes_barrier_helpers \
  tests/test_agent/test_platform_requests.py::test_compile_build_request_preserves_single_barrier_exact_binding_for_t16_targets \
  tests/test_agent/test_import_registry.py::test_single_barrier_helpers_are_visible_to_import_registry \
  tests/test_agent/test_codegen_guardrails.py::test_barrier_family_support_approves_shared_barrier_primitives
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id T16 --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1154_t16_single_barrier_20260703.json
```

The focused regression pass reports `17 passed`; the offline `T16` replay
reports `1/1` passed expectations, first-attempt success, zero actionable
failures, and zero token usage. `T16` prices were `pde_barrier=23418.55`,
`mc_barrier=23700.11`, and `rubinstein=23463.24`, with both comparison lanes
within tolerance. All named `QUA-1154` pack-2 targets are now green and ready
for a final pack replay.

Final `QUA-1154` pack replay:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id P002 --task-id P004 --task-id P006 --task-id P007 \
  --task-id T14 --task-id T15 --task-id T16 \
  --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1154_final_pack_20260703.json
```

The final replay reports `7/7` passed expectations in `200s`, all
first-attempt successes, zero actionable failures, zero lessons/cookbooks
captured, and zero token usage. The successful tasks are `P002`, `P004`,
`P006`, `P007`, `T14`, `T15`, and `T16`.

Full pending-pack replay including previously green and expected-block targets:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/run_tasks.py \
  --task-id P001 --task-id P002 --task-id P004 --task-id P006 \
  --task-id P007 --task-id T14 --task-id T15 --task-id T16 \
  --task-id T17 --task-id T18 \
  --status all --offline-local-agents \
  --recovery-mode assisted --validation standard \
  --output task_results_qua1154_full_pending_pack_20260703.json
/Users/steveyang/miniforge3/bin/python3 scripts/remediate.py \
  --analyze-only \
  --results task_results_qua1154_full_pending_pack_20260703.json \
  --skip-platform-traces
```

The full replay reports `10/10` passed expectations in `248s`, with `9`
pricing successes, `1` honest block (`T18`), `0` actionable failures, all
pricing tasks succeeding on the first attempt, and zero token usage. Bounded
remediation reports `0` total failures.
