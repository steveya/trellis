# Canary Suite Stabilization

## Objective

Turn the curated canary set back into a useful regression and release-readiness
signal by fixing stale inventory, recovering the highest-value failing canaries,
and making canary cost and iteration metrics trustworthy enough to compare over
time.

## Why Now

The live canary run on 2026-04-06 does not support strong release claims. The
current batch is useful as a diagnostic probe, but not yet as a dependable
gate:

- the canary runner currently loads only `pending` tasks, so valid curated
  canaries with lifecycle status `done` (`T01`, `T02`) are misreported as
  stale inventory
- several core canaries fail before any real build/retry loop starts
- several complex canaries still consume large token/time budgets and multiple
  retries before failing with `insufficient_results`
- the repo does not yet carry a trustworthy latency/token/attempt baseline that
  can answer “is the suite getting faster and cheaper?” without ambiguity

The current measured state from the live run is:

- listed canaries: 14
- skipped from stale inventory: 2
- executed: 12
- passed: 2 (`T13`, `T38`)
- failed: 10
- executed pass rate: `16.7%`
- average executed elapsed time: `76.4s`
- average executed tokens: `50.4k`
- average executed attempts: `2.75`

Result artifacts from the run:

- `canary_results_20260406_current.json`
- `canary_T105_20260406_current.json`
- `canary_T39_20260406_current.json`
- `canary_T40_20260406_current.json`
- `canary_T49_20260406_current.json`
- `canary_T65_20260406_current.json`

## Scope

- canary inventory hygiene for the curated task list
- direct remediation of the current high-value failing canaries
- reduction of zero-attempt pre-build failures on simple canaries
- reduction of high-attempt expensive failures on complex canaries
- trustworthy canary replay and telemetry so cost/latency/attempt trends can be
  compared over time
- local gate guidance that eventually turns the canary suite back into a
  practical release surface

## Non-Goals

- broad product-family expansion unrelated to the current canary failures
- redesigning the entire validation architecture again
- replacing the existing canary runner or golden-trace machinery from scratch
- forcing hosted CI/provider rollout in this workstream

## Failure Taxonomy

The current failures cluster into a few distinct buckets:

### 1. Inventory lookup drift

- `T02`

These were originally skipped because the runner filtered the live task
registry to `pending` tasks before building the canary lookup. `T01` is no
longer in this bucket; it now fails on short-rate comparison-regime assembly
and is tracked separately under the short-rate comparison-regime workstream.

### 2. Early clarification / front-door failures

- `T25`
- `T26`
- `T39`

These fail with zero attempts because the task front door still asks for
clarification on simple canonical canaries.

### 3. Cross-validation parity failures

- `T105`

This is also reflected by the known repo-wide non-integration failure in the
quanto comparison test.

### 4. Missing constructive or bound lane

- `T51`

The credit lane is still missing a constructive/evaluable path for the par
spread bootstrap variant.

### 5. Concrete post-build runtime bug

- `T65`

The current failing path contains a direct smoke/runtime bug
(`exercise_time` undefined) after substantial work has already been done.

### 6. Expensive `insufficient_results` cluster

- `T17`
- `T73`
- `T40`
- `T49`

These are the most expensive failures because they consume real iteration
budget before ending without enough successful comparison outputs.

## Ordered Queue

The recommended queue is:

1. canary inventory hygiene
2. quanto comparison parity (`T105`)
3. clarification-gate cleanup for simple MC / FFT canaries (`T25`, `T26`, `T39`)
4. calibration smoke bug fix (`T65`)
5. credit constructive lane for `T51`
6. expensive `insufficient_results` remediation for:
   - `T17`
   - `T73`
   - `T40`
   - `T49`
7. trustworthy replay/telemetry/gate hardening

This order is intentional:

- remove fake failures first
- fix the highest-leverage correctness regression next
- eliminate the cheapest avoidable front-door failures before working on the
  expensive deep-loop failures
- fix direct code bugs before broad iterative tuning
- only after the suite is more stable, invest in lower-cost replay and durable
  gate policy

## Reused Existing Tickets

These existing backlog tickets should be reused instead of duplicated:

- `QUA-458` canary replay: full-task cassette path with diagnosis packet parity
- `QUA-430` CI gate configuration for the test pyramid

This existing ticket is related but broader than the current remediation queue:

- `QUA-428` stale test triage process and tooling

## Agent Intake Bundle

Each agent assigned to this workstream should begin with:

- this plan doc
- `AGENTS.md`
- `CANARY_TASKS.yaml`
- `scripts/run_canary.py`
- the latest result artifacts from the 2026-04-06 run
- `tests/test_agent/test_canary_runner.py`
- the failing canary task definition(s) in `TASKS.yaml`
- the latest task-run history for the canary(s) they own under `task_runs/`

## Linear Mirror

Status mirror last synced: `2026-04-10`

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-700` Canary suite: stabilization, remediation, and gate readiness | Backlog |

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-701` | Inventory hygiene for `CANARY_TASKS.yaml` and `TASKS.yaml` | Done |
| `QUA-702` | Quanto comparison parity for `T105` | Done |
| `QUA-703` | Clarification-gate cleanup for `T25`, `T26`, and `T39` | Done |
| `QUA-704` | Calibration smoke/runtime fix for `T65` | Done |
| `QUA-705` | Credit constructive lane for `T51` | Done |
| `QUA-706` | PDE callable cross-validation recovery for `T17` | Done |
| `QUA-707` | Analytical swaption cross-validation recovery for `T73` | Done |
| `QUA-708` | Transform/Heston cross-validation recovery for `T40` | Done |
| `QUA-709` | Copula cross-validation recovery for `T49` | Done |
| `QUA-458` | Full-task canary replay with diagnosis parity | Done |
| `QUA-710` | Trustworthy canary telemetry and historical baselines | Backlog |
| `QUA-430` | Local gate and release-gate configuration | Backlog |

Note:

- `QUA-700` now remains open only for telemetry / gate closeout
  (`QUA-710`, `QUA-430`). The direct canary recovery tranche and the
  full-task replay tranche (`QUA-458`) are complete after the `2026-04-09`
  full curated rerun plus the `2026-04-10` cassette-backed replay landing.
- `T01` is now green through the short-rate comparison-regime workstream in
  `docs/plans/short-rate-comparison-regime-and-claim-helpers.md` under
  `QUA-746` through `QUA-751`. The recovery path materializes task-level
  short-rate comparison assumptions onto `MarketState` and keeps the tree and
  analytical comparators on the shared short-rate helper layer instead of
  rediscovering model literals from prose.
- `QUA-706` is now satisfied through the event-aware PDE lane landed under
  `docs/plans/event-aware-pde-lane.md`. The acceptance ticket closes the canary
  using the recovered PDE/tree path rather than a local helper-only patch.
- `QUA-707` is now satisfied through `QUA-725` under
  `docs/plans/event-aware-monte-carlo-lane.md`. The recovery path stays on the
  generic event-aware Monte Carlo family and the stabilized helper-backed
  comparison wrappers rather than introducing a swaption-specific Monte Carlo
  lane.
- `QUA-708` and `QUA-709` are now satisfied through the helper-layer extraction
  workstream in `docs/plans/general-helper-layer-extraction.md`. `T40` stays on
  the reusable transform and MC helper surfaces, and `T49` stays on the
  semantic-facing basket-credit helper layer rather than the older raw
  nth-to-default boundary.

## Validation Posture

This workstream should validate at three levels.

### Local

- task-specific reruns for the canary being remediated
- direct subsystem tests around the touched lane
- explicit inspection of attempts, token spend, and elapsed time

### Regional

- re-run the smallest affected family subset after each fix
- compare canary result shape and diagnosis packet shape to existing task-run
  expectations

### Global

- re-run the full curated canary set after the queue stabilizes
- report pass rate, average elapsed time, average tokens, and average attempts
- only claim speed or iteration improvements when the suite is measured against
  a trustworthy recorded baseline

Current full-rerun checkpoint:

- `2026-04-09` curated canary rerun passed `14/14`
- total tokens: `662,446`
- total time: `1,472.1s`
- result artifact: `canary_results_20260409_full_rerun_budget10.json`
