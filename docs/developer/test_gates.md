# Test Gates

`QUA-430` turns Trellis's current test pyramid into explicit local gate
commands from repo root.

## Commands

All commands use the repo-standard interpreter through the root `Makefile`:

```bash
make gate-pr
make gate-canary
make gate-release
```

The `Makefile` defaults `PYTHON` to
`/Users/steveyang/miniforge3/bin/python3`, so the gate commands stay pinned to
the supported interpreter even if your shell `python3` resolves elsewhere.

## Gate Selection

- `make gate-pr`: default PR-ready gate. Runs stale-test hygiene, the core
  non-integration suite, and the tier-2 contract tranche, but skips the
  cross-validation, verification, task-challenge, and cassette-freshness proof
  layers so ordinary merges are not blocked on the slowest numerical evidence
  slices.
- `make gate-canary`: focused core canary gate for agent/runtime/pricing-core
  changes. This calls `scripts/should_run_canary.py` first so the current path
  diff is visible before the live canary subset runs.
- `make gate-release`: broader release-facing gate. Runs `gate-pr`, then the
  cross-validation, verification, task-challenge, and cassette-freshness proof
  layers, followed by the replay-backed canary drift checks for the committed
  full-task cassettes.

## Canary Trigger Helper

Use the trigger helper before paying for the live canary subset:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/should_run_canary.py
```

By default it reads local changed files from `git status --porcelain` and
recommends the focused `core` canary subset when changes touch:

- `trellis/agent/`, `trellis/core/`, `trellis/curves/`, `trellis/models/`, or
  `trellis/instruments/`
- `tests/test_agent/`, `tests/test_contracts/`, or `tests/test_tasks/`
- `TASKS_BENCHMARK_FINANCEPY.yaml`, `TASKS_EXTENSION.yaml`,
  `TASKS_NEGATIVE.yaml`, `TASKS_PROOF_LEGACY.yaml`, `MARKET_SCENARIOS.yaml`,
  `FINANCEPY_BINDINGS.yaml`, `CANARY_TASKS.yaml`, `scripts/run_canary.py`,
  `scripts/canary_common.py`, or `scripts/record_cassettes.py`

For explicit path checks, repeat `--path`:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/should_run_canary.py \
  --path trellis/agent/task_runtime.py \
  --path docs/developer/test_gates.md \
  --json
```

## Cost Notes

- `gate-pr` is the normal correctness gate and does not require live model
  tokens. It intentionally avoids the slower proof/reference strata so the
  merge path stays closer to the core runtime contract.
- `gate-canary` is live by default and intentionally narrower than the full
  canary surface.
- `gate-release` stays deterministic and zero-token by using committed
  full-task replays plus the slower proof/reference suites and the
  drift/freshness checks.

## CI Layout

GitHub Actions now fans the PR gate out into independent jobs instead of one
large serial pytest invocation:

- `repo-checks`: build, whitespace diff check, and stale-marker hygiene
- `pr-gate-shard-1` through `pr-gate-shard-4`: deterministic slices of the
  core PR test surface produced by `scripts/pr_gate_shard.py`
- `pr-gate-tier2-contracts`: the retained tier-2 contract tranche
- `build-and-test`: branch-protection aggregator that goes green only after
  the PR shard jobs succeed
- `typecheck`: the existing non-blocking mypy pass

Local `make gate-pr` stays serial and unchanged so developers still have a
single command for full PR-ready validation. The shard helper only changes the
remote PR workflow shape.

The release gate's drift step uses `scripts/run_canary.py --check-drift`, so
it compares against the latest available decision checkpoint for the replayed
task. If no latest checkpoint exists yet, the runner reports that the drift
check was skipped. Treat that as a signal to refresh the task with a live
canary run before leaning on drift output for a release claim.

Those checkpoints are now binding-first. Fresh replay and live artifacts emit
binding identity at the generation boundary, and older route-era checkpoint
files are normalized on load so drift output is not polluted by the route to
binding stage rename itself.

If you only want to smoke-test the `gate-canary` wiring without spending
tokens, use:

```bash
make gate-canary CANARY_FLAGS=--dry-run
```
