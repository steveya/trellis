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

- `make gate-pr`: default PR-ready gate. Runs stale-test hygiene, the full
  non-integration suite, and the explicit tier-2 contract-test tranche.
- `make gate-canary`: focused core canary gate for agent/runtime/pricing-core
  changes. This calls `scripts/should_run_canary.py` first so the current path
  diff is visible before the live canary subset runs.
- `make gate-release`: broader release-facing gate. Runs `gate-pr`, then the
  cassette freshness contract and replay-backed canary drift checks for the
  committed full-task cassettes.

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
- `TASKS.yaml`, `CANARY_TASKS.yaml`, `scripts/run_canary.py`,
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
  tokens.
- `gate-canary` is live by default and intentionally narrower than the full
  canary surface.
- `gate-release` stays deterministic and zero-token by using committed
  full-task replays plus drift/freshness checks.

The release gate's drift step uses `scripts/run_canary.py --check-drift`, so
it compares against the latest available decision checkpoint for the replayed
task. If no latest checkpoint exists yet, the runner reports that the drift
check was skipped. Treat that as a signal to refresh the task with a live
canary run before leaning on drift output for a release claim.

If you only want to smoke-test the `gate-canary` wiring without spending
tokens, use:

```bash
make gate-canary CANARY_FLAGS=--dry-run
```
