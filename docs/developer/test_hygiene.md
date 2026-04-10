# Test Hygiene

`QUA-428` adds a durable stale-test workflow so Trellis does not quietly
accumulate old `skip`, `xfail`, or quarantine markers as the runtime and gate
surfaces change.

## Local Command

Run the hygiene scan with the repo-standard interpreter:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/test_hygiene.py
```

To make ancient unticketed `xfail` markers fail locally:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/test_hygiene.py --fail-on-ancient-unticketed-xfail
```

The report scans `tests/` for:

- `@pytest.mark.skip(...)`
- `@pytest.mark.skipif(...)`
- `@pytest.mark.xfail(...)`
- `pytest.skip(...)`
- `pytest.xfail(...)`
- `pytest.importorskip(...)`
- `legacy_compat` marker usage

Marker age is approximate. The tool uses `git log` last-touch time for the
owning file and falls back to filesystem mtime when needed.

## Buckets

- `quarantine`: younger than 30 days
- `stale`: 30 to 89 days
- `ancient`: 90 days or older

These buckets are triage hints, not automatic rewrite instructions.

## Decision Tree

1. Does the test still defend a live contract?
2. If yes, update it to the current surface and remove the skip/xfail.
3. If no, delete it instead of leaving dead coverage behind.
4. If the marker is shielding a real bug, include the ticket id in the reason.
5. If the work is host-dependent, token-consuming, or otherwise non-default,
   move it to the correct tier instead of burying it in the baseline unit run.

Examples:

- good: `@pytest.mark.xfail(reason="QUA-428 stale selector still under review")`
- bad: `@pytest.mark.xfail(reason="temporary")`

## Local Enforcement

Pytest collection now blocks `xfail` markers that are both:

- ancient
- missing a linked ticket id such as `QUA-123` or `CR-10`

The enforcement hook is intentionally narrow. It does not fail on every skip,
and it does not rewrite tests for you. It exists to stop long-lived unticketed
xfails from becoming invisible.

For rare local debugging sessions, you can bypass the collection guard with:

```bash
TRELLIS_ALLOW_STALE_XFAIL=1 /Users/steveyang/miniforge3/bin/python3 -m pytest tests -x -q -m "not integration"
```

## Cadence

- run `scripts/test_hygiene.py` when touching test markers
- run it before tightening local or CI gates
- review the report periodically during backlog cleanup and before release work

This tool is advisory for broad stale-test inventory and triage. The normal
non-integration pytest command remains the primary day-to-day correctness gate.

