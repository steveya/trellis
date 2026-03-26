# Shared Memory Evaluation Workflow

Date: 2026-03-25

## Goal

Measure whether the shared-memory substrate is reducing repeated agent mistakes
without relying on anecdotal reruns.

## Five-part report

The deterministic comparison report now covers:

1. outcome summary
2. task transitions
3. failure bucket deltas
4. reviewer signal deltas
5. retry and shared-knowledge deltas

## New tooling

### Compare two task-result tranches

Use:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/evaluate_shared_memory.py \
  task_results_baseline.json \
  task_results_candidate.json \
  --output shared_memory_report.json
```

This prints the five-part report to stdout and optionally saves the full JSON
comparison.

### Per-run sidecar summary

`scripts/run_tasks.py` now writes a sidecar summary next to the main results
file:

- `task_results_foo.json`
- `task_results_foo_summary.json`

The summary includes:

- outcome totals
- failure bucket counts
- retry recovery metrics
- reviewer signal metrics
- shared-knowledge coverage

## Intended usage

1. rerun a fixed failure tranche
2. compare the new result file against the older baseline
3. inspect which failures disappeared vs merely changed bucket
4. inspect whether reviewer-driven recovery improved
5. inspect whether shared-knowledge coverage is increasing across the run

## Current limitation

This workflow is deterministic and result-file based. It measures visible
outcomes and captured reviewer signals, but it does not yet score semantic
quality directly from raw prompt/response pairs or token-level agent behavior.
