## Task Run Persistence

This tranche adds a canonical per-task run store so the latest run for each task is
available without reconstructing it from batch JSON files or raw platform traces.

### What is persisted

Each `run_task(...)` call now writes:

- history record:
  - `task_runs/history/<TASK_ID>/<RUN_ID>.json`
- latest record for the task:
  - `task_runs/latest/<TASK_ID>.json`
- aggregated latest index across tasks:
  - `task_results_latest.json`

### Record contents

The persisted record includes:

- task snapshot from `TASKS.yaml`
- full task result payload
- comparison summary:
  - targets
  - prices
  - deviations
  - reference target
- market context used for the run
- method-level results for comparison tasks
- artifact paths
- platform-trace summaries
- linked Linear/GitHub issue refs extracted from traces
- derived workflow status and next-action summary

### Why this matters

This is the persistence layer needed to answer questions like:

- Why did `T104` succeed?
- Which methods were used?
- What were the prices and deviations?
- Why did `T105` fail?
- Is there an active blocker, running build, or linked issue?

The UI can now read one canonical latest record per task instead of inferring all of
that from merged historical `task_results_*.json` files.

### Known limitation

Older historical batch files are not automatically backfilled into the new
`task_runs/latest/` store. New task runs populate it immediately; a separate backfill
utility can be added later if we want the full historical corpus normalized.
