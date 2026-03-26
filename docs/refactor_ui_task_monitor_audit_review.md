# UI Task Monitor Audit Trail Review

Date: 2026-03-25

## Summary

The Task Monitor now uses the canonical task-run store as its primary source of truth instead of reconstructing state from legacy `task_results_*.json` files alone.

## What Changed

- `trellis-ui` backend now exposes:
  - `GET /api/tasks/runs/latest`
  - `GET /api/tasks/{task_id}/runs`
- These endpoints are backed by:
  - `task_results_latest.json`
  - `task_runs/latest/*.json`
  - `task_runs/history/<TASK_ID>/*.json`
  - the live overlay in `task_results_live.json`
- The Task Monitor frontend now reads canonical latest-run records and renders:
  - workflow status and next action
  - run id and persisted timestamp
  - latest comparison prices and deviations
  - linked Linear/GitHub issues
  - trace summaries and platform-trace links
  - recent run history for the expanded task row

## Why This Fix Was Needed

Previously the monitor still depended on the compatibility `/tasks/results` payload. That meant:

- live runs could be missed if only the canonical latest-run store had the up-to-date state
- expanded rows had only partial result data and had to infer audit details
- persisted workflow/trace/issue context was not surfaced directly even though it already existed on disk

## Current Source Of Truth

For Task Monitor status and audit trails, the source of truth is now:

1. canonical latest task-run records
2. live overlay for in-flight runs
3. legacy `task_results_*.json` only as a fallback when no canonical record exists yet

## UI Task-Run Model Default

The UI task-run path no longer hardcodes `gpt-5-mini` when the request omits a
model. The backend now defers to Trellis' active configured provider/model
default, so UI-triggered runs inherit the same `LLM_PROVIDER` / default-model
selection as the rest of the platform.

## Remaining Limitation

The Task Monitor polls the canonical latest-run endpoint every 5 seconds; it does not yet stream full canonical run-record updates over SSE. SSE still drives same-tab progress updates, while polling keeps the page aligned with externally started runs and persisted audit trails.
