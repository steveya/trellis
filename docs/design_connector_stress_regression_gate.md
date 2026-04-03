# Connector Stress Regression Gate Design

This document keeps the connector-stress idea as a standing design note rather
than a one-off phase plan.

## Purpose

Turn the connector stress tranche into a repeatable regression gate so later
proving-ground failures can be interpreted as substrate problems instead of
mock market-data drift.

## Task Set

The standing stress tranche is:

- `E21`
- `E22`
- `E23`
- `E24`
- `E25`
- `E26`
- `E27`
- `E28`

It covers the current mock connector surface for:

- `discount_curve`
- `forward_curve`
- `black_vol_surface`
- `spot`
- `local_vol_surface`
- `jump_parameters`
- `credit_curve`
- `fx_rates`
- `model_parameters`

## Outcome Classes

Compare-ready:

- `E21`
- `E22`
- `E23`
- `E25`
- `E28`

Honest-block:

- `E24`
- `E26`
- `E27`

## Source Of Truth

The standing gate is intentionally split across two repo surfaces:

- `TASKS.yaml` defines the executable task contract:
  - task identity
  - market assertions
  - cross-validation targets
- `tests/evals/stress_tasks.yaml` defines the gate contract:
  - outcome class
  - required mock capabilities
  - comparison targets
  - reference targets
  - forbidden failure patterns
  - expected blocker categories

Where those two surfaces overlap, they must agree. Deterministic preflight
should fail on drift rather than letting the live tranche run on stale
assumptions.

## Gate Design

1. Freeze the stress manifest and task specs as the source of truth.
2. Run deterministic preflight against each task's mock market, assertions, and
   manifest-to-task contract alignment.
3. Block malformed batches before live execution and write a preflight report
   that says which task/check stopped the run.
4. Persist one named batch report with outcomes, blocker categories, diagnosis
   links, and market context.
5. Treat compare-ready tasks as real comparable runs:
   they should succeed, keep a passed comparison status, and avoid blocker
   categories entirely.
6. Treat honest-block tasks as controlled failures:
   they may fail, but must surface explicit blocker categories that match the
   manifest instead of generic runtime noise.
7. Run this gate whenever connector or task-runtime market wiring changes.
8. Feed repeated failures into durable follow-on tickets rather than leaving
   them as isolated rerun notes.

## Standing Runner Contract

The standing entrypoint is still:

- `scripts/run_stress_tranche.py`

It now supports the operator modes needed for routine use:

- default reuse mode for cheap standing reruns
- `--fresh` for expensive rebuild validation
- `--preflight-only` to stop after deterministic checks
- `--task-id` to narrow the tranche during diagnosis

Each run writes:

- raw results JSON
- batch report JSON
- batch report Markdown

The batch report should be the first place to look after a rerun because it
surfaces:

- compare-ready vs honest-block membership
- preflight failures
- failure buckets
- observed blocker categories
- latest diagnosis packet and dossier paths
- follow-on candidates for repeated failures

## Standing Rerun Rule

Run the standing gate after connector or task-runtime market-wiring changes.

Default local workflow:

1. Run `scripts/run_stress_tranche.py` in reuse mode.
2. If reuse-mode output changes in a meaningful way, inspect the batch report
   and linked diagnosis dossiers first.
3. If the change is still ambiguous, rerun the narrowest failing subset with
   `--fresh`.
4. Only after that should a maintainer widen the investigation into route or
   substrate work.

## Validation Commands

Deterministic:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_stress_task_preflight.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_evals.py -q
```

Live:

- run `E21` through `E28`
- inspect the batch report plus `task_runs/latest/` and
  `task_runs/diagnostics/latest/`

## Related Linear Tickets

Status snapshot as of 2026-04-03:

- `QUA-306` Connector stress ops: E21-E28 standing regression gate
- `QUA-304` Connector stress manifest: deterministic preflight and market-data alignment
- `QUA-307` Connector stress manifest: freeze task contract and source-of-truth rules
