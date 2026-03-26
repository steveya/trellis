# Phase M2.3 Plan: Connector Stress Tranche Regression Gate

## Goal

Turn the existing connector stress tranche into a standing regression gate so we
can trust later proving-ground failures as substrate problems rather than mock
market-data drift.

## Task Set

The active stress tranche is:

- `E21`
- `E22`
- `E23`
- `E24`
- `E25`
- `E26`
- `E27`
- `E28`

These tasks already cover the current mock connector surface:

- `discount_curve`
- `forward_curve`
- `black_vol_surface`
- `spot`
- `local_vol_surface`
- `jump_parameters`
- `credit_curve`
- `fx_rates`
- `model_parameters`

## Intended Outcome Classes

### Compare-ready

These should compile and run against the mock connector without failing for
missing market data:

- `E21`
- `E22`
- `E25`
- `E28`

### Honest-block

These may still fail, but only for real substrate / route reasons:

- `E23`
- `E24`
- `E26`
- `E27`

## Plan

### Step 1: Freeze the tranche contract

Make the stress manifest and task specs the source of truth:

- `TASKS.yaml`
- `tests/evals/stress_tasks.yaml`

Checks:
- every stress task has explicit `market:` and `market_assertions:`
- every stress task has the expected outcome class in the eval manifest
- compare-ready tasks list forbidden market-data failure patterns

## Step 2: Strengthen deterministic preflight

Before any live run:

- load each task from `TASKS.yaml`
- resolve its task-specific mock market state
- validate `market_assertions`
- grade it against `tests/evals/stress_tasks.yaml`

Outputs:
- per-task preflight pass/fail
- resolved capability set
- selected market components

Fail-fast rule:
- if preflight fails, do not run the live batch

## Step 3: Add a canonical batch runner/report for the tranche

Run the tranche as a named batch and persist one report artifact that records:

- task id
- outcome class
- latest run id
- success/failure
- failure bucket
- market context
- comparison status
- forbidden failure-pattern hits

Suggested output:
- `task_results_stress_connector_<timestamp>.json`
- `task_results_stress_connector_<timestamp>_summary.json`

## Step 4: Define pass/fail rules for the tranche

The tranche passes only if:

- compare-ready tasks do not fail for missing connector inputs
- honest-block tasks fail for substrate/unsupported-route reasons, not missing
  connector inputs
- no task fails with stale generic capability errors
- no task fails with empty-response parser errors

The tranche fails if:

- market assertions fail
- compare-ready tasks regress into market-data failures
- forbidden failure patterns appear

## Step 5: Make it a standing gate

Use this tranche before:

- FX proving-ground reruns
- local-vol proving-ground reruns
- major market-data or task-runtime changes

Minimum cadence:
- whenever mock connector or task-runtime market wiring changes
- before starting a proving-ground implementation tranche

## Step 6: Feed the results back

For every stress-batch run:

- classify each failure into:
  - connector regression
  - route/substrate blocker
  - provider noise
  - codegen/validation issue
- update the roadmap note with:
  - current phase
  - blocked/unblocked proving grounds
  - next highest-value task family

## Validation

Deterministic:

```bash
/Users/steveyang/miniforge3/bin/python3 -m pytest \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_stress_task_preflight.py \
  /Users/steveyang/Projects/steveya/trellis/tests/test_agent/test_evals.py -q
```

Live:

- run `E21`-`E28`
- inspect batch summary and per-task latest run records

## Exit Criteria

- the tranche gives a stable answer to “is the mock connector path healthy?”
- proving-ground work can assume the connector baseline is good unless this gate
  says otherwise
