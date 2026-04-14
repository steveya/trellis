# FinancePy Benchmark Task Program

This plan replaces the old mixed `TASKS.yaml` inventory with explicit benchmark, extension, negative, and proof corpora.

## Objective

Build a task program that:

- benchmarks Trellis against FinancePy on explicit `(instrument, method)` parity tasks
- benchmarks Trellis-only nearby extensions that go beyond FinancePy
- benchmarks clarification / honest-block behavior on intentionally vague or unsupported requests
- rebuilds the canary suite around those corpora
- persists immutable timestamped run history for every rerun so progress can be tracked across library and knowledge upgrades

## Corpora

- `TASKS_BENCHMARK_FINANCEPY.yaml`
- `TASKS_EXTENSION.yaml`
- `TASKS_NEGATIVE.yaml`
- `TASKS_MARKET_CONSTRUCTION.yaml`
- `TASKS_PROOF_LEGACY.yaml`
- `MARKET_SCENARIOS.yaml`
- `FINANCEPY_BINDINGS.yaml`
- `CANARY_TASKS.yaml`

## Timestamped Run-History Contract

Every benchmark runner must persist append-only records with at least:

- `run_id`
- `task_id`
- `task_corpus`
- `task_definition_version`
- `task_definition_manifest`
- `market_scenario_id`
- `run_started_at`
- `run_completed_at`
- `execution_mode`
- `git_sha`
- `knowledge_revision` when available
- `success_status`
- `price_outputs`
- `greek_outputs`
- `timing`
- `token_usage`
- `comparison_summary`

`latest` views are convenience pointers only. The append-only history is the system of record.

## Ticket Map

1. `QUA-827`: schema split and manifest cutover
2. `QUA-828`: market-scenario registry and FinancePy binding catalog
3. `QUA-829`: legacy-task audit and migration map
4. `QUA-830`: FinancePy parity tranche 1
5. `QUA-834`: timestamped benchmark persistence and scorecards
6. `QUA-832`: Trellis extension task corpus
7. `QUA-833`: negative clarification / honest-block corpus and runner
8. `QUA-835`: canary revamp
9. `QUA-831`: FinancePy parity tranche 2
10. `QUA-836`: tooling cutover and legacy `TASKS.yaml` deletion

## Current Implementation Surface

- `scripts/run_tasks.py` runs the active pricing corpora
- `scripts/run_financepy_benchmark.py` runs FinancePy parity tasks and persists timestamped history
- `scripts/run_negative_benchmark.py` runs clarification / honest-block tasks and persists timestamped history
- `scripts/run_canary.py` and `scripts/record_cassettes.py` use the rebuilt canary manifest
- `TASKS.yaml` is removed after the cutover; retained proof-only legacy tasks live in `TASKS_PROOF_LEGACY.yaml`

## Market Scenario Follow-On

The benchmark cutover surfaced a broader market-state gap: the mock resolver catalog
is still much thinner than the new benchmark and extension corpora assume, so many
parity tasks currently rely on a benchmark-market overlay rather than first-class
named scenario constructors. That gap is tracked as `QUA-837` with follow-on tickets
for benchmark-aligned scenario schema, flat/textbook constructors, carry unification,
multi-asset support, and coverage audit. The overlay path is acceptable as the
bridge for `QUA-826`, but it is not the intended long-term market-state surface.
