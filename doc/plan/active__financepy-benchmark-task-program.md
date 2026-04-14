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

## Market Scenario Foundation

`QUA-837` is now implemented as the market-state follow-on to the benchmark cutover.
The benchmark/task path no longer depends on an ad hoc benchmark-market overlay as
its primary abstraction. Instead it uses:

- schema-v2 scenario contracts in `MARKET_SCENARIOS.yaml`
- normalized scenario loading in `trellis/agent/market_scenarios.py`
- first-class flat/textbook constructors for single-asset equity, FX, rates, credit,
  and request-only negative tasks
- carry-aware translation from benchmark contract/scenario inputs into runtime market
  state and FinancePy-compatible reference inputs
- multi-asset underlier spot/vol/carry/correlation support for basket and rainbow tasks
- scenario digests persisted into task materialization and append-only run history
- a dedicated coverage audit script for benchmark, extension, negative, and canary corpora

The remaining market-state work is no longer “create the scenario foundation.” It is
scenario expansion: adding richer named benchmark markets and deeper coverage across
the growing parity and extension task sets.
