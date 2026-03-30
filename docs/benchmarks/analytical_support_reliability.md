# Reliability Benchmark: `analytical_support_reliability`
- Created at: `2026-03-28T15:34:32.421877+00:00`
- Baseline mode: `reuse`
- Candidate mode: `fresh-build`
- Tasks: `T97`, `E22`, `E25`

## Notes
- Fresh-build benchmarking uses the same representative tranche as the reuse baseline.
- The report records lesson capture, cookbook enrichment, promotion candidates, and failure buckets.

## Outcome Summary
- Baseline success: `3/3`
- Candidate success: `0/3`
- Baseline avg attempts: `0.0`
- Candidate avg attempts: `7.0`

## Task Transitions
- Improved: `0`
- Regressed: `3`
- Unchanged: `0`

## Failure Buckets
- `comparison_insufficient_results`: `+3`

## Shared Knowledge
- Baseline tasks with shared context: `3`
- Candidate tasks with shared context: `3`
- Baseline tasks with lessons: `3`
- Candidate tasks with lessons: `3`

## Promotion Discipline
- Baseline successful tasks: `3`
- Candidate successful tasks: `0`
- Baseline successful tasks without reusable artifacts: `0`
- Candidate successful tasks without reusable artifacts: `0`

## Knowledge Capture
- Tasks with lessons: E22, E25, T97
- Tasks with cookbook enrichment: none
- Tasks with promotion candidates: none

## Follow-Up Suggestions
- `comparison_insufficient_results` (`3`): Check the comparison-task cross-validation wiring for the fresh-build path.

## Per-Task Details

### `T97` Digital (cash-or-nothing) option: BS formula vs MC vs COS
- Transition: `regressed`
- Baseline: reuse, success, attempts=0, lessons=9
- Candidate: fresh-build, fail, attempts=9, lessons=12

### `E22` Cap/floor: Black caplet stack vs MC rate simulation
- Transition: `regressed`
- Baseline: reuse, success, attempts=0, lessons=8
- Candidate: fresh-build, fail, attempts=6, lessons=10

### `E25` FX option (EURUSD): GK analytical vs MC
- Transition: `regressed`
- Baseline: reuse, success, attempts=0, lessons=9
- Candidate: fresh-build, fail, attempts=6, lessons=11
