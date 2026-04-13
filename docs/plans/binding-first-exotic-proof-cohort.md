# Binding-First Exotic Proof Cohort

## Purpose

This document defines the benchmark cohort for the binding-first exotic
assembly program.

The cohort exists so later tickets can prove concrete capability rather than
making architectural claims in the abstract. Every proof slice under
`QUA-808`, `QUA-809`, and `QUA-815` should use this cohort unless a later
ticket explicitly amends it.

## Design Rules

- Use existing tasks from `TASKS.yaml`; do not invent synthetic benchmark ids
  for the proof program.
- Prefer tasks that exercise composition pressure:
  - event structure
  - control style
  - schedule dependence
  - basket / dependence structure
  - credit / loss-distribution structure
  - cross-currency or multi-factor market binding
- Include at least one honest-block sentinel so the program proves that the
  runtime can refuse unsupported structures cleanly instead of forcing them
  into a legacy route bucket.

## Outcome Taxonomy

Each cohort item must be evaluated against one of these expected outcomes.

### Proved

Use `proved` when the task reaches a valid compare-ready or price-ready result
on the binding-first runtime and the run artifact shows:

- a stable binding identity for each exact helper-backed lane
- no new route ids created to make the task succeed
- no route-card-only notes or prose acting as the critical source of helper or
  kernel selection

### Honest block

Use `honest_block` when the runtime rejects the task through typed
family/binding/primitive blockers that explain why the structure is outside the
current constructable surface.

This is an acceptable result only for tasks explicitly marked as honest-block
sentinels in this cohort.

### Out of scope

Use `out_of_scope` only when the task goes beyond the current program target
even if the runtime blocks honestly. This category should be rare. The default
for unsupported-but-intentional sentinels in this cohort is `honest_block`,
not `out_of_scope`.

## Measurement Protocol

All proof slices must use the same measurement protocol.

1. Fix the repo revision for the whole cohort pass.
2. Run fresh builds for proof tasks; do not rely on stale generated adapters as
   hidden support.
3. Record, for each task and each method lane:
   - selected binding ids
   - comparison or pricing outcome
   - `first_pass`
   - `attempts_to_success`
   - retry taxonomy
   - elapsed time
   - token usage
4. When a task is expected to block honestly, record the blocker taxonomy and
   show that it is family/binding/primitive-first rather than route-first.
5. No ticket may claim proof success if it needed a new route id or route-card
   prose update to make the benchmark pass.

## Cohort

### Event / Control / Schedule Cohort

These tasks define the `QUA-808` proof surface.

| Task | Expected outcome | Binding-first capability under test | Notes |
| --- | --- | --- | --- |
| `T17` Callable bond: HW rate PDE (PSOR) vs HW tree | `proved` | same-day event schedule plus issuer control across PDE and lattice bindings | Tests event transforms, fixed-income schedule semantics, and explicit issuer control |
| `T73` European swaption: Black76 vs HW tree vs HW MC | `proved` | rate-style schedule semantics and method-spanning helper bindings | Good check that schedule-aware analytical, lattice, and MC bindings share one semantic contract |
| `E22` Cap/floor: Black caplet stack vs MC rate simulation | `proved` | rate cap/floor strip decomposition without route-local rescue logic | Confirms typed family lowering plus rate MC/analytical bindings |
| `T105` Quanto option: quanto-adjusted BS vs MC cross-currency | `proved` | cross-currency market binding plus analytical/MC parity | Keeps the proof program honest about multi-currency exact bindings |
| `E27` American Asian barrier under Heston: PDE vs MC vs FFT should block honestly | `honest_block` | honest refusal for unsupported multi-control, path-dependent hybrid structure | Sentinel that the runtime should not force a fake route fit |

### Basket / Credit / Loss Cohort

These tasks define the `QUA-809` proof surface.

This sub-cohort intentionally does not add a second honest-block sentinel.
The program-level sentinel remains `E27` in the event/control/schedule cohort.
`E26` stays `proved` here on purpose even though
`tests/evals/stress_tasks.yaml` currently classifies it as `honest_block` on
the older stress surface. In the binding-first proof program, `E26` is the
constructive basket-credit stress target that `QUA-823` is expected to recover,
not a permanent block sentinel.

| Task | Expected outcome | Binding-first capability under test | Notes |
| --- | --- | --- | --- |
| `T49` CDO tranche: Gaussian vs Student-t copula | `proved` | tranche attachment/detachment and copula-backed loss bindings | Baseline tranche/loss distribution proof |
| `T50` Nth-to-default: MC correlated defaults vs semi-analytical | `proved` | default-time basket-credit structure across MC and analytical lanes | Tests basket-credit composition directly |
| `T53` Multi-name portfolio loss distribution: recursive vs FFT vs MC | `proved` | loss-distribution composition across recursive, transform, and MC bindings | Core proof that loss aggregation is binding-role-driven rather than route-driven |
| `E26` Nth-to-default basket: Gaussian copula vs default-time MC | `proved` | basket-credit hybrid under the stress tranche | Stress follow-on for `T50` |
| `T102` Rainbow option (best-of-two): Stulz formula vs MC | `proved` | multi-underlier basket state and dependence-aware exact binding | Keeps non-credit basket composition in scope |
| `T126` Spread option (Kirk approximation) vs 2D MC vs 2D FFT | `proved` | multi-factor transform and MC parity on a non-credit hybrid payoff | Checks binding-first assembly across analytical, transform, and MC lanes |

## Acceptance Rules For Later Tickets

### `QUA-808`

`QUA-808` is done only when:

- the `QUA-808` cohort has been run on a fixed repo revision with fresh builds
- the run artifacts show binding ids for the successful exact-helper lanes
- any residual failures are either fixed in the same slice or split into
  task-backed follow-on tickets with concrete evidence
- no new route ids were introduced to make the cohort pass

### `QUA-809`

`QUA-809` is done only when:

- the `QUA-809` cohort has been run on a fixed repo revision with fresh builds
- the run artifacts show binding ids for each exact helper-backed lane and the
  basket-credit/loss-distribution surfaces no longer depend on route-local
  exact-lookup authority
- any residual failures are either fixed in the same slice or split into
  task-backed follow-on tickets with concrete evidence

### `QUA-815`

`QUA-815` is the closeout ticket for the whole cohort. It should summarize:

- per-task outcome
- pass vs honest-block rate
- first-pass and retry metrics
- time and token cost
- any remaining blocker families that still prevent the claimed end state

The checked closeout summary lives in:

- `docs/plans/binding-first-exotic-proof-closeout.md`
- `docs/benchmarks/binding_first_exotic_proof_closeout.json`
- `docs/benchmarks/binding_first_exotic_proof_closeout.md`

## Latest Evidence

### `QUA-808` live run on `2026-04-13`

Command:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/run_binding_first_exotic_proof.py \
  --cohort event_control_schedule \
  --output /tmp/qua808_results_live.json \
  --report-json /tmp/qua808_report_live.json \
  --report-md /tmp/qua808_report_live.md
```

Outcome summary:

- `T105` reached `proved`
- `T17` failed because the callable-bond PDE lane still has no exact binding or
  constructive steps
- `T73` failed on analytical/tree/MC parity drift
- `E22` failed on fresh-build analytical and MC instability plus missing
  reference-target evidence
- `E27` refused as expected, but the structured result did not persist typed
  blocker categories, so the proof gate could not certify the `honest_block`

Follow-on tickets opened from that run:

- `QUA-817` callable-bond PDE proof gap (`T17`)
- `QUA-818` swaption parity drift (`T73`)
- `QUA-819` cap/floor fresh-build stability and reference-target evidence (`E22`)
- `QUA-820` structured blocker persistence for the honest-block sentinel (`E27`)
- `QUA-821` residual `unknown` route ids in proof telemetry (`T17`, `E27`)

### `QUA-809` live run on `2026-04-13`

Command:

```bash
/Users/steveyang/miniforge3/bin/python3 scripts/run_binding_first_exotic_proof.py \
  --cohort basket_credit_loss \
  --output /tmp/qua809_results_live.json \
  --report-json /tmp/qua809_report_live.json \
  --report-md /tmp/qua809_report_live.md
```

Outcome summary:

- no `QUA-809` cohort item reached `proved`
- `T49` failed on the Student-t tranche lane rebuilding copula plumbing instead
  of staying on the exact tranche helper contract
- `T50` and `E26` failed around nth-to-default helper invocation and
  basket-credit market parsing
- `T53` failed across recursive, FFT, and MC loss-distribution constructions
- `T102` and `T126` failed on multi-underlier basket market parsing, and `T126`
  also failed its FFT spread lane

Follow-on tickets opened from that run:

- `QUA-822` copula tranche exact-helper contract (`T49`)
- `QUA-823` nth-to-default helper and basket-credit parsing (`T50`, `E26`)
- `QUA-824` loss-distribution recursive/FFT/MC constructive stability (`T53`)
- `QUA-825` multi-underlier basket parsing and FFT spread stability (`T102`, `T126`)
- `QUA-821` broadened to cover residual `unknown` route ids on `T50` and `E26`

## Review Notes

- `T01`, `T13`, `T38`, `T94`, `T108`, and `E25` remain valuable regression
  gates for the binding runtime, but they are not the exotic proof cohort.
  They are support controls, not the program’s primary exotic evidence.
- The cohort intentionally mixes baseline production tasks (`T*`) and stress
  tasks (`E*`). The proof program should show both real throughput and clean
  failure boundaries.
