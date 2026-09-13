# Task Manifest Integrity Program

Linear umbrella: `QUA-1241` — self-contained pricing requests.

## Objective

Make every pricing-task outcome traceable to an authored product, market,
method, and acceptance contract. Incomplete or unsupported requests must ask
for clarification, declare a governed non-pricing disposition, or fail with an
exact capability blocker before Trellis can synthesize economic inputs.

## Historical audit baseline

- 168 request rows were inspected across the pricing and negative corpora.
- 124 of 131 retained legacy proof rows are not self-contained after the
  authored P005, P006, and T102 repairs.
- At the audit, the runtime contained title/id-derived economic bootstraps and a global
  5% comparison fallback.
- P004 could return a value for a non-callable collar after dropping
  callable and irregular-schedule semantics.
- T09 could replace a step-up coupon schedule with one flat 5% coupon.

These are historical findings, not current capability claims. Completed repairs
below now defend P004 and T09 with truthful non-pricing outcomes. The checked
legacy baseline remains an incompleteness inventory, never permission to price.

## Ordered delivery queue

| Order | Linear | Status | Outcome |
| ---: | --- | --- | --- |
| 1 | `QUA-1242` | Done | Fail-closed corpus validation and checked legacy-debt baseline |
| 2 | `QUA-1243` | Done | P004 preserves callable-collar semantics or blocks honestly |
| 3 | `QUA-1248` | Done | T09 cannot price through the flat-coupon bootstrap |
| 4 | `QUA-1249` | Done | T30/T96 get authored market and acceptance contracts |
| 5 | `QUA-1250` | Done | T03/T83/T85 receive current dispositions |
| 6 | `QUA-1244` | Done | P003 monitoring and numerical controls are authored |
| 7 | `QUA-1253` | Done | Reusable convention-aware dual-curve Bermudan swap tails |
| 8 | `QUA-1245` | Done | P005 conventions/model inputs are authored |
| 9 | `QUA-1246` | Done | P006 bounded terminal-protection semantics are authored |
| 10 | `QUA-1247` | In Progress | Six authored pricing/analytics contracts and an exact T82 hold replace seven title-derived bootstraps |
| 10.1 | `QUA-1255` | Done | T02/T17 receive authored callable pricing fixtures |
| 10.2 | `QUA-1260` | Done | Date-preserving parallel curve shifts support the authored callable fixture |
| 10.3 | `QUA-1258` | In Progress | T89 receives an authored same-payoff, constant-zero-OAS duration comparison; prerequisites are Done |
| 10.4 | `QUA-1262` | In Progress | T82 receives an exact proof hold listing missing analytics inputs, with zero execution attempts |
| 10.5 | `QUA-1254` | In Progress | T73 receives an authored payer swap-NPV exercise-value proof, without settlement-lifecycle claims |
| 10.6 | `QUA-1256` | In Progress | E22 receives an authored cap-strip contract with independent lognormal forward-marginal MC |
| 10.7 | `QUA-1257` | Done | T102 receives an authored terminal-basket contract |
| 10.8 | `QUA-1259` | Backlog | Remove obsolete pricing bootstraps; blocked by QUA-1254/1256/1258/1262 |
| 11 | `QUA-1261` | Backlog | Preserve authored market state in generic OAS solving; separate from T89's zero-OAS identity |
| 12 | `QUA-1263` | Backlog | Report exercised pricing seeds truthfully for the remaining authored stochastic tasks |
| 13 | `QUA-1264` | Backlog | Shared comparisons reject invalid numeric/status/unit evidence beyond T89's bounded profile |
| 14 | `QUA-1265` | Backlog | Governed holds cannot bypass admission through removed or spoofed provenance |
| 15 | `QUA-1251` | Backlog | Reusable variable-coupon callable-bond primitive; QUA-1248 prerequisite is Done |
| 16 | `QUA-1252` | Backlog | Remove implicit 5% comparison tolerance after QUA-1247 closes |

Linear is the source of truth; table states were reconciled on 2026-09-13.
Completed upstream tickets may remain in Linear's dependency history and are
not active blockers. Parallel work must use isolated branches/worktrees; merged
manifest changes require a freshly reviewed combined baseline fingerprint.

`QUA-1154` retains a historical dependency on the completed `QUA-1243`; audit
its current P004 disposition before closeout, because the earlier green result
is not economic-equivalence evidence. `QUA-1146` must be audited before more work: the
T18 honest-block behavior is already present, so only genuinely remaining
route-specific bridge work should stay open.

## Current closeout boundary

`QUA-1260` merged in [PR #895](https://github.com/steveya/trellis/pull/895)
as `f779dc09e`. Its local PR gate passed 6,131 core and 32 tier-2 tests, with
15 optional dependency skips; required GitHub checks and reviews passed.
The dated curve retains its conventions under parallel flat-rate shifts; this
does not add bucket shocks, calibration, callable autodiff Greeks, or an OAS
solver repair. Its user and quant documentation shipped with the implementation.

`QUA-1254` and `QUA-1256` remain open while their PR reviews and final gates
complete. Black76 in T73 is tree-normalized, not an independent volatility
oracle; E22's comparator samples independent forward marginals, not a
Hull-White short-rate path. T82's planned outcome is a non-executing hold rather
than an invented analytics fixture. T89's duration comparison is a same-model identity, not
independent numerical accuracy evidence.

`QUA-1247` closes only after six exact authored pricing/analytics replays,
the T82 zero-attempt hold, the preserved T09 block, bootstrap reachability
cleanup, and the release gate pass. Its merged-state cleanup/refactoring and
official quant/developer/user documentation maintenance are required closeout
work. Generic OAS market-state repair (`QUA-1261`) and remaining seed provenance
repair (`QUA-1263`) stay independently tracked under `QUA-1241`.
Review also exposed shared comparison-output validation (`QUA-1264`) and
mutable-provenance admission (`QUA-1265`) gaps. The bounded T89 output checks
and T82/T89 reserved-ID guards do not close those general follow-ons.

## QUA-1242 implementation contract

1. Add immutable issue/report values and corpus-specific validation.
2. Validate task IDs and market/binding references before loading tasks.
3. Keep modern corpora strict and freeze legacy field-level debt by exact
   count/digest identity plus a normalized full-task fingerprint.
4. Add a standalone audit command, Make/CI gate, focused tests, docs, and a
   visible limitation.
5. Run the PR and release gates before closeout.

The legacy baseline is deliberately not a waiver. Any task-content edit,
improvement, or regression changes the issue digest or task fingerprint and
requires a reviewed baseline update in the ticket that changes the manifest
contract. The main runner and specific-id rerunner independently reject an
incomplete selected legacy row before constructing the default market.
