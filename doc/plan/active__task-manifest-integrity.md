# Task Manifest Integrity Program

Linear umbrella: `QUA-1241` — self-contained pricing requests.

## Objective

Make every pricing-task outcome traceable to an authored product, market,
method, and acceptance contract. Incomplete or unsupported requests must ask
for clarification, declare a governed non-pricing disposition, or fail with an
exact capability blocker before Trellis can synthesize economic inputs.

## Audit baseline

- 168 request rows were inspected across the pricing and negative corpora.
- 126 of 131 retained legacy proof rows are not self-contained.
- The runtime still contains title/id-derived economic bootstraps and a global
  5% comparison fallback.
- P004 can currently return a value for a non-callable collar after dropping
  callable and irregular-schedule semantics.
- T09 can currently replace a step-up coupon schedule with one flat 5% coupon.

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
| 10 | `QUA-1247` | Blocked | Seven legacy title-derived bootstraps become named contracts |
| 11 | `QUA-1251` | Blocked | Reusable variable-coupon callable-bond primitive |
| 12 | `QUA-1252` | Blocked | Remove the implicit 5% comparison tolerance |

`QUA-1154` is hard-blocked by `QUA-1243`; its current P004 green result is not
economic-equivalence evidence. `QUA-1146` must be audited before more work: the
T18 honest-block behavior is already present, so only genuinely remaining
route-specific bridge work should stay open.

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
