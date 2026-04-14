# Binding-First Exotic Proof Closeout

## Purpose

This document closes the first proof pass for the binding-first exotic
assembly program.

It is not another implementation ticket. It is the measured capability
statement for the current binding-first runtime, grounded in the checked proof
artifacts and the fixed-revision cohort reruns after the follow-ons landed.

The machine-readable summary lives in:

- `docs/benchmarks/binding_first_exotic_proof_closeout.json`
- `docs/benchmarks/binding_first_exotic_proof_closeout.md`

## Evidence Basis

The checked closeout summary is generated from the reviewed proof bundles
produced by:

- `QUA-808` event/control/schedule cohort rerun on the final fixed revision
- `QUA-809` basket/credit/loss cohort rerun on the final fixed revision

Those bundles were generated on fresh-build proof runs on 2026-04-13 after the
proof follow-ons had landed. The combined closeout therefore reflects one
coherent final support-contract surface rather than the earlier pre-follow-on
snapshot.

## Measured Outcome

Program totals from `docs/benchmarks/binding_first_exotic_proof_closeout.json`:

- `11` proof tasks
- `11` tasks passed the program gate
- `0` tasks failed the program gate
- `10` tasks were expected to be `proved`
- `1` task was expected to be an `honest_block`
- `1` honest-block sentinel was certified by the gate
- first-pass success rate: `10 / 11` (`90.9%`)
- total elapsed time: `304.6s`
- total token usage: `142,551`
- route-identity telemetry no longer carries residual synthetic `unknown`
  placeholders in the checked cohort reports

## Task-Level Outcome Map

| Task | Result | Residual gap | Follow-on |
| --- | --- | --- | --- |
| `T105` | proved | none in this closeout slice | none |
| `T17` | follow-on recovered | title-only callable-bond proof tasks now bootstrap a canonical Bermudan issuer-call schedule and bind the PDE lane to `price_callable_bond_pde(...)` exactly | none |
| `T73` | follow-on recovered | title-only swaption proof tasks now bootstrap a semantic rate-style swaption contract with a shared Hull-White comparison regime across Black76, tree, and MC lanes | none |
| `E22` | follow-on recovered | rate cap/floor proof tasks now bind both analytical and Monte Carlo lanes to exact strip helpers and preserve the expected reference-target evidence under fresh-build retries | none |
| `E27` | follow-on recovered | honest-block sentinel is now certified after structured blocker persistence landed | none |
| `T49` | follow-on recovered | Gaussian and Student-t tranche lanes now stay on `price_credit_basket_tranche(...)` even on fresh builds | none |
| `T50` | follow-on recovered | nth-to-default helper path now proves on the exact helper surface without stale schedule-builder glue | none |
| `E26` | follow-on recovered | nth-to-default basket ingress now resolves to the credit-basket family instead of generic basket parsing | none |
| `T53` | follow-on recovered | recursive / FFT / MC lanes now bind to typed loss-distribution helpers and prove on exact backend surfaces | none |
| `T102` | follow-on recovered | multi-underlier market parsing now stays on typed basket helper surfaces | none |
| `T126` | follow-on recovered | multi-underlier spread parsing and FFT stability now prove on the typed basket helper path | none |

## What This Proves

The proof program did establish several important facts:

- route is no longer required as the primary semantic authority for the proof
  cohort
- binding ids survive into live proof telemetry
- the runtime can support the full agreed binding-first exotic proof cohort on
  a fixed revision without inventing new route identities
- failure surfaces are now concrete enough to split into task-backed
  implementation tickets instead of vague “route is weird” cleanup work, and
  the resolved follow-ons now stay green under the checked cohort reruns

## What This Does Not Prove

This closeout does **not** justify claiming broad support for arbitrary
constructable exotic derivatives.

The current measured state is:

- both proof cohorts now pass their full fixed-revision gate
- all `10` proved tasks in the agreed cohort now finish compare-ready or
  price-ready
- the `E27` honest-block sentinel is now certified from structured blocker data
- residual `unknown` route telemetry has been removed from the checked proof
  reports
- the checked benchmark artifact now records `11/11` gate passes, `1/1`
  certified honest blocks, `304.6s` total elapsed, and `142551` total tokens

So the architecture migration is now backed by a checked cohort-level proof
pass. That is materially stronger than the original closeout snapshot, but the
proof surface is still bounded to the agreed cohort and should be described
that way.

## Support Contract Decision

The correct support statement after this closeout is:

- Trellis now ships a binding-first runtime architecture for exotic assembly.
- Trellis now has checked proof-level evidence across the agreed `11`-task
  binding-first exotic proof cohort (`T17`, `T73`, `T105`, `E22`, `E27`,
  `T49`, `T50`, `T53`, `E26`, `T102`, `T126`).
- Trellis still does **not** have support-contract-correct evidence for
  arbitrary constructable exotics beyond that bounded proof cohort.

This is why `LIMITATIONS.md` now records the exotic proof cohort as an open
limitation in bounded-scope form instead of letting the architecture docs imply
that the checked cohort proves the entire arbitrary-exotic end state.

## Closeout Decision

`QUA-815` should be treated as complete once:

- the checked closeout artifacts are committed
- the measured support contract is reflected in docs and `LIMITATIONS.md`
- the checked cohort reruns and support-contract updates are committed

The closeout ticket ends the first measured proof tranche. It does not imply
that the broader arbitrary-constructable-exotic end state is already complete.
