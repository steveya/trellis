# Binding-First Exotic Proof Closeout

## Purpose

This document closes the first proof pass for the binding-first exotic
assembly program.

It is not another implementation ticket. It is the measured capability
statement for the current binding-first runtime, grounded in the checked proof
artifacts and the follow-on tickets opened from them.

The machine-readable summary lives in:

- `docs/benchmarks/binding_first_exotic_proof_closeout.json`
- `docs/benchmarks/binding_first_exotic_proof_closeout.md`

## Evidence Basis

The closeout summary is generated from the reviewed proof bundles produced by:

- `QUA-808` event/control/schedule cohort
- `QUA-809` basket/credit/loss cohort

Those bundles were generated on consecutive 2026-04-13 binding-first proof
runs with fresh builds. `QUA-809` only changed proof docs and follow-on
tracking relative to the underlying runtime evidence, so the combined closeout
still reflects one coherent support-contract surface.

## Measured Outcome

Program totals from `docs/benchmarks/binding_first_exotic_proof_closeout.json`:

- `11` proof tasks
- `1` task passed the program gate
- `10` tasks failed the program gate
- `10` tasks were expected to be `proved`
- `1` task was expected to be an `honest_block`
- `0` honest-block sentinels were certified by the gate in the initial closeout run
- first-pass success rate: `1 / 11` (`9.1%`)
- total elapsed time: `1143.1s`
- total token usage: `883,606`
- route-identity telemetry was later normalized in `QUA-821`; the residual
  proof blockers are now the task-specific gaps listed below rather than
  synthetic `unknown` route ids

The only proved task in the current closeout is:

- `T105` quanto option: analytical plus Monte Carlo parity with stable binding
  ids and no route-card rescue logic

## Task-Level Outcome Map

| Task | Result | Residual gap | Follow-on |
| --- | --- | --- | --- |
| `T105` | proved | none in this closeout slice | none |
| `T17` | follow-on recovered | title-only callable-bond proof tasks now bootstrap a canonical Bermudan issuer-call schedule and bind the PDE lane to `price_callable_bond_pde(...)` exactly | none |
| `T73` | failed gate | analytical/tree/MC parity drift | `QUA-818` |
| `E22` | failed gate | cap/floor fresh-build instability and missing reference-target evidence | `QUA-819` |
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
- the runtime can support at least one fully binding-first exotic-style parity
  request (`T105`)
- failure surfaces are now concrete enough to split into task-backed
  implementation tickets instead of vague “route is weird” cleanup work

## What This Does Not Prove

This closeout does **not** justify claiming broad support for arbitrary
constructable exotic derivatives.

The current measured state is:

- event/control/schedule proof is still partial because `T73` and `E22` remain open
- basket/credit/loss proof has been recovered across the current cohort
- the honest-block sentinel path was initially uncertified, then recovered by `QUA-820`
- residual `unknown` route telemetry was removed by `QUA-821`

So the architecture migration is meaningfully ahead of the capability proof.
That is acceptable for this closeout, but it must remain explicit.

## Support Contract Decision

The correct support statement after this closeout is:

- Trellis now ships a binding-first runtime architecture for exotic assembly.
- Trellis does **not** yet have proof-level evidence for general constructable
  exotic support across the agreed cohort.
- Current proof-level support is limited to the recovered slices already
  measured in the checked benchmark artifact plus the post-closeout recoveries
  (`T17`, `E27`, `T49`, `T50`, `E26`, `T53`, `T102`, `T126`), with the remaining
  gaps tracked by `QUA-818` and `QUA-819`.

This is why `LIMITATIONS.md` now records the exotic proof cohort as an open
limitation instead of letting the architecture docs imply the proof is already
complete.

## Closeout Decision

`QUA-815` should be treated as complete once:

- the checked closeout artifacts are committed
- the measured support contract is reflected in docs and `LIMITATIONS.md`
- the residual proof gaps remain as concrete follow-on tickets

The closeout ticket ends the measurement tranche. It does not imply the
follow-ons are done.
