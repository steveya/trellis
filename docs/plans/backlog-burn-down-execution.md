# Backlog Burn-Down Execution Program

## Objective

Clear the remaining actionable backlog before starting new project work, while
keeping longer-range expansion epics out of the burn-down queue.

This program turns the audited backlog into a strict execution order with
reviewable delivery slices, validation gates, and closeout rules that match
`AGENTS.md`.

Status mirror last synced: `2026-04-10`

Current status:

- `QUA-458` is complete.
- `QUA-710` is complete.
- `QUA-428` is complete.
- `QUA-430` is complete.
- The next actionable ticket in queue order is `QUA-700`.

## Operating Rules

- Linear is the source of truth for ticket state and scope changes.
- Work one implementation ticket per branch and PR unless the user explicitly
  asks to bundle tickets.
- Follow the ticket workflow in `AGENTS.md` for every delivery slice:
  upstream review, on-screen ticket announcement, TDD, doc update, Linear
  handoff, plan sync, self-review, commit.
- Update `LIMITATIONS.md` whenever a ticket changes the supported surface or
  closes a documented limitation.
- Close umbrella tickets only after their child delivery slices, docs, and
  validation gates are complete.

## In Scope

These tickets are actionable backlog burn-down work:

1. `QUA-458` canary replay: full-task cassette path with diagnosis parity
2. `QUA-710` canary telemetry: trustworthy latency, token, and attempt baselines
3. `QUA-428` stale test triage process and tooling
4. `QUA-430` local gate and release-gate configuration
5. `QUA-700` canary suite closeout after replay / telemetry / gate hardening
6. `QUA-543` stress tranche: remaining `E22` compare-ready regression
7. `QUA-417` route registry cleanup for smaller builder obligation surfaces
8. `QUA-544` proving runs: first-pass knowledge-light reliability
9. `QUA-429` lesson-to-test pipeline base slice
10. `QUA-447` semantic template follow-on for the lesson-to-test path
11. `QUA-545` maintenance tail for residual cleanup that does not justify a
    broader project

## Out of Scope

These remain future-project backlog, not burn-down work:

- `QUA-349`
- `QUA-368`
- `QUA-365`
- `QUA-363`
- `QUA-364`
- `QUA-593` and children `QUA-615` to `QUA-618`
- `QUA-594` and children `QUA-619` to `QUA-621`, `QUA-638` to `QUA-642`
- `QUA-276` roadmap index

If these should move into near-term delivery, they need their own plan and
priority reset rather than being silently treated as backlog cleanup.

## Ordered Waves

### Wave 1: Replay, telemetry, and gate closeout

This wave restores a low-cost, trustworthy canary surface before any new
feature work.

| Order | Ticket | Expected artifact |
| --- | --- | --- |
| 1 | `QUA-458` | Full-task canary replay from cassettes with diagnosis parity |
| 2 | `QUA-710` | Trustworthy canary history for tokens, attempts, and elapsed time |
| 3 | `QUA-428` | Current stale-test inventory and triage flow |
| 4 | `QUA-430` | Stable local gate commands and release guidance |
| 5 | `QUA-700` | Canary umbrella closeout with docs and final rerun evidence |

Wave 1 closeout gate:

- replay path exists and is documented
- canary telemetry is trustworthy enough to compare runs over time
- gate commands and release posture are documented
- `QUA-700` is closed only after the above slices are done

### Wave 2: Remaining correctness and route-surface cleanup

This wave removes the highest-value remaining correctness and builder-surface
debt once canary iteration is cheap again.

| Order | Ticket | Expected artifact |
| --- | --- | --- |
| 6 | `QUA-543` | `E22` stress path restored to a compare-ready state |
| 7 | `QUA-417` | Narrower route-registry / builder obligation surface |

Wave 2 closeout gate:

- the remaining stress-path regression is resolved or split cleanly
- route-surface cleanup lands with the relevant regional validation

### Wave 3: Reliability and durable learning loop

This wave focuses on the first-pass builder quality and the path from lessons
to durable regression coverage.

| Order | Ticket | Expected artifact |
| --- | --- | --- |
| 8 | `QUA-544` | Improved first-pass knowledge-light proving reliability |
| 9 | `QUA-429` | Base lesson-to-test pipeline |
| 10 | `QUA-447` | Semantic template follow-on on top of the base pipeline |

Wave 3 closeout gate:

- first-pass proving reliability is materially better and measured
- the lesson-to-test path exists end-to-end
- semantic follow-on scope stays additive rather than reopening the base slice

### Wave 4: Maintenance tail

| Order | Ticket | Expected artifact |
| --- | --- | --- |
| 11 | `QUA-545` | Residual maintenance cleanup with explicit scope boundary |

## Ticket Selection Rule

Always pick the earliest ticket in the ordered queue whose Linear status is not
`Done`, unless the user explicitly redirects work.

If a ticket has become stale because other landed work already satisfies it:

1. audit the ticket against the current repo
2. update or close it in Linear
3. sync the relevant plan mirror
4. continue to the next actionable ticket without creating duplicate work

## Current Start Point

Execution started with `QUA-458`, continued through `QUA-710`, `QUA-428`, and
`QUA-430`, and now moves next to `QUA-700`.

Plain-English goal:

- close the canary umbrella now that replay, telemetry, hygiene, and gate
  entrypoints are all landed

Primary files and surfaces:

- `docs/plans/canary-suite-stabilization.md`
- the current canary rerun evidence and gate command docs
- any final docs or Linear cleanup needed to turn `QUA-700` from backlog into completed umbrella maintenance
