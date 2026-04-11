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
- `QUA-700` is complete.
- `QUA-543` is complete.
- `QUA-544` is complete with a measured `3/3` proving-success bar, `2/3`
  task-level first-pass success, and the residual `KL01` semantic-validation
  retry split into `QUA-779`.
- Wave 1 is complete.
- Wave 2 is complete.
- `QUA-417` has been rewritten as a low-priority route-registry maintenance
  cleanup rather than a core burn-down tranche.
- The next actionable ticket in queue order is `QUA-429`.

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
7. `QUA-544` proving runs: first-pass knowledge-light reliability
8. `QUA-429` lesson-to-test pipeline base slice
9. `QUA-447` semantic template follow-on for the lesson-to-test path
10. `QUA-545` maintenance tail for residual cleanup that does not justify a
    broader project
11. `QUA-417` route registry cleanup: collapse redundant optional utility
    bindings

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

### Wave 2: Remaining correctness

This wave removed the highest-value remaining correctness debt once canary
iteration was cheap again.

| Order | Ticket | Expected artifact |
| --- | --- | --- |
| 6 | `QUA-543` | `E22` stress path restored to a compare-ready state |

Wave 2 closeout gate:

- the remaining stress-path regression is resolved or split cleanly

### Wave 3: Reliability and durable learning loop

This wave focuses on the first-pass builder quality and the path from lessons
to durable regression coverage.

| Order | Ticket | Expected artifact |
| --- | --- | --- |
| 7 | `QUA-544` | Improved first-pass knowledge-light proving reliability |
| 8 | `QUA-429` | Base lesson-to-test pipeline |
| 9 | `QUA-447` | Semantic template follow-on on top of the base pipeline |

Wave 3 closeout gate:

- first-pass proving reliability is materially better and measured
- the lesson-to-test path exists end-to-end
- semantic follow-on scope stays additive rather than reopening the base slice

Current note:

- `QUA-544` delivered the proving summary and route/packet hardening needed to
  move the tranche to `3/3` success; the remaining non-first-pass retry path
  is isolated in `QUA-779`

### Wave 4: Maintenance tail

| Order | Ticket | Expected artifact |
| --- | --- | --- |
| 10 | `QUA-545` | Residual maintenance cleanup with explicit scope boundary |
| 11 | `QUA-417` | Route-registry metadata compaction for optional utility bindings |

## Ticket Selection Rule

Always pick the earliest ticket in the ordered queue whose Linear status is not
`Done`, unless the user explicitly redirects work.

If a ticket has become stale because other landed work already satisfies it:

1. audit the ticket against the current repo
2. update or close it in Linear
3. sync the relevant plan mirror
4. continue to the next actionable ticket without creating duplicate work

## Current Start Point

Execution started with `QUA-458`, continued through `QUA-710`, `QUA-428`,
`QUA-430`, the `QUA-700` umbrella closeout, and `QUA-543`. `QUA-417` was
rewritten as maintenance cleanup. `QUA-544` then improved the knowledge-light
proving tranche to `3/3` success and split the remaining `KL01`
semantic-validation retry into `QUA-779`, so the active queue now moves next
to `QUA-429`.

Current architectural note:

- the live lesson path is `promotion.py -> store.py -> reflect.py`, with
  `test_resolution.py` and the resolved-failure hook in `executor.py` feeding
  the same promotion pipeline
- older references to `experience.py` or `cookbooks.py` as the core lesson
  surfaces are stale and should not drive ticket scope
- `QUA-429` is the generic deterministic materialization slice from validated
  or promoted lessons into regression payloads
- `QUA-447` remains the semantic-specific follow-on for lowering, validation,
  bridge, and route-boundary template families

Plain-English goal:

- build the base lesson-to-test pipeline now that the proving tranche is
  stable enough to turn repeat misses into durable regression coverage

Primary files and surfaces:

- `trellis/agent/knowledge/promotion.py`
- `trellis/agent/knowledge/store.py`
- `trellis/agent/knowledge/reflect.py`
- `trellis/agent/test_resolution.py`
- `trellis/agent/executor.py` for resolved-failure provenance into the lesson
  pipeline
- a new deterministic lesson-to-test materialization surface and its
  `tests/test_agent/` coverage
