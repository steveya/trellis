# Multi-Curve Rates Completion Plan

## Purpose

This document defines the remaining implementation queue for the Trellis
multi-curve rates workstream.

The goal is not to restart the rates foundation from scratch. The goal is to
make the remaining open multi-curve work agent-ready and stable enough to act
as the rates-side handoff for the bounded calibration-layer model-grammar
rollout.

## Why This Plan Exists

Trellis already landed a substantial multi-curve foundation:

- `QUA-354` curve contract and provenance
- `QUA-355` curve bootstrapping and source selection
- `QUA-357` regression and docs hardening
- `QUA-366` multi-curve cap/floor calibration
- `QUA-367` multi-curve swaption calibration
- `QUA-600` Hull-White term-structure fit
- `QUA-626` bootstrap convention bundle and market-instrument surface
- `QUA-627` Jacobian-aware bootstrap solve path

What is still missing is a single active queue for the remaining work. Right
now those follow-on slices are split across:

- a closed umbrella ticket (`QUA-352`)
- open rates-calibration cleanup (`QUA-381`)
- open market-parameter sourcing work (`QUA-358`, `QUA-362`)
- the broader market-data roadmap in `docs/market_data_workstream.md`

That is enough context for a human, but it is not a good ingestion target for a
local or cloud coding agent.

## Repo-Grounded Current State

### Foundation already present

The repo already has:

- named discount and forecast curves in the market snapshot path
- selected curve-name provenance on the runtime market state
- multi-curve cap/floor and swaption calibration helpers
- a reusable rates bootstrap instrument bundle and solver-ready bootstrap path
- official docs describing the discount/forecast split in user and quant docs

### Remaining gaps

The remaining open gaps that matter for the multi-curve handoff are:

- selected curve roles are not yet surfaced consistently enough through
  runtime/result/replay traces (`QUA-356`)
- the rates calibration helpers still need one stable shared term-builder and a
  single residual/tolerance policy (`QUA-381`)
- market-parameter sourcing still needs an explicit direct-vs-bootstrap source
  branch (`QUA-358`)
- traces and replay artifacts still need a stable market-parameter provenance
  schema (`QUA-362`)

## What This Plan Is Not

This plan does not attempt to:

- replace the shipped rates foundation with a new abstraction stack
- reopen already-completed multi-curve implementation slices without a tested
  bug
- absorb the bounded calibration-layer model-grammar epic
- own synthetic model-consistent mock snapshots for vol and credit
- broaden into live provider connectors or enterprise market-data adapters

## Design Guardrails

### 1. Preserve explicit discount-vs-forecast roles

No ticket in this queue is allowed to collapse back to a single generic "curve"
concept where the runtime or calibration surface actually needs both discount
and forecast roles.

### 2. Provenance must survive the full runtime path

For multi-curve work to be operationally useful, curve-role selection and
source-kind provenance must survive:

`MarketSnapshot -> MarketState -> runtime contract -> task result -> replay/debug artifact`

### 3. Rates calibration helpers must expose residual discipline explicitly

Cap/floor and swaption calibration already work. The remaining job is to make
their shared term vocabulary and residual/tolerance policy explicit so tiny
numeric drift does not look like a semantic failure.

### 4. Source kinds are part of the contract

The system should be able to say whether a rates input came from:

- a direct quote
- a bootstrap source
- a later synthetic/mock prior

That distinction belongs in provenance and trace surfaces, not in reviewer
memory.

## Ordered Delivery Queue

The recommended queue for a single multi-curve agent is:

1. `QUA-356` Rates foundation: trace propagation and cashflow curve selection
2. `QUA-381` Rates calibration: shared term builder and tolerance policy
3. `QUA-358` Market parameters: direct quotes and bootstrap sources
4. `QUA-362` Market parameters: provenance trace schema

This order is intentional:

- finish runtime visibility and curve-role traceability first
- harden the already-landed rates calibration helpers second
- make source-kind semantics explicit once the runtime/calibration boundary is
  stable
- finalize trace/report schema once the runtime and source-kind fields are
  settled

`QUA-358` and `QUA-381` can run in parallel if separate agents are assigned and
their file scopes are kept disjoint. For one agent, the queue above is the
recommended order.

## Critical Path

The critical path is:

`QUA-356 -> QUA-381 -> QUA-358 -> QUA-362`

This is the path that leaves the cleanest rates-side handoff into the bounded
calibration-layer model grammar.

## Parallel Workstream Contract

This plan is meant to run in parallel with
`docs/plans/calibration-layer-model-grammar-unification.md`.

Recommended two-agent split:

- Agent A owns this multi-curve queue.
- Agent B owns the calibration-layer model-grammar queue beginning at
  `QUA-687`.

Shared checkpoints:

- before rates-specific work in `QUA-689`, Agent B should read the latest
  closeout or progress note for `QUA-356`
- before finalizing rates registry entries or synthetic rates fixtures in
  `QUA-691` and `QUA-693`, Agent B should read the latest closeout or progress
  note for `QUA-381` and `QUA-358`
- before `QUA-692` closeout, both streams should reconcile replay and doc
  expectations

The calibration workstream does not need to wait for this entire queue to
finish before starting. It does need to consume the rates outcomes above before
claiming the rates-side boundary is complete.

## Agent Intake Bundle

Each local or cloud coding agent assigned to this workstream should start with:

- this plan doc
- `AGENTS.md`
- `docs/market_data_workstream.md`
- `docs/mathematical/calibration.rst`
- the current Linear ticket body plus any hard blockers
- the current code paths under `trellis/data/`, `trellis/core/market_state.py`,
  `trellis/models/calibration/`, and the relevant tests

## Linear Ticket Mirror

These tables mirror the current Linear tickets for this workstream and their
intended implementation order.

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This file is the repo-local mirror for subsequent agents.
- Implement the earliest ticket in the ordered queue below whose status is not
  `Done` and whose blockers are satisfied.
- Do not mark a row `Done` here before the corresponding Linear issue is
  actually closed.
- When a ticket changes behavior, APIs, runtime workflow, or operator
  expectations, update the relevant official docs in the same implementation
  closeout.

Status mirror last synced: `2026-04-06`

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-694` Rates platform: multi-curve completion and calibration handoff | Backlog |

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-356` | Trace propagation and cashflow curve selection | Done |
| `QUA-381` | Shared rates term builder and tolerance policy | In Progress |
| `QUA-358` | Direct quotes and bootstrap source kinds | Backlog |
| `QUA-362` | Market-parameter provenance trace schema | Backlog |

### Adjacent Shared Work

These are related but not owned by this queue:

| Ticket | Reason |
| --- | --- |
| `QUA-361` | Synthetic priors for mock/proving runs are broader than rates-only multi-curve work |
| `QUA-693` | Model-consistent synthetic snapshots are owned by the bounded calibration/model-grammar epic |
| `QUA-686` | Calibration-layer model grammar consumes this queue but is a separate workstream |

## Validation Posture

This queue should be implemented with explicit local, regional, and global
validation.

### Local

- curve-role propagation tests
- calibration term-builder tests
- residual/tolerance policy tests
- source-kind dispatch tests
- provenance schema rendering tests

### Regional

- snapshot-to-market-state provenance handoff
- calibration helper to result/provenance handoff
- runtime-result-replay alignment for selected curve names and source kinds

### Global

- representative multi-curve cap/floor and swaption replays
- task-result and persisted-run checks for selected curve names
- documentation consistency across quant, developer, and user-guide surfaces

## Official Docs Expected To Change

If this plan is implemented as intended, the main official docs that should be
updated during closeout are:

- `docs/market_data_workstream.md`
- `docs/mathematical/calibration.rst`
- `docs/quant/pricing_stack.rst`
- `docs/developer/task_and_eval_loops.rst`
- `docs/developer/overview.rst`
- `docs/user_guide/market_data.rst`
- `docs/user_guide/pricing.rst`

## Deferred Scope

The following remain explicitly outside this queue:

- live market-data connectors
- enterprise adapter contracts
- non-rates model-grammar work
- model-consistent vol/credit synthetic snapshots
- generalized rates-exotic expansion beyond the already-landed calibration and
  callable-rate foundations
