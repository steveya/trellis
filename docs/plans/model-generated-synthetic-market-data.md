# Model-Generated Synthetic Market Data Plan

## Purpose

This document defines the implementation queue for the next synthetic
market-data workstream in Trellis.

The goal is to make mock and proving snapshots more believable and less
error-prone by generating bounded rates, credit, and volatility fixtures from
explicit model packs rather than from disconnected heuristic tables.

## Why This Plan Exists

Trellis already has a usable synthetic market-data path:

- [trellis/data/mock.py](../../trellis/data/mock.py) produces deterministic full
  snapshots for `source="mock"`
- [docs/market_data_workstream.md](../market_data_workstream.md) describes the
  broader market-data roadmap
- `QUA-693` added a descriptive
  `prior_parameters.model_consistency_contract` to the mock path
- the calibration stack now has explicit model, quote-map, and materialization
  authority surfaces from the completed `QUA-686` workstream

That is a good foundation, but the actual generator in
`trellis/data/mock.py` is still mostly regime-template-based:

- forecast curves still come from fixed basis shifts
- credit curves still start from hand-authored spread grids
- rate vol, local vol, and model-parameter packs are still mostly built from
  regime lookup tables

This is the gap this plan closes.

## Repo-Grounded Current State

### Foundation already present

The repo already has:

- deterministic mock snapshot generation in `trellis/data/mock.py`
- typed quote-map surfaces in `trellis/models/calibration/quote_maps.py`
- typed calibrated-object materialization in
  `trellis/models/calibration/materialization.py`
- canonical model-grammar entries in
  `trellis/agent/knowledge/canonical/model_grammar.yaml`
- benchmark examples that already generate quotes from bounded Heston, SABR,
  Hull-White, local-vol, and credit assumptions in
  `trellis/models/calibration/benchmarking.py`

### Remaining gaps

The remaining gaps that matter for believable synthetic data are:

- quote-space and runtime-space are still not generated from one authority
  surface in the mock path
- rates fixtures preserve multi-curve roles descriptively, but not yet through
  one model-generated rates bundle
- credit still starts from spread grids instead of hazard-first assumptions
- local vol is still generated independently rather than derived from the same
  implied-vol authority surface as the stochastic-volatility pack
- hardening is still focused on descriptive consistency rather than round-trip
  consistency

## What This Plan Is Not

This plan does not attempt to:

- turn the mock path into a production market-data source
- build a generic stochastic market simulator for every asset class
- replace live connectors or file-based snapshot import
- broaden immediately into hybrids, structural credit, rough vol, or
  cross-currency dynamics
- run full expensive calibrations inside `fetch_market_snapshot()` for every
  synthetic request

## Target Pipeline

The target generation pipeline is:

`anchor snapshot/regime + seed -> bounded model packs -> synthetic quote bundles -> runtime market objects -> MarketSnapshot + provenance`

The important shift is that the quote inputs and runtime objects should come
from the same bounded model assumptions.

## Design Guardrails

### 1. Deterministic and seeded

The same snapshot date, request inputs, and seed must produce the same
synthetic snapshot.

### 2. Reuse shipped families only

The generator should only depend on model families Trellis already ships and
documents, such as the bounded Hull-White / SABR / Heston / Dupire /
reduced-form credit slices.

### 3. Quote-space and runtime-space must share the same authority

Synthetic spreads, vol quotes, and curve bundles must be derived from the same
seeded model packs that also produce the runtime `CreditCurve`, `YieldCurve`,
`VolSurface`, local-vol, and model-parameter artifacts.

### 4. Preserve explicit multi-curve roles

Synthetic rates fixtures must continue to preserve discount-vs-forecast curve
roles and basis information explicitly.

### 5. Derived local vol, not guessed local vol

Where the mock path carries both implied-vol and local-vol fixtures, the
local-vol fixture should be derived from the implied-vol authority surface
rather than guessed independently.

### 6. Provenance must stay obviously synthetic

The synthetic origin, seed, model family, and parameterization must remain
clear in snapshot provenance, proving runs, and related diagnostics.

## Ordered Delivery Queue

The recommended queue for a single agent is:

1. `QUA-695` Synthetic market data: seeded generation contract and model-pack authority
2. `QUA-696` Synthetic rates: model-generated multi-curve bundle and rate-vol fixtures
3. `QUA-697` Synthetic credit: hazard-first spread generation and runtime fixtures
4. `QUA-698` Synthetic volatility: Heston-generated smiles and derived local-vol fixtures
5. `QUA-699` Synthetic market data: round-trip validation, task fixtures, and docs hardening

This order is intentional:

- establish one synthetic-generation authority first
- move rates onto that authority while preserving multi-curve roles
- move credit onto a hazard-first authority surface
- move volatility so implied-vol, local-vol, and parameter packs stop drifting
- harden the final path with round-trip tests, task fixtures, and docs

## Critical Path

The critical path is:

`QUA-695 -> QUA-696 -> QUA-697 -> QUA-698 -> QUA-699`

`QUA-697` and `QUA-698` can run in parallel after `QUA-695` if separate agents
are assigned and their write scopes are kept disjoint.

## Shared Dependencies and Coordination

This workstream depends on already-completed foundation from:

- `QUA-693` for the descriptive synthetic model-consistency contract
- `QUA-686` for typed model specs, quote maps, and materialization surfaces

It should stay aligned with:

- `QUA-694` for the remaining multi-curve rates provenance and source-kind work
- `QUA-362` for broader provenance trace/report schema alignment

Practical coordination rules:

- synthetic rates work must preserve the multi-curve roles established by the
  rates platform workstream
- the final hardening slice should read the latest `QUA-362` closeout or
  progress note before it finalizes provenance/report expectations
- the benchmark harness in `trellis/models/calibration/benchmarking.py` should
  be treated as a reuse target, not as throwaway example code

## Agent Intake Bundle

Each local or cloud coding agent assigned to this workstream should start with:

- this plan doc
- `AGENTS.md`
- `docs/market_data_workstream.md`
- `docs/plans/calibration-layer-model-grammar-unification.md`
- `docs/plans/multi-curve-rates-completion.md`
- the current Linear ticket body plus any hard blockers
- the code paths under `trellis/data/mock.py`,
  `trellis/models/calibration/`, `trellis/core/market_state.py`,
  `trellis/curves/`, and the relevant tests

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
| `QUA-361` | Backlog |

### Ordered Queue

| Ticket | Slice | Status |
| --- | --- | --- |
| `QUA-695` | Seeded generation contract and model-pack authority | Done |
| `QUA-696` | Model-generated multi-curve rates bundle and rate-vol fixtures | Done |
| `QUA-697` | Hazard-first synthetic credit generation | Done |
| `QUA-698` | Heston-generated implied-vol and derived local-vol fixtures | Backlog |
| `QUA-699` | Round-trip validation, task fixtures, and docs hardening | Backlog |

### Adjacent Shared Work

These are related but not owned by this queue:

| Ticket | Reason |
| --- | --- |
| `QUA-693` | Completed descriptive foundation for model-consistency metadata |
| `QUA-694` | Multi-curve rates semantics and provenance remain a separate rates workstream |
| `QUA-362` | Broader provenance trace/report schema remains a separate workstream |
| `QUA-686` | The bounded calibration/model-grammar epic already landed the authority surfaces this plan consumes |

## Validation Posture

This queue should be implemented with explicit local, regional, and global
validation.

### Local

- deterministic generation tests for seeds and parameter packs
- family-specific sanity tests for positivity, monotonicity, and bounded
  surfaces
- provenance schema tests for the synthetic generation contract

### Regional

- rates quote bundle to multi-curve runtime handoff
- hazard-first credit generation to typed credit calibration handoff
- Heston-generated implied-vol bundle to local-vol derivation handoff
- benchmark fixture reuse from the bounded calibration workflows

### Global

- round-trip calibration canaries for the synthetic rates, credit, and
  volatility slices
- proving or task-runner fixtures that exercise the improved synthetic path
- replay/debug expectations that keep the synthetic origin explicit

## Official Docs Expected To Change

If this plan is implemented as intended, the main official docs that should be
updated during closeout are:

- `docs/market_data_workstream.md`
- `docs/mathematical/calibration.rst`
- `docs/developer/composition_calibration_design.md`
- `docs/developer/task_and_eval_loops.rst`
- `docs/user_guide/market_data.rst`
- `docs/user_guide/pricing.rst`

## Deferred Scope

The following remain explicitly deferred until this bounded synthetic path is
stable:

- basket or structural credit synthetic generators
- rough-vol, stochastic-local-vol, or hybrid-model synthetic generators
- cross-currency or enterprise-grade multi-curve buildouts
- live provider integration and external market-data adapters
- replacing empirical production inputs with synthetic priors by default
