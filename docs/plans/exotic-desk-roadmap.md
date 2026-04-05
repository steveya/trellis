# Exotic Desk Roadmap

## Purpose

This document defines a product and implementation roadmap for positioning
Trellis as a practical pricing and risk platform for under-supported traders
and small trading pods before it expands into heavier enterprise integration.

The first customer wedge is not "large bank model platform replacement."
The first customer wedge is:

- a trader or small pod with real P&L responsibility
- a platform seat that can book or warehouse the trade but cannot price it well
- limited or no dedicated quant-engineering support
- immediate need for defendable prices, scenarios, and risk summaries on
  awkward products

The canonical example is a rates or exotics trader who needs to price and
analyze a structure such as a range accrual, callable range note, callable
structured note, barrier/range hybrid, or Bermudan-style product on a platform
that does not have native support for the trade.

This plan is intentionally written as a repo-grounded roadmap, not as a
generic fintech market memo. It is meant to guide subsequent Linear epics and
implementation slices.

## Decision Summary

The roadmap direction is:

- lead with unsupported-trade pricing and risk for small pods
- optimize first for explicit trader-supplied trade details and market inputs
- make outputs explainable, auditable, and operationally useful
- treat enterprise ingestion as a later multiplier, not as the first gate
- treat bank-grade counterparty and collateral infrastructure as a later phase
  after the trader workflow is strong

The roadmap is successful only when Trellis can replace "ask a quant to build a
one-off spreadsheet or script" for a meaningful set of real desk workflows.

## Linear Ticket Mirror

These tables mirror the current Linear epic tree for the exotic-desk roadmap
and define the intended implementation order.

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This roadmap file is the repo-local execution mirror for subsequent agents.
- Implement the earliest ticket in the ordered queue below whose status is not
  `Done` and whose upstream prerequisites are already satisfied.
- Skip tickets whose table row is `Done`.
- Epic rows are tracking rows. Do not pick them up before their earlier child
  rows unless the epic has no remaining open child slices.
- Before coding, print the current ticket number and its plain-English goal on
  screen.
- Follow the implementation workflow in `AGENTS.md`:
  - review upstream tickets and notes first
  - implement with TDD
  - update docs after tests pass
  - leave the structured closeout note in Linear
  - mark the Linear ticket `Done`
  - only then update the corresponding row in this file
- Update this table only after the ticket is closed in Linear.

Status mirror last synced: `2026-04-05`

### Ordered Epic Queue

| Ticket | Status |
| --- | --- |
| `QUA-589` Exotic desk MVP | Done |
| `QUA-590` Calibration and market realism | Done |
| `QUA-591` Pod risk and explain | Done |
| `QUA-592` Book ingestion | Done |
| `QUA-593` Enterprise integration | Backlog |
| `QUA-594` Institutional valuation stack | Backlog |

### Ordered Exotic Desk MVP Queue

| Ticket | Status |
| --- | --- |
| `QUA-595` Semantic contract: range accrual trade entry | Done |
| `QUA-313` Market snapshot: file-based curves, surfaces, and fixings import | Done |
| `QUA-596` Range accrual pricing: checked route and validation bundle | Done |
| `QUA-598` Pricing output: trader audit bundle and assumptions summary | Done |
| `QUA-599` HTTP/MCP surface: one-trade pricing workflow | Done |
| `QUA-597` Callable rates structures: exercise schedule and state binding | Done |

### Ordered Calibration And Market Realism Queue

| Ticket | Status |
| --- | --- |
| `QUA-317` Market data: fixing and rate-history schema | Done |
| `QUA-603` Market conventions: accrued interest and YTM completion | Done |
| `QUA-623` Calibration substrate: typed solve request and objective bundle | Done |
| `QUA-624` Calibration substrate: optimizer backend registry and capability checks | Done |
| `QUA-625` Calibration governance: solver provenance and replay artifacts | Done |
| `QUA-626` Rates calibration: bootstrap convention bundle and market instrument surface | Done |
| `QUA-627` Rates calibration: Jacobian-aware bootstrap solve path | Done |
| `QUA-600` Rates calibration: Hull-White term-structure fit | Done |
| `QUA-628` Volatility calibration: SABR smile surface assembly and fit diagnostics | Done |
| `QUA-601` Volatility calibration: SABR smile workflow | Done |
| `QUA-629` Equity-vol process: Heston runtime process and parameter binding | Done |
| `QUA-630` Local-vol calibration: surface hardening and stability checks | Done |
| `QUA-602` Equity-vol calibration: Heston and local-vol workflow | Done |
| `QUA-631` Calibration performance: throughput benchmark and warm-start baseline | Done |
| `QUA-604` Validation: calibration replay and tolerance fixtures | Done |

### Ordered Benchmark Hygiene Queue

| Ticket | Status |
| --- | --- |
| `QUA-680` Benchmark artifacts: portable checked baselines and folder guidance | Done |

### Ordered Pod Risk And Explain Queue

| Ticket | Status |
| --- | --- |
| `QUA-632` Risk substrate: interpolation-aware curve shock engine | Done |
| `QUA-605` Risk analytics: interpolation-aware key rate durations | Done |
| `QUA-606` Risk analytics: twist and butterfly scenario packs | Done |
| `QUA-643` Risk analytics: retire legacy exact-knot KRD path | Done |
| `QUA-644` Risk analytics: quote-space rebuilt curve KRD workflow | Done |
| `QUA-645` Risk outputs: methodology and bucket-convention disclosure | Done |
| `QUA-646` Risk analytics: rebuild-based rates scenario packs | Done |
| `QUA-633` Risk substrate: volatility surface bucket and bump engine | Done |
| `QUA-607` Volatility risk: bucketed vega by expiry and strike | Done |
| `QUA-634` Risk runtime: delta gamma theta measure implementations | Done |
| `QUA-608` Callable analytics: OAS duration and callable scenario explain | Done |
| `QUA-635` Portfolio risk: scenario result cube and aggregation substrate | Done |
| `QUA-609` Portfolio explain: book P&L attribution | Done |
| `QUA-610` Trade explain: driver narrative and scenario commentary | Done |
| `QUA-636` Risk performance: scenario and sensitivity throughput benchmark pack | Done |

### Ordered Book Ingestion Queue

| Ticket | Status |
| --- | --- |
| `QUA-611` Position schema: generic exotic trade import contract | Done |
| `QUA-612` Book ingestion: mixed-instrument CSV/JSON loaders | Done |
| `QUA-613` Snapshot resolution: request-driven component selection | Done |
| `QUA-637` Book execution: scenario batching and reusable compute plan | Done |
| `QUA-614` Book execution: saved scenario templates and batch outputs | Done |

### Ordered Enterprise Integration Queue

| Ticket | Status |
| --- | --- |
| `QUA-615` Connector contract: external trade-store import adapters | Backlog |
| `QUA-616` Connector contract: governed market-data provider adapters | Backlog |
| `QUA-617` Reconciliation: snapshot provenance and stale-data warnings | Backlog |
| `QUA-618` Service integration: read-only enterprise ingestion MVP | Backlog |

### Ordered Institutional Valuation Queue

| Ticket | Status |
| --- | --- |
| `QUA-620` Semantic contract: collateral and netting set representation | Backlog |
| `QUA-638` Counterparty exposure: swap portfolio future value cube | Backlog |
| `QUA-639` Collateral workflow: margin period and collateral state projection | Backlog |
| `QUA-640` Netting workflow: netting-set aggregation and closeout exposure inputs | Backlog |
| `QUA-641` Counterparty exposure: EE EPE PFE aggregation outputs | Backlog |
| `QUA-642` Counterparty performance: exposure simulation benchmark pack | Backlog |
| `QUA-619` Counterparty valuation: xVA engine for swap portfolios | Backlog |
| `QUA-621` Governed workflow: approval and execution gates for production valuation | Backlog |

### Cross-Epic Sequencing Constraints

Use the queue order above, but also respect these concrete handoff rules:

- Do not start `QUA-596` until `QUA-595` and `QUA-313` are landed.
- Do not start `QUA-598` until `QUA-596` is landed.
- Do not start `QUA-599` until `QUA-595`, `QUA-313`, `QUA-596`, and `QUA-598`
  are landed.
- Treat `QUA-597` as the second supported trade-pack slice after the first
  range-accrual path is already working.
- Do not start `QUA-624` until `QUA-623` is landed.
- Do not start `QUA-625` until `QUA-623` and `QUA-624` are landed.
- Do not start `QUA-626` until `QUA-317` and `QUA-603` are landed.
- Do not start `QUA-627` until `QUA-623`, `QUA-624`, and `QUA-626` are
  landed.
- Do not start `QUA-600` until `QUA-623`, `QUA-624`, `QUA-625`, `QUA-626`,
  and `QUA-627` are landed.
- Do not start `QUA-628` until `QUA-623`, `QUA-624`, and `QUA-625` are
  landed.
- Do not start `QUA-601` until `QUA-623`, `QUA-624`, `QUA-625`, and
  `QUA-628` are landed.
- Do not start `QUA-629` or `QUA-630` until `QUA-623` and `QUA-625` are
  landed.
- Do not start `QUA-602` until `QUA-623`, `QUA-624`, `QUA-625`, `QUA-629`,
  and `QUA-630` are landed.
- Do not start `QUA-631` until `QUA-600`, `QUA-601`, and `QUA-602` are
  landed.
- Do not start `QUA-604` until `QUA-600`, `QUA-601`, `QUA-602`, and
  `QUA-631` are landed.
- Do not start `QUA-605` until `QUA-632` is landed.
- Do not start `QUA-607` until `QUA-633` is landed.
- Do not start `QUA-609` until the book-ingestion baseline in `QUA-611`,
  `QUA-612`, and `QUA-614` is landed, and the reusable scenario-result
  substrate in `QUA-635` plus the compute-plan slice in `QUA-637` are also
  landed.
- Do not start `QUA-614` until `QUA-637`, `QUA-611`, and `QUA-612` are
  landed.
- Do not start `QUA-636` until `QUA-635` is landed.
- Do not start `QUA-618` until `QUA-615`, `QUA-616`, and `QUA-617` are landed.
- Do not start `QUA-638` until `QUA-620` is landed.
- Do not start `QUA-639` until `QUA-620` and `QUA-638` are landed.
- Do not start `QUA-640` until `QUA-620`, `QUA-638`, and `QUA-639` are
  landed.
- Do not start `QUA-641` until `QUA-620`, `QUA-638`, `QUA-639`, and
  `QUA-640` are landed.
- Do not start `QUA-642` until `QUA-620`, `QUA-638`, and `QUA-641` are
  landed.
- Do not start `QUA-619` until `QUA-620`, `QUA-638`, `QUA-639`, `QUA-640`,
  and `QUA-641` are landed.
- Do not start `QUA-621` until `QUA-619` is landed.

### Agent Pickup Directive

If you hand this file to a coding agent for end-to-end implementation, the
agent should:

1. start at the top of the epic queue
2. within the active epic, pick the first non-`Done` child ticket whose
   prerequisites are satisfied
3. implement only that ticket's scoped outcome
4. close the ticket in Linear first
5. then update the mirrored status row in this roadmap before moving on

Do not skip ahead to later epics because a later ticket looks easier. This
roadmap is intentionally ordered to preserve the trader-first wedge.

## Why This Plan Exists

Trellis already has many of the ingredients that matter for a commercial
pricing platform:

- strong deterministic pricing substrate across trees, Monte Carlo, PDE,
  transforms, copulas, and analytical routes
- semantic request compilation and typed contract work
- governed execution context, policy, audit, run ledger, and model lifecycle
- a knowledge-backed build path for unsupported products
- explicit market snapshot and provenance work

What the repo does not yet have is a product slice that is clearly optimized for
the operational reality of a lean trading desk:

- real trade-entry flows for awkward products
- market-data loading that works without enterprise plumbing
- risk and explain outputs oriented toward traders rather than framework demos
- simple workflows that do not presume a quant platform team
- direct book and position ingestion beyond narrow bond-centric paths

This document defines that slice and the staged expansion after it.

## Target Customer Order

### Primary initial customer

An exotics or structured-products trader, desk analyst, or small pod on a hedge
fund platform or lean buy-side seat that:

- trades products that the platform does not price natively
- cannot rely on a dedicated quant desk for every trade
- can supply trade terms and market inputs manually when necessary
- needs a defendable price, scenarios, and audit trail quickly

Typical first-batch product families:

- range accruals
- callable range accrual notes
- callable structured notes
- Bermudan or callable rates trades
- barrier/range hybrids
- structured rates and hybrid products with explicit schedules

### Secondary customer

Small funds, regional banks, and specialized desks that:

- have some quant literacy but limited engineering bandwidth
- want repeatable pod-level pricing and risk workflows
- need mixed-book analytics, reusable market templates, and saved scenarios

### Later-stage customer

Larger institutions whose workflows depend on enterprise software and controlled
integration boundaries, including systems such as Murex, Calypso, or Front
Arena, plus broader counterparty, collateral, approval, and reporting layers.

## Product Principles

These are the non-negotiable product rules for the roadmap:

1. trader usefulness comes before enterprise completeness
2. explicit inputs and explicit warnings beat silent inference
3. defendable prices with assumptions and audit trails beat opaque automation
4. first workflows must work with file-based and manually supplied data
5. pricing must ship with risk and explain, not price-only outputs
6. first-batch products should be productized as supported trade packs, not as
   ad hoc research demos
7. enterprise adapters must be additive later, not a prerequisite for initial
   value

## Repo-Grounded Current State

### Strengths to build on

- governed execution, policy, run ledgers, and audit bundles already exist
- typed parse, missing-field reporting, and deterministic match flows are now
  part of the platform surface
- market snapshots, named components, and provenance scaffolding exist
- the pricing substrate already spans many product families and numerical
  methods
- MCP and local HTTP transport already provide thin service surfaces

### Product gaps that block the first wedge

- live market-data auto-resolution remains discount-curve-first instead of
  full snapshot resolution
- direct book and position ingestion remain narrow and mostly bond-shaped
- analytics remain useful but still light for real exotic desk workflows
- several desk-critical calibration and convention details remain incomplete
- fixing/history support and richer market-input workflows are still planned,
  not complete
- enterprise connector work is still future-facing

## Roadmap Objective

The roadmap should move Trellis through three stages:

1. **Trader-in-the-loop MVP**
   Trellis can price one awkward real trade from explicit terms and explicit
   market inputs and return usable price, risk, and audit outputs.
2. **Pod operating system**
   Trellis can support recurring desk workflows across a small book with saved
   scenarios, explain, and reusable market/calibration templates.
3. **Enterprise expansion**
   Trellis can integrate with upstream systems and support broader governance,
   portfolio, and institutional valuation requirements.

## Capability Requirements

These requirements apply across the roadmap and define what "commercially
useful" means for the target customer base.

### Trade-entry requirements

- Trellis must support explicit structured trade capture for first-batch
  products without requiring custom code edits.
- Schedule-bearing products must accept typed schedule fields, fixing dates,
  accrual barriers, call schedules, coupon logic, and observation logic.
- The system must make missing trade terms explicit and return actionable
  missing-field diagnostics.
- The first-batch products should be exposed as supported route or contract
  slices, not only as free-form generated candidates.

### Market-data requirements

- Trellis must load usable market inputs from explicit trader-supplied files
  before it depends on live enterprise connectors.
- Curves, vol surfaces, FX, credit inputs, spots, and fixings/history must be
  representable as named snapshot components.
- Every run must record what market data was used, what was synthetic, what was
  missing, and what assumptions were injected.
- Stale-data and incomplete-market warnings must be first-class outputs.

### Pricing and calibration requirements

- First-batch products must have checked and validated pricing paths.
- Numerical method choice must be explicit enough for a trader or reviewer to
  understand how the price was produced.
- Calibration workflows must be packaged for desk use, not just exposed as raw
  model-fitting helpers.
- Calibration workflows should expose typed solve requests, objective bundles,
  and a pluggable optimizer/backend seam rather than hard-wiring solver calls
  inside model helpers.
- Solver backend identity, options, termination summary, and replay inputs
  must be first-class calibration provenance.
- Supported calibration workflows should carry explicit throughput and
  warm-start baselines, not only fit residuals.
- Known hard-coded assumptions that materially affect desk credibility should be
  retired or surfaced as explicit assumptions.

### Risk requirements

- Every first-batch supported trade must return more than PV.
- Minimum desk outputs should include:
  - PV
  - cashflow or event schedule summary where relevant
  - scenario ladder
  - DV01
  - key-rate risk where applicable
  - usable vega or volatility sensitivity summary
  - product-specific risk outputs when needed, such as OAS-style analytics for
    callable structures
- Risk outputs should be assembled from reusable curve-shock and
  volatility-surface-bump substrate rather than route-local bump code.
- Runtime measure coverage should state supported products and fallback
  behavior explicitly.
- Book-level risk and explain flows should be able to consume a reusable
  scenario-result cube rather than only isolated repricing results.
- Pod-level workflows must later add attribution, scenario packs, richer
  bucketing, and explicit scenario/sensitivity throughput baselines.

### Explain and audit requirements

- Every production-style output must describe assumptions, selected route or
  engine, warnings, and provenance.
- Trellis must explain why a price moved under scenarios in plain desk terms,
  not only through raw tensors or diagnostics.
- Audit bundles must be stable enough to attach to a trade-review or desk
  workflow.

### Workflow requirements

- Initial workflows must work through Python, notebook, MCP, and lightweight
  HTTP usage.
- The system must not presume a centralized quant platform team to operate it.
- Repricing, replay, and saved-snapshot workflows should be easy to run from a
  small team environment.

### Enterprise-later requirements

- Enterprise integrations should target trade-store and market-data software
  only after the explicit-input workflow is robust.
- Read-only ingestion should precede write-back and downstream workflow control.
- Institutional layers such as xVA, collateral, netting, approvals, and
  broader reporting should follow after the trader and pod workflows are proven.

## Phase 1: Trader-In-The-Loop MVP

### Objective

Make Trellis useful for a trader who has one awkward real trade and does not
have a quant desk on call.

### Customer outcome

The user can enter a structured or exotic trade, supply explicit market inputs,
and receive:

- a defendable PV
- a scenario ladder
- core risk outputs
- assumptions and warnings
- a replayable audit bundle

### In-scope requirements

- supported trade-entry packs for first-batch products:
  - range accrual
  - callable range note
  - callable structured note
  - Bermudan-style rates trade
  - barrier/range hybrid where feasible
- file-based market snapshot ingestion:
  - discount and forecast curves
  - vol surfaces
  - FX data
  - credit inputs where relevant
  - fixings/history for products that require them
- checked pricing routes and validation bundles for the supported trade packs
- trader-facing output pack with PV, risk, scenarios, assumptions, and audit
- lightweight runtime surfaces:
  - notebook and Python
  - MCP workflow
  - local HTTP deployment

### Deliverables

#### Trade-pack deliverables

- typed semantic contracts or governed trade schemas for the first-batch
  products
- first-batch route bindings with deterministic validation coverage
- explicit missing-field guidance for each trade pack

#### Market-input deliverables

- file-based snapshot import schema and loader contracts
- named snapshot components for curves, surfaces, FX, credit, and fixings
- warnings for stale, missing, synthetic, or defaulted inputs

#### Output deliverables

- one-trade result projection for desk use
- scenario ladder projection with common stress templates
- cashflow or event schedule projection where relevant
- run audit bundle suitable for review and replay

#### Workflow deliverables

- documented first-batch workflow for pricing an unsupported exotic trade from
  explicit inputs
- MCP prompt or tool flow for guided desk usage
- lightweight HTTP surface for local pod deployment

### Non-goals

- broad enterprise system integration
- full portfolio VaR or ES
- counterparty valuation stack
- automatic coverage for every exotic structure
- replacing all desk spreadsheets on day one

### Phase 1 exit criteria

- a target trader can price a real range accrual or similar structured trade
  without code edits
- Trellis clearly identifies missing trade terms or market components
- the output is strong enough to support quoting, triage, and internal review

## Phase 2: Pod-Level Risk And Explain

### Objective

Move from one-trade assistance to repeatable small-pod workflow support.

### Customer outcome

A small exotic pod can use Trellis for recurring pricing and risk work across a
mixed book without needing bespoke quant support for each structure.

### In-scope requirements

- richer scenario packs:
  - parallel shocks
  - twist and butterfly
  - volatility scenario ladders
  - product-aware callable or exercise scenarios
- richer sensitivity outputs:
  - interpolation-aware KRD built on reusable curve-shock substrate
  - vega bucketing built on reusable volatility-surface bump substrate
  - callable OAS duration
  - fuller Greeks where applicable, with explicit support boundaries
- explain surfaces:
  - driver commentary
  - scenario explain
  - exercise boundary or optionality explain where relevant
- book and pod workflows:
  - mixed-position ingestion
  - reusable scenario-result cube and aggregation substrate
  - scenario batching and reusable compute plans
  - saved scenario templates
  - book-level attribution and summary outputs
- desk-usable calibration workflows:
  - typed, solver-open calibration substrate with provenance and replay
  - Hull-White
  - SABR
  - Heston
  - local volatility where needed
  - calibration, scenario, and sensitivity throughput baselines

### Deliverables

#### Risk deliverables

- reusable curve-shock and volatility-surface-bump substrate with supported KRD
  and vega outputs
- runtime delta, gamma, theta, and richer callable/rates analytics
- reusable scenario-result cube, book-level P&L attribution, and scenario
  summary projections
- benchmark fixtures for scenario and sensitivity throughput

#### Explain deliverables

- structured "what drives this price" output
- scenario commentary output for desk review
- route and assumption summaries that a trader can hand to PM or risk

#### Workflow deliverables

- generic position schema for pod-level books
- reusable scenario batching and compute plans
- saved scenario packs and reusable run templates
- reusable market and calibration templates by product family

### Non-goals

- deep enterprise trade-store integration
- institution-wide risk aggregation
- counterparty, collateral, and netting infrastructure

### Phase 2 exit criteria

- a small pod can run recurring pricing and risk on a mixed exotic book
- Trellis is credible as a daily decision-support tool, not only as a special
  case pricer
- numbers are explainable enough for PM and risk conversations without a
  quant engineer in the loop

## Phase 3: Enterprise Expansion

### Objective

Expand Trellis from a pod operating tool into a governed service that can sit
inside broader institutional workflows.

### Customer outcome

Larger funds and banks can integrate Trellis into upstream trade and market-data
systems while preserving the audit, lifecycle, and controlled execution story.

### In-scope requirements

- read-first connectors for enterprise trade stores and market-data systems
- request-driven snapshot resolution from upstream identifiers
- stronger workflow controls around approvals and governed execution
- broader portfolio and institutional risk layers
- explicit exposure workflow before xVA aggregation
- later-stage counterparty, collateral, and netting work

### Deliverables

#### Integration deliverables

- connector contracts for trade-store and market-data ingestion
- read-only adapters for common enterprise software workflows
- reconciliation, provenance, and stale-data surfacing across imported runs

#### Governance deliverables

- stronger approval and execution gating for production deployments
- broader run, snapshot, and model review tooling
- deployment guidance for shared service usage

#### Institutional valuation deliverables

- future-value exposure cube plus EE, EPE, and PFE outputs
- collateral-state projection and netting-set aggregation semantics
- xVA framework built on those supporting exposure workflows
- exposure-simulation benchmark pack
- later reporting and downstream integration hooks

### Non-goals

- turning Trellis into a general purpose enterprise data warehouse
- requiring enterprise software before value can be realized

### Phase 3 exit criteria

- Trellis can be adopted beyond self-contained pod workflows
- enterprise ingestion is practical and governed
- institutional layers do not compromise the clarity of the original trader
  workflow

## Proposed Linear Epic Structure

This plan should decompose into a small number of clear epics with child slices.

### Epic 1: Exotic desk MVP

Objective:
make one-trade pricing, risk, and audit usable for unsupported exotic and
structured products.

Candidate child slices:

- semantic contract for range accrual trade entry
- supported pricing route and validation bundle for range accrual
- callable structured trade packs
- file-based market snapshot import
- one-trade audit and output bundle
- HTTP and MCP desk workflow surfaces

### Epic 2: Pod risk and explain

Objective:
support recurring small-pod pricing and risk workflows.

Candidate child slices:

- interpolation-aware curve-shock substrate and KRD
- twist and butterfly scenarios
- volatility-surface bump substrate and vega bucketing
- runtime Greeks plus callable OAS and callable explain
- scenario-result cube and book P&L attribution
- trade-driver narrative outputs and throughput benchmarks

### Epic 3: Calibration and market realism

Objective:
make first-batch numbers desk-credible under real market inputs and conventions.

Candidate child slices:

- fixing/history schema and runtime support
- typed calibration solve request and objective bundle
- optimizer backend registry and capability checks
- solver provenance and replay artifacts
- rates bootstrap market-input and solve-path substrate
- rates calibration workflow
- SABR, Heston, and local-vol workflow slices
- market-convention hardening
- calibration performance baselines
- validation fixtures and replay coverage

### Epic 4: Book ingestion

Objective:
support mixed trade books without bespoke code for each desk.

Candidate child slices:

- generic position schema
- mixed-book CSV/JSON import
- request-driven snapshot resolution
- scenario batching and reusable compute plan
- saved scenario templates

### Epic 5: Enterprise adapters

Objective:
bring Trellis into larger operational stacks after the small-pod wedge is proven.

Candidate child slices:

- trade-store connector contract
- market-data connector contract
- read-only enterprise adapters
- provenance and reconciliation tooling

### Epic 6: Institutional valuation stack

Objective:
support broader institutional adoption after the first wedge and pod workflows
are stable.

Candidate child slices:

- institutional semantics
- exposure cube and EE/EPE/PFE outputs
- collateral and netting semantics
- xVA framework
- exposure simulation benchmarks
- stronger governed approval workflows

## Delivery Order

The recommended delivery order is:

1. Phase 1 / Epic 1
2. Phase 1 market-input and calibration-substrate slices from Epic 3
3. Phase 2 / Epic 2
4. Phase 2 book slices from Epic 4
5. Phase 3 / Epic 5
6. Phase 3 / Epic 6

This order preserves the intended wedge:

- unsupported trader first
- pod second
- enterprise later

## Success Metrics

The roadmap should be evaluated against practical adoption metrics, not only
technical completeness.

### Phase 1 metrics

- time from raw terms to first defendable result
- fraction of first-batch trades that can be expressed without code edits
- fraction of runs with explicit market provenance and warnings
- number of trades where traders can proceed without bespoke quant help

### Phase 2 metrics

- repeated pod usage across a mixed book
- number of supported recurring scenario and explain workflows
- reduction in bespoke spreadsheet or one-off script usage

### Phase 3 metrics

- successful read-only integration with upstream systems
- reduction in manual rekeying of trades and market inputs
- controlled production usage under governed policies

## Risks And Failure Modes

- building enterprise adapters too early and delaying the trader wedge
- over-indexing on model breadth without productized trade-entry flows
- shipping price-only results without desk-usable explain and risk
- depending on live connectors before the explicit-input workflow is strong
- treating research candidate generation as a substitute for supported desk
  routes

## Done Criteria For This Plan

This roadmap should be considered translated into execution when:

- the phases above are represented by explicit Linear epics
- each epic is split into reviewable implementation slices
- the first delivery wave is clearly biased toward unsupported exotic desk
  workflows rather than enterprise integration
- the associated docs under `docs/user_guide/`, `docs/developer/`, and
  `docs/quant/` are updated as implementation lands
