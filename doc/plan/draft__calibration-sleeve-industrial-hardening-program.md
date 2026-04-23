# Calibration Sleeve Industrial Hardening Program

## Status

Active execution mirror for the filed calibration-sleeve industrialization
queue, now coordinated with the Autograd Phase 2 implementation queue for
portfolio AAD and derivative governance.

The umbrella `QUA-946`, child tickets `QUA-947` through `QUA-956`, adjacent
Autograd Phase 2 tickets `QUA-966` through `QUA-971`, and post-`QUA-955`
hybrid follow-ons `QUA-972` / `QUA-973` were all filed and completed in
Linear. This document is now the closeout mirror for that queue and should
remain aligned with the landed issue graph.

The adjacent Autograd Phase 2 plan is tracked by `QUA-966` through `QUA-971`.
It should not duplicate calibration curve, surface, or cube plants. It should
consume the stronger market objects produced by this calibration program and
define truthful derivative operators, portfolio-scale sensitivity workflows,
and runtime derivative reporting around those objects.

Status mirror last synced: `2026-04-23`

## Linked Context

- `QUA-590` Calibration and market realism
- `QUA-663` Calibration review: rates, local vol, SABR, Heston, and solve
  requests
- `QUA-686` Semantic model grammar: calibration-layer unification for model
  specs, quote maps, and bindings
- `QUA-946` Calibration sleeve: Trellis-native industrial hardening program
- `QUA-947` Calibration architecture: align Trellis-native docs, runtime
  vocabulary, and plan mirror
- `QUA-948` Equity-vol calibration: carry-consistent pricing and implied-vol
  inversion
- `QUA-949` Credit calibration: CDS-pricer-backed single-name objective and
  diagnostics
- `QUA-950` Equity-vol calibration: industrial surface foundation and staged
  model fits
- `QUA-951` Rates calibration: dated-instrument multi-curve hardening and
  dependency DAG
- `QUA-952` Rates-vol calibration: caplet stripping, swaption cube assembly,
  and model diagnostics
- `QUA-953` Credit curve calibration: schedule-aware CDS workflow, quote
  normalization, and hazard governance
- `QUA-954` Basket credit calibration: base-correlation workflow and
  tranche-surface governance
- `QUA-955` Hybrid calibration: dependency DAG and first cross-asset slice
- `QUA-956` Calibration validation: desk-like fixtures, perturbation
  diagnostics, and latency envelopes
- `QUA-966` Autograd Phase 2: portfolio AAD and gradient governance
- `QUA-967` Autograd backend: JVP VJP and HVP operator implementation
- `QUA-968` Portfolio AAD: book-level reverse-mode sensitivity substrate
- `QUA-969` Discontinuous Greeks: smoothing and custom-adjoint policy
- `QUA-970` Gradient matrix: product-family autograd regression cohort
- `QUA-971` Runtime derivatives: expanded method selection and reporting
- `QUA-972` Hybrid validation: bounded quanto calibration fixtures and
  diagnostics
- `QUA-973` Hybrid derivative governance: bounded quanto matrix and reporting
- `doc/plan/draft__calibration-documentation-and-architecture-alignment.md`
- `doc/plan/draft__autograd-phase-2-aad-and-gradient-governance.md`
- `docs/quant/differentiable_pricing.rst`

## Linear Ticket Mirror

Rules for coding agents:

- Linear is the source of truth for ticket state.
- This file is the repo-local mirror for subsequent agents.
- Implement the earliest ticket in the ordered queue below whose status is not
  `Done` and whose hard prerequisites are satisfied.
- Treat the Autograd Phase 2 mirror as an adjacent derivative-governance lane.
  The combined queue below records cross-program sequencing, but the autograd
  plan remains the detailed source for AD2 ticket scope.
- `CAL.0A`, `CAL.0B`, `CAL.0C`, and `CAL.7` may run in parallel when their
  write scopes do not conflict.
- Do not mark a row `Done` here before the corresponding Linear issue is
  actually closed.
- When a ticket changes behavior, APIs, runtime workflow, or operator
  expectations, update the relevant docs and `LIMITATIONS.md` in the same
  closeout unless the ticket explicitly records why not.

### Workstream Ticket

| Ticket | Status |
| --- | --- |
| `QUA-946` Calibration sleeve: Trellis-native industrial hardening program | Done |
| `QUA-966` Autograd Phase 2: portfolio AAD and gradient governance | Done |

### Ordered Queue

| Queue ID | Linear | Status | Scope | Hard prerequisites |
| --- | --- | --- | --- | --- |
| `CAL.0A` | `QUA-947` | Done | Trellis-native architecture and documentation alignment | none |
| `CAL.0B` | `QUA-948` | Done | equity-vol carry consistency across pricing and implied-vol inversion | none |
| `CAL.0C` | `QUA-949` | Done | CDS-pricer-backed single-name credit objective and diagnostics | none |
| `CAL.1` | `QUA-950` | Done | industrial equity-vol surface foundation and staged model fits | `CAL.0B` |
| `CAL.2` | `QUA-951` | Done | dated-instrument multi-curve hardening and calibration dependency DAG | none; ordered after the Phase 0 slices |
| `CAL.3` | `QUA-952` | Done | caplet stripping, swaption cube assembly, and rates-vol model diagnostics | `CAL.2` |
| `CAL.4` | `QUA-953` | Done | schedule-aware single-name credit curve calibration | `CAL.0C` |
| `CAL.5` | `QUA-954` | Done | basket-credit base-correlation workflow and tranche-surface governance | `CAL.4` |
| `CAL.6` | `QUA-955` | Done | bounded rates + equity/FX quanto-correlation slice on explicit dependency DAGs and runtime materialization | `QUA-950`, `QUA-951`, `QUA-971` |
| `CAL.7` | `QUA-956` | Done | desk-like fixtures, perturbation diagnostics, and latency envelopes | none; extend alongside the active implementation slices |

### Adjacent Autograd Phase 2 Queue

| Queue ID | Linear | Status | Scope | Hard prerequisites |
| --- | --- | --- | --- | --- |
| `AD2.1` | `QUA-967` | Done | JVP, VJP, HVP operator implementation or checked backend decision | `QUA-957`, `QUA-965` |
| `AD2.2` | `QUA-968` | Done | book-level reverse-mode / portfolio AAD substrate | `AD2.1` |
| `AD2.3` | `QUA-969` | Done | smoothing and custom-adjoint policy for discontinuous products | `QUA-957` |
| `AD2.4` | `QUA-970` | Done | product-family gradient matrix and support-contract cohort expansion | consume `AD2.1` / `AD2.3` outcomes as they land |
| `AD2.5` | `QUA-971` | Done | runtime derivative-method taxonomy and reporting integration | `AD2.1`, `AD2.4` |

### Combined Implementation Queue

This is the single cross-program pickup order for the remaining calibration
and Autograd Phase 2 work. It preserves the domain boundaries: calibration
builds market objects and dependency DAGs; Autograd Phase 2 builds derivative
operators, book-level sensitivity flow, discontinuity policy, matrix coverage,
and runtime reporting that consume those objects.

| Queue ID | Linear | Lane | Status | Implementation objective | Hard prerequisites |
| --- | --- | --- | --- | --- | --- |
| `INT.1` | `QUA-954` | Calibration | Done | basket-credit base-correlation / tranche-correlation workflow consuming calibrated single-name curves | `CAL.4` / `QUA-953` |
| `INT.2` | `QUA-967` | Autograd | Done | truthful JVP, VJP, and HVP backend operator support or a checked fail-closed backend decision | `QUA-957`, `QUA-965` |
| `INT.3` | `QUA-956` | Validation | Done | first desk-like fixture, perturbation, and latency tranche for the newly supported calibration slices | none; run after `INT.1` or alongside active implementation slices |
| `INT.4` | `QUA-968` | Autograd | Done | first bounded book-level reverse-mode / portfolio AAD substrate over supported smooth routes | `INT.2` |
| `INT.5` | `QUA-969` | Autograd | Done | governed discontinuous-Greek policy for one bounded barrier, digital, or event/exercise family | `QUA-957`; coordinate with `INT.7` |
| `INT.6` | `QUA-970` | Autograd | Done | product-family derivative matrix covering analytical, curve, surface, MC, and calibration representatives | `QUA-957`; consume `INT.2` / `INT.5` outcomes |
| `INT.7` | `QUA-971` | Autograd | Done | unified runtime derivative-method reporting across analytical, AD, AAD, JVP/VJP/HVP, bump, smoothed/custom-adjoint, and unsupported lanes | `INT.2`, `INT.6`; coordinate with `INT.4` / `INT.5` |
| `INT.8` | `QUA-955` | Calibration | Done | bounded rates + equity/FX quanto-correlation slice on explicit dependency DAGs, using calibrated market objects and existing derivative provenance where useful | `QUA-950`, `QUA-951`, `QUA-971` |
| `INT.9` | `QUA-972` | Validation | Done | desk-like bounded quanto calibration fixtures, perturbation diagnostics, replay coverage, and latency envelopes for the shipped hybrid slice | `INT.8` |
| `INT.10` | `QUA-973` | Autograd | Done | bounded hybrid derivative-matrix row and runtime/reporting governance for the shipped quanto slice | `INT.8`; consume `INT.4`, `INT.6`, `INT.7` |
| `INT.11` | `QUA-946` | Closeout | Done | umbrella cleanup, docs maintenance, plan reconciliation, and follow-on ticket split | `INT.9`, `INT.10` landed or explicitly deferred in closeout |

### Pickup Rule

- start with `CAL.0A`, `CAL.0B`, or `CAL.0C`
- do not start `CAL.1` before `CAL.0B` closes
- do not start `CAL.3` before `CAL.2` closes
- do not start `CAL.4` before `CAL.0C` closes
- do not start `CAL.5` before `CAL.4` closes
- do not start `CAL.6` until the first supported hybrid slice and its concrete
  upstream blockers are explicit in the ticket
- `INT.9` / `QUA-972` and `INT.10` / `QUA-973` both landed after `INT.8`
  and completed the bounded hybrid validation and derivative-governance
  follow-ons
- `INT.11` / `QUA-946` closes only after those follow-ons land and the
  umbrella closeout restores the persisted benchmark artifact to the default
  smoothed baseline (`repeats=3`, `warmups=1`)
- keep `CAL.7` moving alongside the active implementation slices so validation
  does not become a deferred cleanup bucket
- do not start `AD2.2` before `AD2.1` lands a truthful backend operator
  decision
- do not start `AD2.5` before `AD2.1` and `AD2.4` define the operator and
  matrix vocabulary that runtime reporting must expose
- do not use Autograd Phase 2 as a substitute for missing calibration plants;
  derivative work consumes calibrated curves, surfaces, cubes, and correlation
  objects rather than rebuilding them privately

## Purpose

This document defines a bounded plan for upgrading the Trellis calibration
sleeve from its current proving-and-replay grade into something closer to a
mature industrial desk standard.

The calibration sleeve is treated here as a Trellis subsystem, not as a
separate library. It must work through and strengthen existing Trellis
abstractions such as:

- `SemanticContract` and `ValuationContext`
- `MarketBindingSpec` and `RequiredDataSpec`
- `MarketState`
- `trellis.models.calibration.quote_maps`
- `trellis.models.calibration.solve_request`
- `trellis.models.calibration.materialization`
- canonical model-grammar and planner surfaces

The plan starts with the current checked calibration layer and then widens the
scan in the order requested:

1. equity volatility
2. yield curve and multi-curve rates
3. yield-vol models such as SABR and short-rate fits
4. credit curve
5. basket credit and correlation
6. higher-order and cross-asset calibration

This is a planning document, not a claim that the repo already supports the
full industrial target.

## Repo-Grounded Decision Summary

The current repo has a real typed calibration substrate:

- `trellis/models/calibration/solve_request.py`
- `trellis/models/calibration/quote_maps.py`
- `trellis/models/calibration/materialization.py`
- `docs/mathematical/calibration.rst`

That substrate is good enough for bounded replayable workflows, but the
calibration sleeve is still narrow by desk standards.

The main current-state split is:

- shipped calibration workflows exist for bootstrap, flat rates vol, SABR
  single smile, Heston single smile, Dupire local vol, and bounded single-name
  credit
- pricing and market-resolution helpers exist for basket credit, quanto, FX,
  and some exotic/hybrid runtime surfaces
- joint or higher-order calibration workflows are mostly absent
- several of the shipped workflows are still proving-grade approximations
  rather than production-grade market calibrators

The plan should therefore focus on two things at once:

1. fixing correctness and contract weaknesses inside the shipped workflows
2. widening the calibration inventory in the order that a desk would actually
   need it

## End-State Architecture

The target calibration sleeve should be organized as one Trellis-native
three-layer stack.

### Layer 1: Market reconstruction

The first job of the sleeve is to reconstruct liquid, arbitrage-aware market
objects from elementary products.

Representative outputs:

- discount and forecast curves
- basis and collateral-aware rates structures
- equity, FX, cap/floor, and swaption vol surfaces or cubes
- single-name credit curves
- basket-credit correlation or base-correlation surfaces

For many liquid products, this layer is the authoritative result. The
calibration does not need to end in a reduced-form model parameter vector if
the calibrated curve or surface is already the market object that downstream
pricing should consume.

### Layer 2: Model compression

The second job of the sleeve is to fit tractable pricing models to the market
objects from Layer 1 when Trellis needs reduced models for pricing, simulation,
or risk.

Representative outputs:

- SABR parameter packs fitted to rates-vol surfaces
- short-rate parameter sets fitted to rates curve and vol objects
- local-vol or Heston parameterizations fitted to equity-vol objects
- reduced-form credit parameter packs fitted to credit curves
- copula or factor model parameters fitted to correlation objects

This layer should prefer fitting to Trellis-calibrated market objects where the
market standard supports that decomposition, instead of forcing every workflow
to jump directly from raw quotes to final reduced-model parameters.

### Layer 3: Hybrid composition

The third job of the sleeve is to combine already-calibrated single-asset
objects into cross-asset or higher-order systems.

Representative outputs:

- SPX plus variance or SPX plus VIX state models
- FX plus rates domestic/foreign joint setups
- rates plus equity quanto or hybrid discounting setups
- credit plus equity or credit plus rates hybrids

This layer should be dependency-aware. It must consume calibrated curves,
surfaces, credit objects, and correlation objects produced by the first two
layers rather than bypassing them with one monolithic direct fit.

## Trellis-Native Design Constraints

The calibration sleeve should not evolve into an independent sidecar engine.
The end-state design must satisfy these constraints.

### 1. Calibration outputs are Trellis runtime capabilities

Calibrated curves, surfaces, and parameter packs must materialize back onto
`MarketState` and be consumed through existing runtime capability lookups
instead of through route-local payloads.

### 2. Quote semantics remain first-class

Quote families and conventions must continue to flow through
`trellis.models.calibration.quote_maps`, not through per-model custom logic.

### 3. Calibration planning reuses Trellis semantic and binding layers

Where calibration is required before pricing, the planning surface should use
the same Trellis contract and binding abstractions rather than inventing a
parallel calibration DSL.

### 4. Chained calibrations are explicit

Curve builds, surface builds, model fits, and hybrid fits should form explicit
calibration dependency chains or DAGs rather than being hidden in helper-local
control flow.

### 5. Market reconstruction is not second-class

Curve, surface, and correlation object reconstruction should be treated as a
core calibration product of Trellis, not only as a precursor to later model
fits.

## Relationship To Existing Calibration Work

This plan extends, rather than replaces, the completed model-grammar and typed
materialization work captured in:

- `doc/plan/done__calibration-layer-model-grammar-unification.md`
- `doc/plan/draft__calibration-documentation-and-architecture-alignment.md`

That earlier work made calibration routes typed and replayable. This plan is
about raising the numerical, market-data, and cross-asset standard of those
routes.

## Repo-Grounded Current State

### 1. Equity volatility

Current checked workflow surface:

- `trellis/models/calibration/heston_fit.py`
- `trellis/models/calibration/local_vol.py`
- `trellis/models/calibration/implied_vol.py`
- synthetic benchmark inputs in `trellis/models/calibration/benchmarking.py`

What is actually shipped:

- explicit quote-governance on the observed equity-vol grid before model repair,
  with raw-versus-cleaned provenance and node-level adjustment diagnostics
- repaired multi-expiry equity-vol surface authority from per-expiry raw-SVI
  smiles with smile-level and calendar-level no-arbitrage diagnostics
- single-expiry Heston smile calibration
- full-surface Heston compression from the repaired equity-vol authority
- staged comparison helper between repaired-surface authority and one Heston
  compression fit, plus full-surface stage comparison
- Dupire local-vol extraction from an implied-vol grid
- local-vol extraction from the repaired equity-vol authority
- typed solve-request, provenance, replay, and materialization support

What is not yet at desk standard:

- the new surface authority now includes a bounded quote-governance pass, but
  it is still a raw-grid local-outlier cleaner rather than a full bid/ask,
  liquidity, staleness, or exchange-convention governance stack
- the surface authority is still a bounded raw-SVI lane rather than a full
  desk SSVI or broader volatility-governance stack
- Heston now compresses the repaired surface across the full grid, but only
  through one global parameter pack rather than a time-dependent term
  structure, richer loss surface, or stochastic-local-vol bridge
- the Heston objective uses a large sentinel vol on pricing/inversion failure
  instead of a governed failure surface or robust loss
- local-vol can now consume the repaired surface, but the checked path still
  relies on bounded sampled-grid Dupire extraction rather than a full
  arbitrage-repaired price-surface plant
- no stochastic-local-vol bridge is present
- no joint SPX plus variance or SPX plus VIX calibration workflow is present
- equity variance swap pricing exists, but no variance-surface or VIX-style
  calibration workflow was found under `trellis/models/calibration/`

Industrial implication:

- Trellis now has a bounded market-object-first equity-vol surface slice with
  explicit quote governance, repaired-surface authority, and staged
  model-compression comparison
- it still does not yet have a production smile-surface plant

### 2. Yield curve and multi-curve rates

Current checked workflow surface:

- `trellis/curves/bootstrap.py`
- `trellis/models/calibration/rates.py`
- `trellis/core/market_state.py`

What is actually shipped:

- typed bootstrap inputs for deposit, future, and swap quotes
- dated bootstrap inputs for deposits, futures, and swaps with schedule-aware
  accrual generation
- explicit multi-curve bootstrap program support with dependency order and
  dependency-graph payloads
- explicit multi-curve role provenance through selected discount and forecast
  curve names
- typed rates quote maps and materialization onto `MarketState`

What is not yet at desk standard:

- the legacy bootstrap path still works on year-fraction tenors, and even the
  new dated path is still bounded to one first dependency-aware program rather
  than a universal dated bootstrap plant
- the dated path now supports stub-aware schedules and optional business-day
  handling, but it still does not cover IMM logic, turn handling, or a full
  exchange-grade futures and calendar stack
- futures are still handled without exchange-grade convexity adjustments
- there is now a visible chained OIS-then-forecast calibration DAG, but not a
  broader basis or cross-currency dependency plant
- no explicit smoothing or regularized curve family is present beyond the
  current differentiable least-squares setup
- no cross-currency or collateral-aware multi-curve calibration program was
  found in the calibration sleeve

Industrial implication:

- Trellis now has a first dated, dependency-aware multi-curve bootstrap lane
  instead of only parallel tenor-only bundles
- the actual bootstrap engine still remains materially simplified versus a
  full desk curve-construction plant

### 3. Yield-vol models

Current checked workflow surface:

- `trellis/models/calibration/rates.py`
- `trellis/models/calibration/rates_vol_surface.py`
- `trellis/models/calibration/sabr_fit.py`

What is actually shipped:

- cap/floor flat Black-vol inversion
- swaption flat Black-vol inversion
- bounded caplet stripping into a reusable caplet-vol surface
- bounded tenor-aware swaption cube assembly with runtime materialization onto
  `MarketState.vol_surface`
- staged SABR compression over expiry-tenor swaption-cube slices
- SABR single-smile fit
- one strip-level Hull-White fit for a constant `(mean_reversion, sigma)` pair

What is not yet at desk standard:

- caplet stripping still assumes a one-step ladder of cap quotes and does not
  yet cover the broader desk caplet-bootstrap and normal/shifted-vol surface
  plant
- the swaption cube is still bounded to a rectangular absolute-strike grid and
  does not yet provide full desk-style relative-moneyness, arbitrage-repair,
  or bid/ask governance
- SABR is now cube-slice aware, but still fits each expiry-tenor slice
  independently rather than through a smooth arbitrage-aware global cube
- the supported Hull-White fit is constant-parameter across the strip, not a
  time-dependent short-rate calibration
- no G2++, LMM, displaced-diffusion, or normal-vol calibration workflow was
  found as a first-class checked route
- no evident co-calibration loop yet ties the rates-vol authority to a richer
  short-rate or market-model family beyond the bounded Hull-White strip

Industrial implication:

- the rates-vol sleeve now has a first market-object-first layer plus staged
  SABR comparison diagnostics
- it still does not yet cover the broader cube-governance, model-selection, or
  term-structure problems that matter on a desk

### 4. Credit curve

Current checked workflow surface:

- `trellis/models/calibration/credit.py`
- `trellis/curves/credit_curve.py`
- `docs/mathematical/calibration.rst`

What is actually shipped:

- typed single-name reduced-form credit calibration inputs
- spread, upfront, and hazard quote maps
- CDS-pricer-backed schedule-aware normalization across running-spread and
  standard-coupon-plus-upfront quote styles
- repricing, survival-probability, forward-hazard, and hazard-governance
  diagnostics
- credit-curve materialization back onto `MarketState`

What is not yet at desk standard:

- the current workflow is still a bounded single-name strip calibration rather
  than a smoothed or regularized production CDS curve plant
- no bid/ask governance, index-credit conventions, bond/CDS basis, or broader
  curve-policy surface was found
- no structural credit or hybrid credit-equity workflow was found

Industrial implication:

- this is now a typed CDS-pricer-backed single-name CDS strip calibration slice
  with quote-style handling, schedule provenance, and governed diagnostics
- it is still not yet a production CDS bootstrap or broader credit calibration
  engine

### 5. Basket credit and correlation

Current checked pricing and helper surface:

- `trellis/models/credit_basket_copula.py`
- `trellis/instruments/nth_to_default.py`
- `docs/mathematical/copulas.rst`

What is actually shipped:

- Gaussian and Student-t copula pricing helpers
- nth-to-default and tranche-style pricing support
- correlation-aware runtime helper surfaces
- documentation that explains tranche loss and mentions base correlation

What was not found:

- no basket-credit calibration workflow under `trellis/models/calibration/`
- no base-correlation surface calibration routine
- no tranche-implied correlation or factor-loading calibrator
- no joint calibration tying single-name curves to tranche quotes
- no calibration governance for loss-surface smoothing or tranche arbitrage

Industrial implication:

- Trellis can price bounded basket-credit structures
- Trellis does not yet calibrate basket-credit correlation surfaces

### 6. Higher-order and cross-asset calibration

Current checked pricing and resolver surface:

- `trellis/models/resolution/quanto.py`
- `trellis/models/quanto_option.py`
- `trellis/models/fx_vanilla.py`
- correlation-resolution logic in `trellis/data/resolver.py`
- design notes in `docs/design_quanto_runtime_contract.md`

What is actually shipped:

- a narrow single-underlier quanto pricing slice
- explicit runtime binding for FX vol and underlier/FX correlation
- FX vanilla pricing helpers
- correlation provenance and empirical-correlation resolution support

What was not found as first-class calibration workflows:

- no SPX plus VIX calibration route
- no FX plus rates hybrid calibration route
- no rates plus equity hybrid calibration route
- no generic joint-factor or cross-asset calibration DAG
- no stochastic-local-vol or hybrid affine workflow under
  `trellis/models/calibration/`

Industrial implication:

- Trellis has some good runtime ingredients for cross-asset pricing
- it does not yet have a real hybrid calibration sleeve

## Gap Taxonomy Against A Mature Industrial Standard

### A. Market-data conditioning is too thin

Industrial sleeves normally include:

- quote cleaning
- stale or crossed quote detection
- arbitrage repair
- robust weighting and liquidity weighting
- convention-specific normalization

Trellis currently emphasizes typed quote semantics and replayability, but not
yet a strong market-data conditioning layer.

### B. The calibration inventory is too narrow

The repo contains several bounded workflows, but a mature stack needs:

- equity smile and surface parameterizations
- rates curve ladders and basis dependencies
- rates-vol cube handling
- true CDS curve bootstrap
- basket-credit correlation surfaces
- hybrid and cross-asset linkage

### C. The solver layer is typed but still shallow

The current solve substrate is useful, but industrial desks typically need:

- constraints beyond box bounds
- regularization terms
- robust losses
- multi-start or global-search support
- calibration dependency graphs
- parameter freezing and staged solves

### D. Diagnostics are good for replay, weaker for desk review

The repo records provenance and residuals well, but a mature sleeve also needs:

- parameter stability diagnostics
- sensitivity to quote perturbations
- condition and identifiability diagnostics at the model level
- bad-point attribution and quote exclusion policy
- comparative model-fit reporting across candidate models

### E. Validation remains synthetic-heavy

The current replay and benchmark surface is useful, but industrial standards
usually require:

- noisy and imperfect fixtures
- historical backfill or golden-market snapshots
- stress calibration cases
- comparative model-to-model and model-to-market regression packs
- latency budgets on realistic instrument counts

### F. Documentation occasionally overstates the typed workflow relative to the numerical depth

The repo documents the bounded workflows clearly in many places, but there are
still areas where the typed calibration surface looks stronger than the actual
mathematical content. The credit slice is the clearest example.

### G. Documentation and architecture framing are not yet fully aligned

The mathematical framework is already broader and better than the shorthand
"general multivariate SDE setup" suggests, but that framing is still loose in
practice.

The end-state docs should say clearly that:

- the top-level unifier is not one master SDE
- the sleeve is Trellis-native rather than a separate calibration engine
- market reconstruction, model compression, and hybrid composition are distinct
  layers
- quote maps, market binding, and materialization are core Trellis abstractions
  rather than implementation details

## Ordered Work Program

### Phase 0: Architectural and correctness alignment

1. Reframe the calibration sleeve explicitly as a Trellis-native market
   inference layer organized into market reconstruction, model compression, and
   hybrid composition.
2. Update the documentation plan so the mathematical framework, calibration
   docs, and developer docs all describe the same end state.
3. Fix equity-vol carry consistency across pricing and implied-vol inversion.
4. Replace the single-name credit identity solve with a real CDS pricer-backed
   objective.
5. Tighten docs and model-grammar registry language so pricing support is not
   mistaken for calibration support.

Acceptance bar:

- the end-state architecture is documented as Trellis-native rather than as a
  separate library or one master-SDE engine
- Heston and local-vol carry assumptions are internally consistent
- credit calibration reprices CDS-style quotes through a pricing engine rather
  than a direct transform
- docs and benchmark notes clearly distinguish priced-only from calibrated
  surfaces

### Phase 1: Industrial equity-vol foundation

1. Add a governed surface-cleaning layer for equity option quotes.
2. Add an arbitrage-aware parameterized smile or surface family, likely SVI or
   SSVI, as the first production surface authority.
3. Rebuild local-vol extraction off an arbitrage-repaired price or total-variance
   surface rather than raw implied-vol spline interpolation.
4. Widen Heston from single-smile to a controlled term-structure or surface fit.
5. Add staged calibration support so surface-fit and model-fit layers can be
   compared instead of conflated.

Acceptance bar:

- one desk-style SPX surface fixture with stable no-arb diagnostics
- stable parameter fits under small quote perturbations
- explicit comparison between surface-only and model-based fits

### Phase 2: Rates curve and multi-curve hardening

1. Upgrade bootstrap inputs from tenor-only abstractions toward dated
   instrument schedules.
2. Add explicit OIS-discount then forecast then basis dependency handling.
3. Add schedule-aware futures and swap conventions, including business-day and
   stub handling where required.
4. Add regularized curve families or smoothing choices where the current raw
   least-squares formulation is too unstable.
5. Make chained calibration dependencies a first-class contract.

Acceptance bar:

- one realistic USD OIS plus SOFR forecast calibration fixture
- explicit dependency graph and replay artifact for chained curve builds
- stable repricing under realistic quoted instruments and conventions

### Phase 3: Rates-vol surface and model program

1. Add caplet stripping and swaption-cube support.
2. Widen SABR from single smile to expiry-tenor cube with interpolation policy.
3. Add time-dependent short-rate calibration or a second rates model family
   beyond constant-parameter Hull-White.
4. Support model-to-surface and surface-to-model comparison diagnostics.

Acceptance bar:

- one checked swaption cube fixture
- clear fit diagnostics by expiry and tenor
- downstream rates models consume calibrated outputs without hidden
  reconvention

### Phase 4: Real single-name credit calibration

1. Implement schedule-aware CDS quote normalization and pricing.
2. Support standard running plus upfront quote styles.
3. Add hazard smoothing or bootstrap policy with diagnostics on survival and
   forward hazard behavior.
4. Surface recovery assumptions and sensitivity in the calibration result.

Acceptance bar:

- one realistic CDS tenor strip fixture repriced through the CDS pricer
- quote-style coverage for spread and upfront conventions
- diagnostics on survival monotonicity and hazard reasonableness

### Phase 5: Basket credit and correlation calibration

1. Add base-correlation or tranche-implied correlation workflow support.
2. Add a calibration contract that ties single-name curves, tranche quotes, and
   correlation surface outputs together.
3. Add smoothing and monotonicity governance for tranche or base-correlation
   surfaces.
4. Separate homogeneous proving cases from heterogeneous portfolio cases.

Acceptance bar:

- one tranche surface fixture with reproducible implied-correlation outputs
- explicit linkage from single-name curve inputs to basket-correlation outputs
- diagnostics on tranche arbitrage and surface smoothness

### Phase 6: Higher-order and cross-asset calibration

1. Introduce a calibration dependency DAG for chained or joint calibrations.
2. Start with one narrow but real hybrid slice rather than a universal hybrid
   engine.
3. Candidate first slices:
   - SPX plus variance or SPX plus VIX
   - FX plus rates with consistent domestic and foreign curve plus vol binding
   - rates plus equity quanto or hybrid discounting cases
4. Add explicit cross-asset state and correlation parameter materialization.

Acceptance bar:

- one checked cross-asset calibration slice with replay
- explicit dependency and provenance packet across all linked calibrations
- honest failure semantics when required hybrid inputs are missing

### Phase 7: Validation and benchmark hardening

1. Keep the current typed replay pack, but add noisy and desk-like fixtures.
2. Add latency benchmarks at realistic instrument counts.
3. Add perturbation tests so parameter instability is measured rather than
   guessed.
4. Add comparative model packs for at least one asset class per quarter.
5. Keep the doc, plan, and runtime surfaces aligned as the calibration sleeve
   widens so architectural drift does not reappear.

Acceptance bar:

- benchmark pack includes realistic and synthetic fixtures
- replay captures numerical drift and model-instability regressions
- calibration latency and convergence envelopes are explicit

## Recommended Build Order

The completed implementation order through `CAL.6`, plus the parallel
validation and derivative-governance slices, was:

1. Phase 0 correctness fixes
2. Phase 1 equity-vol foundation
3. Phase 2 rates curve and multi-curve hardening
4. Phase 3 rates-vol program
5. Phase 4 real single-name credit calibration
6. Phase 5 bounded homogeneous basket-credit tranche-correlation calibration
7. First desk-like calibration validation tranche
8. First bounded discontinuous Monte Carlo derivative policy
9. Unified runtime derivative-method taxonomy and reporting
10. First bounded hybrid cross-asset calibration slice

The final integrated implementation order was:

1. `QUA-972` bounded hybrid validation tranche over the shipped quanto slice
2. `QUA-973` bounded hybrid derivative-governance tranche over the same route
3. `QUA-946` umbrella closeout and documentation maintenance once those
   follow-ons landed

`QUA-967` landed early as `AD2.1` / `INT.2`, and `QUA-968` consumed that
checked VJP/HVP surface for the first bounded bond-book reverse-mode lane.
`QUA-956` added the first desk-like basket-credit validation tranche with
perturbation diagnostics and latency-envelope metadata. `QUA-969` added the
first bounded discontinuous Monte Carlo derivative policy, keeping barrier and
event discontinuities fail-closed for pathwise AD while reporting the fallback
method explicitly. `QUA-970` added the checked product-family derivative matrix
covering representative analytical, curve, surface, Monte Carlo, calibration,
route-generated, and unsupported discontinuous lanes. `QUA-971` normalized
runtime derivative-method reporting across analytical, AD, AAD, VJP/HVP-backed,
bump, fail-closed, and future smoothed/custom-adjoint lanes.

Reason:

- equity vol is the fastest path to a visibly stronger desk-standard slice
- rates and rates-vol need deeper infrastructure but are central to the broader
  stack
- credit should not widen into basket and hybrid work until the single-name
  slice is mathematically real
- portfolio AAD and richer derivative reporting need truthful backend hooks
  before they can claim throughput or method-selection support
- discontinuous products need explicit policy before they can appear in a
  support matrix or runtime report as anything other than unsupported or bump
- `QUA-971` normalized runtime derivative-method reporting across analytical,
  AD, AAD, JVP/VJP/HVP, bump, fail-closed, and future smoothed/custom-adjoint
  lanes
- cross-asset calibration should be introduced only after at least one strong
  surface exists in each linked asset class and the first derivative-governance
  vocabulary is stable enough to report calibration and risk provenance

## Explicit Non-Goals For The First Tranche

This plan does not propose immediately building:

- a universal calibration engine for all models
- rough-vol calibration
- generic HJM or SPDE calibration
- all hybrid models at once
- every correlation product family simultaneously

The right first industrialization move is to harden a few narrow slices until
they are actually desk-grade, then widen.

## Closeout Result

The final execution slice after the `CAL.6` / `QUA-955` closeout was:

1. `QUA-972`: the first desk-like bounded hybrid fixture pack, replay
   coverage, perturbation diagnostics, and latency envelopes for the shipped
   quanto-correlation route
2. `QUA-973`: the bounded hybrid derivative-matrix and runtime-reporting
   contract on the same shipped route without overstating AD/AAD support
3. `QUA-946`: umbrella closeout once those two follow-ons landed, including
   benchmark-artifact stabilization back to the default smoothed baseline

`QUA-972` and `QUA-973` ran in parallel once the `QUA-955` input schema,
materialization payload, and runtime-reporting surface were stable enough to
treat as the shipped bounded hybrid contract.
