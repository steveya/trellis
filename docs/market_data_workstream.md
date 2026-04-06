# Market Data Workstream

This note defines the market-data roadmap for Trellis and records the first
completed slice, `MD1`.

## Goal

Trellis should treat market data as a first-class subsystem with the same
discipline now used for products and routes:

- typed structure
- deterministic resolution
- explicit provenance
- clean compilation into runtime pricing objects

The end state is:

`connector/raw quotes -> canonical market snapshot -> calibrated market objects -> MarketState -> pricing / greeks / analytics`

## Current State

As of `MD1`, Trellis already has:

- discount curves in [yield_curve.py](/Users/steveyang/Projects/steveya/trellis/trellis/curves/yield_curve.py)
- forward extraction in [forward_curve.py](/Users/steveyang/Projects/steveya/trellis/trellis/curves/forward_curve.py)
- credit curves in [credit_curve.py](/Users/steveyang/Projects/steveya/trellis/trellis/curves/credit_curve.py)
- a volatility-surface protocol in [vol_surface.py](/Users/steveyang/Projects/steveya/trellis/trellis/models/vol_surface.py)
- runtime pricing state in [market_state.py](/Users/steveyang/Projects/steveya/trellis/trellis/core/market_state.py)
- a simple discount-curve resolver in [resolver.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/resolver.py)

What it did not have before `MD1` was a canonical market snapshot layer.

As of `MD4`, Trellis also has a full simulated snapshot path via
`source="mock"`:

- named discount curves
- named forecast curves
- named vol surfaces
- named credit curves
- FX spot quotes
- named underlier spots
- synthetic local-vol surfaces
- synthetic jump/model-parameter packs

Those mock snapshots now carry explicit synthetic-prior provenance, including
the prior family, a stable seed, and the parameterization used to build the
regime bundle.

`QUA-693` first added a bounded descriptive packet to that synthetic path:
``prior_parameters.model_consistency_contract``. That compatibility contract
records the deterministic rates, credit, and volatility assumptions used to
build the mock snapshot, including:

- explicit discount/forecast curve roles plus forecast-basis inputs
- reduced-form credit spread grids, recovery, and the aligned credit workflow
- volatility/model-parameter families and their runtime materialization targets

`QUA-695` adds the new seeded authority surface underneath that compatibility
layer: ``prior_parameters.synthetic_generation_contract``. This contract is the
mock-path generator boundary and explicitly separates:

- seeded model packs
- synthetic quote bundles
- runtime target names

The older ``model_consistency_contract`` is now derived from the seeded
generation contract so existing replay, benchmark, and proving consumers
continue to work while the follow-on family-specific generators migrate onto
the new authority surface.

Both contracts are synthetic metadata only. They exist so proving and demo runs
can explain which bounded model assumptions were used, and they should not be
treated as production market data.

`QUA-692` wires that same contract into calibration hardening as well: the
supported single-name credit benchmark fixture now reads spread/recovery inputs
from ``prior_parameters.model_consistency_contract`` and replays calibration on
the same bounded synthetic assumptions used by mock/proving runs.

Basket and quanto correlation now follow the same provenance discipline:

- explicit correlation matrices and scalar correlation inputs are traced as
  explicit sources
- correlation estimated from historical paths is traced as empirical input
- implied and synthetic correlation sources retain their source family, sample
  size, estimator, seed, and any regularization performed before pricing

That simulated provider is the current stand-in for missing live connectors.

## MD1

`MD1` introduces:

- [schema.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/schema.py)
  - `MarketSnapshot`
- [resolver.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/resolver.py)
  - `resolve_market_snapshot(...)`
  - `resolve_curve(...)` preserved as a compatibility wrapper

`MarketSnapshot` is the market-data analogue of `ProductIR`:

- it is richer than `MarketState`
- it holds named components plus provenance
- it compiles into `MarketState` for runtime pricing

`MD1` is intentionally narrow:

- only the discount-curve path is auto-resolved today
- vol, credit, forecast, and FX components can be carried in the snapshot but
  are not yet fetched from live providers by the generalized resolver

## MD2

`MD2` extends the snapshot layer into the execution surface.

Completed in this phase:

- [session.py](/Users/steveyang/Projects/steveya/trellis/trellis/session.py)
  - `Session` can now be built from `market_snapshot=...`
  - `Session.with_discount_curve(name)` can switch named discount curves
  - snapshot-backed sessions preserve surrounding forecast/vol/FX context when
    discount curves are shifted or replaced
- [pipeline.py](/Users/steveyang/Projects/steveya/trellis/trellis/pipeline.py)
  - `Pipeline.market_data(snapshot=..., discount_curve=...)`
  - batch pricing can now run directly from a named market snapshot

This phase keeps the old `curve=` API intact. The migration path is additive:

- old path: `Session(curve=...)`
- new path: `Session(market_snapshot=..., discount_curve="usd_ois")`

What `MD2` does not do yet:

- resolve named forecast curves from live providers
- resolve live vol/credit/FX snapshots
- make product/request requirements drive automatic component selection

## Phased Roadmap

### MD1: Canonical Snapshot Schema

Status: completed

- add `MarketSnapshot`
- add generalized resolver API
- preserve `resolve_curve(...)`
- keep compatibility with existing `Session` behavior

### MD2: Rates Stack

Status: partial foundation completed

Goal:

- support named discount and forecast curves by currency/index
- compile minimal required curve sets from product/request needs
- stop treating the default curve as the whole market-data story

Planned outputs:

- curve-set schema
- multi-curve resolver
- tighter integration with `Session` and `Pipeline`

Remaining work for `MD2`:

- generalized rates connector inputs, not just a single resolved discount curve
- request-driven named curve selection
- richer rate/curve provenance and diagnostics

### MD3: Volatility Surface Stack

Status: partial foundation completed

Goal:

- move beyond `FlatVol`
- support named surfaces by underlier/family
- separate raw quotes from surface objects

Completed in this phase:

- [vol_surface.py](/Users/steveyang/Projects/steveya/trellis/trellis/models/vol_surface.py)
  - `GridVolSurface`
- [resolver.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/resolver.py)
  - named `vol_surfaces` support
- [session.py](/Users/steveyang/Projects/steveya/trellis/trellis/session.py)
  - `vol_surface_name`
  - `with_vol_surface_name(...)`
- [pipeline.py](/Users/steveyang/Projects/steveya/trellis/trellis/pipeline.py)
  - snapshot-driven named vol-surface selection

Planned outputs:

- surface quote schema
- surface builder/calibration layer
- richer `VolSurface` implementations

Remaining work for `MD3`:

- live connector support for vol quotes/surfaces
- quote-grid schema and calibration/build pipeline
- richer smile/skew conventions and underlier naming

### MD4: Credit / FX / Fixings

Status: partial foundation completed

Goal:

- make non-rates market data first-class in the same framework

Completed in this phase:

- [mock.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/mock.py)
  - `fetch_market_snapshot(...)`
  - simulated discount, forecast, vol, credit, and FX components derived from
    embedded market regimes
- [resolver.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/resolver.py)
  - explicit market-component overrides now merge with provider snapshots
- [session.py](/Users/steveyang/Projects/steveya/trellis/trellis/session.py)
  - `data_source="mock"` now loads full snapshot context automatically
- [schema.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/schema.py)
  - `MarketSnapshot` and `MarketState` now carry generic underlier spots,
    local-vol surfaces, jump-parameter sets, and model-parameter sets
- [capabilities.py](/Users/steveyang/Projects/steveya/trellis/trellis/core/capabilities.py)
  - market-data aliases now bridge legacy task names like `discount_curve`,
    `yield_curve`, `risk_free_curve`, `forward_rate_curve`,
    `volatility_surface`, and `black_vol_surface` onto the modern runtime
    capability model

Planned outputs:

- named credit curves
- FX spot and forward data integration
- fixing/rate history support where products need it

Remaining work for `MD4`:

- fixing/rate-history schema
- FX forward quote or curve-set inputs
- live credit / FX / fixing connectors
- live spot/local-vol/jump/model-parameter providers instead of synthetic packs

### Task-Level Mock Connector Coverage

The task runner now supports task-local market configuration on top of the mock
snapshot path:

- `market:` chooses named snapshot components such as `discount_curve`,
  `forecast_curve`, `vol_surface`, `underlier_spot`, `credit_curve`, `fx_rate`,
  `local_vol_surface`, `jump_parameters`, and `model_parameters`
- `market_assertions:` checks required capabilities and selected component names
  before the build runs
- task results now record market provenance and selected component names

This is the bridge between "mock snapshot exists" and "tasks genuinely exercise
the connector path with explicit market intent."

### MD5: Direct Connector and Book Integration

Goal:

- let external books and position feeds compile directly into Trellis pricing
  requests and market snapshots without going through prompt parsing

Planned outputs:

- connector/adaptor contracts
- book ingestion mappings
- request-driven snapshot resolution

### MD6: Provenance, Validation, and Caching

Goal:

- make snapshots auditable and reproducible

Planned outputs:

- source and timestamp provenance
- source-kind labels for direct quotes vs bootstrapped inputs
- stale/missing-data warnings
- snapshot-level caching
- deterministic golden market fixtures for tests

`QUA-358` now lands the first explicit market-parameter sourcing branch on the
resolver path:

- ``resolve_market_snapshot(...)`` accepts ``model_parameter_sources`` with
  explicit ``source_kind`` declarations
- supported source kinds are currently:
  - ``direct_quote`` for quoted/provider parameter packs
  - ``bootstrap`` for deterministic curve-derived parameter packs
- bootstrap sources persist their entry contract under
  ``provenance.bootstrap_inputs.model_parameters``
- per-pack source metadata persists under
  ``provenance.market_parameter_sources``
- unsupported source kinds or mixed direct/bootstrap payloads fail closed with
  explicit validation errors rather than silently merging ambiguous inputs

## Future Track: Reactive Dataflow

This is explicitly a future design task.

Trellis should continue moving toward immutable reactive dataflow rather than a
QuantLib-style observable/observer graph.

The likely direction is:

- immutable `MarketSnapshot` and request objects
- derived `Session` / `MarketState` / analytics nodes
- explicit dependency tracking
- invalidation and recomputation by value replacement, not observer callbacks

The current scaffolding for that direction is:

- [session.py](/Users/steveyang/Projects/steveya/trellis/trellis/session.py)
- [pipeline.py](/Users/steveyang/Projects/steveya/trellis/trellis/pipeline.py)
- [market_state.py](/Users/steveyang/Projects/steveya/trellis/trellis/core/market_state.py)
- [state_space.py](/Users/steveyang/Projects/steveya/trellis/trellis/core/state_space.py)
- [bootstrap.py](/Users/steveyang/Projects/steveya/trellis/trellis/curves/bootstrap.py)
- [schema.py](/Users/steveyang/Projects/steveya/trellis/trellis/data/schema.py)

## Design Rules

- Keep raw quote connectors separate from calibrated market objects.
- Keep `MarketSnapshot` richer than `MarketState`.
- Make product/request requirements drive market-data resolution.
- Prefer named components over implicit global defaults.
- Preserve backward compatibility while migrating `Session` and `Pipeline`.
- Do not hide missing market data behind silent fallbacks.

## Immediate Next Step

The next real market-data phase is no longer plumbing. It is either:

- live quote/schema support for rates and vol inputs, or
- direct connector/book ingestion on top of the simulated snapshot path

For now, `source="mock"` is the canonical stand-in provider when no external
vendor is available.
