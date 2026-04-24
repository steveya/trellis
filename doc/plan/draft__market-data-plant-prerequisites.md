# Full Market-Data Plant Prerequisites

## Status

Draft planning document. No Linear tickets have been created from this plan.

This document describes the missing data, governance, and computational
contracts required before Trellis can claim a full market-data plant.

## Current Shipped Boundary

Trellis currently has bounded market-data surfaces:

- `trellis.data.schema.MarketSnapshot`
- mock and file-backed snapshot paths in `trellis.data`
- runtime `MarketState` in `trellis.core.market_state`
- selected external adapters such as FRED, treasury, and Bloomberg-oriented
  modules
- synthetic benchmark fixtures for calibration and proving workflows
- calibration materialization back onto `MarketState`

The current system is enough for deterministic tests, synthetic fixtures,
bounded benchmark workflows, and explicit user-supplied inputs. It is not yet
a production market-data plant.

## Missing Gap Before Implementation

The full plant cannot be started until Trellis has a canonical market data
model that separates raw observations, normalized quotes, validated snapshots,
and calibrated runtime objects.

The missing prerequisite is:

`MarketDataObservationModel`: a typed observation and snapshot lifecycle that
tracks source, timestamp, convention, instrument identity, normalization,
quality state, lineage, and downstream calibration eligibility.

Without this model, ingestion, quote governance, stale data handling, replay,
and calibration fixtures will remain separate conventions in each adapter or
workflow.

## Data Lifecycle Contract

The plant should model the lifecycle explicitly:

```text
RawObservation
  -> NormalizedQuote
  -> ValidatedQuote
  -> MarketSnapshot
  -> CalibratedMarketObject
  -> Runtime MarketState
```

Each stage has a different contract.

`RawObservation`:

- source system
- raw ticker or instrument key
- observed value
- observed timestamp
- received timestamp
- raw quote type and units
- raw metadata

`NormalizedQuote`:

- canonical instrument identity
- canonical quote type
- canonical units
- market convention
- calendar and day count
- currency and collateral context
- source lineage

`ValidatedQuote`:

- quality status
- stale/outlier flags
- bid/ask or confidence weight
- replacement or fallback policy
- validation rule IDs

`MarketSnapshot`:

- as-of date/time
- coherent set of validated quotes
- curve, surface, credit, FX, and parameter input packets
- source lineage
- hash or revision ID

`CalibratedMarketObject`:

- output object type
- calibration problem and solve request
- diagnostics
- replay payload
- source quote lineage
- materialization destination

## Mathematical Contract

Raw market observations must be normalized before they enter calibration.

```text
y_j = N_j(x_j, convention_j, context_j)
```

Where:

- `x_j` is the raw source observation
- `N_j` is the convention-specific normalization map
- `y_j` is the canonical quote value

Validation assigns quality and weight:

```text
status_j, w_j = Q_j(y_j, history_j, cross_section_j, rules_j)
```

Calibration should consume validated quotes:

```text
r_i(theta) = w_i * (P_i(theta) - y_i)
```

For robust data handling:

```text
Phi(theta) = sum_i rho(r_i(theta)) + R(theta)
```

Where `rho` may be squared loss, Huber loss, or a fail-closed exclusion policy
depending on governance.

No-arbitrage and structural checks must be first-class constraints:

```text
D(t) > 0
D(t_2) <= D(t_1) for t_2 > t_1 under positive-rate assumptions when enabled
survival(t_2) <= survival(t_1)
total_variance(T, K) >= 0
calendar arbitrage checks for vol surfaces
correlation matrix positive semidefinite
```

The plant must distinguish:

- quote normalization errors
- stale data
- outliers
- missing mandatory observations
- optional observations excluded by policy
- calibration failures
- materialization failures

Those categories should not collapse into a generic exception.

## Computational Contract

Required core objects:

- `RawMarketObservation`
- `MarketInstrumentId`
- `QuoteConvention`
- `NormalizedMarketQuote`
- `QuoteQualityReport`
- `ValidatedMarketQuote`
- `MarketSnapshotRevision`
- `MarketDataLineage`
- `MarketDataPolicy`
- `MarketDataValidationRule`
- `CalibrationInputPacket`

Required services:

- source adapter interface
- quote normalizer registry
- instrument identity resolver
- calendar and convention resolver
- validation rule engine
- stale and outlier detector
- snapshot assembler
- snapshot diff and replay tools
- fixture export/import
- calibration input packet builder

The plant should feed calibration through packets, not through loosely shaped
dictionaries:

```text
CalibrationInputPacket:
  as_of
  quote_set_id
  validated_quotes
  required_market_objects
  policy
  lineage
```

Calibration workflows should report which packet and quote IDs they consumed.

## Governance Requirements

The plant needs policy before breadth.

Policy dimensions:

- source priority
- stale threshold by instrument family
- bid/ask midpoint rules
- quote side selection
- outlier thresholds
- holiday and calendar treatment
- fallback hierarchy
- mandatory versus optional quotes
- quote exclusion approval
- interpolation/extrapolation policy
- calibration eligibility
- benchmark fixture generation

Every exclusion should be explainable:

```text
excluded_quote = {
  quote_id,
  reason,
  rule_id,
  source_value,
  normalized_value,
  replacement,
  severity
}
```

## Product-Family Scope

The first real plant should not try every market at once. Suggested order:

1. rates deposits/OIS/swaps and curve pillars
2. cap/floor and swaption vol quotes
3. equity/FX vanilla option vol quotes
4. CDS quotes and recovery assumptions
5. basket-credit tranche quotes
6. bounded hybrid inputs: FX spot, underlier spot, vol surface, curve links,
   and scalar correlation quotes

Each family must define:

- instrument identity
- raw source mapping
- quote convention
- normalization
- validation rules
- calibration packet output
- replay fixture format

## Required Validation Before Production Claims

The plant should be considered usable only after it has:

- deterministic fixture snapshots
- raw-to-normalized quote tests
- convention tests for day counts, calendars, and units
- stale quote tests
- outlier tests
- missing mandatory quote tests
- source-priority tests
- snapshot hash/replay tests
- calibration packet tests
- end-to-end calibration from a validated snapshot for at least one family
- docs for each supported quote convention

For calibration linkage:

```text
snapshot -> packet -> calibration result -> MarketState
```

must be replayable with stable lineage.

## Implementation Phases

### Phase 1: Observation And Quote Model

Define raw observations, canonical quote identity, normalized quote payloads,
and quality reports.

Deliverables:

- schema dataclasses
- unit tests for serialization
- normalizer interface

### Phase 2: One Source Adapter And One Family

Pick one bounded family, likely rates curve pillars or CDS quotes. Implement
normalization and validation from deterministic fixture data.

Deliverables:

- source adapter fixture
- quote normalizer
- validation rules
- snapshot assembler

### Phase 3: Calibration Packet Builder

Build typed calibration input packets from validated quotes and route one
existing calibration workflow through the packet.

Deliverables:

- packet schema
- workflow adapter
- lineage tests

### Phase 4: Snapshot Replay And Diff

Add snapshot revision IDs, stable serialization, and diff tooling.

Deliverables:

- snapshot hash tests
- replay from fixture
- diff report for quote changes and validation changes

### Phase 5: Governance Expansion

Add stale/outlier policies, source priority, and exclusion reports. Expand to a
second product family.

Deliverables:

- policy docs
- validation rule matrix
- second family packet tests

## Explicit Non-Goals

- Do not call mock snapshots a full market-data plant.
- Do not hard-code vendor tickers inside calibration workflows.
- Do not let calibration silently consume raw source observations.
- Do not treat stale/outlier exclusions as ordinary missing data.
- Do not build live vendor connectivity before deterministic replay and
  governance are stable.

## Open Design Questions

- Should market data schemas live under `trellis.data` or be shared with
  runtime contracts under `trellis.core`?
- Should quote validation be rule-driven data, Python policy objects, or both?
- What is the stable identity format for market instruments across vendors?
- Should fixture snapshots be committed as JSON, YAML, or generated from typed
  Python fixtures?
- How much historical time-series support is needed before outlier governance
  is meaningful?

## First Ticket Shape

Suggested future ticket:

`Market data plant: canonical observation and validated quote model`

Acceptance criteria:

- raw observations and normalized quotes have typed schemas
- one product family can normalize deterministic fixture quotes
- validation reports stale/missing/outlier categories distinctly
- a calibration input packet can be built from validated quotes
- no live vendor integration is required in the first slice
