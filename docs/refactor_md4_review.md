# MD4 Review: Simulated Full Market Snapshots

## Review

Trellis already had the runtime surfaces needed for richer market data:

- `MarketSnapshot`
- snapshot-aware `Session`
- snapshot-aware `Pipeline`
- named discount curves
- named volatility surfaces

The missing piece was provider-side: `source="mock"` still only resolved a
single discount curve, so caps, credit products, FX flows, and multi-curve
rates examples still needed manual inputs.

## Plan

Build a deterministic simulated snapshot provider as the stand-in for missing
live connectors.

The stand-in should:

- derive the rates regime from the embedded yield snapshots
- emit named discount and forecast curves
- emit a named rate-vol surface set
- emit named credit curves
- emit FX spot quotes
- preserve explicit caller overrides on top of the provider snapshot

## Tests First

The red tests for this phase required:

- `MockDataProvider.fetch_market_snapshot(...)` returns a full named snapshot
- `resolve_market_snapshot(source="mock")` exposes the same richer context
- explicit resolver overrides merge with provider data rather than replacing it
- `Session(data_source="mock")` loads full snapshot context automatically
- a cap can be priced from mock source without manually supplying vol or
  forecast curves

## Implementation

Completed in this phase:

- `BaseDataProvider.fetch_market_snapshot(...)` default hook
- `MockDataProvider.fetch_market_snapshot(...)`
- richer simulated components for:
  - `usd_ois`, `eur_ois`, `gbp_ois`
  - forecast curves keyed by rate index
  - `usd_rates_atm`, `usd_rates_smile`
  - `usd_ig`, `usd_hy`
  - `EURUSD`, `GBPUSD`, `USDJPY`
- resolver overlay logic so explicit vol/credit/forecast/FX inputs can augment
  the provider snapshot

## Validation

Focused:

- `tests/test_data/test_mock.py`
- `tests/test_data/test_resolver.py`
- `tests/test_session.py`

Broader:

- `tests/test_data/test_market_snapshot.py`
- `tests/test_data/test_mock.py`
- `tests/test_data/test_resolver.py`
- `tests/test_models/test_vol_surface.py`
- `tests/test_session.py`
- `tests/test_pipeline.py`
- `tests/test_instruments/test_cap.py`

## Outcome

Trellis still does not have real external market-data connectors, but it now
has a coherent stand-in provider that exercises the full snapshot path end to
end. That is enough to keep the library and agent workflows moving while the
live connector and quote-schema phases remain deferred.
