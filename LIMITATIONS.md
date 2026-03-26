# Known Limitations

Tracked issues to tackle proactively. Ordered by impact.

## Resolved

| # | Limitation | Resolution |
|---|-----------|-----------|
| L1 | ~~Trinomial trees not implemented~~ | `build_rate_lattice()` and `build_generic_lattice()` now support `branching=3`. HW trinomial with mean-reversion probabilities. |
| L4 | ~~Old CN/implicit FD still have coefficient bug~~ | `__init__.py` now imports from `theta_method_1d`. Old names are backward-compat wrappers. Capabilities updated. |
| L15 | ~~Critic/validator failures silently caught~~ | All `except Exception: pass` blocks in `_validate_build()` now log warnings via `logging.warning()`. |
| L16 | ~~Experience recording silently fails~~ | `_record_resolved_failures()` now validates LLM output, logs errors at each stage. Knowledge system capture logged separately. |

## Numerical Methods

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L2 | **YieldCurve.bump() only hits exact tenor matches** — interpolation-unaware, KRDs are zero for tenors not in the curve | Key rate durations broken on flat or sparse curves | `yield_curve.py:bump()` |
| L3 | **Vega bump assumes FlatVol** — analytics Vega creates a new FlatVol at bumped level, ignoring smile/surface shape | Vega wrong for non-flat vol surfaces | `measures.py:Vega.compute()` |

## Calibration

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L5 | **Mean reversion hard-coded to 0.1** — not calibrated to swaptions or caps | HW model may misprice for different market regimes | `callable_bond.py:63`, `cookbooks.py:90` |
| L6 | **HW vol not calibrated to cap/swaption market** — sigma_HW derived from a single Black vol quote | Correct calibration would fit to a vol term structure | all rate tree pricers |
| L7 | **Prepayment CPR hard-coded to 6%** — PSA model uses fixed base CPR | MBS pricing uses assumed prepayment, not market-implied | `prepayment.py:26` |
| L8 | **Recovery rate hard-coded to 40%** — default for all credit instruments | Should be per-issuer or per-seniority | `nth_to_default.py:30` |

## Analytics

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L9 | **YTM not computed** — PricingResult.ytm is always None | Bond analytics incomplete | `pricer.py:75` |
| L10 | **Accrued interest simplified** — does not handle all market conventions | May not match Bloomberg/QuantLib for edge cases | `pricer.py:44` |
| L11 | **Barrier monitoring is discrete** — continuous barrier approximated by time steps | Barrier option prices need many steps to converge | `barrier_option.py:39` |
| L12 | **Duration/convexity via finite differences** — not analytical | Small numerical error, slower than closed-form for bonds | `measures.py` |

## Infrastructure

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L13 | **Bloomberg data provider is a placeholder** — raises NotImplementedError | Cannot connect to Bloomberg terminals | `bloomberg.py:26` |
| L14 | **Agent tests require LLM API key** — skipped without credentials | Agent pipeline untestable in CI without API access | `test_callable_bond.py`, `test_swaption_demo.py` |
| L23 | **Sandboxed FinancePy imports need a writable Numba cache** — `financepy` may fail at import time when Numba cannot write to package or user cache directories | FinancePy cross-validation/tests can fail in restricted runners unless `NUMBA_CACHE_DIR` points to a writable path such as `/tmp/numba_cache` | `docs/validation/numba_cache.md` |
| L24 | **External market-data auto-resolution is still discount-curve only** — `source="mock"` now provides a richer simulated snapshot with named rates/forecast/vol/credit/FX components plus underlier spots and synthetic local-vol/jump/model-parameter packs, but live providers still only auto-build the default discount curve today | Direct connector and real full-snapshot resolution are still incomplete | `data/resolver.py`, `data/mock.py`, `data/schema.py`, `session.py` |

## Dupire / Local Vol

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L17 | **Local vol falls back to implied vol when denominator ≤ 0** — numerically unstable regions | Local vol surface may have discontinuities | `local_vol.py:55` |

## Knowledge System (discovered from task runs)

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L18 | **Heston class naming confusion** — registry says `Heston`, agent sometimes guesses `HestonProcess` | Rare import hallucination | `processes/heston.py` |
| L19 | **No StateSpace in default test market data** — tasks needing scenario-weighted pricing fail | T93 and similar tasks | `scripts/run_tasks.py` |
| L20 | **Agent sometimes omits MarketState import** — generates code using MarketState without importing it | Runtime NameError | `executor.py` code gen |
| L21 | **No copula data contracts** — credit tasks lack vol/unit conventions | Soft failures on credit instruments | `knowledge/canonical/data_contracts.yaml` |
| L22 | **No analytical method requirements** — no modeling constraints for simple discounting | Missing guidance for basic tasks | `knowledge/canonical/method_requirements.yaml` |
