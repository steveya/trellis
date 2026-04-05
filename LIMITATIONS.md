# Known Limitations

This file is a repo-grounded snapshot of current limitations, not a wishlist.

Open entries below were manually revalidated on 2026-04-04 from the checked-in
code and docs, with emphasis on the mathematical, numerical, and computational
gaps that block Trellis from being desk-credible for calibration, exotic-book
pricing/risk, and counterparty-style workflows.

Legacy entries that were present in older versions of this file but were not
rechecked in this pass are preserved under `Needs Revalidation` so their IDs do
not disappear from surrounding docs. They should not be treated as current
ground truth until revalidated.

## Resolved

| # | Limitation | Resolution |
|---|-----------|-----------|
| L1 | ~~Trinomial trees not implemented~~ | `build_rate_lattice()` and `build_generic_lattice()` now support `branching=3`. HW trinomial with mean-reversion probabilities. |
| L4 | ~~Old CN/implicit FD still have coefficient bug~~ | `__init__.py` now imports from `theta_method_1d`. Old names are backward-compat wrappers. Capabilities updated. |
| L9 | ~~YTM not computed~~ | `trellis.engine.pricer.price_instrument()` now solves a coupon-frequency nominal yield-to-maturity from the reported dirty price and projects it through the direct/session pricing surfaces. |
| L15 | ~~Critic/validator failures silently caught~~ | All `except Exception: pass` blocks in `_validate_build()` now log warnings via `logging.warning()`. |
| L16 | ~~Experience recording silently fails~~ | `_record_resolved_failures()` now validates LLM output, logs errors at each stage. Knowledge system capture logged separately. |
| L10 | ~~Accrued interest simplified~~ | Bond accrued interest now uses the explicit coupon schedule plus the selected day-count convention instead of a flat period approximation. |
| L5 | ~~Hull-White helper routes still default mean reversion to `0.1`~~ | Callable-bond, Bermudan-swaption, and ZCB-option helpers now resolve Hull-White parameters from explicit inputs or `MarketState.model_parameters` / `model_parameter_sets`, with legacy heuristics only as fallback. |
| L2 | ~~`YieldCurve.bump()` only hits exact tenor matches~~ | The shared curve-shock substrate now supports off-grid bucket nodes, and `KeyRateDurations` consumes the same bucket grid so interpolation-aware KRD requests no longer collapse to exact-knot-only sensitivities. |

## Revalidated Open Limitations

### Numerical Methods And Runtime Risk

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L3 | **Runtime vega is flat-vol centric** — true autodiff is only used for `FlatVol`; non-flat surfaces are reduced to one representative scalar vol and rewrapped as `FlatVol` | Smile/surface vega is materially wrong for anything beyond flat vol | `trellis/analytics/measures.py` |
| L11 | **Barrier monitoring is discrete fixed-step Monte Carlo** — continuous barriers are approximated by checking a finite path grid | Barrier prices and Greeks require many steps to converge and are not desk-grade for tight barrier risk | `trellis/instruments/barrier_option.py` |
| L12 | **Runtime risk surface is still incomplete and bump-heavy** — duration/convexity remain finite-difference; `delta`, `gamma`, and `theta` have no runtime implementation; rate scenarios now include named twist/butterfly packs, but the broader scenario and explain surface is still narrow | Risk coverage is stronger than simple parallel-rate shifts alone, but still too thin for full daily exotic-book risk or explain workflows | `trellis/analytics/measures.py`, `tests/test_agent/test_measure_protocol.py`, `trellis/pipeline.py` |
| L25 | **Early-exercise Monte Carlo upper bounds are diagnostic, not production dual bounds** — the current upper bound is a perfect-information pathwise maximum, not a full Andersen-Broadie-style nested dual construction | American/Bermudan MC error control is not strong enough for institutional exercise risk | `trellis/models/monte_carlo/primal_dual.py` |

### Calibration And Model Realism

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L6 | **Supported Hull-White calibration is still constant-parameter only** — the checked workflow fits one `(mean_reversion, sigma)` pair across the strip rather than a time-dependent sigma term structure or cap-surface fit | Tree-based rates pricing now supports a reusable calibrated parameter set, but not a full term-structure calibration workflow | `trellis/models/calibration/rates.py` |
| L17 | **Dupire local vol now flags unstable regions, but still repairs them with implied-vol fallback rather than a regularized surface fit** | Review tools can now inspect explicit diagnostics and warnings, but the checked path is still a hardening layer rather than a production arbitrage-repair workflow | `trellis/models/calibration/local_vol.py` |
| L26 | **Curve bootstrap is not actually fully differentiable or desk-complete** — Jacobians are finite-difference, Newton solves use raw NumPy, and the instrument setup hardcodes simple deposit/future/swap assumptions | Curve construction is too simplified for production multi-curve calibration or stable adjoint sensitivities | `trellis/curves/bootstrap.py` |
| L27 | **Flat-Black rates helpers are still quote-local** — cap/floor and European swaption helpers solve one flat Black vol per quote instead of assembling a reusable surface or regularized workflow | Hull-White strip calibration is now supported, but broader rates-vol surface calibration remains incomplete | `trellis/models/calibration/rates.py` |
| L28 | **SABR calibration is still single-smile only** — the smile-input assembly and fit diagnostics are now explicit, but the supported fit still calibrates one expiry smile with fixed `beta` at a time | No full SABR term-structure or multi-expiry surface workflow yet | `trellis/models/calibration/sabr_fit.py` |
| L29 | **Advanced stochastic-vol integration is still partial** — supported Heston smile calibration and runtime binding now exist, but the checked path is still single-smile only and broader cross-engine stochastic-vol reuse remains incomplete | Trellis can now fit and reuse one Heston smile parameter set, but not a full stochastic-vol surface plant or uniform reuse across all stochastic-vol families | `trellis/models/calibration/heston_fit.py`, `trellis/models/processes/heston.py`, `trellis/models/processes/sabr.py` |

### Autograd And Sensitivities

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L30 | **The AD layer is still a thin `autograd` wrapper** — there is no reverse-mode portfolio AAD stack, no dedicated Jacobian/VJP/JVP surface, and no industrial adjoint workflow | Sensitivity throughput will not scale to large books, calibration loops, or scenario grids | `trellis/core/differentiable.py` |
| L31 | **Large parts of the numerical stack are intentionally forward-only** — smile surfaces, Numba kernels, reduced-storage MC accumulation, discontinuous payoffs, and many custom schemes sit outside the traced region | The library cannot rely on one consistent differentiable path across pricing and risk | `docs/quant/differentiable_pricing.rst` |
| L32 | **Differentiable Monte Carlo is narrow and explicit-shock only** — deterministic pathwise gradients require explicit shocks, and supported differentiable schemes are limited | Autograd-based MC Greeks are useful for demos but too constrained for production risk | `trellis/models/monte_carlo/engine.py` |

### Portfolio, Scenario, And Counterparty Risk

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L33 | **Book and pipeline abstractions are still thin for mixed exotic books** — book aggregation stops at MV/DV01/duration and the pipeline scenario surface remains simple | Trellis is not yet a strong book-level pricing/risk engine even before xVA enters the picture | `trellis/book.py`, `trellis/pipeline.py`, `trellis/analytics/measures.py` |
| L34 | **There is no exposure or xVA engine** — no exposure cube, no `EE/EPE/PFE`, no `CVA/DVA/FVA/MVA/KVA`, and no implemented collateral/netting semantics | Counterparty-risk analysis is not currently supported beyond basic credit-product pricing | `docs/plans/exotic-desk-roadmap.md` |

### Performance And Scaling

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L35 | **Performance acceleration is local and selective** — there is optional Numba on some kernels, but no multicore, distributed, or GPU execution path | Calibration sweeps, large scenario sets, and portfolio risk runs will hit a scaling ceiling quickly | `trellis/models/_numba.py`, `trellis/models/monte_carlo/engine.py` |
| L36 | **Benchmark coverage is still too narrow** — checked calibration throughput baselines now exist alongside the Monte Carlo path-kernel benchmark, but exotic-pricing, scenario, and book-risk benchmark surfaces are still missing | Trellis can now compare supported calibration workflows over time, but there is still no broad evidence trail for exotic pricing throughput or book-scale risk performance | `docs/benchmarks/monte_carlo_path_kernels.json`, `docs/benchmarks/calibration_workflows.json` |
| L37 | **QMC / variance-reduction tooling is still basic** — Sobol normals, Brownian bridge, antithetics, and a simple control variate exist, but there is no broader simulation-optimization toolkit around them | Useful building blocks exist, but not yet a comprehensive industrial simulation stack | `trellis/models/qmc/__init__.py`, `trellis/models/monte_carlo/variance_reduction.py` |

## Needs Revalidation

These IDs existed in older versions of this file but were not rechecked during
the 2026-04-04 mathematical/numerical/computational pass.

| # | Legacy entry | Prior focus area |
|---|-------------|------------------|
| L7 | Prepayment CPR hard-coded to 6% | mortgage / prepayment |
| L8 | Recovery rate hard-coded to 40% | credit products |
| L13 | Bloomberg provider is a placeholder | data integration |
| L14 | Agent tests require API key | CI / infrastructure |
| L18 | Heston class naming confusion | knowledge system |
| L19 | No `StateSpace` in default test market data | task runner |
| L20 | Agent sometimes omits `MarketState` import | code generation |
| L21 | No copula data contracts | knowledge system |
| L22 | No analytical method requirements | knowledge system |
| L23 | FinancePy imports need writable Numba cache | validation infrastructure |
| L24 | External market-data auto-resolution still discount-curve first | market-data plumbing |
