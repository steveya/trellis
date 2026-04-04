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
| L15 | ~~Critic/validator failures silently caught~~ | All `except Exception: pass` blocks in `_validate_build()` now log warnings via `logging.warning()`. |
| L16 | ~~Experience recording silently fails~~ | `_record_resolved_failures()` now validates LLM output, logs errors at each stage. Knowledge system capture logged separately. |

## Revalidated Open Limitations

### Numerical Methods And Runtime Risk

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L2 | **`YieldCurve.bump()` only hits exact tenor matches** — key-rate shocks do not propagate through interpolation, so off-grid tenors are effectively ignored | KRDs and tenor scenarios are unreliable on sparse or flat curves | `trellis/curves/yield_curve.py`, `trellis/analytics/measures.py` |
| L3 | **Runtime vega is flat-vol centric** — true autodiff is only used for `FlatVol`; non-flat surfaces are reduced to one representative scalar vol and rewrapped as `FlatVol` | Smile/surface vega is materially wrong for anything beyond flat vol | `trellis/analytics/measures.py` |
| L11 | **Barrier monitoring is discrete fixed-step Monte Carlo** — continuous barriers are approximated by checking a finite path grid | Barrier prices and Greeks require many steps to converge and are not desk-grade for tight barrier risk | `trellis/instruments/barrier_option.py` |
| L12 | **Runtime risk surface is still incomplete and bump-heavy** — duration/convexity remain finite-difference; `delta`, `gamma`, and `theta` have no runtime implementation; scenario support is centered on simple rate shifts | Risk coverage is too thin for daily exotic-book risk or explain workflows | `trellis/analytics/measures.py`, `tests/test_agent/test_measure_protocol.py`, `trellis/pipeline.py` |
| L25 | **Early-exercise Monte Carlo upper bounds are diagnostic, not production dual bounds** — the current upper bound is a perfect-information pathwise maximum, not a full Andersen-Broadie-style nested dual construction | American/Bermudan MC error control is not strong enough for institutional exercise risk | `trellis/models/monte_carlo/primal_dual.py` |

### Calibration And Model Realism

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L5 | **Hull-White helper routes still default mean reversion to `0.1`** instead of calibrating or requiring an explicit market-fitted input | Rates optionality can be mispriced across regimes and tenors | `trellis/instruments/callable_bond.py`, `trellis/models/callable_bond_tree.py`, `trellis/models/zcb_option.py`, `trellis/models/zcb_option_tree.py`, `trellis/models/bermudan_swaption_tree.py` |
| L6 | **Hull-White / rate-tree volatility calibration is still flat-Black conversion, not a term-structure fit** | Tree-based rates pricing is not calibrated to a real cap/floor or swaption surface | `trellis/models/calibration/rates.py`, `trellis/agent/data_contract.py`, `trellis/agent/executor.py` |
| L17 | **Dupire local vol falls back to implied vol when the denominator becomes non-positive** | Local-vol surfaces can become discontinuous or unstable exactly where robust production smoothing is needed | `trellis/models/calibration/local_vol.py` |
| L26 | **Curve bootstrap is not actually fully differentiable or desk-complete** — Jacobians are finite-difference, Newton solves use raw NumPy, and the instrument setup hardcodes simple deposit/future/swap assumptions | Curve construction is too simplified for production multi-curve calibration or stable adjoint sensitivities | `trellis/curves/bootstrap.py` |
| L27 | **Rates calibration is helper-grade, not workflow-grade** — cap/floor and swaption calibration solve one flat Black vol per quote rather than a surface or model parameter set | No desk-usable rates calibration stack for replay, regularization, or market-surface fitting | `trellis/models/calibration/rates.py` |
| L28 | **SABR calibration is slice-level only** — it fits `(alpha, rho, nu)` for one expiry/strike smile with fixed `beta` | No term-structure or surface calibration workflow for rates/equity vol desks | `trellis/models/calibration/sabr_fit.py` |
| L29 | **Advanced stochastic-vol models are not fully integrated into the generic process stack** — `Heston` and `SABRProcess` exist as model/formula helpers, but they are not wired into the common `StochasticProcess` interface used by the Monte Carlo engine | Model breadth exists on paper, but cross-engine calibration/risk reuse is limited | `trellis/models/processes/base.py`, `trellis/models/processes/heston.py`, `trellis/models/processes/sabr.py` |

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
| L36 | **Benchmark coverage is too narrow** — the checked-in benchmark surface only measures small GBM path kernels | There is no evidence trail for calibration throughput, exotic pricing throughput, or book-scale risk performance | `docs/benchmarks/monte_carlo_path_kernels.json` |
| L37 | **QMC / variance-reduction tooling is still basic** — Sobol normals, Brownian bridge, antithetics, and a simple control variate exist, but there is no broader simulation-optimization toolkit around them | Useful building blocks exist, but not yet a comprehensive industrial simulation stack | `trellis/models/qmc/__init__.py`, `trellis/models/monte_carlo/variance_reduction.py` |

## Needs Revalidation

These IDs existed in older versions of this file but were not rechecked during
the 2026-04-04 mathematical/numerical/computational pass.

| # | Legacy entry | Prior focus area |
|---|-------------|------------------|
| L7 | Prepayment CPR hard-coded to 6% | mortgage / prepayment |
| L8 | Recovery rate hard-coded to 40% | credit products |
| L9 | YTM not computed | bond analytics |
| L10 | Accrued interest simplified | bond conventions |
| L13 | Bloomberg provider is a placeholder | data integration |
| L14 | Agent tests require API key | CI / infrastructure |
| L18 | Heston class naming confusion | knowledge system |
| L19 | No `StateSpace` in default test market data | task runner |
| L20 | Agent sometimes omits `MarketState` import | code generation |
| L21 | No copula data contracts | knowledge system |
| L22 | No analytical method requirements | knowledge system |
| L23 | FinancePy imports need writable Numba cache | validation infrastructure |
| L24 | External market-data auto-resolution still discount-curve first | market-data plumbing |
