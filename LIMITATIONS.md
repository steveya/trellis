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
| L39 | ~~Build-loop rejected legitimate signed-PV payoffs~~ | `check_non_negativity` in `trellis/agent/invariants.py` now short-circuits via `_payoff_is_signed_linear` for single-name CDS (and any future signed linear products declared through the same trait helper), so protection-buyer CDS clean PV no longer fails the invariant suite. Family-specific CDS invariants (`check_cds_spread_quote_normalization`, `check_cds_credit_curve_sensitivity`) remain in force. (QUA-851.) |
| L40 | ~~Build-loop enforced vol-monotonicity on digital options~~ | `check_vol_monotonicity` in `trellis/agent/invariants.py` now short-circuits via `_payoff_has_nonmonotonic_vol` for cash-or-nothing / asset-or-nothing digital payoffs (classified by class name, spec name, `cash_payoff` attribute, or `payoff_style` marker), so digital options no longer fail the invariant suite for their legitimately non-monotonic price-vs-vol curves. Pricing-exception detection is preserved; only the monotonicity assertion is skipped. (QUA-879.) |

## Revalidated Open Limitations

### Numerical Methods And Runtime Risk

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L3 | **Scalar runtime vega is only partially surface-native** — `FlatVol` now uses autodiff, `GridVolSurface` scalar vega now uses an explicit parallel bucket bump, and unsupported surfaces now disclose representative-flat-vol fallback metadata instead of silently collapsing; but there is still no true scalar autodiff path for realistic non-flat surfaces | Runtime vega is now auditable, but scalar smile/surface vega outside the declared support contracts remains approximate rather than a true surface-native Greek | `trellis/analytics/measures.py`, `trellis/models/vol_surface.py` |
| L11 | **Barrier monitoring is discrete fixed-step Monte Carlo** — continuous barriers are approximated by checking a finite path grid | Barrier prices and Greeks require many steps to converge and are not desk-grade for tight barrier risk | `trellis/instruments/barrier_option.py` |
| L12 | **Runtime risk is broader and provenance-aware, but still bump-heavy outside the declared AD lanes** — public `YieldCurve` DV01/duration/convexity and exact-node KRDs can now resolve through `autodiff_public_curve`, and `delta`/`gamma`/`theta` now exist as runtime measures with explicit metadata; but spot risk, theta, off-grid KRDs, rebuild-based rates risk, and most scenario/explain workflows still depend on finite differences or structured bumps | The runtime now states which derivative method it used instead of silently mixing paths, but the overall risk surface is still too bump-oriented for broad exotic-book daily risk or explain use | `trellis/analytics/measures.py`, `trellis/models/vol_surface.py`, `docs/user_guide/pricing.rst` |
| L25 | **Early-exercise Monte Carlo upper bounds are diagnostic, not production dual bounds** — the current upper bound is a perfect-information pathwise maximum, not a full Andersen-Broadie-style nested dual construction | American/Bermudan MC error control is not strong enough for institutional exercise risk | `trellis/models/monte_carlo/primal_dual.py` |

### Calibration And Model Realism

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L6 | **Supported Hull-White calibration is still constant-parameter only** — the checked workflow fits one `(mean_reversion, sigma)` pair across the strip rather than a time-dependent sigma term structure or cap-surface fit | Tree-based rates pricing now supports a reusable calibrated parameter set, but not a full term-structure calibration workflow | `trellis/models/calibration/rates.py` |
| L17 | **Dupire local vol now flags unstable regions, but still repairs them with implied-vol fallback rather than a regularized surface fit** | Review tools can now inspect explicit diagnostics and warnings, but the checked path is still a hardening layer rather than a production arbitrage-repair workflow | `trellis/models/calibration/local_vol.py` |
| L26 | **Curve bootstrap is still not desk-complete even though the checked solve now carries an explicit autodiff repricing Jacobian** — the supported bootstrap lane now exposes ``autodiff_vector_jacobian`` provenance, but the instrument setup still hardcodes a simplified deposit/future/swap world and does not represent a production multi-curve calibration plant | Curve construction is now more auditable and derivative-aware, but still too simplified for desk-grade multi-curve buildout or industrial adjoint sensitivities | `trellis/curves/bootstrap.py`, `trellis/models/calibration/solve_request.py` |
| L27 | **Flat-Black rates helpers are still quote-local** — cap/floor and European swaption helpers solve one flat Black vol per quote instead of assembling a reusable surface or regularized workflow | Hull-White strip calibration is now supported, but broader rates-vol surface calibration remains incomplete | `trellis/models/calibration/rates.py` |
| L28 | **SABR calibration is still single-smile only** — the smile-input assembly and fit diagnostics are now explicit, but the supported fit still calibrates one expiry smile with fixed `beta` at a time | No full SABR term-structure or multi-expiry surface workflow yet | `trellis/models/calibration/sabr_fit.py` |
| L29 | **Advanced stochastic-vol integration is still partial** — supported Heston smile calibration and runtime binding now exist, but the checked path is still single-smile only and broader cross-engine stochastic-vol reuse remains incomplete | Trellis can now fit and reuse one Heston smile parameter set, but not a full stochastic-vol surface plant or uniform reuse across all stochastic-vol families | `trellis/models/calibration/heston_fit.py`, `trellis/models/processes/heston.py`, `trellis/models/processes/sabr.py` |

### Autograd And Sensitivities

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L30 | **The AD layer is still a thin `autograd` wrapper** — there is no reverse-mode portfolio AAD stack, no dedicated Jacobian/VJP/JVP surface, and no industrial adjoint workflow | Sensitivity throughput will not scale to large books, calibration loops, or scenario grids | `trellis/core/differentiable.py` |
| L31 | **Large parts of the numerical stack are still intentionally forward-only** — the public payoff contract and the public curve/surface abstractions now preserve traced scalars for supported smooth node-value workflows, runtime risk now records the resolved derivative method on scalar and structured outputs, and deterministic Monte Carlo can now keep smooth terminal/snapshot state contracts on the traced path; but streaming reduced-state MC accumulation, discontinuous payoffs, Numba kernels, unsupported surface/risk contracts, and many custom schemes still sit outside the traced region | Trellis can now differentiate through the public payoff boundary, through public yield/credit/grid-vol market objects on supported node-value workflows, through explicit-shock smooth state-aware Monte Carlo contracts, and it can now state when it fell back to bumps; but it still cannot rely on one consistent differentiable path across pricing and risk | `trellis/core/payoff.py`, `trellis/curves/yield_curve.py`, `trellis/curves/credit_curve.py`, `trellis/models/vol_surface.py`, `trellis/analytics/measures.py`, `trellis/models/monte_carlo/engine.py`, `docs/quant/differentiable_pricing.rst` |
| L32 | **Differentiable Monte Carlo is still explicit-shock and smooth-contract only** — terminal-only and smooth event-replay state-aware payoffs can now stay on the traced path, but deterministic pathwise gradients still require explicit shocks, supported differentiable schemes remain limited, and barrier/exercise-style event logic or true streaming reduced-state accumulation still fall back or fail closed | Autograd-based MC Greeks now cover a more reusable runtime subset, but they are still too constrained for broad production risk, especially for discontinuous event logic or memory-scaled state streaming | `trellis/models/monte_carlo/engine.py`, `trellis/models/monte_carlo/event_aware.py`, `trellis/models/monte_carlo/event_state.py`, `trellis/models/monte_carlo/path_state.py` |

### Portfolio, Scenario, And Counterparty Risk

| # | Limitation | Impact | Files |
|---|-----------|--------|-------|
| L41 | **Route-free ContractIR cutover is still bounded by decomposition coverage and excludes arithmetic Asians** — admitted requests that decompose into a checked `ContractIR` shape now select exact helpers route-free on the fresh-build path, but structurally unsupported or under-specified requests still fall back to the compatibility route path, and arithmetic Asians still fail closed pending a real checked helper surface | Trellis can now claim route-free fresh-build authority for the migrated bounded cohort, but it is still incorrect to claim universal route retirement or arithmetic-Asian structural support | `trellis/agent/contract_ir_solver_compiler.py`, `trellis/agent/semantic_contract_compiler.py`, `trellis/agent/platform_requests.py`, `docs/quant/contract_ir.rst`, `docs/developer/contract_ir_solver_compiler.rst` |
| L42 | **Quoted-observable closure is selection-only today** — `CurveQuote` / `SurfaceQuote` nodes, bounded decomposition, and a route-free admission registry now exist for terminal linear quote spreads, but Trellis still has no checked executable lowering for future curve/surface quote products | Trellis can now represent and structurally admit the first quoted-observable cohort without route ids, but it is still not support-contract-correct to claim end-to-end pricing coverage for quoted-spread or vol-skew products until a checked lowering lane lands | `trellis/agent/contract_ir.py`, `trellis/agent/knowledge/decompose.py`, `trellis/agent/quoted_observable_admission.py`, `docs/quant/contract_ir.rst` |
| L43 | **Static-leg route-free authority is still bounded to the admitted cohort** — route-free decomposition and checked lowering admission now exist for vanilla fixed-float IRS, SOFR/FF-style basis swaps, fixed coupon bonds, and scheduled `period_rate_option_strip` contracts; the cap/floor strip benchmark cohort (`F003`-`F005`) now also executes on the deterministic exact-binding fresh-build path without LLM code generation, but richer leg structures outside the admitted constant-notional coupon, basis, and scheduled-strip families remain unsupported | Trellis can now honestly claim authoritative fresh-build route-free execution for the admitted static-leg strip cohort and checked lowering for the broader bounded static-leg surface, but it is still not support-contract-correct to claim generic leg-product coverage or full cutover for richer leg structures | `trellis/agent/static_leg_contract.py`, `trellis/agent/static_leg_admission.py`, `trellis/agent/executor.py`, `trellis/agent/knowledge/decompose.py`, `trellis/models/rate_basis_swap.py`, `docs/quant/static_leg_contract_ir.rst` |
| L44 | **Dynamic event/state/control closure is still not executable today** — `DynamicContractIR` now carries explicit inventory and control-magnitude semantics, bounded decompositions exist for autocallable/TARN, callable-bond/swing, and GMWB-style proving fixtures, `compile_dynamic_lane_admission(...)` emits admitted automatic, discrete, and continuous lane contracts with benchmark plans, and overlay-bearing GMWB requests now have an explicit `InsuranceOverlayContractIR` representation instead of only blocker prose; but none of those dynamic or overlay lanes is yet wired into the authoritative fresh-build pricing compiler | Trellis can now classify and structurally admit the first dynamic cohorts without route-local product authority, and it can now represent mortality/lapse/fee-bearing control relatives honestly without flattening them into the financial-control core; but it is still not support-contract-correct to claim route-free executable pricing for autocallables, TARN/TARF, callable coupon structures, swing options, or GMxB-style control or insurance-overlay families until a checked numerical compiler path lands | `trellis/agent/dynamic_contract_ir.py`, `trellis/agent/insurance_overlay_contract.py`, `trellis/agent/dynamic_lane_admission.py`, `trellis/agent/semantic_track_classifier.py`, `trellis/agent/knowledge/decompose.py`, `docs/quant/dynamic_contract_ir.rst` |
| L38 | **Binding-first exotic proof coverage is still bounded to the agreed cohort** — the checked proof closeout now passes the full `11`-task benchmark cohort and certifies the honest-block sentinel, but that evidence is still limited to the explicitly measured constructable structures | Trellis can now honestly claim the agreed proof cohort, but it is still not support-contract-correct to claim arbitrary constructable-exotic coverage beyond that bounded benchmark surface | `doc/plan/done__binding-first-exotic-proof-closeout.md`, `docs/benchmarks/binding_first_exotic_proof_closeout.json`, `tests/evals/binding_first_exotic_proof.yaml`, `trellis/agent/backend_bindings.py`, `trellis/agent/family_lowering_ir.py`, `trellis/agent/dsl_lowering.py` |
| L33 | **Book and pipeline abstractions are still thin for mixed exotic books** — book aggregation stops at MV/DV01/duration and the pipeline scenario surface remains simple | Trellis is not yet a strong book-level pricing/risk engine even before xVA enters the picture | `trellis/book.py`, `trellis/pipeline.py`, `trellis/analytics/measures.py` |
| L34 | **There is still no netting, collateral, or xVA engine** — Trellis now has a reusable factor-state simulation substrate plus clean future-value cubes for supported vanilla interest-rate swap positions and shared-path swap portfolios, but it still has no netting-set aggregation, collateral recursion, `EE/EPE/PFE` service, or `CVA/DVA/FVA/MVA/KVA` stack | Counterparty-risk building blocks now exist for supported swap workflows, but institutional exposure and xVA analytics remain unsupported | `trellis/models/monte_carlo/simulation_substrate.py`, `trellis/book.py`, `doc/plan/done__exotic-desk-roadmap.md` |

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
