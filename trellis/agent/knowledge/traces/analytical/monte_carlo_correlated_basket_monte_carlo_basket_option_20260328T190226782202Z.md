# Analytical Trace: `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z`
- Trace type: `analytical`
- Route family: `monte_carlo`
- Route name: `correlated_basket_monte_carlo`
- Model: `monte_carlo`
- Status: `ok`
- Created at: `2026-03-28T19:02:26.782247+00:00`
- Updated at: `2026-03-28T19:02:26.937986+00:00`

## Context
- `class_name`: 'HimalayaBasketPayoff'
- `generation_plan`: {'method': 'monte_carlo', 'instrument_type': 'basket_option', 'inspected_modules': ['trellis.core.date_utils', 'trellis.core.market_state', 'trellis.core.payoff', 'trellis.core.types', 'trellis.instruments.barrier_option', 'trellis.models.black', 'trellis.models.monte_carlo'], 'approved_modules': ['trellis.core.date_utils', 'trellis.core.differentiable', 'trellis.core.market_state', 'trellis.core.payoff', 'trellis.core.types', 'trellis.instruments.barrier_option', 'trellis.models.black', 'trellis.models.monte_carlo', 'trellis.models.monte_carlo.basket_state', 'trellis.models.monte_carlo.brownian_bridge', 'trellis.models.monte_carlo.discretization', 'trellis.models.monte_carlo.early_exercise', 'trellis.models.monte_carlo.engine', 'trellis.models.monte_carlo.event_state', 'trellis.models.monte_carlo.local_vol', 'trellis.models.monte_carlo.lsm', 'trellis.models.monte_carlo.path_state', 'trellis.models.monte_carlo.primal_dual', 'trellis.models.monte_carlo.profiling', 'trellis.models.monte_carlo.quanto', 'trellis.models.monte_carlo.ranked_observation_payoffs', 'trellis.models.monte_carlo.schemes', 'trellis.models.monte_carlo.semantic_basket', 'trellis.models.monte_carlo.stochastic_mesh', 'trellis.models.monte_carlo.tv_regression', 'trellis.models.monte_carlo.variance_reduction', 'trellis.models.processes.gbm', 'trellis.models.resolution.basket_semantics'], 'symbols_to_reuse': ['Antithetic', 'BarrierMonitor', 'BarrierOptionPayoff', 'BarrierOptionSpec', 'BasisFunction', 'BasketSpecLike', 'CashflowSchedule', 'Cashflows', 'ChebyshevBasis', 'ContinuationEstimator', 'CorrelationPreflightError', 'CorrelationPreflightReport', 'DataProvider', 'DayCountConvention', 'DeterministicCashflowPayoff', 'DiscountCurve', 'DiscretizationScheme', 'EarlyExerciseDiagnostics', 'EarlyExercisePolicyResult', 'Euler', 'Exact', 'FastLaguerreContinuationEstimator', 'FastPolynomialContinuationEstimator', 'Frequency', 'GBM', 'HermiteBasis', 'Instrument', 'LaguerreBasis', 'LeastSquaresContinuationEstimator', 'LocalVolMonteCarloResult', 'LogEuler', 'MarketState', 'Milstein', 'MissingCapabilityError', 'MonteCarloEngine', 'MonteCarloPathKernelBenchmark', 'MonteCarloPathPayoff', 'MonteCarloPathRequirement', 'MonteCarloPathState', 'PathEventRecord', 'PathEventSpec', 'PathEventState', 'PathEventTimeline', 'PathReducer', 'Payoff', 'PolynomialBasis', 'PresentValue', 'PricingResult', 'QuantoMonteCarloSpecLike', 'RankedObservationBasketMonteCarloPayoff', 'RankedObservationBasketSpec', 'RankedObservationBasketSpecLike', 'ResolvedBasketSemantics', 'ResolvedInputPayoff', 'StateAwarePayoff', 'add_months', 'antithetic', 'antithetic_normals', 'apply_path_event_spec', 'barrier_payoff', 'benchmark_path_kernel', 'black76_asset_or_nothing_call', 'black76_asset_or_nothing_put', 'black76_call', 'black76_cash_or_nothing_call', 'black76_cash_or_nothing_put', 'black76_put', 'brownian_bridge', 'brownian_bridge_increments', 'build_basket_path_requirement', 'build_event_path_requirement', 'build_quanto_mc_initial_state', 'build_quanto_mc_process', 'build_ranked_observation_basket_initial_state', 'build_ranked_observation_basket_process', 'build_ranked_observation_basket_state_payoff', 'control_variate', 'default_continuation_estimator', 'euler_maruyama', 'evaluate_ranked_observation_basket_paths', 'evaluate_ranked_observation_basket_state', 'event_step_indices', 'exact_simulation', 'garman_kohlhagen_call', 'garman_kohlhagen_put', 'generate_schedule', 'get_accrual_fraction', 'get_bracketing_dates', 'get_numpy', 'gradient', 'hessian', 'laguerre_basis', 'local_vol_european_vanilla_price', 'local_vol_european_vanilla_price_result', 'longstaff_schwartz', 'longstaff_schwartz_result', 'milstein', 'observation_step_indices', 'polynomial_basis', 'price_quanto_option_monte_carlo', 'price_ranked_observation_basket_monte_carlo', 'primal_dual_mc', 'primal_dual_mc_result', 'recommended_quanto_mc_engine_kwargs', 'recommended_ranked_observation_basket_mc_engine_kwargs', 'replay_path_event_timeline', 'resolve_basket_semantics', 'sobol_normals', 'stochastic_mesh', 'stochastic_mesh_result', 'terminal_quanto_option_payoff', 'terminal_ranked_observation_basket_payoff', 'terminal_value_payoff', 'tsitsiklis_van_roy', 'tsitsiklis_van_roy_result', 'year_fraction'], 'proposed_tests': ['tests/test_agent/test_build_loop.py'], 'uncertainty_flags': [], 'repo_revision': '2204f054d09cfaa88de56326480925626f4ca2b3', 'primitive_plan': {'route': 'correlated_basket_monte_carlo', 'engine_family': 'monte_carlo', 'score': 4.0}}
- `route_card`: '## Structured Route Card\n- Method family: `monte_carlo`\n- Instrument type: `basket_option`\n- Route: `correlated_basket_monte_carlo`\n- Engine family: `monte_carlo`\n- Required primitives:\n  - `trellis.models.resolution.basket_semantics.resolve_basket_semantics` (market_binding)\n  - `trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo` (route_helper)\n- Resolved instructions:\n  - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n  - [route_hint] Parse `spec.underlyings` into a Python list of ticker strings and `spec.observation_dates` into `date` objects before basket resolution; do not pass raw comma-separated strings into the resolver.\n  - [route_hint] Bind the market state with `resolve_basket_semantics(...)`, then delegate straight to `price_ranked_observation_basket_monte_carlo(...)` through a thin adapter; do not import process primitives directly in the generated adapter.\n  - [route_hint] Reuse `RankedObservationBasketSpec` and `RankedObservationBasketMonteCarloPayoff` from `trellis.models.monte_carlo.semantic_basket`, import only `trellis.models.resolution.basket_semantics` and `trellis.models.monte_carlo.semantic_basket` for the basket adapter, and do not invent extra `trellis.models.*` basket or payoff subpackages.\n  - [route_hint] Do not introduce a bespoke basket spec name such as `HimalayaBasketSpec` or reimplement rank/remove/aggregate logic inline.\n  - [historical_note] Basket Monte Carlo routing should use generic basket semantics and snapshot-based state instead of a product-specific mountain-range branch.\n- Required adapters:\n  - `reuse_shared_basket_market_binding`\n  - `reuse_shared_basket_mc_route_helper`\n- Primary modules to inspect/reuse:\n  - `trellis.core.date_utils`\n  - `trellis.core.market_state`\n  - `trellis.core.payoff`\n  - `trellis.core.types`\n  - `trellis.instruments.barrier_option`\n  - `trellis.models.black`\n- Post-build test targets:\n  - `tests/test_agent/test_build_loop.py`\n- Instruction precedence: follow the approved modules, primitives, and route helper in this card. If older guidance conflicts, treat it as stale and obey this plan.\n- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels.'
- `route_plan`: '## Structured Generation Plan\n- Method family: `monte_carlo`\n- Instrument type: `basket_option`\n- Repo revision: `2204f054d09cfaa88de56326480925626f4ca2b3`\n- Inspected modules:\n  - `trellis.core.date_utils`\n  - `trellis.core.market_state`\n  - `trellis.core.payoff`\n  - `trellis.core.types`\n  - `trellis.instruments.barrier_option`\n  - `trellis.models.black`\n  - `trellis.models.monte_carlo`\n- Approved Trellis modules for imports:\n  - `trellis.core.date_utils`\n  - `trellis.core.differentiable`\n  - `trellis.core.market_state`\n  - `trellis.core.payoff`\n  - `trellis.core.types`\n  - `trellis.instruments.barrier_option`\n  - `trellis.models.black`\n  - `trellis.models.monte_carlo`\n  - `trellis.models.monte_carlo.basket_state`\n  - `trellis.models.monte_carlo.brownian_bridge`\n  - `trellis.models.monte_carlo.discretization`\n  - `trellis.models.monte_carlo.early_exercise`\n  - `trellis.models.monte_carlo.engine`\n  - `trellis.models.monte_carlo.event_state`\n  - `trellis.models.monte_carlo.local_vol`\n  - `trellis.models.monte_carlo.lsm`\n  - `trellis.models.monte_carlo.path_state`\n  - `trellis.models.monte_carlo.primal_dual`\n  - `trellis.models.monte_carlo.profiling`\n  - `trellis.models.monte_carlo.quanto`\n  - `trellis.models.monte_carlo.ranked_observation_payoffs`\n  - `trellis.models.monte_carlo.schemes`\n  - `trellis.models.monte_carlo.semantic_basket`\n  - `trellis.models.monte_carlo.stochastic_mesh`\n  - `trellis.models.monte_carlo.tv_regression`\n  - `trellis.models.monte_carlo.variance_reduction`\n  - `trellis.models.processes.gbm`\n  - `trellis.models.resolution.basket_semantics`\n- Public symbols available from the approved modules:\n  - `Antithetic`\n  - `BarrierMonitor`\n  - `BarrierOptionPayoff`\n  - `BarrierOptionSpec`\n  - `BasisFunction`\n  - `BasketSpecLike`\n  - `CashflowSchedule`\n  - `Cashflows`\n  - `ChebyshevBasis`\n  - `ContinuationEstimator`\n  - `CorrelationPreflightError`\n  - `CorrelationPreflightReport`\n  - `DataProvider`\n  - `DayCountConvention`\n  - `DeterministicCashflowPayoff`\n  - `DiscountCurve`\n  - `DiscretizationScheme`\n  - `EarlyExerciseDiagnostics`\n  - `EarlyExercisePolicyResult`\n  - `Euler`\n  - `Exact`\n  - `FastLaguerreContinuationEstimator`\n  - `FastPolynomialContinuationEstimator`\n  - `Frequency`\n  - `GBM`\n  - `HermiteBasis`\n  - `Instrument`\n  - `LaguerreBasis`\n  - `LeastSquaresContinuationEstimator`\n  - `LocalVolMonteCarloResult`\n  - `LogEuler`\n  - `MarketState`\n  - `Milstein`\n  - `MissingCapabilityError`\n  - `MonteCarloEngine`\n  - `MonteCarloPathKernelBenchmark`\n  - `MonteCarloPathPayoff`\n  - `MonteCarloPathRequirement`\n  - `MonteCarloPathState`\n  - `PathEventRecord`\n  - `PathEventSpec`\n  - `PathEventState`\n  - `PathEventTimeline`\n  - `PathReducer`\n  - `Payoff`\n  - `PolynomialBasis`\n  - `PresentValue`\n  - `PricingResult`\n  - `QuantoMonteCarloSpecLike`\n  - `RankedObservationBasketMonteCarloPayoff`\n  - `RankedObservationBasketSpec`\n  - `RankedObservationBasketSpecLike`\n  - `ResolvedBasketSemantics`\n  - `ResolvedInputPayoff`\n  - `StateAwarePayoff`\n  - `add_months`\n  - `antithetic`\n  - `antithetic_normals`\n  - `apply_path_event_spec`\n  - `barrier_payoff`\n  - `benchmark_path_kernel`\n  - `black76_asset_or_nothing_call`\n  - `black76_asset_or_nothing_put`\n  - `black76_call`\n  - `black76_cash_or_nothing_call`\n  - `black76_cash_or_nothing_put`\n  - `black76_put`\n  - `brownian_bridge`\n  - `brownian_bridge_increments`\n  - `build_basket_path_requirement`\n  - `build_event_path_requirement`\n  - `build_quanto_mc_initial_state`\n  - `build_quanto_mc_process`\n  - `build_ranked_observation_basket_initial_state`\n  - `build_ranked_observation_basket_process`\n  - `build_ranked_observation_basket_state_payoff`\n  - `control_variate`\n  - `default_continuation_estimator`\n  - `euler_maruyama`\n  - `evaluate_ranked_observation_basket_paths`\n- Tests to run after generation:\n  - `tests/test_agent/test_build_loop.py`\n- Primitive route:\n  - Route: `correlated_basket_monte_carlo`\n  - Engine family: `monte_carlo`\n  - Route score: `4.00`\n  - Selected primitives:\n    - `trellis.models.resolution.basket_semantics.resolve_basket_semantics` (market_binding)\n    - `trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo` (route_helper)\n  - Resolved instructions:\n    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n    - [route_hint] Parse `spec.underlyings` into a Python list of ticker strings and `spec.observation_dates` into `date` objects before basket resolution; do not pass raw comma-separated strings into the resolver.\n    - [route_hint] Bind the market state with `resolve_basket_semantics(...)`, then delegate straight to `price_ranked_observation_basket_monte_carlo(...)` through a thin adapter; do not import process primitives directly in the generated adapter.\n    - [route_hint] Reuse `RankedObservationBasketSpec` and `RankedObservationBasketMonteCarloPayoff` from `trellis.models.monte_carlo.semantic_basket`, import only `trellis.models.resolution.basket_semantics` and `trellis.models.monte_carlo.semantic_basket` for the basket adapter, and do not invent extra `trellis.models.*` basket or payoff subpackages.\n    - [route_hint] Do not introduce a bespoke basket spec name such as `HimalayaBasketSpec` or reimplement rank/remove/aggregate logic inline.\n    - [historical_note] Basket Monte Carlo routing should use generic basket semantics and snapshot-based state instead of a product-specific mountain-range branch.\n  - Required adapters:\n    - `reuse_shared_basket_market_binding`\n    - `reuse_shared_basket_mc_route_helper`\n  - Instruction precedence: follow the approved modules, primitives, and route helper in this plan. If older guidance conflicts, treat it as stale and obey this plan.\n- Every `trellis.*` import in the generated code MUST come from the approved module list above.\n- If you need functionality outside the approved list, say so explicitly instead of inventing an import.'
- `selected_curve_names`: {'discount_curve': 'usd_ois', 'forecast_curve': 'USD-SOFR-3M', 'credit_curve': 'usd_ig'}
- `spec_name`: 'HimalayaBasketSpec'

## Steps
- **trace** `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:root`
  - Label: Analytical build
  - Status: `ok`
  - Notes:
    - The trace mirrors the deterministic GenerationPlan used to assemble the route.
  - Inputs:
    - `issue_id`: None
    - `model`: monte_carlo
    - `route_family`: monte_carlo
    - `route_name`: correlated_basket_monte_carlo
    - `task_id`: None
  - Outputs:
    - `route`: {
  "family": "monte_carlo",
  "model": "monte_carlo",
  "name": "correlated_basket_monte_carlo"
}
    - `status`: ok
  - **semantic_resolution** `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:semantic_resolution`
    - Label: Resolve contract and route
    - Status: `ok`
    - Parent: `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:root`
    - Notes:
      - Record the semantic contract that drives route selection, not just the final code path.
    - Inputs:
      - `approved_modules`: [
  "trellis.core.date_utils",
  "trellis.core.differentiable",
  "trellis.core.market_state",
  "trellis.core.payoff",
  "trellis.core.types",
  "trellis.instruments.barrier_option",
  "trellis.models.black",
  "trellis.models.monte_carlo",
  "trellis.models.monte_carlo.basket_state",
  "trellis.models.monte_carlo.brownian_bridge",
  "trellis.models.monte_carlo.discretization",
  "trellis.models.monte_carlo.early_exercise",
  "trellis.models.monte_carlo.engine",
  "trellis.models.monte_carlo.event_state",
  "trellis.models.monte_carlo.local_vol",
  "trellis.models.monte_carlo.lsm",
  "trellis.models.monte_carlo.path_state",
  "trellis.models.monte_carlo.primal_dual",
  "trellis.models.monte_carlo.profiling",
  "trellis.models.monte_carlo.quanto",
  "trellis.models.monte_carlo.ranked_observation_payoffs",
  "trellis.models.monte_carlo.schemes",
  "trellis.models.monte_carlo.semantic_basket",
  "trellis.models.monte_carlo.stochastic_mesh",
  "trellis.models.monte_carlo.tv_regression",
  "trellis.models.monte_carlo.variance_reduction",
  "trellis.models.processes.gbm",
  "trellis.models.resolution.basket_semantics"
]
      - `inspected_modules`: [
  "trellis.core.date_utils",
  "trellis.core.market_state",
  "trellis.core.payoff",
  "trellis.core.types",
  "trellis.instruments.barrier_option",
  "trellis.models.black",
  "trellis.models.monte_carlo"
]
      - `instrument_type`: basket_option
      - `method`: monte_carlo
      - `repo_revision`: 2204f054d09cfaa88de56326480925626f4ca2b3
      - `symbols_to_reuse`: [
  "Antithetic",
  "BarrierMonitor",
  "BarrierOptionPayoff",
  "BarrierOptionSpec",
  "BasisFunction",
  "BasketSpecLike",
  "CashflowSchedule",
  "Cashflows",
  "ChebyshevBasis",
  "ContinuationEstimator",
  "CorrelationPreflightError",
  "CorrelationPreflightReport",
  "DataProvider",
  "DayCountConvention",
  "DeterministicCashflowPayoff",
  "DiscountCurve",
  "DiscretizationScheme",
  "EarlyExerciseDiagnostics",
  "EarlyExercisePolicyResult",
  "Euler",
  "Exact",
  "FastLaguerreContinuationEstimator",
  "FastPolynomialContinuationEstimator",
  "Frequency",
  "GBM",
  "HermiteBasis",
  "Instrument",
  "LaguerreBasis",
  "LeastSquaresContinuationEstimator",
  "LocalVolMonteCarloResult",
  "LogEuler",
  "MarketState",
  "Milstein",
  "MissingCapabilityError",
  "MonteCarloEngine",
  "MonteCarloPathKernelBenchmark",
  "MonteCarloPathPayoff",
  "MonteCarloPathRequirement",
  "MonteCarloPathState",
  "PathEventRecord"
]
      - `uncertainty_flags`: []
    - Outputs:
      - `model`: monte_carlo
      - `primitive_plan_score`: 4.0
      - `route_family`: monte_carlo
      - `route_name`: correlated_basket_monte_carlo
  - **decomposition** `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:decomposition`
    - Label: Select reusable kernels
    - Status: `ok`
    - Parent: `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:root`
    - Notes:
      - Capture the reusable valuation components and any exact basis-claim assembly.
    - Inputs:
      - `adapters`: [
  "reuse_shared_basket_market_binding",
  "reuse_shared_basket_mc_route_helper"
]
      - `blockers`: []
      - `notes`: [
  "Basket Monte Carlo routing should use generic basket semantics and snapshot-based state instead of a product-specific mountain-range branch.",
  "Parse `spec.underlyings` into a Python list of ticker strings and `spec.observation_dates` into `date` objects before basket resolution; do not pass raw comma-separated strings into the resolver.",
  "Bind the market state with `resolve_basket_semantics(...)`, then delegate straight to `price_ranked_observation_basket_monte_carlo(...)` through a thin adapter; do not import process primitives directly in the generated adapter.",
  "Reuse `RankedObservationBasketSpec` and `RankedObservationBasketMonteCarloPayoff` from `trellis.models.monte_carlo.semantic_basket`, import only `trellis.models.resolution.basket_semantics` and `trellis.models.monte_carlo.semantic_basket` for the basket adapter, and do not invent extra `trellis.models.*` basket or payoff subpackages.",
  "Do not introduce a bespoke basket spec name such as `HimalayaBasketSpec` or reimplement rank/remove/aggregate logic inline."
]
      - `primitives`: [
  {
    "module": "trellis.models.resolution.basket_semantics",
    "required": true,
    "role": "market_binding",
    "symbol": "resolve_basket_semantics"
  },
  {
    "module": "trellis.models.monte_carlo.semantic_basket",
    "required": true,
    "role": "route_helper",
    "symbol": "price_ranked_observation_basket_monte_carlo"
  }
]
    - Outputs:
      - `reuse_decision`: route_local
      - `selected_primitives`: [
  {
    "module": "trellis.models.resolution.basket_semantics",
    "required": true,
    "role": "market_binding",
    "symbol": "resolve_basket_semantics"
  },
  {
    "module": "trellis.models.monte_carlo.semantic_basket",
    "required": true,
    "role": "route_helper",
    "symbol": "price_ranked_observation_basket_monte_carlo"
  }
]
  - **assembly** `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:assembly`
    - Label: Assemble route from kernels
    - Status: `ok`
    - Parent: `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:root`
    - Notes:
      - Prefer thin orchestration around existing analytical kernels and route helpers.
    - Inputs:
      - `adapters`: [
  "reuse_shared_basket_market_binding",
  "reuse_shared_basket_mc_route_helper"
]
      - `approved_modules`: [
  "trellis.core.date_utils",
  "trellis.core.differentiable",
  "trellis.core.market_state",
  "trellis.core.payoff",
  "trellis.core.types",
  "trellis.instruments.barrier_option",
  "trellis.models.black",
  "trellis.models.monte_carlo",
  "trellis.models.monte_carlo.basket_state",
  "trellis.models.monte_carlo.brownian_bridge",
  "trellis.models.monte_carlo.discretization",
  "trellis.models.monte_carlo.early_exercise",
  "trellis.models.monte_carlo.engine",
  "trellis.models.monte_carlo.event_state",
  "trellis.models.monte_carlo.local_vol",
  "trellis.models.monte_carlo.lsm",
  "trellis.models.monte_carlo.path_state",
  "trellis.models.monte_carlo.primal_dual",
  "trellis.models.monte_carlo.profiling",
  "trellis.models.monte_carlo.quanto",
  "trellis.models.monte_carlo.ranked_observation_payoffs",
  "trellis.models.monte_carlo.schemes",
  "trellis.models.monte_carlo.semantic_basket",
  "trellis.models.monte_carlo.stochastic_mesh",
  "trellis.models.monte_carlo.tv_regression",
  "trellis.models.monte_carlo.variance_reduction",
  "trellis.models.processes.gbm",
  "trellis.models.resolution.basket_semantics"
]
      - `route_helper`: trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo
    - Outputs:
      - `assembly_card`: ## Structured Route Card
- Method family: `monte_carlo`
- Instrument type: `basket_option`
- Route: `correlated_basket_monte_carlo`
- Engine family: `monte_carlo`
- Required primitives:
  - `trellis.models.resolution.basket_semantics.resolve_basket_semantics` (market_binding)
  - `trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo` (route_helper)
- Resolved instructions:
  - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
  - [route_hint] Parse `spec.underlyings` into a Python list of ticker strings and `spec.observation_dates` into `date` objects before basket resolution; do not pass raw comma-separated strings into the resolver.
  - [route_hint] Bind the market state with `resolve_basket_semantics(...)`, then delegate straight to `price_ranked_observation_basket_monte_carlo(...)` through a thin adapter; do not import process primitives directly in the generated adapter.
  - [route_hint] Reuse `RankedObservationBasketSpec` and `RankedObservationBasketMonteCarloPayoff` from `trellis.models.monte_carlo.semantic_basket`, import only `trellis.models.resolution.basket_semantics` and `trellis.models.monte_carlo.semantic_basket` for the basket adapter, and do not invent extra `trellis.models.*` basket or payoff subpackages.
  - [route_hint] Do not introduce a bespoke basket spec name such as `HimalayaBasketSpec` or reimplement rank/remove/aggregate logic inline.
  - [historical_note] Basket Monte Carlo routing should use generic basket semantics and snapshot-based state instead of a product-specific mountain-range branch.
- Required adapters:
  - `reuse_shared_basket_market_binding`
  - `reuse_shared_basket_mc_route_helper`
- Primary modules to inspect/reuse:
  - `trellis.core.date_utils`
  - `trellis.core.market_state`
  - `trellis.core.payoff`
  - `trellis.core.types`
  - `trellis.instruments.barrier_option`
  - `trellis.models.black`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the approved modules, primitives, and route helper in this card. If older guidance conflicts, treat it as stale and obey this plan.
- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels.
      - `helper_modules`: [
  "trellis.models.resolution.basket_semantics",
  "trellis.models.monte_carlo.semantic_basket"
]
      - `route_helper`: trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo
  - **validation** `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:validation`
    - Label: Validate route and fallbacks
    - Status: `ok`
    - Parent: `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:root`
    - Notes:
      - Record proposed tests, blocker state, and any fallback or reuse notes.
    - Inputs:
      - `blockers`: []
      - `proposed_tests`: [
  "tests/test_agent/test_build_loop.py"
]
      - `uncertainty_flags`: []
      - `validation`: None
    - Outputs:
      - `blocker_report_present`: False
      - `new_primitive_workflow_present`: False
      - `validation_state`: planned
  - **output** `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:output`
    - Label: Final analytical artifact
    - Status: `ok`
    - Parent: `monte_carlo_correlated_basket_monte_carlo_basket_option_20260328T190226782202Z:root`
    - Notes:
      - Persist both the machine-readable trace and the text rendering from the same source of truth.
    - Inputs:
      - `model`: monte_carlo
      - `route`: correlated_basket_monte_carlo
    - Outputs:
      - `route_card`: ## Structured Route Card
- Method family: `monte_carlo`
- Instrument type: `basket_option`
- Route: `correlated_basket_monte_carlo`
- Engine family: `monte_carlo`
- Required primitives:
  - `trellis.models.resolution.basket_semantics.resolve_basket_semantics` (market_binding)
  - `trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo` (route_helper)
- Resolved instructions:
  - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
  - [route_hint] Parse `spec.underlyings` into a Python list of ticker strings and `spec.observation_dates` into `date` objects before basket resolution; do not pass raw comma-separated strings into the resolver.
  - [route_hint] Bind the market state with `resolve_basket_semantics(...)`, then delegate straight to `price_ranked_observation_basket_monte_carlo(...)` through a thin adapter; do not import process primitives directly in the generated adapter.
  - [route_hint] Reuse `RankedObservationBasketSpec` and `RankedObservationBasketMonteCarloPayoff` from `trellis.models.monte_carlo.semantic_basket`, import only `trellis.models.resolution.basket_semantics` and `trellis.models.monte_carlo.semantic_basket` for the basket adapter, and do not invent extra `trellis.models.*` basket or payoff subpackages.
  - [route_hint] Do not introduce a bespoke basket spec name such as `HimalayaBasketSpec` or reimplement rank/remove/aggregate logic inline.
  - [historical_note] Basket Monte Carlo routing should use generic basket semantics and snapshot-based state instead of a product-specific mountain-range branch.
- Required adapters:
  - `reuse_shared_basket_market_binding`
  - `reuse_shared_basket_mc_route_helper`
- Primary modules to inspect/reuse:
  - `trellis.core.date_utils`
  - `trellis.core.market_state`
  - `trellis.core.payoff`
  - `trellis.core.types`
  - `trellis.instruments.barrier_option`
  - `trellis.models.black`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the approved modules, primitives, and route helper in this card. If older guidance conflicts, treat it as stale and obey this plan.
- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels.
      - `route_plan`: ## Structured Generation Plan
- Method family: `monte_carlo`
- Instrument type: `basket_option`
- Repo revision: `2204f054d09cfaa88de56326480925626f4ca2b3`
- Inspected modules:
  - `trellis.core.date_utils`
  - `trellis.core.market_state`
  - `trellis.core.payoff`
  - `trellis.core.types`
  - `trellis.instruments.barrier_option`
  - `trellis.models.black`
  - `trellis.models.monte_carlo`
- Approved Trellis modules for imports:
  - `trellis.core.date_utils`
  - `trellis.core.differentiable`
  - `trellis.core.market_state`
  - `trellis.core.payoff`
  - `trellis.core.types`
  - `trellis.instruments.barrier_option`
  - `trellis.models.black`
  - `trellis.models.monte_carlo`
  - `trellis.models.monte_carlo.basket_state`
  - `trellis.models.monte_carlo.brownian_bridge`
  - `trellis.models.monte_carlo.discretization`
  - `trellis.models.monte_carlo.early_exercise`
  - `trellis.models.monte_carlo.engine`
  - `trellis.models.monte_carlo.event_state`
  - `trellis.models.monte_carlo.local_vol`
  - `trellis.models.monte_carlo.lsm`
  - `trellis.models.monte_carlo.path_state`
  - `trellis.models.monte_carlo.primal_dual`
  - `trellis.models.monte_carlo.profiling`
  - `trellis.models.monte_carlo.quanto`
  - `trellis.models.monte_carlo.ranked_observation_payoffs`
  - `trellis.models.monte_carlo.schemes`
  - `trellis.models.monte_carlo.semantic_basket`
  - `trellis.models.monte_carlo.stochastic_mesh`
  - `trellis.models.monte_carlo.tv_regression`
  - `trellis.models.monte_carlo.variance_reduction`
  - `trellis.models.processes.gbm`
  - `trellis.models.resolution.basket_semantics`
- Public symbols available from the approved modules:
  - `Antithetic`
  - `BarrierMonitor`
  - `BarrierOptionPayoff`
  - `BarrierOptionSpec`
  - `BasisFunction`
  - `BasketSpecLike`
  - `CashflowSchedule`
  - `Cashflows`
  - `ChebyshevBasis`
  - `ContinuationEstimator`
  - `CorrelationPreflightError`
  - `CorrelationPreflightReport`
  - `DataProvider`
  - `DayCountConvention`
  - `DeterministicCashflowPayoff`
  - `DiscountCurve`
  - `DiscretizationScheme`
  - `EarlyExerciseDiagnostics`
  - `EarlyExercisePolicyResult`
  - `Euler`
  - `Exact`
  - `FastLaguerreContinuationEstimator`
  - `FastPolynomialContinuationEstimator`
  - `Frequency`
  - `GBM`
  - `HermiteBasis`
  - `Instrument`
  - `LaguerreBasis`
  - `LeastSquaresContinuationEstimator`
  - `LocalVolMonteCarloResult`
  - `LogEuler`
  - `MarketState`
  - `Milstein`
  - `MissingCapabilityError`
  - `MonteCarloEngine`
  - `MonteCarloPathKernelBenchmark`
  - `MonteCarloPathPayoff`
  - `MonteCarloPathRequirement`
  - `MonteCarloPathState`
  - `PathEventRecord`
  - `PathEventSpec`
  - `PathEventState`
  - `PathEventTimeline`
  - `PathReducer`
  - `Payoff`
  - `PolynomialBasis`
  - `PresentValue`
  - `PricingResult`
  - `QuantoMonteCarloSpecLike`
  - `RankedObservationBasketMonteCarloPayoff`
  - `RankedObservationBasketSpec`
  - `RankedObservationBasketSpecLike`
  - `ResolvedBasketSemantics`
  - `ResolvedInputPayoff`
  - `StateAwarePayoff`
  - `add_months`
  - `antithetic`
  - `antithetic_normals`
  - `apply_path_event_spec`
  - `barrier_payoff`
  - `benchmark_path_kernel`
  - `black76_asset_or_nothing_call`
  - `black76_asset_or_nothing_put`
  - `black76_call`
  - `black76_cash_or_nothing_call`
  - `black76_cash_or_nothing_put`
  - `black76_put`
  - `brownian_bridge`
  - `brownian_bridge_increments`
  - `build_basket_path_requirement`
  - `build_event_path_requirement`
  - `build_quanto_mc_initial_state`
  - `build_quanto_mc_process`
  - `build_ranked_observation_basket_initial_state`
  - `build_ranked_observation_basket_process`
  - `build_ranked_observation_basket_state_payoff`
  - `control_variate`
  - `default_continuation_estimator`
  - `euler_maruyama`
  - `evaluate_ranked_observation_basket_paths`
- Tests to run after generation:
  - `tests/test_agent/test_build_loop.py`
- Primitive route:
  - Route: `correlated_basket_monte_carlo`
  - Engine family: `monte_carlo`
  - Route score: `4.00`
  - Selected primitives:
    - `trellis.models.resolution.basket_semantics.resolve_basket_semantics` (market_binding)
    - `trellis.models.monte_carlo.semantic_basket.price_ranked_observation_basket_monte_carlo` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
    - [route_hint] Parse `spec.underlyings` into a Python list of ticker strings and `spec.observation_dates` into `date` objects before basket resolution; do not pass raw comma-separated strings into the resolver.
    - [route_hint] Bind the market state with `resolve_basket_semantics(...)`, then delegate straight to `price_ranked_observation_basket_monte_carlo(...)` through a thin adapter; do not import process primitives directly in the generated adapter.
    - [route_hint] Reuse `RankedObservationBasketSpec` and `RankedObservationBasketMonteCarloPayoff` from `trellis.models.monte_carlo.semantic_basket`, import only `trellis.models.resolution.basket_semantics` and `trellis.models.monte_carlo.semantic_basket` for the basket adapter, and do not invent extra `trellis.models.*` basket or payoff subpackages.
    - [route_hint] Do not introduce a bespoke basket spec name such as `HimalayaBasketSpec` or reimplement rank/remove/aggregate logic inline.
    - [historical_note] Basket Monte Carlo routing should use generic basket semantics and snapshot-based state instead of a product-specific mountain-range branch.
  - Required adapters:
    - `reuse_shared_basket_market_binding`
    - `reuse_shared_basket_mc_route_helper`
  - Instruction precedence: follow the approved modules, primitives, and route helper in this plan. If older guidance conflicts, treat it as stale and obey this plan.
- Every `trellis.*` import in the generated code MUST come from the approved module list above.
- If you need functionality outside the approved list, say so explicitly instead of inventing an import.
      - `trace_type`: analytical