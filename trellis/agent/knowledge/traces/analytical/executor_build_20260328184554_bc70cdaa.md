# Analytical Trace: `executor_build_20260328184554_bc70cdaa`
- Trace type: `analytical`
- Route family: `analytical`
- Route name: `quanto_adjustment_analytical`
- Model: `unknown`
- Status: `ok`
- Created at: `2026-03-28T18:45:54.407466+00:00`
- Updated at: `2026-03-28T18:45:54.548824+00:00`
- Task ID: `executor_build_20260328184554_bc70cdaa`

## Context
- `class_name`: 'QuantoOptionAnalyticalPayoff'
- `generation_plan`: {'method': 'analytical', 'instrument_type': 'quanto_option', 'inspected_modules': ['trellis.models.black', 'trellis.models.monte_carlo.engine', 'trellis.models.processes.correlated_gbm'], 'approved_modules': ['trellis.core.date_utils', 'trellis.core.differentiable', 'trellis.core.market_state', 'trellis.core.payoff', 'trellis.core.types', 'trellis.models.analytical', 'trellis.models.analytical.barrier', 'trellis.models.analytical.jamshidian', 'trellis.models.analytical.quanto', 'trellis.models.analytical.support.cross_asset', 'trellis.models.analytical.support.discounting', 'trellis.models.analytical.support.forwards', 'trellis.models.analytical.support.payoffs', 'trellis.models.black', 'trellis.models.monte_carlo.engine', 'trellis.models.monte_carlo.quanto', 'trellis.models.processes.correlated_gbm', 'trellis.models.resolution.quanto'], 'symbols_to_reuse': ['CashflowSchedule', 'Cashflows', 'CorrelatedGBM', 'DataProvider', 'DayCountConvention', 'DeterministicCashflowPayoff', 'DiscountCurve', 'Frequency', 'Instrument', 'MarketState', 'MissingCapabilityError', 'MonteCarloEngine', 'MonteCarloPathPayoff', 'Payoff', 'PresentValue', 'PricingResult', 'QuantoAnalyticalSpecLike', 'QuantoMonteCarloSpecLike', 'QuantoSpecLike', 'ResolvedBarrierInputs', 'ResolvedInputPayoff', 'ResolvedJamshidianInputs', 'ResolvedQuantoInputs', 'add_months', 'asset_or_nothing_intrinsic', 'barrier_image_raw', 'barrier_option_price', 'barrier_regime_selector_raw', 'black76_asset_or_nothing_call', 'black76_asset_or_nothing_put', 'black76_call', 'black76_cash_or_nothing_call', 'black76_cash_or_nothing_put', 'black76_put', 'build_quanto_mc_initial_state', 'build_quanto_mc_process', 'call_put_parity_gap', 'cash_or_nothing_intrinsic', 'continuous_rate_from_simple_rate', 'discount_factor_from_zero_rate', 'discounted_value', 'down_and_in_call', 'down_and_in_call_raw', 'down_and_out_call', 'down_and_out_call_raw', 'effective_covariance_term', 'exchange_option_effective_vol', 'foreign_to_domestic_forward_bridge', 'forward_discount_ratio', 'forward_from_carry_rate', 'forward_from_discount_factors', 'forward_from_dividend_yield', 'garman_kohlhagen_call', 'garman_kohlhagen_put', 'generate_schedule', 'get_accrual_fraction', 'get_bracketing_dates', 'get_numpy', 'gradient', 'hessian', 'implied_zero_rate', 'normalized_option_type', 'price_quanto_option_analytical', 'price_quanto_option_monte_carlo', 'price_quanto_option_raw', 'quanto_adjusted_forward', 'rebate_raw', 'recommended_quanto_mc_engine_kwargs', 'resolve_quanto_correlation', 'resolve_quanto_foreign_curve', 'resolve_quanto_inputs', 'resolve_quanto_underlier_spot', 'safe_time_fraction', 'simple_rate_from_discount_factor', 'terminal_intrinsic', 'terminal_quanto_option_payoff', 'terminal_vanilla_from_basis', 'vanilla_call_raw', 'year_fraction', 'zcb_option_hw', 'zcb_option_hw_raw'], 'proposed_tests': ['tests/test_agent/test_build_loop.py'], 'uncertainty_flags': [], 'repo_revision': '2204f054d09cfaa88de56326480925626f4ca2b3', 'primitive_plan': {'route': 'quanto_adjustment_analytical', 'engine_family': 'unknown', 'score': 2.0}}
- `route_card`: '## Structured Route Card\n- Method family: `analytical`\n- Instrument type: `quanto_option`\n- Route: `quanto_adjustment_analytical`\n- Engine family: `unknown`\n- Required primitives:\n  - `trellis.models.resolution.quanto.resolve_quanto_inputs` (market_binding)\n  - `trellis.models.black.black76_call` (pricing_kernel)\n  - `trellis.models.black.black76_put` (pricing_kernel)\n  - `trellis.models.analytical.quanto.price_quanto_option_analytical` (route_helper)\n- Resolved instructions:\n  - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n  - [historical_note] Quanto analytical routing should preserve domestic payout semantics instead of degrading to vanilla FX pricing.\n  - [historical_note] Resolve shared market inputs first, then price through the checked-in quanto analytical helper.\n- Required adapters:\n  - `reuse_shared_quanto_market_binding`\n  - `apply_quanto_adjustment_terms`\n  - `reuse_shared_quanto_analytical_route_helper`\n- Primary modules to inspect/reuse:\n  - `trellis.models.black`\n  - `trellis.models.monte_carlo.engine`\n  - `trellis.models.processes.correlated_gbm`\n- Post-build test targets:\n  - `tests/test_agent/test_build_loop.py`\n- Instruction precedence: follow the approved modules, primitives, and route helper in this card. If older guidance conflicts, treat it as stale and obey this plan.\n- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels.'
- `route_plan`: '## Structured Generation Plan\n- Method family: `analytical`\n- Instrument type: `quanto_option`\n- Repo revision: `2204f054d09cfaa88de56326480925626f4ca2b3`\n- Inspected modules:\n  - `trellis.models.black`\n  - `trellis.models.monte_carlo.engine`\n  - `trellis.models.processes.correlated_gbm`\n- Approved Trellis modules for imports:\n  - `trellis.core.date_utils`\n  - `trellis.core.differentiable`\n  - `trellis.core.market_state`\n  - `trellis.core.payoff`\n  - `trellis.core.types`\n  - `trellis.models.analytical`\n  - `trellis.models.analytical.barrier`\n  - `trellis.models.analytical.jamshidian`\n  - `trellis.models.analytical.quanto`\n  - `trellis.models.analytical.support.cross_asset`\n  - `trellis.models.analytical.support.discounting`\n  - `trellis.models.analytical.support.forwards`\n  - `trellis.models.analytical.support.payoffs`\n  - `trellis.models.black`\n  - `trellis.models.monte_carlo.engine`\n  - `trellis.models.monte_carlo.quanto`\n  - `trellis.models.processes.correlated_gbm`\n  - `trellis.models.resolution.quanto`\n- Public symbols available from the approved modules:\n  - `CashflowSchedule`\n  - `Cashflows`\n  - `CorrelatedGBM`\n  - `DataProvider`\n  - `DayCountConvention`\n  - `DeterministicCashflowPayoff`\n  - `DiscountCurve`\n  - `Frequency`\n  - `Instrument`\n  - `MarketState`\n  - `MissingCapabilityError`\n  - `MonteCarloEngine`\n  - `MonteCarloPathPayoff`\n  - `Payoff`\n  - `PresentValue`\n  - `PricingResult`\n  - `QuantoAnalyticalSpecLike`\n  - `QuantoMonteCarloSpecLike`\n  - `QuantoSpecLike`\n  - `ResolvedBarrierInputs`\n  - `ResolvedInputPayoff`\n  - `ResolvedJamshidianInputs`\n  - `ResolvedQuantoInputs`\n  - `add_months`\n  - `asset_or_nothing_intrinsic`\n  - `barrier_image_raw`\n  - `barrier_option_price`\n  - `barrier_regime_selector_raw`\n  - `black76_asset_or_nothing_call`\n  - `black76_asset_or_nothing_put`\n  - `black76_call`\n  - `black76_cash_or_nothing_call`\n  - `black76_cash_or_nothing_put`\n  - `black76_put`\n  - `build_quanto_mc_initial_state`\n  - `build_quanto_mc_process`\n  - `call_put_parity_gap`\n  - `cash_or_nothing_intrinsic`\n  - `continuous_rate_from_simple_rate`\n  - `discount_factor_from_zero_rate`\n  - `discounted_value`\n  - `down_and_in_call`\n  - `down_and_in_call_raw`\n  - `down_and_out_call`\n  - `down_and_out_call_raw`\n  - `effective_covariance_term`\n  - `exchange_option_effective_vol`\n  - `foreign_to_domestic_forward_bridge`\n  - `forward_discount_ratio`\n  - `forward_from_carry_rate`\n  - `forward_from_discount_factors`\n  - `forward_from_dividend_yield`\n  - `garman_kohlhagen_call`\n  - `garman_kohlhagen_put`\n  - `generate_schedule`\n  - `get_accrual_fraction`\n  - `get_bracketing_dates`\n  - `get_numpy`\n  - `gradient`\n  - `hessian`\n  - `implied_zero_rate`\n  - `normalized_option_type`\n  - `price_quanto_option_analytical`\n  - `price_quanto_option_monte_carlo`\n  - `price_quanto_option_raw`\n  - `quanto_adjusted_forward`\n  - `rebate_raw`\n  - `recommended_quanto_mc_engine_kwargs`\n  - `resolve_quanto_correlation`\n  - `resolve_quanto_foreign_curve`\n  - `resolve_quanto_inputs`\n  - `resolve_quanto_underlier_spot`\n  - `safe_time_fraction`\n  - `simple_rate_from_discount_factor`\n  - `terminal_intrinsic`\n  - `terminal_quanto_option_payoff`\n  - `terminal_vanilla_from_basis`\n  - `vanilla_call_raw`\n  - `year_fraction`\n  - `zcb_option_hw`\n- Tests to run after generation:\n  - `tests/test_agent/test_build_loop.py`\n- Primitive route:\n  - Route: `quanto_adjustment_analytical`\n  - Engine family: `unknown`\n  - Route score: `2.00`\n  - Selected primitives:\n    - `trellis.models.resolution.quanto.resolve_quanto_inputs` (market_binding)\n    - `trellis.models.black.black76_call` (pricing_kernel)\n    - `trellis.models.black.black76_put` (pricing_kernel)\n    - `trellis.models.analytical.quanto.price_quanto_option_analytical` (route_helper)\n  - Resolved instructions:\n    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n    - [historical_note] Quanto analytical routing should preserve domestic payout semantics instead of degrading to vanilla FX pricing.\n    - [historical_note] Resolve shared market inputs first, then price through the checked-in quanto analytical helper.\n  - Required adapters:\n    - `reuse_shared_quanto_market_binding`\n    - `apply_quanto_adjustment_terms`\n    - `reuse_shared_quanto_analytical_route_helper`\n  - Instruction precedence: follow the approved modules, primitives, and route helper in this plan. If older guidance conflicts, treat it as stale and obey this plan.\n- Every `trellis.*` import in the generated code MUST come from the approved module list above.\n- If you need functionality outside the approved list, say so explicitly instead of inventing an import.'
- `selected_curve_names`: {}
- `spec_name`: 'QuantoOptionSpec'

## Steps
- **trace** `executor_build_20260328184554_bc70cdaa:root`
  - Label: Analytical build
  - Status: `ok`
  - Notes:
    - The trace mirrors the deterministic GenerationPlan used to assemble the route.
  - Inputs:
    - `issue_id`: None
    - `model`: unknown
    - `route_family`: analytical
    - `route_name`: quanto_adjustment_analytical
    - `task_id`: executor_build_20260328184554_bc70cdaa
  - Outputs:
    - `route`: {
  "family": "analytical",
  "model": "unknown",
  "name": "quanto_adjustment_analytical"
}
    - `status`: ok
  - **semantic_resolution** `executor_build_20260328184554_bc70cdaa:semantic_resolution`
    - Label: Resolve contract and route
    - Status: `ok`
    - Parent: `executor_build_20260328184554_bc70cdaa:root`
    - Notes:
      - Record the semantic contract that drives route selection, not just the final code path.
    - Inputs:
      - `approved_modules`: [
  "trellis.core.date_utils",
  "trellis.core.differentiable",
  "trellis.core.market_state",
  "trellis.core.payoff",
  "trellis.core.types",
  "trellis.models.analytical",
  "trellis.models.analytical.barrier",
  "trellis.models.analytical.jamshidian",
  "trellis.models.analytical.quanto",
  "trellis.models.analytical.support.cross_asset",
  "trellis.models.analytical.support.discounting",
  "trellis.models.analytical.support.forwards",
  "trellis.models.analytical.support.payoffs",
  "trellis.models.black",
  "trellis.models.monte_carlo.engine",
  "trellis.models.monte_carlo.quanto",
  "trellis.models.processes.correlated_gbm",
  "trellis.models.resolution.quanto"
]
      - `inspected_modules`: [
  "trellis.models.black",
  "trellis.models.monte_carlo.engine",
  "trellis.models.processes.correlated_gbm"
]
      - `instrument_type`: quanto_option
      - `method`: analytical
      - `repo_revision`: 2204f054d09cfaa88de56326480925626f4ca2b3
      - `symbols_to_reuse`: [
  "CashflowSchedule",
  "Cashflows",
  "CorrelatedGBM",
  "DataProvider",
  "DayCountConvention",
  "DeterministicCashflowPayoff",
  "DiscountCurve",
  "Frequency",
  "Instrument",
  "MarketState",
  "MissingCapabilityError",
  "MonteCarloEngine",
  "MonteCarloPathPayoff",
  "Payoff",
  "PresentValue",
  "PricingResult",
  "QuantoAnalyticalSpecLike",
  "QuantoMonteCarloSpecLike",
  "QuantoSpecLike",
  "ResolvedBarrierInputs",
  "ResolvedInputPayoff",
  "ResolvedJamshidianInputs",
  "ResolvedQuantoInputs",
  "add_months",
  "asset_or_nothing_intrinsic",
  "barrier_image_raw",
  "barrier_option_price",
  "barrier_regime_selector_raw",
  "black76_asset_or_nothing_call",
  "black76_asset_or_nothing_put",
  "black76_call",
  "black76_cash_or_nothing_call",
  "black76_cash_or_nothing_put",
  "black76_put",
  "build_quanto_mc_initial_state",
  "build_quanto_mc_process",
  "call_put_parity_gap",
  "cash_or_nothing_intrinsic",
  "continuous_rate_from_simple_rate",
  "discount_factor_from_zero_rate"
]
      - `uncertainty_flags`: []
    - Outputs:
      - `model`: unknown
      - `primitive_plan_score`: 2.0
      - `route_family`: analytical
      - `route_name`: quanto_adjustment_analytical
  - **decomposition** `executor_build_20260328184554_bc70cdaa:decomposition`
    - Label: Select reusable kernels
    - Status: `ok`
    - Parent: `executor_build_20260328184554_bc70cdaa:root`
    - Notes:
      - Capture the reusable valuation components and any exact basis-claim assembly.
    - Inputs:
      - `adapters`: [
  "reuse_shared_quanto_market_binding",
  "apply_quanto_adjustment_terms",
  "reuse_shared_quanto_analytical_route_helper"
]
      - `blockers`: []
      - `notes`: [
  "Quanto analytical routing should preserve domestic payout semantics instead of degrading to vanilla FX pricing.",
  "Resolve shared market inputs first, then price through the checked-in quanto analytical helper."
]
      - `primitives`: [
  {
    "module": "trellis.models.resolution.quanto",
    "required": true,
    "role": "market_binding",
    "symbol": "resolve_quanto_inputs"
  },
  {
    "module": "trellis.models.black",
    "required": true,
    "role": "pricing_kernel",
    "symbol": "black76_call"
  },
  {
    "module": "trellis.models.black",
    "required": true,
    "role": "pricing_kernel",
    "symbol": "black76_put"
  },
  {
    "module": "trellis.models.analytical.quanto",
    "required": true,
    "role": "route_helper",
    "symbol": "price_quanto_option_analytical"
  }
]
    - Outputs:
      - `reuse_decision`: exact_decomposition
      - `selected_primitives`: [
  {
    "module": "trellis.models.resolution.quanto",
    "required": true,
    "role": "market_binding",
    "symbol": "resolve_quanto_inputs"
  },
  {
    "module": "trellis.models.black",
    "required": true,
    "role": "pricing_kernel",
    "symbol": "black76_call"
  },
  {
    "module": "trellis.models.black",
    "required": true,
    "role": "pricing_kernel",
    "symbol": "black76_put"
  },
  {
    "module": "trellis.models.analytical.quanto",
    "required": true,
    "role": "route_helper",
    "symbol": "price_quanto_option_analytical"
  }
]
  - **assembly** `executor_build_20260328184554_bc70cdaa:assembly`
    - Label: Assemble route from kernels
    - Status: `ok`
    - Parent: `executor_build_20260328184554_bc70cdaa:root`
    - Notes:
      - Prefer thin orchestration around existing analytical kernels and route helpers.
    - Inputs:
      - `adapters`: [
  "reuse_shared_quanto_market_binding",
  "apply_quanto_adjustment_terms",
  "reuse_shared_quanto_analytical_route_helper"
]
      - `approved_modules`: [
  "trellis.core.date_utils",
  "trellis.core.differentiable",
  "trellis.core.market_state",
  "trellis.core.payoff",
  "trellis.core.types",
  "trellis.models.analytical",
  "trellis.models.analytical.barrier",
  "trellis.models.analytical.jamshidian",
  "trellis.models.analytical.quanto",
  "trellis.models.analytical.support.cross_asset",
  "trellis.models.analytical.support.discounting",
  "trellis.models.analytical.support.forwards",
  "trellis.models.analytical.support.payoffs",
  "trellis.models.black",
  "trellis.models.monte_carlo.engine",
  "trellis.models.monte_carlo.quanto",
  "trellis.models.processes.correlated_gbm",
  "trellis.models.resolution.quanto"
]
      - `route_helper`: trellis.models.analytical.quanto.price_quanto_option_analytical
    - Outputs:
      - `assembly_card`: ## Structured Route Card
- Method family: `analytical`
- Instrument type: `quanto_option`
- Route: `quanto_adjustment_analytical`
- Engine family: `unknown`
- Required primitives:
  - `trellis.models.resolution.quanto.resolve_quanto_inputs` (market_binding)
  - `trellis.models.black.black76_call` (pricing_kernel)
  - `trellis.models.black.black76_put` (pricing_kernel)
  - `trellis.models.analytical.quanto.price_quanto_option_analytical` (route_helper)
- Resolved instructions:
  - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
  - [historical_note] Quanto analytical routing should preserve domestic payout semantics instead of degrading to vanilla FX pricing.
  - [historical_note] Resolve shared market inputs first, then price through the checked-in quanto analytical helper.
- Required adapters:
  - `reuse_shared_quanto_market_binding`
  - `apply_quanto_adjustment_terms`
  - `reuse_shared_quanto_analytical_route_helper`
- Primary modules to inspect/reuse:
  - `trellis.models.black`
  - `trellis.models.monte_carlo.engine`
  - `trellis.models.processes.correlated_gbm`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the approved modules, primitives, and route helper in this card. If older guidance conflicts, treat it as stale and obey this plan.
- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels.
      - `helper_modules`: [
  "trellis.models.resolution.quanto",
  "trellis.models.black",
  "trellis.models.black",
  "trellis.models.analytical.quanto"
]
      - `route_helper`: trellis.models.analytical.quanto.price_quanto_option_analytical
  - **validation** `executor_build_20260328184554_bc70cdaa:validation`
    - Label: Validate route and fallbacks
    - Status: `ok`
    - Parent: `executor_build_20260328184554_bc70cdaa:root`
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
  - **output** `executor_build_20260328184554_bc70cdaa:output`
    - Label: Final analytical artifact
    - Status: `ok`
    - Parent: `executor_build_20260328184554_bc70cdaa:root`
    - Notes:
      - Persist both the machine-readable trace and the text rendering from the same source of truth.
    - Inputs:
      - `model`: unknown
      - `route`: quanto_adjustment_analytical
    - Outputs:
      - `route_card`: ## Structured Route Card
- Method family: `analytical`
- Instrument type: `quanto_option`
- Route: `quanto_adjustment_analytical`
- Engine family: `unknown`
- Required primitives:
  - `trellis.models.resolution.quanto.resolve_quanto_inputs` (market_binding)
  - `trellis.models.black.black76_call` (pricing_kernel)
  - `trellis.models.black.black76_put` (pricing_kernel)
  - `trellis.models.analytical.quanto.price_quanto_option_analytical` (route_helper)
- Resolved instructions:
  - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
  - [historical_note] Quanto analytical routing should preserve domestic payout semantics instead of degrading to vanilla FX pricing.
  - [historical_note] Resolve shared market inputs first, then price through the checked-in quanto analytical helper.
- Required adapters:
  - `reuse_shared_quanto_market_binding`
  - `apply_quanto_adjustment_terms`
  - `reuse_shared_quanto_analytical_route_helper`
- Primary modules to inspect/reuse:
  - `trellis.models.black`
  - `trellis.models.monte_carlo.engine`
  - `trellis.models.processes.correlated_gbm`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the approved modules, primitives, and route helper in this card. If older guidance conflicts, treat it as stale and obey this plan.
- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels.
      - `route_plan`: ## Structured Generation Plan
- Method family: `analytical`
- Instrument type: `quanto_option`
- Repo revision: `2204f054d09cfaa88de56326480925626f4ca2b3`
- Inspected modules:
  - `trellis.models.black`
  - `trellis.models.monte_carlo.engine`
  - `trellis.models.processes.correlated_gbm`
- Approved Trellis modules for imports:
  - `trellis.core.date_utils`
  - `trellis.core.differentiable`
  - `trellis.core.market_state`
  - `trellis.core.payoff`
  - `trellis.core.types`
  - `trellis.models.analytical`
  - `trellis.models.analytical.barrier`
  - `trellis.models.analytical.jamshidian`
  - `trellis.models.analytical.quanto`
  - `trellis.models.analytical.support.cross_asset`
  - `trellis.models.analytical.support.discounting`
  - `trellis.models.analytical.support.forwards`
  - `trellis.models.analytical.support.payoffs`
  - `trellis.models.black`
  - `trellis.models.monte_carlo.engine`
  - `trellis.models.monte_carlo.quanto`
  - `trellis.models.processes.correlated_gbm`
  - `trellis.models.resolution.quanto`
- Public symbols available from the approved modules:
  - `CashflowSchedule`
  - `Cashflows`
  - `CorrelatedGBM`
  - `DataProvider`
  - `DayCountConvention`
  - `DeterministicCashflowPayoff`
  - `DiscountCurve`
  - `Frequency`
  - `Instrument`
  - `MarketState`
  - `MissingCapabilityError`
  - `MonteCarloEngine`
  - `MonteCarloPathPayoff`
  - `Payoff`
  - `PresentValue`
  - `PricingResult`
  - `QuantoAnalyticalSpecLike`
  - `QuantoMonteCarloSpecLike`
  - `QuantoSpecLike`
  - `ResolvedBarrierInputs`
  - `ResolvedInputPayoff`
  - `ResolvedJamshidianInputs`
  - `ResolvedQuantoInputs`
  - `add_months`
  - `asset_or_nothing_intrinsic`
  - `barrier_image_raw`
  - `barrier_option_price`
  - `barrier_regime_selector_raw`
  - `black76_asset_or_nothing_call`
  - `black76_asset_or_nothing_put`
  - `black76_call`
  - `black76_cash_or_nothing_call`
  - `black76_cash_or_nothing_put`
  - `black76_put`
  - `build_quanto_mc_initial_state`
  - `build_quanto_mc_process`
  - `call_put_parity_gap`
  - `cash_or_nothing_intrinsic`
  - `continuous_rate_from_simple_rate`
  - `discount_factor_from_zero_rate`
  - `discounted_value`
  - `down_and_in_call`
  - `down_and_in_call_raw`
  - `down_and_out_call`
  - `down_and_out_call_raw`
  - `effective_covariance_term`
  - `exchange_option_effective_vol`
  - `foreign_to_domestic_forward_bridge`
  - `forward_discount_ratio`
  - `forward_from_carry_rate`
  - `forward_from_discount_factors`
  - `forward_from_dividend_yield`
  - `garman_kohlhagen_call`
  - `garman_kohlhagen_put`
  - `generate_schedule`
  - `get_accrual_fraction`
  - `get_bracketing_dates`
  - `get_numpy`
  - `gradient`
  - `hessian`
  - `implied_zero_rate`
  - `normalized_option_type`
  - `price_quanto_option_analytical`
  - `price_quanto_option_monte_carlo`
  - `price_quanto_option_raw`
  - `quanto_adjusted_forward`
  - `rebate_raw`
  - `recommended_quanto_mc_engine_kwargs`
  - `resolve_quanto_correlation`
  - `resolve_quanto_foreign_curve`
  - `resolve_quanto_inputs`
  - `resolve_quanto_underlier_spot`
  - `safe_time_fraction`
  - `simple_rate_from_discount_factor`
  - `terminal_intrinsic`
  - `terminal_quanto_option_payoff`
  - `terminal_vanilla_from_basis`
  - `vanilla_call_raw`
  - `year_fraction`
  - `zcb_option_hw`
- Tests to run after generation:
  - `tests/test_agent/test_build_loop.py`
- Primitive route:
  - Route: `quanto_adjustment_analytical`
  - Engine family: `unknown`
  - Route score: `2.00`
  - Selected primitives:
    - `trellis.models.resolution.quanto.resolve_quanto_inputs` (market_binding)
    - `trellis.models.black.black76_call` (pricing_kernel)
    - `trellis.models.black.black76_put` (pricing_kernel)
    - `trellis.models.analytical.quanto.price_quanto_option_analytical` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
    - [historical_note] Quanto analytical routing should preserve domestic payout semantics instead of degrading to vanilla FX pricing.
    - [historical_note] Resolve shared market inputs first, then price through the checked-in quanto analytical helper.
  - Required adapters:
    - `reuse_shared_quanto_market_binding`
    - `apply_quanto_adjustment_terms`
    - `reuse_shared_quanto_analytical_route_helper`
  - Instruction precedence: follow the approved modules, primitives, and route helper in this plan. If older guidance conflicts, treat it as stale and obey this plan.
- Every `trellis.*` import in the generated code MUST come from the approved module list above.
- If you need functionality outside the approved list, say so explicitly instead of inventing an import.
      - `trace_type`: analytical