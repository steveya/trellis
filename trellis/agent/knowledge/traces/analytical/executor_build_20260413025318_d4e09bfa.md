# Analytical Trace: `executor_build_20260413025318_d4e09bfa`
- Trace type: `analytical`
- Route family: `analytical`
- Route name: `analytical_black76`
- Model: `analytical`
- Status: `warning`
- Created at: `2026-04-13T02:53:19.184495+00:00`
- Updated at: `2026-04-13T02:53:19.227962+00:00`
- Task ID: `executor_build_20260413025318_d4e09bfa`

## Context
- `class_name`: 'SwaptionPayoff'
- `generation_plan`: {'method': 'analytical', 'instrument_type': None, 'inspected_modules': ['trellis.models.black'], 'approved_modules': ['trellis.core.date_utils', 'trellis.core.differentiable', 'trellis.core.market_state', 'trellis.core.payoff', 'trellis.core.types', 'trellis.models.analytical', 'trellis.models.analytical.barrier', 'trellis.models.analytical.fx', 'trellis.models.analytical.jamshidian', 'trellis.models.analytical.quanto', 'trellis.models.analytical.support.cross_asset', 'trellis.models.analytical.support.discounting', 'trellis.models.analytical.support.forwards', 'trellis.models.analytical.support.payoffs', 'trellis.models.black', 'trellis.models.rate_style_swaption'], 'symbols_to_reuse': ['BermudanSwaptionLowerBoundSpecLike', 'CashflowSchedule', 'Cashflows', 'ContractTimeline', 'DataProvider', 'DayCountConvention', 'DeterministicCashflowPayoff', 'DiscountCurve', 'DslMeasure', 'EuropeanSwaptionSpecLike', 'EventSchedule', 'Frequency', 'Instrument', 'MarketState', 'MissingCapabilityError', 'MonteCarloPathPayoff', 'Payoff', 'PresentValue', 'PricingResult', 'QuantoAnalyticalSpecLike', 'RateStyleSwaptionSpecLike', 'ResolvedBarrierInputs', 'ResolvedGarmanKohlhagenInputs', 'ResolvedInputPayoff', 'ResolvedJamshidianInputs', 'ResolvedSwaptionBlack76Inputs', 'SchedulePeriod', 'TimelineRole', 'add_months', 'asset_or_nothing_intrinsic', 'barrier_image_raw', 'barrier_option_price', 'barrier_regime_selector_raw', 'black76_asset_or_nothing_call', 'black76_asset_or_nothing_put', 'black76_call', 'black76_cash_or_nothing_call', 'black76_cash_or_nothing_put', 'black76_put', 'build_contract_timeline', 'build_contract_timeline_from_dates', 'build_exercise_timeline_from_dates', 'build_observation_timeline', 'build_payment_timeline', 'build_period_schedule', 'call_put_parity_gap', 'cash_or_nothing_intrinsic', 'coerce_contract_timeline_from_dates', 'continuous_rate_from_simple_rate', 'discount_factor_from_zero_rate', 'discounted_value', 'down_and_in_call', 'down_and_in_call_raw', 'down_and_out_call', 'down_and_out_call_raw', 'effective_covariance_term', 'exchange_option_effective_vol', 'foreign_to_domestic_forward_bridge', 'forward_discount_ratio', 'forward_from_carry_rate', 'forward_from_discount_factors', 'forward_from_dividend_yield', 'garman_kohlhagen_call', 'garman_kohlhagen_call_raw', 'garman_kohlhagen_price_raw', 'garman_kohlhagen_put', 'garman_kohlhagen_put_raw', 'generate_schedule', 'get_accrual_fraction', 'get_bracketing_dates', 'get_numpy', 'gradient', 'hessian', 'implied_zero_rate', 'jacobian', 'normalize_dsl_measure', 'normalize_explicit_dates', 'normalized_option_type', 'price_bermudan_swaption_black76_lower_bound', 'price_quanto_option_analytical', 'price_quanto_option_raw', 'price_swaption_black76', 'price_swaption_black76_raw', 'price_swaption_monte_carlo', 'quanto_adjusted_forward', 'rebate_raw', 'resolve_swaption_black76_inputs', 'resolve_swaption_curve_basis_spread', 'resolve_swaption_monte_carlo_problem', 'safe_time_fraction', 'simple_rate_from_discount_factor', 'terminal_intrinsic', 'terminal_vanilla_from_basis', 'vanilla_call_raw', 'year_fraction', 'zcb_option_hw', 'zcb_option_hw_raw'], 'proposed_tests': ['tests/test_agent/test_build_loop.py'], 'uncertainty_flags': ['instrument_type_not_provided'], 'repo_revision': '3e05e8e455bb894825c363ce38d73be7f6ef7c15', 'instruction_resolution': {'route': 'analytical_black76', 'effective_instruction_count': 1, 'dropped_instruction_count': 0, 'conflict_count': 0, 'effective_instructions': [{'id': 'analytical_black76:route-helper', 'title': 'Use the selected route helper directly', 'instruction_type': 'hard_constraint', 'status': 'active', 'source_kind': 'route_card', 'source_id': 'analytical_black76:route-helper', 'source_revision': '', 'scope_methods': ('analytical',), 'scope_instruments': (), 'scope_routes': ('analytical_black76',), 'scope_modules': ('trellis.models.rate_style_swaption',), 'scope_features': (), 'precedence_rank': 100, 'supersedes': (), 'conflict_policy': 'prefer_newer', 'statement': 'Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.', 'rationale': 'The helper already owns the route-specific engine and payoff mapping.', 'created_at': '', 'updated_at': ''}], 'dropped_instructions': [], 'conflicts': []}, 'primitive_plan': {'route': 'analytical_black76', 'engine_family': 'analytical', 'route_family': 'analytical', 'score': 5.75}}
- `route_card`: "## Structured Lane Card\n- Method family: `analytical`\n- Instrument type: `unknown`\n- Lane boundary: family=`analytical`, kind=`exact_target_binding`, timeline_roles=`payment`, exact_bindings=`trellis.models.rate_style_swaption.price_swaption_black76`\n- Lane obligations:\n  - Lane family: `analytical`\n  - Plan kind: `exact_target_binding`\n  - Timeline roles: `payment`\n  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`\n  - State obligations: `foreign_discount_curve`, `expiry_black_vol`\n  - Construction steps:\n    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.\n    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.\n  - Exact backend bindings:\n    - `trellis.models.rate_style_swaption.price_swaption_black76`\n  - Exact binding signatures:\n    - `price_swaption_black76(market_state: 'MarketState', spec: 'EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike', *, expiry_date: 'date | None' = None, mean_reversion: 'float | None' = None, sigma: 'float | None' = None) -> 'float'`\n- Route authority:\n  - binding=`trellis.models.rate_style_swaption.price_swaption_black76`, engine=`analytical`, authority=`exact_backend_fit`\n  - Validation bundle: `analytical:swaption`\n  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_rate_style_swaption_helper_consistency`\n  - Canary coverage: canaries=`T73`\n  - Helper authority: `trellis.models.rate_style_swaption.price_swaption_black76`\n  - Exact target bindings: `trellis.models.rate_style_swaption.price_swaption_black76`\n- Backend binding:\n  - Route: `analytical_black76`\n  - Engine family: `analytical`\n  - Route family: `analytical`\n  - Selected primitives:\n    - `trellis.models.rate_style_swaption.price_swaption_black76` (route_helper)\n  - Resolved instructions:\n    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n- Primary modules to inspect/reuse:\n  - `trellis.models.black`\n- Post-build test targets:\n  - `tests/test_agent/test_build_loop.py`\n- Instruction precedence: follow the lane obligations in this card first. Treat backend route/helper details as exact-fit bindings, not as permission to invent a different numerical path.\n- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan.\n- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires."
- `route_plan`: "## Structured Generation Plan\n- Method family: `analytical`\n- Instrument type: `unknown`\n- Lane boundary: family=`analytical`, kind=`exact_target_binding`, timeline_roles=`payment`, exact_bindings=`trellis.models.rate_style_swaption.price_swaption_black76`\n- Lane obligations:\n  - Lane family: `analytical`\n  - Plan kind: `exact_target_binding`\n  - Timeline roles: `payment`\n  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`\n  - State obligations: `foreign_discount_curve`, `expiry_black_vol`\n  - Construction steps:\n    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.\n    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.\n  - Exact backend bindings:\n    - `trellis.models.rate_style_swaption.price_swaption_black76`\n  - Exact binding signatures:\n    - `price_swaption_black76(market_state: 'MarketState', spec: 'EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike', *, expiry_date: 'date | None' = None, mean_reversion: 'float | None' = None, sigma: 'float | None' = None) -> 'float'`\n- Route authority:\n  - binding=`trellis.models.rate_style_swaption.price_swaption_black76`, engine=`analytical`, authority=`exact_backend_fit`\n  - Validation bundle: `analytical:swaption`\n  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_rate_style_swaption_helper_consistency`\n  - Canary coverage: canaries=`T73`\n  - Helper authority: `trellis.models.rate_style_swaption.price_swaption_black76`\n  - Exact target bindings: `trellis.models.rate_style_swaption.price_swaption_black76`\n- Repo revision: `3e05e8e455bb894825c363ce38d73be7f6ef7c15`\n- Inspected modules:\n  - `trellis.models.black`\n- Approved Trellis modules for imports:\n  - `trellis.core.date_utils`\n  - `trellis.core.differentiable`\n  - `trellis.core.market_state`\n  - `trellis.core.payoff`\n  - `trellis.core.types`\n  - `trellis.models.analytical`\n  - `trellis.models.analytical.barrier`\n  - `trellis.models.analytical.fx`\n  - `trellis.models.analytical.jamshidian`\n  - `trellis.models.analytical.quanto`\n  - `trellis.models.analytical.support.cross_asset`\n  - `trellis.models.analytical.support.discounting`\n  - `trellis.models.analytical.support.forwards`\n  - `trellis.models.analytical.support.payoffs`\n  - `trellis.models.black`\n  - `trellis.models.rate_style_swaption`\n- Public symbols available from the approved modules:\n  - `BermudanSwaptionLowerBoundSpecLike`\n  - `CashflowSchedule`\n  - `Cashflows`\n  - `ContractTimeline`\n  - `DataProvider`\n  - `DayCountConvention`\n  - `DeterministicCashflowPayoff`\n  - `DiscountCurve`\n  - `DslMeasure`\n  - `EuropeanSwaptionSpecLike`\n  - `EventSchedule`\n  - `Frequency`\n  - `Instrument`\n  - `MarketState`\n  - `MissingCapabilityError`\n  - `MonteCarloPathPayoff`\n  - `Payoff`\n  - `PresentValue`\n  - `PricingResult`\n  - `QuantoAnalyticalSpecLike`\n  - `RateStyleSwaptionSpecLike`\n  - `ResolvedBarrierInputs`\n  - `ResolvedGarmanKohlhagenInputs`\n  - `ResolvedInputPayoff`\n- Tests to run after generation:\n  - `tests/test_agent/test_build_loop.py`\n- Likely tests for reused symbols:\n  - `Cashflows` -> tests/test_models/test_contingent_cashflows.py\n- Backend binding:\n  - Route: `analytical_black76`\n  - Engine family: `analytical`\n  - Route family: `analytical`\n  - Route score: `5.75`\n  - Selected primitives:\n    - `trellis.models.rate_style_swaption.price_swaption_black76` (route_helper)\n  - Resolved instructions:\n    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n  - Instruction precedence: follow the compiler-emitted lane obligations first, then satisfy the exact backend binding and approved imports. If older guidance conflicts, treat it as stale and obey this plan.\n- Uncertainty flags:\n  - `instrument_type_not_provided`\n- Every `trellis.*` import in the generated code MUST come from the approved module list above.\n- If you need functionality outside the approved list, say so explicitly instead of inventing an import."
- `selected_curve_names`: {}
- `spec_name`: 'SwaptionSpec'

## Steps
- **trace** `executor_build_20260413025318_d4e09bfa:root`
  - Label: Analytical build
  - Status: `warning`
  - Notes:
    - The trace mirrors the deterministic GenerationPlan used to assemble the route.
  - Inputs:
    - `issue_id`: None
    - `model`: analytical
    - `route_family`: analytical
    - `route_name`: analytical_black76
    - `task_id`: executor_build_20260413025318_d4e09bfa
  - Outputs:
    - `route`: {
  "family": "analytical",
  "model": "analytical",
  "name": "analytical_black76"
}
    - `status`: warning
  - **semantic_resolution** `executor_build_20260413025318_d4e09bfa:semantic_resolution`
    - Label: Resolve contract and route
    - Status: `ok`
    - Parent: `executor_build_20260413025318_d4e09bfa:root`
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
  "trellis.models.analytical.fx",
  "trellis.models.analytical.jamshidian",
  "trellis.models.analytical.quanto",
  "trellis.models.analytical.support.cross_asset",
  "trellis.models.analytical.support.discounting",
  "trellis.models.analytical.support.forwards",
  "trellis.models.analytical.support.payoffs",
  "trellis.models.black",
  "trellis.models.rate_style_swaption"
]
      - `inspected_modules`: [
  "trellis.models.black"
]
      - `instrument_type`: None
      - `method`: analytical
      - `repo_revision`: 3e05e8e455bb894825c363ce38d73be7f6ef7c15
      - `symbols_to_reuse`: [
  "BermudanSwaptionLowerBoundSpecLike",
  "CashflowSchedule",
  "Cashflows",
  "ContractTimeline",
  "DataProvider",
  "DayCountConvention",
  "DeterministicCashflowPayoff",
  "DiscountCurve",
  "DslMeasure",
  "EuropeanSwaptionSpecLike",
  "EventSchedule",
  "Frequency",
  "Instrument",
  "MarketState",
  "MissingCapabilityError",
  "MonteCarloPathPayoff",
  "Payoff",
  "PresentValue",
  "PricingResult",
  "QuantoAnalyticalSpecLike",
  "RateStyleSwaptionSpecLike",
  "ResolvedBarrierInputs",
  "ResolvedGarmanKohlhagenInputs",
  "ResolvedInputPayoff",
  "ResolvedJamshidianInputs",
  "ResolvedSwaptionBlack76Inputs",
  "SchedulePeriod",
  "TimelineRole",
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
  "build_contract_timeline"
]
      - `uncertainty_flags`: [
  "instrument_type_not_provided"
]
    - Outputs:
      - `instruction_resolution`: {
  "conflict_count": 0,
  "conflicts": [],
  "dropped_instruction_count": 0,
  "dropped_instructions": [],
  "effective_instruction_count": 1,
  "effective_instructions": [
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "analytical_black76:route-helper",
      "instruction_type": "hard_constraint",
      "precedence_rank": 100,
      "rationale": "The helper already owns the route-specific engine and payoff mapping.",
      "scope_features": [],
      "scope_instruments": [],
      "scope_methods": [
        "analytical"
      ],
      "scope_modules": [
        "trellis.models.rate_style_swaption"
      ],
      "scope_routes": [
        "analytical_black76"
      ],
      "source_id": "analytical_black76:route-helper",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.",
      "status": "active",
      "supersedes": [],
      "title": "Use the selected route helper directly",
      "updated_at": ""
    }
  ],
  "route": "analytical_black76"
}
      - `model`: analytical
      - `primitive_plan_score`: 5.75
      - `route_family`: analytical
      - `route_name`: analytical_black76
  - **instruction_lifecycle** `executor_build_20260413025318_d4e09bfa:instruction_lifecycle`
    - Label: Resolve route guidance lifecycle
    - Status: `ok`
    - Parent: `executor_build_20260413025318_d4e09bfa:root`
    - Notes:
      - List the effective, dropped, and conflicting route guidance records so replay consumers can see what actually governed the build.
    - Inputs:
      - `conflict_count`: 0
      - `dropped_instruction_count`: 0
      - `effective_instruction_count`: 1
      - `route`: analytical_black76
    - Outputs:
      - `instruction_resolution`: {
  "conflict_count": 0,
  "conflicts": [],
  "dropped_instruction_count": 0,
  "dropped_instructions": [],
  "effective_instruction_count": 1,
  "effective_instructions": [
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "analytical_black76:route-helper",
      "instruction_type": "hard_constraint",
      "precedence_rank": 100,
      "rationale": "The helper already owns the route-specific engine and payoff mapping.",
      "scope_features": [],
      "scope_instruments": [],
      "scope_methods": [
        "analytical"
      ],
      "scope_modules": [
        "trellis.models.rate_style_swaption"
      ],
      "scope_routes": [
        "analytical_black76"
      ],
      "source_id": "analytical_black76:route-helper",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.",
      "status": "active",
      "supersedes": [],
      "title": "Use the selected route helper directly",
      "updated_at": ""
    }
  ],
  "route": "analytical_black76"
}
  - **decomposition** `executor_build_20260413025318_d4e09bfa:decomposition`
    - Label: Select reusable kernels
    - Status: `ok`
    - Parent: `executor_build_20260413025318_d4e09bfa:root`
    - Notes:
      - Capture the reusable valuation components and any exact basis-claim assembly.
    - Inputs:
      - `adapters`: []
      - `blockers`: []
      - `notes`: []
      - `primitives`: [
  {
    "excluded": false,
    "module": "trellis.models.rate_style_swaption",
    "required": true,
    "role": "route_helper",
    "symbol": "price_swaption_black76"
  }
]
    - Outputs:
      - `reuse_decision`: exact_decomposition
      - `selected_primitives`: [
  {
    "excluded": false,
    "module": "trellis.models.rate_style_swaption",
    "required": true,
    "role": "route_helper",
    "symbol": "price_swaption_black76"
  }
]
  - **assembly** `executor_build_20260413025318_d4e09bfa:assembly`
    - Label: Assemble route from kernels
    - Status: `ok`
    - Parent: `executor_build_20260413025318_d4e09bfa:root`
    - Notes:
      - Prefer thin orchestration around existing analytical kernels and route helpers.
    - Inputs:
      - `adapters`: []
      - `approved_modules`: [
  "trellis.core.date_utils",
  "trellis.core.differentiable",
  "trellis.core.market_state",
  "trellis.core.payoff",
  "trellis.core.types",
  "trellis.models.analytical",
  "trellis.models.analytical.barrier",
  "trellis.models.analytical.fx",
  "trellis.models.analytical.jamshidian",
  "trellis.models.analytical.quanto",
  "trellis.models.analytical.support.cross_asset",
  "trellis.models.analytical.support.discounting",
  "trellis.models.analytical.support.forwards",
  "trellis.models.analytical.support.payoffs",
  "trellis.models.black",
  "trellis.models.rate_style_swaption"
]
      - `route_helper`: trellis.models.rate_style_swaption.price_swaption_black76
    - Outputs:
      - `assembly_card`: ## Structured Lane Card
- Method family: `analytical`
- Instrument type: `unknown`
- Lane boundary: family=`analytical`, kind=`exact_target_binding`, timeline_roles=`payment`, exact_bindings=`trellis.models.rate_style_swaption.price_swaption_black76`
- Lane obligations:
  - Lane family: `analytical`
  - Plan kind: `exact_target_binding`
  - Timeline roles: `payment`
  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`
  - State obligations: `foreign_discount_curve`, `expiry_black_vol`
  - Construction steps:
    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.
    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.
  - Exact backend bindings:
    - `trellis.models.rate_style_swaption.price_swaption_black76`
  - Exact binding signatures:
    - `price_swaption_black76(market_state: 'MarketState', spec: 'EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike', *, expiry_date: 'date | None' = None, mean_reversion: 'float | None' = None, sigma: 'float | None' = None) -> 'float'`
- Route authority:
  - binding=`trellis.models.rate_style_swaption.price_swaption_black76`, engine=`analytical`, authority=`exact_backend_fit`
  - Validation bundle: `analytical:swaption`
  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_rate_style_swaption_helper_consistency`
  - Canary coverage: canaries=`T73`
  - Helper authority: `trellis.models.rate_style_swaption.price_swaption_black76`
  - Exact target bindings: `trellis.models.rate_style_swaption.price_swaption_black76`
- Backend binding:
  - Route: `analytical_black76`
  - Engine family: `analytical`
  - Route family: `analytical`
  - Selected primitives:
    - `trellis.models.rate_style_swaption.price_swaption_black76` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
- Primary modules to inspect/reuse:
  - `trellis.models.black`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the lane obligations in this card first. Treat backend route/helper details as exact-fit bindings, not as permission to invent a different numerical path.
- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan.
- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires.
      - `helper_modules`: [
  "trellis.models.rate_style_swaption"
]
      - `route_helper`: trellis.models.rate_style_swaption.price_swaption_black76
  - **validation** `executor_build_20260413025318_d4e09bfa:validation`
    - Label: Validate route and fallbacks
    - Status: `warning`
    - Parent: `executor_build_20260413025318_d4e09bfa:root`
    - Notes:
      - Record proposed tests, blocker state, and any fallback or reuse notes.
    - Inputs:
      - `blockers`: []
      - `proposed_tests`: [
  "tests/test_agent/test_build_loop.py"
]
      - `uncertainty_flags`: [
  "instrument_type_not_provided"
]
      - `validation`: None
    - Outputs:
      - `blocker_report_present`: False
      - `new_primitive_workflow_present`: False
      - `validation_state`: planned
  - **output** `executor_build_20260413025318_d4e09bfa:output`
    - Label: Final analytical artifact
    - Status: `warning`
    - Parent: `executor_build_20260413025318_d4e09bfa:root`
    - Notes:
      - Persist both the machine-readable trace and the text rendering from the same source of truth.
    - Inputs:
      - `model`: analytical
      - `route`: analytical_black76
    - Outputs:
      - `route_card`: ## Structured Lane Card
- Method family: `analytical`
- Instrument type: `unknown`
- Lane boundary: family=`analytical`, kind=`exact_target_binding`, timeline_roles=`payment`, exact_bindings=`trellis.models.rate_style_swaption.price_swaption_black76`
- Lane obligations:
  - Lane family: `analytical`
  - Plan kind: `exact_target_binding`
  - Timeline roles: `payment`
  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`
  - State obligations: `foreign_discount_curve`, `expiry_black_vol`
  - Construction steps:
    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.
    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.
  - Exact backend bindings:
    - `trellis.models.rate_style_swaption.price_swaption_black76`
  - Exact binding signatures:
    - `price_swaption_black76(market_state: 'MarketState', spec: 'EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike', *, expiry_date: 'date | None' = None, mean_reversion: 'float | None' = None, sigma: 'float | None' = None) -> 'float'`
- Route authority:
  - binding=`trellis.models.rate_style_swaption.price_swaption_black76`, engine=`analytical`, authority=`exact_backend_fit`
  - Validation bundle: `analytical:swaption`
  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_rate_style_swaption_helper_consistency`
  - Canary coverage: canaries=`T73`
  - Helper authority: `trellis.models.rate_style_swaption.price_swaption_black76`
  - Exact target bindings: `trellis.models.rate_style_swaption.price_swaption_black76`
- Backend binding:
  - Route: `analytical_black76`
  - Engine family: `analytical`
  - Route family: `analytical`
  - Selected primitives:
    - `trellis.models.rate_style_swaption.price_swaption_black76` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
- Primary modules to inspect/reuse:
  - `trellis.models.black`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the lane obligations in this card first. Treat backend route/helper details as exact-fit bindings, not as permission to invent a different numerical path.
- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan.
- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires.
      - `route_plan`: ## Structured Generation Plan
- Method family: `analytical`
- Instrument type: `unknown`
- Lane boundary: family=`analytical`, kind=`exact_target_binding`, timeline_roles=`payment`, exact_bindings=`trellis.models.rate_style_swaption.price_swaption_black76`
- Lane obligations:
  - Lane family: `analytical`
  - Plan kind: `exact_target_binding`
  - Timeline roles: `payment`
  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`
  - State obligations: `foreign_discount_curve`, `expiry_black_vol`
  - Construction steps:
    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.
    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.
  - Exact backend bindings:
    - `trellis.models.rate_style_swaption.price_swaption_black76`
  - Exact binding signatures:
    - `price_swaption_black76(market_state: 'MarketState', spec: 'EuropeanSwaptionSpecLike | RateStyleSwaptionSpecLike', *, expiry_date: 'date | None' = None, mean_reversion: 'float | None' = None, sigma: 'float | None' = None) -> 'float'`
- Route authority:
  - binding=`trellis.models.rate_style_swaption.price_swaption_black76`, engine=`analytical`, authority=`exact_backend_fit`
  - Validation bundle: `analytical:swaption`
  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_rate_style_swaption_helper_consistency`
  - Canary coverage: canaries=`T73`
  - Helper authority: `trellis.models.rate_style_swaption.price_swaption_black76`
  - Exact target bindings: `trellis.models.rate_style_swaption.price_swaption_black76`
- Repo revision: `3e05e8e455bb894825c363ce38d73be7f6ef7c15`
- Inspected modules:
  - `trellis.models.black`
- Approved Trellis modules for imports:
  - `trellis.core.date_utils`
  - `trellis.core.differentiable`
  - `trellis.core.market_state`
  - `trellis.core.payoff`
  - `trellis.core.types`
  - `trellis.models.analytical`
  - `trellis.models.analytical.barrier`
  - `trellis.models.analytical.fx`
  - `trellis.models.analytical.jamshidian`
  - `trellis.models.analytical.quanto`
  - `trellis.models.analytical.support.cross_asset`
  - `trellis.models.analytical.support.discounting`
  - `trellis.models.analytical.support.forwards`
  - `trellis.models.analytical.support.payoffs`
  - `trellis.models.black`
  - `trellis.models.rate_style_swaption`
- Public symbols available from the approved modules:
  - `BermudanSwaptionLowerBoundSpecLike`
  - `CashflowSchedule`
  - `Cashflows`
  - `ContractTimeline`
  - `DataProvider`
  - `DayCountConvention`
  - `DeterministicCashflowPayoff`
  - `DiscountCurve`
  - `DslMeasure`
  - `EuropeanSwaptionSpecLike`
  - `EventSchedule`
  - `Frequency`
  - `Instrument`
  - `MarketState`
  - `MissingCapabilityError`
  - `MonteCarloPathPayoff`
  - `Payoff`
  - `PresentValue`
  - `PricingResult`
  - `QuantoAnalyticalSpecLike`
  - `RateStyleSwaptionSpecLike`
  - `ResolvedBarrierInputs`
  - `ResolvedGarmanKohlhagenInputs`
  - `ResolvedInputPayoff`
- Tests to run after generation:
  - `tests/test_agent/test_build_loop.py`
- Likely tests for reused symbols:
  - `Cashflows` -> tests/test_models/test_contingent_cashflows.py
- Backend binding:
  - Route: `analytical_black76`
  - Engine family: `analytical`
  - Route family: `analytical`
  - Route score: `5.75`
  - Selected primitives:
    - `trellis.models.rate_style_swaption.price_swaption_black76` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
  - Instruction precedence: follow the compiler-emitted lane obligations first, then satisfy the exact backend binding and approved imports. If older guidance conflicts, treat it as stale and obey this plan.
- Uncertainty flags:
  - `instrument_type_not_provided`
- Every `trellis.*` import in the generated code MUST come from the approved module list above.
- If you need functionality outside the approved list, say so explicitly instead of inventing an import.
      - `trace_type`: analytical