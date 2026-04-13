# Analytical Trace: `executor_build_20260413025327_aaebedcb`
- Trace type: `analytical`
- Route family: `analytical`
- Route name: `analytical_garman_kohlhagen`
- Model: `analytical`
- Status: `ok`
- Created at: `2026-04-13T02:53:28.887490+00:00`
- Updated at: `2026-04-13T02:53:28.931641+00:00`
- Task ID: `executor_build_20260413025327_aaebedcb`

## Context
- `class_name`: 'FXVanillaAnalyticalPayoff'
- `generation_plan`: {'method': 'analytical', 'instrument_type': 'european_option', 'inspected_modules': ['trellis.models.black'], 'approved_modules': ['trellis.core.date_utils', 'trellis.core.differentiable', 'trellis.core.market_state', 'trellis.core.payoff', 'trellis.core.types', 'trellis.models.analytical', 'trellis.models.analytical.barrier', 'trellis.models.analytical.fx', 'trellis.models.analytical.jamshidian', 'trellis.models.analytical.quanto', 'trellis.models.analytical.support.cross_asset', 'trellis.models.analytical.support.discounting', 'trellis.models.analytical.support.forwards', 'trellis.models.analytical.support.payoffs', 'trellis.models.black', 'trellis.models.fx_vanilla'], 'symbols_to_reuse': ['CashflowSchedule', 'Cashflows', 'ContractTimeline', 'DataProvider', 'DayCountConvention', 'DeterministicCashflowPayoff', 'DiscountCurve', 'DslMeasure', 'EventSchedule', 'FXVanillaSpecLike', 'Frequency', 'Instrument', 'MarketState', 'MissingCapabilityError', 'MonteCarloPathPayoff', 'Payoff', 'PresentValue', 'PricingResult', 'QuantoAnalyticalSpecLike', 'ResolvedBarrierInputs', 'ResolvedFXVanillaInputs', 'ResolvedGarmanKohlhagenInputs', 'ResolvedInputPayoff', 'ResolvedJamshidianInputs', 'SchedulePeriod', 'TimelineRole', 'add_months', 'asset_or_nothing_intrinsic', 'barrier_image_raw', 'barrier_option_price', 'barrier_regime_selector_raw', 'black76_asset_or_nothing_call', 'black76_asset_or_nothing_put', 'black76_call', 'black76_cash_or_nothing_call', 'black76_cash_or_nothing_put', 'black76_put', 'build_contract_timeline', 'build_contract_timeline_from_dates', 'build_exercise_timeline_from_dates', 'build_observation_timeline', 'build_payment_timeline', 'build_period_schedule', 'call_put_parity_gap', 'cash_or_nothing_intrinsic', 'coerce_contract_timeline_from_dates', 'continuous_rate_from_simple_rate', 'discount_factor_from_zero_rate', 'discounted_value', 'down_and_in_call', 'down_and_in_call_raw', 'down_and_out_call', 'down_and_out_call_raw', 'effective_covariance_term', 'exchange_option_effective_vol', 'foreign_to_domestic_forward_bridge', 'forward_discount_ratio', 'forward_from_carry_rate', 'forward_from_discount_factors', 'forward_from_dividend_yield', 'garman_kohlhagen_call', 'garman_kohlhagen_call_raw', 'garman_kohlhagen_price_raw', 'garman_kohlhagen_put', 'garman_kohlhagen_put_raw', 'generate_schedule', 'get_accrual_fraction', 'get_bracketing_dates', 'get_numpy', 'gradient', 'hessian', 'implied_zero_rate', 'jacobian', 'normalize_dsl_measure', 'normalize_explicit_dates', 'normalized_option_type', 'price_fx_vanilla_analytical', 'price_fx_vanilla_monte_carlo', 'price_quanto_option_analytical', 'price_quanto_option_raw', 'quanto_adjusted_forward', 'rebate_raw', 'resolve_fx_vanilla_inputs', 'safe_time_fraction', 'simple_rate_from_discount_factor', 'terminal_intrinsic', 'terminal_vanilla_from_basis', 'vanilla_call_raw', 'year_fraction', 'zcb_option_hw', 'zcb_option_hw_raw'], 'proposed_tests': ['tests/test_agent/test_build_loop.py'], 'uncertainty_flags': [], 'repo_revision': '3e05e8e455bb894825c363ce38d73be7f6ef7c15', 'instruction_resolution': {'route': 'analytical_garman_kohlhagen', 'effective_instruction_count': 2, 'dropped_instruction_count': 0, 'conflict_count': 0, 'effective_instructions': [{'id': 'analytical_garman_kohlhagen:route-helper', 'title': 'Use the selected route helper directly', 'instruction_type': 'hard_constraint', 'status': 'active', 'source_kind': 'route_card', 'source_id': 'analytical_garman_kohlhagen:route-helper', 'source_revision': '', 'scope_methods': ('analytical',), 'scope_instruments': ('european_option',), 'scope_routes': ('analytical_garman_kohlhagen',), 'scope_modules': ('trellis.models.fx_vanilla',), 'scope_features': (), 'precedence_rank': 100, 'supersedes': (), 'conflict_policy': 'prefer_newer', 'statement': 'Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.', 'rationale': 'The helper already owns the route-specific engine and payoff mapping.', 'created_at': '', 'updated_at': ''}, {'id': 'analytical_garman_kohlhagen:note:1', 'title': 'Route note 1', 'instruction_type': 'historical_note', 'status': 'active', 'source_kind': 'route_card', 'source_id': 'analytical_garman_kohlhagen:note:1', 'source_revision': '', 'scope_methods': ('analytical',), 'scope_instruments': ('european_option',), 'scope_routes': ('analytical_garman_kohlhagen',), 'scope_modules': (), 'scope_features': (), 'precedence_rank': 49, 'supersedes': (), 'conflict_policy': 'prefer_newer', 'statement': 'Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.', 'rationale': 'Route notes are retained as structured guidance and resolved by precedence.', 'created_at': '', 'updated_at': ''}], 'dropped_instructions': [], 'conflicts': []}, 'primitive_plan': {'route': 'analytical_garman_kohlhagen', 'engine_family': 'analytical', 'route_family': 'analytical', 'score': 5.5}}
- `route_card`: "## Structured Lane Card\n- Method family: `analytical`\n- Instrument type: `european_option`\n- Lane boundary: family=`analytical`, kind=`exact_target_binding`, exact_bindings=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`\n- Lane obligations:\n  - Lane family: `analytical`\n  - Plan kind: `exact_target_binding`\n  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`, `fx_rates`\n  - State obligations: `spot`, `strike`, `expiry`, `fx_rate_scalar_spot`\n  - Construction steps:\n    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.\n    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.\n  - Exact backend bindings:\n    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical`\n  - Exact binding signatures:\n    - `price_fx_vanilla_analytical(market_state: 'MarketState', spec: 'FXVanillaSpecLike') -> 'float'`\n- Route authority:\n  - binding=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`, engine=`analytical`, authority=`exact_backend_fit`\n  - Route alias: `analytical_garman_kohlhagen`\n  - Validation bundle: `analytical:european_option`\n  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`\n  - Helper authority: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`\n  - Exact target bindings: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`\n- Backend binding:\n  - Route: `analytical_garman_kohlhagen`\n  - Engine family: `analytical`\n  - Route family: `analytical`\n  - Selected primitives:\n    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical` (route_helper)\n  - Resolved instructions:\n    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n    - [historical_note] Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.\n  - Backend notes:\n    - Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.\n- Primary modules to inspect/reuse:\n  - `trellis.models.black`\n- Post-build test targets:\n  - `tests/test_agent/test_build_loop.py`\n- Instruction precedence: follow the lane obligations in this card first. Treat backend route/helper details as exact-fit bindings, not as permission to invent a different numerical path.\n- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan.\n- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires."
- `route_plan`: "## Structured Generation Plan\n- Method family: `analytical`\n- Instrument type: `european_option`\n- Lane boundary: family=`analytical`, kind=`exact_target_binding`, exact_bindings=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`\n- Lane obligations:\n  - Lane family: `analytical`\n  - Plan kind: `exact_target_binding`\n  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`, `fx_rates`, `spot`\n  - State obligations: `spot`, `strike`, `expiry`, `fx_rate_scalar_spot`, `foreign_discount_curve`, `expiry_black_vol`\n  - Construction steps:\n    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.\n    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.\n  - Exact backend bindings:\n    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical`\n  - Exact binding signatures:\n    - `price_fx_vanilla_analytical(market_state: 'MarketState', spec: 'FXVanillaSpecLike') -> 'float'`\n- Route authority:\n  - binding=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`, engine=`analytical`, authority=`exact_backend_fit`\n  - Route alias: `analytical_garman_kohlhagen`\n  - Validation bundle: `analytical:european_option`\n  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`, `check_zero_vol_intrinsic`\n  - Helper authority: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`\n  - Exact target bindings: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`\n- Repo revision: `3e05e8e455bb894825c363ce38d73be7f6ef7c15`\n- Inspected modules:\n  - `trellis.models.black`\n- Approved Trellis modules for imports:\n  - `trellis.core.date_utils`\n  - `trellis.core.differentiable`\n  - `trellis.core.market_state`\n  - `trellis.core.payoff`\n  - `trellis.core.types`\n  - `trellis.models.analytical`\n  - `trellis.models.analytical.barrier`\n  - `trellis.models.analytical.fx`\n  - `trellis.models.analytical.jamshidian`\n  - `trellis.models.analytical.quanto`\n  - `trellis.models.analytical.support.cross_asset`\n  - `trellis.models.analytical.support.discounting`\n  - `trellis.models.analytical.support.forwards`\n  - `trellis.models.analytical.support.payoffs`\n  - `trellis.models.black`\n  - `trellis.models.fx_vanilla`\n- Public symbols available from the approved modules:\n  - `CashflowSchedule`\n  - `Cashflows`\n  - `ContractTimeline`\n  - `DataProvider`\n  - `DayCountConvention`\n  - `DeterministicCashflowPayoff`\n  - `DiscountCurve`\n  - `DslMeasure`\n  - `EventSchedule`\n  - `FXVanillaSpecLike`\n  - `Frequency`\n  - `Instrument`\n  - `MarketState`\n  - `MissingCapabilityError`\n  - `MonteCarloPathPayoff`\n  - `Payoff`\n  - `PresentValue`\n  - `PricingResult`\n  - `QuantoAnalyticalSpecLike`\n  - `ResolvedBarrierInputs`\n  - `ResolvedFXVanillaInputs`\n  - `ResolvedGarmanKohlhagenInputs`\n  - `ResolvedInputPayoff`\n  - `ResolvedJamshidianInputs`\n- Tests to run after generation:\n  - `tests/test_agent/test_build_loop.py`\n- Likely tests for reused symbols:\n  - `Cashflows` -> tests/test_models/test_contingent_cashflows.py\n- Backend binding:\n  - Route: `analytical_garman_kohlhagen`\n  - Engine family: `analytical`\n  - Route family: `analytical`\n  - Route score: `5.50`\n  - Selected primitives:\n    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical` (route_helper)\n  - Resolved instructions:\n    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n    - [historical_note] Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.\n  - Instruction precedence: follow the compiler-emitted lane obligations first, then satisfy the exact backend binding and approved imports. If older guidance conflicts, treat it as stale and obey this plan.\n- Every `trellis.*` import in the generated code MUST come from the approved module list above.\n- If you need functionality outside the approved list, say so explicitly instead of inventing an import."
- `selected_curve_names`: {}
- `spec_name`: 'FXVanillaOptionSpec'

## Steps
- **trace** `executor_build_20260413025327_aaebedcb:root`
  - Label: Analytical build
  - Status: `ok`
  - Notes:
    - The trace mirrors the deterministic GenerationPlan used to assemble the route.
  - Inputs:
    - `issue_id`: None
    - `model`: analytical
    - `route_family`: analytical
    - `route_name`: analytical_garman_kohlhagen
    - `task_id`: executor_build_20260413025327_aaebedcb
  - Outputs:
    - `route`: {
  "family": "analytical",
  "model": "analytical",
  "name": "analytical_garman_kohlhagen"
}
    - `status`: ok
  - **semantic_resolution** `executor_build_20260413025327_aaebedcb:semantic_resolution`
    - Label: Resolve contract and route
    - Status: `ok`
    - Parent: `executor_build_20260413025327_aaebedcb:root`
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
  "trellis.models.fx_vanilla"
]
      - `inspected_modules`: [
  "trellis.models.black"
]
      - `instrument_type`: european_option
      - `method`: analytical
      - `repo_revision`: 3e05e8e455bb894825c363ce38d73be7f6ef7c15
      - `symbols_to_reuse`: [
  "CashflowSchedule",
  "Cashflows",
  "ContractTimeline",
  "DataProvider",
  "DayCountConvention",
  "DeterministicCashflowPayoff",
  "DiscountCurve",
  "DslMeasure",
  "EventSchedule",
  "FXVanillaSpecLike",
  "Frequency",
  "Instrument",
  "MarketState",
  "MissingCapabilityError",
  "MonteCarloPathPayoff",
  "Payoff",
  "PresentValue",
  "PricingResult",
  "QuantoAnalyticalSpecLike",
  "ResolvedBarrierInputs",
  "ResolvedFXVanillaInputs",
  "ResolvedGarmanKohlhagenInputs",
  "ResolvedInputPayoff",
  "ResolvedJamshidianInputs",
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
  "build_contract_timeline",
  "build_contract_timeline_from_dates",
  "build_exercise_timeline_from_dates"
]
      - `uncertainty_flags`: []
    - Outputs:
      - `instruction_resolution`: {
  "conflict_count": 0,
  "conflicts": [],
  "dropped_instruction_count": 0,
  "dropped_instructions": [],
  "effective_instruction_count": 2,
  "effective_instructions": [
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "analytical_garman_kohlhagen:route-helper",
      "instruction_type": "hard_constraint",
      "precedence_rank": 100,
      "rationale": "The helper already owns the route-specific engine and payoff mapping.",
      "scope_features": [],
      "scope_instruments": [
        "european_option"
      ],
      "scope_methods": [
        "analytical"
      ],
      "scope_modules": [
        "trellis.models.fx_vanilla"
      ],
      "scope_routes": [
        "analytical_garman_kohlhagen"
      ],
      "source_id": "analytical_garman_kohlhagen:route-helper",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.",
      "status": "active",
      "supersedes": [],
      "title": "Use the selected route helper directly",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "analytical_garman_kohlhagen:note:1",
      "instruction_type": "historical_note",
      "precedence_rank": 49,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "european_option"
      ],
      "scope_methods": [
        "analytical"
      ],
      "scope_modules": [],
      "scope_routes": [
        "analytical_garman_kohlhagen"
      ],
      "source_id": "analytical_garman_kohlhagen:note:1",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 1",
      "updated_at": ""
    }
  ],
  "route": "analytical_garman_kohlhagen"
}
      - `model`: analytical
      - `primitive_plan_score`: 5.5
      - `route_family`: analytical
      - `route_name`: analytical_garman_kohlhagen
  - **instruction_lifecycle** `executor_build_20260413025327_aaebedcb:instruction_lifecycle`
    - Label: Resolve route guidance lifecycle
    - Status: `ok`
    - Parent: `executor_build_20260413025327_aaebedcb:root`
    - Notes:
      - List the effective, dropped, and conflicting route guidance records so replay consumers can see what actually governed the build.
    - Inputs:
      - `conflict_count`: 0
      - `dropped_instruction_count`: 0
      - `effective_instruction_count`: 2
      - `route`: analytical_garman_kohlhagen
    - Outputs:
      - `instruction_resolution`: {
  "conflict_count": 0,
  "conflicts": [],
  "dropped_instruction_count": 0,
  "dropped_instructions": [],
  "effective_instruction_count": 2,
  "effective_instructions": [
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "analytical_garman_kohlhagen:route-helper",
      "instruction_type": "hard_constraint",
      "precedence_rank": 100,
      "rationale": "The helper already owns the route-specific engine and payoff mapping.",
      "scope_features": [],
      "scope_instruments": [
        "european_option"
      ],
      "scope_methods": [
        "analytical"
      ],
      "scope_modules": [
        "trellis.models.fx_vanilla"
      ],
      "scope_routes": [
        "analytical_garman_kohlhagen"
      ],
      "source_id": "analytical_garman_kohlhagen:route-helper",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.",
      "status": "active",
      "supersedes": [],
      "title": "Use the selected route helper directly",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "analytical_garman_kohlhagen:note:1",
      "instruction_type": "historical_note",
      "precedence_rank": 49,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "european_option"
      ],
      "scope_methods": [
        "analytical"
      ],
      "scope_modules": [],
      "scope_routes": [
        "analytical_garman_kohlhagen"
      ],
      "source_id": "analytical_garman_kohlhagen:note:1",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 1",
      "updated_at": ""
    }
  ],
  "route": "analytical_garman_kohlhagen"
}
  - **decomposition** `executor_build_20260413025327_aaebedcb:decomposition`
    - Label: Select reusable kernels
    - Status: `ok`
    - Parent: `executor_build_20260413025327_aaebedcb:root`
    - Notes:
      - Capture the reusable valuation components and any exact basis-claim assembly.
    - Inputs:
      - `adapters`: []
      - `blockers`: []
      - `notes`: [
  "Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests."
]
      - `primitives`: [
  {
    "excluded": false,
    "module": "trellis.models.fx_vanilla",
    "required": true,
    "role": "route_helper",
    "symbol": "price_fx_vanilla_analytical"
  }
]
    - Outputs:
      - `reuse_decision`: exact_decomposition
      - `selected_primitives`: [
  {
    "excluded": false,
    "module": "trellis.models.fx_vanilla",
    "required": true,
    "role": "route_helper",
    "symbol": "price_fx_vanilla_analytical"
  }
]
  - **assembly** `executor_build_20260413025327_aaebedcb:assembly`
    - Label: Assemble route from kernels
    - Status: `ok`
    - Parent: `executor_build_20260413025327_aaebedcb:root`
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
  "trellis.models.fx_vanilla"
]
      - `route_helper`: trellis.models.fx_vanilla.price_fx_vanilla_analytical
    - Outputs:
      - `assembly_card`: ## Structured Lane Card
- Method family: `analytical`
- Instrument type: `european_option`
- Lane boundary: family=`analytical`, kind=`exact_target_binding`, exact_bindings=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`
- Lane obligations:
  - Lane family: `analytical`
  - Plan kind: `exact_target_binding`
  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`, `fx_rates`
  - State obligations: `spot`, `strike`, `expiry`, `fx_rate_scalar_spot`
  - Construction steps:
    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.
    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.
  - Exact backend bindings:
    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical`
  - Exact binding signatures:
    - `price_fx_vanilla_analytical(market_state: 'MarketState', spec: 'FXVanillaSpecLike') -> 'float'`
- Route authority:
  - binding=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`, engine=`analytical`, authority=`exact_backend_fit`
  - Route alias: `analytical_garman_kohlhagen`
  - Validation bundle: `analytical:european_option`
  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`
  - Helper authority: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`
  - Exact target bindings: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`
- Backend binding:
  - Route: `analytical_garman_kohlhagen`
  - Engine family: `analytical`
  - Route family: `analytical`
  - Selected primitives:
    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
    - [historical_note] Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.
  - Backend notes:
    - Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.
- Primary modules to inspect/reuse:
  - `trellis.models.black`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the lane obligations in this card first. Treat backend route/helper details as exact-fit bindings, not as permission to invent a different numerical path.
- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan.
- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires.
      - `helper_modules`: [
  "trellis.models.fx_vanilla"
]
      - `route_helper`: trellis.models.fx_vanilla.price_fx_vanilla_analytical
  - **validation** `executor_build_20260413025327_aaebedcb:validation`
    - Label: Validate route and fallbacks
    - Status: `ok`
    - Parent: `executor_build_20260413025327_aaebedcb:root`
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
  - **output** `executor_build_20260413025327_aaebedcb:output`
    - Label: Final analytical artifact
    - Status: `ok`
    - Parent: `executor_build_20260413025327_aaebedcb:root`
    - Notes:
      - Persist both the machine-readable trace and the text rendering from the same source of truth.
    - Inputs:
      - `model`: analytical
      - `route`: analytical_garman_kohlhagen
    - Outputs:
      - `route_card`: ## Structured Lane Card
- Method family: `analytical`
- Instrument type: `european_option`
- Lane boundary: family=`analytical`, kind=`exact_target_binding`, exact_bindings=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`
- Lane obligations:
  - Lane family: `analytical`
  - Plan kind: `exact_target_binding`
  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`, `fx_rates`
  - State obligations: `spot`, `strike`, `expiry`, `fx_rate_scalar_spot`
  - Construction steps:
    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.
    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.
  - Exact backend bindings:
    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical`
  - Exact binding signatures:
    - `price_fx_vanilla_analytical(market_state: 'MarketState', spec: 'FXVanillaSpecLike') -> 'float'`
- Route authority:
  - binding=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`, engine=`analytical`, authority=`exact_backend_fit`
  - Route alias: `analytical_garman_kohlhagen`
  - Validation bundle: `analytical:european_option`
  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`
  - Helper authority: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`
  - Exact target bindings: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`
- Backend binding:
  - Route: `analytical_garman_kohlhagen`
  - Engine family: `analytical`
  - Route family: `analytical`
  - Selected primitives:
    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
    - [historical_note] Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.
  - Backend notes:
    - Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.
- Primary modules to inspect/reuse:
  - `trellis.models.black`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the lane obligations in this card first. Treat backend route/helper details as exact-fit bindings, not as permission to invent a different numerical path.
- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan.
- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires.
      - `route_plan`: ## Structured Generation Plan
- Method family: `analytical`
- Instrument type: `european_option`
- Lane boundary: family=`analytical`, kind=`exact_target_binding`, exact_bindings=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`
- Lane obligations:
  - Lane family: `analytical`
  - Plan kind: `exact_target_binding`
  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`, `fx_rates`, `spot`
  - State obligations: `spot`, `strike`, `expiry`, `fx_rate_scalar_spot`, `foreign_discount_curve`, `expiry_black_vol`
  - Construction steps:
    - Resolve scalar market inputs and contract terms before applying the closed-form kernel.
    - Keep discounting and market-binding semantics explicit instead of hiding them behind invented helpers.
  - Exact backend bindings:
    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical`
  - Exact binding signatures:
    - `price_fx_vanilla_analytical(market_state: 'MarketState', spec: 'FXVanillaSpecLike') -> 'float'`
- Route authority:
  - binding=`trellis.models.fx_vanilla.price_fx_vanilla_analytical`, engine=`analytical`, authority=`exact_backend_fit`
  - Route alias: `analytical_garman_kohlhagen`
  - Validation bundle: `analytical:european_option`
  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`, `check_zero_vol_intrinsic`
  - Helper authority: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`
  - Exact target bindings: `trellis.models.fx_vanilla.price_fx_vanilla_analytical`
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
  - `trellis.models.fx_vanilla`
- Public symbols available from the approved modules:
  - `CashflowSchedule`
  - `Cashflows`
  - `ContractTimeline`
  - `DataProvider`
  - `DayCountConvention`
  - `DeterministicCashflowPayoff`
  - `DiscountCurve`
  - `DslMeasure`
  - `EventSchedule`
  - `FXVanillaSpecLike`
  - `Frequency`
  - `Instrument`
  - `MarketState`
  - `MissingCapabilityError`
  - `MonteCarloPathPayoff`
  - `Payoff`
  - `PresentValue`
  - `PricingResult`
  - `QuantoAnalyticalSpecLike`
  - `ResolvedBarrierInputs`
  - `ResolvedFXVanillaInputs`
  - `ResolvedGarmanKohlhagenInputs`
  - `ResolvedInputPayoff`
  - `ResolvedJamshidianInputs`
- Tests to run after generation:
  - `tests/test_agent/test_build_loop.py`
- Likely tests for reused symbols:
  - `Cashflows` -> tests/test_models/test_contingent_cashflows.py
- Backend binding:
  - Route: `analytical_garman_kohlhagen`
  - Engine family: `analytical`
  - Route family: `analytical`
  - Route score: `5.50`
  - Selected primitives:
    - `trellis.models.fx_vanilla.price_fx_vanilla_analytical` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
    - [historical_note] Use `trellis.models.fx_vanilla.price_fx_vanilla_analytical(...)` as the exact backend helper for analytical vanilla FX requests.
  - Instruction precedence: follow the compiler-emitted lane obligations first, then satisfy the exact backend binding and approved imports. If older guidance conflicts, treat it as stale and obey this plan.
- Every `trellis.*` import in the generated code MUST come from the approved module list above.
- If you need functionality outside the approved list, say so explicitly instead of inventing an import.
      - `trace_type`: analytical