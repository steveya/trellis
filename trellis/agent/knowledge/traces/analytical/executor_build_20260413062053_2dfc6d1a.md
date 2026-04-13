# Analytical Trace: `executor_build_20260413062053_2dfc6d1a`
- Trace type: `analytical`
- Route family: `rate_lattice`
- Route name: `exercise_lattice`
- Model: `lattice`
- Status: `ok`
- Created at: `2026-04-13T06:20:54.048265+00:00`
- Updated at: `2026-04-13T06:20:54.102686+00:00`
- Task ID: `executor_build_20260413062053_2dfc6d1a`

## Context
- `class_name`: 'BermudanSwaptionPayoff'
- `generation_plan`: {'method': 'rate_tree', 'instrument_type': 'bermudan_swaption', 'inspected_modules': ['trellis.models.trees.lattice'], 'approved_modules': ['trellis.core.date_utils', 'trellis.core.differentiable', 'trellis.core.market_state', 'trellis.core.payoff', 'trellis.core.types', 'trellis.models.bermudan_swaption_tree', 'trellis.models.black', 'trellis.models.trees.algebra', 'trellis.models.trees.backward_induction', 'trellis.models.trees.binomial', 'trellis.models.trees.control', 'trellis.models.trees.lattice', 'trellis.models.trees.models', 'trellis.models.trees.product_lattice', 'trellis.models.trees.trinomial'], 'symbols_to_reuse': ['AnalyticalCalibration', 'BermudanSwaptionSpecLike', 'BermudanSwaptionTreeMarketStateLike', 'BermudanSwaptionTreeSpec', 'BinomialTree', 'CalibratedLatticeData', 'CalibrationDiagnostics', 'CalibrationStrategy', 'CashflowSchedule', 'Cashflows', 'ContractTimeline', 'DataProvider', 'DayCountConvention', 'DeterministicCashflowPayoff', 'DiscountCurve', 'DiscountCurveLike', 'DslMeasure', 'EventOverlaySpec', 'EventSchedule', 'ExerciseObjective', 'Frequency', 'Instrument', 'LatticeAlgebraEligibilityDecision', 'LatticeContractSpec', 'LatticeControlSpec', 'LatticeExercisePolicy', 'LatticeLinearClaimSpec', 'LatticeMeshSpec', 'LatticeModelSpec', 'LatticeRecipe', 'LatticeTopologySpec', 'LocalVolCalibration', 'MarketState', 'MissingCapabilityError', 'MonteCarloPathPayoff', 'NO_CALIBRATION_TARGET', 'NoCalibrationTarget', 'Payoff', 'PresentValue', 'PricingResult', 'ProductRecombiningLattice2D', 'RecombiningLattice', 'ResolvedBermudanSwaptionTreeInputs', 'ResolvedInputPayoff', 'SchedulePeriod', 'TERM_STRUCTURE_TARGET', 'TermStructureCalibration', 'TermStructureTarget', 'TimelineRole', 'TreeModel', 'TrinomialTree', 'TwoFactorAnalyticalCalibration', 'VolSurfaceLike', 'VolSurfaceTarget', 'add_months', 'backward_induction', 'bdt_displacement', 'bdt_mean_reversion_probabilities', 'binomial_mean_reversion_probabilities_from_metric', 'black76_asset_or_nothing_call', 'black76_asset_or_nothing_put', 'black76_call', 'black76_cash_or_nothing_call', 'black76_cash_or_nothing_put', 'black76_put', 'build_bermudan_swaption_coupon_map', 'build_bermudan_swaption_exercise_policy', 'build_bermudan_swaption_lattice', 'build_contract_timeline', 'build_contract_timeline_from_dates', 'build_exercise_timeline_from_dates', 'build_generic_lattice', 'build_lattice', 'build_observation_timeline', 'build_payment_timeline', 'build_period_schedule', 'build_product_spot_lattice_2d', 'build_rate_lattice', 'build_spot_lattice', 'calibrate_lattice', 'coerce_contract_timeline_from_dates', 'compile_bermudan_swaption_contract_spec', 'compile_lattice_recipe', 'equal_probabilities', 'equity_tree', 'garman_kohlhagen_call', 'garman_kohlhagen_put', 'generate_schedule', 'get_accrual_fraction', 'get_bracketing_dates', 'get_numpy', 'gradient', 'hessian', 'ho_lee_displacement', 'hw_displacement', 'hw_mean_reversion_probabilities', 'identity_state_metric', 'jacobian', 'lattice_algebra_eligible', 'lattice_backward_induction', 'lattice_step_from_time', 'lattice_steps_from_timeline', 'log_rate_state_metric', 'lognormal_rate', 'merge_lattice_exercise_policy', 'normal_rate', 'normalize_dsl_measure', 'normalize_explicit_dates', 'price_bermudan_swaption_on_lattice', 'price_bermudan_swaption_tree', 'price_on_lattice', 'resolve_bermudan_swaption_tree_inputs', 'resolve_lattice_exercise_policy', 'resolve_lattice_exercise_policy_from_control_style', 'shifted_lognormal_rate', 'short_rate_tree', 'standard_discount', 'trinomial_mean_reversion_probabilities_from_metric', 'with_control', 'with_overlay', 'year_fraction'], 'proposed_tests': ['tests/test_agent/test_build_loop.py', 'tests/test_agent/test_callable_bond.py', 'tests/test_tasks/test_t04_bermudan_swaption.py'], 'uncertainty_flags': [], 'repo_revision': '7b4e104a06010bbe869a4c3666e707423c9307ec', 'instruction_resolution': {'route': 'exercise_lattice', 'effective_instruction_count': 1, 'dropped_instruction_count': 0, 'conflict_count': 0, 'effective_instructions': [{'id': 'exercise_lattice:route-helper', 'title': 'Use the selected route helper directly', 'instruction_type': 'hard_constraint', 'status': 'active', 'source_kind': 'route_card', 'source_id': 'exercise_lattice:route-helper', 'source_revision': '', 'scope_methods': ('rate_tree',), 'scope_instruments': ('bermudan_swaption',), 'scope_routes': ('exercise_lattice',), 'scope_modules': ('trellis.models.bermudan_swaption_tree',), 'scope_features': (), 'precedence_rank': 100, 'supersedes': (), 'conflict_policy': 'prefer_newer', 'statement': 'Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.', 'rationale': 'The helper already owns the route-specific engine and payoff mapping.', 'created_at': '', 'updated_at': ''}], 'dropped_instructions': [], 'conflicts': []}, 'primitive_plan': {'route': 'exercise_lattice', 'engine_family': 'lattice', 'route_family': 'rate_lattice', 'score': 7.75}}
- `route_card`: "## Structured Lane Card\n- Method family: `rate_tree`\n- Instrument type: `bermudan_swaption`\n- Lane boundary: family=`lattice`, kind=`exact_target_binding`, timeline_roles=`payment`, `exercise`, exact_bindings=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`\n- Lane obligations:\n  - Lane family: `lattice`\n  - Plan kind: `exact_target_binding`\n  - Timeline roles: `payment`, `exercise`\n  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`\n  - State obligations: `foreign_discount_curve`, `expiry_black_vol`\n  - Construction steps:\n    - Resolve the pricing state and early-exercise contract before building the lattice.\n    - Keep continuation, discounting, and exercise semantics explicit during backward induction.\n  - Exact backend bindings:\n    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`\n  - Exact binding signatures:\n    - `price_bermudan_swaption_tree(market_state: 'BermudanSwaptionTreeMarketStateLike', spec: 'BermudanSwaptionSpecLike', *, model: 'str' = 'hull_white', mean_reversion: 'float | None' = None, sigma: 'float | None' = None, n_steps: 'int | None' = None) -> 'float'`\n- Route authority:\n  - binding=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`, engine=`lattice`, authority=`exact_backend_fit`\n  - Route alias: `exercise_lattice`\n  - Validation bundle: `rate_tree:bermudan_swaption`\n  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`\n  - Canary coverage: canaries=`T01`\n  - Helper authority: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`\n  - Exact target bindings: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`\n- Backend binding:\n  - Route: `exercise_lattice`\n  - Engine family: `lattice`\n  - Route family: `rate_lattice`\n  - Selected primitives:\n    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree` (route_helper)\n  - Resolved instructions:\n    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n- Primary modules to inspect/reuse:\n  - `trellis.models.trees.lattice`\n- Post-build test targets:\n  - `tests/test_agent/test_build_loop.py`\n  - `tests/test_agent/test_callable_bond.py`\n  - `tests/test_tasks/test_t04_bermudan_swaption.py`\n- Instruction precedence: follow the lane obligations in this card first. Treat backend route/helper details as exact-fit bindings, not as permission to invent a different numerical path.\n- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan.\n- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires."
- `route_plan`: "## Structured Generation Plan\n- Method family: `rate_tree`\n- Instrument type: `bermudan_swaption`\n- Lane boundary: family=`lattice`, kind=`exact_target_binding`, timeline_roles=`payment`, `exercise`, exact_bindings=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`\n- Lane obligations:\n  - Lane family: `lattice`\n  - Plan kind: `exact_target_binding`\n  - Timeline roles: `payment`, `exercise`\n  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`\n  - State obligations: `foreign_discount_curve`, `expiry_black_vol`\n  - Construction steps:\n    - Resolve the pricing state and early-exercise contract before building the lattice.\n    - Keep continuation, discounting, and exercise semantics explicit during backward induction.\n  - Exact backend bindings:\n    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`\n  - Exact binding signatures:\n    - `price_bermudan_swaption_tree(market_state: 'BermudanSwaptionTreeMarketStateLike', spec: 'BermudanSwaptionSpecLike', *, model: 'str' = 'hull_white', mean_reversion: 'float | None' = None, sigma: 'float | None' = None, n_steps: 'int | None' = None) -> 'float'`\n- Route authority:\n  - binding=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`, engine=`lattice`, authority=`exact_backend_fit`\n  - Route alias: `exercise_lattice`\n  - Validation bundle: `rate_tree:bermudan_swaption`\n  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`\n  - Canary coverage: canaries=`T01`\n  - Helper authority: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`\n  - Exact target bindings: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`\n- Repo revision: `7b4e104a06010bbe869a4c3666e707423c9307ec`\n- Inspected modules:\n  - `trellis.models.trees.lattice`\n- Approved Trellis modules for imports:\n  - `trellis.core.date_utils`\n  - `trellis.core.differentiable`\n  - `trellis.core.market_state`\n  - `trellis.core.payoff`\n  - `trellis.core.types`\n  - `trellis.models.bermudan_swaption_tree`\n  - `trellis.models.black`\n  - `trellis.models.trees.algebra`\n  - `trellis.models.trees.backward_induction`\n  - `trellis.models.trees.binomial`\n  - `trellis.models.trees.control`\n  - `trellis.models.trees.lattice`\n  - `trellis.models.trees.models`\n  - `trellis.models.trees.product_lattice`\n  - `trellis.models.trees.trinomial`\n- Public symbols available from the approved modules:\n  - `AnalyticalCalibration`\n  - `BermudanSwaptionSpecLike`\n  - `BermudanSwaptionTreeMarketStateLike`\n  - `BermudanSwaptionTreeSpec`\n  - `BinomialTree`\n  - `CalibratedLatticeData`\n  - `CalibrationDiagnostics`\n  - `CalibrationStrategy`\n  - `CashflowSchedule`\n  - `Cashflows`\n  - `ContractTimeline`\n  - `DataProvider`\n  - `DayCountConvention`\n  - `DeterministicCashflowPayoff`\n  - `DiscountCurve`\n  - `DiscountCurveLike`\n  - `DslMeasure`\n  - `EventOverlaySpec`\n  - `EventSchedule`\n  - `ExerciseObjective`\n  - `Frequency`\n  - `Instrument`\n  - `LatticeAlgebraEligibilityDecision`\n  - `LatticeContractSpec`\n- Tests to run after generation:\n  - `tests/test_agent/test_build_loop.py`\n  - `tests/test_agent/test_callable_bond.py`\n  - `tests/test_tasks/test_t04_bermudan_swaption.py`\n- Backend binding:\n  - Route: `exercise_lattice`\n  - Engine family: `lattice`\n  - Route family: `rate_lattice`\n  - Route score: `7.75`\n  - Selected primitives:\n    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree` (route_helper)\n  - Resolved instructions:\n    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.\n  - Instruction precedence: follow the compiler-emitted lane obligations first, then satisfy the exact backend binding and approved imports. If older guidance conflicts, treat it as stale and obey this plan.\n- Every `trellis.*` import in the generated code MUST come from the approved module list above.\n- If you need functionality outside the approved list, say so explicitly instead of inventing an import."
- `selected_curve_names`: {}
- `spec_name`: 'BermudanSwaptionSpec'

## Steps
- **trace** `executor_build_20260413062053_2dfc6d1a:root`
  - Label: Analytical build
  - Status: `ok`
  - Notes:
    - The trace mirrors the deterministic GenerationPlan used to assemble the route.
  - Inputs:
    - `issue_id`: None
    - `model`: lattice
    - `route_family`: rate_lattice
    - `route_name`: exercise_lattice
    - `task_id`: executor_build_20260413062053_2dfc6d1a
  - Outputs:
    - `route`: {
  "family": "rate_lattice",
  "model": "lattice",
  "name": "exercise_lattice"
}
    - `status`: ok
  - **semantic_resolution** `executor_build_20260413062053_2dfc6d1a:semantic_resolution`
    - Label: Resolve contract and route
    - Status: `ok`
    - Parent: `executor_build_20260413062053_2dfc6d1a:root`
    - Notes:
      - Record the semantic contract that drives route selection, not just the final code path.
    - Inputs:
      - `approved_modules`: [
  "trellis.core.date_utils",
  "trellis.core.differentiable",
  "trellis.core.market_state",
  "trellis.core.payoff",
  "trellis.core.types",
  "trellis.models.bermudan_swaption_tree",
  "trellis.models.black",
  "trellis.models.trees.algebra",
  "trellis.models.trees.backward_induction",
  "trellis.models.trees.binomial",
  "trellis.models.trees.control",
  "trellis.models.trees.lattice",
  "trellis.models.trees.models",
  "trellis.models.trees.product_lattice",
  "trellis.models.trees.trinomial"
]
      - `inspected_modules`: [
  "trellis.models.trees.lattice"
]
      - `instrument_type`: bermudan_swaption
      - `method`: rate_tree
      - `repo_revision`: 7b4e104a06010bbe869a4c3666e707423c9307ec
      - `symbols_to_reuse`: [
  "AnalyticalCalibration",
  "BermudanSwaptionSpecLike",
  "BermudanSwaptionTreeMarketStateLike",
  "BermudanSwaptionTreeSpec",
  "BinomialTree",
  "CalibratedLatticeData",
  "CalibrationDiagnostics",
  "CalibrationStrategy",
  "CashflowSchedule",
  "Cashflows",
  "ContractTimeline",
  "DataProvider",
  "DayCountConvention",
  "DeterministicCashflowPayoff",
  "DiscountCurve",
  "DiscountCurveLike",
  "DslMeasure",
  "EventOverlaySpec",
  "EventSchedule",
  "ExerciseObjective",
  "Frequency",
  "Instrument",
  "LatticeAlgebraEligibilityDecision",
  "LatticeContractSpec",
  "LatticeControlSpec",
  "LatticeExercisePolicy",
  "LatticeLinearClaimSpec",
  "LatticeMeshSpec",
  "LatticeModelSpec",
  "LatticeRecipe",
  "LatticeTopologySpec",
  "LocalVolCalibration",
  "MarketState",
  "MissingCapabilityError",
  "MonteCarloPathPayoff",
  "NO_CALIBRATION_TARGET",
  "NoCalibrationTarget",
  "Payoff",
  "PresentValue",
  "PricingResult"
]
      - `uncertainty_flags`: []
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
      "id": "exercise_lattice:route-helper",
      "instruction_type": "hard_constraint",
      "precedence_rank": 100,
      "rationale": "The helper already owns the route-specific engine and payoff mapping.",
      "scope_features": [],
      "scope_instruments": [
        "bermudan_swaption"
      ],
      "scope_methods": [
        "rate_tree"
      ],
      "scope_modules": [
        "trellis.models.bermudan_swaption_tree"
      ],
      "scope_routes": [
        "exercise_lattice"
      ],
      "source_id": "exercise_lattice:route-helper",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.",
      "status": "active",
      "supersedes": [],
      "title": "Use the selected route helper directly",
      "updated_at": ""
    }
  ],
  "route": "exercise_lattice"
}
      - `model`: lattice
      - `primitive_plan_score`: 7.75
      - `route_family`: rate_lattice
      - `route_name`: exercise_lattice
  - **instruction_lifecycle** `executor_build_20260413062053_2dfc6d1a:instruction_lifecycle`
    - Label: Resolve route guidance lifecycle
    - Status: `ok`
    - Parent: `executor_build_20260413062053_2dfc6d1a:root`
    - Notes:
      - List the effective, dropped, and conflicting route guidance records so replay consumers can see what actually governed the build.
    - Inputs:
      - `conflict_count`: 0
      - `dropped_instruction_count`: 0
      - `effective_instruction_count`: 1
      - `route`: exercise_lattice
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
      "id": "exercise_lattice:route-helper",
      "instruction_type": "hard_constraint",
      "precedence_rank": 100,
      "rationale": "The helper already owns the route-specific engine and payoff mapping.",
      "scope_features": [],
      "scope_instruments": [
        "bermudan_swaption"
      ],
      "scope_methods": [
        "rate_tree"
      ],
      "scope_modules": [
        "trellis.models.bermudan_swaption_tree"
      ],
      "scope_routes": [
        "exercise_lattice"
      ],
      "source_id": "exercise_lattice:route-helper",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.",
      "status": "active",
      "supersedes": [],
      "title": "Use the selected route helper directly",
      "updated_at": ""
    }
  ],
  "route": "exercise_lattice"
}
  - **decomposition** `executor_build_20260413062053_2dfc6d1a:decomposition`
    - Label: Select reusable kernels
    - Status: `ok`
    - Parent: `executor_build_20260413062053_2dfc6d1a:root`
    - Notes:
      - Capture the reusable valuation components and any exact basis-claim assembly.
    - Inputs:
      - `adapters`: []
      - `blockers`: []
      - `notes`: []
      - `primitives`: [
  {
    "excluded": false,
    "module": "trellis.models.bermudan_swaption_tree",
    "required": true,
    "role": "route_helper",
    "symbol": "price_bermudan_swaption_tree"
  }
]
    - Outputs:
      - `reuse_decision`: route_local
      - `selected_primitives`: [
  {
    "excluded": false,
    "module": "trellis.models.bermudan_swaption_tree",
    "required": true,
    "role": "route_helper",
    "symbol": "price_bermudan_swaption_tree"
  }
]
  - **assembly** `executor_build_20260413062053_2dfc6d1a:assembly`
    - Label: Assemble route from kernels
    - Status: `ok`
    - Parent: `executor_build_20260413062053_2dfc6d1a:root`
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
  "trellis.models.bermudan_swaption_tree",
  "trellis.models.black",
  "trellis.models.trees.algebra",
  "trellis.models.trees.backward_induction",
  "trellis.models.trees.binomial",
  "trellis.models.trees.control",
  "trellis.models.trees.lattice",
  "trellis.models.trees.models",
  "trellis.models.trees.product_lattice",
  "trellis.models.trees.trinomial"
]
      - `route_helper`: trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree
    - Outputs:
      - `assembly_card`: ## Structured Lane Card
- Method family: `rate_tree`
- Instrument type: `bermudan_swaption`
- Lane boundary: family=`lattice`, kind=`exact_target_binding`, timeline_roles=`payment`, `exercise`, exact_bindings=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
- Lane obligations:
  - Lane family: `lattice`
  - Plan kind: `exact_target_binding`
  - Timeline roles: `payment`, `exercise`
  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`
  - State obligations: `foreign_discount_curve`, `expiry_black_vol`
  - Construction steps:
    - Resolve the pricing state and early-exercise contract before building the lattice.
    - Keep continuation, discounting, and exercise semantics explicit during backward induction.
  - Exact backend bindings:
    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
  - Exact binding signatures:
    - `price_bermudan_swaption_tree(market_state: 'BermudanSwaptionTreeMarketStateLike', spec: 'BermudanSwaptionSpecLike', *, model: 'str' = 'hull_white', mean_reversion: 'float | None' = None, sigma: 'float | None' = None, n_steps: 'int | None' = None) -> 'float'`
- Route authority:
  - binding=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`, engine=`lattice`, authority=`exact_backend_fit`
  - Route alias: `exercise_lattice`
  - Validation bundle: `rate_tree:bermudan_swaption`
  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`
  - Canary coverage: canaries=`T01`
  - Helper authority: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
  - Exact target bindings: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
- Backend binding:
  - Route: `exercise_lattice`
  - Engine family: `lattice`
  - Route family: `rate_lattice`
  - Selected primitives:
    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
- Primary modules to inspect/reuse:
  - `trellis.models.trees.lattice`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
  - `tests/test_agent/test_callable_bond.py`
  - `tests/test_tasks/test_t04_bermudan_swaption.py`
- Instruction precedence: follow the lane obligations in this card first. Treat backend route/helper details as exact-fit bindings, not as permission to invent a different numerical path.
- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan.
- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires.
      - `helper_modules`: [
  "trellis.models.bermudan_swaption_tree"
]
      - `route_helper`: trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree
  - **validation** `executor_build_20260413062053_2dfc6d1a:validation`
    - Label: Validate route and fallbacks
    - Status: `ok`
    - Parent: `executor_build_20260413062053_2dfc6d1a:root`
    - Notes:
      - Record proposed tests, blocker state, and any fallback or reuse notes.
    - Inputs:
      - `blockers`: []
      - `proposed_tests`: [
  "tests/test_agent/test_build_loop.py",
  "tests/test_agent/test_callable_bond.py",
  "tests/test_tasks/test_t04_bermudan_swaption.py"
]
      - `uncertainty_flags`: []
      - `validation`: None
    - Outputs:
      - `blocker_report_present`: False
      - `new_primitive_workflow_present`: False
      - `validation_state`: planned
  - **output** `executor_build_20260413062053_2dfc6d1a:output`
    - Label: Final analytical artifact
    - Status: `ok`
    - Parent: `executor_build_20260413062053_2dfc6d1a:root`
    - Notes:
      - Persist both the machine-readable trace and the text rendering from the same source of truth.
    - Inputs:
      - `model`: lattice
      - `route`: exercise_lattice
    - Outputs:
      - `route_card`: ## Structured Lane Card
- Method family: `rate_tree`
- Instrument type: `bermudan_swaption`
- Lane boundary: family=`lattice`, kind=`exact_target_binding`, timeline_roles=`payment`, `exercise`, exact_bindings=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
- Lane obligations:
  - Lane family: `lattice`
  - Plan kind: `exact_target_binding`
  - Timeline roles: `payment`, `exercise`
  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`
  - State obligations: `foreign_discount_curve`, `expiry_black_vol`
  - Construction steps:
    - Resolve the pricing state and early-exercise contract before building the lattice.
    - Keep continuation, discounting, and exercise semantics explicit during backward induction.
  - Exact backend bindings:
    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
  - Exact binding signatures:
    - `price_bermudan_swaption_tree(market_state: 'BermudanSwaptionTreeMarketStateLike', spec: 'BermudanSwaptionSpecLike', *, model: 'str' = 'hull_white', mean_reversion: 'float | None' = None, sigma: 'float | None' = None, n_steps: 'int | None' = None) -> 'float'`
- Route authority:
  - binding=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`, engine=`lattice`, authority=`exact_backend_fit`
  - Route alias: `exercise_lattice`
  - Validation bundle: `rate_tree:bermudan_swaption`
  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`
  - Canary coverage: canaries=`T01`
  - Helper authority: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
  - Exact target bindings: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
- Backend binding:
  - Route: `exercise_lattice`
  - Engine family: `lattice`
  - Route family: `rate_lattice`
  - Selected primitives:
    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
- Primary modules to inspect/reuse:
  - `trellis.models.trees.lattice`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
  - `tests/test_agent/test_callable_bond.py`
  - `tests/test_tasks/test_t04_bermudan_swaption.py`
- Instruction precedence: follow the lane obligations in this card first. Treat backend route/helper details as exact-fit bindings, not as permission to invent a different numerical path.
- Treat route authority as backend-fit evidence, not as permission to invent a different synthesis plan.
- Use approved Trellis imports only. Prefer thin adapters when the compiler found an exact backend; otherwise build the smallest lane-consistent kernel the plan requires.
      - `route_plan`: ## Structured Generation Plan
- Method family: `rate_tree`
- Instrument type: `bermudan_swaption`
- Lane boundary: family=`lattice`, kind=`exact_target_binding`, timeline_roles=`payment`, `exercise`, exact_bindings=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
- Lane obligations:
  - Lane family: `lattice`
  - Plan kind: `exact_target_binding`
  - Timeline roles: `payment`, `exercise`
  - Market bindings: `black_vol_surface`, `discount_curve`, `forward_curve`
  - State obligations: `foreign_discount_curve`, `expiry_black_vol`
  - Construction steps:
    - Resolve the pricing state and early-exercise contract before building the lattice.
    - Keep continuation, discounting, and exercise semantics explicit during backward induction.
  - Exact backend bindings:
    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
  - Exact binding signatures:
    - `price_bermudan_swaption_tree(market_state: 'BermudanSwaptionTreeMarketStateLike', spec: 'BermudanSwaptionSpecLike', *, model: 'str' = 'hull_white', mean_reversion: 'float | None' = None, sigma: 'float | None' = None, n_steps: 'int | None' = None) -> 'float'`
- Route authority:
  - binding=`trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`, engine=`lattice`, authority=`exact_backend_fit`
  - Route alias: `exercise_lattice`
  - Validation bundle: `rate_tree:bermudan_swaption`
  - Validation checks: `check_non_negativity`, `check_price_sanity`, `check_vol_sensitivity`, `check_vol_monotonicity`
  - Canary coverage: canaries=`T01`
  - Helper authority: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
  - Exact target bindings: `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree`
- Repo revision: `7b4e104a06010bbe869a4c3666e707423c9307ec`
- Inspected modules:
  - `trellis.models.trees.lattice`
- Approved Trellis modules for imports:
  - `trellis.core.date_utils`
  - `trellis.core.differentiable`
  - `trellis.core.market_state`
  - `trellis.core.payoff`
  - `trellis.core.types`
  - `trellis.models.bermudan_swaption_tree`
  - `trellis.models.black`
  - `trellis.models.trees.algebra`
  - `trellis.models.trees.backward_induction`
  - `trellis.models.trees.binomial`
  - `trellis.models.trees.control`
  - `trellis.models.trees.lattice`
  - `trellis.models.trees.models`
  - `trellis.models.trees.product_lattice`
  - `trellis.models.trees.trinomial`
- Public symbols available from the approved modules:
  - `AnalyticalCalibration`
  - `BermudanSwaptionSpecLike`
  - `BermudanSwaptionTreeMarketStateLike`
  - `BermudanSwaptionTreeSpec`
  - `BinomialTree`
  - `CalibratedLatticeData`
  - `CalibrationDiagnostics`
  - `CalibrationStrategy`
  - `CashflowSchedule`
  - `Cashflows`
  - `ContractTimeline`
  - `DataProvider`
  - `DayCountConvention`
  - `DeterministicCashflowPayoff`
  - `DiscountCurve`
  - `DiscountCurveLike`
  - `DslMeasure`
  - `EventOverlaySpec`
  - `EventSchedule`
  - `ExerciseObjective`
  - `Frequency`
  - `Instrument`
  - `LatticeAlgebraEligibilityDecision`
  - `LatticeContractSpec`
- Tests to run after generation:
  - `tests/test_agent/test_build_loop.py`
  - `tests/test_agent/test_callable_bond.py`
  - `tests/test_tasks/test_t04_bermudan_swaption.py`
- Backend binding:
  - Route: `exercise_lattice`
  - Engine family: `lattice`
  - Route family: `rate_lattice`
  - Route score: `7.75`
  - Selected primitives:
    - `trellis.models.bermudan_swaption_tree.price_bermudan_swaption_tree` (route_helper)
  - Resolved instructions:
    - [hard_constraint] Use the route helper directly inside `evaluate()`; do not rebuild the process, engine, or discount glue manually.
  - Instruction precedence: follow the compiler-emitted lane obligations first, then satisfy the exact backend binding and approved imports. If older guidance conflicts, treat it as stale and obey this plan.
- Every `trellis.*` import in the generated code MUST come from the approved module list above.
- If you need functionality outside the approved list, say so explicitly instead of inventing an import.
      - `trace_type`: analytical