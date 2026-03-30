# Analytical Trace: `executor_build_20260329154841_f9c8ff3f`
- Trace type: `analytical`
- Route family: `pde_solver`
- Route name: `pde_theta_1d`
- Model: `pde_solver`
- Status: `ok`
- Created at: `2026-03-29T15:50:11.516841+00:00`
- Updated at: `2026-03-29T15:50:11.641261+00:00`
- Task ID: `executor_build_20260329154841_f9c8ff3f`

## Context
- `class_name`: 'AmericanPutPricer'
- `generation_plan`: {'method': 'pde_solver', 'instrument_type': 'american_put', 'inspected_modules': ['trellis.models.pde.theta_method'], 'approved_modules': ['trellis.core.date_utils', 'trellis.core.differentiable', 'trellis.core.market_state', 'trellis.core.payoff', 'trellis.core.types', 'trellis.models.black', 'trellis.models.pde', 'trellis.models.pde.crank_nicolson', 'trellis.models.pde.grid', 'trellis.models.pde.implicit_fd', 'trellis.models.pde.operator', 'trellis.models.pde.psor', 'trellis.models.pde.rate_operator', 'trellis.models.pde.theta_method', 'trellis.models.pde.thomas'], 'symbols_to_reuse': ['BlackScholesOperator', 'CEVOperator', 'CashflowSchedule', 'Cashflows', 'DataProvider', 'DayCountConvention', 'DeterministicCashflowPayoff', 'DiscountCurve', 'Frequency', 'Grid', 'HeatOperator', 'HullWhitePDEOperator', 'Instrument', 'MarketState', 'MissingCapabilityError', 'MonteCarloPathPayoff', 'PDEOperator', 'Payoff', 'PresentValue', 'PricingResult', 'ResolvedInputPayoff', 'add_months', 'black76_asset_or_nothing_call', 'black76_asset_or_nothing_put', 'black76_call', 'black76_cash_or_nothing_call', 'black76_cash_or_nothing_put', 'black76_put', 'crank_nicolson_1d', 'garman_kohlhagen_call', 'garman_kohlhagen_put', 'generate_schedule', 'get_accrual_fraction', 'get_bracketing_dates', 'get_numpy', 'gradient', 'hessian', 'implicit_fd_1d', 'psor_1d', 'theta_method_1d', 'thomas_solve', 'year_fraction'], 'proposed_tests': ['tests/test_agent/test_build_loop.py'], 'uncertainty_flags': [], 'repo_revision': '2204f054d09cfaa88de56326480925626f4ca2b3', 'instruction_resolution': {'route': 'pde_theta_1d', 'effective_instruction_count': 5, 'dropped_instruction_count': 0, 'conflict_count': 0, 'effective_instructions': [{'id': 'pde_theta_1d:note:2', 'title': 'Route note 2', 'instruction_type': 'route_hint', 'status': 'active', 'source_kind': 'route_card', 'source_id': 'pde_theta_1d:note:2', 'source_revision': '', 'scope_methods': ('pde_solver',), 'scope_instruments': ('american_put',), 'scope_routes': ('pde_theta_1d',), 'scope_modules': (), 'scope_features': (), 'precedence_rank': 48, 'supersedes': (), 'conflict_policy': 'prefer_newer', 'statement': 'Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.', 'rationale': 'Route notes are retained as structured guidance and resolved by precedence.', 'created_at': '', 'updated_at': ''}, {'id': 'pde_theta_1d:note:5', 'title': 'Route note 5', 'instruction_type': 'route_hint', 'status': 'active', 'source_kind': 'route_card', 'source_id': 'pde_theta_1d:note:5', 'source_revision': '', 'scope_methods': ('pde_solver',), 'scope_instruments': ('american_put',), 'scope_routes': ('pde_theta_1d',), 'scope_modules': (), 'scope_features': (), 'precedence_rank': 45, 'supersedes': (), 'conflict_policy': 'prefer_newer', 'statement': 'Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.', 'rationale': 'Route notes are retained as structured guidance and resolved by precedence.', 'created_at': '', 'updated_at': ''}, {'id': 'pde_theta_1d:note:1', 'title': 'Route note 1', 'instruction_type': 'historical_note', 'status': 'active', 'source_kind': 'route_card', 'source_id': 'pde_theta_1d:note:1', 'source_revision': '', 'scope_methods': ('pde_solver',), 'scope_instruments': ('american_put',), 'scope_routes': ('pde_theta_1d',), 'scope_modules': (), 'scope_features': (), 'precedence_rank': 49, 'supersedes': (), 'conflict_policy': 'prefer_newer', 'statement': 'Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.', 'rationale': 'Route notes are retained as structured guidance and resolved by precedence.', 'created_at': '', 'updated_at': ''}, {'id': 'pde_theta_1d:note:3', 'title': 'Route note 3', 'instruction_type': 'historical_note', 'status': 'active', 'source_kind': 'route_card', 'source_id': 'pde_theta_1d:note:3', 'source_revision': '', 'scope_methods': ('pde_solver',), 'scope_instruments': ('american_put',), 'scope_routes': ('pde_theta_1d',), 'scope_modules': (), 'scope_features': (), 'precedence_rank': 47, 'supersedes': (), 'conflict_policy': 'prefer_newer', 'statement': 'For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.', 'rationale': 'Route notes are retained as structured guidance and resolved by precedence.', 'created_at': '', 'updated_at': ''}, {'id': 'pde_theta_1d:note:4', 'title': 'Route note 4', 'instruction_type': 'historical_note', 'status': 'active', 'source_kind': 'route_card', 'source_id': 'pde_theta_1d:note:4', 'source_revision': '', 'scope_methods': ('pde_solver',), 'scope_instruments': ('american_put',), 'scope_routes': ('pde_theta_1d',), 'scope_modules': (), 'scope_features': (), 'precedence_rank': 46, 'supersedes': (), 'conflict_policy': 'prefer_newer', 'statement': 'For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.', 'rationale': 'Route notes are retained as structured guidance and resolved by precedence.', 'created_at': '', 'updated_at': ''}], 'dropped_instructions': [], 'conflicts': []}, 'primitive_plan': {'route': 'pde_theta_1d', 'engine_family': 'pde_solver', 'route_family': 'pde_solver', 'score': 2.0}}
- `route_card`: '## Structured Route Card\n- Method family: `pde_solver`\n- Instrument type: `american_put`\n- Route: `pde_theta_1d`\n- Engine family: `pde_solver`\n- Route family: `pde_solver`\n- Required primitives:\n  - `trellis.models.pde.grid.Grid` (grid)\n  - `trellis.models.pde.operator.BlackScholesOperator` (spatial_operator)\n  - `trellis.models.pde.theta_method.theta_method_1d` (time_stepping)\n- Resolved instructions:\n  - [route_hint] Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.\n  - [route_hint] Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.\n  - [historical_note] Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.\n  - [historical_note] For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.\n  - [historical_note] For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.\n- Required adapters:\n  - `define_operator_boundary_terminal_conditions`\n- Primary modules to inspect/reuse:\n  - `trellis.models.pde.theta_method`\n- Post-build test targets:\n  - `tests/test_agent/test_build_loop.py`\n- Instruction precedence: follow the approved modules, primitives, and route helper in this card. If older guidance conflicts, treat it as stale and obey this plan.\n- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels.'
- `route_plan`: '## Structured Generation Plan\n- Method family: `pde_solver`\n- Instrument type: `american_put`\n- Repo revision: `2204f054d09cfaa88de56326480925626f4ca2b3`\n- Inspected modules:\n  - `trellis.models.pde.theta_method`\n- Approved Trellis modules for imports:\n  - `trellis.core.date_utils`\n  - `trellis.core.differentiable`\n  - `trellis.core.market_state`\n  - `trellis.core.payoff`\n  - `trellis.core.types`\n  - `trellis.models.black`\n  - `trellis.models.pde`\n  - `trellis.models.pde.crank_nicolson`\n  - `trellis.models.pde.grid`\n  - `trellis.models.pde.implicit_fd`\n  - `trellis.models.pde.operator`\n  - `trellis.models.pde.psor`\n  - `trellis.models.pde.rate_operator`\n  - `trellis.models.pde.theta_method`\n  - `trellis.models.pde.thomas`\n- Public symbols available from the approved modules:\n  - `BlackScholesOperator`\n  - `CEVOperator`\n  - `CashflowSchedule`\n  - `Cashflows`\n  - `DataProvider`\n  - `DayCountConvention`\n  - `DeterministicCashflowPayoff`\n  - `DiscountCurve`\n  - `Frequency`\n  - `Grid`\n  - `HeatOperator`\n  - `HullWhitePDEOperator`\n  - `Instrument`\n  - `MarketState`\n  - `MissingCapabilityError`\n  - `MonteCarloPathPayoff`\n  - `PDEOperator`\n  - `Payoff`\n  - `PresentValue`\n  - `PricingResult`\n  - `ResolvedInputPayoff`\n  - `add_months`\n  - `black76_asset_or_nothing_call`\n  - `black76_asset_or_nothing_put`\n  - `black76_call`\n  - `black76_cash_or_nothing_call`\n  - `black76_cash_or_nothing_put`\n  - `black76_put`\n  - `crank_nicolson_1d`\n  - `garman_kohlhagen_call`\n  - `garman_kohlhagen_put`\n  - `generate_schedule`\n  - `get_accrual_fraction`\n  - `get_bracketing_dates`\n  - `get_numpy`\n  - `gradient`\n  - `hessian`\n  - `implicit_fd_1d`\n  - `psor_1d`\n  - `theta_method_1d`\n  - `thomas_solve`\n  - `year_fraction`\n- Tests to run after generation:\n  - `tests/test_agent/test_build_loop.py`\n- Primitive route:\n  - Route: `pde_theta_1d`\n  - Engine family: `pde_solver`\n  - Route family: `pde_solver`\n  - Route score: `2.00`\n  - Selected primitives:\n    - `trellis.models.pde.grid.Grid` (grid)\n    - `trellis.models.pde.operator.BlackScholesOperator` (spatial_operator)\n    - `trellis.models.pde.theta_method.theta_method_1d` (time_stepping)\n  - Resolved instructions:\n    - [route_hint] Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.\n    - [route_hint] Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.\n    - [historical_note] Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.\n    - [historical_note] For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.\n    - [historical_note] For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.\n  - Required adapters:\n    - `define_operator_boundary_terminal_conditions`\n  - Instruction precedence: follow the approved modules, primitives, and route helper in this plan. If older guidance conflicts, treat it as stale and obey this plan.\n- Every `trellis.*` import in the generated code MUST come from the approved module list above.\n- If you need functionality outside the approved list, say so explicitly instead of inventing an import.'
- `selected_curve_names`: {'discount_curve': 'usd_ois', 'credit_curve': 'usd_ig'}
- `spec_name`: 'AmericanPutSpec'

## Steps
- **trace** `executor_build_20260329154841_f9c8ff3f:root`
  - Label: Analytical build
  - Status: `ok`
  - Notes:
    - The trace mirrors the deterministic GenerationPlan used to assemble the route.
  - Inputs:
    - `issue_id`: None
    - `model`: pde_solver
    - `route_family`: pde_solver
    - `route_name`: pde_theta_1d
    - `task_id`: executor_build_20260329154841_f9c8ff3f
  - Outputs:
    - `route`: {
  "family": "pde_solver",
  "model": "pde_solver",
  "name": "pde_theta_1d"
}
    - `status`: ok
  - **semantic_resolution** `executor_build_20260329154841_f9c8ff3f:semantic_resolution`
    - Label: Resolve contract and route
    - Status: `ok`
    - Parent: `executor_build_20260329154841_f9c8ff3f:root`
    - Notes:
      - Record the semantic contract that drives route selection, not just the final code path.
    - Inputs:
      - `approved_modules`: [
  "trellis.core.date_utils",
  "trellis.core.differentiable",
  "trellis.core.market_state",
  "trellis.core.payoff",
  "trellis.core.types",
  "trellis.models.black",
  "trellis.models.pde",
  "trellis.models.pde.crank_nicolson",
  "trellis.models.pde.grid",
  "trellis.models.pde.implicit_fd",
  "trellis.models.pde.operator",
  "trellis.models.pde.psor",
  "trellis.models.pde.rate_operator",
  "trellis.models.pde.theta_method",
  "trellis.models.pde.thomas"
]
      - `inspected_modules`: [
  "trellis.models.pde.theta_method"
]
      - `instrument_type`: american_put
      - `method`: pde_solver
      - `repo_revision`: 2204f054d09cfaa88de56326480925626f4ca2b3
      - `symbols_to_reuse`: [
  "BlackScholesOperator",
  "CEVOperator",
  "CashflowSchedule",
  "Cashflows",
  "DataProvider",
  "DayCountConvention",
  "DeterministicCashflowPayoff",
  "DiscountCurve",
  "Frequency",
  "Grid",
  "HeatOperator",
  "HullWhitePDEOperator",
  "Instrument",
  "MarketState",
  "MissingCapabilityError",
  "MonteCarloPathPayoff",
  "PDEOperator",
  "Payoff",
  "PresentValue",
  "PricingResult",
  "ResolvedInputPayoff",
  "add_months",
  "black76_asset_or_nothing_call",
  "black76_asset_or_nothing_put",
  "black76_call",
  "black76_cash_or_nothing_call",
  "black76_cash_or_nothing_put",
  "black76_put",
  "crank_nicolson_1d",
  "garman_kohlhagen_call",
  "garman_kohlhagen_put",
  "generate_schedule",
  "get_accrual_fraction",
  "get_bracketing_dates",
  "get_numpy",
  "gradient",
  "hessian",
  "implicit_fd_1d",
  "psor_1d",
  "theta_method_1d"
]
      - `uncertainty_flags`: []
    - Outputs:
      - `instruction_resolution`: {
  "conflict_count": 0,
  "conflicts": [],
  "dropped_instruction_count": 0,
  "dropped_instructions": [],
  "effective_instruction_count": 5,
  "effective_instructions": [
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:2",
      "instruction_type": "route_hint",
      "precedence_rank": 48,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:2",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 2",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:5",
      "instruction_type": "route_hint",
      "precedence_rank": 45,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:5",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 5",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:1",
      "instruction_type": "historical_note",
      "precedence_rank": 49,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:1",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 1",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:3",
      "instruction_type": "historical_note",
      "precedence_rank": 47,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:3",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 3",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:4",
      "instruction_type": "historical_note",
      "precedence_rank": 46,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:4",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 4",
      "updated_at": ""
    }
  ],
  "route": "pde_theta_1d"
}
      - `model`: pde_solver
      - `primitive_plan_score`: 2.0
      - `route_family`: pde_solver
      - `route_name`: pde_theta_1d
  - **instruction_lifecycle** `executor_build_20260329154841_f9c8ff3f:instruction_lifecycle`
    - Label: Resolve route guidance lifecycle
    - Status: `ok`
    - Parent: `executor_build_20260329154841_f9c8ff3f:root`
    - Notes:
      - List the effective, dropped, and conflicting route guidance records so replay consumers can see what actually governed the build.
    - Inputs:
      - `conflict_count`: 0
      - `dropped_instruction_count`: 0
      - `effective_instruction_count`: 5
      - `route`: pde_theta_1d
    - Outputs:
      - `instruction_resolution`: {
  "conflict_count": 0,
  "conflicts": [],
  "dropped_instruction_count": 0,
  "dropped_instructions": [],
  "effective_instruction_count": 5,
  "effective_instructions": [
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:2",
      "instruction_type": "route_hint",
      "precedence_rank": 48,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:2",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 2",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:5",
      "instruction_type": "route_hint",
      "precedence_rank": 45,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:5",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 5",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:1",
      "instruction_type": "historical_note",
      "precedence_rank": 49,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:1",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 1",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:3",
      "instruction_type": "historical_note",
      "precedence_rank": 47,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:3",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 3",
      "updated_at": ""
    },
    {
      "conflict_policy": "prefer_newer",
      "created_at": "",
      "id": "pde_theta_1d:note:4",
      "instruction_type": "historical_note",
      "precedence_rank": 46,
      "rationale": "Route notes are retained as structured guidance and resolved by precedence.",
      "scope_features": [],
      "scope_instruments": [
        "american_put"
      ],
      "scope_methods": [
        "pde_solver"
      ],
      "scope_modules": [],
      "scope_routes": [
        "pde_theta_1d"
      ],
      "source_id": "pde_theta_1d:note:4",
      "source_kind": "route_card",
      "source_revision": "",
      "statement": "For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.",
      "status": "active",
      "supersedes": [],
      "title": "Route note 4",
      "updated_at": ""
    }
  ],
  "route": "pde_theta_1d"
}
  - **decomposition** `executor_build_20260329154841_f9c8ff3f:decomposition`
    - Label: Select reusable kernels
    - Status: `ok`
    - Parent: `executor_build_20260329154841_f9c8ff3f:root`
    - Notes:
      - Capture the reusable valuation components and any exact basis-claim assembly.
    - Inputs:
      - `adapters`: [
  "define_operator_boundary_terminal_conditions"
]
      - `blockers`: []
      - `notes`: [
  "Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.",
  "Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.",
  "For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.",
  "For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.",
  "Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally."
]
      - `primitives`: [
  {
    "module": "trellis.models.pde.grid",
    "required": true,
    "role": "grid",
    "symbol": "Grid"
  },
  {
    "module": "trellis.models.pde.operator",
    "required": true,
    "role": "spatial_operator",
    "symbol": "BlackScholesOperator"
  },
  {
    "module": "trellis.models.pde.theta_method",
    "required": true,
    "role": "time_stepping",
    "symbol": "theta_method_1d"
  }
]
    - Outputs:
      - `reuse_decision`: route_local
      - `selected_primitives`: [
  {
    "module": "trellis.models.pde.grid",
    "required": true,
    "role": "grid",
    "symbol": "Grid"
  },
  {
    "module": "trellis.models.pde.operator",
    "required": true,
    "role": "spatial_operator",
    "symbol": "BlackScholesOperator"
  },
  {
    "module": "trellis.models.pde.theta_method",
    "required": true,
    "role": "time_stepping",
    "symbol": "theta_method_1d"
  }
]
  - **assembly** `executor_build_20260329154841_f9c8ff3f:assembly`
    - Label: Assemble route from kernels
    - Status: `ok`
    - Parent: `executor_build_20260329154841_f9c8ff3f:root`
    - Notes:
      - Prefer thin orchestration around existing analytical kernels and route helpers.
    - Inputs:
      - `adapters`: [
  "define_operator_boundary_terminal_conditions"
]
      - `approved_modules`: [
  "trellis.core.date_utils",
  "trellis.core.differentiable",
  "trellis.core.market_state",
  "trellis.core.payoff",
  "trellis.core.types",
  "trellis.models.black",
  "trellis.models.pde",
  "trellis.models.pde.crank_nicolson",
  "trellis.models.pde.grid",
  "trellis.models.pde.implicit_fd",
  "trellis.models.pde.operator",
  "trellis.models.pde.psor",
  "trellis.models.pde.rate_operator",
  "trellis.models.pde.theta_method",
  "trellis.models.pde.thomas"
]
      - `route_helper`: None
    - Outputs:
      - `assembly_card`: ## Structured Route Card
- Method family: `pde_solver`
- Instrument type: `american_put`
- Route: `pde_theta_1d`
- Engine family: `pde_solver`
- Route family: `pde_solver`
- Required primitives:
  - `trellis.models.pde.grid.Grid` (grid)
  - `trellis.models.pde.operator.BlackScholesOperator` (spatial_operator)
  - `trellis.models.pde.theta_method.theta_method_1d` (time_stepping)
- Resolved instructions:
  - [route_hint] Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.
  - [route_hint] Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.
  - [historical_note] Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.
  - [historical_note] For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.
  - [historical_note] For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.
- Required adapters:
  - `define_operator_boundary_terminal_conditions`
- Primary modules to inspect/reuse:
  - `trellis.models.pde.theta_method`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the approved modules, primitives, and route helper in this card. If older guidance conflicts, treat it as stale and obey this plan.
- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels.
      - `helper_modules`: [
  "trellis.models.pde.grid",
  "trellis.models.pde.operator",
  "trellis.models.pde.theta_method"
]
      - `route_helper`: None
  - **validation** `executor_build_20260329154841_f9c8ff3f:validation`
    - Label: Validate route and fallbacks
    - Status: `ok`
    - Parent: `executor_build_20260329154841_f9c8ff3f:root`
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
  - **output** `executor_build_20260329154841_f9c8ff3f:output`
    - Label: Final analytical artifact
    - Status: `ok`
    - Parent: `executor_build_20260329154841_f9c8ff3f:root`
    - Notes:
      - Persist both the machine-readable trace and the text rendering from the same source of truth.
    - Inputs:
      - `model`: pde_solver
      - `route`: pde_theta_1d
    - Outputs:
      - `route_card`: ## Structured Route Card
- Method family: `pde_solver`
- Instrument type: `american_put`
- Route: `pde_theta_1d`
- Engine family: `pde_solver`
- Route family: `pde_solver`
- Required primitives:
  - `trellis.models.pde.grid.Grid` (grid)
  - `trellis.models.pde.operator.BlackScholesOperator` (spatial_operator)
  - `trellis.models.pde.theta_method.theta_method_1d` (time_stepping)
- Resolved instructions:
  - [route_hint] Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.
  - [route_hint] Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.
  - [historical_note] Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.
  - [historical_note] For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.
  - [historical_note] For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.
- Required adapters:
  - `define_operator_boundary_terminal_conditions`
- Primary modules to inspect/reuse:
  - `trellis.models.pde.theta_method`
- Post-build test targets:
  - `tests/test_agent/test_build_loop.py`
- Instruction precedence: follow the approved modules, primitives, and route helper in this card. If older guidance conflicts, treat it as stale and obey this plan.
- Use approved Trellis imports only. Prefer thin adapters over bespoke numerical kernels.
      - `route_plan`: ## Structured Generation Plan
- Method family: `pde_solver`
- Instrument type: `american_put`
- Repo revision: `2204f054d09cfaa88de56326480925626f4ca2b3`
- Inspected modules:
  - `trellis.models.pde.theta_method`
- Approved Trellis modules for imports:
  - `trellis.core.date_utils`
  - `trellis.core.differentiable`
  - `trellis.core.market_state`
  - `trellis.core.payoff`
  - `trellis.core.types`
  - `trellis.models.black`
  - `trellis.models.pde`
  - `trellis.models.pde.crank_nicolson`
  - `trellis.models.pde.grid`
  - `trellis.models.pde.implicit_fd`
  - `trellis.models.pde.operator`
  - `trellis.models.pde.psor`
  - `trellis.models.pde.rate_operator`
  - `trellis.models.pde.theta_method`
  - `trellis.models.pde.thomas`
- Public symbols available from the approved modules:
  - `BlackScholesOperator`
  - `CEVOperator`
  - `CashflowSchedule`
  - `Cashflows`
  - `DataProvider`
  - `DayCountConvention`
  - `DeterministicCashflowPayoff`
  - `DiscountCurve`
  - `Frequency`
  - `Grid`
  - `HeatOperator`
  - `HullWhitePDEOperator`
  - `Instrument`
  - `MarketState`
  - `MissingCapabilityError`
  - `MonteCarloPathPayoff`
  - `PDEOperator`
  - `Payoff`
  - `PresentValue`
  - `PricingResult`
  - `ResolvedInputPayoff`
  - `add_months`
  - `black76_asset_or_nothing_call`
  - `black76_asset_or_nothing_put`
  - `black76_call`
  - `black76_cash_or_nothing_call`
  - `black76_cash_or_nothing_put`
  - `black76_put`
  - `crank_nicolson_1d`
  - `garman_kohlhagen_call`
  - `garman_kohlhagen_put`
  - `generate_schedule`
  - `get_accrual_fraction`
  - `get_bracketing_dates`
  - `get_numpy`
  - `gradient`
  - `hessian`
  - `implicit_fd_1d`
  - `psor_1d`
  - `theta_method_1d`
  - `thomas_solve`
  - `year_fraction`
- Tests to run after generation:
  - `tests/test_agent/test_build_loop.py`
- Primitive route:
  - Route: `pde_theta_1d`
  - Engine family: `pde_solver`
  - Route family: `pde_solver`
  - Route score: `2.00`
  - Selected primitives:
    - `trellis.models.pde.grid.Grid` (grid)
    - `trellis.models.pde.operator.BlackScholesOperator` (spatial_operator)
    - `trellis.models.pde.theta_method.theta_method_1d` (time_stepping)
  - Resolved instructions:
    - [route_hint] Use `BlackScholesOperator(sigma_fn, r_fn)` with callable inputs; do not invent `log_grid_pde` or `uniform_grid_pde` helper names.
    - [route_hint] Do not pass a callable terminal payoff into `theta_method_1d`; the solver expects an ndarray and will copy it internally.
    - [historical_note] Construct `Grid(x_min, x_max, n_x, T, n_t, log_spacing=...)` and a terminal-condition ndarray before calling `theta_method_1d(grid, operator, terminal_condition, theta=...)`.
    - [historical_note] For American puts, pass `exercise_values` and `exercise_fn=max` instead of a bare `AMERICAN` constant or a `psor_pde` alias.
    - [historical_note] For barrier or digital payoffs, keep `lower_bc_fn` / `upper_bc_fn` callables explicit and use `rannacher_timesteps` to smooth the first backward steps.
  - Required adapters:
    - `define_operator_boundary_terminal_conditions`
  - Instruction precedence: follow the approved modules, primitives, and route helper in this plan. If older guidance conflicts, treat it as stale and obey this plan.
- Every `trellis.*` import in the generated code MUST come from the approved module list above.
- If you need functionality outside the approved list, say so explicitly instead of inventing an import.
      - `trace_type`: analytical