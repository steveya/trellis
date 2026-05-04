"""Execution visitors that lower route-free IR into numerical runtimes."""

from trellis.execution.visitors.aggregation import (
    DiscountedExecutionSummary,
    FutureValueExecutionSummary,
    summarize_discounted_execution_ir,
    summarize_future_value_execution_ir,
)
from trellis.execution.visitors.cashflow_expand import known_cashflow_obligations
from trellis.execution.visitors.event_compile import (
    compile_callable_bond_spec_from_execution_ir,
)
from trellis.execution.visitors.normalize import normalize_execution_ir
from trellis.execution.visitors.requirements import derive_requirement_hints
from trellis.execution.visitors.schedule import (
    ExecutionScheduleEntry,
    execution_event_schedule,
)
from trellis.execution.visitors.simulation_bridge import (
    BermudanBestOfBasketLatticeControls,
    BermudanBestOfBasketLatticeResult,
    BermudanBestOfBasketMCControls,
    BermudanBestOfBasketMCInputs,
    BermudanBestOfBasketMCResult,
    build_future_value_cube_from_execution_ir,
    compile_factor_state_simulation_ir_from_execution_ir,
    compile_swap_spec_from_execution_ir,
    price_bermudan_best_of_basket_lattice,
    price_bermudan_best_of_basket_monte_carlo,
)

__all__ = [
    "BermudanBestOfBasketLatticeControls",
    "BermudanBestOfBasketLatticeResult",
    "BermudanBestOfBasketMCControls",
    "BermudanBestOfBasketMCInputs",
    "BermudanBestOfBasketMCResult",
    "DiscountedExecutionSummary",
    "ExecutionScheduleEntry",
    "FutureValueExecutionSummary",
    "build_future_value_cube_from_execution_ir",
    "compile_callable_bond_spec_from_execution_ir",
    "compile_factor_state_simulation_ir_from_execution_ir",
    "compile_swap_spec_from_execution_ir",
    "derive_requirement_hints",
    "execution_event_schedule",
    "known_cashflow_obligations",
    "normalize_execution_ir",
    "price_bermudan_best_of_basket_lattice",
    "price_bermudan_best_of_basket_monte_carlo",
    "summarize_discounted_execution_ir",
    "summarize_future_value_execution_ir",
]
