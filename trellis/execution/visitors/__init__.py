"""Execution visitors that lower route-free IR into numerical runtimes."""

from trellis.execution.visitors.cashflow_expand import known_cashflow_obligations
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
    price_bermudan_best_of_basket_lattice,
    price_bermudan_best_of_basket_monte_carlo,
)

__all__ = [
    "BermudanBestOfBasketLatticeControls",
    "BermudanBestOfBasketLatticeResult",
    "BermudanBestOfBasketMCControls",
    "BermudanBestOfBasketMCInputs",
    "BermudanBestOfBasketMCResult",
    "ExecutionScheduleEntry",
    "derive_requirement_hints",
    "execution_event_schedule",
    "known_cashflow_obligations",
    "normalize_execution_ir",
    "price_bermudan_best_of_basket_lattice",
    "price_bermudan_best_of_basket_monte_carlo",
]
