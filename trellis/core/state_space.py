"""StateSpace: discrete states with market-implied probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trellis.core.market_state import MarketState


@dataclass(frozen=True)
class StateSpace:
    """A set of mutually exclusive states with probabilities and conditional MarketStates.

    Parameters
    ----------
    states : dict[str, tuple[float, MarketState]]
        Mapping from state name to ``(probability, conditional_market_state)``.
        Probabilities should sum to ~1.0.
    """

    states: dict[str, tuple[float, MarketState]]

    def __post_init__(self):
        total = sum(prob for prob, _ in self.states.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"State probabilities sum to {total:.4f}, expected ~1.0"
            )

    @property
    def state_names(self) -> list[str]:
        return list(self.states.keys())

    def probability(self, state: str) -> float:
        return self.states[state][0]

    def market_state(self, state: str) -> MarketState:
        return self.states[state][1]
