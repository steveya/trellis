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
        """Validate that the discrete-state probabilities sum to approximately one."""
        total = sum(prob for prob, _ in self.states.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"State probabilities sum to {total:.4f}, expected ~1.0"
            )

    @property
    def state_names(self) -> list[str]:
        """Return the configured scenario/state labels in insertion order."""
        return list(self.states.keys())

    def probability(self, state: str) -> float:
        """Return the probability weight attached to ``state``."""
        return self.states[state][0]

    def market_state(self, state: str) -> MarketState:
        """Return the conditional market state associated with ``state``."""
        return self.states[state][1]
