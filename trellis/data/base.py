"""Abstract base for data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date


class BaseDataProvider(ABC):
    """Base class for market data providers."""

    @abstractmethod
    def fetch_yields(self, as_of: date | None = None) -> dict[float, float]:
        """Return {tenor_years: yield} for the given (or most recent) date."""
        ...

    def fetch_market_snapshot(self, as_of: date | None = None):
        """Return a richer market snapshot when the provider supports one.

        Providers may override this to supply named discount/forecast curves,
        vol surfaces, credit curves, and FX rates together. The default
        implementation keeps legacy providers yield-only.
        """
        raise NotImplementedError
