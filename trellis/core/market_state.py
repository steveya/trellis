"""MarketState: frozen bag of market capabilities for payoff evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

from trellis.core.types import DiscountCurve

if TYPE_CHECKING:
    from trellis.core.state_space import StateSpace
    from trellis.curves.credit_curve import CreditCurve
    from trellis.curves.forward_curve import ForwardCurve
    from trellis.instruments.fx import FXRate
    from trellis.models.vol_surface import VolSurface


class MissingCapabilityError(Exception):
    """Raised when a MarketState lacks market data required by a Payoff."""

    def __init__(self, missing: set[str], available: set[str],
                 details: list[str] | None = None):
        self.missing = missing
        self.available = available
        self.details = details or []
        msg = f"Missing market data: {sorted(missing)}."
        if self.details:
            msg += "\n" + "\n".join(self.details)
        else:
            msg += f" Available: {sorted(available)}"
        super().__init__(msg)


@dataclass(frozen=True)
class MarketState:
    """Immutable bag of market data for payoff evaluation.

    Parameters
    ----------
    as_of : date
        Observation date for market data.
    settlement : date
        Cash settlement date.
    discount : DiscountCurve or None
        Discount curve (OIS curve for discounting cashflows).
    forward_curve : ForwardCurve or None
        Default forward rate curve. Auto-constructed from *discount* if not provided.
    vol_surface : VolSurface or None
        Volatility surface for option pricing.
    state_space : StateSpace or None
        Discrete states with probabilities for scenario-weighted pricing.
    credit_curve : CreditCurve or None
        Credit / survival probability curve.
    forecast_curves : dict[str, DiscountCurve] or None
        Forecast curves keyed by rate index name (e.g. ``"USD-SOFR-3M"``).
        Also used for foreign discount curves (e.g. ``"EUR-DISC"``).
    fx_rates : dict[str, FXRate] or None
        FX spot rates keyed by pair (e.g. ``"EURUSD"``).
    """

    as_of: date
    settlement: date
    discount: DiscountCurve | None = None
    forward_curve: ForwardCurve | None = None
    vol_surface: VolSurface | None = None
    state_space: StateSpace | None = None
    credit_curve: CreditCurve | None = None
    forecast_curves: dict[str, DiscountCurve] | None = None
    fx_rates: dict[str, FXRate] | None = None

    def __post_init__(self):
        if self.discount is not None and self.forward_curve is None:
            from trellis.curves.forward_curve import ForwardCurve
            object.__setattr__(self, "forward_curve", ForwardCurve(self.discount))

    def forecast_forward_curve(self, rate_index: str | None = None):
        """Get a ForwardCurve for the given rate index.

        In multi-curve mode, looks up the forecast curve keyed by *rate_index*.
        Falls back to the default forward_curve (from discount) when
        *rate_index* is None or not found in forecast_curves.
        """
        from trellis.curves.forward_curve import ForwardCurve
        if rate_index is not None and self.forecast_curves is not None:
            if rate_index in self.forecast_curves:
                return ForwardCurve(self.forecast_curves[rate_index])
        if self.forward_curve is not None:
            return self.forward_curve
        raise ValueError("No forward curve available")

    @property
    def available_capabilities(self) -> set[str]:
        caps: set[str] = set()
        if self.discount is not None:
            caps.add("discount")
            caps.add("forward_rate")
        if self.forward_curve is not None:
            caps.add("forward_rate")
        if self.vol_surface is not None:
            caps.add("black_vol")
        if self.state_space is not None:
            caps.add("state_space")
        if self.credit_curve is not None:
            caps.add("credit")
        if self.forecast_curves is not None:
            caps.add("forecast_rate")
        if self.fx_rates is not None:
            caps.add("fx")
        return caps
