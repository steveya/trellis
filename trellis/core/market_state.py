"""MarketState: frozen bag of market capabilities for payoff evaluation."""

from __future__ import annotations

from dataclasses import dataclass
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
        """Store the missing and available capability sets and build a readable error."""
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
    spot : float or None
        Default underlier spot for equity-like tasks when a single primary
        underlier is selected from the market snapshot.
    underlier_spots : dict[str, float] or None
        Named underlier spots keyed by ticker or underlier identifier.
    local_vol_surface : callable or None
        Default local-vol function ``sigma(S, t)`` for local-vol tasks.
    local_vol_surfaces : dict[str, callable] or None
        Named local-vol functions keyed by model/surface name.
    jump_parameters : dict[str, object] or None
        Default jump-diffusion parameter pack for jump models.
    jump_parameter_sets : dict[str, dict[str, object]] or None
        Named jump-diffusion parameter packs.
    model_parameters : dict[str, object] or None
        Default model parameter pack (e.g. Heston).
    model_parameter_sets : dict[str, dict[str, object]] or None
        Named model parameter packs.
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
    spot: float | None = None
    underlier_spots: dict[str, float] | None = None
    local_vol_surface: object | None = None
    local_vol_surfaces: dict[str, object] | None = None
    jump_parameters: dict[str, object] | None = None
    jump_parameter_sets: dict[str, dict[str, object]] | None = None
    model_parameters: dict[str, object] | None = None
    model_parameter_sets: dict[str, dict[str, object]] | None = None

    def __post_init__(self):
        """Populate convenient scalar defaults when only one named input is available."""
        if self.discount is not None and self.forward_curve is None:
            from trellis.curves.forward_curve import ForwardCurve
            object.__setattr__(self, "forward_curve", ForwardCurve(self.discount))
        if self.spot is None and self.underlier_spots:
            if len(self.underlier_spots) == 1:
                object.__setattr__(self, "spot", next(iter(self.underlier_spots.values())))
        if self.local_vol_surface is None and self.local_vol_surfaces:
            if len(self.local_vol_surfaces) == 1:
                object.__setattr__(
                    self,
                    "local_vol_surface",
                    next(iter(self.local_vol_surfaces.values())),
                )
        if self.jump_parameters is None and self.jump_parameter_sets:
            if len(self.jump_parameter_sets) == 1:
                object.__setattr__(
                    self,
                    "jump_parameters",
                    next(iter(self.jump_parameter_sets.values())),
                )
        if self.model_parameters is None and self.model_parameter_sets:
            if len(self.model_parameter_sets) == 1:
                object.__setattr__(
                    self,
                    "model_parameters",
                    next(iter(self.model_parameter_sets.values())),
                )

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
        """Return the normalized capability names exposed by this market state."""
        caps: set[str] = set()
        if self.discount is not None:
            caps.add("discount_curve")
            caps.add("forward_curve")
        if self.forward_curve is not None or self.forecast_curves is not None:
            caps.add("forward_curve")
        if self.vol_surface is not None:
            caps.add("black_vol_surface")
        if self.state_space is not None:
            caps.add("state_space")
        if self.credit_curve is not None:
            caps.add("credit_curve")
        if self.fx_rates is not None:
            caps.add("fx_rates")
        if self.spot is not None or self.underlier_spots is not None:
            caps.add("spot")
        if self.local_vol_surface is not None or self.local_vol_surfaces is not None:
            caps.add("local_vol_surface")
        if self.jump_parameters is not None or self.jump_parameter_sets is not None:
            caps.add("jump_parameters")
        if self.model_parameters is not None or self.model_parameter_sets is not None:
            caps.add("model_parameters")
        return caps
