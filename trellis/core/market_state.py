"""MarketState: immutable container of market data used for pricing.

A MarketState holds everything a payoff needs to compute a price:
discount curves, volatility surfaces, spot prices, credit curves, etc.
It is frozen (immutable) so that multiple payoffs can safely share the
same market snapshot without risk of accidental modification.
"""

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
    fixing_histories : dict[str, dict[date, float]] or None
        Historical observed fixings keyed by rate index or fixing-history name.
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
    selected_curve_names : dict[str, str] or None
        Canonical names chosen when a snapshot is compiled into this runtime
        state. Keys are curve kinds such as ``"discount_curve"`` and
        ``"forecast_curve"``.
    market_provenance : dict[str, object] or None
        Snapshot-level provenance for the resolved market inputs, when available.
    """

    as_of: date
    settlement: date
    discount: DiscountCurve | None = None
    forward_curve: ForwardCurve | None = None
    vol_surface: VolSurface | None = None
    state_space: StateSpace | None = None
    credit_curve: CreditCurve | None = None
    fixing_histories: dict[str, dict[date, float]] | None = None
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
    selected_curve_names: dict[str, str] | None = None
    market_provenance: dict[str, object] | None = None

    def __post_init__(self):
        """Auto-fill single-valued defaults from their dict counterparts.

        For example, if underlier_spots has exactly one entry and spot is None,
        spot is set to that single value. Same logic for local_vol_surface,
        jump_parameters, and model_parameters. Also builds a default
        forward_curve from the discount curve if none was provided.
        """
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
        """Return a ForwardCurve for projecting future interest rates.

        If rate_index is given (e.g. "USD-SOFR-3M") and a matching curve
        exists in forecast_curves, that curve is used. Otherwise falls back
        to the default forward_curve derived from the discount curve.

        Raises ValueError if no forward curve is available at all.
        """
        from trellis.curves.forward_curve import ForwardCurve
        if rate_index is not None and self.forecast_curves is not None:
            if rate_index in self.forecast_curves:
                return ForwardCurve(self.forecast_curves[rate_index])
        if self.forward_curve is not None:
            return self.forward_curve
        raise ValueError("No forward curve available")

    def fixing_history(self, rate_index: str | None = None):
        """Return a selected fixing history, if one is available."""
        if not self.fixing_histories:
            return None
        if rate_index is not None:
            if rate_index in self.fixing_histories:
                return self.fixing_histories[rate_index]
            raise ValueError(f"Unknown fixing history: {rate_index}")
        selected_name = self.selected_curve_name("fixing_history")
        if selected_name is not None and selected_name in self.fixing_histories:
            return self.fixing_histories[selected_name]
        if len(self.fixing_histories) == 1:
            return next(iter(self.fixing_histories.values()))
        raise ValueError("Multiple fixing histories available; set a default fixing history name")

    def selected_curve_name(self, kind: str) -> str | None:
        """Return the selected canonical curve name for a curve kind."""
        if self.selected_curve_names is None:
            return None
        return self.selected_curve_names.get(kind)

    def materialized_calibrated_object(
        self,
        *,
        object_kind: str,
        object_name: str | None = None,
    ) -> dict[str, object] | None:
        """Return one calibrated-object materialization record, when available."""
        from trellis.models.calibration.materialization import resolve_materialized_object

        return resolve_materialized_object(
            self,
            object_kind=object_kind,
            object_name=object_name,
        )

    @property
    def valuation_date(self) -> date:
        """Backward-compatible alias used by generated pricers for valuation time."""
        return self.settlement

    @property
    def available_capabilities(self) -> set[str]:
        """Return the canonical capability names exposed by this market state."""
        caps: set[str] = set()
        if self.discount is not None:
            caps.add("discount_curve")
            caps.add("forward_curve")
        if self.forward_curve is not None or self.forecast_curves:
            caps.add("forward_curve")
        if self.vol_surface is not None:
            caps.add("black_vol_surface")
        if self.state_space is not None:
            caps.add("state_space")
        if self.credit_curve is not None:
            caps.add("credit_curve")
        if self.fixing_histories:
            caps.add("fixing_history")
        if self.fx_rates:
            caps.add("fx_rates")
        if self.spot is not None or self.underlier_spots:
            caps.add("spot")
        if self.local_vol_surface is not None or self.local_vol_surfaces:
            caps.add("local_vol_surface")
        if self.jump_parameters is not None or self.jump_parameter_sets:
            caps.add("jump_parameters")
        if self.model_parameters is not None or self.model_parameter_sets:
            caps.add("model_parameters")
        return caps

    def summarize_for_audit(self) -> dict:
        """Serialize auditable market state snapshot for model audit trail.

        Samples curve objects at standard tenors and returns a JSON-serializable
        dict suitable for inclusion in a ModelAuditRecord.
        """
        _TENORS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
        summary: dict = {
            "as_of": self.as_of.isoformat(),
            "settlement": self.settlement.isoformat(),
            "spot": self.spot,
            "selected_curve_names": dict(self.selected_curve_names or {}),
            "market_provenance": {
                k: str(v) for k, v in (self.market_provenance or {}).items()
            },
        }
        selected_calibrated = (
            dict(self.market_provenance.get("selected_calibrated_objects", {}))
            if isinstance(self.market_provenance, dict)
            else {}
        )
        if selected_calibrated:
            summary["selected_calibrated_objects"] = selected_calibrated
        if self.discount is not None:
            try:
                summary["discount_factors"] = {
                    str(t): float(self.discount.discount(t)) for t in _TENORS
                }
            except Exception:
                summary["discount_factors"] = "error_sampling"
        if self.vol_surface is not None:
            try:
                atm_strike = self.spot or 100.0
                summary["vol_surface_atm"] = {
                    str(t): float(self.vol_surface.black_vol(t, atm_strike))
                    for t in _TENORS
                }
            except Exception:
                summary["vol_surface_atm"] = "error_sampling"
        if self.credit_curve is not None:
            try:
                summary["credit_survival"] = {
                    str(t): float(self.credit_curve.survival_probability(t))
                    for t in _TENORS
                }
            except Exception:
                summary["credit_survival"] = "error_sampling"
        if self.underlier_spots:
            summary["underlier_spots"] = dict(self.underlier_spots)
        if self.fx_rates:
            summary["fx_rates"] = {
                k: float(v.spot) for k, v in self.fx_rates.items()
            }
        if self.fixing_histories:
            summary["fixing_histories"] = {
                name: {
                    fix_date.isoformat(): float(value)
                    for fix_date, value in sorted(history.items())
                }
                for name, history in sorted(self.fixing_histories.items())
            }
        return summary
