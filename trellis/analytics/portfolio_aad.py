"""Typed portfolio-AAD request and result contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable

from trellis.analytics.risk_factors import (
    RiskFactorCoordinate,
    RiskFactorId,
    RiskFactorRegistry,
    SparseRiskVector,
)


_EARLY_EXERCISE_AAD_POLICY = "hard_exercise_projection_smooth_interior"
_DEFAULT_EARLY_EXERCISE_TREE_STEPS = 128
_DEFAULT_EARLY_EXERCISE_BOUNDARY_TOLERANCE = 1.0e-12


def _sorted_unique_factors(factors: Iterable[RiskFactorId]) -> tuple[RiskFactorId, ...]:
    factor_map: dict[RiskFactorId, RiskFactorId] = {}
    for factor in factors:
        if not isinstance(factor, RiskFactorId):
            raise TypeError("selected factors must be RiskFactorId instances")
        factor_map[factor] = factor
    return tuple(factor for _, factor in sorted(factor_map.items(), key=lambda item: item[0].key))


def _copy_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(metadata or {})


def _copy_diagnostics(
    diagnostics: Iterable[Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], ...]:
    return tuple(dict(diagnostic) for diagnostic in (diagnostics or ()))


def _axis_key(value: str) -> str | float:
    try:
        return float(value)
    except ValueError:
        return value


@dataclass(frozen=True)
class AADSupportDecision:
    """Adapter support decision for one position and request."""

    supported: bool
    reason: str
    factor_dependencies: tuple[RiskFactorId, ...] = field(default_factory=tuple)
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported", bool(self.supported))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(
            self,
            "factor_dependencies",
            _sorted_unique_factors(self.factor_dependencies),
        )
        object.__setattr__(self, "diagnostics", _copy_diagnostics(self.diagnostics))

    @property
    def support_status(self) -> str:
        """Return a compact support status string."""
        return "supported" if self.supported else "unsupported"

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly support-decision payload."""
        return {
            "supported": self.supported,
            "support_status": self.support_status,
            "reason": self.reason,
            "factor_dependencies": [
                factor.to_payload()
                for factor in self.factor_dependencies
            ],
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> AADSupportDecision:
        """Build a support decision from :meth:`to_payload` output."""
        return cls(
            supported=bool(payload["supported"]),
            reason=str(payload["reason"]),
            factor_dependencies=tuple(
                RiskFactorId.from_payload(entry)
                for entry in payload.get("factor_dependencies", ())
            ),
            diagnostics=payload.get("diagnostics") or (),
        )


@runtime_checkable
class TradeAADAdapter(Protocol):
    """Structural protocol for product-family portfolio-AAD adapters."""

    def support_decision(
        self,
        position_name: str,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> AADSupportDecision:
        """Return whether this adapter supports a position under *request*."""
        ...

    def factor_dependencies(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> tuple[RiskFactorId, ...]:
        """Return the factors needed to differentiate the supplied position."""
        ...

    def value(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> float:
        """Return the scalar value represented by this adapter."""
        ...

    def vjp(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
        weight: float = 1.0,
    ) -> SparseRiskVector:
        """Return a sparse VJP risk vector for the supplied position."""
        ...


@dataclass(frozen=True)
class DefaultUnsupportedAADPolicy:
    """Default fail-closed policy for positions outside portfolio-AAD support."""

    include_value_when_priced: bool = True

    def record(
        self,
        *,
        position_name: str,
        instrument: object,
        reason: str,
        request: PortfolioAADRequest,
        priced_value_available: bool = False,
    ) -> UnsupportedAADPosition:
        """Return the typed unsupported-position record for a failed adapter match."""
        return UnsupportedAADPosition(
            position_name=position_name,
            instrument_type=type(instrument).__name__,
            reason=reason,
            requested_factors=request.selected_factors,
            included_in_value=bool(self.include_value_when_priced and priced_value_available),
            included_in_risk=False,
            fallback_method=None,
        )


@dataclass(frozen=True)
class BondCurveAADMarketContext:
    """Market context for the bounded shared-curve bond AAD lane."""

    curve: object
    settlement: date
    curve_name: str = "shared_curve"
    currency: str | None = None
    provenance_namespace: str | None = "portfolio_aad"
    object_path: str = ""

    def coordinates(self) -> tuple[RiskFactorCoordinate, ...]:
        """Return canonical curve coordinates for this context."""
        return RiskFactorRegistry().discover_yield_curve(
            self.curve,
            object_name=self.curve_name,
            currency=self.currency,
            object_path=self.object_path,
            provenance_namespace=self.provenance_namespace,
        )


@dataclass(frozen=True)
class BondCurveAADAdapter:
    """Concrete adapter for fixed-rate bond risk over one shared yield curve."""

    def support_decision(
        self,
        position_name: str,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> AADSupportDecision:
        """Return whether this adapter supports a bond under *market_context*."""
        from trellis.instruments.bond import Bond

        if not isinstance(instrument, Bond):
            return AADSupportDecision(False, "unsupported_instrument_type")
        if instrument.maturity_date is None:
            return AADSupportDecision(False, "bond_maturity_date_required")
        try:
            dependencies = self.factor_dependencies(instrument, market_context, request)
        except Exception as exc:
            return AADSupportDecision(
                False,
                "yield_curve_nodes_unavailable",
                diagnostics=(
                    {
                        "position_name": str(position_name),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ),
            )
        return AADSupportDecision(
            True,
            "supported_bond_curve_aad",
            factor_dependencies=dependencies,
        )

    def factor_dependencies(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> tuple[RiskFactorId, ...]:
        """Return shared-curve zero-rate factors for a supported bond."""
        context = self._context(market_context)
        return tuple(coordinate.factor_id for coordinate in context.coordinates())

    def value(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> float:
        """Return the bond value for this adapter's market context."""
        context = self._context(market_context)
        return float(instrument.price(context.curve, context.settlement))

    def vjp(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
        weight: float = 1.0,
    ) -> SparseRiskVector:
        """Return the sparse VJP vector for one bond position."""
        from trellis.core.differentiable import get_numpy, vjp

        context = self._context(market_context)
        curve = context.curve
        coordinates = context.coordinates()
        tenors = getattr(curve, "tenors", None)
        rates = getattr(curve, "rates", None)
        if tenors is None or rates is None:
            raise ValueError("bond curve AAD requires curve tenors and rates")
        np = get_numpy()
        curve_cls = type(curve)
        tenors_arr = np.asarray(tenors, dtype=float)
        rates_arr = np.asarray(rates, dtype=float)

        def value_from_rates(rates_vec):
            traced_curve = curve_cls(tenors_arr, rates_vec)
            return instrument.price(traced_curve, context.settlement)

        _value, pullback = vjp(value_from_rates, rates_arr)
        gradient = np.asarray(pullback(float(weight)), dtype=float)
        vector = SparseRiskVector.from_items(
            (coordinate.factor_id, sensitivity)
            for coordinate, sensitivity in zip(coordinates, gradient)
        )
        return request.filter_vector(vector)

    @staticmethod
    def _context(market_context: object) -> BondCurveAADMarketContext:
        if not isinstance(market_context, BondCurveAADMarketContext):
            raise TypeError("BondCurveAADAdapter requires BondCurveAADMarketContext")
        return market_context


@dataclass(frozen=True)
class VanillaEquityOptionVolAADMarketContext:
    """Market context for the bounded vanilla-equity vol-surface AAD lane."""

    market_state: object
    vol_surface_name: str = "default_vol_surface"
    currency: str | None = None
    provenance_namespace: str | None = "portfolio_aad"
    object_path: str = ""

    def coordinates(self) -> tuple[RiskFactorCoordinate, ...]:
        """Return supported vol coordinates for this context."""
        market_state = self.market_state
        vol_surface = getattr(market_state, "vol_surface", None)
        if vol_surface is None:
            raise ValueError("vanilla equity option AAD requires market_state.vol_surface")
        from trellis.models.vol_surface import FlatVol, GridVolSurface

        registry = RiskFactorRegistry()
        if isinstance(vol_surface, FlatVol):
            return registry.discover_flat_vol_surface(
                vol_surface,
                object_name=self.vol_surface_name,
                currency=self.currency,
                object_path=self.object_path,
                provenance_namespace=self.provenance_namespace,
                support_status="supported",
            )
        if isinstance(vol_surface, GridVolSurface):
            return registry.discover_grid_vol_surface(
                vol_surface,
                object_name=self.vol_surface_name,
                currency=self.currency,
                object_path=self.object_path,
                provenance_namespace=self.provenance_namespace,
                support_status="supported",
            )
        raise ValueError(
            "unsupported vol surface parameterization for vanilla equity option AAD"
        )


@dataclass(frozen=True)
class VanillaEquityOptionVolAADAdapter:
    """Concrete adapter for vanilla European equity-option risk to one surface."""

    def support_decision(
        self,
        position_name: str,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> AADSupportDecision:
        """Return whether this adapter supports a vanilla option under *market_context*."""
        try:
            context = self._context(market_context)
        except TypeError:
            return AADSupportDecision(False, "unsupported_market_context")

        market_state = context.market_state
        if getattr(market_state, "discount", None) is None:
            return AADSupportDecision(False, "discount_curve_required")
        if getattr(market_state, "vol_surface", None) is None:
            return AADSupportDecision(False, "vol_surface_required")

        from trellis.models.vol_surface import FlatVol, GridVolSurface

        vol_surface = getattr(market_state, "vol_surface", None)
        if not isinstance(vol_surface, (FlatVol, GridVolSurface)):
            return AADSupportDecision(False, "unsupported_vol_surface_parameterization")

        missing_fields = tuple(
            field_name
            for field_name in ("spot", "strike", "expiry_date")
            if not hasattr(instrument, field_name)
        )
        if missing_fields:
            return AADSupportDecision(
                False,
                "vanilla_equity_contract_fields_required",
                diagnostics=(
                    {
                        "position_name": str(position_name),
                        "missing_fields": list(missing_fields),
                    },
                ),
            )

        exercise_style = _vanilla_equity_option_exercise_style(instrument)
        if exercise_style not in {"european", "american", "bermudan"}:
            return AADSupportDecision(False, "unsupported_exercise_style")

        option_type = str(getattr(instrument, "option_type", "call")).strip().lower()
        if option_type not in {"call", "put"}:
            return AADSupportDecision(False, "unsupported_option_type")

        try:
            maturity = _vanilla_equity_option_maturity(instrument, market_state)
        except Exception as exc:
            return AADSupportDecision(
                False,
                "maturity_unavailable",
                diagnostics=(
                    {
                        "position_name": str(position_name),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ),
            )
        if maturity <= 0.0:
            return AADSupportDecision(False, "positive_maturity_required")

        if exercise_style in {"american", "bermudan"}:
            if not isinstance(vol_surface, FlatVol):
                return AADSupportDecision(
                    False,
                    "unsupported_early_exercise_vol_surface_parameterization",
                )
            base_vol = float(getattr(vol_surface, "vol"))
            if base_vol <= 0.0:
                return AADSupportDecision(False, "positive_vol_required")
            if exercise_style == "bermudan" and not _vanilla_equity_option_exercise_steps(
                instrument,
                market_state,
                maturity=maturity,
                n_steps=_vanilla_equity_option_tree_steps(instrument),
            ):
                return AADSupportDecision(False, "bermudan_exercise_schedule_required")
            try:
                _, boundary_margin = _vanilla_equity_option_early_exercise_value_from_vol(
                    instrument,
                    market_state,
                    base_vol,
                    return_boundary_margin=True,
                )
            except Exception as exc:
                return AADSupportDecision(
                    False,
                    "early_exercise_policy_unavailable",
                    diagnostics=(
                        {
                            "position_name": str(position_name),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    ),
                )
            diagnostic = _vanilla_equity_option_early_exercise_diagnostic(
                instrument,
                boundary_margin=boundary_margin,
            )
            tolerance = _vanilla_equity_option_boundary_tolerance(instrument)
            if boundary_margin is not None and boundary_margin <= tolerance:
                return AADSupportDecision(
                    False,
                    "early_exercise_boundary_kink",
                    diagnostics=(diagnostic,),
                )
            try:
                dependencies = self.factor_dependencies(instrument, context, request)
            except Exception as exc:
                return AADSupportDecision(
                    False,
                    "vol_surface_coordinate_unavailable",
                    diagnostics=(
                        {
                            "position_name": str(position_name),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    ),
                )
            return AADSupportDecision(
                True,
                "supported_vanilla_equity_early_exercise_flat_vol_aad",
                factor_dependencies=dependencies,
                diagnostics=(diagnostic,),
            )

        try:
            dependencies = self.factor_dependencies(instrument, context, request)
        except Exception as exc:
            return AADSupportDecision(
                False,
                "vol_surface_coordinate_unavailable",
                diagnostics=(
                    {
                        "position_name": str(position_name),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ),
            )
        return AADSupportDecision(
            True,
            _vanilla_equity_option_support_reason(vol_surface),
            factor_dependencies=dependencies,
        )

    def factor_dependencies(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> tuple[RiskFactorId, ...]:
        """Return shared vol-surface factors for a supported vanilla option."""
        context = self._context(market_context)
        return _sorted_unique_factors(
            coordinate.factor_id for coordinate in context.coordinates()
        )

    def value(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
    ) -> float:
        """Return the Black76-equivalent value for this adapter's market context."""
        context = self._context(market_context)
        market_state = context.market_state
        vol_surface = getattr(market_state, "vol_surface", None)
        maturity = _vanilla_equity_option_maturity(instrument, market_state)
        sigma = vol_surface.black_vol(maturity, float(getattr(instrument, "strike")))
        if _vanilla_equity_option_exercise_style(instrument) in {"american", "bermudan"}:
            return float(
                _vanilla_equity_option_early_exercise_value_from_vol(
                    instrument,
                    market_state,
                    sigma,
                )
            )
        return float(_vanilla_equity_option_value_from_vol(instrument, market_state, sigma))

    def vjp(
        self,
        instrument: object,
        market_context: object,
        request: PortfolioAADRequest,
        weight: float = 1.0,
    ) -> SparseRiskVector:
        """Return the sparse vol-surface VJP vector for one option position."""
        from trellis.core.differentiable import get_numpy, vjp
        from trellis.models.vol_surface import FlatVol, GridVolSurface

        context = self._context(market_context)
        market_state = context.market_state
        coordinates = context.coordinates()
        vol_surface = getattr(market_state, "vol_surface", None)
        if isinstance(vol_surface, FlatVol):
            base_vol = float(getattr(vol_surface, "vol"))

            def value_from_vol(vol):
                if _vanilla_equity_option_exercise_style(instrument) in {
                    "american",
                    "bermudan",
                }:
                    return _vanilla_equity_option_early_exercise_value_from_vol(
                        instrument,
                        market_state,
                        vol,
                    )
                return _vanilla_equity_option_value_from_vol(
                    instrument,
                    market_state,
                    vol,
                )

            _value, pullback = vjp(value_from_vol, base_vol)
            np = get_numpy()
            gradient = np.asarray(pullback(float(weight)), dtype=float)
            sensitivity = float(np.reshape(gradient, (-1,))[0])
            vector = SparseRiskVector.from_items(
                ((coordinates[0].factor_id, sensitivity),)
            )
            return request.filter_vector(vector)
        if isinstance(vol_surface, GridVolSurface):
            np = get_numpy()
            expiries = tuple(float(expiry) for expiry in vol_surface.expiries)
            strikes = tuple(float(strike) for strike in vol_surface.strikes)
            base_vols = np.asarray(vol_surface.vols, dtype=float)

            def value_from_nodes(vol_nodes):
                traced_surface = GridVolSurface(expiries, strikes, vol_nodes)
                maturity = _vanilla_equity_option_maturity(instrument, market_state)
                sigma = traced_surface.black_vol(
                    maturity,
                    float(getattr(instrument, "strike")),
                )
                return _vanilla_equity_option_value_from_vol(
                    instrument,
                    market_state,
                    sigma,
                )

            _value, pullback = vjp(value_from_nodes, base_vols)
            gradient = np.asarray(pullback(float(weight)), dtype=float).reshape(-1)
            vector = SparseRiskVector.from_items(
                (coordinate.factor_id, sensitivity)
                for coordinate, sensitivity in zip(coordinates, gradient)
            )
            return request.filter_vector(vector)
        raise ValueError(
            "unsupported vol surface parameterization for vanilla equity option AAD"
        )

    @staticmethod
    def _context(market_context: object) -> VanillaEquityOptionVolAADMarketContext:
        if not isinstance(market_context, VanillaEquityOptionVolAADMarketContext):
            raise TypeError(
                "VanillaEquityOptionVolAADAdapter requires "
                "VanillaEquityOptionVolAADMarketContext"
            )
        return market_context


def _vanilla_equity_option_maturity(instrument: object, market_state: object) -> float:
    from trellis.core.date_utils import year_fraction
    from trellis.core.types import DayCountConvention

    settlement = (
        getattr(market_state, "settlement", None)
        or getattr(market_state, "as_of", None)
    )
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of")
    day_count = getattr(instrument, "day_count", DayCountConvention.ACT_365)
    return float(year_fraction(settlement, getattr(instrument, "expiry_date"), day_count))


def _vanilla_equity_option_exercise_style(instrument: object) -> str:
    return str(getattr(instrument, "exercise_style", "european")).strip().lower()


def _vanilla_equity_option_support_reason(vol_surface: object) -> str:
    """Return the support reason for the active vanilla-option vol surface."""
    from trellis.models.vol_surface import GridVolSurface

    if isinstance(vol_surface, GridVolSurface):
        return "supported_vanilla_equity_grid_vol_aad"
    return "supported_vanilla_equity_flat_vol_aad"


def _vanilla_equity_option_value_from_vol(
    instrument: object,
    market_state: object,
    sigma: object,
) -> object:
    from trellis.core.differentiable import get_numpy
    from trellis.models.black import black76_call, black76_put

    np = get_numpy()
    maturity = _vanilla_equity_option_maturity(instrument, market_state)
    spot = float(getattr(instrument, "spot"))
    strike = float(getattr(instrument, "strike"))
    notional = float(getattr(instrument, "notional", 1.0))
    option_type = str(getattr(instrument, "option_type", "call")).strip().lower()
    discount = getattr(market_state, "discount", None)
    if discount is None:
        raise ValueError("vanilla equity option AAD requires market_state.discount")

    if maturity <= 0.0:
        if option_type == "put":
            return notional * np.maximum(strike - spot, 0.0)
        return notional * np.maximum(spot - strike, 0.0)

    df = discount.discount(maturity)
    forward = spot / np.maximum(df, 1e-12)
    if option_type == "put":
        return notional * df * black76_put(forward, strike, sigma, maturity)
    if option_type == "call":
        return notional * df * black76_call(forward, strike, sigma, maturity)
    raise ValueError(f"unsupported option_type: {option_type!r}")


def _vanilla_equity_option_tree_steps(instrument: object) -> int:
    raw_steps = getattr(
        instrument,
        "aad_tree_steps",
        getattr(instrument, "tree_steps", _DEFAULT_EARLY_EXERCISE_TREE_STEPS),
    )
    n_steps = int(raw_steps)
    if n_steps < 2:
        raise ValueError("early-exercise option AAD requires at least two tree steps")
    return n_steps


def _vanilla_equity_option_boundary_tolerance(instrument: object) -> float:
    raw_tolerance = getattr(
        instrument,
        "early_exercise_boundary_tolerance",
        _DEFAULT_EARLY_EXERCISE_BOUNDARY_TOLERANCE,
    )
    return max(float(raw_tolerance), 0.0)


def _vanilla_equity_option_exercise_steps(
    instrument: object,
    market_state: object,
    *,
    maturity: float,
    n_steps: int,
) -> frozenset[int]:
    style = _vanilla_equity_option_exercise_style(instrument)
    if style == "american":
        return frozenset(range(0, n_steps))
    if style != "bermudan":
        return frozenset()
    exercise_dates = tuple(getattr(instrument, "exercise_dates", ()) or ())
    if not exercise_dates:
        return frozenset()

    from trellis.core.date_utils import year_fraction
    from trellis.core.types import DayCountConvention

    settlement = (
        getattr(market_state, "settlement", None)
        or getattr(market_state, "as_of", None)
    )
    if settlement is None:
        raise ValueError("market_state must provide settlement or as_of")
    day_count = getattr(instrument, "day_count", DayCountConvention.ACT_365)
    steps: set[int] = set()
    for exercise_date in exercise_dates:
        exercise_time = float(year_fraction(settlement, exercise_date, day_count))
        if exercise_time <= 0.0 or exercise_time >= maturity:
            continue
        step = int(round(exercise_time / maturity * n_steps))
        if 0 < step < n_steps:
            steps.add(step)
    return frozenset(steps)


def _vanilla_equity_option_intrinsic_from_spots(
    instrument: object,
    spots: object,
) -> object:
    from trellis.core.differentiable import get_numpy

    np = get_numpy()
    strike = float(getattr(instrument, "strike"))
    notional = float(getattr(instrument, "notional", 1.0))
    option_type = str(getattr(instrument, "option_type", "call")).strip().lower()
    if option_type == "put":
        return notional * np.maximum(strike - spots, 0.0)
    if option_type == "call":
        return notional * np.maximum(spots - strike, 0.0)
    raise ValueError(f"unsupported option_type: {option_type!r}")


def _vanilla_equity_option_early_exercise_value_from_vol(
    instrument: object,
    market_state: object,
    sigma: object,
    *,
    return_boundary_margin: bool = False,
) -> object:
    from trellis.core.differentiable import get_numpy

    np = get_numpy()
    maturity = _vanilla_equity_option_maturity(instrument, market_state)
    if maturity <= 0.0:
        value = _vanilla_equity_option_intrinsic_from_spots(
            instrument,
            float(getattr(instrument, "spot")),
        )
        if return_boundary_margin:
            return value, None
        return value

    discount = getattr(market_state, "discount", None)
    if discount is None:
        raise ValueError("early-exercise option AAD requires market_state.discount")
    n_steps = _vanilla_equity_option_tree_steps(instrument)
    dt = maturity / float(n_steps)
    maturity_df = discount.discount(maturity)
    rate = -np.log(np.maximum(maturity_df, 1.0e-12)) / maturity
    one_step_df = np.exp(-rate * dt)
    up = np.exp(sigma * np.sqrt(dt))
    down = 1.0 / up
    growth = np.exp(rate * dt)
    probability = (growth - down) / (up - down)

    spot = float(getattr(instrument, "spot"))
    terminal_nodes = np.arange(n_steps + 1)
    terminal_spots = spot * (up ** terminal_nodes) * (
        down ** (n_steps - terminal_nodes)
    )
    values = _vanilla_equity_option_intrinsic_from_spots(instrument, terminal_spots)
    exercise_steps = _vanilla_equity_option_exercise_steps(
        instrument,
        market_state,
        maturity=maturity,
        n_steps=n_steps,
    )

    min_boundary_margin = None
    for step in range(n_steps - 1, -1, -1):
        continuation = one_step_df * (
            (1.0 - probability) * values[:-1] + probability * values[1:]
        )
        if step in exercise_steps:
            nodes = np.arange(step + 1)
            spots = spot * (up ** nodes) * (down ** (step - nodes))
            exercise = _vanilla_equity_option_intrinsic_from_spots(instrument, spots)
            if return_boundary_margin:
                spread = np.abs(continuation - exercise)
                active_scale = np.maximum(np.abs(continuation), np.abs(exercise))
                active = active_scale > 1.0e-12
                if bool(np.any(active)):
                    margin = float(np.min(spread[active]))
                    min_boundary_margin = (
                        margin
                        if min_boundary_margin is None
                        else min(min_boundary_margin, margin)
                    )
            values = np.maximum(continuation, exercise)
        else:
            values = continuation

    value = values[0]
    if return_boundary_margin:
        return value, min_boundary_margin
    return value


def _vanilla_equity_option_early_exercise_diagnostic(
    instrument: object,
    *,
    boundary_margin: float | None,
) -> dict[str, Any]:
    return {
        "code": "early_exercise_control_policy",
        "severity": "info",
        "exercise_style": _vanilla_equity_option_exercise_style(instrument),
        "derivative_policy": _EARLY_EXERCISE_AAD_POLICY,
        "tree_steps": _vanilla_equity_option_tree_steps(instrument),
        "boundary_tolerance": _vanilla_equity_option_boundary_tolerance(instrument),
        "boundary_margin": boundary_margin,
    }


@dataclass(frozen=True)
class PortfolioAADRequest:
    """Request policy for factorized portfolio AAD."""

    selected_factors: tuple[RiskFactorId, ...] = field(default_factory=tuple)
    unsupported_position_policy: str = "exclude_from_risk"
    include_unsupported_value: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_factors",
            _sorted_unique_factors(self.selected_factors),
        )
        object.__setattr__(
            self,
            "unsupported_position_policy",
            str(self.unsupported_position_policy).strip() or "exclude_from_risk",
        )
        object.__setattr__(
            self,
            "include_unsupported_value",
            bool(self.include_unsupported_value),
        )

    @property
    def selects_all_factors(self) -> bool:
        """Return whether the request leaves the full factor set selected."""
        return not self.selected_factors

    def filter_vector(self, vector: SparseRiskVector) -> SparseRiskVector:
        """Apply this request's selected-factor policy to a sparse vector."""
        if self.selects_all_factors:
            return vector
        return vector.filter(self.selected_factors)

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly request payload."""
        return {
            "selected_factors": "all"
            if self.selects_all_factors
            else [factor.to_payload() for factor in self.selected_factors],
            "unsupported_position_policy": self.unsupported_position_policy,
            "include_unsupported_value": self.include_unsupported_value,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PortfolioAADRequest:
        """Build a request from :meth:`to_payload` output."""
        selected_payload = payload.get("selected_factors", "all")
        if selected_payload == "all":
            selected_factors: tuple[RiskFactorId, ...] = ()
        else:
            selected_factors = tuple(
                RiskFactorId.from_payload(entry)
                for entry in selected_payload
            )
        return cls(
            selected_factors=selected_factors,
            unsupported_position_policy=str(
                payload.get("unsupported_position_policy", "exclude_from_risk")
            ),
            include_unsupported_value=bool(payload.get("include_unsupported_value", True)),
        )


@dataclass(frozen=True)
class UnsupportedAADPosition:
    """Typed record for a position excluded from portfolio-AAD risk."""

    position_name: str
    instrument_type: str
    reason: str
    requested_factors: tuple[RiskFactorId, ...] = field(default_factory=tuple)
    included_in_value: bool = False
    included_in_risk: bool = False
    fallback_method: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_name", str(self.position_name))
        object.__setattr__(self, "instrument_type", str(self.instrument_type))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(
            self,
            "requested_factors",
            _sorted_unique_factors(self.requested_factors),
        )
        object.__setattr__(self, "included_in_value", bool(self.included_in_value))
        object.__setattr__(self, "included_in_risk", bool(self.included_in_risk))
        if self.fallback_method is not None:
            object.__setattr__(self, "fallback_method", str(self.fallback_method))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly unsupported-position payload."""
        return {
            "position_name": self.position_name,
            "instrument_type": self.instrument_type,
            "reason": self.reason,
            "requested_factors": [
                factor.to_payload()
                for factor in self.requested_factors
            ],
            "included_in_value": self.included_in_value,
            "included_in_risk": self.included_in_risk,
            "fallback_method": self.fallback_method,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> UnsupportedAADPosition:
        """Build an unsupported-position record from :meth:`to_payload` output."""
        return cls(
            position_name=str(payload["position_name"]),
            instrument_type=str(payload["instrument_type"]),
            reason=str(payload["reason"]),
            requested_factors=tuple(
                RiskFactorId.from_payload(entry)
                for entry in payload.get("requested_factors", ())
            ),
            included_in_value=bool(payload.get("included_in_value", False)),
            included_in_risk=bool(payload.get("included_in_risk", False)),
            fallback_method=(
                None
                if payload.get("fallback_method") is None
                else str(payload.get("fallback_method"))
            ),
        )


@dataclass(frozen=True)
class PortfolioAADResult:
    """Factorized portfolio-AAD result with metadata and diagnostics."""

    portfolio_value: float | None = None
    risk_vector: SparseRiskVector = field(default_factory=SparseRiskVector)
    coordinates: tuple[RiskFactorCoordinate, ...] = field(default_factory=tuple)
    unsupported_positions: tuple[UnsupportedAADPosition, ...] = field(default_factory=tuple)
    method_metadata: Mapping[str, Any] = field(default_factory=dict, hash=False)
    diagnostics: tuple[Mapping[str, Any], ...] = field(default_factory=tuple, hash=False)

    def __post_init__(self) -> None:
        if self.portfolio_value is not None:
            object.__setattr__(self, "portfolio_value", float(self.portfolio_value))
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "unsupported_positions", tuple(self.unsupported_positions))
        object.__setattr__(self, "method_metadata", _copy_metadata(self.method_metadata))
        object.__setattr__(self, "diagnostics", _copy_diagnostics(self.diagnostics))

    @property
    def support_status(self) -> str:
        """Return the normalized derivative support status."""
        return str(
            self.method_metadata.get(
                "derivative_method_support",
                self.method_metadata.get("support_status", "unsupported"),
            )
        )

    def apply_request(self, request: PortfolioAADRequest) -> PortfolioAADResult:
        """Return this result filtered according to *request*."""
        filtered_vector = request.filter_vector(self.risk_vector)
        selected = set(filtered_vector)
        filtered_coordinates = tuple(
            coordinate
            for coordinate in self.coordinates
            if coordinate.factor_id in selected
        )
        return PortfolioAADResult(
            portfolio_value=self.portfolio_value,
            risk_vector=filtered_vector,
            coordinates=filtered_coordinates,
            unsupported_positions=self.unsupported_positions,
            method_metadata=self.method_metadata,
            diagnostics=self.diagnostics,
        )

    def missing_selected_factors(self, request: PortfolioAADRequest) -> tuple[RiskFactorId, ...]:
        """Return explicitly requested factors absent from this result."""
        if request.selects_all_factors:
            return ()
        available_factors = set(self.risk_vector)
        available_factors.update(coordinate.factor_id for coordinate in self.coordinates)
        return tuple(
            factor
            for factor in request.selected_factors
            if factor not in available_factors
        )

    def values_by_axis(self, axis_name: str) -> dict[str | float, float]:
        """Expose sparse values keyed by one factor-axis value for legacy views."""
        values: dict[str | float, float] = {}
        for factor_id, sensitivity in self.risk_vector.items():
            axes = dict(factor_id.axes)
            if axis_name not in axes:
                continue
            values[_axis_key(axes[axis_name])] = float(sensitivity)
        return dict(sorted(values.items(), key=lambda item: str(item[0])))

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly result payload."""
        return {
            "portfolio_value": self.portfolio_value,
            "risk_vector": self.risk_vector.to_payload(),
            "coordinates": [
                coordinate.to_payload()
                for coordinate in self.coordinates
            ],
            "unsupported_positions": [
                position.to_payload()
                for position in self.unsupported_positions
            ],
            "metadata": dict(self.method_metadata),
            "diagnostics": [dict(diagnostic) for diagnostic in self.diagnostics],
            "support_status": self.support_status,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> PortfolioAADResult:
        """Build a result from :meth:`to_payload` output."""
        return cls(
            portfolio_value=payload.get("portfolio_value"),
            risk_vector=SparseRiskVector.from_payload(payload.get("risk_vector") or {}),
            coordinates=tuple(
                RiskFactorCoordinate.from_payload(entry)
                for entry in payload.get("coordinates", ())
            ),
            unsupported_positions=tuple(
                UnsupportedAADPosition.from_payload(entry)
                for entry in payload.get("unsupported_positions", ())
            ),
            method_metadata=payload.get("metadata") or {},
            diagnostics=payload.get("diagnostics") or (),
        )


__all__ = [
    "AADSupportDecision",
    "BondCurveAADAdapter",
    "BondCurveAADMarketContext",
    "DefaultUnsupportedAADPolicy",
    "PortfolioAADRequest",
    "PortfolioAADResult",
    "TradeAADAdapter",
    "UnsupportedAADPosition",
    "VanillaEquityOptionVolAADAdapter",
    "VanillaEquityOptionVolAADMarketContext",
]
