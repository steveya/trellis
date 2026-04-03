"""Shared market-resolution helpers for ranked-observation basket routes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from types import SimpleNamespace
from typing import Any, Protocol

from trellis.core.date_utils import normalize_explicit_dates, year_fraction
from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.core.types import ContractTimeline, DayCountConvention, Frequency

np = get_numpy()


class BasketSpecLike(Protocol):
    """Minimal spec surface required by the shared basket resolver."""

    strike: float
    expiry_date: date
    constituents: str
    observation_dates: ContractTimeline | tuple[date, ...] | None
    selection_rule: str
    lock_rule: str
    aggregation_rule: str
    option_type: str
    selection_count: int
    day_count: DayCountConvention
    observation_frequency: Frequency
    correlation_matrix_key: str | None


@dataclass(frozen=True)
class ResolvedBasketSemantics:
    """Normalized market inputs consumed by basket pricing routes."""

    constituent_names: tuple[str, ...]
    constituent_spots: tuple[float, ...]
    constituent_vols: tuple[float, ...]
    constituent_carry: tuple[float, ...]
    correlation_matrix: tuple[tuple[float, ...], ...]
    observation_dates: tuple[date, ...]
    observation_times: tuple[float, ...]
    valuation_date: date
    T: float
    domestic_df: float
    selection_rule: str
    lock_rule: str
    aggregation_rule: str
    selection_count: int
    correlation_preflight: "CorrelationPreflightReport | None" = None

    _ALIASES = {
        "constituents": "constituent_names",
        "spots": "constituent_spots",
        "vols": "constituent_vols",
        "carries": "constituent_carry",
        "corr": "correlation_matrix",
        "correlation": "correlation_matrix",
        "time_to_expiry": "T",
        "domestic_discount_factor": "domestic_df",
        "observation_schedule": "observation_dates",
        "observation_year_fractions": "observation_times",
        "as_of_date": "valuation_date",
        "settlement_date": "valuation_date",
    }

    @property
    def constituent_spot_vector(self):
        """Return the constituent spot vector under a friendly alias."""
        return self.constituent_spots

    @property
    def constituent_vol_vector(self):
        """Return the constituent vol vector under a friendly alias."""
        return self.constituent_vols

    @property
    def constituent_carry_vector(self):
        """Return the constituent carry vector under a friendly alias."""
        return self.constituent_carry

    @property
    def spots(self):
        """Backward-compatible alias for generated code that expects basket spots."""
        return self.constituent_spots

    @property
    def vols(self):
        """Backward-compatible alias for generated code that expects basket vols."""
        return self.constituent_vols

    @property
    def carries(self):
        """Backward-compatible alias for generated code that expects basket carries."""
        return self.constituent_carry

    @property
    def underlier_spots(self):
        """Backward-compatible alias for generated code that expects underlier spots."""
        return self.constituent_spots

    @property
    def time_to_expiry(self) -> float:
        """Backward-compatible alias for the expiry horizon."""
        return self.T

    @property
    def domestic_discount_factor(self) -> float:
        """Backward-compatible alias for the domestic discount factor."""
        return self.domestic_df

    @property
    def as_of_date(self) -> date:
        """Backward-compatible alias for the valuation date."""
        return self.valuation_date

    @property
    def settlement_date(self) -> date:
        """Backward-compatible alias for the valuation date."""
        return self.valuation_date

    @property
    def correlation(self):
        """Return the correlation matrix under a short alias."""
        return self.correlation_matrix

    @property
    def risk_free_rates(self):
        """Backward-compatible alias for generated code that expects per-asset rates."""
        if self.T <= 0.0:
            rate = 0.0
        else:
            rate = float(-np.log(self.domestic_df) / self.T)
        return tuple(rate for _ in self.constituent_spots)

    @property
    def div_yields(self):
        """Backward-compatible alias for generated code that expects dividend yields."""
        return self.constituent_carry

    @property
    def rates(self):
        """Backward-compatible alias for generated code that expects basket rates."""
        return self.risk_free_rates

    @property
    def dividends(self):
        """Backward-compatible alias for generated code that expects dividend yields."""
        return self.div_yields

    def __getitem__(self, key: str) -> Any:
        field_name = self._ALIASES.get(key, key)
        if not hasattr(self, field_name):
            raise KeyError(key)
        return getattr(self, field_name)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


@dataclass(frozen=True)
class CorrelationPreflightReport:
    """Traceable summary of basket-correlation validation and repair."""

    source_key: str | None
    source_kind: str
    requested_assets: int
    correlation_status: str
    was_regularized: bool
    regularization_floor: float
    max_asymmetry: float
    max_diagonal_deviation: float
    min_eigenvalue_before: float
    min_eigenvalue_after: float
    source_family: str = ""
    source_estimator: str | None = None
    sample_size: int | None = None
    source_seed: int | None = None
    source_parameters: dict[str, object] = field(default_factory=dict)


class CorrelationPreflightError(ValueError):
    """Raised when basket correlation inputs fail deterministic preflight."""

    def __init__(self, message: str, *, report: CorrelationPreflightReport) -> None:
        super().__init__(message)
        self.report = report


def resolve_basket_semantics(
    market_state: MarketState,
    spec: BasketSpecLike | None = None,
    **kwargs,
) -> ResolvedBasketSemantics:
    """Resolve the deterministic market inputs needed by basket routes."""
    if kwargs:
        spec = _coerce_basket_spec(spec, **kwargs)
    if spec is None:
        raise TypeError("resolve_basket_semantics() missing required spec or basket fields")
    if market_state.discount is None:
        raise ValueError("market_state.discount is required for basket pricing")
    if market_state.vol_surface is None and market_state.local_vol_surface is None:
        raise ValueError("market_state.vol_surface is required for basket pricing")

    constituent_names = _parse_constituents(spec.constituents)
    if not constituent_names:
        raise ValueError("Basket pricing requires at least one constituent name")

    observation_dates = _resolve_observation_dates(market_state, spec)
    T = year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)
    domestic_df = 1.0 if T <= 0.0 else float(market_state.discount.discount(T))

    spots = tuple(
        _resolve_constituent_spot(market_state, constituent)
        for constituent in constituent_names
    )

    if T <= 0.0:
        vols = tuple(0.0 for _ in constituent_names)
        carry = tuple(0.0 for _ in constituent_names)
        corr = _identity_correlation_matrix(len(constituent_names))
        return ResolvedBasketSemantics(
            constituent_names=constituent_names,
            constituent_spots=spots,
            constituent_vols=vols,
            constituent_carry=carry,
            correlation_matrix=corr,
            observation_dates=observation_dates,
            observation_times=tuple(0.0 for _ in observation_dates),
            valuation_date=market_state.settlement,
            T=0.0,
            domestic_df=domestic_df,
            selection_rule=_normalize_rule(spec.selection_rule, "best_of_remaining"),
            lock_rule=_normalize_rule(spec.lock_rule, "remove_selected"),
            aggregation_rule=_normalize_rule(spec.aggregation_rule, "average_locked_returns"),
            selection_count=max(int(spec.selection_count or 1), 1),
            correlation_preflight=CorrelationPreflightReport(
                source_key=None,
                source_kind="identity_default",
                requested_assets=len(constituent_names),
                correlation_status="accepted",
                was_regularized=False,
                regularization_floor=0.0,
                max_asymmetry=0.0,
                max_diagonal_deviation=0.0,
                min_eigenvalue_before=1.0,
                min_eigenvalue_after=1.0,
                source_family="identity",
            ),
        )

    vols = tuple(
        _resolve_constituent_vol(market_state, constituent, spot, T)
        for constituent, spot in zip(constituent_names, spots, strict=True)
    )
    carry = tuple(
        _resolve_constituent_carry(market_state, constituent, T)
        for constituent in constituent_names
    )
    corr, correlation_preflight = _resolve_correlation_matrix(
        market_state,
        constituent_names,
        spec.correlation_matrix_key,
    )
    return ResolvedBasketSemantics(
        constituent_names=constituent_names,
        constituent_spots=spots,
        constituent_vols=vols,
        constituent_carry=carry,
        correlation_matrix=corr,
        observation_dates=observation_dates,
        observation_times=tuple(
            _resolve_observation_time(market_state.settlement, obs_date, spec.day_count)
            for obs_date in observation_dates
            if obs_date > market_state.settlement
        ),
        valuation_date=market_state.settlement,
        T=T,
        domestic_df=domestic_df,
        selection_rule=_normalize_rule(spec.selection_rule, "best_of_remaining"),
        lock_rule=_normalize_rule(spec.lock_rule, "remove_selected"),
        aggregation_rule=_normalize_rule(spec.aggregation_rule, "average_locked_returns"),
        selection_count=max(int(spec.selection_count or 1), 1),
        correlation_preflight=correlation_preflight,
    )


def _coerce_basket_spec(spec: BasketSpecLike | None, **kwargs) -> BasketSpecLike:
    """Build a lightweight spec adapter from common generated-field aliases."""
    attrs: dict[str, Any] = {}
    if spec is not None:
        for name in (
            "constituents",
            "observation_dates",
            "selection_rule",
            "lock_rule",
            "aggregation_rule",
            "option_type",
            "selection_count",
            "day_count",
            "observation_frequency",
            "correlation_matrix_key",
            "strike",
            "expiry_date",
        ):
            if hasattr(spec, name):
                attrs[name] = getattr(spec, name)

    attrs.update(kwargs)
    if "constituents" not in attrs and "underlyings" in attrs:
        attrs["constituents"] = attrs["underlyings"]
    if "correlation_matrix_key" not in attrs and "correlation_source" in attrs:
        attrs["correlation_matrix_key"] = attrs["correlation_source"]

    return SimpleNamespace(
        strike=float(attrs.get("strike", 0.0)),
        expiry_date=attrs.get("expiry_date"),
        constituents=attrs.get("constituents", ""),
        observation_dates=attrs.get("observation_dates"),
        selection_rule=attrs.get("selection_rule", "best_of_remaining"),
        lock_rule=attrs.get("lock_rule", "remove_selected"),
        aggregation_rule=attrs.get("aggregation_rule", "average_locked_returns"),
        option_type=attrs.get("option_type", "call"),
        selection_count=int(attrs.get("selection_count", 1) or 1),
        day_count=attrs.get("day_count", DayCountConvention.ACT_365),
        observation_frequency=attrs.get("observation_frequency", Frequency.QUARTERLY),
        correlation_matrix_key=attrs.get("correlation_matrix_key"),
    )


def _parse_constituents(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = value.replace(";", ",").split(",")
        return tuple(item.strip() for item in items if item.strip())
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                items.append(text)
        return tuple(dict.fromkeys(items))
    return (str(value).strip(),)


def _resolve_observation_dates(
    market_state: MarketState,
    spec: BasketSpecLike,
) -> tuple[date, ...]:
    if spec.observation_dates:
        return normalize_explicit_dates(spec.observation_dates)

    from trellis.core.date_utils import generate_schedule

    schedule = generate_schedule(
        market_state.settlement,
        spec.expiry_date,
        spec.observation_frequency,
    )
    return tuple(schedule)


def _resolve_observation_time(
    settlement: date,
    observation_date: date,
    day_count: DayCountConvention,
) -> float:
    return max(float(year_fraction(settlement, observation_date, day_count)), 0.0)


def _resolve_constituent_spot(market_state: MarketState, constituent: str) -> float:
    spots = market_state.underlier_spots or {}
    for key in (
        constituent,
        constituent.upper(),
        constituent.lower(),
        constituent.replace(" ", "_"),
    ):
        if key in spots:
            return float(spots[key])
    if market_state.spot is not None:
        return float(market_state.spot)
    raise ValueError(
        f"Basket pricing requires market_state.underlier_spots[{constituent!r}] or a fallback spot"
    )


def _resolve_constituent_vol(
    market_state: MarketState,
    constituent: str,
    spot: float,
    T: float,
) -> float:
    if market_state.local_vol_surface is not None and callable(market_state.local_vol_surface):
        return float(market_state.local_vol_surface(spot, T))
    if market_state.vol_surface is not None:
        return float(market_state.vol_surface.black_vol(T, spot))
    raise ValueError("Basket pricing requires a volatility surface")


def _resolve_constituent_carry(
    market_state: MarketState,
    constituent: str,
    T: float,
) -> float:
    forecast_curves = market_state.forecast_curves or {}
    for key in (
        constituent,
        constituent.upper(),
        constituent.lower(),
        f"{constituent}-DISC",
        f"{constituent}_DISC",
        f"{constituent}-CARRY",
        f"{constituent}_CARRY",
    ):
        if key in forecast_curves:
            curve = forecast_curves[key]
            if T <= 0.0:
                return 0.0
            return float(curve.forward_rate(0.0, T, compounding="continuous"))
    if market_state.forward_curve is not None and T > 0.0:
        try:
            return float(market_state.forward_curve.forward_rate(0.0, T, compounding="continuous"))
        except Exception:
            return 0.0
    return 0.0


def _resolve_correlation_matrix(
    market_state: MarketState,
    constituent_names: tuple[str, ...],
    key: str | None,
) -> tuple[tuple[tuple[float, ...], ...], CorrelationPreflightReport]:
    params = market_state.model_parameters or {}
    n_assets = len(constituent_names)

    descriptor = params.get("correlation_source")
    if descriptor is not None:
        return _resolve_correlation_from_descriptor(
            market_state,
            constituent_names,
            descriptor,
            source_key=key,
        )

    candidate_keys = [
        key,
        "correlation_matrix",
        "corr_matrix",
        "basket_correlation",
        "correlation",
        "rho",
    ]
    for candidate in candidate_keys:
        if candidate is None or candidate not in params:
            continue
        value = params[candidate]
        if isinstance(value, (int, float)):
            matrix = _constant_correlation_matrix(n_assets, float(value))
            return _preflight_correlation_matrix(
                matrix,
                n_assets,
                source_key=candidate,
                source_kind="explicit_scalar",
                source_family="explicit",
                source_parameters={"value": float(value)},
            )
        return _preflight_correlation_matrix(
            value,
            n_assets,
            source_key=candidate,
            source_kind="explicit_matrix",
            source_family="explicit",
            source_parameters={"shape": tuple(np.asarray(value, dtype=float).shape)},
        )

    empirical_value, empirical_meta = _resolve_empirical_correlation_matrix(
        params,
        constituent_names,
    )
    if empirical_value is not None:
        return _preflight_correlation_matrix(
            empirical_value,
            n_assets,
            source_key=empirical_meta.get("source_key"),
            source_kind="empirical_observations",
            source_family="empirical",
            source_estimator=str(empirical_meta.get("source_estimator") or "sample_pearson"),
            sample_size=empirical_meta.get("sample_size"),
            source_seed=empirical_meta.get("source_seed"),
            source_parameters=empirical_meta.get("source_parameters"),
        )

    implied_value, implied_meta = _resolve_implied_correlation_matrix(params)
    if implied_value is not None:
        return _preflight_correlation_matrix(
            implied_value,
            n_assets,
            source_key=implied_meta.get("source_key"),
            source_kind=implied_meta.get("source_kind", "implied_matrix"),
            source_family="implied",
            source_estimator=str(implied_meta.get("source_estimator") or "implied_surface"),
            sample_size=implied_meta.get("sample_size"),
            source_seed=implied_meta.get("source_seed"),
            source_parameters=implied_meta.get("source_parameters"),
        )

    synthetic_value, synthetic_meta = _resolve_synthetic_correlation_matrix(
        market_state,
        n_assets,
    )
    if synthetic_value is not None:
        return _preflight_correlation_matrix(
            synthetic_value,
            n_assets,
            source_key=synthetic_meta.get("source_key"),
            source_kind=synthetic_meta.get("source_kind", "identity_default"),
            source_family=str(synthetic_meta.get("source_family") or "synthetic"),
            source_estimator=synthetic_meta.get("source_estimator"),
            sample_size=synthetic_meta.get("sample_size"),
            source_seed=synthetic_meta.get("source_seed"),
            source_parameters=synthetic_meta.get("source_parameters"),
        )

    matrix = _identity_correlation_matrix(n_assets)
    report = CorrelationPreflightReport(
        source_key=None,
        source_kind="identity_default",
        requested_assets=n_assets,
        correlation_status="accepted",
        was_regularized=False,
        regularization_floor=0.0,
        max_asymmetry=0.0,
        max_diagonal_deviation=0.0,
        min_eigenvalue_before=1.0,
        min_eigenvalue_after=1.0,
        source_family="identity",
    )
    return matrix, report


def _resolve_correlation_from_descriptor(
    market_state: MarketState,
    constituent_names: tuple[str, ...],
    descriptor: Any,
    *,
    source_key: str | None,
) -> tuple[tuple[tuple[float, ...], ...], CorrelationPreflightReport]:
    """Resolve a correlation source descriptor with explicit provenance."""
    n_assets = len(constituent_names)
    if isinstance(descriptor, str):
        payload: dict[str, Any] = {"kind": descriptor}
    elif isinstance(descriptor, dict):
        payload = dict(descriptor)
    else:
        raise ValueError("correlation_source must be a mapping or string")

    kind = _normalize_source_kind(
        str(payload.pop("kind", payload.pop("source_kind", "")) or "").strip()
    )
    source_ref = payload.pop("source_ref", None)
    source_seed = payload.pop("seed", payload.pop("prior_seed", None))
    source_estimator = payload.pop("estimator", None)
    source_parameters = dict(payload.pop("parameters", {}) or {})
    if source_ref is not None:
        source_parameters.setdefault("source_ref", source_ref)
    if source_seed is not None:
        source_parameters.setdefault("seed", int(source_seed))

    if kind == "explicit":
        value = _descriptor_matrix_or_scalar(payload, explicit_keys=("matrix", "correlation_matrix", "value", "rho"))
        if value is None:
            raise ValueError("explicit correlation_source requires a matrix or scalar value")
        source_kind = "explicit_scalar" if isinstance(value, (int, float)) else "explicit_matrix"
        source_parameters.setdefault(
            "shape",
            tuple(np.asarray(value, dtype=float).shape) if not isinstance(value, (int, float)) else (n_assets, n_assets),
        )
        return _preflight_correlation_matrix(
            _constant_correlation_matrix(n_assets, float(value))
            if isinstance(value, (int, float))
            else value,
            n_assets,
            source_key=source_key or source_ref,
            source_kind=source_kind,
            source_family="explicit",
            source_estimator=None,
            source_seed=source_seed if source_seed is None else int(source_seed),
            source_parameters=source_parameters,
        )

    if kind in {"empirical", "estimated"}:
        observations = _descriptor_observations(
            payload,
            constituent_names,
        )
        if observations is None:
            value = _descriptor_matrix_or_scalar(payload, explicit_keys=("matrix", "correlation_matrix"))
            if value is None:
                raise ValueError(
                    "empirical correlation_source requires observations or a precomputed matrix"
                )
            return _preflight_correlation_matrix(
                value,
                n_assets,
                source_key=source_key or source_ref,
                source_kind="empirical_matrix",
                source_family="empirical",
                source_estimator=str(source_estimator or "precomputed_empirical"),
                source_seed=source_seed if source_seed is None else int(source_seed),
                source_parameters=source_parameters,
            )
        matrix, sample_size, observation_meta = _estimate_empirical_correlation_matrix(
            observations,
            constituent_names,
        )
        source_parameters.update(observation_meta)
        return _preflight_correlation_matrix(
            matrix,
            n_assets,
            source_key=source_key or source_ref,
            source_kind="empirical_observations",
            source_family="empirical",
            source_estimator=str(source_estimator or "sample_pearson"),
            sample_size=sample_size,
            source_seed=source_seed if source_seed is None else int(source_seed),
            source_parameters=source_parameters,
        )

    if kind == "implied":
        value = _descriptor_matrix_or_scalar(
            payload,
            explicit_keys=("matrix", "correlation_matrix", "value", "rho"),
        )
        if value is None:
            raise ValueError("implied correlation_source requires a matrix or scalar value")
        source_kind = "implied_scalar" if isinstance(value, (int, float)) else "implied_matrix"
        source_parameters.setdefault(
            "shape",
            tuple(np.asarray(value, dtype=float).shape) if not isinstance(value, (int, float)) else (n_assets, n_assets),
        )
        return _preflight_correlation_matrix(
            _constant_correlation_matrix(n_assets, float(value))
            if isinstance(value, (int, float))
            else value,
            n_assets,
            source_key=source_key or source_ref,
            source_kind=source_kind,
            source_family="implied",
            source_estimator=str(source_estimator or "implied_surface"),
            source_seed=source_seed if source_seed is None else int(source_seed),
            source_parameters=source_parameters,
        )

    if kind == "synthetic":
        value = _descriptor_matrix_or_scalar(
            payload,
            explicit_keys=("matrix", "correlation_matrix", "value", "rho"),
        )
        if value is None:
            raise ValueError("synthetic correlation_source requires a matrix or scalar value")
        source_kind = "synthetic_scalar" if isinstance(value, (int, float)) else "synthetic_matrix"
        source_parameters.setdefault(
            "shape",
            tuple(np.asarray(value, dtype=float).shape) if not isinstance(value, (int, float)) else (n_assets, n_assets),
        )
        return _preflight_correlation_matrix(
            _constant_correlation_matrix(n_assets, float(value))
            if isinstance(value, (int, float))
            else value,
            n_assets,
            source_key=source_key or source_ref,
            source_kind=source_kind,
            source_family="synthetic",
            source_estimator=str(source_estimator or "synthetic_prior"),
            source_seed=source_seed if source_seed is None else int(source_seed),
            source_parameters=source_parameters,
        )

    raise ValueError(f"Unsupported correlation_source kind {kind!r}")


def _resolve_empirical_correlation_matrix(
    params: dict[str, object],
    constituent_names: tuple[str, ...],
) -> tuple[tuple[tuple[float, ...], ...] | None, dict[str, object]]:
    """Resolve an empirical correlation matrix from historical observations."""
    for key in ("correlation_observations", "historical_returns", "path_returns", "observed_paths"):
        if key not in params:
            continue
        observations = params[key]
        matrix, sample_size, observation_meta = _estimate_empirical_correlation_matrix(
            observations,
            constituent_names,
        )
        meta = {
            "source_key": key,
            "source_estimator": "sample_pearson",
            "sample_size": sample_size,
            "source_parameters": observation_meta,
        }
        return matrix, meta
    return None, {}


def _resolve_implied_correlation_matrix(
    params: dict[str, object],
) -> tuple[tuple[tuple[float, ...], ...] | None, dict[str, object]]:
    """Resolve an implied correlation matrix or scalar from model parameters."""
    for key in ("implied_correlation_matrix", "implied_correlation", "option_surface_correlation"):
        if key not in params:
            continue
        value = params[key]
        source_kind = "implied_scalar" if isinstance(value, (int, float)) else "implied_matrix"
        source_parameters = {"source_key": key}
        if not isinstance(value, (int, float)):
            source_parameters["shape"] = tuple(np.asarray(value, dtype=float).shape)
        else:
            source_parameters["value"] = float(value)
        return value, {
            "source_key": key,
            "source_kind": source_kind,
            "source_estimator": "implied_surface",
            "source_parameters": source_parameters,
        }
    return None, {}


def _resolve_synthetic_correlation_matrix(
    market_state: MarketState,
    n_assets: int,
) -> tuple[tuple[tuple[float, ...], ...] | None, dict[str, object]]:
    """Fall back to a synthetic prior when the market snapshot is synthetic."""
    provenance = dict(getattr(market_state, "market_provenance", None) or {})
    source_kind = str(provenance.get("source_kind") or "")
    if source_kind not in {"synthetic_snapshot", "user_supplied_snapshot"}:
        return None, {}

    source_parameters = dict(provenance.get("prior_parameters") or {})
    for key in ("source_ref", "source", "prior_family"):
        if provenance.get(key) is not None:
            source_parameters.setdefault(key, provenance[key])
    return _identity_correlation_matrix(n_assets), {
        "source_key": provenance.get("source_ref"),
        "source_kind": "identity_default",
        "source_family": "synthetic",
        "source_seed": provenance.get("prior_seed"),
        "source_parameters": source_parameters,
    }


def _descriptor_matrix_or_scalar(
    payload: dict[str, Any],
    *,
    explicit_keys: tuple[str, ...],
) -> Any | None:
    for key in explicit_keys:
        if key in payload:
            return payload[key]
    return None


def _descriptor_observations(
    payload: dict[str, Any],
    constituent_names: tuple[str, ...],
) -> Any | None:
    for key in ("observations", "historical_returns", "path_returns", "samples"):
        if key in payload:
            return payload[key]
    if all(name in payload for name in constituent_names):
        return {name: payload[name] for name in constituent_names}
    return None


def _normalize_source_kind(kind: str) -> str:
    normalized = kind.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", "explicit", "explicit_matrix", "explicit_scalar", "calibrated", "quoted", "bootstrapped"}:
        return "explicit"
    if normalized in {"estimated", "empirical", "empirical_observations", "empirical_matrix"}:
        return "empirical"
    if normalized in {"implied", "implied_matrix", "implied_scalar"}:
        return "implied"
    if normalized in {"synthetic", "synthetic_matrix", "synthetic_scalar", "sampled"}:
        return "synthetic"
    raise ValueError(f"Unsupported correlation_source kind {kind!r}")


def _source_family_from_kind(kind: str) -> str:
    """Map a detailed correlation kind onto its broad source family."""
    normalized = str(kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized.startswith("empirical") or normalized.startswith("estimated"):
        return "empirical"
    if normalized.startswith("implied"):
        return "implied"
    if normalized.startswith("synthetic") or normalized == "sampled":
        return "synthetic"
    if normalized.startswith("identity"):
        return "identity"
    return "explicit"


def _estimate_empirical_correlation_matrix(
    observations: Any,
    constituent_names: tuple[str, ...],
) -> tuple[tuple[tuple[float, ...], ...], int, dict[str, object]]:
    """Estimate a correlation matrix from historical return observations."""
    n_assets = len(constituent_names)

    if isinstance(observations, dict):
        series: list[np.ndarray] = []
        sample_size: int | None = None
        for name in constituent_names:
            key_candidates = (
                name,
                name.upper(),
                name.lower(),
                name.replace(" ", "_"),
            )
            series_values = None
            for key in key_candidates:
                if key in observations:
                    series_values = observations[key]
                    break
            if series_values is None:
                raise ValueError(
                    f"Empirical correlation observations are missing a series for {name!r}"
                )
            arr = np.asarray(series_values, dtype=float).reshape(-1)
            if arr.size < 2:
                raise ValueError(
                    f"Empirical correlation observations for {name!r} need at least two samples"
                )
            if sample_size is None:
                sample_size = int(arr.size)
            elif sample_size != int(arr.size):
                raise ValueError("Empirical correlation observations must share a common sample size")
            series.append(arr)
        data = np.column_stack(series)
    else:
        data = np.asarray(observations, dtype=float)
        if data.ndim != 2:
            raise ValueError("Empirical correlation observations must be two-dimensional")
        if data.shape[1] == n_assets:
            pass
        elif data.shape[0] == n_assets:
            data = data.T
        else:
            raise ValueError(
                f"Empirical correlation observations must align with {n_assets} assets; got {data.shape}"
            )
        if data.shape[0] < 2:
            raise ValueError("Empirical correlation observations need at least two samples")
        sample_size = int(data.shape[0])

    if n_assets == 1:
        matrix = np.array([[1.0]], dtype=float)
    else:
        matrix = np.corrcoef(data, rowvar=False)
    observation_meta = {
        "observation_shape": tuple(int(value) for value in data.shape),
        "asset_names": constituent_names,
    }
    return (
        tuple(tuple(float(cell) for cell in row) for row in matrix),
        int(data.shape[0]),
        observation_meta,
    )


def _rejected_correlation_report(
    *,
    n_assets: int,
    source_key: str | None,
    source_kind: str,
    source_family: str = "",
    source_estimator: str | None = None,
    sample_size: int | None = None,
    source_seed: int | None = None,
    source_parameters: dict[str, object] | None = None,
    max_asymmetry: float = 0.0,
    max_diagonal_deviation: float = 0.0,
) -> CorrelationPreflightReport:
    """Build a deterministic rejected preflight report for malformed inputs."""
    return CorrelationPreflightReport(
        source_key=source_key,
        source_kind=source_kind,
        requested_assets=n_assets,
        correlation_status="rejected",
        was_regularized=False,
        regularization_floor=0.0,
        max_asymmetry=max_asymmetry,
        max_diagonal_deviation=max_diagonal_deviation,
        min_eigenvalue_before=float("nan"),
        min_eigenvalue_after=float("nan"),
        source_family=source_family or _source_family_from_kind(source_kind),
        source_estimator=source_estimator,
        sample_size=sample_size,
        source_seed=source_seed,
        source_parameters=dict(source_parameters or {}),
    )


def _preflight_correlation_matrix(
    value: Any,
    n_assets: int,
    *,
    source_key: str | None,
    source_kind: str,
    source_family: str = "",
    source_estimator: str | None = None,
    sample_size: int | None = None,
    source_seed: int | None = None,
    source_parameters: dict[str, object] | None = None,
) -> tuple[tuple[tuple[float, ...], ...], CorrelationPreflightReport]:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        report = _rejected_correlation_report(
            n_assets=n_assets,
            source_key=source_key,
            source_kind=source_kind,
            source_family=source_family,
            source_estimator=source_estimator,
            sample_size=sample_size,
            source_seed=source_seed,
            source_parameters=source_parameters,
        )
        raise CorrelationPreflightError(
            f"Basket correlation matrix must be two-dimensional; got shape {matrix.shape}",
            report=report,
        )
    if matrix.shape != (n_assets, n_assets):
        report = _rejected_correlation_report(
            n_assets=n_assets,
            source_key=source_key,
            source_kind=source_kind,
            source_family=source_family,
            source_estimator=source_estimator,
            sample_size=sample_size,
            source_seed=source_seed,
            source_parameters=source_parameters,
        )
        raise CorrelationPreflightError(
            f"Basket correlation matrix must have shape ({n_assets}, {n_assets}); got {matrix.shape}",
            report=report,
        )
    if not np.all(np.isfinite(matrix)):
        report = _rejected_correlation_report(
            n_assets=n_assets,
            source_key=source_key,
            source_kind=source_kind,
            source_family=source_family,
            source_estimator=source_estimator,
            sample_size=sample_size,
            source_seed=source_seed,
            source_parameters=source_parameters,
        )
        raise CorrelationPreflightError(
            "Basket correlation matrix must contain only finite values",
            report=report,
        )

    max_asymmetry = float(np.max(np.abs(matrix - matrix.T))) if n_assets else 0.0
    max_diagonal_deviation = (
        float(np.max(np.abs(np.diag(matrix) - 1.0))) if n_assets else 0.0
    )

    symmetric = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(symmetric, 1.0)
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    min_before = float(np.min(eigvals)) if eigvals.size else 1.0

    was_regularized = min_before <= 1e-12
    regularization_floor = 1e-12
    if was_regularized:
        symmetric = _nearest_pd_correlation_matrix(
            eigvals,
            eigvecs,
            floor=regularization_floor,
        )

    min_after = float(np.min(np.linalg.eigvalsh(symmetric))) if n_assets else 1.0
    report = CorrelationPreflightReport(
        source_key=source_key,
        source_kind=source_kind,
        requested_assets=n_assets,
        correlation_status="accepted" if not was_regularized else "regularized",
        was_regularized=was_regularized,
        regularization_floor=regularization_floor if was_regularized else 0.0,
        max_asymmetry=max_asymmetry,
        max_diagonal_deviation=max_diagonal_deviation,
        min_eigenvalue_before=min_before,
        min_eigenvalue_after=min_after,
        source_family=source_family or _source_family_from_kind(source_kind),
        source_estimator=source_estimator,
        sample_size=sample_size,
        source_seed=source_seed,
        source_parameters=dict(source_parameters or {}),
    )
    return (
        tuple(tuple(float(cell) for cell in row) for row in symmetric),
        report,
    )


def _nearest_pd_correlation_matrix(
    eigvals,
    eigvecs,
    *,
    floor: float,
):
    clipped = np.maximum(eigvals, floor)
    repaired = eigvecs @ np.diag(clipped) @ eigvecs.T
    repaired = 0.5 * (repaired + repaired.T)
    scale = np.sqrt(np.clip(np.diag(repaired), floor, None))
    repaired = repaired / np.outer(scale, scale)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _constant_correlation_matrix(n_assets: int, rho: float) -> tuple[tuple[float, ...], ...]:
    clipped = float(np.clip(rho, -0.999, 0.999))
    matrix = np.full((n_assets, n_assets), clipped, dtype=float)
    np.fill_diagonal(matrix, 1.0)
    return tuple(tuple(float(cell) for cell in row) for row in matrix)


def _identity_correlation_matrix(n_assets: int) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(1.0 if i == j else 0.0 for j in range(n_assets))
        for i in range(n_assets)
    )


def _normalize_rule(value: str | None, default: str) -> str:
    rule = (value or default).strip().lower().replace("-", "_").replace(" ", "_")
    return rule or default


__all__ = [
    "BasketSpecLike",
    "ResolvedBasketSemantics",
    "resolve_basket_semantics",
]
