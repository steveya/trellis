"""Compatibility shims that delegate legacy adapter specs into execution IR."""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

from trellis.core.date_utils import year_fraction
from trellis.core.types import DayCountConvention
from trellis.execution.compiler import compile_bermudan_best_of_basket_execution_ir
from trellis.execution.visitors import (
    BermudanBestOfBasketLatticeControls,
    BermudanBestOfBasketMCControls,
    BermudanBestOfBasketMCInputs,
    price_bermudan_best_of_basket_lattice,
    price_bermudan_best_of_basket_monte_carlo,
)


def price_bermudan_best_of_basket_from_compat_spec(
    market_state: Any,
    spec: Any,
    *,
    method: str = "monte_carlo",
) -> float:
    """Price a legacy rainbow adapter spec through route-free execution visitors."""
    valuation_date = _valuation_date(market_state, spec)
    expiry_date = _coerce_date(_attr(spec, "expiry_date"), "expiry_date")
    day_count = _day_count(_attr(spec, "day_count", DayCountConvention.ACT_365))
    underliers = _string_sequence(
        _attr(spec, "underliers", None) or _attr(spec, "constituents", None),
        label="underliers",
    )
    if len(underliers) != 2:
        raise ValueError("P001 compatibility shim requires exactly two underliers")

    observation_dates = _date_sequence(_attr(spec, "observation_dates", ()))
    exercise_dates = _date_sequence(_attr(spec, "exercise_dates", ())) or (expiry_date,)
    notional = float(_attr(spec, "notional", 1.0) or 1.0)
    strike = float(_attr(spec, "strike"))
    currency = str(_attr(spec, "currency", "USD") or "USD")

    inputs = BermudanBestOfBasketMCInputs(
        valuation_date=valuation_date,
        spot_values=_named_float_mapping(underliers, _attr(spec, "spots"), "spots"),
        volatilities=_named_float_mapping(
            underliers,
            _attr(spec, "vols", None) or _attr(spec, "volatilities", None),
            "vols",
        ),
        carry_rates=_named_float_mapping(
            underliers,
            _attr(spec, "dividend_yields", None)
            or _attr(spec, "carry_rates", None)
            or _attr(spec, "dividend_rates", None),
            "dividend_yields",
            default=0.0,
        ),
        correlation_matrix=_correlation_matrix(_attr(spec, "correlation"), len(underliers)),
        risk_free_rate=_risk_free_rate(market_state, spec, valuation_date, expiry_date, day_count),
        day_count=day_count,
    )
    ir = compile_bermudan_best_of_basket_execution_ir(
        semantic_id=str(_attr(spec, "semantic_id", "P001") or "P001"),
        underliers=underliers,
        strike=strike,
        expiry_date=expiry_date,
        observation_dates=observation_dates,
        exercise_dates=exercise_dates,
        notional=notional,
        currency=currency,
        source_ref="compat_agent:rainbowoption",
    )

    normalized_method = _normalized_method(method)
    if normalized_method == "lattice":
        result = price_bermudan_best_of_basket_lattice(
            ir,
            inputs,
            controls=BermudanBestOfBasketLatticeControls(
                n_steps=int(_attr(spec, "lattice_n_steps", None) or _attr(spec, "n_steps", 48) or 48),
            ),
        )
        return float(result.price)
    if normalized_method == "monte_carlo":
        result = price_bermudan_best_of_basket_monte_carlo(
            ir,
            inputs,
            controls=BermudanBestOfBasketMCControls(
                n_paths=int(_attr(spec, "n_paths", 4096) or 4096),
                n_steps=int(_attr(spec, "n_steps", 96) or 96),
                seed=None if _attr(spec, "seed", 42) is None else int(_attr(spec, "seed", 42)),
            ),
        )
        return float(result.price)
    raise ValueError(f"Unsupported semantic execution shim method {method!r}")


def _attr(spec: Any, name: str, default: Any = None) -> Any:
    return getattr(spec, name, default)


def _valuation_date(market_state: Any, spec: Any) -> date:
    value = getattr(market_state, "settlement", None) or _attr(spec, "valuation_date", None)
    return _coerce_date(value, "valuation_date")


def _coerce_date(value: Any, label: str) -> date:
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return date.fromisoformat(text)


def _date_sequence(value: Any) -> tuple[date, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    return tuple(_coerce_date(item, "date sequence item") for item in values)


def _string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    if value is None or value == "":
        raise ValueError(f"{label} is required")
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        values = [str(item).strip() for item in value]
    else:
        values = [str(value).strip()]
    result = tuple(item for item in values if item)
    if not result:
        raise ValueError(f"{label} is required")
    return result


def _float_sequence(value: Any, *, label: str) -> tuple[float, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]
    try:
        return tuple(float(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric values") from exc


def _named_float_mapping(
    names: tuple[str, ...],
    value: Any,
    label: str,
    *,
    default: float | None = None,
) -> dict[str, float]:
    values = _float_sequence(value, label=label)
    if not values and default is not None:
        return {name: float(default) for name in names}
    if len(values) != len(names):
        raise ValueError(f"{label} length must match underliers")
    return {name: float(item) for name, item in zip(names, values)}


def _correlation_matrix(value: Any, n_assets: int) -> tuple[tuple[float, ...], ...]:
    if isinstance(value, str):
        rows = [row.strip() for row in value.split(";") if row.strip()]
        matrix = tuple(
            tuple(float(cell.strip()) for cell in row.split(",") if cell.strip())
            for row in rows
        )
    elif isinstance(value, Sequence):
        matrix = tuple(tuple(float(cell) for cell in row) for row in value)
    else:
        matrix = ()
    if len(matrix) != n_assets or any(len(row) != n_assets for row in matrix):
        raise ValueError("correlation must be a square matrix aligned to underliers")
    return matrix


def _risk_free_rate(
    market_state: Any,
    spec: Any,
    valuation_date: date,
    expiry_date: date,
    day_count: DayCountConvention,
) -> float:
    explicit = _attr(spec, "risk_free_rate", None)
    if explicit not in {None, ""}:
        return float(explicit)
    domestic = _attr(spec, "domestic_rate", None)
    if domestic not in {None, ""}:
        return float(domestic)
    discount = getattr(market_state, "discount", None)
    if discount is not None and hasattr(discount, "zero_rate"):
        maturity = max(year_fraction(valuation_date, expiry_date, day_count), 1e-12)
        return float(discount.zero_rate(maturity))
    return 0.0


def _day_count(value: Any) -> DayCountConvention:
    if isinstance(value, DayCountConvention):
        return value
    text = str(value or "").strip()
    if not text:
        return DayCountConvention.ACT_365
    try:
        return DayCountConvention[text]
    except KeyError:
        return DayCountConvention(text)


def _normalized_method(method: str) -> str:
    text = str(method or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"rate_tree", "tree", "lattice", "lattices"}:
        return "lattice"
    if text in {"monte_carlo", "montecarlo", "mc"}:
        return "monte_carlo"
    return text


__all__ = [
    "price_bermudan_best_of_basket_from_compat_spec",
]
