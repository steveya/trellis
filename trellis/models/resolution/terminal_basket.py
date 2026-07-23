"""Typed market resolution for terminal multi-asset basket options."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from datetime import date
from typing import Protocol

from trellis.core.market_state import MarketState
from trellis.core.runtime_contract import wrap_market_state_with_contract
from trellis.core.types import DayCountConvention
from trellis.models.resolution.basket_semantics import (
    ResolvedBasketSemantics,
    resolve_basket_semantics,
)


class TerminalBasketSpecLike(Protocol):
    """Minimal contract surface needed to resolve a terminal basket."""

    notional: float
    underliers: str
    strike: float
    expiry_date: date
    correlation: str
    weights: str | None
    spots: str | None
    vols: str | None
    dividend_yields: str | None
    basket_style: str
    option_type: str
    day_count: DayCountConvention


@dataclass(frozen=True)
class ResolvedTerminalBasketInputs:
    """Canonical market coordinates and payoff terms for a terminal basket."""

    semantics: ResolvedBasketSemantics
    weights: tuple[float, ...]
    basket_style: str
    option_type: str
    strike: float
    comparison_target: str | None = None

    @property
    def notional_spots(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.semantics.constituent_spots)

    @property
    def vols(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.semantics.constituent_vols)

    @property
    def carry(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.semantics.constituent_carry)

    @property
    def correlation_matrix(self) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(cell) for cell in row)
            for row in self.semantics.correlation_matrix
        )


def resolve_terminal_basket_inputs(
    market_state: MarketState,
    spec: TerminalBasketSpecLike,
    *,
    comparison_target: str | None = None,
) -> ResolvedTerminalBasketInputs:
    """Resolve names, spots, vols, carry, correlation, style, and weights."""
    target = str(comparison_target or "").strip().lower() or None
    underliers = _parse_name_vector(getattr(spec, "underliers", None))
    if not underliers:
        spot_names = tuple((market_state.underlier_spots or {}).keys())
        underliers = tuple(str(item) for item in spot_names[:2])
    if len(underliers) != 2:
        raise ValueError("Terminal basket resolution requires exactly two underliers")

    correlation_source = _correlation_source_descriptor(
        getattr(spec, "correlation", None),
        n_assets=len(underliers),
    )
    market_state_for_resolution = market_state
    market_state_updates: dict[str, object] = {}
    if correlation_source is not None:
        model_parameters = dict(
            getattr(market_state, "model_parameters", None) or {}
        )
        model_parameters["correlation_source"] = correlation_source
        market_state_updates["model_parameters"] = model_parameters
    if (
        getattr(market_state, "vol_surface", None) is not None
        and getattr(market_state, "local_vol_surface", None) is not None
    ):
        market_state_updates["local_vol_surface"] = None
        market_state_updates["local_vol_surfaces"] = {}
    if market_state_updates:
        market_state_for_resolution = _replace_market_state_like(
            market_state,
            **market_state_updates,
        )

    semantics = resolve_basket_semantics(
        market_state_for_resolution,
        constituents=",".join(underliers),
        strike=float(getattr(spec, "strike", 0.0)),
        expiry_date=getattr(spec, "expiry_date"),
        option_type=getattr(spec, "option_type", "call"),
        day_count=(
            getattr(spec, "day_count", None)
            or DayCountConvention.ACT_365
        ),
    )

    spot_override = _parse_float_vector(
        getattr(spec, "spots", None),
        expected=len(underliers),
    )
    vol_override = _parse_float_vector(
        getattr(spec, "vols", None),
        expected=len(underliers),
    )
    carry_override = _parse_float_vector(
        getattr(spec, "dividend_yields", None),
        expected=len(underliers),
    )
    if spot_override is not None:
        semantics = replace(semantics, constituent_spots=spot_override)
    if vol_override is not None:
        semantics = replace(semantics, constituent_vols=vol_override)
    if carry_override is not None:
        semantics = replace(semantics, constituent_carry=carry_override)

    basket_style = _normalized_basket_style(
        getattr(spec, "basket_style", None),
        comparison_target=target,
    )
    weights = _resolve_basket_weights(
        getattr(spec, "weights", None),
        expected=len(underliers),
        basket_style=basket_style,
    )
    option_type = _normalized_option_type(
        getattr(spec, "option_type", "call")
    )
    return ResolvedTerminalBasketInputs(
        semantics=semantics,
        weights=weights,
        basket_style=basket_style,
        option_type=option_type,
        strike=float(getattr(spec, "strike", 0.0)),
        comparison_target=target,
    )


def _replace_market_state_like(market_state: object, **updates):
    """Clone a market state while preserving runtime-contract wrappers."""
    if is_dataclass(market_state):
        return replace(market_state, **updates)

    raw_market_state = getattr(market_state, "raw_market_state", None)
    if raw_market_state is not None and is_dataclass(raw_market_state):
        replaced = replace(raw_market_state, **updates)
        return wrap_market_state_with_contract(
            replaced,
            requirements=getattr(market_state, "_requirements", ()),
            context=str(getattr(market_state, "_context", "") or ""),
        )

    if hasattr(market_state, "__dict__"):
        payload = dict(vars(market_state))
        payload.update(updates)
        market_state_type = type(market_state)
        try:
            return market_state_type(**payload)
        except Exception:
            pass

    raise TypeError(
        "terminal basket resolution could not clone the market state"
    )


def _normalized_basket_style(
    value: object,
    *,
    comparison_target: str | None,
) -> str:
    # The comparison target is retained as provenance on resolved inputs, but
    # payoff semantics must come from the explicit spec/contract rather than
    # target naming conventions.
    del comparison_target
    style = str(value or "weighted_sum").strip().lower()
    aliases = {
        "best_of_two": "best_of",
        "bestof": "best_of",
        "best": "best_of",
        "worstof": "worst_of",
        "worst": "worst_of",
    }
    normalized = aliases.get(style, style)
    if normalized not in {"weighted_sum", "spread", "best_of", "worst_of"}:
        raise ValueError(
            "basket_style must be weighted_sum, spread, best_of, or worst_of"
        )
    return normalized


def _normalized_option_type(value: object) -> str:
    option_type = str(value or "call").strip().lower()
    if option_type not in {"call", "put"}:
        raise ValueError(
            f"Unsupported option_type {value!r}; expected 'call' or 'put'"
        )
    return option_type


def _resolve_basket_weights(
    value: object,
    *,
    expected: int,
    basket_style: str,
) -> tuple[float, ...]:
    parsed = _parse_float_vector(value, expected=expected)
    if parsed is not None:
        return parsed
    if basket_style == "spread":
        if expected < 2:
            raise ValueError("Spread basket resolution requires at least two underliers")
        return (1.0, -1.0) + tuple(0.0 for _ in range(expected - 2))
    return tuple(1.0 / float(expected) for _ in range(expected))


def _parse_name_vector(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(";", ",").split(",")]
        return tuple(item for item in items if item)
    if isinstance(value, (list, tuple)):
        return tuple(
            str(item).strip()
            for item in value
            if str(item).strip()
        )
    text = str(value).strip()
    return (text,) if text else ()


def _parse_float_vector(
    value: object,
    *,
    expected: int,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (int, float)):
        if expected != 1:
            return tuple(float(value) for _ in range(expected))
        return (float(value),)
    if isinstance(value, str):
        items = [
            item.strip()
            for item in value.replace(";", ",").split(",")
            if item.strip()
        ]
        parsed = tuple(float(item) for item in items)
    else:
        parsed = tuple(float(item) for item in value)
    if len(parsed) != expected:
        raise ValueError(f"Expected {expected} numeric entries, got {len(parsed)}")
    return parsed


def _correlation_source_descriptor(value: object, *, n_assets: int):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return {"kind": "explicit", "value": float(value)}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if ";" not in text and "," not in text:
            return {"kind": "explicit", "value": float(text)}
        rows = [
            [
                float(cell.strip())
                for cell in row.split(",")
                if cell.strip()
            ]
            for row in text.split(";")
            if row.strip()
        ]
        if len(rows) == 1 and len(rows[0]) == 1:
            return {"kind": "explicit", "value": rows[0][0]}
        matrix = tuple(tuple(row) for row in rows)
        if len(matrix) != n_assets or any(
            len(row) != n_assets for row in matrix
        ):
            raise ValueError(
                f"Expected a {n_assets}x{n_assets} correlation matrix, "
                f"got {len(matrix)} row(s)"
            )
        return {"kind": "explicit", "matrix": matrix}
    return value


__all__ = [
    "ResolvedTerminalBasketInputs",
    "TerminalBasketSpecLike",
    "resolve_terminal_basket_inputs",
]
