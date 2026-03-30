"""Common analytical payoff transforms."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy

np = get_numpy()


def normalized_option_type(option_type: str) -> str:
    """Return a canonical call/put label or raise for unsupported option types."""
    normalized = str(option_type).strip().lower()
    if normalized in {"call", "put"}:
        return normalized
    raise ValueError(
        f"Unsupported option_type {option_type!r}; expected 'call' or 'put'"
    )


def terminal_intrinsic(option_type: str, *, spot: float, strike: float) -> float:
    """Return terminal intrinsic value for a vanilla call or put payoff."""
    normalized_type = normalized_option_type(option_type)
    if normalized_type == "call":
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


def cash_or_nothing_intrinsic(
    option_type: str,
    *,
    spot: float,
    strike: float,
    cash: float = 1.0,
) -> float:
    """Return the terminal cash-or-nothing payoff for a vanilla call or put."""
    normalized_type = normalized_option_type(option_type)
    if normalized_type == "call":
        return np.where(spot > strike, cash, 0.0)
    return np.where(spot < strike, cash, 0.0)


def asset_or_nothing_intrinsic(option_type: str, *, spot: float, strike: float) -> float:
    """Return the terminal asset-or-nothing payoff for a vanilla call or put."""
    return spot * cash_or_nothing_intrinsic(
        option_type,
        spot=spot,
        strike=strike,
        cash=1.0,
    )


def terminal_vanilla_from_basis(
    option_type: str,
    *,
    asset_value: float,
    cash_value: float,
    strike: float,
) -> float:
    """Assemble a vanilla payoff from asset-or-nothing and cash-or-nothing basis claims."""
    normalized_type = normalized_option_type(option_type)
    if normalized_type == "call":
        return asset_value - strike * cash_value
    return strike * cash_value - asset_value


def call_put_parity_gap(
    *,
    call_value: float,
    put_value: float,
    forward: float,
    strike: float,
    discount_factor: float = 1.0,
) -> float:
    """Return the discounted call-put parity residual on a forward-style surface."""
    return call_value - put_value - discount_factor * (forward - strike)
