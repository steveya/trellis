"""Bounded credit-index spread option pricing helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as raw_np

from trellis.core.differentiable import get_numpy
from trellis.models.black import black76_call, black76_put

np = get_numpy()


@dataclass(frozen=True)
class CreditIndexOptionSpec:
    """Contract inputs for a bounded credit-index option on quoted spread."""

    notional: float = 10_000_000.0
    forward_spread: float = 0.0125
    strike_spread: float = 0.0100
    spread_volatility: float = 0.30
    maturity_years: float = 1.25
    index_annuity: float = 4.2
    discount_rate: float = 0.04
    recovery_rate: float = 0.40
    option_type: str = "call"
    loss_convention: str = "spread_annuity"

    def __post_init__(self) -> None:
        fields = {
            "notional": float(self.notional),
            "forward_spread": _normalize_spread_quote(self.forward_spread),
            "strike_spread": _normalize_spread_quote(self.strike_spread),
            "spread_volatility": float(self.spread_volatility),
            "maturity_years": float(self.maturity_years),
            "index_annuity": float(self.index_annuity),
            "discount_rate": float(self.discount_rate),
            "recovery_rate": float(self.recovery_rate),
            "option_type": str(self.option_type or "").strip().lower(),
            "loss_convention": str(self.loss_convention or "").strip().lower(),
        }
        for name in (
            "notional",
            "forward_spread",
            "strike_spread",
            "spread_volatility",
            "maturity_years",
            "index_annuity",
        ):
            if fields[name] < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 <= fields["recovery_rate"] <= 1.0:
            raise ValueError("recovery_rate must satisfy 0 <= recovery_rate <= 1")
        if fields["option_type"] not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        if fields["loss_convention"] not in {"spread_annuity", "loss_given_default"}:
            raise ValueError(
                "loss_convention must be 'spread_annuity' or 'loss_given_default'"
            )
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    @property
    def loss_multiplier(self) -> float:
        """Return the notional multiplier implied by the loss convention."""
        if self.loss_convention == "loss_given_default":
            return 1.0 - float(self.recovery_rate)
        return 1.0


def price_credit_index_option_black_on_spread(
    market_state,
    spec: CreditIndexOptionSpec,
) -> float:
    """Price a credit-index spread option with Black-76 on the forward spread."""
    maturity = max(float(spec.maturity_years), 0.0)
    discount_factor = _discount_factor(market_state, spec, maturity)
    unit_price = _black_unit_spread_option(spec, maturity)
    return float(_spread_option_multiplier(spec) * discount_factor * unit_price)


def price_credit_index_option_monte_carlo(
    market_state,
    spec: CreditIndexOptionSpec,
    *,
    n_paths: int = 65_536,
    seed: int | None = 55,
) -> float:
    """Price a credit-index spread option by antithetic lognormal spread MC."""
    path_count = max(int(n_paths), 1)
    maturity = max(float(spec.maturity_years), 0.0)
    discount_factor = _discount_factor(market_state, spec, maturity)
    if maturity <= 0.0 or spec.spread_volatility <= 0.0:
        unit_price = _intrinsic_unit_spread_option(spec)
        return float(_spread_option_multiplier(spec) * discount_factor * unit_price)

    rng = raw_np.random.default_rng(seed)
    half_count = (path_count + 1) // 2
    shocks = rng.standard_normal(half_count)
    shocks = raw_np.concatenate([shocks, -shocks])[:path_count]
    sigma_sqrt_t = float(spec.spread_volatility) * float(raw_np.sqrt(maturity))
    terminal_spreads = float(spec.forward_spread) * raw_np.exp(
        -0.5 * float(spec.spread_volatility) ** 2 * maturity
        + sigma_sqrt_t * shocks
    )
    if spec.option_type == "call":
        payoffs = raw_np.maximum(terminal_spreads - float(spec.strike_spread), 0.0)
    else:
        payoffs = raw_np.maximum(float(spec.strike_spread) - terminal_spreads, 0.0)
    return float(_spread_option_multiplier(spec) * discount_factor * raw_np.mean(payoffs))


def _black_unit_spread_option(spec: CreditIndexOptionSpec, maturity: float) -> float:
    if maturity <= 0.0 or spec.spread_volatility <= 0.0:
        return _intrinsic_unit_spread_option(spec)
    if spec.option_type == "call":
        return float(
            black76_call(
                float(spec.forward_spread),
                float(spec.strike_spread),
                float(spec.spread_volatility),
                maturity,
            )
        )
    return float(
        black76_put(
            float(spec.forward_spread),
            float(spec.strike_spread),
            float(spec.spread_volatility),
            maturity,
        )
    )


def _intrinsic_unit_spread_option(spec: CreditIndexOptionSpec) -> float:
    if spec.option_type == "call":
        return max(float(spec.forward_spread) - float(spec.strike_spread), 0.0)
    return max(float(spec.strike_spread) - float(spec.forward_spread), 0.0)


def _spread_option_multiplier(spec: CreditIndexOptionSpec) -> float:
    return float(spec.notional) * float(spec.index_annuity) * float(spec.loss_multiplier)


def _discount_factor(market_state, spec: CreditIndexOptionSpec, maturity: float) -> float:
    discount = getattr(market_state, "discount", None) if market_state is not None else None
    if discount is not None and hasattr(discount, "discount"):
        try:
            return float(discount.discount(maturity))
        except Exception:
            pass
    return float(np.exp(-float(spec.discount_rate) * maturity))


def _normalize_spread_quote(value: float) -> float:
    spread = float(value)
    if spread > 1.0:
        return spread * 1.0e-4
    return spread


__all__ = [
    "CreditIndexOptionSpec",
    "price_credit_index_option_black_on_spread",
    "price_credit_index_option_monte_carlo",
]
