"""Analytical equity-exotic helpers used by deterministic benchmark wrappers.

These helpers intentionally cover the FinancePy parity tranche with shared
spot/rate/dividend/vol resolution from ``MarketState``. They are exact helper
surfaces for benchmark-style European contracts, not a generic path-dependent
engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from typing import Protocol

from scipy.optimize import newton
from scipy.stats import multivariate_normal

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention


def _normal_cdf(value: float) -> float:
    return float(0.5 * (1.0 + erf(float(value) / sqrt(2.0))))


def _bivariate_normal_cdf(x: float, y: float, rho: float) -> float:
    rho_clamped = max(min(float(rho), 0.999999), -0.999999)
    return float(
        multivariate_normal.cdf(
            [float(x), float(y)],
            mean=[0.0, 0.0],
            cov=[[1.0, rho_clamped], [rho_clamped, 1.0]],
        )
    )


def _linear_interp(x: float, xs: list[float], ys: list[float]) -> float:
    if not xs or len(xs) != len(ys):
        raise ValueError("Interpolation requires aligned strike and value grids")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for left, right, y_left, y_right in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        if left <= x <= right:
            if abs(right - left) <= 1e-12:
                return y_left
            weight = (x - left) / (right - left)
            return y_left + weight * (y_right - y_left)
    return ys[-1]


def _float_grid(value: object | None) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    text = str(value or "").strip()
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _forward_from_spot(spot: float, rate: float, dividend: float, maturity: float) -> float:
    return float(spot) * exp((float(rate) - float(dividend)) * float(maturity))


def _discount_factor(rate: float, maturity: float) -> float:
    return exp(-float(rate) * float(maturity))


def _black_scholes_value(
    *,
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    sigma: float,
    maturity: float,
    option_type: str,
) -> float:
    option = str(option_type or "call").strip().lower()
    if maturity <= 0.0:
        intrinsic = max(float(spot) - float(strike), 0.0)
        if option == "put":
            intrinsic = max(float(strike) - float(spot), 0.0)
        return intrinsic
    sigma_sqrt_t = max(float(sigma), 0.0) * sqrt(float(maturity))
    if sigma_sqrt_t <= 0.0:
        forward = _forward_from_spot(spot, rate, dividend, maturity)
        intrinsic = max(forward - float(strike), 0.0)
        if option == "put":
            intrinsic = max(float(strike) - forward, 0.0)
        return _discount_factor(rate, maturity) * intrinsic
    d1 = (log(float(spot) / float(strike)) + (float(rate) - float(dividend) + 0.5 * float(sigma) ** 2) * float(maturity)) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    df_r = _discount_factor(rate, maturity)
    df_q = _discount_factor(dividend, maturity)
    if option == "put":
        return float(strike) * df_r * _normal_cdf(-d2) - float(spot) * df_q * _normal_cdf(-d1)
    return float(spot) * df_q * _normal_cdf(d1) - float(strike) * df_r * _normal_cdf(d2)


@dataclass(frozen=True)
class ResolvedEquityAnalyticalInputs:
    spot: float
    rate: float
    dividend: float
    sigma: float
    notional: float


class _EquityOptionLike(Protocol):
    spot: float
    day_count: DayCountConvention
    notional: float


def _carry_rate_from_market_state(market_state: MarketState, spec) -> float:
    if getattr(spec, "dividend_yield", None) is not None:
        return float(spec.dividend_yield)
    if getattr(spec, "dividend_rate", None) is not None:
        return float(spec.dividend_rate)
    params = dict(getattr(market_state, "model_parameters", None) or {})
    carry_rates = dict(params.get("underlier_carry_rates") or {})
    if carry_rates:
        return float(next(iter(carry_rates.values())))
    return 0.0


def _resolve_common_inputs(market_state: MarketState, spec: _EquityOptionLike) -> ResolvedEquityAnalyticalInputs:
    if market_state.discount is None:
        raise ValueError("Analytical equity benchmark pricing requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("Analytical equity benchmark pricing requires market_state.vol_surface")
    settlement = getattr(market_state, "settlement", None)
    day_count = getattr(spec, "time_day_count", None) or getattr(spec, "day_count", DayCountConvention.ACT_365)
    maturity = 0.0
    expiry_candidates = (
        getattr(spec, "expiry_date", None),
        getattr(spec, "inner_expiry_date", None),
        getattr(spec, "call_expiry_date", None),
        getattr(spec, "put_expiry_date", None),
        getattr(spec, "outer_expiry_date", None),
    )
    if settlement is not None:
        for expiry_date in expiry_candidates:
            if expiry_date is None:
                continue
            maturity = max(maturity, float(year_fraction(settlement, expiry_date, day_count)))
    if maturity <= 0.0:
        maturity = 1e-12
    strike_candidates = (
        getattr(spec, "strike", None),
        getattr(spec, "inner_strike", None),
        getattr(spec, "call_strike", None),
        getattr(spec, "put_strike", None),
        getattr(spec, "outer_strike", None),
    )
    strike = next((float(candidate) for candidate in strike_candidates if candidate not in {None, ""}), 0.0)
    sigma = float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike))
    rate = float(market_state.discount.zero_rate(max(maturity, 1e-6)))
    return ResolvedEquityAnalyticalInputs(
        spot=float(spec.spot),
        rate=rate,
        dividend=_carry_rate_from_market_state(market_state, spec),
        sigma=sigma,
        notional=float(getattr(spec, "notional", 1.0) or 1.0),
    )


def price_equity_digital_option_analytical(market_state: MarketState, spec) -> float:
    resolved = _resolve_common_inputs(market_state, spec)
    maturity = max(float(year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)), 0.0)
    forward = _forward_from_spot(resolved.spot, resolved.rate, resolved.dividend, maturity)
    discount = _discount_factor(resolved.rate, maturity)
    sigma_sqrt_t = resolved.sigma * sqrt(max(maturity, 1e-12))
    if sigma_sqrt_t <= 0.0:
        unit_price = 1.0 if (forward > float(spec.strike)) == (str(spec.option_type).lower() == "call") else 0.0
    else:
        d2 = (log(forward / float(spec.strike)) - 0.5 * resolved.sigma * resolved.sigma * maturity) / sigma_sqrt_t
        unit_price = _normal_cdf(d2) if str(spec.option_type).lower() == "call" else _normal_cdf(-d2)
    cash_payoff = float(getattr(spec, "cash_payoff", 1.0) or 1.0)
    return resolved.notional * discount * cash_payoff * unit_price


def price_equity_fixed_lookback_option_analytical(market_state: MarketState, spec) -> float:
    resolved = _resolve_common_inputs(market_state, spec)
    maturity = max(float(year_fraction(market_state.settlement, spec.expiry_date, spec.day_count)), 0.0)
    if maturity <= 0.0:
        return 0.0
    option_type = str(getattr(spec, "option_type", "call") or "call").strip().lower()
    s0 = resolved.spot
    strike = float(spec.strike)
    running_extreme = float(getattr(spec, "running_extreme", s0) or s0)
    rate = resolved.rate
    dividend = resolved.dividend
    sigma = resolved.sigma
    carry = rate - dividend
    if abs(carry) < 1e-12:
        carry = 1e-12 if carry >= 0.0 else -1e-12
    u = sigma * sigma / (2.0 * carry)
    w = 2.0 * carry / (sigma * sigma)
    expbt = exp(carry * maturity)
    sqrt_t = sqrt(maturity)

    if option_type == "call":
        s_max = running_extreme
        if strike > s_max:
            d1 = (log(s0 / strike) + (carry + sigma * sigma / 2.0) * maturity) / (sigma * sqrt_t)
            d2 = d1 - sigma * sqrt_t
            if abs(s0 - strike) <= 1e-12:
                term = -_normal_cdf(d1 - 2.0 * carry * sqrt_t / sigma) + expbt * _normal_cdf(d1)
            elif s0 < strike and w > 100.0:
                term = expbt * _normal_cdf(d1)
            else:
                term = -(s0 / strike) ** (-w) * _normal_cdf(d1 - 2.0 * carry * sqrt_t / sigma) + expbt * _normal_cdf(d1)
            price = (
                s0 * exp(-dividend * maturity) * _normal_cdf(d1)
                - strike * exp(-rate * maturity) * _normal_cdf(d2)
                + s0 * exp(-rate * maturity) * u * term
            )
        else:
            e1 = (log(s0 / s_max) + (carry + sigma * sigma / 2.0) * maturity) / (sigma * sqrt_t)
            e2 = e1 - sigma * sqrt_t
            if abs(s0 - s_max) <= 1e-12:
                term = -_normal_cdf(e1 - 2.0 * carry * sqrt_t / sigma) + expbt * _normal_cdf(e1)
            elif s0 < s_max and w > 100.0:
                term = expbt * _normal_cdf(e1)
            else:
                term = -((s0 / s_max) ** (-w)) * _normal_cdf(e1 - 2.0 * carry * sqrt_t / sigma) + expbt * _normal_cdf(e1)
            price = (
                exp(-rate * maturity) * (s_max - strike)
                + s0 * exp(-dividend * maturity) * _normal_cdf(e1)
                - s_max * exp(-rate * maturity) * _normal_cdf(e2)
                + s0 * exp(-rate * maturity) * u * term
            )
    else:
        s_min = running_extreme
        if strike >= s_min:
            f1 = (log(s0 / s_min) + (carry + sigma * sigma / 2.0) * maturity) / (sigma * sqrt_t)
            f2 = f1 - sigma * sqrt_t
            if abs(s0 - s_min) <= 1e-12:
                term = _normal_cdf(-f1 + 2.0 * carry * sqrt_t / sigma) - expbt * _normal_cdf(-f1)
            elif s0 > s_min and w < -100.0:
                term = -expbt * _normal_cdf(-f1)
            else:
                term = ((s0 / s_min) ** (-w)) * _normal_cdf(-f1 + 2.0 * carry * sqrt_t / sigma) - expbt * _normal_cdf(-f1)
            price = (
                exp(-rate * maturity) * (strike - s_min)
                - s0 * exp(-dividend * maturity) * _normal_cdf(-f1)
                + s_min * exp(-rate * maturity) * _normal_cdf(-f2)
                + s0 * exp(-rate * maturity) * u * term
            )
        else:
            d1 = (log(s0 / strike) + (carry + sigma * sigma / 2.0) * maturity) / (sigma * sqrt_t)
            d2 = d1 - sigma * sqrt_t
            if abs(s0 - strike) <= 1e-12:
                term = _normal_cdf(-d1 + 2.0 * carry * sqrt_t / sigma) - expbt * _normal_cdf(-d1)
            elif s0 > strike and w < -100.0:
                term = -expbt * _normal_cdf(-d1)
            else:
                term = ((s0 / strike) ** (-w)) * _normal_cdf(-d1 + 2.0 * carry * sqrt_t / sigma) - expbt * _normal_cdf(-d1)
            price = (
                strike * exp(-rate * maturity) * _normal_cdf(-d2)
                - s0 * exp(-dividend * maturity) * _normal_cdf(-d1)
                + s0 * exp(-rate * maturity) * u * term
            )
    return resolved.notional * float(price)


def price_equity_chooser_option_analytical(market_state: MarketState, spec) -> float:
    resolved = _resolve_common_inputs(market_state, spec)
    choose_time = max(float(year_fraction(market_state.settlement, spec.choose_date, spec.day_count)), 1e-12)
    call_time = max(float(year_fraction(market_state.settlement, spec.call_expiry_date, spec.day_count)), choose_time)
    put_time = max(float(year_fraction(market_state.settlement, spec.put_expiry_date, spec.day_count)), choose_time)
    sigma = resolved.sigma
    rate = resolved.rate
    dividend = resolved.dividend
    spot = resolved.spot
    call_strike = float(spec.call_strike)
    put_strike = float(spec.put_strike)

    def chooser_balance(stock_level: float) -> float:
        call_value = _black_scholes_value(
            spot=stock_level,
            strike=call_strike,
            rate=rate,
            dividend=dividend,
            sigma=sigma,
            maturity=call_time - choose_time,
            option_type="call",
        )
        put_value = _black_scholes_value(
            spot=stock_level,
            strike=put_strike,
            rate=rate,
            dividend=dividend,
            sigma=sigma,
            maturity=put_time - choose_time,
            option_type="put",
        )
        return call_value - put_value

    try:
        critical_spot = float(newton(chooser_balance, x0=spot, tol=1e-8, maxiter=50))
    except RuntimeError as exc:
        raise ValueError(
            "Chooser option critical-spot search failed to converge. "
            "Check that choose_date precedes both expiry dates and that "
            "market parameters are in a reasonable range."
        ) from exc
    sigma_choose = sigma * sqrt(choose_time)
    d1 = (log(spot / critical_spot) + (rate - dividend + 0.5 * sigma * sigma) * choose_time) / sigma_choose
    d2 = d1 - sigma_choose
    y1 = (log(spot / call_strike) + (rate - dividend + 0.5 * sigma * sigma) * call_time) / (sigma * sqrt(call_time))
    y2 = (log(spot / put_strike) + (rate - dividend + 0.5 * sigma * sigma) * put_time) / (sigma * sqrt(put_time))
    rho1 = sqrt(choose_time / call_time)
    rho2 = sqrt(choose_time / put_time)

    price = (
        spot * exp(-dividend * call_time) * _bivariate_normal_cdf(d1, y1, rho1)
        - call_strike * exp(-rate * call_time) * _bivariate_normal_cdf(d2, y1 - sigma * sqrt(call_time), rho1)
        - spot * exp(-dividend * put_time) * _bivariate_normal_cdf(-d1, -y2, rho2)
        + put_strike * exp(-rate * put_time) * _bivariate_normal_cdf(-d2, -y2 + sigma * sqrt(put_time), rho2)
    )
    return resolved.notional * float(price)


def price_equity_compound_option_analytical(market_state: MarketState, spec) -> float:
    resolved = _resolve_common_inputs(market_state, spec)
    outer_time = max(float(year_fraction(market_state.settlement, spec.outer_expiry_date, spec.day_count)), 1e-12)
    inner_time = max(float(year_fraction(market_state.settlement, spec.inner_expiry_date, spec.day_count)), outer_time)
    sigma = resolved.sigma
    rate = resolved.rate
    dividend = resolved.dividend
    spot = resolved.spot
    outer_option_type = str(spec.outer_option_type or "call").strip().lower()
    inner_option_type = str(spec.inner_option_type or "call").strip().lower()
    outer_strike = float(spec.outer_strike)
    inner_strike = float(spec.inner_strike)

    def underlying_minus_outer(stock_level: float) -> float:
        return _black_scholes_value(
            spot=stock_level,
            strike=inner_strike,
            rate=rate,
            dividend=dividend,
            sigma=sigma,
            maturity=inner_time - outer_time,
            option_type=inner_option_type,
        ) - outer_strike

    try:
        critical_spot = float(newton(underlying_minus_outer, x0=spot, tol=1e-8, maxiter=50))
    except RuntimeError as exc:
        raise ValueError(
            "Compound option critical-spot search failed to converge. "
            "Verify that outer_strike is reachable by the inner option value "
            "and that market parameters are in a reasonable range."
        ) from exc
    a1 = (log(spot / critical_spot) + (rate - dividend + 0.5 * sigma * sigma) * outer_time) / (sigma * sqrt(outer_time))
    a2 = a1 - sigma * sqrt(outer_time)
    b1 = (log(spot / inner_strike) + (rate - dividend + 0.5 * sigma * sigma) * inner_time) / (sigma * sqrt(inner_time))
    b2 = b1 - sigma * sqrt(inner_time)
    corr = sqrt(outer_time / inner_time)
    df_outer = exp(-rate * outer_time)
    df_inner_div = exp(-dividend * inner_time)
    df_inner_rate = exp(-rate * inner_time)

    if outer_option_type == "call" and inner_option_type == "call":
        price = (
            spot * df_inner_div * _bivariate_normal_cdf(a1, b1, corr)
            - inner_strike * df_inner_rate * _bivariate_normal_cdf(a2, b2, corr)
            - outer_strike * df_outer * _normal_cdf(a2)
        )
    elif outer_option_type == "put" and inner_option_type == "call":
        price = (
            inner_strike * df_inner_rate * _bivariate_normal_cdf(-a2, b2, -corr)
            - spot * df_inner_div * _bivariate_normal_cdf(-a1, b1, -corr)
            + outer_strike * df_outer * _normal_cdf(-a2)
        )
    elif outer_option_type == "call" and inner_option_type == "put":
        price = (
            inner_strike * df_inner_rate * _bivariate_normal_cdf(-a2, -b2, corr)
            - spot * df_inner_div * _bivariate_normal_cdf(-a1, -b1, corr)
            - outer_strike * df_outer * _normal_cdf(-a2)
        )
    elif outer_option_type == "put" and inner_option_type == "put":
        price = (
            spot * df_inner_div * _bivariate_normal_cdf(a1, -b1, -corr)
            - inner_strike * df_inner_rate * _bivariate_normal_cdf(a2, -b2, -corr)
            + outer_strike * df_outer * _normal_cdf(a2)
        )
    else:
        raise ValueError(
            f"Unsupported compound option types outer={outer_option_type!r}, inner={inner_option_type!r}"
        )
    return resolved.notional * float(price)


def price_equity_cliquet_option_analytical(market_state: MarketState, spec) -> float:
    resolved = _resolve_common_inputs(market_state, spec)
    settlement = getattr(market_state, "settlement", None)
    if settlement is None:
        raise ValueError("Cliquet analytical pricing requires market_state.settlement")
    if market_state.discount is None:
        raise ValueError("Cliquet analytical pricing requires market_state.discount")
    if market_state.vol_surface is None:
        raise ValueError("Cliquet analytical pricing requires market_state.vol_surface")

    observation_dates = tuple(sorted(getattr(spec, "observation_dates", ()) or ()))
    if not observation_dates:
        raise ValueError("Cliquet analytical pricing requires non-empty observation_dates")

    option_type = str(getattr(spec, "option_type", "call") or "call").strip().lower()
    carry = resolved.dividend
    time_day_count = getattr(spec, "time_day_count", None) or getattr(spec, "day_count", DayCountConvention.ACT_365)
    total = 0.0
    previous_time = 0.0
    for observation_date in observation_dates:
        t_exp = max(float(year_fraction(settlement, observation_date, time_day_count)), 0.0)
        if t_exp <= previous_time + 1e-12:
            continue
        tau = t_exp - previous_time
        rate = float(market_state.discount.zero_rate(max(t_exp, 1e-6)))
        sigma = float(market_state.vol_surface.black_vol(max(tau, 1e-6), resolved.spot))
        dq_prev = exp(-carry * previous_time)
        unit_price = _black_scholes_value(
            spot=1.0,
            strike=1.0,
            rate=rate,
            dividend=carry,
            sigma=sigma,
            maturity=tau,
            option_type=option_type,
        )
        total += resolved.spot * dq_prev * unit_price
        previous_time = t_exp
    return resolved.notional * float(total)


def equity_variance_swap_outputs_analytical(market_state: MarketState, spec) -> dict[str, float]:
    if market_state.discount is None:
        raise ValueError("Variance swap analytical pricing requires market_state.discount")
    settlement = getattr(market_state, "settlement", None)
    if settlement is None:
        raise ValueError("Variance swap analytical pricing requires market_state.settlement")

    maturity = max(float(year_fraction(settlement, spec.expiry_date, spec.day_count)), 0.0)
    if maturity <= 0.0:
        return {
            "price": 0.0,
            "fair_strike_variance": float(getattr(spec, "strike_variance", 0.0) or 0.0),
        }

    spot = float(spec.spot)
    strike_grid = _float_grid(getattr(spec, "replication_strikes", None))
    if not strike_grid:
        strike_grid = [0.6 * spot, 0.8 * spot, 1.0 * spot, 1.2 * spot, 1.4 * spot]

    vol_grid = _float_grid(getattr(spec, "replication_volatilities", None))
    if not vol_grid:
        if market_state.vol_surface is None:
            raise ValueError("Variance swap analytical pricing requires a black vol surface or explicit replication_volatilities")
        vol_grid = [
            float(market_state.vol_surface.black_vol(max(maturity, 1e-6), strike))
            for strike in strike_grid
        ]
    if len(strike_grid) != len(vol_grid):
        raise ValueError("replication_strikes and replication_volatilities must have the same length")

    atm_vol = _linear_interp(spot, strike_grid, vol_grid)
    strike_span = strike_grid[-1] - strike_grid[0]
    slope = 0.0 if abs(strike_span) <= 1e-12 else spot * (vol_grid[-1] - vol_grid[0]) / strike_span
    fair_strike_variance = float(atm_vol**2 * sqrt(1.0 + 3.0 * maturity * slope * slope))
    discount = exp(-float(market_state.discount.zero_rate(max(maturity, 1e-6))) * maturity)
    price = float(spec.notional) * discount * (fair_strike_variance - float(spec.strike_variance))
    return {
        "price": price,
        "fair_strike_variance": fair_strike_variance,
    }


def price_equity_variance_swap_analytical(market_state: MarketState, spec) -> float:
    return float(equity_variance_swap_outputs_analytical(market_state, spec)["price"])


__all__ = [
    "ResolvedEquityAnalyticalInputs",
    "equity_variance_swap_outputs_analytical",
    "price_equity_cliquet_option_analytical",
    "price_equity_chooser_option_analytical",
    "price_equity_compound_option_analytical",
    "price_equity_digital_option_analytical",
    "price_equity_fixed_lookback_option_analytical",
    "price_equity_variance_swap_analytical",
]
