"""Checked analytical adapter for the FinancePy fixed-lookback parity task."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import exp, isfinite, log, pi, sqrt

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import (
    discount_factor_from_zero_rate,
    normalized_option_type,
)
from trellis.models.analytical.support.probability import standard_normal_cdf
from trellis.models.resolution.single_state_diffusion import (
    resolve_scalar_diffusion_market_inputs,
)


_CARRY_LIMIT = 1e-8


@dataclass(frozen=True)
class LookbackOptionSpec:
    """Admitted continuously monitored European fixed-strike contract."""

    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = "call"
    lookback_type: str = "fixed_strike"
    running_extreme: float | None = None
    dividend_yield: float | None = None
    monitoring_style: str = "continuous"
    exercise_style: str = "european"
    day_count: DayCountConvention = DayCountConvention.ACT_365


@dataclass(frozen=True)
class _ScalarMarketCoordinate:
    spot: float
    expiry_date: date
    day_count: DayCountConvention
    dividend_yield: float | None = None


def _standard_normal_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)


class LookbackOptionPayoff:
    """Compose the admitted closed form from reusable Trellis primitives."""

    def __init__(self, spec: LookbackOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> LookbackOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        if str(spec.lookback_type).strip().lower() != "fixed_strike":
            raise ValueError("analytical lookback pricing requires fixed_strike")
        if str(spec.monitoring_style).strip().lower() != "continuous":
            raise ValueError(
                "analytical lookback pricing requires continuous monitoring"
            )
        if str(spec.exercise_style).strip().lower() != "european":
            raise ValueError("analytical lookback pricing requires European exercise")
        try:
            option_type = normalized_option_type(spec.option_type)
        except ValueError as exc:
            raise ValueError("lookback option_type must be call or put") from exc

        settlement = market_state.settlement or market_state.as_of
        if settlement is None:
            raise ValueError("lookback pricing requires settlement or as_of")
        maturity = float(year_fraction(settlement, spec.expiry_date, spec.day_count))
        if maturity < 0.0:
            raise ValueError("lookback expiry must not precede settlement")

        contract_spot = float(spec.spot)
        runtime_spot = market_state.spot
        spot = float(contract_spot if runtime_spot is None else runtime_spot)
        strike = float(spec.strike)
        notional = float(spec.notional)
        historical_extreme = float(
            contract_spot if spec.running_extreme is None else spec.running_extreme
        )
        if not all(
            isfinite(value)
            for value in (contract_spot, spot, strike, notional, historical_extreme)
        ):
            raise ValueError("lookback scalar contract values must be finite")
        if contract_spot <= 0.0 or spot <= 0.0 or strike <= 0.0:
            raise ValueError("lookback spot and strike must be positive")
        if historical_extreme <= 0.0:
            raise ValueError("lookback running extreme must be positive")

        if option_type == "call":
            if historical_extreme < contract_spot:
                raise ValueError(
                    "lookback running maximum must be at least contract spot"
                )
            running_extreme = max(historical_extreme, spot)
        else:
            if historical_extreme > contract_spot:
                raise ValueError(
                    "lookback running minimum must not exceed contract spot"
                )
            running_extreme = min(historical_extreme, spot)

        if maturity == 0.0:
            intrinsic = (
                max(running_extreme - strike, 0.0)
                if option_type == "call"
                else max(strike - running_extreme, 0.0)
            )
            return notional * intrinsic

        coordinate = _ScalarMarketCoordinate(
            spot=spot,
            expiry_date=spec.expiry_date,
            day_count=spec.day_count,
            dividend_yield=spec.dividend_yield,
        )
        resolved = resolve_scalar_diffusion_market_inputs(
            market_state,
            coordinate,
            volatility_coordinate=strike,
        )
        rate = resolved.rate
        dividend_yield = resolved.dividend_yield
        sigma = resolved.sigma
        if sigma <= 0.0:
            raise ValueError("analytical lookback pricing requires positive volatility")

        sigma_squared = sigma * sigma
        sigma_sqrt_time = sigma * sqrt(maturity)
        carry = rate - dividend_yield
        rate_discount = discount_factor_from_zero_rate(rate, maturity)
        dividend_discount = discount_factor_from_zero_rate(
            dividend_yield,
            maturity,
        )

        def carry_correction(boundary: float) -> tuple[float, float, float]:
            log_moneyness = log(spot / boundary)
            d1 = (
                log_moneyness + (carry + 0.5 * sigma_squared) * maturity
            ) / sigma_sqrt_time
            d2 = d1 - sigma_sqrt_time
            if abs(carry) <= _CARRY_LIMIT:
                density = _standard_normal_pdf(d1)
                if option_type == "call":
                    correction = (
                        log_moneyness + 0.5 * sigma_squared * maturity
                    ) * standard_normal_cdf(d1) + sigma_sqrt_time * density
                else:
                    correction = (
                        -log_moneyness - 0.5 * sigma_squared * maturity
                    ) * standard_normal_cdf(-d1) + sigma_sqrt_time * density
                return d1, d2, correction

            weight = 2.0 * carry / sigma_squared
            carry_growth = exp(carry * maturity)
            shifted_d1 = d1 - 2.0 * carry * sqrt(maturity) / sigma
            at_boundary = abs(log_moneyness) <= 1e-14
            if option_type == "call":
                if at_boundary:
                    term = -standard_normal_cdf(
                        shifted_d1
                    ) + carry_growth * standard_normal_cdf(d1)
                elif weight > 100.0:
                    term = carry_growth * standard_normal_cdf(d1)
                else:
                    term = -exp(-weight * log_moneyness) * standard_normal_cdf(
                        shifted_d1
                    ) + carry_growth * standard_normal_cdf(d1)
            else:
                if at_boundary:
                    term = standard_normal_cdf(
                        -shifted_d1
                    ) - carry_growth * standard_normal_cdf(-d1)
                elif weight < -100.0:
                    term = -carry_growth * standard_normal_cdf(-d1)
                else:
                    term = exp(-weight * log_moneyness) * standard_normal_cdf(
                        -shifted_d1
                    ) - carry_growth * standard_normal_cdf(-d1)
            return d1, d2, sigma_squared * term / (2.0 * carry)

        if option_type == "call":
            boundary = max(strike, running_extreme)
            d1, d2, correction = carry_correction(boundary)
            unit_price = (
                rate_discount * max(running_extreme - strike, 0.0)
                + spot * dividend_discount * standard_normal_cdf(d1)
                - boundary * rate_discount * standard_normal_cdf(d2)
                + spot * rate_discount * correction
            )
        else:
            boundary = min(strike, running_extreme)
            d1, d2, correction = carry_correction(boundary)
            unit_price = (
                rate_discount * max(strike - running_extreme, 0.0)
                - spot * dividend_discount * standard_normal_cdf(-d1)
                + boundary * rate_discount * standard_normal_cdf(-d2)
                + spot * rate_discount * correction
            )

        price = float(notional * unit_price)
        if not isfinite(price):
            raise ValueError("analytical lookback formula returned a non-finite price")
        return price
