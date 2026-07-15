"""Agent-generated payoff: Build a pricer for: FinancePy parity: equity compound option

compound_option

Spot: 100.0.
Outer option type: call.
Inner option type: call.
Outer strike: 12.0.
Inner strike: 100.0.
Outer expiry date: 2025-05-16.
Inner expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.compound.black_scholes
Benchmark product: compound_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite, log, sqrt

from trellis.core.date_utils import year_fraction
from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import (
    discount_factor_from_zero_rate,
    discounted_value,
    forward_from_dividend_yield,
)
from trellis.models.analytical.support.probability import (
    bivariate_standard_normal_cdf,
    standard_normal_cdf,
)
from trellis.models.black import black76_call, black76_put
from trellis.models.calibration.solve_request import (
    ObjectiveBundle,
    SolveBounds,
    SolveRequest,
    execute_solve_request,
)
from trellis.models.resolution.single_state_diffusion import (
    resolve_scalar_diffusion_market_inputs,
)

@dataclass(frozen=True)
class CompoundOptionSpec:
    """European option on a European option under constant diffusion inputs."""

    notional: float
    spot: float
    outer_expiry_date: date
    inner_expiry_date: date
    outer_strike: float
    inner_strike: float
    outer_option_type: str = 'call'
    inner_option_type: str = 'call'
    dividend_yield: float | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


@dataclass(frozen=True)
class _ScalarMarketCoordinate:
    """Product-neutral coordinate for the admitted constant-input projection."""

    spot: float
    expiry_date: date
    day_count: DayCountConvention
    dividend_yield: float | None = None


class CompoundOptionPayoff:
    """Build a pricer for: FinancePy parity: equity compound option

compound_option

Spot: 100.0.
Outer option type: call.
Inner option type: call.
Outer strike: 12.0.
Inner strike: 100.0.
Outer expiry date: 2025-05-16.
Inner expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.compound.black_scholes
Benchmark product: compound_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

    def __init__(self, spec: CompoundOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> CompoundOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        settlement = market_state.settlement or market_state.as_of
        if settlement is None:
            raise ValueError("compound option pricing requires settlement or as_of")

        outer_time = float(
            year_fraction(settlement, spec.outer_expiry_date, spec.day_count)
        )
        inner_time = float(
            year_fraction(settlement, spec.inner_expiry_date, spec.day_count)
        )
        if not (0.0 < outer_time < inner_time):
            raise ValueError(
                "outer expiry must lie strictly between settlement and inner expiry"
            )

        runtime_spot = market_state.spot
        spot = float(spec.spot if runtime_spot is None else runtime_spot)
        outer_strike = float(spec.outer_strike)
        inner_strike = float(spec.inner_strike)
        notional = float(spec.notional)
        if not all(
            isfinite(value)
            for value in (spot, outer_strike, inner_strike, notional)
        ):
            raise ValueError("compound option scalar contract values must be finite")
        if spot <= 0.0 or outer_strike <= 0.0 or inner_strike <= 0.0:
            raise ValueError("compound option spot and strikes must be positive")

        outer_option_type = str(spec.outer_option_type or "call").strip().lower()
        inner_option_type = str(spec.inner_option_type or "call").strip().lower()
        if outer_option_type not in {"call", "put"} or inner_option_type not in {
            "call",
            "put",
        }:
            raise ValueError(
                "unsupported compound option types "
                f"outer={outer_option_type!r}, inner={inner_option_type!r}"
            )

        coordinate = _ScalarMarketCoordinate(
            spot=spot,
            expiry_date=spec.inner_expiry_date,
            day_count=spec.day_count,
            dividend_yield=spec.dividend_yield,
        )
        resolved = resolve_scalar_diffusion_market_inputs(
            market_state,
            coordinate,
            volatility_coordinate=inner_strike,
        )
        rate = resolved.rate
        dividend_yield = resolved.dividend_yield
        sigma = resolved.sigma
        if sigma <= 0.0:
            raise ValueError("compound analytical pricing requires positive volatility")

        remaining_time = inner_time - outer_time

        def inner_value_at_outer(stock_level: float) -> float:
            forward = forward_from_dividend_yield(
                spot=stock_level,
                domestic_rate=rate,
                dividend_yield=dividend_yield,
                T=remaining_time,
            )
            kernel = black76_call if inner_option_type == "call" else black76_put
            return float(
                discounted_value(
                    kernel(forward, inner_strike, sigma, remaining_time),
                    discount_factor_from_zero_rate(rate, remaining_time),
                )
            )

        def critical_state_balance(stock_level: object) -> float:
            return inner_value_at_outer(float(stock_level)) - outer_strike

        state_scale = max(spot, outer_strike, inner_strike, 1.0)
        lower_state = state_scale * 1e-8
        upper_state = state_scale * 1e6
        lower_balance = critical_state_balance(lower_state)
        upper_balance = critical_state_balance(upper_state)
        if not isfinite(lower_balance) or not isfinite(upper_balance):
            raise ValueError("compound critical-state bracket values must be finite")
        if not (
            lower_balance <= 0.0 <= upper_balance
            or upper_balance <= 0.0 <= lower_balance
        ):
            raise ValueError(
                "compound critical-state objective has no sign change on the admitted bracket"
            )

        solve_result = execute_solve_request(
            SolveRequest(
                request_id="compound-critical-state",
                problem_kind="root_scalar",
                parameter_names=("critical_stock",),
                initial_guess=(spot,),
                objective=ObjectiveBundle(
                    objective_kind="root_scalar",
                    labels=("inner_value_minus_outer_strike",),
                    target_values=(0.0,),
                    scalar_objective_fn=critical_state_balance,
                ),
                bounds=SolveBounds(lower=(lower_state,), upper=(upper_state,)),
                solver_hint="brentq",
                metadata={"contract": "european_compound"},
                options={"tol": 1e-10},
            )
        )
        if not solve_result.success or len(solve_result.solution) != 1:
            raise ValueError("compound pricing requires a successful critical-stock solve")
        critical_spot = solve_result.solution[0]
        if not isfinite(critical_spot) or critical_spot <= 0.0:
            raise ValueError(
                "compound solve must return a positive finite critical stock state"
            )

        outer_sigma_time = sigma * sqrt(outer_time)
        inner_sigma_time = sigma * sqrt(inner_time)
        a1 = (
            log(spot / critical_spot)
            + (rate - dividend_yield + 0.5 * sigma * sigma) * outer_time
        ) / outer_sigma_time
        a2 = a1 - outer_sigma_time
        b1 = (
            log(spot / inner_strike)
            + (rate - dividend_yield + 0.5 * sigma * sigma) * inner_time
        ) / inner_sigma_time
        b2 = b1 - inner_sigma_time
        correlation = sqrt(outer_time / inner_time)
        outer_discount = discount_factor_from_zero_rate(rate, outer_time)
        inner_rate_discount = discount_factor_from_zero_rate(rate, inner_time)
        inner_dividend_discount = discount_factor_from_zero_rate(
            dividend_yield, inner_time
        )

        if outer_option_type == "call" and inner_option_type == "call":
            value = (
                spot
                * inner_dividend_discount
                * bivariate_standard_normal_cdf(a1, b1, correlation)
                - inner_strike
                * inner_rate_discount
                * bivariate_standard_normal_cdf(a2, b2, correlation)
                - outer_strike * outer_discount * standard_normal_cdf(a2)
            )
        elif outer_option_type == "put" and inner_option_type == "call":
            value = (
                inner_strike
                * inner_rate_discount
                * bivariate_standard_normal_cdf(-a2, b2, -correlation)
                - spot
                * inner_dividend_discount
                * bivariate_standard_normal_cdf(-a1, b1, -correlation)
                + outer_strike * outer_discount * standard_normal_cdf(-a2)
            )
        elif outer_option_type == "call" and inner_option_type == "put":
            value = (
                inner_strike
                * inner_rate_discount
                * bivariate_standard_normal_cdf(-a2, -b2, correlation)
                - spot
                * inner_dividend_discount
                * bivariate_standard_normal_cdf(-a1, -b1, correlation)
                - outer_strike * outer_discount * standard_normal_cdf(-a2)
            )
        else:
            value = (
                spot
                * inner_dividend_discount
                * bivariate_standard_normal_cdf(a1, -b1, -correlation)
                - inner_strike
                * inner_rate_discount
                * bivariate_standard_normal_cdf(a2, -b2, -correlation)
                + outer_strike * outer_discount * standard_normal_cdf(a2)
            )
        return float(notional * value)
