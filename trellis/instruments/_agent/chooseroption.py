"""Agent-generated payoff: Build a pricer for: FinancePy parity: equity chooser option

chooser_option

Spot: 100.0.
Choose date: 2025-05-16.
Call expiry date: 2025-11-15.
Put expiry date: 2025-11-15.
Call strike: 100.0.
Put strike: 100.0.

Preferred method family: analytical
FinancePy binding: financepy.equity.chooser.black_scholes
Benchmark product: chooser_option

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
class ChooserOptionSpec:
    """Simple European chooser contract under constant diffusion inputs."""
    notional: float
    spot: float
    choose_date: date
    call_expiry_date: date
    put_expiry_date: date
    call_strike: float
    put_strike: float
    dividend_yield: float | None = None
    day_count: DayCountConvention = DayCountConvention.ACT_365


@dataclass(frozen=True)
class _ScalarMarketCoordinate:
    """Product-neutral coordinate used for one explicit market projection."""

    spot: float
    expiry_date: date
    day_count: DayCountConvention
    dividend_yield: float | None = None


class ChooserOptionPayoff:
    """Build a pricer for: FinancePy parity: equity chooser option

chooser_option

Spot: 100.0.
Choose date: 2025-05-16.
Call expiry date: 2025-11-15.
Put expiry date: 2025-11-15.
Call strike: 100.0.
Put strike: 100.0.

Preferred method family: analytical
FinancePy binding: financepy.equity.chooser.black_scholes
Benchmark product: chooser_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

    def __init__(self, spec: ChooserOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> ChooserOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        settlement = market_state.settlement or market_state.as_of
        if settlement is None:
            raise ValueError("chooser option pricing requires settlement or as_of")

        choose_time = float(year_fraction(settlement, spec.choose_date, spec.day_count))
        call_time = float(year_fraction(settlement, spec.call_expiry_date, spec.day_count))
        put_time = float(year_fraction(settlement, spec.put_expiry_date, spec.day_count))
        if not (0.0 < choose_time < call_time and choose_time < put_time):
            raise ValueError(
                "choice date must lie strictly between settlement and both expiries"
            )

        runtime_spot = market_state.spot
        spot = float(spec.spot if runtime_spot is None else runtime_spot)
        call_strike = float(spec.call_strike)
        put_strike = float(spec.put_strike)
        notional = float(spec.notional)
        if not all(isfinite(value) for value in (spot, call_strike, put_strike, notional)):
            raise ValueError("chooser option scalar contract values must be finite")
        if spot <= 0.0 or call_strike <= 0.0 or put_strike <= 0.0:
            raise ValueError("chooser option spot and strikes must be positive")

        coordinate = _ScalarMarketCoordinate(
            spot=spot,
            expiry_date=max(spec.call_expiry_date, spec.put_expiry_date),
            day_count=spec.day_count,
            dividend_yield=spec.dividend_yield,
        )
        resolved = resolve_scalar_diffusion_market_inputs(
            market_state,
            coordinate,
            volatility_coordinate=call_strike,
        )
        rate = resolved.rate
        dividend_yield = resolved.dividend_yield
        sigma = resolved.sigma
        if sigma <= 0.0:
            raise ValueError("simple chooser analytical pricing requires positive volatility")

        def option_value_at_choice(
            stock_level: float,
            *,
            strike: float,
            remaining_time: float,
            option_type: str,
        ) -> float:
            forward = forward_from_dividend_yield(
                spot=stock_level,
                domestic_rate=rate,
                dividend_yield=dividend_yield,
                T=remaining_time,
            )
            kernel = black76_call if option_type == "call" else black76_put
            return float(
                discounted_value(
                    kernel(forward, strike, sigma, remaining_time),
                    discount_factor_from_zero_rate(rate, remaining_time),
                )
            )

        def chooser_balance(stock_level: object) -> float:
            state = float(stock_level)
            call_value = option_value_at_choice(
                state,
                strike=call_strike,
                remaining_time=call_time - choose_time,
                option_type="call",
            )
            put_value = option_value_at_choice(
                state,
                strike=put_strike,
                remaining_time=put_time - choose_time,
                option_type="put",
            )
            return call_value - put_value

        state_scale = max(spot, call_strike, put_strike, 1.0)
        lower_state = state_scale * 1e-8
        upper_state = state_scale * 1e6
        lower_balance = chooser_balance(lower_state)
        upper_balance = chooser_balance(upper_state)
        if not isfinite(lower_balance) or not isfinite(upper_balance):
            raise ValueError("chooser critical-state bracket values must be finite")
        if lower_balance > 0.0 or upper_balance < 0.0:
            raise ValueError(
                "chooser critical-state objective has no sign change on the admitted bracket"
            )

        solve_result = execute_solve_request(
            SolveRequest(
                request_id="chooser-critical-state",
                problem_kind="root_scalar",
                parameter_names=("critical_stock",),
                initial_guess=(spot,),
                objective=ObjectiveBundle(
                    objective_kind="root_scalar",
                    labels=("call_value_minus_put_value",),
                    target_values=(0.0,),
                    scalar_objective_fn=chooser_balance,
                ),
                bounds=SolveBounds(
                    lower=(lower_state,),
                    upper=(upper_state,),
                ),
                solver_hint="brentq",
                metadata={"contract": "simple_chooser"},
                options={"tol": 1e-10},
            )
        )
        critical_spot = solve_result.solution[0]
        if not isfinite(critical_spot) or critical_spot <= 0.0:
            raise ValueError("chooser solve must return a positive finite critical stock state")

        sigma_choose = sigma * sqrt(choose_time)
        d1 = (
            log(spot / critical_spot)
            + (rate - dividend_yield + 0.5 * sigma * sigma) * choose_time
        ) / sigma_choose
        d2 = d1 - sigma_choose
        call_sigma_time = sigma * sqrt(call_time)
        put_sigma_time = sigma * sqrt(put_time)
        y1 = (
            log(spot / call_strike)
            + (rate - dividend_yield + 0.5 * sigma * sigma) * call_time
        ) / call_sigma_time
        y2 = (
            log(spot / put_strike)
            + (rate - dividend_yield + 0.5 * sigma * sigma) * put_time
        ) / put_sigma_time
        call_correlation = sqrt(choose_time / call_time)
        put_correlation = sqrt(choose_time / put_time)

        call_asset = (
            spot
            * discount_factor_from_zero_rate(dividend_yield, call_time)
            * bivariate_standard_normal_cdf(d1, y1, call_correlation)
        )
        call_cash = (
            call_strike
            * discount_factor_from_zero_rate(rate, call_time)
            * bivariate_standard_normal_cdf(
                d2,
                y1 - call_sigma_time,
                call_correlation,
            )
        )
        put_asset = (
            spot
            * discount_factor_from_zero_rate(dividend_yield, put_time)
            * bivariate_standard_normal_cdf(-d1, -y2, put_correlation)
        )
        put_cash = (
            put_strike
            * discount_factor_from_zero_rate(rate, put_time)
            * bivariate_standard_normal_cdf(
                -d2,
                -y2 + put_sigma_time,
                put_correlation,
            )
        )
        return float(notional * (call_asset - call_cash - put_asset + put_cash))
