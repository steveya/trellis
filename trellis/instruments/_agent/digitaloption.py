"""Agent-generated payoff: Build a pricer for: FinancePy parity: equity digital cash-or-nothing

digital_option

Spot: 100.0.
Strike: 100.0.
Option type: call.
Payout type: cash_or_nothing.
Cash payoff: 10.0.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.digital.black_scholes
Benchmark product: digital_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from trellis.core.market_state import MarketState
from trellis.core.types import DayCountConvention
from trellis.models.analytical.support import (
    asset_or_nothing_intrinsic,
    cash_or_nothing_intrinsic,
    discounted_value,
    forward_from_dividend_yield,
)
from trellis.models.black import (
    black76_asset_or_nothing_call,
    black76_asset_or_nothing_put,
    black76_cash_or_nothing_call,
    black76_cash_or_nothing_put,
)
from trellis.models.resolution.single_state_diffusion import (
    resolve_single_state_diffusion_inputs,
)


@dataclass(frozen=True)
class DigitalOptionSpec:
    """European equity digital contract.

    Supports cash-or-nothing and asset-or-nothing call/put settlement with an
    explicit continuous dividend yield.
    """
    notional: float
    spot: float
    strike: float
    expiry_date: date
    option_type: str = 'call'
    payout_type: str = 'cash_or_nothing'
    cash_payoff: float = 1.0
    dividend_yield: float = 0.0
    day_count: DayCountConvention = DayCountConvention.ACT_365


class DigitalOptionPayoff:
    """Build a pricer for: FinancePy parity: equity digital cash-or-nothing

digital_option

Spot: 100.0.
Strike: 100.0.
Option type: call.
Payout type: cash_or_nothing.
Cash payoff: 10.0.
Expiry date: 2025-11-15.

Preferred method family: analytical
FinancePy binding: financepy.equity.digital.black_scholes
Benchmark product: digital_option

Construct methods: analytical
Comparison targets: analytical (analytical)
Cross-validation harness:
  external targets: financepy

Implementation target: analytical."""

    def __init__(self, spec: DigitalOptionSpec):
        self._spec = spec

    @property
    def spec(self) -> DigitalOptionSpec:
        return self._spec

    @property
    def requirements(self) -> set[str]:
        return {"black_vol_surface", "discount_curve"}

    def evaluate(self, market_state: MarketState) -> float:
        spec = self._spec
        resolved = resolve_single_state_diffusion_inputs(market_state, spec)
        payout_type = str(spec.payout_type or "cash_or_nothing").strip().lower()
        if payout_type not in {"cash_or_nothing", "asset_or_nothing"}:
            raise ValueError(f"Unsupported payout_type {spec.payout_type!r}")

        if resolved.maturity <= 0.0:
            if payout_type == "cash_or_nothing":
                settlement_amount = cash_or_nothing_intrinsic(
                    resolved.option_type,
                    spot=resolved.spot,
                    strike=resolved.strike,
                    cash=spec.cash_payoff,
                )
            else:
                settlement_amount = asset_or_nothing_intrinsic(
                    resolved.option_type,
                    spot=resolved.spot,
                    strike=resolved.strike,
                )
            return float(resolved.notional * settlement_amount)

        discount_curve = market_state.discount
        if discount_curve is None:
            raise ValueError("digital option pricing requires market_state.discount")
        discount_factor = float(discount_curve.discount(resolved.maturity))
        forward = forward_from_dividend_yield(
            spot=resolved.spot,
            domestic_rate=resolved.rate,
            dividend_yield=resolved.dividend_yield,
            T=resolved.maturity,
        )

        if payout_type == "cash_or_nothing":
            kernel = (
                black76_cash_or_nothing_call
                if resolved.option_type == "call"
                else black76_cash_or_nothing_put
            )
            payout_scale = float(spec.cash_payoff)
        else:
            kernel = (
                black76_asset_or_nothing_call
                if resolved.option_type == "call"
                else black76_asset_or_nothing_put
            )
            payout_scale = 1.0

        undiscounted = kernel(
            forward,
            resolved.strike,
            resolved.sigma,
            resolved.maturity,
        )
        return float(
            discounted_value(
                undiscounted,
                discount_factor,
                scale=resolved.notional * payout_scale,
            )
        )
