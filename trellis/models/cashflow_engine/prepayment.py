"""Prepayment models for MBS/ABS pricing."""

from __future__ import annotations

import numpy as raw_np


class PSA:
    """Public Securities Association prepayment benchmark.

    CPR ramps linearly from 0.2% to 6% over the first 30 months,
    then stays at 6%. Multiply by ``speed`` for faster/slower.

    Parameters
    ----------
    speed : float
        PSA speed (1.0 = 100% PSA, 2.0 = 200% PSA).
    """

    def __init__(self, speed: float = 1.0):
        self.speed = speed

    def cpr(self, month: int) -> float:
        """Conditional prepayment rate (annualized) at given month of seasoning."""
        if month <= 30:
            base = 0.06 * month / 30
        else:
            base = 0.06
        return base * self.speed

    def smm(self, month: int) -> float:
        """Single monthly mortality (monthly prepayment rate)."""
        annual_cpr = self.cpr(month)
        return 1 - (1 - annual_cpr) ** (1 / 12)


class CPR:
    """Constant Prepayment Rate model.

    Parameters
    ----------
    rate : float
        Annualized CPR (e.g. 0.06 for 6%).
    """

    def __init__(self, rate: float):
        self.rate = rate

    def cpr(self, month: int) -> float:
        return self.rate

    def smm(self, month: int) -> float:
        return 1 - (1 - self.rate) ** (1 / 12)


class RateDependent:
    """Rate-dependent prepayment model (simplified Richard-Roll style).

    CPR increases as rates fall below the coupon rate (refinancing incentive).

    Parameters
    ----------
    coupon : float
        Mortgage coupon rate.
    base_cpr : float
        Base CPR when rates equal coupon.
    incentive_mult : float
        How much CPR increases per 100bp of incentive.
    burnout : float
        Burnout factor (0-1). Higher = more burnout over time.
    """

    def __init__(self, coupon: float, base_cpr: float = 0.06,
                 incentive_mult: float = 0.3, burnout: float = 0.01):
        self.coupon = coupon
        self.base_cpr = base_cpr
        self.incentive_mult = incentive_mult
        self.burnout = burnout

    def cpr(self, month: int, current_rate: float) -> float:
        """CPR given current mortgage rate."""
        incentive = max(self.coupon - current_rate, 0) * 100  # in bps / 100
        burnout_factor = raw_np.exp(-self.burnout * month)
        return self.base_cpr + self.incentive_mult * incentive * burnout_factor

    def smm(self, month: int, current_rate: float) -> float:
        annual = self.cpr(month, current_rate)
        return 1 - (1 - min(annual, 0.99)) ** (1 / 12)
