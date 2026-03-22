"""Amortization schedule generators."""

from __future__ import annotations

import numpy as raw_np


def level_pay(
    balance: float,
    rate: float,
    n_periods: int,
) -> list[tuple[float, float]]:
    """Level-pay (fully amortizing) schedule.

    Parameters
    ----------
    balance : float
        Initial loan balance.
    rate : float
        Periodic interest rate (e.g. annual rate / 12 for monthly).
    n_periods : int
        Total number of periods.

    Returns
    -------
    list of (interest, principal) tuples per period.
    """
    if rate == 0:
        payment = balance / n_periods
        return [(0.0, payment)] * n_periods

    payment = balance * rate / (1 - (1 + rate) ** (-n_periods))
    schedule = []
    remaining = balance

    for _ in range(n_periods):
        interest = remaining * rate
        principal = payment - interest
        principal = min(principal, remaining)
        schedule.append((interest, principal))
        remaining -= principal

    return schedule


def scheduled(
    balance: float,
    rate: float,
    principal_schedule: list[float],
) -> list[tuple[float, float]]:
    """Scheduled amortization (specified principal payments).

    Parameters
    ----------
    balance : float
    rate : float
        Periodic rate.
    principal_schedule : list[float]
        Principal payment for each period.

    Returns
    -------
    list of (interest, principal) tuples.
    """
    result = []
    remaining = balance
    for prin in principal_schedule:
        interest = remaining * rate
        prin = min(prin, remaining)
        result.append((interest, prin))
        remaining -= prin
    return result


def custom(
    balance: float,
    rate_schedule: list[float],
    principal_schedule: list[float],
) -> list[tuple[float, float]]:
    """Custom schedule with time-varying rates and principal.

    Parameters
    ----------
    balance : float
    rate_schedule : list[float]
        Periodic rate for each period.
    principal_schedule : list[float]
        Principal for each period.
    """
    result = []
    remaining = balance
    for rate, prin in zip(rate_schedule, principal_schedule):
        interest = remaining * rate
        prin = min(prin, remaining)
        result.append((interest, prin))
        remaining -= prin
    return result
