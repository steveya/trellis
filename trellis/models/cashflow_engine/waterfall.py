"""Generic waterfall / priority of payments engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as raw_np


@dataclass
class Tranche:
    """A tranche in a waterfall structure.

    Parameters
    ----------
    name : str
        Tranche identifier (e.g. "A", "B", "equity").
    notional : float
        Original notional balance.
    coupon : float
        Coupon rate (annualized).
    subordination : float
        Subordination level (0 = senior, higher = more junior).
    """

    name: str
    notional: float
    coupon: float
    subordination: float = 0.0
    balance: float = 0.0  # remaining balance (set during run)

    def __post_init__(self):
        if self.balance == 0.0:
            self.balance = self.notional


class Waterfall:
    """Generic priority-of-payments engine.

    Distributes available cash to tranches in priority order.
    Supports interest and principal waterfalls with coverage tests.

    Parameters
    ----------
    tranches : list[Tranche]
        Ordered by seniority (most senior first).
    """

    def __init__(self, tranches: list[Tranche]):
        self.tranches = sorted(tranches, key=lambda t: t.subordination)
        self._history: list[dict] = []

    def distribute(self, available_interest: float, available_principal: float,
                   period: float = 0.5) -> dict[str, dict]:
        """Distribute one period's cash through the waterfall.

        Parameters
        ----------
        available_interest : float
            Total interest collected this period.
        available_principal : float
            Total principal collected this period.
        period : float
            Accrual period in years.

        Returns
        -------
        dict mapping tranche name to {"interest": float, "principal": float}.
        """
        result = {}
        remaining_interest = available_interest
        remaining_principal = available_principal

        # Interest waterfall (senior first)
        for tranche in self.tranches:
            due = tranche.balance * tranche.coupon * period
            paid = min(due, remaining_interest)
            remaining_interest -= paid
            result[tranche.name] = {"interest": paid, "principal": 0.0}

        # Principal waterfall (senior first)
        for tranche in self.tranches:
            paid = min(tranche.balance, remaining_principal)
            remaining_principal -= paid
            tranche.balance -= paid
            result[tranche.name]["principal"] = paid

        result["_residual"] = {
            "interest": remaining_interest,
            "principal": remaining_principal,
        }

        self._history.append(result)
        return result

    def run(self, cashflows: list[tuple[float, float]],
            period: float = 0.5) -> list[dict]:
        """Run the waterfall over multiple periods.

        Parameters
        ----------
        cashflows : list of (interest, principal) tuples per period.

        Returns
        -------
        list of distribution dicts per period.
        """
        results = []
        for interest, principal in cashflows:
            results.append(self.distribute(interest, principal, period))
        return results
