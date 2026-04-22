"""Option-Adjusted Spread (OAS) for callable/puttable bonds.

OAS is the constant spread over the treasury curve such that repricing
the bond (with its embedded option) on the shifted curve equals the
observed market price.

For callable bonds, the repricing uses a Hull-White rate tree on the
shifted curve — the tree must be recalibrated at each trial spread.

This is the spread that compensates for credit risk AFTER accounting for
the option (unlike z-spread which ignores optionality).
"""

from __future__ import annotations

from datetime import date

from scipy.optimize import brentq

from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve


def compute_oas(
    payoff,
    market_price: float,
    curve: YieldCurve,
    settlement: date,
    vol_surface=None,
    spread_range: tuple[float, float] = (-500, 500),
    tol: float = 0.01,
) -> float:
    """Compute the option-adjusted spread for any Payoff.

    Parameters
    ----------
    payoff : Payoff
        Must implement ``evaluate(market_state)`` and return a present-value
        scalar.
    market_price : float
        Observed market price (clean or dirty, depending on payoff convention).
    curve : YieldCurve
        Base treasury curve (OAS is spread over this).
    settlement : date
        Settlement date.
    vol_surface : VolSurface or None
        Volatility surface for option pricing.
    spread_range : tuple[float, float]
        Search bounds in basis points.
    tol : float
        Root-finding tolerance in basis points.

    Returns
    -------
    float
        OAS in basis points.
    """

    def objective(bps: float) -> float:
        """Return the callable-payoff pricing error after a parallel spread shift."""
        shifted_curve = curve.shift(bps)
        ms = MarketState(
            as_of=settlement,
            settlement=settlement,
            discount=shifted_curve,
            vol_surface=vol_surface,
        )
        model_price = payoff.evaluate(ms)
        return float(model_price - market_price)

    oas = brentq(objective, spread_range[0], spread_range[1], xtol=tol)
    return oas
