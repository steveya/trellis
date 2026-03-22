"""Trellis: AI-augmented, self-evolving quantitative pricing library."""

from trellis.instruments.bond import Bond, ParBond
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.pricer import price_instrument as price
from trellis.book import Book, BookResult
from trellis.session import Session
from trellis.pipeline import Pipeline
from trellis.samples import sample_bond_10y, sample_book, sample_curve
from trellis.core.market_state import MarketState, MissingCapabilityError
from trellis.core.payoff import Payoff, DeterministicCashflowPayoff, Cashflows, PresentValue
from trellis.core.state_space import StateSpace
from trellis.engine.payoff_pricer import price_payoff
from trellis.curves.forward_curve import ForwardCurve
from trellis.models.black import black76_call, black76_put
from trellis.models.vol_surface import VolSurface, FlatVol
from trellis.instruments.cap import CapPayoff, FloorPayoff, CapFloorSpec
from trellis.instruments.scenario_weighted import ScenarioWeightedPayoff
from trellis.conventions.calendar import (
    Calendar, BusinessDayAdjustment,
    US_SETTLEMENT, UK_SETTLEMENT, TARGET,
)
from trellis.conventions.rate_index import RateIndex, SOFR_ON, SOFR_3M, SONIA
from trellis.curves.credit_curve import CreditCurve
from trellis.instruments.swap import SwapPayoff, SwapSpec, par_swap_rate
from trellis.instruments.fx import FXRate, FXForward, FXForwardPayoff
from trellis.curves.bootstrap import BootstrapInstrument, bootstrap_yield_curve
from trellis.core.capabilities import analyze_gap, capability_summary


def ask(query: str, session=None, **kwargs):
    """Price an instrument described in natural language.

    Uses mock market data by default (flat 4.5% curve, 20% vol).
    Pass a ``session`` to use a specific market snapshot.

    Example::

        result = trellis.ask("Price a 5Y cap at 4% on $10M")
        print(result.price)
    """
    from trellis.agent.ask import ask_session
    if session is None:
        from trellis.samples import sample_session
        from trellis.models.vol_surface import FlatVol as _FV
        session = sample_session()
        # Add a default vol surface for option pricing
        if session.vol_surface is None:
            session = session.with_vol_surface(_FV(0.20))
    return ask_session(query, session, **kwargs)


def quickstart() -> Session:
    """Get a Session with mock market data — no network, no API keys.

    Example::

        s = trellis.quickstart()
        result = s.price(trellis.sample_bond_10y())
    """
    from trellis.samples import sample_session
    return sample_session()


__all__ = [
    "Bond", "ParBond", "YieldCurve",
    "price", "ask",
    "Book", "BookResult", "Session", "Pipeline",
    "quickstart", "sample_bond_10y", "sample_book", "sample_curve",
    "MarketState", "MissingCapabilityError",
    "Payoff", "DeterministicCashflowPayoff", "Cashflows", "PresentValue",
    "StateSpace", "price_payoff",
    "ForwardCurve",
    "black76_call", "black76_put",
    "VolSurface", "FlatVol",
    "CapPayoff", "FloorPayoff", "CapFloorSpec",
    "ScenarioWeightedPayoff",
    "Calendar", "BusinessDayAdjustment",
    "US_SETTLEMENT", "UK_SETTLEMENT", "TARGET",
    "RateIndex", "SOFR_ON", "SOFR_3M", "SONIA",
    "CreditCurve",
    "SwapPayoff", "SwapSpec", "par_swap_rate",
    "FXRate", "FXForward", "FXForwardPayoff",
    "BootstrapInstrument", "bootstrap_yield_curve",
    "analyze_gap", "capability_summary",
]
