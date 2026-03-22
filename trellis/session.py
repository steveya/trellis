"""Session: immutable market snapshot for interactive pricing."""

from __future__ import annotations

from datetime import date

from trellis.book import Book, BookResult
from trellis.core.market_state import MarketState
from trellis.core.payoff import Payoff
from trellis.core.types import DayCountConvention, GreeksSpec, PricingResult
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff as _price_payoff
from trellis.engine.pricer import price_instrument
from trellis.instruments.bond import Bond


class Session:
    """An immutable market snapshot for pricing.

    Parameters
    ----------
    curve : YieldCurve or None
        If *None*, auto-resolves via :func:`resolve_curve`.
    settlement : date or None
        Settlement date (defaults to today).
    as_of : date, str, or None
        Date for market data resolution (only used when *curve* is None).
    data_source : str
        ``"treasury_gov"``, ``"fred"``, or ``"mock"``.
    agent : bool
        Enable LLM agent fallback for unsupported instruments.
    vol_surface : VolSurface or None
        Volatility surface for option pricing.
    state_space : StateSpace or None
        Discrete states for scenario-weighted pricing.
    credit_curve : CreditCurve or None
        Credit / survival probability curve.
    forecast_curves : dict[str, DiscountCurve] or None
        Forecast curves keyed by rate index name.
    fx_rates : dict[str, FXRate] or None
        FX spot rates keyed by pair.
    """

    __slots__ = (
        "_curve", "_settlement", "_agent",
        "_vol_surface", "_state_space",
        "_credit_curve", "_forecast_curves", "_fx_rates",
    )

    def __init__(
        self,
        curve: YieldCurve | None = None,
        settlement: date | None = None,
        *,
        as_of: date | str | None = None,
        data_source: str = "treasury_gov",
        agent: bool = False,
        vol_surface=None,
        state_space=None,
        credit_curve=None,
        forecast_curves=None,
        fx_rates=None,
    ):
        if curve is None:
            from trellis.data.resolver import resolve_curve
            curve = resolve_curve(as_of=as_of, source=data_source)
        object.__setattr__(self, "_curve", curve)
        object.__setattr__(self, "_settlement", settlement or date.today())
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_vol_surface", vol_surface)
        object.__setattr__(self, "_state_space", state_space)
        object.__setattr__(self, "_credit_curve", credit_curve)
        object.__setattr__(self, "_forecast_curves", forecast_curves)
        object.__setattr__(self, "_fx_rates", fx_rates)

    def __setattr__(self, name, value):
        raise AttributeError("Session is immutable")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def curve(self) -> YieldCurve:
        return self._curve

    @property
    def settlement(self) -> date:
        return self._settlement

    @property
    def agent_enabled(self) -> bool:
        return self._agent

    @property
    def vol_surface(self):
        return self._vol_surface

    @property
    def state_space(self):
        return self._state_space

    @property
    def credit_curve(self):
        return self._credit_curve

    @property
    def forecast_curves(self):
        return self._forecast_curves

    @property
    def fx_rates(self):
        return self._fx_rates

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def price(
        self,
        instrument: Bond | Book,
        *,
        greeks: GreeksSpec = "all",
    ) -> PricingResult | BookResult:
        """Price a single instrument or a Book."""
        if isinstance(instrument, Book):
            return self._price_book(instrument, greeks=greeks)
        try:
            return price_instrument(
                instrument, self._curve, self._settlement, greeks=greeks,
            )
        except (NotImplementedError, TypeError):
            if self._agent:
                return self._agent_price(instrument)
            raise

    def _price_book(self, book: Book, *, greeks: GreeksSpec = "all") -> BookResult:
        results = {}
        for name in book:
            results[name] = price_instrument(
                book[name], self._curve, self._settlement, greeks=greeks,
            )
        return BookResult(results, book)

    def price_payoff(
        self,
        payoff: Payoff,
        *,
        day_count: DayCountConvention = DayCountConvention.ACT_365,
    ) -> float:
        """Price a Payoff by constructing a MarketState and delegating."""
        ms = MarketState(
            as_of=self._settlement,
            settlement=self._settlement,
            discount=self._curve,
            vol_surface=self._vol_surface,
            state_space=self._state_space,
            credit_curve=self._credit_curve,
            forecast_curves=self._forecast_curves,
            fx_rates=self._fx_rates,
        )
        return _price_payoff(payoff, ms, day_count=day_count)

    def ask(self, description: str, model: str | None = None):
        """Price an instrument described in natural language.

        Example::

            s = Session(curve=YieldCurve.flat(0.05), vol_surface=FlatVol(0.20))
            result = s.ask("Price a 5Y cap at 4% on $10M SOFR")
            print(result.price)
        """
        from trellis.agent.ask import ask_session
        return ask_session(description, self, model=model)

    def build_and_price(
        self,
        payoff_description: str,
        requirements: set[str],
        payoff_kwargs: dict,
        *,
        day_count: DayCountConvention = DayCountConvention.ACT_365,
        model: str = "claude-sonnet-4-6",
    ) -> float:
        """Build a payoff class via the agent, instantiate it, and price it.

        Parameters
        ----------
        payoff_description : str
            e.g. "European payer swaption"
        requirements : set[str]
            MarketState capabilities needed.
        payoff_kwargs : dict
            Constructor arguments for the generated payoff spec.
        day_count : DayCountConvention
            Day count for discounting.
        model : str
            LLM model to use.
        """
        from trellis.agent.executor import build_payoff
        payoff_cls = build_payoff(payoff_description, requirements, model=model)
        # The generated class should have a spec dataclass; instantiate it
        payoff = payoff_cls(**payoff_kwargs)
        return self.price_payoff(payoff, day_count=day_count)

    def _agent_price(self, instrument):
        from trellis.agent.executor import execute
        return execute(f"Price instrument: {instrument!r}")

    def greeks(
        self,
        instrument: Bond,
        *,
        measures: list[str] | None = None,
    ) -> dict:
        """Compute Greeks for an instrument."""
        spec = list(measures) if measures else "all"
        result = price_instrument(
            instrument, self._curve, self._settlement, greeks=spec,
        )
        return result.greeks

    # ------------------------------------------------------------------
    # Scenario methods (return new Session)
    # ------------------------------------------------------------------

    def _clone(self, **overrides) -> Session:
        new = object.__new__(Session)
        for slot in self.__slots__:
            key = slot.lstrip("_")
            val = overrides.get(key, getattr(self, slot))
            object.__setattr__(new, slot, val)
        return new

    def with_curve_shift(self, bps: float) -> Session:
        return self._clone(curve=self._curve.shift(bps))

    def with_tenor_bumps(self, bumps: dict[float, float]) -> Session:
        return self._clone(curve=self._curve.bump(bumps))

    def with_curve(self, curve: YieldCurve) -> Session:
        return self._clone(curve=curve)

    def with_vol_surface(self, vol_surface) -> Session:
        return self._clone(vol_surface=vol_surface)

    def with_state_space(self, state_space) -> Session:
        return self._clone(state_space=state_space)

    def with_credit_curve(self, credit_curve) -> Session:
        return self._clone(credit_curve=credit_curve)

    def with_forecast_curves(self, forecast_curves) -> Session:
        return self._clone(forecast_curves=forecast_curves)

    def with_fx_rates(self, fx_rates) -> Session:
        return self._clone(fx_rates=fx_rates)

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def spread_to_curve(self, instrument: Bond, market_price: float) -> float:
        from scipy.optimize import brentq

        def objective(bps: float) -> float:
            shifted = self._curve.shift(bps)
            result = price_instrument(
                instrument, shifted, self._settlement, greeks=None,
            )
            return result.clean_price - market_price

        return brentq(objective, -500, 500, xtol=1e-6)

    def risk_report(self, book: Book) -> dict:
        br = self._price_book(book, greeks="all")
        agg_krd: dict[str, float] = {}
        tmv = br.total_mv
        positions = {}
        for name in br:
            r = br[name]
            ntl = book.notional(name)
            mv = r.dirty_price * ntl
            positions[name] = {
                "dirty_price": r.dirty_price,
                "clean_price": r.clean_price,
                "notional": ntl,
                "mv": mv,
                "dv01": r.greeks.get("dv01", 0.0),
                "duration": r.greeks.get("duration", 0.0),
            }
            krd = r.greeks.get("key_rate_durations", {})
            for k, v in krd.items():
                weight = mv / tmv if tmv else 0.0
                agg_krd[k] = agg_krd.get(k, 0.0) + v * weight
        return {
            "total_mv": br.total_mv,
            "book_dv01": br.book_dv01,
            "book_duration": br.book_duration,
            "book_krd": agg_krd,
            "positions": positions,
        }
