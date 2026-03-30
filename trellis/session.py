"""Session: immutable market snapshot for interactive pricing."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from trellis.book import Book, BookResult
from trellis.core.market_state import MarketState
from trellis.core.payoff import Payoff
from trellis.core.types import DayCountConvention, GreeksSpec, PricingResult
from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot
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
        "_market_snapshot", "_discount_curve_name", "_vol_surface_name",
    )

    def __init__(
        self,
        curve: YieldCurve | None = None,
        settlement: date | None = None,
        *,
        as_of: date | str | None = None,
        data_source: str = "treasury_gov",
        agent: bool = False,
        market_snapshot: MarketSnapshot | None = None,
        discount_curve: str | None = None,
        vol_surface_name: str | None = None,
        vol_surface=None,
        state_space=None,
        credit_curve=None,
        forecast_curves=None,
        fx_rates=None,
    ):
        """Create an immutable pricing session from explicit or resolved data.

        The session is the main user-facing state object in Trellis. It wraps
        the discount curve plus optional volatility, credit, forecast, and FX
        inputs into a reusable pricing context. If ``curve`` is omitted, the
        constructor resolves a :class:`MarketSnapshot` from the requested data
        source and extracts the default named components from it.

        Typical usage::

            s = Session(curve=YieldCurve.flat(0.045), settlement=date(2024, 11, 15))
            price = s.price(trellis.sample_bond_10y()).clean_price

            s = Session(data_source="mock")
            result = s.ask("Price a 5Y SOFR cap at 4% on $10M")
        """
        if market_snapshot is not None and curve is not None:
            raise ValueError("Pass either curve= or market_snapshot=, not both")

        if market_snapshot is None and curve is None:
            from trellis.data.resolver import resolve_market_snapshot

            market_snapshot = resolve_market_snapshot(
                as_of=as_of,
                source=data_source,
                vol_surface=vol_surface,
                vol_surfaces=None,
                default_vol_surface=None,
                forecast_curves=forecast_curves,
                credit_curve=credit_curve,
                fx_rates=fx_rates,
                metadata=None,
            )

        if market_snapshot is not None:
            curve = market_snapshot.discount_curve(discount_curve)
            if vol_surface is None:
                vol_surface = market_snapshot.vol_surface(vol_surface_name)
            if credit_curve is None:
                credit_curve = market_snapshot.credit_curve()
            if forecast_curves is None:
                forecast_curves = dict(market_snapshot.forecast_curves) or None
            if fx_rates is None:
                fx_rates = dict(market_snapshot.fx_rates) or None
            discount_curve = discount_curve or market_snapshot.default_discount_curve
            vol_surface_name = vol_surface_name or market_snapshot.default_vol_surface
        else:
            market_snapshot = MarketSnapshot(
                as_of=_resolve_as_of(as_of, settlement),
                source="explicit",
                discount_curves={"discount": curve},
                forecast_curves=forecast_curves or {},
                vol_surfaces={"default": vol_surface} if vol_surface is not None else {},
                credit_curves={"default": credit_curve} if credit_curve is not None else {},
                fx_rates=fx_rates or {},
                default_discount_curve="discount",
                default_vol_surface="default" if vol_surface is not None else None,
                default_credit_curve="default" if credit_curve is not None else None,
                provenance={
                    "source": "explicit",
                    "source_kind": "explicit_input",
                    "source_ref": "Session(curve=...)",
                    "as_of": _resolve_as_of(as_of, settlement).isoformat(),
                },
            )
            discount_curve = "discount"
            vol_surface_name = "default" if vol_surface is not None else None

        object.__setattr__(self, "_curve", curve)
        object.__setattr__(self, "_settlement", settlement or date.today())
        object.__setattr__(self, "_agent", agent)
        object.__setattr__(self, "_vol_surface", vol_surface)
        object.__setattr__(self, "_state_space", state_space)
        object.__setattr__(self, "_credit_curve", credit_curve)
        object.__setattr__(self, "_forecast_curves", forecast_curves)
        object.__setattr__(self, "_fx_rates", fx_rates)
        object.__setattr__(self, "_market_snapshot", market_snapshot)
        object.__setattr__(self, "_discount_curve_name", discount_curve)
        object.__setattr__(self, "_vol_surface_name", vol_surface_name)

    def __setattr__(self, name, value):
        """Reject in-place mutation and preserve session immutability."""
        raise AttributeError("Session is immutable")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def curve(self) -> YieldCurve:
        """Primary discount curve used for direct instrument pricing."""
        return self._curve

    @property
    def settlement(self) -> date:
        """Cash settlement date used for pricing and time-to-maturity math."""
        return self._settlement

    @property
    def agent_enabled(self) -> bool:
        """Whether unsupported direct pricing falls back to the agent path."""
        return self._agent

    @property
    def market_snapshot(self) -> MarketSnapshot | None:
        """Resolved named market snapshot backing this session, if any."""
        return self._market_snapshot

    @property
    def discount_curve_name(self) -> str | None:
        """Name of the active discount curve inside the backing snapshot."""
        return self._discount_curve_name

    @property
    def vol_surface_name(self) -> str | None:
        """Name of the active volatility surface inside the backing snapshot."""
        return self._vol_surface_name

    @property
    def vol_surface(self):
        """Volatility surface used by Black, tree, MC, or PDE-style payoffs."""
        return self._vol_surface

    @property
    def state_space(self):
        """Discrete state space for scenario-weighted or finite-state payoffs."""
        return self._state_space

    @property
    def credit_curve(self):
        """Credit or survival curve carried by the session, if configured."""
        return self._credit_curve

    @property
    def forecast_curves(self):
        """Named forecast curves keyed by rate index or foreign discount key."""
        return self._forecast_curves

    @property
    def fx_rates(self):
        """Spot FX rates keyed by pair, such as ``EURUSD``."""
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
        compiled_request = None
        try:
            compiled_request = self.to_platform_request(
                instrument if not isinstance(instrument, Book) else None,
                book=instrument if isinstance(instrument, Book) else None,
                request_type="price",
                measures=["price"] if greeks is None else ["price", *([*greeks] if isinstance(greeks, list) else ([] if greeks == "all" else []))],
            )
            from trellis.agent.platform_requests import compile_platform_request
            compiled_request = compile_platform_request(compiled_request)
            self._append_platform_event(
                compiled_request,
                "request_compiled",
                status="ok",
                details={
                    "action": compiled_request.execution_plan.action,
                    "route_method": compiled_request.execution_plan.route_method,
                    "requires_build": compiled_request.execution_plan.requires_build,
                },
            )
        except Exception:
            compiled_request = None

        if isinstance(instrument, Book):
            try:
                result = self._price_book(instrument, greeks=greeks)
                self._record_platform_trace(compiled_request, success=True, outcome="priced")
                return result
            except Exception as exc:
                self._record_platform_trace(
                    compiled_request,
                    success=False,
                    outcome="price_failed",
                    details={"error_type": type(exc).__name__, "error": str(exc)},
                )
                raise
        try:
            result = price_instrument(
                instrument, self._curve, self._settlement, greeks=greeks,
            )
            self._record_platform_trace(compiled_request, success=True, outcome="priced")
            return result
        except (NotImplementedError, TypeError) as exc:
            if self._agent:
                try:
                    result = self._agent_price(instrument)
                    self._record_platform_trace(compiled_request, success=True, outcome="agent_priced")
                    return result
                except Exception as agent_exc:
                    self._record_platform_trace(
                        compiled_request,
                        success=False,
                        outcome="price_failed",
                        details={
                            "error_type": type(agent_exc).__name__,
                            "error": str(agent_exc),
                        },
                    )
                    raise
            self._record_platform_trace(
                compiled_request,
                success=False,
                outcome="price_failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        except Exception as exc:
            self._record_platform_trace(
                compiled_request,
                success=False,
                outcome="price_failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise

    def _price_book(self, book: Book, *, greeks: GreeksSpec = "all") -> BookResult:
        """Price each position in a book under the current session state."""
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

    def ask(self, description: str, measures: list | None = None,
            model: str | None = None):
        """Price (and optionally analyze) an instrument described in natural language.

        Examples::

            # Just price
            result = s.ask("Price a 5Y cap at 4% on $10M SOFR")
            print(result.price)

            # Price + analytics
            result = s.ask(
                "Price a callable bond ...",
                measures=["price", "dv01", "vega", {"oas": {"market_price": 95.0}}],
            )
            print(result.analytics.oas)
        """
        from trellis.agent.ask import ask_session
        return ask_session(description, self, measures=measures, model=model)

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
        """Delegate unsupported direct pricing to the agent executor."""
        from trellis.agent.executor import execute
        return execute(f"Price instrument: {instrument!r}")

    def greeks(
        self,
        instrument: Bond,
        *,
        measures: list[str] | None = None,
    ) -> dict:
        """Compute Greeks for an instrument."""
        compiled_request = None
        try:
            compiled_request = self.to_platform_request(
                instrument,
                request_type="greeks",
                measures=measures or ["dv01"],
            )
            from trellis.agent.platform_requests import compile_platform_request
            compiled_request = compile_platform_request(compiled_request)
            self._append_platform_event(
                compiled_request,
                "request_compiled",
                status="ok",
                details={
                    "action": compiled_request.execution_plan.action,
                    "route_method": compiled_request.execution_plan.route_method,
                },
            )
        except Exception:
            compiled_request = None
        spec = list(measures) if measures else "all"
        try:
            result = price_instrument(
                instrument, self._curve, self._settlement, greeks=spec,
            )
            self._record_platform_trace(compiled_request, success=True, outcome="greeks_computed")
            return result.greeks
        except Exception as exc:
            self._record_platform_trace(
                compiled_request,
                success=False,
                outcome="greeks_failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise

    # ------------------------------------------------------------------
    # Scenario methods (return new Session)
    # ------------------------------------------------------------------

    def _clone(self, **overrides) -> Session:
        """Create a shallow immutable copy, replacing only selected slots."""
        new = object.__new__(Session)
        for slot in self.__slots__:
            key = slot.lstrip("_")
            val = overrides.get(key, getattr(self, slot))
            object.__setattr__(new, slot, val)
        return new

    def with_curve_shift(self, bps: float) -> Session:
        """Return a new session with a parallel curve shift in basis points."""
        return self.with_curve(self._curve.shift(bps))

    def with_tenor_bumps(self, bumps: dict[float, float]) -> Session:
        """Return a new session with individual tenor bumps in basis points."""
        return self.with_curve(self._curve.bump(bumps))

    def with_curve(self, curve: YieldCurve) -> Session:
        """Return a new session with the primary discount curve replaced."""
        snapshot = self._replace_snapshot_discount_curve(curve)
        return self._clone(curve=curve, market_snapshot=snapshot)

    def with_discount_curve(self, name: str) -> Session:
        """Switch to another named discount curve from the backing snapshot."""
        if self._market_snapshot is None:
            raise ValueError("No market snapshot available for named curve selection")
        curve = self._market_snapshot.discount_curve(name)
        return self._clone(curve=curve, discount_curve_name=name)

    def with_vol_surface_name(self, name: str) -> Session:
        """Switch to another named volatility surface from the backing snapshot."""
        if self._market_snapshot is None:
            raise ValueError("No market snapshot available for named vol surface selection")
        vol_surface = self._market_snapshot.vol_surface(name)
        return self._clone(vol_surface=vol_surface, vol_surface_name=name)

    def with_vol_surface(self, vol_surface) -> Session:
        """Return a new session with an explicit volatility surface override."""
        return self._clone(
            vol_surface=vol_surface,
            market_snapshot=self._replace_snapshot_vol_surface(vol_surface),
        )

    def with_state_space(self, state_space) -> Session:
        """Return a new session with a replacement discrete state space."""
        return self._clone(state_space=state_space)

    def with_credit_curve(self, credit_curve) -> Session:
        """Return a new session with a replacement credit curve."""
        return self._clone(
            credit_curve=credit_curve,
            market_snapshot=self._replace_snapshot_credit_curve(credit_curve),
        )

    def with_forecast_curves(self, forecast_curves) -> Session:
        """Return a new session with replacement named forecast curves."""
        return self._clone(
            forecast_curves=forecast_curves,
            market_snapshot=self._replace_snapshot_forecast_curves(forecast_curves),
        )

    def with_fx_rates(self, fx_rates) -> Session:
        """Return a new session with replacement FX spot rates."""
        return self._clone(
            fx_rates=fx_rates,
            market_snapshot=self._replace_snapshot_fx_rates(fx_rates),
        )

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def spread_to_curve(self, instrument: Bond, market_price: float) -> float:
        """Solve for the parallel spread that reproduces an observed market price.

        This computes the z-spread style shift ``s`` such that repricing the
        instrument on ``curve.shift(s)`` matches ``market_price``. The root is
        solved in basis points over ``[-500, 500]``.
        """
        from scipy.optimize import brentq

        def objective(bps: float) -> float:
            """Return the pricing error after applying a parallel spread in basis points."""
            shifted = self._curve.shift(bps)
            result = price_instrument(
                instrument, shifted, self._settlement, greeks=None,
            )
            return result.clean_price - market_price

        return brentq(objective, -500, 500, xtol=1e-6)

    def analyze(
        self,
        instrument,
        measures: list | None = None,
        **kwargs,
    ):
        """Compute analytics for any instrument (Payoff or Book).

        Parameters
        ----------
        instrument : Payoff or Book
            Single instrument or a book of instruments.
        measures : list
            What to compute. Each element can be:
            - str: "price", "dv01", "vega" → use defaults
            - dict: {"oas": {"market_price": 95.0}} → parameterized
            - Measure object: OAS(market_price=95.0) → full control
            If None, defaults to ["price", "dv01", "duration"].
        **kwargs
            Extra context passed to all measures (e.g., market_price for OAS).

        Returns
        -------
        AnalyticsResult or BookAnalyticsResult
        """
        from trellis.analytics.measures import resolve_measures
        from trellis.analytics.result import AnalyticsResult, BookAnalyticsResult

        if measures is None:
            measures = ["price", "dv01", "duration"]

        resolved = resolve_measures(measures)

        compiled_request = None
        try:
            compiled_request = self.to_platform_request(
                instrument if not isinstance(instrument, Book) else None,
                book=instrument if isinstance(instrument, Book) else None,
                request_type="analytics",
                measures=measures,
            )
            from trellis.agent.platform_requests import compile_platform_request
            compiled_request = compile_platform_request(compiled_request)
            self._append_platform_event(
                compiled_request,
                "request_compiled",
                status="ok",
                details={
                    "action": compiled_request.execution_plan.action,
                    "route_method": compiled_request.execution_plan.route_method,
                },
            )
        except Exception:
            compiled_request = None

        try:
            if isinstance(instrument, Book):
                result = self._analyze_book(instrument, resolved, **kwargs)
                self._record_platform_trace(compiled_request, success=True, outcome="analytics_computed")
                return result

            # Single payoff
            ms = self._build_market_state()
            ctx = dict(kwargs)
            ctx.setdefault("_cache", {})
            data = {}
            for m in resolved:
                data[m.name] = m.compute(instrument, ms, **ctx)
            result = AnalyticsResult(data)
            self._record_platform_trace(compiled_request, success=True, outcome="analytics_computed")
            return result
        except Exception as exc:
            self._record_platform_trace(
                compiled_request,
                success=False,
                outcome="analytics_failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise

    def _analyze_book(self, book, measures, **kwargs):
        """Compute the requested analytics measures for every book position."""
        from trellis.analytics.result import AnalyticsResult, BookAnalyticsResult

        ms = self._build_market_state()
        positions = {}
        for name in book:
            payoff = book[name]
            ctx = dict(kwargs)
            ctx.setdefault("_cache", {})
            data = {}
            for m in measures:
                data[m.name] = m.compute(payoff, ms, **ctx)
            positions[name] = AnalyticsResult(data)

        notionals = {name: book.notional(name) for name in book}
        return BookAnalyticsResult(positions, notionals)

    def _build_market_state(self) -> MarketState:
        """Materialize the session's pricing inputs as a ``MarketState``."""
        return MarketState(
            as_of=self._market_snapshot.as_of if self._market_snapshot is not None else self._settlement,
            settlement=self._settlement,
            discount=self._curve,
            vol_surface=self._vol_surface,
            state_space=self._state_space,
            credit_curve=self._credit_curve,
            forecast_curves=self._forecast_curves,
            fx_rates=self._fx_rates,
        )

    def _replace_snapshot_discount_curve(self, curve: YieldCurve):
        """Clone the backing snapshot with an updated active discount curve."""
        if self._market_snapshot is None:
            return None
        name = self._discount_curve_name or self._market_snapshot.default_discount_curve or "discount"
        curves = dict(self._market_snapshot.discount_curves)
        curves[name] = curve
        return replace(self._market_snapshot, discount_curves=curves)

    def _replace_snapshot_vol_surface(self, vol_surface):
        """Clone the backing snapshot with an updated default vol surface."""
        if self._market_snapshot is None:
            return None
        surfaces = dict(self._market_snapshot.vol_surfaces)
        default_name = self._vol_surface_name or self._market_snapshot.default_vol_surface or "default"
        if vol_surface is None:
            surfaces = {}
            default_name = None
        else:
            surfaces[default_name] = vol_surface
        return replace(
            self._market_snapshot,
            vol_surfaces=surfaces,
            default_vol_surface=default_name,
        )

    def _replace_snapshot_credit_curve(self, credit_curve):
        """Clone the backing snapshot with an updated default credit curve."""
        if self._market_snapshot is None:
            return None
        curves = dict(self._market_snapshot.credit_curves)
        default_name = self._market_snapshot.default_credit_curve or "default"
        if credit_curve is None:
            curves = {}
            default_name = None
        else:
            curves[default_name] = credit_curve
        return replace(
            self._market_snapshot,
            credit_curves=curves,
            default_credit_curve=default_name,
        )

    def _replace_snapshot_forecast_curves(self, forecast_curves):
        """Clone the backing snapshot with replacement forecast curves."""
        if self._market_snapshot is None:
            return None
        return replace(
            self._market_snapshot,
            forecast_curves=dict(forecast_curves or {}),
        )

    def _replace_snapshot_fx_rates(self, fx_rates):
        """Clone the backing snapshot with replacement FX spot data."""
        if self._market_snapshot is None:
            return None
        return replace(
            self._market_snapshot,
            fx_rates=dict(fx_rates or {}),
        )

    def oas(self, payoff, market_price: float) -> float:
        """Compute Option-Adjusted Spread for a Payoff (callable bond, etc.).

        OAS is the constant spread (in bps) over the treasury curve such that
        repricing the instrument on the shifted curve equals the market price.

        Parameters
        ----------
        payoff : Payoff
            Must implement ``evaluate(market_state) -> float``.
        market_price : float
            Observed market price.

        Returns
        -------
        float
            OAS in basis points.
        """
        from trellis.analytics.oas import compute_oas
        return compute_oas(
            payoff, market_price, self._curve, self._settlement,
            vol_surface=self._vol_surface,
        )

    def risk_report(self, book: Book) -> dict:
        """Return position- and portfolio-level risk aggregates for a book.

        The report combines market value, DV01, duration, and aggregated key
        rate durations. It is intended as a lightweight portfolio summary for
        notebook or API use rather than a full reporting engine.
        """
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

    def to_platform_request(
        self,
        instrument=None,
        *,
        book=None,
        request_type: str = "price",
        measures: list | None = None,
        description: str | None = None,
        model: str | None = None,
    ):
        """Compile the current session context into a canonical platform request.

        Platform requests normalize session, book, and measure metadata into
        the agent-facing request model used by tracing, blocking, comparison,
        and build orchestration layers.
        """
        from trellis.agent.platform_requests import make_session_request

        return make_session_request(
            self,
            instrument=instrument,
            book=book,
            request_type=request_type,
            measures=measures,
            description=description,
            model=model,
        )

    @staticmethod
    def _append_platform_event(
        compiled_request,
        event: str,
        *,
        status: str = "info",
        details: dict | None = None,
    ) -> None:
        """Best-effort helper for appending a trace event to a compiled request."""
        if compiled_request is None:
            return
        try:
            from trellis.agent.platform_traces import append_platform_trace_event

            append_platform_trace_event(
                compiled_request,
                event,
                status=status,
                details=details,
            )
        except Exception:
            pass

    @staticmethod
    def _record_platform_trace(
        compiled_request,
        *,
        success: bool,
        outcome: str,
        details: dict | None = None,
    ) -> None:
        """Best-effort helper for writing terminal trace state for a request."""
        if compiled_request is None:
            return
        try:
            from trellis.agent.platform_traces import record_platform_trace

            record_platform_trace(
                compiled_request,
                success=success,
                outcome=outcome,
                details=details,
            )
        except Exception:
            pass


def _resolve_as_of(as_of: date | str | None, settlement: date | None) -> date:
    """Resolve an as-of date for explicit component snapshots."""
    if as_of is None:
        return settlement or date.today()
    if isinstance(as_of, str):
        if as_of == "latest":
            return date.today()
        return date.fromisoformat(as_of)
    return as_of
