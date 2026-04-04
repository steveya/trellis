"""Session: immutable market snapshot for interactive pricing."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from uuid import uuid4

from trellis.book import Book, BookResult
from trellis.core.market_state import MarketState
from trellis.core.payoff import Payoff
from trellis.core.types import DayCountConvention, GreeksSpec, PricingResult
from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot
from trellis.engine.payoff_pricer import price_payoff as _price_payoff
from trellis.engine.pricer import price_instrument
from trellis.instruments.bond import Bond


def _execution_result_trace_details_or_empty(result) -> dict[str, object]:
    """Return trace details for one execution result when available."""
    if result is None:
        return {}
    from trellis.platform.results import execution_result_trace_details

    return execution_result_trace_details(result)


def _platform_failure_details(exc: Exception, result) -> dict[str, object]:
    """Build the common trace payload for one governed session failure."""
    return {
        **_execution_result_trace_details_or_empty(result),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


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
        "_session_id",
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
        object.__setattr__(self, "_session_id", f"session_{uuid4().hex[:12]}")

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
    def session_id(self) -> str:
        """Stable identifier for the immutable session state object."""
        return self._session_id

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
        return self._run_governed_request(
            instrument=instrument if not isinstance(instrument, Book) else None,
            book=instrument if isinstance(instrument, Book) else None,
            request_type="price",
            measures=["price"] if greeks is None else ["price", *([*greeks] if isinstance(greeks, list) else ([] if greeks == "all" else []))],
            metadata={"greeks_mode": _pricing_greeks_mode(greeks)},
            success_outcome="priced",
            failure_outcome="price_failed",
        )

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

    def greeks(
        self,
        instrument: Bond,
        *,
        measures: list[str] | None = None,
    ) -> dict:
        """Compute Greeks for an instrument."""
        return self._run_governed_request(
            instrument=instrument,
            request_type="greeks",
            measures=measures or ["dv01"],
            success_outcome="greeks_computed",
            failure_outcome="greeks_failed",
        )

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
        if measures is None:
            measures = ["price", "dv01", "duration"]

        return self._run_governed_request(
            instrument=instrument if not isinstance(instrument, Book) else None,
            book=instrument if isinstance(instrument, Book) else None,
            request_type="analytics",
            measures=measures,
            measure_context=kwargs,
            success_outcome="analytics_computed",
            failure_outcome="analytics_failed",
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
        br = self.price(book, greeks="all")
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
        measure_context: dict | None = None,
        description: str | None = None,
        model: str | None = None,
        metadata: dict | None = None,
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
            measure_context=measure_context,
            description=description,
            model=model,
            metadata=metadata,
        )

    def _compile_platform_request(
        self,
        instrument=None,
        *,
        book=None,
        request_type: str = "price",
        measures: list | None = None,
        measure_context: dict | None = None,
        description: str | None = None,
        model: str | None = None,
        metadata: dict | None = None,
    ):
        """Compile and trace one governed request for a public Session entry point."""
        from trellis.agent.platform_requests import compile_platform_request

        compiled_request = compile_platform_request(
            self.to_platform_request(
                instrument,
                book=book,
                request_type=request_type,
                measures=measures,
                measure_context=measure_context,
                description=description,
                model=model,
                metadata=metadata,
            )
        )
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
        return compiled_request

    def _execute_platform_request(self, compiled_request):
        """Execute one compiled request through the governed platform core."""
        from trellis.platform.executor import execute_compiled_request

        return execute_compiled_request(
            compiled_request,
            self.to_execution_context(),
        )

    def _run_governed_request(
        self,
        instrument=None,
        *,
        book=None,
        request_type: str = "price",
        measures: list | None = None,
        measure_context: dict | None = None,
        description: str | None = None,
        model: str | None = None,
        metadata: dict | None = None,
        success_outcome: str,
        failure_outcome: str,
        default_message: str | None = None,
        success_details: dict | None = None,
    ):
        """Compile, execute, project, and trace one governed Session request."""
        compiled_request = None
        result = None
        try:
            compiled_request = self._compile_platform_request(
                instrument,
                book=book,
                request_type=request_type,
                measures=measures,
                measure_context=measure_context,
                description=description,
                model=model,
                metadata=metadata,
            )
            result = self._execute_platform_request(compiled_request)
            from trellis.platform.results import (
                execution_result_trace_details,
                project_execution_result_value,
            )

            projected = project_execution_result_value(
                result,
                default_message=default_message,
            )
            details = dict(execution_result_trace_details(result))
            if success_details:
                details.update(success_details)
            self._record_platform_trace(
                compiled_request,
                success=True,
                outcome=success_outcome,
                details=details,
            )
            return projected
        except Exception as exc:
            self._record_platform_trace(
                compiled_request,
                success=False,
                outcome=failure_outcome,
                details=_platform_failure_details(exc, result),
            )
            raise

    def to_execution_context(
        self,
        *,
        run_mode=None,
        provider_bindings=None,
        policy_bundle_id: str | None = None,
        allow_mock_data: bool | None = None,
        require_provider_disclosure: bool | None = None,
        default_output_mode: str = "result_only",
        default_audit_mode: str = "summary",
        requested_persistence: str = "ephemeral",
        requested_snapshot_policy: str = "prefer_bound_snapshot",
        metadata: dict | None = None,
    ):
        """Normalize the session's convenience state into explicit governed runtime context."""
        from trellis.platform.context import execution_context_from_session

        return execution_context_from_session(
            self,
            run_mode=run_mode,
            provider_bindings=provider_bindings,
            policy_bundle_id=policy_bundle_id,
            allow_mock_data=allow_mock_data,
            require_provider_disclosure=require_provider_disclosure,
            default_output_mode=default_output_mode,
            default_audit_mode=default_audit_mode,
            requested_persistence=requested_persistence,
            requested_snapshot_policy=requested_snapshot_policy,
            metadata=metadata,
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


def _pricing_greeks_mode(greeks: GreeksSpec) -> str:
    """Return the pricing greeks mode implied by the public Session API."""
    if greeks is None:
        return "none"
    if isinstance(greeks, list):
        return "explicit"
    return "all"
