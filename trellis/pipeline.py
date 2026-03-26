"""Pipeline: declarative batch pricing."""

from __future__ import annotations

from datetime import date

from trellis.book import Book, BookResult
from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot
from trellis.session import Session


class Pipeline:
    """Declarative, chainable batch-pricing pipeline.

    Example::

        results = (
            Pipeline()
            .instruments_from_csv("portfolio.csv")
            .market_data(source="treasury_gov", as_of="2024-11-15")
            .compute(["price", "dv01", "duration"])
            .scenarios([{"shift_bps": 0}, {"shift_bps": +100}])
            .run()
        )
    """

    def __init__(self):
        """Initialize an empty pipeline with no instruments or scenarios yet."""
        self._book: Book | None = None
        self._curve: YieldCurve | None = None
        self._market_snapshot: MarketSnapshot | None = None
        self._discount_curve_name: str | None = None
        self._vol_surface_name: str | None = None
        self._data_source: str = "treasury_gov"
        self._as_of: date | str | None = None
        self._measures: list[str] | None = None
        self._scenarios: list[dict] | None = None
        self._outputs: list[dict] = []

    # ------------------------------------------------------------------
    # Builder methods (return self for chaining)
    # ------------------------------------------------------------------

    def instruments_from_csv(self, path: str) -> Pipeline:
        """Load a bond book from CSV and return the pipeline for chaining."""
        self._book = Book.from_csv(path)
        return self

    def instruments_from_dataframe(self, df) -> Pipeline:
        """Load a bond book from a pandas DataFrame and return ``self``."""
        self._book = Book.from_dataframe(df)
        return self

    def instruments(self, book: Book) -> Pipeline:
        """Attach an already-constructed book to the pipeline."""
        self._book = book
        return self

    def market_data(
        self,
        source: str = "treasury_gov",
        as_of: date | str | None = "latest",
        curve: YieldCurve | None = None,
        snapshot: MarketSnapshot | None = None,
        discount_curve: str | None = None,
        vol_surface_name: str | None = None,
    ) -> Pipeline:
        """Configure the base market data source or explicit snapshot.

        Use ``curve=`` for a single explicit discount curve or ``snapshot=``
        for a richer named market snapshot with discount, forecast, vol,
        credit, and FX components.
        """
        if snapshot is not None and curve is not None:
            raise ValueError("Pass either curve= or snapshot=, not both")
        self._data_source = source
        self._as_of = as_of
        self._curve = curve
        self._market_snapshot = snapshot
        self._discount_curve_name = discount_curve
        self._vol_surface_name = vol_surface_name
        return self

    def compute(self, measures: list[str]) -> Pipeline:
        """Choose the measures to compute for each scenario run."""
        self._measures = measures
        return self

    def scenarios(self, specs: list[dict]) -> Pipeline:
        """Register scenario specifications such as parallel shifts or tenor bumps."""
        self._scenarios = specs
        return self

    def output_parquet(self, path_template: str) -> Pipeline:
        """Write each scenario result to a parquet path template."""
        self._outputs.append({"format": "parquet", "path": path_template})
        return self

    def output_csv(self, path_template: str) -> Pipeline:
        """Write each scenario result to a CSV path template."""
        self._outputs.append({"format": "csv", "path": path_template})
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def compile_request(self):
        """Compile the pipeline input into the canonical platform request."""
        if self._book is None:
            raise ValueError("No instruments configured — cannot compile request")

        base = Session(
            curve=self._curve,
            market_snapshot=self._market_snapshot,
            discount_curve=self._discount_curve_name,
            vol_surface_name=self._vol_surface_name,
            settlement=None,
            as_of=self._as_of,
            data_source=self._data_source,
        )

        from trellis.agent.platform_requests import make_pipeline_request

        return make_pipeline_request(
            book=self._book,
            market_snapshot=base.market_snapshot,
            settlement=base.settlement,
            measures=self._measures or ["price"],
            metadata={
                "scenario_count": len(self._scenarios or [{"name": "base"}]),
                "data_source": self._data_source,
            },
        )

    def run(self) -> dict[str, BookResult]:
        """Execute the pipeline and return ``{scenario_name: BookResult}``."""
        if self._book is None:
            raise ValueError("No instruments configured — call instruments(), "
                             "instruments_from_csv(), or instruments_from_dataframe()")

        # Build base session
        base = Session(
            curve=self._curve,
            market_snapshot=self._market_snapshot,
            discount_curve=self._discount_curve_name,
            vol_surface_name=self._vol_surface_name,
            settlement=None,
            as_of=self._as_of,
            data_source=self._data_source,
        )
        compiled_request = None
        terminal_recorded = False
        try:
            from trellis.agent.platform_requests import compile_platform_request
            from trellis.agent.platform_traces import append_platform_trace_event

            compiled_request = compile_platform_request(self.compile_request())
            append_platform_trace_event(
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

        # Determine greeks spec from measures
        greeks_spec = self._resolve_greeks_spec()

        # Build scenario list
        scenario_list = self._scenarios or [{"name": "base"}]

        try:
            results: dict[str, BookResult] = {}
            for spec in scenario_list:
                name = spec.get("name", self._scenario_name(spec))
                session = self._apply_scenario(base, spec)
                br = session.price(self._book, greeks=greeks_spec)
                results[name] = br

                # Write outputs
                for out in self._outputs:
                    self._write_output(out, name, br)

            try:
                if compiled_request is not None:
                    from trellis.agent.platform_traces import record_platform_trace

                    record_platform_trace(
                        compiled_request,
                        success=True,
                        outcome="pipeline_priced",
                        details={"scenario_names": list(results.keys())},
                    )
                    terminal_recorded = True
            except Exception:
                pass

            return results
        except Exception as exc:
            if compiled_request is not None and not terminal_recorded:
                try:
                    from trellis.agent.platform_traces import record_platform_trace

                    record_platform_trace(
                        compiled_request,
                        success=False,
                        outcome="pipeline_failed",
                        details={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                except Exception:
                    pass
            raise

    def _resolve_greeks_spec(self):
        """Translate requested measures into the greeks spec used by pricing."""
        if self._measures is None:
            return "all"
        greek_names = [m for m in self._measures if m != "price"]
        return greek_names if greek_names else None

    @staticmethod
    def _scenario_name(spec: dict) -> str:
        """Derive a readable scenario name from a scenario specification."""
        if "shift_bps" in spec:
            return f"shift_{spec['shift_bps']:+d}bp"
        if "tenor_bumps" in spec:
            return "tenor_bump"
        return "base"

    @staticmethod
    def _apply_scenario(base: Session, spec: dict) -> Session:
        """Apply a scenario spec to a base session and return the shocked session."""
        session = base
        if "shift_bps" in spec:
            session = session.with_curve_shift(spec["shift_bps"])
        if "tenor_bumps" in spec:
            session = session.with_tenor_bumps(spec["tenor_bumps"])
        return session

    def _write_output(self, out: dict, scenario_name: str, br: BookResult):
        """Persist a scenario result using the configured output formatter."""
        path = out["path"].format(
            date=date.today().isoformat(),
            scenario=scenario_name,
        )
        df = br.to_dataframe()
        if out["format"] == "parquet":
            df.to_parquet(path, index=False)
        elif out["format"] == "csv":
            df.to_csv(path, index=False)
