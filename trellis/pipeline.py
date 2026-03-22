"""Pipeline: declarative batch pricing."""

from __future__ import annotations

from datetime import date

from trellis.book import Book, BookResult
from trellis.curves.yield_curve import YieldCurve
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
        self._book: Book | None = None
        self._curve: YieldCurve | None = None
        self._data_source: str = "treasury_gov"
        self._as_of: date | str | None = None
        self._measures: list[str] | None = None
        self._scenarios: list[dict] | None = None
        self._outputs: list[dict] = []

    # ------------------------------------------------------------------
    # Builder methods (return self for chaining)
    # ------------------------------------------------------------------

    def instruments_from_csv(self, path: str) -> Pipeline:
        self._book = Book.from_csv(path)
        return self

    def instruments_from_dataframe(self, df) -> Pipeline:
        self._book = Book.from_dataframe(df)
        return self

    def instruments(self, book: Book) -> Pipeline:
        self._book = book
        return self

    def market_data(
        self,
        source: str = "treasury_gov",
        as_of: date | str | None = "latest",
        curve: YieldCurve | None = None,
    ) -> Pipeline:
        self._data_source = source
        self._as_of = as_of
        self._curve = curve
        return self

    def compute(self, measures: list[str]) -> Pipeline:
        self._measures = measures
        return self

    def scenarios(self, specs: list[dict]) -> Pipeline:
        self._scenarios = specs
        return self

    def output_parquet(self, path_template: str) -> Pipeline:
        self._outputs.append({"format": "parquet", "path": path_template})
        return self

    def output_csv(self, path_template: str) -> Pipeline:
        self._outputs.append({"format": "csv", "path": path_template})
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self) -> dict[str, BookResult]:
        """Execute the pipeline and return ``{scenario_name: BookResult}``."""
        if self._book is None:
            raise ValueError("No instruments configured — call instruments(), "
                             "instruments_from_csv(), or instruments_from_dataframe()")

        # Build base session
        base = Session(
            curve=self._curve,
            settlement=None,
            as_of=self._as_of,
            data_source=self._data_source,
        )

        # Determine greeks spec from measures
        greeks_spec = self._resolve_greeks_spec()

        # Build scenario list
        scenario_list = self._scenarios or [{"name": "base"}]

        results: dict[str, BookResult] = {}
        for spec in scenario_list:
            name = spec.get("name", self._scenario_name(spec))
            session = self._apply_scenario(base, spec)
            br = session.price(self._book, greeks=greeks_spec)
            results[name] = br

            # Write outputs
            for out in self._outputs:
                self._write_output(out, name, br)

        return results

    def _resolve_greeks_spec(self):
        if self._measures is None:
            return "all"
        greek_names = [m for m in self._measures if m != "price"]
        return greek_names if greek_names else None

    @staticmethod
    def _scenario_name(spec: dict) -> str:
        if "shift_bps" in spec:
            return f"shift_{spec['shift_bps']:+d}bp"
        if "tenor_bumps" in spec:
            return "tenor_bump"
        return "base"

    @staticmethod
    def _apply_scenario(base: Session, spec: dict) -> Session:
        session = base
        if "shift_bps" in spec:
            session = session.with_curve_shift(spec["shift_bps"])
        if "tenor_bumps" in spec:
            session = session.with_tenor_bumps(spec["tenor_bumps"])
        return session

    def _write_output(self, out: dict, scenario_name: str, br: BookResult):
        path = out["path"].format(
            date=date.today().isoformat(),
            scenario=scenario_name,
        )
        df = br.to_dataframe()
        if out["format"] == "parquet":
            df.to_parquet(path, index=False)
        elif out["format"] == "csv":
            df.to_csv(path, index=False)
