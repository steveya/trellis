"""Book (instrument collection) and BookResult (aggregated pricing)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from datetime import date
from typing import Any

from trellis.analytics.result import RiskMeasureOutput
from trellis.core.types import Instrument, PricingResult


class Book:
    """A named collection of instruments with optional notionals.

    Parameters
    ----------
    instruments : list or dict
        If a list, names are auto-generated (``"inst_0"``, …).
        If a dict, keys are used as names.
    notionals : dict or None
        Per-name notional amounts. Defaults to 1.0 for each.
    """

    def __init__(
        self,
        instruments: list | dict[str, Instrument],
        notionals: dict[str, float] | None = None,
    ):
        """Store instruments and optional notionals under stable position names."""
        if isinstance(instruments, dict):
            self._instruments = dict(instruments)
        else:
            self._instruments = {
                f"inst_{i}": inst for i, inst in enumerate(instruments)
            }
        self._notionals = dict(notionals) if notionals else {}

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Instrument:
        """Return the instrument stored under ``key``."""
        return self._instruments[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over position names in insertion order."""
        return iter(self._instruments)

    def __len__(self) -> int:
        """Return the number of positions in the book."""
        return len(self._instruments)

    @property
    def names(self) -> list[str]:
        """Return all position names as a list."""
        return list(self._instruments.keys())

    @property
    def instruments(self) -> dict[str, Instrument]:
        """Return a shallow copy of the underlying name-to-instrument mapping."""
        return dict(self._instruments)

    def notional(self, name: str) -> float:
        """Return the stored notional for a position, defaulting to ``1.0``."""
        return self._notionals.get(name, 1.0)

    # ------------------------------------------------------------------
    # Constructors from tabular data
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(cls, df) -> Book:
        """Build a Book from a pandas DataFrame.

        Expected columns: ``face``, ``coupon``, ``maturity_date``, ``frequency``.
        Optional: ``name`` (or index), ``notional``, ``maturity``.
        """
        from trellis.instruments.bond import Bond

        instruments: dict[str, Instrument] = {}
        notionals: dict[str, float] = {}

        for idx, row in df.iterrows():
            name = str(row["name"]) if "name" in df.columns else str(idx)
            mat_date = row["maturity_date"]
            if hasattr(mat_date, "date"):
                mat_date = mat_date.date()

            bond = Bond(
                face=float(row["face"]),
                coupon=float(row["coupon"]),
                maturity_date=mat_date,
                frequency=int(row.get("frequency", 2)),
                maturity=int(row["maturity"]) if "maturity" in df.columns else None,
            )
            instruments[name] = bond
            if "notional" in df.columns:
                notionals[name] = float(row["notional"])

        return cls(instruments, notionals if notionals else None)

    @classmethod
    def from_csv(cls, path: str) -> Book:
        """Build a Book from a CSV file. Delegates to :meth:`from_dataframe`."""
        import pandas as pd

        df = pd.read_csv(path, parse_dates=["maturity_date"])
        return cls.from_dataframe(df)


class BookResult:
    """Aggregated pricing results for a :class:`Book`.

    Parameters
    ----------
    results : dict
        ``{name: PricingResult}`` for every instrument in the book.
    book : Book
        The book that was priced (used for notional weighting).
    """

    def __init__(self, results: dict[str, PricingResult], book: Book):
        """Bind per-position pricing results to the originating book definition."""
        self._results = results
        self._book = book

    def __getitem__(self, key: str) -> PricingResult:
        """Return the pricing result for a named position."""
        return self._results[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over position names with available pricing results."""
        return iter(self._results)

    def __len__(self) -> int:
        """Return the number of priced positions."""
        return len(self._results)

    # ------------------------------------------------------------------
    # Aggregated analytics
    # ------------------------------------------------------------------

    @property
    def total_mv(self) -> float:
        """Total market value: sum(dirty_price * notional)."""
        return sum(
            r.dirty_price * self._book.notional(name)
            for name, r in self._results.items()
        )

    @property
    def book_dv01(self) -> float:
        """Portfolio DV01: sum(dv01 * notional / 100)."""
        total = 0.0
        for name, r in self._results.items():
            dv01 = r.greeks.get("dv01", 0.0)
            total += dv01 * self._book.notional(name) / 100.0
        return total

    @property
    def book_duration(self) -> float:
        """Market-value-weighted average duration."""
        tmv = self.total_mv
        if tmv == 0:
            return 0.0
        weighted = sum(
            r.greeks.get("duration", 0.0)
            * r.dirty_price
            * self._book.notional(name)
            for name, r in self._results.items()
        )
        return weighted / tmv

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """JSON-serializable dict representation."""
        rows = {}
        for name, r in self._results.items():
            row = {
                "clean_price": r.clean_price,
                "dirty_price": r.dirty_price,
                "accrued_interest": r.accrued_interest,
                "notional": self._book.notional(name),
            }
            row.update(r.greeks)
            rows[name] = row
        return {
            "positions": rows,
            "total_mv": self.total_mv,
            "book_dv01": self.book_dv01,
            "book_duration": self.book_duration,
        }

    def to_dataframe(self):
        """Return a pandas DataFrame with one row per position."""
        import pandas as pd

        records = []
        for name, r in self._results.items():
            rec = {
                "name": name,
                "clean_price": r.clean_price,
                "dirty_price": r.dirty_price,
                "accrued_interest": r.accrued_interest,
                "notional": self._book.notional(name),
            }
            for k, v in r.greeks.items():
                if not isinstance(v, dict):
                    rec[k] = v
            records.append(rec)
        return pd.DataFrame(records)


class ScenarioResultCube(Mapping[str, BookResult]):
    """Dict-like scenario result cube with explicit scenario metadata.

    The cube preserves the concrete scenario specification and provenance for
    each scenario while exposing reusable aggregation helpers for later
    attribution and summary workflows.
    """

    def __init__(
        self,
        results: dict[str, BookResult],
        *,
        scenario_specs: dict[str, dict[str, Any]] | None = None,
        scenario_provenance: dict[str, dict[str, Any]] | None = None,
        compute_plan: dict[str, Any] | None = None,
        base_scenario: str | None = None,
        compute_plan_object: object | None = None,
    ):
        self._results = dict(results)
        self._scenario_specs = {
            str(name): deepcopy(spec)
            for name, spec in (scenario_specs or {}).items()
        }
        self._scenario_provenance = {
            str(name): deepcopy(spec)
            for name, spec in (scenario_provenance or {}).items()
        }
        self._compute_plan = deepcopy(compute_plan or {})
        self._base_scenario = base_scenario
        self._compute_plan_object = compute_plan_object
        if self._base_scenario is None and self._results:
            self._base_scenario = "base" if "base" in self._results else next(iter(self._results))
        if self._base_scenario is not None and self._base_scenario not in self._results:
            raise KeyError(
                f"Base scenario {self._base_scenario!r} is not present in the cube."
            )
        for name in self._results:
            self._scenario_specs.setdefault(name, {"name": name})
            self._scenario_provenance.setdefault(name, {})

    def __getitem__(self, key: str) -> BookResult:
        return self._results[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    def __repr__(self) -> str:
        return f"ScenarioResultCube({list(self._results)})"

    @property
    def base_name(self) -> str:
        """Return the baseline scenario name used for deltas."""
        if self._base_scenario is None:
            raise ValueError("ScenarioResultCube is empty and has no base scenario.")
        return self._base_scenario

    @property
    def scenario_specs(self) -> dict[str, dict[str, Any]]:
        """Return a defensive copy of the scenario-spec map."""
        return deepcopy(self._scenario_specs)

    @property
    def scenario_provenance(self) -> dict[str, dict[str, Any]]:
        """Return a defensive copy of the per-scenario provenance map."""
        return deepcopy(self._scenario_provenance)

    @property
    def compute_plan(self) -> dict[str, Any]:
        """Return the serialized compute plan used to produce this cube."""
        return deepcopy(self._compute_plan)

    def book_ladder(
        self,
        metric: str,
        *,
        baseline_scenario: str | None = None,
    ) -> RiskMeasureOutput:
        """Return one aggregated book metric across scenarios plus deltas."""
        baseline = self._resolve_baseline_scenario(baseline_scenario)
        resolved_metric = self._resolve_book_metric_name(metric)
        values = {
            name: self._extract_book_metric(result, resolved_metric)
            for name, result in self._results.items()
        }
        base_value = values[baseline]
        deltas = {
            name: float(value - base_value)
            for name, value in values.items()
        }
        return RiskMeasureOutput(
            values,
            metadata=self._ladder_metadata(
                metric=resolved_metric,
                requested_metric=metric,
                aggregation_level="book",
                baseline_scenario=baseline,
                deltas=deltas,
            ),
        )

    def position_ladder(
        self,
        metric: str,
        *,
        baseline_scenario: str | None = None,
    ) -> RiskMeasureOutput:
        """Return one per-position metric across scenarios plus deltas."""
        baseline = self._resolve_baseline_scenario(baseline_scenario)
        position_names = self._position_names()
        values: dict[str, dict[str, float]] = {name: {} for name in position_names}
        for scenario_name, result in self._results.items():
            for position_name in position_names:
                values[position_name][scenario_name] = self._extract_position_metric(
                    result,
                    position_name,
                    metric,
                )
        deltas = {
            position_name: {
                scenario_name: float(value - values[position_name][baseline])
                for scenario_name, value in scenario_values.items()
            }
            for position_name, scenario_values in values.items()
        }
        return RiskMeasureOutput(
            values,
            metadata=self._ladder_metadata(
                metric=metric,
                requested_metric=metric,
                aggregation_level="position",
                baseline_scenario=baseline,
                deltas=deltas,
            ),
        )

    def book_pnl(self, *, baseline_scenario: str | None = None) -> RiskMeasureOutput:
        """Return book-level market-value deltas across scenarios."""
        ladder = self.book_ladder("total_mv", baseline_scenario=baseline_scenario)
        metadata = dict(ladder.metadata)
        metadata["levels"] = dict(ladder)
        return RiskMeasureOutput(
            dict(metadata["deltas"]),
            metadata=metadata,
        )

    def position_pnl(self, *, baseline_scenario: str | None = None) -> RiskMeasureOutput:
        """Return per-position market-value deltas across scenarios."""
        ladder = self.position_ladder("mv", baseline_scenario=baseline_scenario)
        metadata = dict(ladder.metadata)
        metadata["levels"] = dict(ladder)
        return RiskMeasureOutput(
            dict(metadata["deltas"]),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly nested projection of the scenario cube."""
        return {
            "scenarios": {
                name: result.to_dict()
                for name, result in self._results.items()
            },
            "scenario_specs": self.scenario_specs,
            "scenario_provenance": self.scenario_provenance,
            "compute_plan": self.compute_plan,
        }

    def to_batch_output(
        self,
        *,
        baseline_scenario: str | None = None,
    ) -> dict[str, Any]:
        """Project the cube into a stable batch-review payload."""
        baseline = self._resolve_baseline_scenario(baseline_scenario)
        return {
            "baseline_scenario": baseline,
            "compute_plan": self.compute_plan,
            "scenario_specs": self.scenario_specs,
            "scenario_provenance": self.scenario_provenance,
            "book_pnl": self.book_pnl(baseline_scenario=baseline).to_payload(),
            "position_pnl": self.position_pnl(baseline_scenario=baseline).to_payload(),
            "pnl_attribution": self.pnl_attribution(baseline_scenario=baseline),
        }

    def pnl_attribution(
        self,
        *,
        baseline_scenario: str | None = None,
        top_positions: int = 5,
    ) -> dict[str, Any]:
        """Return a book-level P&L attribution view for scenario review."""
        baseline = self._resolve_baseline_scenario(baseline_scenario)
        resolved_top_positions = max(int(top_positions), 1)
        book_pnl = self.book_pnl(baseline_scenario=baseline)
        position_pnl = self.position_pnl(baseline_scenario=baseline)
        net_position_pnl = {
            position_name: 0.0
            for position_name in position_pnl
        }
        scenario_attribution: dict[str, Any] = {}
        for scenario_name in self._results:
            total_pnl = float(book_pnl.metadata["deltas"][scenario_name])
            contributors: list[dict[str, Any]] = []
            for position_name in position_pnl:
                pnl = float(position_pnl.metadata["deltas"][position_name][scenario_name])
                net_position_pnl[position_name] += pnl
                contributors.append(
                    {
                        "position_name": position_name,
                        "pnl": pnl,
                        "share_of_total": 0.0 if total_pnl == 0.0 else float(pnl / total_pnl),
                    }
                )
            contributors.sort(
                key=lambda item: (-abs(item["pnl"]), item["position_name"])
            )
            top_contributors = contributors[:resolved_top_positions]
            scenario_attribution[scenario_name] = {
                "total_pnl": total_pnl,
                "top_contributors": top_contributors,
                "residual_pnl": float(
                    total_pnl - sum(item["pnl"] for item in top_contributors)
                ),
            }
        return {
            "baseline_scenario": baseline,
            "scenario_order": list(self._results),
            "net_position_pnl": net_position_pnl,
            "scenario_attribution": scenario_attribution,
            "book_pnl": book_pnl.to_payload(),
            "position_pnl": position_pnl.to_payload(),
        }

    def _resolve_baseline_scenario(self, baseline_scenario: str | None) -> str:
        if baseline_scenario is None:
            return self.base_name
        if baseline_scenario not in self._results:
            raise KeyError(
                f"Baseline scenario {baseline_scenario!r} is not present in the cube."
            )
        return baseline_scenario

    @staticmethod
    def _resolve_book_metric_name(metric: str) -> str:
        aliases = {
            "mv": "total_mv",
            "dv01": "book_dv01",
            "duration": "book_duration",
        }
        return aliases.get(metric, metric)

    @staticmethod
    def _extract_book_metric(result: BookResult, metric: str) -> float:
        value = getattr(result, metric, None)
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Book metric {metric!r} is not a scalar aggregated field."
            )
        return float(value)

    def _extract_position_metric(
        self,
        result: BookResult,
        position_name: str,
        metric: str,
    ) -> float:
        pricing_result = result[position_name]
        if metric == "mv":
            return float(pricing_result.dirty_price * result._book.notional(position_name))
        if metric == "notional":
            return float(result._book.notional(position_name))
        if metric in {"clean_price", "dirty_price", "accrued_interest"}:
            return float(getattr(pricing_result, metric))
        greek = pricing_result.greeks.get(metric)
        if isinstance(greek, (int, float)):
            return float(greek)
        raise ValueError(
            f"Position metric {metric!r} is not available as a scalar scenario ladder."
        )

    def _position_names(self) -> tuple[str, ...]:
        if not self._results:
            return ()
        first_name = next(iter(self._results))
        position_names = tuple(self._results[first_name])
        expected = set(position_names)
        for scenario_name, result in self._results.items():
            actual = set(result)
            if actual != expected:
                raise ValueError(
                    "ScenarioResultCube requires identical position names across scenarios; "
                    f"{scenario_name!r} differs from {first_name!r}."
                )
        return position_names

    def _ladder_metadata(
        self,
        *,
        metric: str,
        requested_metric: str,
        aggregation_level: str,
        baseline_scenario: str,
        deltas: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "metric": metric,
            "requested_metric": requested_metric,
            "aggregation_level": aggregation_level,
            "baseline_scenario": baseline_scenario,
            "deltas": deepcopy(deltas),
            "scenario_specs": self.scenario_specs,
            "scenario_provenance": self.scenario_provenance,
        }
