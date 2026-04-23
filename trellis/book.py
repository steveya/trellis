"""Book (instrument collection) and BookResult (aggregated pricing)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from typing import Any

from trellis.analytics.result import RiskMeasureOutput
from trellis.core.differentiable import get_backend_capabilities, get_numpy, vjp
from trellis.core.types import Instrument, PricingResult
from trellis.instruments.bond import Bond
from trellis.curves.yield_curve import YieldCurve

np = get_numpy()


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


def _bond_position_support_report(book: Book) -> tuple[list[tuple[str, Bond]], list[dict[str, Any]]]:
    """Split a book into supported bond positions and explicit exclusions."""
    supported: list[tuple[str, Bond]] = []
    exclusions: list[dict[str, Any]] = []
    for name in book:
        instrument = book[name]
        if not isinstance(instrument, Bond):
            exclusions.append(
                {
                    "position_name": name,
                    "instrument_type": type(instrument).__name__,
                    "reason": "unsupported_instrument_type",
                }
            )
            continue
        if instrument.maturity_date is None:
            exclusions.append(
                {
                    "position_name": name,
                    "instrument_type": type(instrument).__name__,
                    "reason": "bond_maturity_date_required",
                }
            )
            continue
        supported.append((name, instrument))
    return supported, exclusions


def portfolio_aad_curve_risk(
    book: Book,
    curve: YieldCurve,
    settlement: date,
) -> RiskMeasureOutput:
    """Return a reverse-mode curve risk vector for supported bond book positions.

    The returned values are key-rate durations computed from one aggregate book
    value differentiated through the backend VJP surface. Unsupported positions
    are excluded explicitly in the attached metadata.
    """
    supported_positions, exclusions = _bond_position_support_report(book)
    capabilities = get_backend_capabilities()
    tenors = getattr(curve, "tenors", None)
    rates = getattr(curve, "rates", None)
    base_metadata: dict[str, Any] = {
        "resolved_derivative_method": "portfolio_aad_vjp",
        "backend_id": capabilities.backend_id,
        "backend_operator": "vjp",
        "curve_type": type(curve).__name__,
        "curve_tenors": [] if tenors is None else [float(tenor) for tenor in np.asarray(tenors, dtype=float)],
        "supported_position_names": [name for name, _ in supported_positions],
        "unsupported_positions": deepcopy(exclusions),
        "supported_position_count": len(supported_positions),
        "unsupported_position_count": len(exclusions),
        "book_position_count": len(book),
        "support_status": "supported" if supported_positions and not exclusions else (
            "partial" if supported_positions else "unsupported"
        ),
    }

    if tenors is None or rates is None:
        base_metadata["fallback_reason"] = {
            "code": "portfolio_aad_curve_unavailable",
            "message": "The supplied curve does not expose tenors and rates for reverse-mode book differentiation.",
        }
        return RiskMeasureOutput({}, metadata=base_metadata)

    if not supported_positions:
        base_metadata["fallback_reason"] = {
            "code": "portfolio_aad_book_unavailable",
            "message": "No supported bond positions were found in the book.",
        }
        return RiskMeasureOutput({}, metadata=base_metadata)

    curve_cls = type(curve)
    tenors_arr = np.asarray(tenors, dtype=float)
    rates_arr = np.asarray(rates, dtype=float)

    def book_value_from_rates(rates_vec):
        traced_curve = curve_cls(tenors_arr, rates_vec)
        value = np.array(0.0)
        for name, instrument in supported_positions:
            value = value + book.notional(name) * instrument.price(traced_curve, settlement)
        return value

    try:
        book_value, pullback = vjp(book_value_from_rates, rates_arr)
        gradient = np.asarray(pullback(1.0), dtype=float)
    except Exception as exc:
        base_metadata["fallback_reason"] = {
            "code": "portfolio_aad_trace_failed",
            "message": "Reverse-mode differentiation of the aggregate book value failed.",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        return RiskMeasureOutput({}, metadata=base_metadata)

    book_value = float(book_value)
    gradient = np.asarray(gradient, dtype=float)
    book_dv01 = -float(np.sum(gradient)) * 0.0001
    book_duration = 0.0 if book_value == 0.0 else -float(np.sum(gradient)) / book_value
    key_rate_durations: dict[float, float] = {}
    for tenor, sensitivity in zip(tenors_arr, gradient):
        tenor_key = float(tenor)
        key_rate_durations[tenor_key] = 0.0 if book_value == 0.0 else -float(sensitivity) / book_value

    metadata = {
        **base_metadata,
        "book_value": book_value,
        "book_dv01": book_dv01,
        "book_duration": book_duration,
        "gradient": [float(value) for value in gradient],
        "unsupported_positions": deepcopy(exclusions),
        "supported_position_market_value": book_value,
    }
    return RiskMeasureOutput(key_rate_durations, metadata=metadata)


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


@dataclass(frozen=True)
class FutureValueCubeMetadata:
    """Explicit semantic metadata for one trade/date/path future-value cube."""

    measure: str = "risk_neutral"
    numeraire: str = "discount_curve"
    value_semantics: str = "clean_future_value"
    phase_semantics: str = "post_event"
    state_names: tuple[str, ...] = ()
    process_family: str = ""
    compute_plan: dict[str, Any] | None = None
    position_provenance: dict[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "measure", str(self.measure or "risk_neutral"))
        object.__setattr__(self, "numeraire", str(self.numeraire or "discount_curve"))
        object.__setattr__(
            self,
            "value_semantics",
            str(self.value_semantics or "clean_future_value"),
        )
        object.__setattr__(
            self,
            "phase_semantics",
            str(self.phase_semantics or "post_event"),
        )
        object.__setattr__(
            self,
            "state_names",
            tuple(str(name) for name in (self.state_names or ())),
        )
        object.__setattr__(self, "process_family", str(self.process_family or ""))
        object.__setattr__(self, "compute_plan", deepcopy(self.compute_plan or {}))
        object.__setattr__(
            self,
            "position_provenance",
            {
                str(name): deepcopy(payload)
                for name, payload in (self.position_provenance or {}).items()
            },
        )

    @classmethod
    def from_value(
        cls,
        metadata: "FutureValueCubeMetadata | Mapping[str, Any] | None",
    ) -> "FutureValueCubeMetadata":
        """Coerce one metadata payload into the frozen metadata value type."""
        if metadata is None:
            return cls()
        if isinstance(metadata, cls):
            return metadata
        payload = dict(metadata)
        return cls(
            measure=payload.get("measure", "risk_neutral"),
            numeraire=payload.get("numeraire", "discount_curve"),
            value_semantics=payload.get("value_semantics", "clean_future_value"),
            phase_semantics=payload.get("phase_semantics", "post_event"),
            state_names=tuple(payload.get("state_names", ())),
            process_family=payload.get("process_family", ""),
            compute_plan=payload.get("compute_plan"),
            position_provenance=payload.get("position_provenance"),
        )


class FutureValueCube:
    """Trade/date/path valuation tensor with explicit future-value semantics."""

    def __init__(
        self,
        values,
        *,
        position_names: tuple[str, ...] | list[str],
        observation_times: tuple[float, ...] | list[float],
        observation_dates: tuple[date, ...] | list[date] | None = None,
        metadata: FutureValueCubeMetadata | Mapping[str, Any] | None = None,
    ):
        cube = np.asarray(values, dtype=float)
        if cube.ndim != 3:
            raise ValueError(
                "FutureValueCube values must have shape (position, observation_time, path)"
            )
        names = tuple(str(name) for name in position_names)
        if cube.shape[0] != len(names):
            raise ValueError(
                "FutureValueCube position_names must match the trade axis of values"
            )
        times = tuple(float(time) for time in observation_times)
        if cube.shape[1] != len(times):
            raise ValueError(
                "FutureValueCube observation_times must match the date axis of values"
            )
        if any(later < earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("FutureValueCube observation_times must be sorted ascending")
        dates = tuple(observation_dates or ())
        if dates and len(dates) != len(times):
            raise ValueError(
                "FutureValueCube observation_dates must match the observation axis of values"
            )

        self._values = cube.copy()
        self._position_names = names
        self._position_index = {name: index for index, name in enumerate(names)}
        if len(self._position_index) != len(names):
            raise ValueError("FutureValueCube position_names must be unique")
        self._observation_times = times
        self._time_index = {time: index for index, time in enumerate(times)}
        self._observation_dates = dates
        self._date_index = {obs_date: index for index, obs_date in enumerate(dates)}
        self._metadata = FutureValueCubeMetadata.from_value(metadata)

    def __repr__(self) -> str:
        return (
            "FutureValueCube("
            f"positions={list(self._position_names)}, "
            f"observation_times={list(self._observation_times)}, "
            f"n_paths={self.n_paths}"
            ")"
        )

    @property
    def values(self):
        """Return a defensive copy of the raw tensor."""
        return np.asarray(self._values, dtype=float).copy()

    @property
    def metadata(self) -> FutureValueCubeMetadata:
        """Return the frozen cube metadata."""
        return self._metadata

    @property
    def position_names(self) -> tuple[str, ...]:
        return self._position_names

    @property
    def observation_times(self) -> tuple[float, ...]:
        return self._observation_times

    @property
    def observation_dates(self) -> tuple[date, ...]:
        return self._observation_dates

    @property
    def measure(self) -> str:
        return self._metadata.measure

    @property
    def numeraire(self) -> str:
        return self._metadata.numeraire

    @property
    def value_semantics(self) -> str:
        return self._metadata.value_semantics

    @property
    def phase_semantics(self) -> str:
        return self._metadata.phase_semantics

    @property
    def state_names(self) -> tuple[str, ...]:
        return self._metadata.state_names

    @property
    def process_family(self) -> str:
        return self._metadata.process_family

    @property
    def compute_plan(self) -> dict[str, Any]:
        return deepcopy(self._metadata.compute_plan or {})

    @property
    def position_provenance(self) -> dict[str, dict[str, Any]]:
        return deepcopy(self._metadata.position_provenance or {})

    @property
    def n_positions(self) -> int:
        return int(self._values.shape[0])

    @property
    def n_observation_times(self) -> int:
        return int(self._values.shape[1])

    @property
    def n_paths(self) -> int:
        return int(self._values.shape[2])

    def position_index(self, name: str) -> int:
        try:
            return self._position_index[str(name)]
        except KeyError as exc:
            raise KeyError(f"Unknown FutureValueCube position {name!r}") from exc

    def time_index(self, observation_time: float) -> int:
        resolved = float(observation_time)
        try:
            return self._time_index[resolved]
        except KeyError as exc:
            raise KeyError(
                f"Unknown FutureValueCube observation_time {observation_time!r}"
            ) from exc

    def date_index(self, observation_date: date) -> int:
        if not self._observation_dates:
            raise KeyError("FutureValueCube has no observation_dates")
        try:
            return self._date_index[observation_date]
        except KeyError as exc:
            raise KeyError(
                f"Unknown FutureValueCube observation_date {observation_date!r}"
            ) from exc

    def values_for_position(self, name: str):
        """Return the date/path matrix for one named position."""
        return np.asarray(self._values[self.position_index(name)], dtype=float).copy()

    def portfolio_values(self):
        """Return the aggregated portfolio date/path matrix."""
        return np.sum(np.asarray(self._values, dtype=float), axis=0)

    def positive_values(self):
        """Return the positive part of each trade/date/path value."""
        return np.maximum(np.asarray(self._values, dtype=float), 0.0)

    def positive_portfolio_values(self):
        """Return the positive part of the aggregated portfolio date/path matrix."""
        return np.maximum(self.portfolio_values(), 0.0)

    def expected_positive_exposure(self):
        """Return `EE(t_i) = N^{-1} sum_n max(sum_a C_{a,i,n}, 0)`."""
        return np.mean(self.positive_portfolio_values(), axis=1)

    def potential_future_exposure(self, alpha: float):
        """Return `PFE_alpha(t_i)` from the positive portfolio distribution."""
        level = float(alpha)
        if not 0.0 <= level <= 1.0:
            raise ValueError("FutureValueCube alpha must satisfy 0 <= alpha <= 1")
        return np.quantile(self.positive_portfolio_values(), level, axis=1)

    def path_slice(self, path_index: int) -> dict[str, Any]:
        """Return one pathwise date-ladder per position."""
        resolved = int(path_index)
        if resolved < 0 or resolved >= self.n_paths:
            raise IndexError(
                f"FutureValueCube path_index {path_index} is out of bounds for {self.n_paths} paths"
            )
        return {
            name: np.asarray(self._values[index, :, resolved], dtype=float).copy()
            for name, index in self._position_index.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly serialization."""
        return {
            "position_names": list(self.position_names),
            "observation_times": list(self.observation_times),
            "observation_dates": [obs.isoformat() for obs in self.observation_dates],
            "values": np.asarray(self._values, dtype=float).tolist(),
            "metadata": {
                "measure": self.measure,
                "numeraire": self.numeraire,
                "value_semantics": self.value_semantics,
                "phase_semantics": self.phase_semantics,
                "state_names": list(self.state_names),
                "process_family": self.process_family,
                "compute_plan": self.compute_plan,
                "position_provenance": self.position_provenance,
            },
        }
