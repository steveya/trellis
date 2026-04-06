"""Tests for trellis.book — Book and BookResult."""

import json
import tempfile
from datetime import date

import pytest

from trellis.book import Book, BookResult, ScenarioResultCube
from trellis.core.types import PricingResult
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.pricer import price_instrument
from trellis.instruments.bond import Bond


def _make_bond(**kwargs):
    defaults = dict(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                    maturity=10, frequency=2)
    defaults.update(kwargs)
    return Bond(**defaults)


class TestBook:
    """Book construction and accessors."""

    def test_from_dict(self):
        b = Book({"A": _make_bond(), "B": _make_bond(coupon=0.04)})
        assert len(b) == 2
        assert "A" in b.names
        assert "B" in b.names

    def test_from_list(self):
        b = Book([_make_bond(), _make_bond()])
        assert len(b) == 2
        assert b.names == ["inst_0", "inst_1"]

    def test_getitem(self):
        bond = _make_bond()
        b = Book({"X": bond})
        assert b["X"] is bond

    def test_iter(self):
        b = Book({"A": _make_bond(), "B": _make_bond()})
        assert list(b) == ["A", "B"]

    def test_notional_default(self):
        b = Book({"A": _make_bond()})
        assert b.notional("A") == 1.0

    def test_notional_custom(self):
        b = Book({"A": _make_bond()}, notionals={"A": 1_000_000})
        assert b.notional("A") == 1_000_000

    def test_empty_book(self):
        b = Book({})
        assert len(b) == 0
        assert b.names == []

    def test_single_instrument(self):
        b = Book([_make_bond()])
        assert len(b) == 1

    def test_instruments_property(self):
        bond = _make_bond()
        b = Book({"A": bond})
        assert b.instruments == {"A": bond}

    def test_from_dataframe(self):
        import pandas as pd
        df = pd.DataFrame({
            "name": ["10Y"],
            "face": [100.0],
            "coupon": [0.05],
            "maturity_date": [date(2034, 11, 15)],
            "frequency": [2],
            "maturity": [10],
            "notional": [1_000_000.0],
        })
        b = Book.from_dataframe(df)
        assert len(b) == 1
        assert "10Y" in b.names
        assert b.notional("10Y") == 1_000_000.0
        inst = b["10Y"]
        assert inst.coupon_rate == 0.05

    def test_from_csv_roundtrip(self):
        import pandas as pd
        df = pd.DataFrame({
            "name": ["5Y", "10Y"],
            "face": [100.0, 100.0],
            "coupon": [0.04, 0.05],
            "maturity_date": ["2029-11-15", "2034-11-15"],
            "frequency": [2, 2],
            "maturity": [5, 10],
        })
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            df.to_csv(f, index=False)
            path = f.name

        b = Book.from_csv(path)
        assert len(b) == 2
        assert "5Y" in b.names


class TestBookResult:
    """BookResult aggregation and export."""

    def _price_book(self, greeks="all"):
        curve = YieldCurve.flat(0.045)
        bond_a = _make_bond()
        bond_b = _make_bond(coupon=0.04)
        book = Book(
            {"A": bond_a, "B": bond_b},
            notionals={"A": 1_000_000, "B": 500_000},
        )
        results = {}
        for name in book:
            results[name] = price_instrument(
                book[name], curve, date(2024, 11, 15), greeks=greeks,
            )
        return BookResult(results, book), book

    def test_total_mv(self):
        br, _ = self._price_book()
        assert br.total_mv > 0

    def test_book_dv01(self):
        br, _ = self._price_book()
        assert br.book_dv01 > 0

    def test_book_duration(self):
        br, _ = self._price_book()
        assert 0 < br.book_duration < 15

    def test_to_dict_json_serializable(self):
        br, _ = self._price_book()
        d = br.to_dict()
        # Should not raise
        json.dumps(d)
        assert "total_mv" in d
        assert "positions" in d

    def test_to_dataframe_shape(self):
        br, book = self._price_book()
        df = br.to_dataframe()
        assert len(df) == len(book)
        assert "clean_price" in df.columns
        assert "notional" in df.columns

    def test_getitem(self):
        br, _ = self._price_book()
        assert isinstance(br["A"], PricingResult)

    def test_len_and_iter(self):
        br, book = self._price_book()
        assert len(br) == len(book)
        assert set(br) == set(book)

    def test_zero_notional(self):
        curve = YieldCurve.flat(0.045)
        bond = _make_bond()
        book = Book({"A": bond}, notionals={"A": 0.0})
        results = {"A": price_instrument(bond, curve, date(2024, 11, 15))}
        br = BookResult(results, book)
        assert br.total_mv == 0.0
        assert br.book_duration == 0.0


class TestScenarioResultCube:
    """Scenario-aware aggregation substrate for book workflows."""

    def _make_cube(self) -> ScenarioResultCube:
        curve = YieldCurve.flat(0.045)
        bond_a = _make_bond()
        bond_b = _make_bond(coupon=0.04)
        book = Book(
            {"A": bond_a, "B": bond_b},
            notionals={"A": 1_000_000, "B": 500_000},
        )
        base_results = {
            name: price_instrument(book[name], curve, date(2024, 11, 15), greeks="all")
            for name in book
        }
        shocked_results = {
            name: PricingResult(
                clean_price=result.clean_price - 1.0,
                dirty_price=result.dirty_price - 1.0,
                accrued_interest=result.accrued_interest,
                greeks=result.greeks,
                curve_sensitivities=result.curve_sensitivities,
            )
            for name, result in base_results.items()
        }
        return ScenarioResultCube(
            {
                "base": BookResult(base_results, book),
                "up100": BookResult(shocked_results, book),
            },
            scenario_specs={
                "base": {"name": "base", "shift_bps": 0.0},
                "up100": {"name": "up100", "shift_bps": 100.0},
            },
            scenario_provenance={
                "base": {"source": "unit", "run_mode": "test"},
                "up100": {"source": "unit", "run_mode": "test"},
            },
        )

    def test_preserves_specs_and_provenance(self):
        cube = self._make_cube()

        assert list(cube) == ["base", "up100"]
        assert cube.base_name == "base"
        assert cube.scenario_specs["up100"]["shift_bps"] == pytest.approx(100.0)
        assert cube.scenario_provenance["up100"]["source"] == "unit"

    def test_book_ladder_carries_deltas_and_metadata(self):
        cube = self._make_cube()

        ladder = cube.book_ladder("total_mv")

        assert ladder["base"] > ladder["up100"]
        assert ladder.metadata["metric"] == "total_mv"
        assert ladder.metadata["aggregation_level"] == "book"
        assert ladder.metadata["baseline_scenario"] == "base"
        assert ladder.metadata["deltas"]["base"] == pytest.approx(0.0)
        assert ladder.metadata["deltas"]["up100"] == pytest.approx(
            ladder["up100"] - ladder["base"]
        )
        assert ladder.metadata["scenario_specs"]["up100"]["shift_bps"] == pytest.approx(
            100.0
        )

    def test_position_ladder_supports_market_value_projection(self):
        cube = self._make_cube()

        ladder = cube.position_ladder("mv")

        assert ladder["A"]["base"] > ladder["A"]["up100"]
        assert ladder["B"]["base"] > ladder["B"]["up100"]
        assert ladder.metadata["metric"] == "mv"
        assert ladder.metadata["aggregation_level"] == "position"
        assert ladder.metadata["deltas"]["A"]["base"] == pytest.approx(0.0)
        assert ladder.metadata["deltas"]["A"]["up100"] == pytest.approx(
            ladder["A"]["up100"] - ladder["A"]["base"]
        )

    def test_batch_output_projection_preserves_plan_and_pnl_views(self):
        cube = self._make_cube()
        cube = ScenarioResultCube(
            dict(cube),
            scenario_specs=cube.scenario_specs,
            scenario_provenance=cube.scenario_provenance,
            compute_plan={
                "plan_type": "book_scenario_batch",
                "scenario_count": 2,
            },
        )

        payload = cube.to_batch_output()

        assert payload["compute_plan"]["plan_type"] == "book_scenario_batch"
        assert payload["book_pnl"]["metadata"]["baseline_scenario"] == "base"
        assert payload["book_pnl"]["values"]["base"] == pytest.approx(0.0)
        assert payload["book_pnl"]["values"]["up100"] < 0.0
        assert payload["book_pnl"]["metadata"]["levels"]["base"] > payload["book_pnl"]["metadata"]["levels"]["up100"]
        assert payload["position_pnl"]["values"]["A"]["base"] == pytest.approx(0.0)
        assert payload["position_pnl"]["values"]["A"]["up100"] < 0.0
        assert (
            payload["position_pnl"]["metadata"]["levels"]["A"]["base"]
            > payload["position_pnl"]["metadata"]["levels"]["A"]["up100"]
        )
        assert payload["pnl_attribution"]["scenario_attribution"]["up100"]["top_contributors"][0]["position_name"] == "A"

    def test_book_pnl_values_are_deltas_and_levels_live_in_metadata(self):
        cube = self._make_cube()

        pnl = cube.book_pnl()

        assert pnl["base"] == pytest.approx(0.0)
        assert pnl["up100"] == pytest.approx(pnl.metadata["deltas"]["up100"])
        assert pnl.metadata["levels"]["base"] > pnl.metadata["levels"]["up100"]

    def test_position_pnl_values_are_deltas_and_levels_live_in_metadata(self):
        cube = self._make_cube()

        pnl = cube.position_pnl()

        assert pnl["A"]["base"] == pytest.approx(0.0)
        assert pnl["A"]["up100"] == pytest.approx(pnl.metadata["deltas"]["A"]["up100"])
        assert pnl.metadata["levels"]["A"]["base"] > pnl.metadata["levels"]["A"]["up100"]

    def test_pnl_attribution_ranks_top_contributors_per_scenario(self):
        cube = self._make_cube()

        attribution = cube.pnl_attribution()

        assert attribution["baseline_scenario"] == "base"
        assert attribution["scenario_attribution"]["up100"]["total_pnl"] < 0.0
        assert attribution["scenario_attribution"]["up100"]["top_contributors"][0]["position_name"] == "A"
        assert abs(attribution["net_position_pnl"]["A"]) > abs(attribution["net_position_pnl"]["B"])
