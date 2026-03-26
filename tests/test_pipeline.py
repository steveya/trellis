"""Tests for trellis.pipeline — Pipeline (declarative batch)."""

import tempfile
from datetime import date
from pathlib import Path

import pytest

from trellis.book import Book, BookResult
from trellis.data.schema import MarketSnapshot
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.bond import Bond
from trellis.models.vol_surface import FlatVol, GridVolSurface
from trellis.pipeline import Pipeline


def _curve():
    return YieldCurve.flat(0.045)


def _book():
    bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
                maturity=10, frequency=2)
    return Book({"10Y": bond})


def _snapshot():
    return MarketSnapshot(
        as_of=date(2024, 11, 15),
        source="unit",
        discount_curves={
            "usd_ois": YieldCurve.flat(0.045),
            "eur_ois": YieldCurve.flat(0.025),
        },
        vol_surfaces={
            "atm": FlatVol(0.20),
            "smile": GridVolSurface(
                expiries=(1.0, 2.0),
                strikes=(90.0, 110.0),
                vols=((0.25, 0.22), (0.27, 0.24)),
            ),
        },
        default_discount_curve="usd_ois",
        default_vol_surface="atm",
    )


class TestPipeline:

    def test_basic_run(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .run()
        )
        assert "base" in results
        assert isinstance(results["base"], BookResult)
        assert results["base"].total_mv > 0

    def test_multiple_scenarios(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .scenarios([
                {"name": "base", "shift_bps": 0},
                {"name": "up100", "shift_bps": 100},
            ])
            .run()
        )
        assert "base" in results
        assert "up100" in results
        assert results["up100"].total_mv < results["base"].total_mv

    def test_compute_price_only(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .compute(["price"])
            .run()
        )
        br = results["base"]
        assert br["10Y"].greeks == {}

    def test_compute_selective_greeks(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(curve=_curve())
            .compute(["price", "dv01"])
            .run()
        )
        br = results["base"]
        assert "dv01" in br["10Y"].greeks
        assert "convexity" not in br["10Y"].greeks

    def test_missing_instruments_raises(self):
        with pytest.raises(ValueError, match="No instruments"):
            Pipeline().market_data(curve=_curve()).run()

    def test_output_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_template = str(Path(tmpdir) / "{scenario}.csv")
            (
                Pipeline()
                .instruments(_book())
                .market_data(curve=_curve())
                .output_csv(path_template)
                .run()
            )
            assert Path(tmpdir, "base.csv").exists()

    def test_output_parquet(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_template = str(Path(tmpdir) / "{scenario}.parquet")
            (
                Pipeline()
                .instruments(_book())
                .market_data(curve=_curve())
                .output_parquet(path_template)
                .run()
            )
            assert Path(tmpdir, "base.parquet").exists()

    def test_mock_source_no_mocking_needed(self):
        """Full offline pipeline — no patches, no network."""
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(source="mock", as_of="2024-11-15")
            .run()
        )
        assert results["base"].total_mv > 0

    def test_market_snapshot_input(self):
        snapshot = _snapshot()
        usd_results = (
            Pipeline()
            .instruments(_book())
            .market_data(snapshot=snapshot, discount_curve="usd_ois")
            .run()
        )
        eur_results = (
            Pipeline()
            .instruments(_book())
            .market_data(snapshot=snapshot, discount_curve="eur_ois")
            .run()
        )
        assert eur_results["base"].total_mv > usd_results["base"].total_mv

    def test_market_snapshot_accepts_named_vol_surface(self):
        results = (
            Pipeline()
            .instruments(_book())
            .market_data(snapshot=_snapshot(), vol_surface_name="smile")
            .run()
        )
        assert results["base"].total_mv > 0
