"""Tests for the package-level public API surface.

These tests lock down the canonical package entry points introduced in Tranche 2C.
They complement the existing v2 API tests by covering `trellis.core`,
`trellis.models`, and the public docs/metadata surface.
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_core_package_exports():
    import trellis.core as core
    from trellis.core.capabilities import (
        analyze_gap,
        capability_summary,
        check_market_data,
        discover_capabilities,
    )
    from trellis.core.market_state import MarketState, MissingCapabilityError
    from trellis.core.payoff import (
        Cashflows,
        DeterministicCashflowPayoff,
        Payoff,
        PresentValue,
    )
    from trellis.core.state_space import StateSpace
    from trellis.core.types import DayCountConvention, Frequency, PricingResult

    assert core.MarketState is MarketState
    assert core.MissingCapabilityError is MissingCapabilityError
    assert core.Payoff is Payoff
    assert core.DeterministicCashflowPayoff is DeterministicCashflowPayoff
    assert core.Cashflows is Cashflows
    assert core.PresentValue is PresentValue
    assert core.StateSpace is StateSpace
    assert core.Frequency is Frequency
    assert core.DayCountConvention is DayCountConvention
    assert core.PricingResult is PricingResult
    assert core.analyze_gap is analyze_gap
    assert core.check_market_data is check_market_data
    assert core.discover_capabilities is discover_capabilities
    assert core.capability_summary is capability_summary


def test_models_package_exports():
    import trellis.models as models
    from trellis.models.black import (
        black76_call,
        black76_put,
        garman_kohlhagen_call,
        garman_kohlhagen_put,
    )
    from trellis.models.vol_surface import FlatVol, GridVolSurface, VolSurface

    assert models.black76_call is black76_call
    assert models.black76_put is black76_put
    assert models.garman_kohlhagen_call is garman_kohlhagen_call
    assert models.garman_kohlhagen_put is garman_kohlhagen_put
    assert models.FlatVol is FlatVol
    assert models.GridVolSurface is GridVolSurface
    assert models.VolSurface is VolSurface

    for name in (
        "analytical",
        "trees",
        "monte_carlo",
        "qmc",
        "pde",
        "transforms",
        "processes",
        "copulas",
        "calibration",
        "cashflow_engine",
    ):
        assert hasattr(models, name), f"trellis.models missing `{name}` package export"


def test_family_package_exports_are_canonical():
    import trellis.models.monte_carlo as monte_carlo
    import trellis.models.pde as pde
    import trellis.models.qmc as qmc
    import trellis.models.transforms as transforms
    import trellis.models.trees as trees
    from trellis.models.monte_carlo.brownian_bridge import brownian_bridge
    from trellis.models.monte_carlo.variance_reduction import sobol_normals

    assert hasattr(trees, "BinomialTree")
    assert hasattr(trees, "TrinomialTree")
    assert hasattr(trees, "backward_induction")

    assert hasattr(monte_carlo, "MonteCarloEngine")
    assert hasattr(monte_carlo, "euler_maruyama")
    assert hasattr(monte_carlo, "milstein")

    assert qmc.sobol_normals is sobol_normals
    assert qmc.brownian_bridge is brownian_bridge

    assert hasattr(pde, "theta_method_1d")
    assert hasattr(pde, "crank_nicolson_1d")
    assert hasattr(pde, "implicit_fd_1d")
    assert hasattr(pde, "Grid")

    assert hasattr(transforms, "fft_price")
    assert hasattr(transforms, "cos_price")


def test_models_docs_use_package_level_entry_points():
    text = (REPO_ROOT / "docs" / "api" / "models.rst").read_text()
    assert "trellis.models.pde.theta_method_1d" in text
    assert "trellis.models.pde.crank_nicolson.crank_nicolson_1d" not in text
    assert "trellis.models.pde.implicit_fd.implicit_fd_1d" not in text
    assert "trellis.models.GridVolSurface" in text
    assert "trellis.models.trees.BinomialTree" in text
    assert "trellis.models.monte_carlo.MonteCarloEngine" in text
    assert "trellis.models.qmc.sobol_normals" in text
    assert "trellis.models.qmc.brownian_bridge" in text
    assert "trellis.models.transforms.fft_price" in text


def test_core_docs_use_package_level_entry_points():
    text = (REPO_ROOT / "docs" / "api" / "core.rst").read_text()
    assert "trellis.core.MarketState" in text
    assert "trellis.core.Payoff" in text
    assert "trellis.core.Frequency" in text
    assert "trellis.core.capability_summary" in text


def test_readme_uses_trellis_name():
    first_line = (REPO_ROOT / "README.md").read_text().splitlines()[0].strip().lower()
    assert first_line == "# trellis"


def test_setup_metadata_uses_trellis_name():
    setup_text = (REPO_ROOT / "setup.py").read_text()
    assert 'name="trellis"' in setup_text or "name='trellis'" in setup_text


def test_migration_notes_cover_qmc_canonical_path():
    text = (REPO_ROOT / "docs" / "migration_notes.md").read_text()
    assert "trellis.models.qmc" in text
    assert "trellis.models.monte_carlo.variance_reduction" in text
    assert "trellis.models.monte_carlo.brownian_bridge" in text
