"""Product-family gradient matrix for the public autograd support contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as raw_np
import pytest

from trellis.conventions.day_count import DayCountConvention
from trellis.core.differentiable import get_numpy, gradient
from trellis.core.market_state import MarketState
from trellis.core.payoff import ResolvedInputPayoff
from trellis.core.types import Frequency
from trellis.curves.bootstrap import (
    BootstrapConventionBundle,
    BootstrapCurveInputBundle,
    BootstrapInstrument,
    bootstrap_curve_result,
)
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.instruments.fx import FXRate
from trellis.models.analytical.quanto import price_quanto_option_analytical
from trellis.models.black import black76_call
from trellis.models.monte_carlo.engine import (
    MonteCarloEngine,
    describe_monte_carlo_derivative_policy,
)
from trellis.models.monte_carlo.path_state import barrier_payoff
from trellis.models.processes.gbm import GBM
from trellis.models.resolution.quanto import resolve_quanto_inputs
from trellis.models.vol_surface import FlatVol, GridVolSurface
from trellis.session import Session


anp = get_numpy()
SETTLE = date(2024, 11, 15)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GradientMatrixRow:
    """One checked row in the product-family derivative support matrix."""

    family_id: str
    product_family: str
    category: str
    support_status: str
    expected_derivative_method: str
    fallback_derivative_method: str | None
    documentation_terms: tuple[str, ...]


GRADIENT_MATRIX: tuple[GradientMatrixRow, ...] = (
    GradientMatrixRow(
        family_id="analytical_black76",
        product_family="Black76 closed-form route",
        category="analytical",
        support_status="supported",
        expected_derivative_method="autodiff_scalar_gradient",
        fallback_derivative_method=None,
        documentation_terms=("analytical_black76", "Black76", "autodiff_scalar_gradient"),
    ),
    GradientMatrixRow(
        family_id="public_curve_nodes",
        product_family="public yield and credit curve node routes",
        category="public_curve",
        support_status="supported",
        expected_derivative_method="autodiff_public_curve",
        fallback_derivative_method=None,
        documentation_terms=("public_curve_nodes", "YieldCurve", "CreditCurve", "autodiff_public_curve"),
    ),
    GradientMatrixRow(
        family_id="grid_vol_surface_bucketed",
        product_family="grid volatility surface node and bucketed-vega route",
        category="vol_surface",
        support_status="partial",
        expected_derivative_method="surface_bucket_bump",
        fallback_derivative_method=None,
        documentation_terms=("grid_vol_surface_bucketed", "GridVolSurface", "surface_bucket_bump"),
    ),
    GradientMatrixRow(
        family_id="smooth_monte_carlo_pathwise",
        product_family="smooth Monte Carlo pathwise route",
        category="smooth_monte_carlo",
        support_status="supported",
        expected_derivative_method="autodiff_pathwise",
        fallback_derivative_method=None,
        documentation_terms=("smooth_monte_carlo_pathwise", "autodiff_pathwise", "simulate_with_shocks"),
    ),
    GradientMatrixRow(
        family_id="rates_bootstrap_calibration",
        product_family="rates bootstrap calibration route",
        category="calibration",
        support_status="supported",
        expected_derivative_method="autodiff_vector_jacobian",
        fallback_derivative_method=None,
        documentation_terms=("rates_bootstrap_calibration", "autodiff_vector_jacobian", "solver_provenance"),
    ),
    GradientMatrixRow(
        family_id="quanto_generated_helper",
        product_family="route-generated quanto analytical helper route",
        category="route_generated",
        support_status="supported",
        expected_derivative_method="autodiff_scalar_gradient",
        fallback_derivative_method=None,
        documentation_terms=("quanto_generated_helper", "route-generated", "autodiff_scalar_gradient"),
    ),
    GradientMatrixRow(
        family_id="barrier_mc_discontinuous_policy",
        product_family="barrier Monte Carlo discontinuous policy route",
        category="unsupported_discontinuous",
        support_status="unsupported_with_declared_fallback",
        expected_derivative_method="unsupported_discontinuous_pathwise",
        fallback_derivative_method="finite_difference_bump_reprice",
        documentation_terms=(
            "barrier_mc_discontinuous_policy",
            "unsupported_discontinuous_pathwise",
            "finite_difference_bump_reprice",
        ),
    ),
)


@dataclass(frozen=True)
class _ResolvedVolProbe:
    expiry: float
    strike: float
    vol: object


class _VolProbePayoff(ResolvedInputPayoff[None, _ResolvedVolProbe]):
    requirements = {"black_vol_surface"}

    def __init__(self, *, expiry: float, strike: float):
        super().__init__(None)
        self.expiry = float(expiry)
        self.strike = float(strike)

    def resolve_inputs(self, market_state: MarketState) -> _ResolvedVolProbe:
        return _ResolvedVolProbe(
            expiry=self.expiry,
            strike=self.strike,
            vol=market_state.vol_surface.black_vol(self.expiry, self.strike),
        )

    def evaluate_from_resolved(self, resolved: _ResolvedVolProbe):
        return resolved.vol


def _finite_difference(fn, x: float, bump: float = 1e-4) -> float:
    return float((fn(x + bump) - fn(x - bump)) / (2.0 * bump))


def test_gradient_matrix_has_required_product_family_coverage():
    required_categories = {
        "analytical",
        "public_curve",
        "vol_surface",
        "smooth_monte_carlo",
        "calibration",
        "route_generated",
        "unsupported_discontinuous",
    }
    categories = {row.category for row in GRADIENT_MATRIX}
    assert required_categories <= categories

    family_ids = [row.family_id for row in GRADIENT_MATRIX]
    assert len(family_ids) == len(set(family_ids))

    for row in GRADIENT_MATRIX:
        assert row.expected_derivative_method
        assert row.support_status in {
            "supported",
            "partial",
            "unsupported_with_declared_fallback",
        }
        if row.support_status.startswith("unsupported"):
            assert row.fallback_derivative_method


@pytest.mark.parametrize("row", GRADIENT_MATRIX, ids=lambda row: row.family_id)
def test_gradient_matrix_row_matches_checked_runtime_or_governance(row: GradientMatrixRow):
    check = _ROW_CHECKS[row.family_id]
    check(row)


def test_gradient_matrix_is_documented_and_prevents_stale_support_claims():
    differentiable_pricing = (REPO_ROOT / "docs/quant/differentiable_pricing.rst").read_text()
    limitations = (REPO_ROOT / "LIMITATIONS.md").read_text()

    assert "Product-Family Gradient Matrix" in differentiable_pricing
    for row in GRADIENT_MATRIX:
        for term in row.documentation_terms:
            assert term in differentiable_pricing
        assert row.expected_derivative_method in differentiable_pricing
        if row.fallback_derivative_method:
            assert row.fallback_derivative_method in differentiable_pricing

    stale_or_overbroad_claims = (
        "jvp=True",
        "portfolio_aad=True",
        "supports universal portfolio AAD",
        "automatic discontinuous Greeks are supported",
        "all generated routes are differentiable by default",
        "surface-native scalar vega for every smile surface is supported",
    )
    for claim in stale_or_overbroad_claims:
        assert claim not in differentiable_pricing
        assert claim not in limitations


def _check_analytical_black76(row: GradientMatrixRow) -> None:
    assert row.expected_derivative_method == "autodiff_scalar_gradient"
    autodiff_vega = gradient(lambda sigma: black76_call(100.0, 100.0, sigma, 1.0))(0.20)
    fd_vega = _finite_difference(lambda sigma: black76_call(100.0, 100.0, sigma, 1.0), 0.20)
    assert autodiff_vega == pytest.approx(fd_vega, rel=1e-6, abs=1e-8)


def _check_public_curve_nodes(row: GradientMatrixRow) -> None:
    assert row.expected_derivative_method == "autodiff_public_curve"

    curve_gradient = gradient(
        lambda rates: YieldCurve((1.0, 2.0), rates).discount(1.5)
    )(anp.array([0.04, 0.06]))
    expected_discount = raw_np.exp(-0.05 * 1.5)
    raw_np.testing.assert_allclose(
        curve_gradient,
        raw_np.array([-0.75 * expected_discount, -0.75 * expected_discount]),
        atol=1e-12,
    )

    credit_gradient = gradient(
        lambda hazards: CreditCurve((1.0, 3.0), hazards).survival_probability(2.0)
    )(anp.array([0.01, 0.02]))
    expected_survival = raw_np.exp(-0.03)
    raw_np.testing.assert_allclose(
        credit_gradient,
        raw_np.array([-expected_survival, -expected_survival]),
        atol=1e-12,
    )


def _check_grid_vol_surface_bucketed(row: GradientMatrixRow) -> None:
    assert row.expected_derivative_method == "surface_bucket_bump"

    surface_gradient = gradient(
        lambda vols: GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=vols,
        ).black_vol(1.5, 100.0)
    )(anp.array([[0.25, 0.22], [0.27, 0.24]]))
    raw_np.testing.assert_allclose(surface_gradient, raw_np.full((2, 2), 0.25), atol=1e-12)

    session = Session(
        curve=YieldCurve.flat(0.04),
        settlement=SETTLE,
        vol_surface=GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=((0.25, 0.22), (0.27, 0.24)),
        ),
    )
    vega = session.analyze(
        _VolProbePayoff(expiry=1.5, strike=100.0),
        measures=[
            {
                "vega": {
                    "expiries": (1.0, 2.0),
                    "strikes": (90.0, 110.0),
                    "bump_pct": 1.0,
                }
            }
        ],
    ).vega
    assert vega.metadata["resolved_derivative_method"] == "surface_bucket_bump"


def _check_smooth_monte_carlo_pathwise(row: GradientMatrixRow) -> None:
    assert row.expected_derivative_method == "autodiff_pathwise"

    engine = MonteCarloEngine(GBM(mu=0.05, sigma=0.20), n_paths=64, n_steps=8, seed=43, method="exact")
    shocks = raw_np.random.default_rng(47).standard_normal((engine.n_paths, engine.n_steps))

    def price_from_spot(spot):
        return engine.price(
            spot,
            1.0,
            lambda paths: anp.maximum(paths[:, -1] - 100.0, 0.0),
            discount_rate=0.05,
            shocks=shocks,
            differentiable=True,
            return_paths=False,
        )["price"]

    metadata = engine.price(
        100.0,
        1.0,
        lambda paths: anp.maximum(paths[:, -1] - 100.0, 0.0),
        discount_rate=0.05,
        shocks=shocks,
        differentiable=True,
        return_paths=False,
    )["derivative_metadata"]
    assert metadata["resolved_derivative_method"] == "autodiff_pathwise"

    autodiff_delta = gradient(price_from_spot)(100.0)
    finite_difference_delta = (price_from_spot(100.01) - price_from_spot(99.99)) / 0.02
    assert autodiff_delta == pytest.approx(finite_difference_delta, rel=1e-4, abs=1e-4)


def _check_rates_bootstrap_calibration(row: GradientMatrixRow) -> None:
    assert row.expected_derivative_method == "autodiff_vector_jacobian"

    bundle = BootstrapCurveInputBundle(
        curve_name="usd_ois_boot",
        currency="USD",
        rate_index="USD-SOFR-3M",
        conventions=BootstrapConventionBundle(
            deposit_day_count=DayCountConvention.ACT_360,
            future_day_count=DayCountConvention.ACT_360,
            swap_fixed_frequency=Frequency.ANNUAL,
            swap_fixed_day_count=DayCountConvention.THIRTY_360_US,
            swap_float_frequency=Frequency.QUARTERLY,
            swap_float_day_count=DayCountConvention.ACT_360,
        ),
        instruments=(
            BootstrapInstrument(tenor=0.25, quote=0.04, instrument_type="deposit", label="DEP3M"),
            BootstrapInstrument(tenor=2.0, quote=0.045, instrument_type="swap", label="SWAP2Y"),
            BootstrapInstrument(tenor=5.0, quote=0.048, instrument_type="swap", label="SWAP5Y"),
        ),
    )
    result = bootstrap_curve_result(bundle, max_iter=75, tol=1e-12)

    assert result.solve_result.metadata["derivative_method"] == "autodiff_vector_jacobian"
    assert result.solver_provenance.backend["derivative_method"] == "autodiff_vector_jacobian"
    assert result.diagnostics.max_abs_residual < 1e-8


def _check_quanto_generated_helper(row: GradientMatrixRow) -> None:
    assert row.expected_derivative_method == "autodiff_scalar_gradient"

    from trellis.instruments._agent.quantooptionanalytical import QuantoOptionSpec

    spec = QuantoOptionSpec(
        notional=100_000.0,
        strike=100.0,
        expiry_date=date(2025, 11, 15),
        fx_pair="EURUSD",
        underlier_currency="EUR",
        domestic_currency="USD",
    )

    def price_from_spot(spot):
        market_state = MarketState(
            as_of=SETTLE,
            settlement=SETTLE,
            discount=YieldCurve.flat(0.05),
            forecast_curves={"EUR-DISC": YieldCurve.flat(0.03)},
            fx_rates={"EURUSD": FXRate(spot=1.10, domestic="USD", foreign="EUR")},
            spot=spot,
            underlier_spots={"EUR": spot},
            vol_surface=FlatVol(0.20),
            model_parameters={"quanto_correlation": 0.35},
        )
        return price_quanto_option_analytical(spec, resolve_quanto_inputs(market_state, spec))

    autodiff_delta = gradient(price_from_spot)(100.0)
    finite_difference_delta = (price_from_spot(100.01) - price_from_spot(99.99)) / 0.02
    assert autodiff_delta == pytest.approx(finite_difference_delta, rel=1e-6)


def _check_barrier_mc_discontinuous_policy(row: GradientMatrixRow) -> None:
    assert row.expected_derivative_method == "unsupported_discontinuous_pathwise"
    assert row.fallback_derivative_method == "finite_difference_bump_reprice"

    payoff = barrier_payoff(
        barrier=95.0,
        direction="down",
        knock="out",
        terminal_payoff_fn=lambda terminal: raw_np.maximum(terminal - 100.0, 0.0),
    )
    metadata = describe_monte_carlo_derivative_policy(
        payoff.path_requirement,
        differentiable=True,
    )

    assert metadata["resolved_derivative_method"] == "unsupported_discontinuous_pathwise"
    assert metadata["pathwise_autodiff_supported"] is False
    assert metadata["fallback_derivative_method"] == "finite_difference_bump_reprice"
    assert metadata["discontinuous_derivative_policy"] == "fail_closed"


_ROW_CHECKS = {
    "analytical_black76": _check_analytical_black76,
    "public_curve_nodes": _check_public_curve_nodes,
    "grid_vol_surface_bucketed": _check_grid_vol_surface_bucketed,
    "smooth_monte_carlo_pathwise": _check_smooth_monte_carlo_pathwise,
    "rates_bootstrap_calibration": _check_rates_bootstrap_calibration,
    "quanto_generated_helper": _check_quanto_generated_helper,
    "barrier_mc_discontinuous_policy": _check_barrier_mc_discontinuous_policy,
}
