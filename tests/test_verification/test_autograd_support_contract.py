"""Verification cohort for the public autograd support contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as raw_np
import pytest

from trellis.core.differentiable import get_numpy, gradient
from trellis.core.market_state import MarketState
from trellis.core.payoff import DeterministicCashflowPayoff, ResolvedInputPayoff
from trellis.curves.credit_curve import CreditCurve
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.instruments.bond import Bond
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.path_state import terminal_value_payoff
from trellis.models.processes.gbm import GBM
from trellis.models.vol_surface import FlatVol, GridVolSurface
from trellis.session import Session


anp = get_numpy()
SETTLE = date(2024, 11, 15)
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _ResolvedSpot:
    T: float
    spot: object


class _PublicResolvedSpotPayoff(ResolvedInputPayoff[None, _ResolvedSpot]):
    requirements = {"spot"}

    def resolve_inputs(self, market_state: MarketState) -> _ResolvedSpot:
        return _ResolvedSpot(T=1.0, spot=market_state.spot)

    def evaluate_from_resolved(self, resolved: _ResolvedSpot):
        return resolved.spot * resolved.spot + 0.25 * resolved.spot


def test_gradient_cohort_public_payoff_curve_credit_and_grid_vol_paths():
    """One representative cohort defends the public traced pricing map."""

    payoff_delta = gradient(
        lambda spot: price_payoff(
            _PublicResolvedSpotPayoff(None),
            MarketState(as_of=SETTLE, settlement=SETTLE, spot=spot),
        )
    )(100.0)
    assert payoff_delta == pytest.approx(200.25)

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

    surface_gradient = gradient(
        lambda vols: GridVolSurface(
            expiries=(1.0, 2.0),
            strikes=(90.0, 110.0),
            vols=vols,
        ).black_vol(1.5, 100.0)
    )(anp.array([[0.25, 0.22], [0.27, 0.24]]))
    raw_np.testing.assert_allclose(surface_gradient, raw_np.full((2, 2), 0.25), atol=1e-12)


def test_gradient_cohort_runtime_risk_and_pathwise_mc_provenance():
    """Runtime outputs should expose the derivative method, not hide the lane."""

    bond = Bond(
        face=100.0,
        coupon=0.045,
        maturity_date=date(2034, 11, 15),
        maturity=10,
        frequency=2,
    )
    rate_session = Session(
        curve=YieldCurve([1.0, 2.0, 5.0, 10.0], [0.04, 0.042, 0.045, 0.047]),
        settlement=SETTLE,
    )
    rate_risk = rate_session.analyze(
        DeterministicCashflowPayoff(bond),
        measures=["dv01", "duration", "convexity", "key_rate_durations"],
    )
    for measure in (rate_risk.dv01, rate_risk.duration, rate_risk.convexity, rate_risk.key_rate_durations):
        assert measure.metadata["resolved_derivative_method"] == "autodiff_public_curve"

    flat_vol_session = Session(curve=YieldCurve.flat(0.04), settlement=SETTLE, vol_surface=FlatVol(0.20))
    vega = flat_vol_session.analyze(
        _VolProbePayoff(expiry=1.0, strike=100.0),
        measures=[{"vega": {"bump_pct": 1.0}}],
    ).vega
    assert vega.metadata["resolved_derivative_method"] == "autodiff_flat_vol"

    engine = MonteCarloEngine(GBM(mu=0.05, sigma=0.20), n_paths=64, n_steps=8, seed=43, method="exact")
    shocks = raw_np.random.default_rng(47).standard_normal((engine.n_paths, engine.n_steps))
    payoff = terminal_value_payoff(lambda terminal: anp.maximum(terminal - 100.0, 0.0))

    def price_from_spot(spot):
        return engine.price(
            spot,
            1.0,
            payoff,
            discount_rate=0.05,
            shocks=shocks,
            differentiable=True,
            return_paths=False,
        )["price"]

    autodiff_delta = gradient(price_from_spot)(100.0)
    finite_difference_delta = (price_from_spot(100.01) - price_from_spot(99.99)) / 0.02
    assert autodiff_delta == pytest.approx(finite_difference_delta, rel=1e-4, abs=1e-4)


class _VolProbePayoff:
    requirements = {"black_vol_surface"}

    def __init__(self, *, expiry: float, strike: float):
        self.expiry = float(expiry)
        self.strike = float(strike)

    def evaluate(self, market_state: MarketState):
        return market_state.vol_surface.black_vol(self.expiry, self.strike)


def test_support_contract_docs_and_limitations_name_the_checked_lanes():
    """Docs and limitations should describe the same supported AD surface as tests."""

    differentiable_pricing = (REPO_ROOT / "docs/quant/differentiable_pricing.rst").read_text()
    limitations = (REPO_ROOT / "LIMITATIONS.md").read_text()

    required_doc_terms = {
        "PricingValue",
        "YieldCurve",
        "CreditCurve",
        "GridVolSurface",
        "autodiff_public_curve",
        "surface_bucket_bump",
        "autodiff_flat_vol",
        "autodiff_vector_jacobian",
        "autodiff_scalar_gradient",
        "finite_difference_vector_jacobian",
        "fit_heston_surface",
        "get_backend_capabilities",
        "hessian_vector_product",
        "portfolio_aad",
        "simulate_with_shocks",
        "price_event_aware_monte_carlo",
    }
    missing_doc_terms = sorted(term for term in required_doc_terms if term not in differentiable_pricing)
    assert missing_doc_terms == []

    required_limitation_terms = {
        "L30",
        "L31",
        "L32",
        "PricingValue",
        "public payoff boundary",
        "public yield/credit/grid-vol market objects",
        "finite_difference_vector_jacobian",
        "get_backend_capabilities",
        "portfolio_aad",
        "runtime risk now records",
    }
    missing_limitation_terms = sorted(term for term in required_limitation_terms if term not in limitations)
    assert missing_limitation_terms == []

    stale_claims = (
        "spot delta / gamma / theta are not implemented",
        "evaluate() returns a float PV",
        "YieldCurve is not trace-safe",
        "GridVolSurface.black_vol(...) returns float",
        "AD layer is still a thin `autograd` wrapper",
    )
    assert not any(claim in limitations for claim in stale_claims)
