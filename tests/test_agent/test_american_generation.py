"""Proving-ground tests for the generated American put payoff."""

from __future__ import annotations

from datetime import date
from importlib import import_module
from math import exp, log, sqrt

from scipy.stats import norm

from trellis.agent.codegen_guardrails import build_generation_plan
from trellis.agent.quant import PricingPlan
from trellis.agent.semantic_validation import validate_semantics
from trellis.agent.knowledge.decompose import decompose_to_ir
from trellis.core.market_state import MarketState
from trellis.curves.yield_curve import YieldCurve
from trellis.engine.payoff_pricer import price_payoff
from trellis.models.trees.lattice import build_spot_lattice, lattice_backward_induction
from trellis.models.vol_surface import FlatVol


def _american_artifact_source() -> str:
    mod = import_module("trellis.instruments._agent.americanputpayoff")
    return mod.__file__ and open(mod.__file__).read()


def _american_generation_plan():
    return build_generation_plan(
        pricing_plan=PricingPlan(
            method="monte_carlo",
            method_modules=["trellis.models.monte_carlo.engine"],
            required_market_data={"discount", "black_vol"},
            model_to_build="american_option",
            reasoning="test",
        ),
        instrument_type="american_option",
        inspected_modules=(
            "trellis.models.monte_carlo.engine",
            "trellis.models.monte_carlo.lsm",
            "trellis.models.monte_carlo.schemes",
            "trellis.models.processes.gbm",
        ),
        product_ir=decompose_to_ir(
            "American put option on equity",
            instrument_type="american_option",
        ),
    )


def _euro_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def test_current_american_artifact_is_route_compliant():
    report = validate_semantics(
        _american_artifact_source(),
        product_ir=decompose_to_ir(
            "American put option on equity",
            instrument_type="american_option",
        ),
        generation_plan=_american_generation_plan(),
    )

    assert report.ok


def test_current_american_artifact_is_thin_lsm_adapter():
    source = _american_artifact_source()

    assert "longstaff_schwartz" in source
    assert "LaguerreBasis" in source
    assert 'method="lsm"' not in source
    assert "engine.price(" not in source
    assert "engine.simulate(" in source


def test_current_american_artifact_prices_plausibly_against_tree():
    mod = import_module("trellis.instruments._agent.americanputpayoff")
    settle = date(2024, 11, 15)
    expiry = date(2025, 11, 15)
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    T = (expiry - settle).days / 365.25

    spec = mod.AmericanPutEquitySpec(
        spot=S0,
        strike=K,
        expiry_date=expiry,
    )
    market_state = MarketState(
        as_of=settle,
        settlement=settle,
        discount=YieldCurve.flat(r),
        vol_surface=FlatVol(sigma),
    )

    lsm_price = price_payoff(mod.AmericanOptionPayoff(spec), market_state)

    lattice = build_spot_lattice(S0, r, sigma, T, 500)

    def payoff(step, node, tree):
        return max(K - tree.get_state(step, node), 0.0)

    tree_price = lattice_backward_induction(
        lattice,
        payoff,
        exercise_value=payoff,
        exercise_type="american",
    )
    euro_price = _euro_put(S0, K, r, sigma, T)

    assert lsm_price >= euro_price - 0.10
    assert abs(lsm_price - tree_price) / tree_price < 0.10
