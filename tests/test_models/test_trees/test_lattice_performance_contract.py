from __future__ import annotations

from statistics import median
import time
import warnings

import pytest

from trellis.curves.yield_curve import YieldCurve
from trellis.core.capabilities import METHODS
from trellis.models.trees.algebra import (
    BINOMIAL_1F_TOPOLOGY,
    LATTICE_MODEL_REGISTRY,
    LOG_SPOT_MESH,
    build_lattice,
    compile_lattice_recipe,
    equity_tree,
    price_on_lattice,
    with_overlay,
)
from trellis.models.trees.lattice import build_rate_lattice, build_spot_lattice, lattice_backward_induction


def _build_plain_call_contract():
    _, _, _, contract = compile_lattice_recipe(
        equity_tree(model_family="crr", strike=100.0, option_type="call")
    )
    return contract


def _median_elapsed(fn, *, batches: int = 7, iterations: int = 3) -> tuple[float, float]:
    result = 0.0
    samples: list[float] = []
    for _ in range(batches):
        t0 = time.perf_counter()
        for _ in range(iterations):
            result = float(fn())
        samples.append((time.perf_counter() - t0) / float(iterations))
    return result, float(median(samples))


def test_price_on_lattice_stays_close_to_direct_backward_induction_timing():
    lattice = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        LOG_SPOT_MESH,
        LATTICE_MODEL_REGISTRY["crr"],
        calibration_target=None,
        spot=100.0,
        rate=0.03,
        sigma=0.20,
        maturity=1.0,
        n_steps=400,
    )
    contract = _build_plain_call_contract()

    def direct() -> float:
        return lattice_backward_induction(
            lattice,
            terminal_payoff=lambda step, node, lattice_: max(lattice_.get_state(step, node) - 100.0, 0.0),
        )

    # Warm both paths before timing so JIT/import costs do not dominate.
    price_on_lattice(lattice, contract)
    direct()

    fast_price, generalized = _median_elapsed(lambda: price_on_lattice(lattice, contract))
    direct_price, direct_elapsed = _median_elapsed(direct)

    assert fast_price == pytest.approx(direct_price, rel=1e-12)
    assert generalized <= 3.0 * direct_elapsed
    assert lattice._lattice_last_pricing_path.startswith("fast_")


def test_price_on_lattice_warns_when_overlay_forces_python_fallback():
    lattice = build_lattice(
        BINOMIAL_1F_TOPOLOGY,
        LOG_SPOT_MESH,
        LATTICE_MODEL_REGISTRY["crr"],
        calibration_target=None,
        spot=100.0,
        rate=0.03,
        sigma=0.20,
        maturity=1.0,
        n_steps=120,
    )
    _, _, _, contract = compile_lattice_recipe(
        with_overlay(
            equity_tree(model_family="crr", strike=100.0, option_type="call"),
            "knock_out_barrier",
            barrier=130.0,
        )
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        price = price_on_lattice(lattice, contract)

    assert price >= 0.0
    assert any("python fallback" in str(w.message).lower() for w in caught)
    assert lattice._lattice_last_pricing_path == "python_overlay_fallback"


@pytest.mark.legacy_compat
def test_legacy_tree_entry_points_emit_deprecation_warnings():
    curve = YieldCurve.flat(0.03)

    with pytest.deprecated_call():
        build_spot_lattice(100.0, 0.03, 0.20, 1.0, 32)

    with pytest.deprecated_call():
        build_rate_lattice(0.03, 0.01, 0.1, 1.0, 32, discount_curve=curve)


def test_rate_tree_capability_examples_reference_unified_api():
    capability = next(method for method in METHODS if method.name == "rate_tree")

    assert "build_lattice" in capability.example_usage
    assert "BinomialTree.crr" not in capability.example_usage
