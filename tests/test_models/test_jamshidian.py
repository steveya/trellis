"""Autograd regressions for the Jamshidian ZCB option kernel."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trellis.core.differentiable import gradient
from trellis.curves.yield_curve import YieldCurve
from trellis.models.analytical.jamshidian import (
    ResolvedJamshidianInputs,
    zcb_option_hw,
    zcb_option_hw_raw,
)


FLAT_RATE = 0.05
SIGMA = 0.01
A_HW = 0.1
T_EXP = 3.0
T_BOND = 9.0
K_UNIT = 0.63


@pytest.fixture(scope="module")
def flat_curve():
    return YieldCurve.flat(FLAT_RATE, max_tenor=max(T_BOND + 1.0, 31.0))


def _finite_difference(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


def _resolved(flat_curve, *, sigma: float = SIGMA, a: float = A_HW):
    return ResolvedJamshidianInputs(
        discount_factor_expiry=float(flat_curve.discount(T_EXP)),
        discount_factor_bond=float(flat_curve.discount(T_BOND)),
        strike=K_UNIT,
        T_exp=T_EXP,
        T_bond=T_BOND,
        sigma=sigma,
        a=a,
    )


class TestJamshidianRawKernel:

    def test_raw_kernel_matches_adapter(self, flat_curve):
        resolved = _resolved(flat_curve)

        raw = zcb_option_hw_raw(resolved)
        adapted = zcb_option_hw(flat_curve, K_UNIT, T_EXP, T_BOND, SIGMA, A_HW)

        assert raw["call"] == pytest.approx(adapted["call"], rel=1e-12)
        assert raw["put"] == pytest.approx(adapted["put"], rel=1e-12)

    def test_call_sigma_gradient_matches_finite_difference(self, flat_curve):
        resolved = _resolved(flat_curve)

        autodiff = gradient(
            lambda sigma: zcb_option_hw_raw(replace(resolved, sigma=sigma))["call"]
        )(resolved.sigma)
        fd = _finite_difference(
            lambda sigma: zcb_option_hw_raw(replace(resolved, sigma=sigma))["call"],
            resolved.sigma,
        )

        assert autodiff == pytest.approx(fd, rel=1e-5, abs=1e-8)

    def test_call_mean_reversion_gradient_matches_finite_difference(self, flat_curve):
        resolved = _resolved(flat_curve)

        autodiff = gradient(
            lambda a: zcb_option_hw_raw(replace(resolved, a=a))["call"]
        )(resolved.a)
        fd = _finite_difference(
            lambda a: zcb_option_hw_raw(replace(resolved, a=a))["call"],
            resolved.a,
        )

        assert autodiff == pytest.approx(fd, rel=1e-5, abs=1e-8)

