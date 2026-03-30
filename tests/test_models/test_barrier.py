"""Tests for the route-local barrier analytical kernel pack."""

from __future__ import annotations

from dataclasses import replace

import pytest

from trellis.core.differentiable import gradient
from trellis.models.analytical.barrier import (
    ResolvedBarrierInputs,
    barrier_image_raw,
    barrier_option_price,
    down_and_in_call,
    down_and_in_call_raw,
    down_and_out_call,
    down_and_out_call_raw,
    vanilla_call_raw,
)


def _finite_difference(fn, x, eps=1e-6):
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


class TestBarrierKernelPack:
    """T09 barrier route assembly from reusable analytical pieces."""

    def setup_method(self):
        self.resolved = ResolvedBarrierInputs(
            spot=100.0,
            strike=100.0,
            barrier=90.0,
            rate=0.05,
            sigma=0.20,
            T=1.0,
        )

    def test_kernel_pack_recomposes_t09_route(self):
        vanilla = vanilla_call_raw(self.resolved)
        image = barrier_image_raw(self.resolved)
        do = down_and_out_call_raw(self.resolved)
        di = down_and_in_call_raw(self.resolved)

        assert do == pytest.approx(vanilla - image, abs=1e-12)
        assert di == pytest.approx(image, abs=1e-12)
        assert do + di == pytest.approx(vanilla, abs=1e-12)

    def test_raw_down_and_out_call_vega_matches_finite_difference(self):
        autodiff_vega = gradient(
            lambda vol: down_and_out_call_raw(replace(self.resolved, sigma=vol))
        )(self.resolved.sigma)
        fd_vega = _finite_difference(
            lambda vol: down_and_out_call_raw(replace(self.resolved, sigma=vol)),
            self.resolved.sigma,
        )

        assert autodiff_vega == pytest.approx(fd_vega, rel=1e-6, abs=1e-8)

    def test_raw_down_and_out_call_delta_matches_finite_difference(self):
        autodiff_delta = gradient(
            lambda spot: down_and_out_call_raw(replace(self.resolved, spot=spot))
        )(self.resolved.spot)
        fd_delta = _finite_difference(
            lambda spot: down_and_out_call_raw(replace(self.resolved, spot=spot)),
            self.resolved.spot,
        )

        assert autodiff_delta == pytest.approx(fd_delta, rel=1e-6, abs=1e-8)

    def test_public_wrappers_and_generic_dispatch_stay_scalar_returning(self):
        do = down_and_out_call(
            self.resolved.spot,
            self.resolved.strike,
            self.resolved.barrier,
            self.resolved.rate,
            self.resolved.sigma,
            self.resolved.T,
        )
        di = down_and_in_call(
            self.resolved.spot,
            self.resolved.strike,
            self.resolved.barrier,
            self.resolved.rate,
            self.resolved.sigma,
            self.resolved.T,
        )
        generic = barrier_option_price(
            self.resolved.spot,
            self.resolved.strike,
            self.resolved.barrier,
            self.resolved.rate,
            self.resolved.sigma,
            self.resolved.T,
            barrier_type="down_and_out",
            option_type="call",
        )

        assert isinstance(do, float)
        assert isinstance(di, float)
        assert generic == pytest.approx(do, abs=1e-12)
