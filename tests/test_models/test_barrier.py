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


class TestBarrierOptionPriceDispatcher:
    """Full Reiner-Rubinstein generic dispatcher coverage."""

    def setup_method(self):
        # Standard parameters: spot above barrier, K > B
        self.S = 100.0
        self.K = 100.0
        self.B = 90.0
        self.r = 0.05
        self.sigma = 0.20
        self.T = 1.0

    # --- In/out parity ---

    def test_down_call_in_out_parity(self):
        vanilla = barrier_option_price(
            self.S, self.K, self.B, self.r, self.sigma, self.T,
            barrier_type="down_and_out", option_type="call",
        )
        knock_in = barrier_option_price(
            self.S, self.K, self.B, self.r, self.sigma, self.T,
            barrier_type="down_and_in", option_type="call",
        )
        # in + out == vanilla (Black-Scholes) for zero rebate
        from scipy.stats import norm as _norm
        import math
        d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * math.sqrt(self.T)
        )
        d2 = d1 - self.sigma * math.sqrt(self.T)
        bs_call = self.S * _norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * _norm.cdf(d2)
        assert vanilla + knock_in == pytest.approx(bs_call, rel=1e-6)

    def test_up_call_in_out_parity(self):
        """Up-barrier call: out + in should sum to vanilla (K < B, spot < B)."""
        S, K, B = 80.0, 75.0, 110.0
        out = barrier_option_price(S, K, B, self.r, self.sigma, self.T,
                                   barrier_type="up_and_out", option_type="call")
        inn = barrier_option_price(S, K, B, self.r, self.sigma, self.T,
                                   barrier_type="up_and_in", option_type="call")
        from scipy.stats import norm as _norm
        import math
        d1 = (math.log(S / K) + (self.r + 0.5 * self.sigma**2) * self.T) / (
            self.sigma * math.sqrt(self.T)
        )
        d2 = d1 - self.sigma * math.sqrt(self.T)
        bs_call = S * _norm.cdf(d1) - K * math.exp(-self.r * self.T) * _norm.cdf(d2)
        assert out + inn == pytest.approx(bs_call, rel=1e-3)

    # --- Edge cases ---

    def test_expired_option_returns_zero(self):
        price = barrier_option_price(
            self.S, self.K, self.B, self.r, self.sigma, T=0.0,
            barrier_type="down_and_out", option_type="call",
        )
        assert price == 0.0

    def test_spot_at_barrier_returns_rebate(self):
        """When spot ≤ barrier the down-and-out option is knocked out."""
        price = barrier_option_price(
            self.B, self.K, self.B, self.r, self.sigma, self.T,
            barrier_type="down_and_out", option_type="call",
            rebate=5.0,
        )
        assert price == pytest.approx(5.0, abs=1e-12)

    def test_generic_matches_public_wrapper_down_and_out(self):
        generic = barrier_option_price(
            self.S, self.K, self.B, self.r, self.sigma, self.T,
            barrier_type="down_and_out", option_type="call",
        )
        direct = down_and_out_call(self.S, self.K, self.B, self.r, self.sigma, self.T)
        assert generic == pytest.approx(direct, abs=1e-10)

    def test_generic_matches_public_wrapper_down_and_in(self):
        generic = barrier_option_price(
            self.S, self.K, self.B, self.r, self.sigma, self.T,
            barrier_type="down_and_in", option_type="call",
        )
        direct = down_and_in_call(self.S, self.K, self.B, self.r, self.sigma, self.T)
        assert generic == pytest.approx(direct, abs=1e-10)

    def test_k_le_barrier_down_and_out_call_nonnegative(self):
        """K ≤ B case: down-and-out call uses the B_val - D branch, price ≥ 0."""
        price = barrier_option_price(
            self.S, K=85.0, B=self.B, r=self.r, sigma=self.sigma, T=self.T,
            barrier_type="down_and_out", option_type="call",
        )
        assert price >= 0.0

    def test_unknown_barrier_type_raises(self):
        with pytest.raises(ValueError, match="Unknown barrier_type"):
            barrier_option_price(
                self.S, self.K, self.B, self.r, self.sigma, self.T,
                barrier_type="sideways_and_out", option_type="call",
            )
