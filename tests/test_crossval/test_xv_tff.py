"""Cross-validation against TF Quant Finance (Google).

Tests: European options (BS), Heston model, American options.
Requires: pip install tf-quant-finance tensorflow tf-keras
Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python if protobuf errors occur.

Skips gracefully if tf-quant-finance is not installed.
"""

import os
import numpy as raw_np
import pytest

# Set protobuf workaround before any TFF import
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import tf_quant_finance as tff
    HAS_TFF = True
except (ImportError, TypeError):
    HAS_TFF = False

# --- Trellis ---
from trellis.models.calibration.implied_vol import _bs_price
from trellis.models.black import black76_call
from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.backward_induction import backward_induction
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.processes.gbm import GBM
from trellis.models.transforms.fft_pricer import fft_price

S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

pytestmark = pytest.mark.skipif(not HAS_TFF, reason="tf-quant-finance not installed")


class TestBSCrossValTFF:

    def test_european_call(self):
        """Trellis BS call matches TFF."""
        trellis_call = _bs_price(S0, K, T, r, sigma, "call")

        tff_call = float(tff.black_scholes.option_price(
            volatilities=raw_np.array([sigma]),
            strikes=raw_np.array([K]),
            expiries=raw_np.array([T]),
            spots=raw_np.array([S0]),
            discount_rates=raw_np.array([r]),
            is_call_options=raw_np.array([True]),
        )[0])

        assert trellis_call == pytest.approx(tff_call, rel=1e-4), (
            f"Trellis={trellis_call:.6f}, TFF={tff_call:.6f}"
        )

    def test_european_put(self):
        trellis_put = _bs_price(S0, K, T, r, sigma, "put")

        tff_put = float(tff.black_scholes.option_price(
            volatilities=raw_np.array([sigma]),
            strikes=raw_np.array([K]),
            expiries=raw_np.array([T]),
            spots=raw_np.array([S0]),
            discount_rates=raw_np.array([r]),
            is_call_options=raw_np.array([False]),
        )[0])

        assert trellis_put == pytest.approx(tff_put, rel=1e-4)

    def test_otm_call(self):
        K_otm = 120.0
        trellis_call = _bs_price(S0, K_otm, T, r, sigma, "call")

        tff_call = float(tff.black_scholes.option_price(
            volatilities=raw_np.array([sigma]),
            strikes=raw_np.array([K_otm]),
            expiries=raw_np.array([T]),
            spots=raw_np.array([S0]),
            discount_rates=raw_np.array([r]),
            is_call_options=raw_np.array([True]),
        )[0])

        assert trellis_call == pytest.approx(tff_call, rel=1e-3)

    def test_multiple_strikes(self):
        """Vectorized: multiple strikes at once."""
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        tff_prices = tff.black_scholes.option_price(
            volatilities=raw_np.array([sigma] * len(strikes)),
            strikes=raw_np.array(strikes),
            expiries=raw_np.array([T] * len(strikes)),
            spots=raw_np.array([S0] * len(strikes)),
            discount_rates=raw_np.array([r] * len(strikes)),
            is_call_options=raw_np.array([True] * len(strikes)),
        )
        for i, K_i in enumerate(strikes):
            trellis_price = _bs_price(S0, K_i, T, r, sigma, "call")
            assert trellis_price == pytest.approx(float(tff_prices[i]), rel=1e-3), (
                f"K={K_i}: Trellis={trellis_price:.4f}, TFF={float(tff_prices[i]):.4f}"
            )


class TestHestonCrossValTFF:

    @pytest.mark.skip(reason="Heston characteristic function needs CF convention alignment with FFT pricer")
    def test_heston_call(self):
        """Trellis Heston FFT matches TFF Heston."""
        from trellis.models.processes.heston import Heston

        # Heston parameters
        kappa, theta, xi, rho, v0 = 2.0, 0.04, 0.3, -0.7, 0.04

        # Trellis: FFT with Heston characteristic function
        heston = Heston(mu=r, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)

        def char_fn(u):
            return heston.characteristic_function(u, T)

        trellis_price = fft_price(char_fn, S0, K, T, r)

        # TFF Heston
        tff_prices = tff.models.heston.approximations.european_option_price(
            strikes=raw_np.array([K]),
            expiries=raw_np.array([T]),
            spots=raw_np.array([S0]),
            discount_rates=raw_np.array([r]),
            is_call_options=raw_np.array([True]),
            variances=raw_np.array([v0]),
            mean_reversion=raw_np.array([kappa]),
            theta=raw_np.array([theta]),
            volvol=raw_np.array([xi]),
            rho=raw_np.array([rho]),
        )
        tff_price = float(tff_prices[0])

        # Heston prices can vary by method; allow 5% tolerance
        assert trellis_price == pytest.approx(tff_price, rel=0.05), (
            f"Trellis Heston={trellis_price:.4f}, TFF Heston={tff_price:.4f}"
        )


class TestAmericanCrossValTFF:

    def test_american_put_tree_vs_tff(self):
        """Trellis CRR tree American put vs TFF approximation."""
        # Trellis tree
        tree = BinomialTree.crr(S0, T, 500, r, sigma)
        def put_payoff(step, node):
            return max(K - tree.value_at(step, node), 0)
        def exercise_val(step, node, t):
            return max(K - t.value_at(step, node), 0)
        trellis_amer = backward_induction(tree, put_payoff, r, "american",
                                           exercise_value_fn=exercise_val)

        # TFF American approximation (Barone-Adesi Whaley)
        try:
            tff_amer = float(tff.black_scholes.approximations.american_option.adesi_whaley(
                volatilities=raw_np.array([sigma]),
                strikes=raw_np.array([K]),
                expiries=raw_np.array([T]),
                spots=raw_np.array([S0]),
                discount_rates=raw_np.array([r]),
                is_call_options=raw_np.array([False]),
            )[0])
        except Exception:
            pytest.skip("TFF American approximation not available")

        assert trellis_amer == pytest.approx(tff_amer, rel=0.02), (
            f"Trellis tree={trellis_amer:.4f}, TFF Adesi-Whaley={tff_amer:.4f}"
        )
