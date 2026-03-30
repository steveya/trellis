"""Tests for binomial/trinomial trees and backward induction."""

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.core.differentiable import gradient, get_numpy
from trellis.models.trees.binomial import BinomialTree
from trellis.models.trees.trinomial import TrinomialTree
from trellis.models.trees.backward_induction import backward_induction

np = get_numpy()


def bs_call(S, K, T, r, sigma):
    """Black-Scholes European call price."""
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return S * norm.cdf(d1) - K * raw_np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    """Black-Scholes European put price."""
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return K * raw_np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------------
# BinomialTree.crr
# ---------------------------------------------------------------------------


class TestBinomialCRR:
    def test_terminal_nodes_count(self):
        """CRR tree with n steps has n+1 terminal nodes."""
        n = 50
        tree = BinomialTree.crr(S0=100, T=1.0, n_steps=n, r=0.05, sigma=0.20)
        assert len(tree.terminal_values()) == n + 1

    def test_u_times_d_approx_one(self):
        """CRR: u * d = 1."""
        tree = BinomialTree.crr(S0=100, T=1.0, n_steps=100, r=0.05, sigma=0.20)
        assert tree.u * tree.d == pytest.approx(1.0, rel=1e-12)

    def test_p_between_zero_and_one(self):
        tree = BinomialTree.crr(S0=100, T=1.0, n_steps=100, r=0.05, sigma=0.20)
        assert 0 < tree.p < 1

    def test_initial_value(self):
        tree = BinomialTree.crr(S0=100, T=1.0, n_steps=50, r=0.05, sigma=0.20)
        assert tree.value_at(0, 0) == pytest.approx(100.0, rel=1e-12)


# ---------------------------------------------------------------------------
# BinomialTree.jarrow_rudd
# ---------------------------------------------------------------------------


class TestBinomialJarrowRudd:
    def test_p_equals_half(self):
        tree = BinomialTree.jarrow_rudd(S0=100, T=1.0, n_steps=100, r=0.05, sigma=0.20)
        assert tree.p == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# European call via backward_induction converges to BS
# ---------------------------------------------------------------------------


class TestBackwardInductionEuropeanCall:
    def test_european_call_converges_to_bs(self):
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        n_steps = 200
        tree = BinomialTree.crr(S0, T, n_steps, r, sigma)

        def payoff(step, node):
            return max(tree.value_at(step, node) - K, 0.0)

        price = backward_induction(tree, payoff, discount_rate=r, exercise_type="european")
        bs_ref = bs_call(S0, K, T, r, sigma)
        assert price == pytest.approx(bs_ref, rel=0.01)

    def test_european_put_converges_to_bs(self):
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        n_steps = 200
        tree = BinomialTree.crr(S0, T, n_steps, r, sigma)

        def payoff(step, node):
            return max(K - tree.value_at(step, node), 0.0)

        price = backward_induction(tree, payoff, discount_rate=r, exercise_type="european")
        bs_ref = bs_put(S0, K, T, r, sigma)
        assert price == pytest.approx(bs_ref, rel=0.01)


class TestBackwardInductionDifferentiable:
    def test_binomial_forward_price_has_unit_spot_delta(self):
        """Differentiable tree rollback should preserve a linear payoff exactly."""
        S0, r, sigma, T = 100.0, 0.05, 0.20, 1.0
        n_steps = 200

        def price_from_spot(spot):
            tree = BinomialTree.crr(spot, T, n_steps, r, sigma)

            def payoff(step, node):
                return tree.value_at(n_steps, node)

            return backward_induction(
                tree,
                payoff,
                discount_rate=r,
                exercise_type="european",
                differentiable=True,
            )

        delta = gradient(price_from_spot)(S0)
        assert price_from_spot(S0) == pytest.approx(S0, rel=1e-12, abs=1e-12)
        assert delta == pytest.approx(1.0, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# American put >= European put
# ---------------------------------------------------------------------------


class TestAmericanVsEuropean:
    def test_american_put_geq_european_put(self):
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        n_steps = 100
        tree = BinomialTree.crr(S0, T, n_steps, r, sigma)

        def payoff(step, node):
            return max(K - tree.value_at(step, node), 0.0)

        def exercise_val(step, node, t):
            return max(K - t.value_at(step, node), 0.0)

        euro_price = backward_induction(tree, payoff, discount_rate=r,
                                        exercise_type="european")
        amer_price = backward_induction(tree, payoff, discount_rate=r,
                                        exercise_type="american",
                                        exercise_value_fn=exercise_val)
        assert amer_price >= euro_price - 1e-10


# ---------------------------------------------------------------------------
# Put-call parity on tree
# ---------------------------------------------------------------------------


class TestPutCallParity:
    def test_put_call_parity(self):
        """call - put ~ S0 - K*exp(-rT)."""
        S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
        n_steps = 200
        tree = BinomialTree.crr(S0, T, n_steps, r, sigma)

        def call_payoff(step, node):
            return max(tree.value_at(step, node) - K, 0.0)

        def put_payoff(step, node):
            return max(K - tree.value_at(step, node), 0.0)

        call_price = backward_induction(tree, call_payoff, discount_rate=r,
                                        exercise_type="european")
        put_price = backward_induction(tree, put_payoff, discount_rate=r,
                                       exercise_type="european")
        parity_rhs = S0 - K * raw_np.exp(-r * T)
        assert call_price - put_price == pytest.approx(parity_rhs, rel=0.01)


# ---------------------------------------------------------------------------
# TrinomialTree
# ---------------------------------------------------------------------------


class TestTrinomialTree:
    def test_probabilities_sum_to_one(self):
        tree = TrinomialTree.standard(S0=100, T=1.0, n_steps=50, r=0.05, sigma=0.20)
        assert tree.pu + tree.pm + tree.pd == pytest.approx(1.0, abs=1e-12)

    def test_probabilities_positive(self):
        tree = TrinomialTree.standard(S0=100, T=1.0, n_steps=50, r=0.05, sigma=0.20)
        assert tree.pu > 0
        assert tree.pm > 0
        assert tree.pd > 0

    def test_initial_value(self):
        tree = TrinomialTree.standard(S0=100, T=1.0, n_steps=50, r=0.05, sigma=0.20)
        assert tree.value_at(0, 0) == pytest.approx(100.0, rel=1e-12)
