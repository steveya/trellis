"""Tests for the generic recombining lattice."""

from dataclasses import FrozenInstanceError

import numpy as raw_np
import pytest
from scipy.stats import norm

from trellis.models.trees.lattice import (
    RecombiningLattice,
    build_rate_lattice,
    build_spot_lattice,
    lattice_backward_induction,
    lattice_backward_induction_result,
)
from trellis.models.trees.control import resolve_lattice_exercise_policy
from tests.lattice_builders import build_equity_lattice, build_short_rate_lattice


S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0


def bs_call(S, K, T, r, sigma):
    d1 = (raw_np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * raw_np.sqrt(T))
    d2 = d1 - sigma * raw_np.sqrt(T)
    return S * norm.cdf(d1) - K * raw_np.exp(-r * T) * norm.cdf(d2)


class TestRecombiningLattice:

    def test_construction(self):
        lat = RecombiningLattice(10, 0.1, branching=2, state_dim=1)
        assert lat.n_steps == 10
        assert lat.n_nodes(0) == 1
        assert lat.n_nodes(10) == 11

    def test_set_get_state(self):
        lat = RecombiningLattice(5, 0.2, state_dim=1)
        lat.set_state(0, 0, 100.0)
        assert lat.get_state(0, 0) == 100.0

    def test_multidim_state(self):
        lat = RecombiningLattice(5, 0.2, state_dim=2)
        lat.set_state(0, 0, (100.0, 50.0))
        assert lat.get_state(0, 0) == (100.0, 50.0)

    def test_trinomial_nodes(self):
        lat = RecombiningLattice(5, 0.2, branching=3)
        assert lat.n_nodes(0) == 1
        assert lat.n_nodes(5) == 11  # 2*5+1


class TestSpotLattice:

    def test_european_call_converges(self):
        """CRR spot lattice European call → BS."""
        n = 200
        lattice = build_equity_lattice(S0, r, sigma, T, n)

        def payoff(step, node, lat):
            return max(lat.get_state(step, node) - K, 0)

        price = lattice_backward_induction(lattice, payoff)
        bs_ref = bs_call(S0, K, T, r, sigma)
        assert price == pytest.approx(bs_ref, rel=0.02)

    def test_american_put_geq_european(self):
        n = 200
        lattice = build_equity_lattice(S0, r, sigma, T, n)

        def payoff(step, node, lat):
            return max(K - lat.get_state(step, node), 0)

        euro = lattice_backward_induction(lattice, payoff)
        amer = lattice_backward_induction(
            lattice, payoff, exercise_value=payoff, exercise_type="american",
        )
        assert amer >= euro - 0.01

    def test_jarrow_rudd_call_converges(self):
        lattice = build_equity_lattice(S0, r, sigma, T, 200, model="jarrow_rudd")

        def payoff(step, node, lat):
            return max(lat.get_state(step, node) - K, 0.0)

        price = lattice_backward_induction(lattice, payoff)
        bs_ref = bs_call(S0, K, T, r, sigma)
        assert price == pytest.approx(bs_ref, rel=0.03)


class TestRateLattice:

    def test_rate_at_root(self):
        lattice = build_short_rate_lattice(0.05, 0.01, 0.1, 1.0, 50)
        assert lattice.get_state(0, 0) == pytest.approx(0.05)

    def test_rate_dispersion_increases_with_vol(self):
        """Higher vol → wider rate dispersion at terminal step."""
        lat_low = build_short_rate_lattice(0.05, 0.005, 0.1, 5.0, 100)
        lat_high = build_short_rate_lattice(0.05, 0.02, 0.1, 5.0, 100)

        # Check dispersion at final step
        n = 100
        rates_low = [lat_low.get_state(n, j) for j in range(n + 1)]
        rates_high = [lat_high.get_state(n, j) for j in range(n + 1)]
        assert raw_np.std(rates_high) > raw_np.std(rates_low)

    def test_callable_bond_vol_sensitive(self):
        """A callable bond on a rate lattice MUST be vol-sensitive."""
        from datetime import date
        from trellis.core.date_utils import year_fraction
        from trellis.core.types import DayCountConvention

        T = 10.0
        r0 = 0.05
        notional = 100.0
        coupon = 0.05

        prices = []
        for vol in [0.005, 0.01, 0.02]:
            lattice = build_short_rate_lattice(r0, vol, 0.1, T, 100)
            dt = T / 100

            # Call steps at 3Y, 5Y, 7Y
            exercise_steps = [30, 50, 70]

            def payoff(step, node, lat):
                return notional + notional * coupon * dt

            def exercise(step, node, lat):
                return notional + notional * coupon * dt  # call at par + accrued

            p = lattice_backward_induction(
                lattice, payoff, exercise, "bermudan", exercise_steps,
            )
            prices.append(p)

        # Price must change meaningfully with vol
        total_change = abs(prices[-1] - prices[0])
        assert total_change > 0.01, f"Prices nearly unchanged with vol: {prices}"

    @pytest.mark.legacy_compat
    def test_exercise_policy_matches_legacy_kwargs(self):
        lattice = build_rate_lattice(0.05, 0.01, 0.1, 10.0, 100)
        dt = 10.0 / 100
        exercise_steps = [30, 50, 70]

        def payoff(step, node, lat):
            return 100.0 + 100.0 * 0.05 * dt

        def exercise(step, node, lat):
            return 100.0 + 100.0 * 0.05 * dt

        legacy = lattice_backward_induction(
            lattice,
            payoff,
            exercise,
            exercise_type="bermudan",
            exercise_steps=exercise_steps,
            exercise_fn=min,
        )
        policy = resolve_lattice_exercise_policy(
            "issuer_call",
            exercise_steps=exercise_steps,
        )
        policy_price = lattice_backward_induction(
            lattice,
            payoff,
            exercise,
            exercise_policy=policy,
        )

        assert policy_price == pytest.approx(legacy)

    @pytest.mark.legacy_compat
    def test_lattice_backward_induction_accepts_legacy_terminal_value_and_exercise_value_fn(self):
        lattice = build_rate_lattice(0.05, 0.01, 0.1, 1.0, 4)
        policy = resolve_lattice_exercise_policy("issuer_call", exercise_steps=[2, 3])

        def cashflow_at_node(step, node):
            return 1.0 if step > 0 else 0.0

        def exercise_value_fn(step, node, continuation):
            return 100.0 if continuation > 100.0 else 101.0

        price = lattice_backward_induction(
            lattice,
            terminal_value=101.0,
            cashflow_at_node=cashflow_at_node,
            exercise_value_fn=exercise_value_fn,
            exercise_policy=policy,
        )

        assert price > 0.0


class TestLatticeRollbackObservations:
    @staticmethod
    def _two_step_lattice() -> RecombiningLattice:
        lattice = RecombiningLattice(2, 1.0, branching=2, state_dim=1)
        for step in range(3):
            for node in range(lattice.n_nodes(step)):
                lattice.set_state(step, node, float(node))
        lattice.set_probabilities(0, 0, [0.5, 0.5])
        lattice.set_probabilities(1, 0, [0.5, 0.5])
        lattice.set_probabilities(1, 1, [0.5, 0.5])
        lattice.set_discount(0, 0, 1.0)
        lattice.set_discount(1, 0, 1.0)
        lattice.set_discount(1, 1, 1.0)
        return lattice

    def test_result_captures_continuation_cashflow_and_control_phases(self):
        lattice = self._two_step_lattice()

        result = lattice_backward_induction_result(
            lattice,
            terminal_payoff=lambda step, node, lattice_: 100.0 + 10.0 * node,
            cashflow_at_node=lambda step, node, lattice_: 5.0 if step == 1 else 0.0,
            exercise_value=lambda step, node, lattice_, continuation: (
                110.0 if node == 0 else 130.0
            ),
            exercise_type="bermudan",
            exercise_steps=[1],
            observation_steps=[1],
        )

        observation = result.observation_at(1)
        assert observation.continuation_values == pytest.approx((105.0, 115.0))
        assert observation.post_cashflow_values == pytest.approx((110.0, 120.0))
        assert observation.post_control_values == pytest.approx((110.0, 130.0))
        assert result.root_value == pytest.approx(120.0)

    def test_result_normalizes_requested_steps_and_captures_terminal_values(self):
        lattice = self._two_step_lattice()

        result = lattice_backward_induction_result(
            lattice,
            terminal_payoff=lambda step, node, lattice_: 100.0 + 10.0 * node,
            observation_steps=[2, 0, 1, 1],
        )

        assert tuple(observation.step for observation in result.observations) == (0, 1, 2)
        terminal = result.observation_at(2)
        assert terminal.continuation_values == pytest.approx((100.0, 110.0, 120.0))
        assert terminal.post_cashflow_values == terminal.continuation_values
        assert terminal.post_control_values == terminal.continuation_values

    @pytest.mark.parametrize("observation_steps", [(-1,), (3,), (0.5,), (True,)])
    def test_result_rejects_invalid_observation_steps(self, observation_steps):
        lattice = self._two_step_lattice()

        error = TypeError if observation_steps in {(0.5,), (True,)} else ValueError
        with pytest.raises(error):
            lattice_backward_induction_result(
                lattice,
                terminal_payoff=0.0,
                observation_steps=observation_steps,
            )

    def test_result_is_immutable_and_missing_observations_fail_closed(self):
        lattice = self._two_step_lattice()
        result = lattice_backward_induction_result(
            lattice,
            terminal_payoff=1.0,
            observation_steps=(1,),
        )

        with pytest.raises(FrozenInstanceError):
            result.root_value = 2.0
        with pytest.raises(TypeError):
            result.observation_at(1).continuation_values[0] = 2.0
        with pytest.raises(KeyError, match="No lattice rollback observation"):
            result.observation_at(0)

    @pytest.mark.parametrize(
        ("exercise_fn", "exercise_value"),
        [
            (max, 130.0),
            (min, 90.0),
        ],
    )
    def test_result_root_matches_scalar_backward_induction(self, exercise_fn, exercise_value):
        lattice = self._two_step_lattice()
        kwargs = {
            "terminal_payoff": lambda step, node, lattice_: 100.0 + 10.0 * node,
            "cashflow_at_node": lambda step, node, lattice_: 2.0 if step == 1 else 0.0,
            "exercise_value": lambda step, node, lattice_, continuation: exercise_value,
            "exercise_type": "bermudan",
            "exercise_steps": [1],
            "exercise_fn": exercise_fn,
        }

        result = lattice_backward_induction_result(
            lattice,
            observation_steps=(1,),
            **kwargs,
        )
        scalar = lattice_backward_induction(lattice, **kwargs)

        assert result.root_value == pytest.approx(scalar)


class TestLatticeBackwardInductionCompatibility:
    @pytest.mark.legacy_compat
    def test_lattice_backward_induction_accepts_legacy_callable_signatures(self):
        lattice = build_rate_lattice(0.05, 0.01, 0.1, 1.0, 4)
        policy = resolve_lattice_exercise_policy("issuer_call", exercise_steps=[2, 3])

        def terminal_payoff(node, _t):
            return 101.0

        def cashflow_at_node(step, node, _t):
            return 1.0 if step > 0 else 0.0

        def exercise_value(step, node, continuation, _t):
            return 100.0 if continuation > 100.0 else 101.0

        price = lattice_backward_induction(
            lattice,
            terminal_payoff=terminal_payoff,
            cashflow_at_node=cashflow_at_node,
            exercise_value=exercise_value,
            exercise_policy=policy,
        )

        assert price > 0.0

    def test_lattice_backward_induction_accepts_scalar_terminal_payoff(self):
        lattice = build_short_rate_lattice(0.05, 0.01, 0.1, 1.0, 4)
        policy = resolve_lattice_exercise_policy("issuer_call", exercise_steps=[2, 3])

        def cashflow_at_node(step, node):
            return 1.0 if step > 0 else 0.0

        def exercise_value(step, node):
            return 100.0

        price = lattice_backward_induction(
            lattice,
            101.0,
            exercise_value=exercise_value,
            cashflow_at_node=cashflow_at_node,
            exercise_policy=policy,
        )

        assert price > 0.0
