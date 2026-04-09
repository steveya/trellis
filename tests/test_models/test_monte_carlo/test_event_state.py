"""Tests for the reusable path-event Monte Carlo substrate."""

from __future__ import annotations

import numpy as np
import pytest


def test_event_step_indices_and_requirement_are_deterministic():
    from trellis.models.monte_carlo.event_state import (
        build_event_path_requirement,
        event_step_indices,
    )

    assert event_step_indices((0.5, 1.0), 1.0, 2) == (1, 2)
    requirement = build_event_path_requirement((0.5, 1.0), 1.0, 2)
    assert requirement.snapshot_steps == (1, 2)


def test_ranked_observation_event_state_tracks_selection_and_settlement():
    from trellis.models.monte_carlo.event_state import (
        PathEventSpec,
        PathEventTimeline,
        replay_path_event_timeline,
    )

    paths = np.array(
        [
            [
                [100.0, 100.0],
                [110.0, 105.0],
                [120.0, 112.0],
            ]
        ],
        dtype=float,
    )
    timeline = PathEventTimeline(
        (
            PathEventSpec(
                name="basket_settlement",
                kind="settlement",
                step=2,
                priority=1,
                payload={"rule": "average_locked_returns"},
            ),
            PathEventSpec(
                name="observation_1",
                kind="observation",
                step=1,
                payload={
                    "selection_rule": "best_of_remaining",
                    "lock_rule": "remove_selected",
                    "selection_count": 1,
                },
            ),
            PathEventSpec(
                name="observation_2",
                kind="observation",
                step=2,
                payload={
                    "selection_rule": "best_of_remaining",
                    "lock_rule": "remove_selected",
                    "selection_count": 1,
                },
            ),
        )
    )

    state = replay_path_event_timeline(
        (
            paths[:, 1, :],
            paths[:, 2, :],
            paths[:, 2, :],
        ),
        initial_values=np.array([100.0, 100.0], dtype=float),
        event_timeline=timeline,
    )

    assert timeline.steps == (1, 2, 2)
    assert state.event_count == 3
    np.testing.assert_array_equal(state.selected_counts, np.array([2.0]))
    np.testing.assert_allclose(state.locked_returns, np.array([0.22]))
    np.testing.assert_allclose(state.settlement_value("basket_settlement"), np.array([0.11]))
    np.testing.assert_array_equal(state.selected_indices["observation_1"], np.array([0]))
    np.testing.assert_array_equal(state.selected_indices["observation_2"], np.array([1]))


def test_barrier_event_state_tracks_hit_and_knock_out_settlement():
    from trellis.models.monte_carlo.event_state import (
        PathEventSpec,
        PathEventTimeline,
        replay_path_event_timeline,
    )

    paths = np.array(
        [
            [100.0, 95.0, 92.0],
            [100.0, 97.0, 104.0],
        ],
        dtype=float,
    )
    timeline = PathEventTimeline(
        (
            PathEventSpec(
                name="barrier_settlement",
                kind="settlement",
                step=2,
                priority=1,
                payload={"rule": "knock_out_terminal", "barrier_event": "down_barrier"},
            ),
            PathEventSpec(
                name="down_barrier",
                kind="barrier",
                step=1,
                payload={"direction": "down", "level": 96.0},
            ),
        )
    )

    state = replay_path_event_timeline(
        (
            paths[:, 1],
            paths[:, 2],
        ),
        initial_values=100.0,
        event_timeline=timeline,
    )

    np.testing.assert_array_equal(state.barrier_hit("down_barrier"), np.array([True, False]))
    np.testing.assert_array_equal(state.settlement_value("barrier_settlement"), np.array([0.0, 104.0]))


def test_callable_event_state_tracks_coupon_and_exercise_settlement():
    from trellis.models.monte_carlo.event_state import (
        PathEventSpec,
        PathEventTimeline,
        replay_path_event_timeline,
    )

    paths = np.array(
        [
            [100.0, 102.0, 103.0],
            [100.0, 98.0, 96.0],
        ],
        dtype=float,
    )
    timeline = PathEventTimeline(
        (
            PathEventSpec(
                name="coupon_1",
                kind="coupon",
                step=1,
                payload={"amount": 5.0},
            ),
            PathEventSpec(
                name="issuer_call",
                kind="exercise",
                step=2,
                payload={
                    "exercise_rule": "issuer_call",
                    "direction": "up",
                    "threshold": 100.0,
                    "exercise_value": 100.0,
                },
            ),
            PathEventSpec(
                name="callable_settlement",
                kind="settlement",
                step=2,
                priority=1,
                payload={
                    "rule": "exercise_or_terminal",
                    "exercise_event": "issuer_call",
                    "coupon_events": ("coupon_1",),
                },
            ),
        )
    )

    state = replay_path_event_timeline(
        (
            paths[:, 1],
            paths[:, 2],
            paths[:, 2],
        ),
        initial_values=100.0,
        event_timeline=timeline,
    )

    np.testing.assert_array_equal(state.coupon_cashflow("coupon_1"), np.array([5.0, 5.0]))
    np.testing.assert_array_equal(state.exercise_triggered("issuer_call"), np.array([True, False]))
    np.testing.assert_array_equal(state.settlement_value("callable_settlement"), np.array([105.0, 101.0]))


def test_path_event_timeline_sorts_events_deterministically():
    from trellis.models.monte_carlo.event_state import PathEventSpec, PathEventTimeline

    timeline = PathEventTimeline(
        (
            PathEventSpec(name="late", kind="settlement", step=2, priority=1),
            PathEventSpec(name="early", kind="observation", step=1, priority=0),
        )
    )

    assert [event.name for event in timeline] == ["early", "late"]
    assert timeline.steps == (1, 2)


def test_path_event_state_rejects_invalid_directions():
    from trellis.models.monte_carlo.event_state import (
        PathEventSpec,
        PathEventTimeline,
        replay_path_event_timeline,
    )

    barrier_timeline = PathEventTimeline(
        (
            PathEventSpec(
                name="bad_barrier",
                kind="barrier",
                step=1,
                payload={"direction": "sideways", "level": 100.0},
            ),
        )
    )

    with pytest.raises(ValueError, match="Unsupported direction"):
        replay_path_event_timeline((np.array([100.0], dtype=float),), 100.0, barrier_timeline)


def test_discounted_swap_pv_settlement_uses_short_rate_anchor_and_discount_reducer():
    from trellis.models.monte_carlo.event_state import (
        PathEventSpec,
        PathEventTimeline,
        replay_path_event_timeline,
    )

    exercise_time = 1.0
    payment_times = (1.5, 2.0)
    accrual_fractions = (0.5, 0.5)
    anchor_discount_to_exercise = 0.96
    anchor_discount_factors = (0.94, 0.92)
    anchor_short_rate = -np.log(anchor_discount_to_exercise) / exercise_time
    mean_reversion = 0.10
    curve_basis_spread = 0.002
    short_rate_at_exercise = np.array([0.050, 0.032], dtype=float)
    discount_to_exercise = np.array([0.97, 0.99], dtype=float)

    timeline = PathEventTimeline(
        (
            PathEventSpec(
                name="expiry_settlement",
                kind="settlement",
                step=2,
                payload={
                    "rule": "discounted_swap_pv",
                    "discount_reducer_name": "discount_to_expiry",
                    "exercise_time": exercise_time,
                    "payment_times": payment_times,
                    "accrual_fractions": accrual_fractions,
                    "anchor_discount_to_exercise": anchor_discount_to_exercise,
                    "anchor_discount_factors": anchor_discount_factors,
                    "anchor_short_rate": anchor_short_rate,
                    "mean_reversion": mean_reversion,
                    "curve_basis_spread": curve_basis_spread,
                    "notional": 100.0,
                    "strike": 0.045,
                    "is_payer": True,
                },
            ),
        )
    )

    state = replay_path_event_timeline(
        (short_rate_at_exercise,),
        initial_values=0.04,
        event_timeline=timeline,
        reducer_values={"discount_to_expiry": discount_to_exercise},
    )

    def _expected_settlement(short_rate: float, discount_factor: float) -> float:
        taus = np.asarray(payment_times, dtype=float) - exercise_time
        B = (1.0 - np.exp(-mean_reversion * taus)) / mean_reversion
        anchor_ratio = np.asarray(anchor_discount_factors, dtype=float) / anchor_discount_to_exercise
        bond_prices = anchor_ratio * np.exp(-B * (short_rate - anchor_short_rate))
        annuity = float(np.sum(np.asarray(accrual_fractions, dtype=float) * bond_prices))
        forward = (1.0 - float(bond_prices[-1])) / annuity
        adjusted_forward = forward + curve_basis_spread
        intrinsic = max(adjusted_forward - 0.045, 0.0)
        return discount_factor * 100.0 * annuity * intrinsic

    expected = np.asarray(
        [
            _expected_settlement(short_rate_at_exercise[0], discount_to_exercise[0]),
            _expected_settlement(short_rate_at_exercise[1], discount_to_exercise[1]),
        ],
        dtype=float,
    )

    np.testing.assert_allclose(
        state.settlement_value("expiry_settlement"),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
