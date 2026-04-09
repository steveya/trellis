"""Tests for the generic event-aware 1D PDE rollback substrate."""

from __future__ import annotations

import math

import numpy as raw_np
import pytest

from trellis.models.pde.grid import Grid
from trellis.models.pde.operator import BlackScholesOperator
from trellis.models.pde.rate_operator import HullWhitePDEOperator
from trellis.models.pde.theta_method import _theta_step, theta_method_1d


def _interp_grid(values: raw_np.ndarray, x: raw_np.ndarray, x0: float) -> float:
    idx = raw_np.searchsorted(x, x0)
    idx = max(1, min(idx, len(x) - 1))
    weight = (x0 - x[idx - 1]) / (x[idx] - x[idx - 1])
    return float(values[idx - 1] * (1.0 - weight) + values[idx] * weight)


class TestEventAwarePDEAssembly:
    @pytest.mark.parametrize(
        ("family", "kwargs", "operator_type"),
        [
            (
                "black_scholes_1d",
                {"sigma": 0.20, "r": 0.05},
                BlackScholesOperator,
            ),
            (
                "local_vol_1d",
                {
                    "sigma_fn": lambda s, t: 0.15 + 0.05 * raw_np.exp(-0.5 * t),
                    "r": 0.03,
                },
                BlackScholesOperator,
            ),
            (
                "hull_white_1f",
                {
                    "sigma": 0.01,
                    "mean_reversion": 0.10,
                    "theta_fn": lambda t: 0.05,
                },
                HullWhitePDEOperator,
            ),
        ],
    )
    def test_build_problem_resolves_supported_operator_families(self, family, kwargs, operator_type):
        from trellis.models.pde.event_aware import (
            EventAwarePDEBoundarySpec,
            EventAwarePDEGridSpec,
            EventAwarePDEOperatorSpec,
            EventAwarePDEProblemSpec,
            build_event_aware_pde_problem,
        )

        spec = EventAwarePDEProblemSpec(
            grid_spec=EventAwarePDEGridSpec(
                x_min=0.0 if family != "hull_white_1f" else -0.05,
                x_max=300.0 if family != "hull_white_1f" else 0.15,
                n_x=61,
                maturity=1.0,
                n_t=40,
            ),
            operator_spec=EventAwarePDEOperatorSpec(family=family, **kwargs),
            terminal_condition=lambda x: raw_np.maximum(x - 100.0, 0.0),
            boundary_spec=EventAwarePDEBoundarySpec(
                lower=lambda t: 0.0,
                upper=lambda t: 200.0,
            ),
        )

        problem = build_event_aware_pde_problem(spec)

        assert isinstance(problem.operator, operator_type)
        assert problem.grid.n_x == 61
        assert problem.grid.n_t == 40
        assert problem.terminal_condition.shape == (61,)

    def test_build_problem_rejects_unsupported_operator_family(self):
        from trellis.models.pde.event_aware import (
            EventAwarePDEBoundarySpec,
            EventAwarePDEGridSpec,
            EventAwarePDEOperatorSpec,
            EventAwarePDEProblemSpec,
            build_event_aware_pde_problem,
        )

        spec = EventAwarePDEProblemSpec(
            grid_spec=EventAwarePDEGridSpec(
                x_min=0.0,
                x_max=100.0,
                n_x=21,
                maturity=1.0,
                n_t=20,
            ),
            operator_spec=EventAwarePDEOperatorSpec(family="unsupported"),
            terminal_condition=lambda x: raw_np.maximum(x - 50.0, 0.0),
            boundary_spec=EventAwarePDEBoundarySpec(lower=lambda t: 0.0, upper=lambda t: 50.0),
        )

        with pytest.raises(ValueError, match="Unsupported PDE operator family"):
            build_event_aware_pde_problem(spec)

    def test_build_problem_rejects_unsupported_boundary_policy(self):
        from trellis.models.pde.event_aware import (
            EventAwarePDEBoundarySpec,
            EventAwarePDEGridSpec,
            EventAwarePDEOperatorSpec,
            EventAwarePDEProblemSpec,
            build_event_aware_pde_problem,
        )

        spec = EventAwarePDEProblemSpec(
            grid_spec=EventAwarePDEGridSpec(
                x_min=0.0,
                x_max=100.0,
                n_x=21,
                maturity=1.0,
                n_t=20,
            ),
            operator_spec=EventAwarePDEOperatorSpec(family="black_scholes_1d", sigma=0.2, r=0.05),
            terminal_condition=lambda x: raw_np.maximum(x - 50.0, 0.0),
            boundary_spec=EventAwarePDEBoundarySpec(
                lower=lambda t: 0.0,
                upper=lambda t: 50.0,
                post_step_policy="unsupported",
            ),
        )

        with pytest.raises(ValueError, match="Unsupported boundary post_step_policy"):
            build_event_aware_pde_problem(spec)


class TestEventAwarePDETransforms:
    def test_apply_event_bucket_orders_cashflow_then_min_projection(self):
        from trellis.models.pde.event_aware import (
            EventAwarePDEEventBucket,
            EventAwarePDETransform,
            apply_event_bucket,
        )

        values = raw_np.array([99.0, 100.0, 101.5, 104.0, 107.0])
        x = raw_np.linspace(0.0, 4.0, len(values))
        bucket = EventAwarePDEEventBucket(
            time=0.5,
            transforms=(
                EventAwarePDETransform(kind="add_cashflow", payload=5.0),
                EventAwarePDETransform(kind="project_min", payload=102.0),
            ),
        )

        updated = apply_event_bucket(values, x, bucket.time, bucket)

        raw_np.testing.assert_allclose(updated, raw_np.array([102.0, 102.0, 102.0, 102.0, 102.0]))

    def test_state_remap_transform_uses_callable_payload(self):
        from trellis.models.pde.event_aware import EventAwarePDETransform, apply_event_transform

        x = raw_np.linspace(0.0, 4.0, 5)
        values = raw_np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        transform = EventAwarePDETransform(
            kind="state_remap",
            payload=lambda x_grid, current, t: raw_np.interp(
                raw_np.clip(x_grid + 0.5, x_grid[0], x_grid[-1]),
                x_grid,
                current,
            ),
        )

        updated = apply_event_transform(values, x, 0.25, transform)

        assert updated[0] > values[0]
        assert updated[-1] == pytest.approx(values[-1])

    def test_build_problem_rejects_unsupported_transform_kind(self):
        from trellis.models.pde.event_aware import (
            EventAwarePDEBoundarySpec,
            EventAwarePDEEventBucket,
            EventAwarePDEGridSpec,
            EventAwarePDEOperatorSpec,
            EventAwarePDEProblemSpec,
            EventAwarePDETransform,
            build_event_aware_pde_problem,
        )

        spec = EventAwarePDEProblemSpec(
            grid_spec=EventAwarePDEGridSpec(
                x_min=0.0,
                x_max=100.0,
                n_x=21,
                maturity=1.0,
                n_t=20,
            ),
            operator_spec=EventAwarePDEOperatorSpec(family="black_scholes_1d", sigma=0.2, r=0.05),
            terminal_condition=lambda x: raw_np.maximum(x - 50.0, 0.0),
            boundary_spec=EventAwarePDEBoundarySpec(lower=lambda t: 0.0, upper=lambda t: 50.0),
            event_buckets=(
                EventAwarePDEEventBucket(
                    time=0.5,
                    transforms=(EventAwarePDETransform(kind="unsupported"),),
                ),
            ),
        )

        with pytest.raises(ValueError, match="Unsupported event transform kind"):
            build_event_aware_pde_problem(spec)


class TestEventAwarePDERollback:
    def test_identity_problem_matches_direct_theta_method(self):
        from trellis.models.pde.event_aware import (
            EventAwarePDEBoundarySpec,
            EventAwarePDEGridSpec,
            EventAwarePDEOperatorSpec,
            EventAwarePDEProblemSpec,
            build_event_aware_pde_problem,
            solve_event_aware_pde,
        )

        S0 = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.20
        maturity = 1.0
        s_max = 4.0 * S0

        spec = EventAwarePDEProblemSpec(
            grid_spec=EventAwarePDEGridSpec(
                x_min=0.0,
                x_max=s_max,
                n_x=201,
                maturity=maturity,
                n_t=200,
            ),
            operator_spec=EventAwarePDEOperatorSpec(
                family="black_scholes_1d",
                sigma=sigma,
                r=r,
            ),
            terminal_condition=lambda x: raw_np.maximum(x - K, 0.0),
            boundary_spec=EventAwarePDEBoundarySpec(
                lower=lambda t: 0.0,
                upper=lambda t: s_max - K * raw_np.exp(-r * (maturity - t)),
            ),
            theta=0.5,
        )

        problem = build_event_aware_pde_problem(spec)
        values = solve_event_aware_pde(problem)

        direct_grid = Grid(x_min=0.0, x_max=s_max, n_x=201, T=maturity, n_t=200)
        direct_operator = BlackScholesOperator(lambda s, t: sigma, lambda t: r)
        direct_values = theta_method_1d(
            direct_grid,
            direct_operator,
            raw_np.maximum(direct_grid.x - K, 0.0),
            theta=0.5,
            lower_bc_fn=lambda t: 0.0,
            upper_bc_fn=lambda t: s_max - K * raw_np.exp(-r * (maturity - t)),
        )

        raw_np.testing.assert_allclose(values, direct_values, atol=1e-10, rtol=1e-10)
        assert _interp_grid(values, problem.grid.x, S0) == pytest.approx(
            _interp_grid(direct_values, direct_grid.x, S0),
            rel=1e-10,
            abs=1e-10,
        )

    def test_hull_white_event_rollback_matches_manual_loop(self):
        from trellis.models.pde.event_aware import (
            EventAwarePDEBoundarySpec,
            EventAwarePDEEventBucket,
            EventAwarePDEGridSpec,
            EventAwarePDEOperatorSpec,
            EventAwarePDEProblemSpec,
            EventAwarePDETransform,
            apply_event_bucket,
            build_event_aware_pde_problem,
            solve_event_aware_pde,
        )

        maturity = 2.0
        n_t = 80
        n_x = 81
        dt = maturity / n_t

        spec = EventAwarePDEProblemSpec(
            grid_spec=EventAwarePDEGridSpec(
                x_min=-0.05,
                x_max=0.15,
                n_x=n_x,
                maturity=maturity,
                n_t=n_t,
            ),
            operator_spec=EventAwarePDEOperatorSpec(
                family="hull_white_1f",
                sigma=0.01,
                mean_reversion=0.10,
                theta_fn=lambda t: 0.05,
            ),
            terminal_condition=lambda x: raw_np.full_like(x, 100.0),
            boundary_spec=EventAwarePDEBoundarySpec(
                lower=lambda t, values, grid: values[0] * math.exp(-grid.x[0] * grid.dt),
                upper=lambda t, values, grid: values[-1] * math.exp(-grid.x[-1] * grid.dt),
                post_step_policy="linear_extrapolation",
            ),
            event_buckets=(
                EventAwarePDEEventBucket(
                    time=1.0,
                    transforms=(
                        EventAwarePDETransform(kind="add_cashflow", payload=2.5),
                        EventAwarePDETransform(kind="project_min", payload=102.5),
                    ),
                ),
            ),
            theta=0.5,
        )

        problem = build_event_aware_pde_problem(spec)
        values = solve_event_aware_pde(problem)

        operator = HullWhitePDEOperator(sigma=0.01, a=0.10, theta_fn=lambda t: 0.05)
        manual = raw_np.full(n_x, 100.0)
        event_step = int(round(1.0 / dt))
        n_int = n_x - 2
        for step in range(n_t - 1, -1, -1):
            t = step * dt
            a_coeff, b_coeff, c_coeff = operator.coefficients(problem.grid.x, t, dt)
            lower = manual[0] * math.exp(-problem.grid.x[0] * dt)
            upper = manual[-1] * math.exp(-problem.grid.x[-1] * dt)
            manual = _theta_step(manual, a_coeff, b_coeff, c_coeff, 0.5, lower, upper, n_int)
            manual[0] = 2.0 * manual[1] - manual[2]
            manual[-1] = 2.0 * manual[-2] - manual[-3]
            if step == event_step:
                manual = apply_event_bucket(manual, problem.grid.x, t, problem.event_buckets[0])

        raw_np.testing.assert_allclose(values, manual, atol=1e-10, rtol=1e-10)
