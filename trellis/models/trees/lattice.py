"""Generic recombining lattice for tree-based pricing.

The lattice is a data structure. The agent (or hand-coded pricer) defines:
- What state each node holds (rate, price, (price, avg), ...)
- How states evolve (dynamics)
- Discount factors at each node
- Payoff and exercise logic

This separates the tree structure from the financial model.
"""

from __future__ import annotations

import inspect
import warnings
import numpy as raw_np

from trellis.models._numba import NUMBA_AVAILABLE, maybe_njit
from trellis.models.trees.control import LatticeExercisePolicy, merge_lattice_exercise_policy
from trellis.models.trees.models import (
    binomial_mean_reversion_probabilities_from_metric,
    trinomial_mean_reversion_probabilities_from_metric,
)


class RecombiningLattice:
    """Generic N-dimensional recombining lattice.

    Parameters
    ----------
    n_steps : int
        Number of time steps.
    dt : float
        Time step size.
    branching : int
        Branching factor (2 = binomial, 3 = trinomial).
    state_dim : int
        Dimension of the state vector at each node (1 = scalar).
    """

    def __init__(self, n_steps: int, dt: float, branching: int = 2,
                 state_dim: int = 1):
        """Allocate state, probability, and discount storage for the full tree."""
        self.n_steps = n_steps
        self.dt = dt
        self.branching = branching
        self.state_dim = state_dim

        # For a recombining binomial: step i has i+1 nodes
        # For trinomial: step i has 2*i+1 nodes
        if branching == 2:
            max_nodes = n_steps + 1
        else:
            max_nodes = 2 * n_steps + 1

        # State storage: (n_steps+1, max_nodes, state_dim)
        self._states = raw_np.full((n_steps + 1, max_nodes, state_dim), raw_np.nan)

        # Transition probabilities: (n_steps, max_nodes, branching)
        self._probs = raw_np.zeros((n_steps, max_nodes, branching))

        # Discount factors at each node: (n_steps+1, max_nodes)
        self._discounts = raw_np.ones((n_steps + 1, max_nodes))

    def n_nodes(self, step: int) -> int:
        """Number of nodes at a given step."""
        if self.branching == 2:
            return step + 1
        else:
            return 2 * step + 1

    def set_state(self, step: int, node: int, state):
        """Set the state at (step, node). State can be scalar or tuple."""
        if self.state_dim == 1:
            self._states[step, node, 0] = float(state)
        else:
            for d in range(self.state_dim):
                self._states[step, node, d] = float(state[d])

    def get_state(self, step: int, node: int):
        """Get the state at (step, node). Returns float if 1D, tuple if multi-D."""
        if self.state_dim == 1:
            return float(self._states[step, node, 0])
        return tuple(float(self._states[step, node, d]) for d in range(self.state_dim))

    def set_probabilities(self, step: int, node: int, probs: list[float]):
        """Set transition probabilities from (step, node) to children."""
        for b, p in enumerate(probs):
            self._probs[step, node, b] = p

    def get_probabilities(self, step: int, node: int) -> list[float]:
        """Return the outgoing transition probabilities from ``(step, node)``."""
        return [float(self._probs[step, node, b]) for b in range(self.branching)]

    def set_discount(self, step: int, node: int, df: float):
        """Set the one-step discount factor at (step, node)."""
        self._discounts[step, node] = df

    def get_discount(self, step: int, node: int) -> float:
        """Return the one-step discount factor stored at ``(step, node)``."""
        return float(self._discounts[step, node])

    def child_indices(self, step: int, node: int) -> list[int]:
        """Return indices of child nodes at step+1."""
        if self.branching == 2:
            return [node, node + 1]  # down, up
        else:
            # Trinomial: down, mid, up relative to center
            return [node, node + 1, node + 2]


@maybe_njit(cache=False)
def _binomial_lattice_continuation_numba(
    values: raw_np.ndarray,
    discounts: raw_np.ndarray,
    probs: raw_np.ndarray,
) -> raw_np.ndarray:
    """Return one discounted binomial lattice rollback step."""
    out = raw_np.empty(len(discounts), dtype=values.dtype)
    for j in range(len(discounts)):
        out[j] = discounts[j] * (probs[j, 0] * values[j] + probs[j, 1] * values[j + 1])
    return out


@maybe_njit(cache=False)
def _trinomial_lattice_continuation_numba(
    values: raw_np.ndarray,
    discounts: raw_np.ndarray,
    probs: raw_np.ndarray,
) -> raw_np.ndarray:
    """Return one discounted trinomial lattice rollback step."""
    out = raw_np.empty(len(discounts), dtype=values.dtype)
    for j in range(len(discounts)):
        out[j] = discounts[j] * (
            probs[j, 0] * values[j]
            + probs[j, 1] * values[j + 1]
            + probs[j, 2] * values[j + 2]
        )
    return out


@maybe_njit(cache=False)
def _propagate_binomial_arrow_debreu_numba(
    q_current: raw_np.ndarray,
    discounts: raw_np.ndarray,
    probs: raw_np.ndarray,
) -> raw_np.ndarray:
    """Propagate Arrow-Debreu prices forward through one binomial step."""
    out = raw_np.zeros(len(q_current) + 1, dtype=q_current.dtype)
    for j in range(len(q_current)):
        weighted = q_current[j] * discounts[j]
        out[j] += weighted * probs[j, 0]
        out[j + 1] += weighted * probs[j, 1]
    return out


@maybe_njit(cache=False)
def _propagate_trinomial_arrow_debreu_numba(
    q_current: raw_np.ndarray,
    discounts: raw_np.ndarray,
    probs: raw_np.ndarray,
) -> raw_np.ndarray:
    """Propagate Arrow-Debreu prices forward through one trinomial step."""
    out = raw_np.zeros(len(q_current) + 2, dtype=q_current.dtype)
    for j in range(len(q_current)):
        weighted = q_current[j] * discounts[j]
        out[j] += weighted * probs[j, 0]
        out[j + 1] += weighted * probs[j, 1]
        out[j + 2] += weighted * probs[j, 2]
    return out


def _node_values(count: int, generator) -> raw_np.ndarray:
    """Collect node values from a Python callback with minimal overhead."""
    return raw_np.fromiter(generator, dtype=float, count=count)


def _positional_callback_params(callback) -> tuple[tuple[str, ...], bool]:
    """Return positional parameter names and whether the callable accepts ``*args``."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return (), True

    names: list[str] = []
    has_varargs = False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
            continue
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            names.append(parameter.name)
    return tuple(names), has_varargs


def _terminal_payoff_adapter(terminal_payoff):
    """Adapt common legacy terminal-payoff callback shapes to the lattice contract."""
    param_names, has_varargs = _positional_callback_params(terminal_payoff)
    arity = len(param_names)

    if has_varargs or arity >= 3:
        return lambda step, node, lattice: terminal_payoff(step, node, lattice)
    if arity == 2:
        first = param_names[0].lower()
        if "step" in first:
            return lambda step, node, lattice: terminal_payoff(step, node)
        return lambda step, node, lattice: terminal_payoff(node, lattice)
    if arity == 1:
        first = param_names[0].lower()
        if "node" in first:
            return lambda step, node, lattice: terminal_payoff(node)
        return lambda step, node, lattice: terminal_payoff(step)
    return lambda step, node, lattice: terminal_payoff()


def _cashflow_adapter(cashflow_at_node):
    """Adapt common legacy cashflow callback shapes to the lattice contract."""
    param_names, has_varargs = _positional_callback_params(cashflow_at_node)
    arity = len(param_names)

    if has_varargs or arity >= 3:
        return lambda step, node, lattice: cashflow_at_node(step, node, lattice)
    if arity == 2:
        return lambda step, node, lattice: cashflow_at_node(step, node)
    if arity == 1:
        return lambda step, node, lattice: cashflow_at_node(step)
    return lambda step, node, lattice: cashflow_at_node()


def _exercise_value_adapter(exercise_value):
    """Adapt legacy exercise-value callback shapes to the lattice contract."""
    param_names, has_varargs = _positional_callback_params(exercise_value)
    arity = len(param_names)

    if has_varargs or arity >= 4:
        third = param_names[2].lower() if arity >= 3 else ""
        if "continuation" in third:
            return lambda step, node, lattice, continuation: exercise_value(
                step,
                node,
                continuation,
                lattice,
            )
        return lambda step, node, lattice, continuation: exercise_value(
            step,
            node,
            lattice,
            continuation,
        )
    if arity == 3:
        third = param_names[2].lower()
        if "continuation" in third:
            return lambda step, node, lattice, continuation: exercise_value(
                step,
                node,
                continuation,
            )
        return lambda step, node, lattice, continuation: exercise_value(step, node, lattice)
    if arity == 2:
        return lambda step, node, lattice, continuation: exercise_value(step, node)
    if arity == 1:
        return lambda step, node, lattice, continuation: exercise_value(step)
    return lambda step, node, lattice, continuation: exercise_value()


def _apply_exercise_rule(
    continuation: raw_np.ndarray,
    exercise: raw_np.ndarray,
    exercise_fn,
) -> raw_np.ndarray:
    """Combine continuation and exercise values using the supplied rule."""
    if exercise_fn is max:
        return raw_np.maximum(continuation, exercise)
    if exercise_fn is min:
        return raw_np.minimum(continuation, exercise)
    return raw_np.fromiter(
        (exercise_fn(float(c), float(e)) for c, e in zip(continuation, exercise)),
        dtype=float,
        count=len(continuation),
    )


def _rollback_continuation(
    values: raw_np.ndarray,
    discounts: raw_np.ndarray,
    probs: raw_np.ndarray,
    branching: int,
) -> raw_np.ndarray:
    """Return one discounted rollback step for the chosen branching factor."""
    if branching == 2:
        if NUMBA_AVAILABLE:
            return _binomial_lattice_continuation_numba(values, discounts, probs)
        return discounts * (probs[:, 0] * values[:-1] + probs[:, 1] * values[1:])

    if NUMBA_AVAILABLE:
        return _trinomial_lattice_continuation_numba(values, discounts, probs)
    return discounts * (
        probs[:, 0] * values[:-2] + probs[:, 1] * values[1:-1] + probs[:, 2] * values[2:]
    )


def _propagate_arrow_debreu(
    q_current: raw_np.ndarray,
    discounts: raw_np.ndarray,
    probs: raw_np.ndarray,
    branching: int,
) -> raw_np.ndarray:
    """Propagate Arrow-Debreu state prices forward one time step."""
    if branching == 2:
        if NUMBA_AVAILABLE:
            return _propagate_binomial_arrow_debreu_numba(q_current, discounts, probs)
        weighted = q_current * discounts
        out = raw_np.zeros(len(q_current) + 1, dtype=weighted.dtype)
        out[:-1] += weighted * probs[:, 0]
        out[1:] += weighted * probs[:, 1]
        return out

    if NUMBA_AVAILABLE:
        return _propagate_trinomial_arrow_debreu_numba(q_current, discounts, probs)
    weighted = q_current * discounts
    out = raw_np.zeros(len(q_current) + 2, dtype=weighted.dtype)
    out[:-2] += weighted * probs[:, 0]
    out[1:-1] += weighted * probs[:, 1]
    out[2:] += weighted * probs[:, 2]
    return out


def _step_displacements(step: int, n_nodes: int, displacement_fn, dr: float = 0.0) -> raw_np.ndarray:
    """Evaluate the displacement function across one full lattice step."""
    return _node_values(
        n_nodes,
        (displacement_fn(step, j, dr) for j in range(n_nodes)),
    )


def _check_standard_discount(discount_fn) -> bool:
    """Check whether the supplied discount function matches exp(-r * dt)."""
    try:
        return (
            abs(discount_fn(0.0, 1.0) - 1.0) < 1e-12
            and abs(discount_fn(0.1, 0.25) - raw_np.exp(-0.025)) < 1e-12
        )
    except Exception:
        return False


def _check_lognormal_model(rate_fn) -> bool:
    """Check whether the supplied rate function behaves like exp(phi + x)."""
    try:
        return (
            abs(rate_fn(0.0, 0.0) - 1.0) < 1e-12
            and abs(rate_fn(raw_np.log(2.0), 0.0) - 2.0) < 1e-12
            and abs(rate_fn(0.0, raw_np.log(3.0)) - 3.0) < 1e-12
        )
    except Exception:
        return False


def _rate_array(phi: float, displacements: raw_np.ndarray, rate_fn, rate_mode: str) -> raw_np.ndarray:
    """Evaluate node rates for one time step."""
    if rate_mode == "normal":
        return phi + displacements
    if rate_mode == "lognormal":
        return raw_np.exp(phi + displacements)
    return _node_values(
        len(displacements),
        (rate_fn(phi, float(x)) for x in displacements),
    )


def _discount_array(
    rates: raw_np.ndarray,
    dt: float,
    discount_fn,
    standard_discount: bool,
) -> raw_np.ndarray:
    """Evaluate one-step node discount factors for one time step."""
    if standard_discount:
        return raw_np.exp(-rates * dt)
    return _node_values(
        len(rates),
        (discount_fn(float(r), dt) for r in rates),
    )


def lattice_backward_induction(
    lattice: RecombiningLattice,
    terminal_payoff=None,
    exercise_value=None,
    exercise_type: str = "european",
    exercise_steps: list[int] | None = None,
    cashflow_at_node=None,
    exercise_fn=None,
    exercise_policy: LatticeExercisePolicy | None = None,
    terminal_value: float | None = None,
    exercise_value_fn=None,
) -> float:
    """Generic backward induction on a RecombiningLattice.

    Parameters
    ----------
    lattice : RecombiningLattice
    terminal_payoff : callable(step, node, lattice) -> float, optional
        Payoff at terminal nodes. Required unless ``terminal_value`` is supplied.
    exercise_value : callable(step, node, lattice) -> float, optional
        Exercise value at a node. Used for American/Bermudan.
    exercise_type : str
        "european", "american", or "bermudan".
    exercise_steps : list[int] or None
        Steps where exercise is allowed (Bermudan only).
    cashflow_at_node : callable(step, node, lattice) -> float, optional
        Intermediate cashflows (e.g., coupons) received at each node.
        Added to the continuation value during rollback.
    exercise_fn : callable or None
        How to combine continuation and exercise values.
        Default: ``max`` (holder exercises to maximize value — puts, American options).
        Use ``min`` for issuer-callable instruments (issuer calls to minimize liability).
    exercise_policy : LatticeExercisePolicy or None
        Checked-in normalized exercise contract. When supplied, this becomes the
        authoritative source for lattice exercise semantics and step timing.
    terminal_value : float or None
        Compatibility alias for a constant terminal payoff.
    exercise_value_fn : callable or None
        Compatibility alias for ``exercise_value``.

    Returns
    -------
    float
        Price at root node (step=0, node=0).
    """
    if terminal_payoff is not None and not callable(terminal_payoff):
        if terminal_value is not None:
            raise TypeError("lattice_backward_induction() received multiple terminal payoff specifications")
        terminal_value = float(terminal_payoff)
        terminal_payoff = None

    if terminal_payoff is None:
        if terminal_value is None:
            raise TypeError(
                "lattice_backward_induction() missing 1 required positional argument: 'terminal_payoff'"
            )
        terminal_payoff = lambda step, node, lat: float(terminal_value)
    elif terminal_value is not None:
        raise TypeError("lattice_backward_induction() received both terminal_payoff and terminal_value")

    if exercise_value is None and exercise_value_fn is not None:
        exercise_value = exercise_value_fn

    exercise_type, effective_exercise_steps, exercise_fn = merge_lattice_exercise_policy(
        exercise_policy=exercise_policy,
        exercise_type=exercise_type,
        exercise_steps=exercise_steps,
        exercise_fn=exercise_fn,
    )
    if exercise_fn is None:
        exercise_fn = max
    n = lattice.n_steps
    branching = lattice.branching
    exercise_set = {int(step) for step in effective_exercise_steps}
    terminal_payoff = _terminal_payoff_adapter(terminal_payoff)
    if cashflow_at_node is not None:
        cashflow_at_node = _cashflow_adapter(cashflow_at_node)
    if exercise_value is not None:
        exercise_value = _exercise_value_adapter(exercise_value)

    # Terminal values
    n_terminal = lattice.n_nodes(n)
    values = _node_values(
        n_terminal,
        (terminal_payoff(n, j, lattice) for j in range(n_terminal)),
    )

    # Roll back
    for i in range(n - 1, -1, -1):
        n_nodes_i = lattice.n_nodes(i)
        discounts = lattice._discounts[i, :n_nodes_i]
        probs = lattice._probs[i, :n_nodes_i, :branching]
        new_values = _rollback_continuation(values, discounts, probs, branching)

        if cashflow_at_node is not None:
            new_values = new_values + _node_values(
                n_nodes_i,
                (cashflow_at_node(i, j, lattice) for j in range(n_nodes_i)),
            )

        # Exercise decisions
        if exercise_value is not None and (
            exercise_type == "american"
            or (exercise_type == "bermudan" and i in exercise_set)
        ):
            exercise = _node_values(
                n_nodes_i,
                (exercise_value(i, j, lattice, float(new_values[j])) for j in range(n_nodes_i)),
            )
            new_values = _apply_exercise_rule(new_values, exercise, exercise_fn)

        values = new_values

    return float(values[0])


# ---------------------------------------------------------------------------
# Universal analytical lattice calibration (Brigo-Mercurio framework)
# ---------------------------------------------------------------------------

def calibrate_lattice(
    lattice: RecombiningLattice,
    discount_curve,
    displacement_fn,
    rate_fn=None,
    discount_fn=None,
) -> list[float]:
    r"""Analytically calibrate a recombining lattice to a discount curve.

    This implements the universal analytical calibration from Brigo & Mercurio
    (2006), Chapter 15.  It is the same method used by FinancePy and (in
    iterative form) QuantLib.

    **Mathematical framework**

    The short rate at node (m, j) decomposes as:

    .. math::
        r(m, j) = \phi(m) + x(j)

    where :math:`x(j)` is a *displacement* that depends only on the node index
    (provided by ``displacement_fn``), and :math:`\phi(m)` is a time-dependent
    drift chosen so that the tree exactly reprices zero-coupon bonds from the
    input ``discount_curve``.

    At each step *m* the drift is solved **analytically** (no Newton iteration):

    .. math::
        \phi(m) = \frac{1}{\Delta t} \ln\!\Bigl(
            \frac{\displaystyle\sum_j Q(m,j)\,e^{-x(j)\,\Delta t}}
                 {P^{\text{mkt}}(0,\,(m{+}1)\Delta t)}
        \Bigr)

    where :math:`Q(m,j)` are Arrow-Debreu state prices propagated forward:

    .. math::
        Q(m{+}1, k) = \sum_{j \to k} Q(m,j)\;p(j \to k)\;
                       e^{-r(m,j)\,\Delta t}

    Parameters
    ----------
    lattice : RecombiningLattice
        Lattice with probabilities already set.
    discount_curve
        Must support ``discount_curve.discount(t) -> float``.
    displacement_fn : callable(step: int, node: int, dr: float) -> float
        Returns *x(j)* — the displacement at node *j* of step *m*.
        For Hull-White binomial: ``lambda m, j, dr: (2*j - m) * dr``.
    rate_fn : callable(phi, displacement) -> float, optional
        Combines calibrated drift and displacement into the short rate.
        Default: ``lambda phi, x: phi + x`` (normal/additive model).
        For lognormal: ``lambda phi, x: exp(phi + x)``.
    discount_fn : callable(rate, dt) -> float, optional
        One-step discount factor. Default: ``lambda r, dt: exp(-r * dt)``.

    Returns
    -------
    phis : list[float]
        The calibrated drift at each step, length ``n_steps + 1``.
    """
    # Default rate_fn/discount_fn: normal (additive) model
    if rate_fn is None:
        rate_fn = lambda phi, x: phi + x
    if discount_fn is None:
        discount_fn = lambda r_val, dt_val: raw_np.exp(-r_val * dt_val)

    # Detect if we can use the analytical shortcut (normal model only).
    # For normal models: r = phi + x, so exp(-r*dt) = exp(-phi*dt)*exp(-x*dt)
    # and phi factors out of the sum, giving a closed-form solution.
    # For lognormal: r = exp(phi + x), no closed form — use Newton.
    _is_normal = _check_normal_model(rate_fn)
    _is_lognormal = _check_lognormal_model(rate_fn)
    _standard_discount = _check_standard_discount(discount_fn)
    rate_mode = "normal" if _is_normal else "lognormal" if _is_lognormal else "generic"

    n = lattice.n_steps
    dt = lattice.dt
    dr = 0.0  # displacement_fn captures the actual dr via closure
    branching = lattice.branching
    states = lattice._states[:, :, 0]
    discounts = lattice._discounts
    probs = lattice._probs

    phis = raw_np.zeros(n + 1, dtype=float)
    Q_current = raw_np.array([1.0], dtype=float)  # Arrow-Debreu state prices at current step

    # --- Calibrate phi at each step ---
    for m in range(n + 1):
        n_nodes_curr = lattice.n_nodes(m)
        market_df = float(discount_curve.discount((m + 1) * dt))
        displacements = _step_displacements(m, n_nodes_curr, displacement_fn, dr)

        if _is_normal:
            # Analytical: phi = (1/dt) * ln(sum_j Q_j * exp(-x_j*dt) / DF)
            sum_qz = float(raw_np.dot(Q_current, raw_np.exp(-displacements * dt)))
            if sum_qz > 0 and market_df > 0:
                phis[m] = raw_np.log(sum_qz / market_df) / dt
            else:
                phis[m] = phis[m - 1] if m > 0 else 0.0
        else:
            # Newton iteration for non-normal models (e.g., lognormal)
            # Solve: sum_j Q_j * discount_fn(rate_fn(phi, x_j), dt) = market_df
            phi = phis[m - 1] if m > 0 else raw_np.log(max(float(
                discount_curve.zero_rate(max(dt, 0.001))), 1e-10))
            for _ in range(30):
                rates = _rate_array(phi, displacements, rate_fn, rate_mode)
                step_discounts = _discount_array(rates, dt, discount_fn, _standard_discount)
                f_val = float(raw_np.dot(Q_current, step_discounts))
                eps = 1e-8
                rates_up = _rate_array(phi + eps, displacements, rate_fn, rate_mode)
                step_discounts_up = _discount_array(
                    rates_up, dt, discount_fn, _standard_discount,
                )
                f_deriv = float(raw_np.dot(Q_current, (step_discounts_up - step_discounts) / eps))
                err = f_val - market_df
                if abs(err) < 1e-12:
                    break
                if abs(f_deriv) < 1e-15:
                    break
                phi -= err / f_deriv
            phis[m] = phi

        # Set states and discounts at this step
        r_nodes = _rate_array(float(phis[m]), displacements, rate_fn, rate_mode)
        step_discounts = _discount_array(r_nodes, dt, discount_fn, _standard_discount)
        states[m, :n_nodes_curr] = r_nodes
        discounts[m, :n_nodes_curr] = step_discounts

        # Propagate Arrow-Debreu prices to next step
        if m < n:
            Q_current = _propagate_arrow_debreu(
                Q_current,
                step_discounts,
                probs[m, :n_nodes_curr, :branching],
                branching,
            )

    return [float(phi) for phi in phis]


def _check_normal_model(rate_fn) -> bool:
    """Check if rate_fn is additive (phi + x) by testing linearity."""
    try:
        # If rate_fn(1, 2) == 3 and rate_fn(0, 1) == 1, it's additive
        return (abs(rate_fn(1.0, 2.0) - 3.0) < 1e-10 and
                abs(rate_fn(0.0, 1.0) - 1.0) < 1e-10)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Convenience builders — the agent calls these to specialize the generic lattice
# ---------------------------------------------------------------------------

def build_rate_lattice(
    r0: float,
    sigma: float,
    a: float,
    T: float,
    n_steps: int,
    discount_curve=None,
    branching: int = 2,
) -> RecombiningLattice:
    r"""Build a calibrated mean-reverting short-rate lattice (Hull-White).

    Uses the Brigo-Mercurio analytical calibration framework via
    :func:`calibrate_lattice`.  The short rate decomposes as:

    .. math::
        r(m, j) = \phi(m) + (2j - m)\,\sigma\sqrt{\Delta t}

    where :math:`\phi(m)` is solved analytically at each step to reprice
    zero-coupon bonds from the input ``discount_curve``.

    Parameters
    ----------
    r0 : float
        Initial short rate (typically ``curve.zero_rate(dt)``).
    sigma : float
        Hull-White rate volatility (absolute, NOT Black vol).
    a : float
        Mean reversion speed.
    T : float
        Time horizon.
    n_steps : int
        Number of steps.
    discount_curve : DiscountCurve or None
        If provided, calibrate to this curve (analytical, exact to machine
        precision).  If None, use constant r0 at all nodes (for testing only).
    branching : int
        2 (binomial) or 3 (trinomial).
    """
    warnings.warn(
        "build_rate_lattice() is deprecated; use trellis.models.trees.build_lattice(...) with a lattice model spec",
        DeprecationWarning,
        stacklevel=2,
    )
    dt = T / n_steps
    lattice = RecombiningLattice(n_steps, dt, branching, state_dim=1)
    dr = sigma * raw_np.sqrt(dt)
    probs = lattice._probs
    states = lattice._states[:, :, 0]
    discounts = lattice._discounts

    if branching not in (2, 3):
        raise ValueError(f"branching must be 2 or 3, got {branching}")

    if branching == 2:
        # --- Binomial (existing logic) ---
        # Step 1: Set mean-reversion-adjusted probabilities
        # In the HW binomial tree the displacement is symmetric:
        #   x(m, j) = (2*j - m) * dr
        # Probabilities encode mean reversion: p_up = 0.5 + a*(alpha - r)*dt/(2*dr)
        # But we need rates to compute drift, so we do two passes:
        #   Pass 1: equal probs → calibrate phi → get rates
        #   Pass 2: recompute probs with mean-reversion drift

        # --- Pass 1: equal probabilities, analytical calibration ---
        for i in range(n_steps):
            n_nodes_i = i + 1
            probs[i, :n_nodes_i, 0] = 0.5
            probs[i, :n_nodes_i, 1] = 0.5

        if discount_curve is not None:
            # Hull-White normal displacement: x(m, j) = (2*j - m) * dr
            def hw_displacement(m, j, _dr):
                """Return the additive Hull-White displacement for a binomial node."""
                return (2 * j - m) * dr

            phis = calibrate_lattice(lattice, discount_curve, hw_displacement)

            # --- Pass 2: mean-reversion-adjusted probabilities ---
            for i in range(n_steps):
                n_nodes_i = i + 1
                r_nodes = states[i, :n_nodes_i]
                target = phis[min(i + 1, n_steps)]
                drift = a * (target - r_nodes) * dt
                p_up = 0.5 + drift / (2 * dr) if dr > 0 else raw_np.full(n_nodes_i, 0.5)
                p_up = raw_np.clip(p_up, 0.01, 0.99)
                probs[i, :n_nodes_i, 0] = 1.0 - p_up
                probs[i, :n_nodes_i, 1] = p_up

            # --- Pass 3: re-calibrate with final probabilities ---
            # The probability adjustment changes the AD prices, so we re-calibrate
            # phi to maintain exact curve fitting.
            phis = calibrate_lattice(lattice, discount_curve, hw_displacement)
        else:
            # Uncalibrated: use r0 as phi at all steps, but keep displacement structure
            for i in range(n_steps + 1):
                n_nodes_i = i + 1
                offsets = 2.0 * raw_np.arange(n_nodes_i) - i
                r_nodes = r0 + offsets * dr
                states[i, :n_nodes_i] = r_nodes
                discounts[i, :n_nodes_i] = raw_np.exp(-r_nodes * dt)

    else:
        # --- Trinomial (branching == 3) ---
        # Hull-White trinomial: step m has 2*m+1 nodes, j=0..2*m
        # Displacement: x(m, j) = (j - m) * dr  (center node at j=m)
        # Standard HW trinomial probabilities: [1/6, 2/3, 1/6] (equal probs)
        # Mean-reversion adjusted:
        #   p_up   = 1/6 + (a*(target-r)*dt) / (2*dr)
        #   p_mid  = 2/3
        #   p_down = 1/6 - (a*(target-r)*dt) / (2*dr)

        # --- Pass 1: equal trinomial probabilities, analytical calibration ---
        for i in range(n_steps):
            n_nodes_i = 2 * i + 1
            probs[i, :n_nodes_i, 0] = 1.0 / 6
            probs[i, :n_nodes_i, 1] = 2.0 / 3
            probs[i, :n_nodes_i, 2] = 1.0 / 6

        def hw_tri_displacement(m, j, _dr):
            """Return the additive Hull-White displacement for a trinomial node."""
            return (j - m) * dr

        if discount_curve is not None:
            phis = calibrate_lattice(lattice, discount_curve, hw_tri_displacement)

            # --- Pass 2: mean-reversion-adjusted probabilities ---
            for i in range(n_steps):
                n_nodes_i = 2 * i + 1
                r_nodes = states[i, :n_nodes_i]
                target = phis[min(i + 1, n_steps)]
                drift = a * (target - r_nodes) * dt
                half_drift = drift / (2 * dr) if dr > 0 else raw_np.zeros(n_nodes_i)
                p_up = raw_np.clip(1.0 / 6 + half_drift, 0.01, 0.98)
                p_down = raw_np.clip(1.0 / 6 - half_drift, 0.01, 0.98)
                p_mid = raw_np.maximum(0.01, 1.0 - p_up - p_down)
                total = p_up + p_mid + p_down
                probs[i, :n_nodes_i, 0] = p_down / total
                probs[i, :n_nodes_i, 1] = p_mid / total
                probs[i, :n_nodes_i, 2] = p_up / total

            # --- Pass 3: re-calibrate with final probabilities ---
            phis = calibrate_lattice(lattice, discount_curve, hw_tri_displacement)
        else:
            # Uncalibrated: use r0 as phi at all steps
            for i in range(n_steps + 1):
                n_nodes_i = 2 * i + 1
                offsets = raw_np.arange(n_nodes_i) - i
                r_nodes = r0 + offsets * dr
                states[i, :n_nodes_i] = r_nodes
                discounts[i, :n_nodes_i] = raw_np.exp(-r_nodes * dt)

    return lattice


def _build_spot_lattice_impl(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    *,
    model: str = "crr",
) -> RecombiningLattice:
    """Build a recombining spot-price lattice for low-dimensional equity trees.

    Parameters
    ----------
    S0, r, sigma, T, n_steps
        Standard spot-lattice inputs.
    model : str, default ``"crr"``
        Supported one-factor spot parameterizations:

        - ``"crr"`` / ``"cox_ross_rubinstein"``
        - ``"jarrow_rudd"`` / ``"jr"``
    """
    dt = T / n_steps
    model_key = str(model).strip().lower()
    if model_key in {"crr", "cox_ross_rubinstein"}:
        u = raw_np.exp(sigma * raw_np.sqrt(dt))
        d = 1.0 / u
        p = (raw_np.exp(r * dt) - d) / (u - d)
    elif model_key in {"jarrow_rudd", "jr"}:
        u = raw_np.exp((r - 0.5 * sigma ** 2) * dt + sigma * raw_np.sqrt(dt))
        d = raw_np.exp((r - 0.5 * sigma ** 2) * dt - sigma * raw_np.sqrt(dt))
        p = 0.5
    else:
        raise ValueError(
            f"Unsupported spot lattice model {model!r}. Supported models: "
            "'crr', 'jarrow_rudd'."
        )

    lattice = RecombiningLattice(n_steps, dt, branching=2, state_dim=1)
    states = lattice._states[:, :, 0]
    discounts = lattice._discounts
    probs = lattice._probs

    for i in range(n_steps + 1):
        n_nodes_i = i + 1
        j = raw_np.arange(n_nodes_i)
        s_nodes = S0 * (u ** j) * (d ** (i - j))
        states[i, :n_nodes_i] = s_nodes
        discounts[i, :n_nodes_i] = raw_np.exp(-r * dt)

    for i in range(n_steps):
        n_nodes_i = i + 1
        probs[i, :n_nodes_i, 0] = 1.0 - p
        probs[i, :n_nodes_i, 1] = p

    return lattice


def build_spot_lattice(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    *,
    model: str = "crr",
) -> RecombiningLattice:
    """Backward-compatible wrapper for the legacy spot-lattice helper."""
    warnings.warn(
        "build_spot_lattice() is deprecated; use trellis.models.trees.build_lattice(...) with an equity lattice model spec",
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_spot_lattice_impl(
        S0,
        r,
        sigma,
        T,
        n_steps,
        model=model,
    )


def build_generic_lattice(
    model,
    r0: float,
    sigma: float,
    a: float,
    T: float,
    n_steps: int,
    discount_curve=None,
    branching: int = 2,
) -> RecombiningLattice:
    """Build a calibrated rate lattice from a TreeModel specification.

    This is the universal entry point. The ``model`` object (from
    ``trellis.models.trees.models``) defines the displacement, probability,
    rate, and discount functions. The calibration adapts automatically:
    analytical for normal models, Newton for lognormal.

    Parameters
    ----------
    model : TreeModel
        Model specification (e.g., HULL_WHITE, BLACK_DERMAN_TOY).
    r0 : float
        Initial short rate.
    sigma : float
        Volatility (interpretation depends on model.vol_type).
    a : float
        Mean reversion speed.
    T : float
        Time horizon.
    n_steps : int
        Number of time steps.
    discount_curve
        Discount curve for calibration. Required.
    branching : int
        2 (binomial) or 3 (trinomial).
    """
    if discount_curve is None:
        raise ValueError("discount_curve is required for calibrated lattice building")
    if branching not in (2, 3):
        raise ValueError(f"branching must be 2 or 3, got {branching}")
    supported_branchings = getattr(model, "supported_branchings", (2, 3))
    if branching not in supported_branchings:
        raise ValueError(
            f"Model {getattr(model, 'name', model)!r} does not support branching={branching}. "
            f"Supported: {supported_branchings}"
        )

    dt = T / n_steps
    lattice = RecombiningLattice(n_steps, dt, branching, state_dim=1)
    dr = sigma * raw_np.sqrt(dt)
    probs = lattice._probs

    # Wrap displacement to capture dr in closure.
    # For binomial: model's displacement_fn uses (2*j - m)*dr directly.
    # For trinomial: j ranges 0..2*m, so the natural displacement is (j - m)*dr.
    # We remap so the model's fn (written for binomial indexing) works for both.
    _disp = model.displacement_fn

    if branching == 2:
        def displacement(m, j, _dr):
            """Adapt the model displacement rule to the binomial node indexing."""
            return _disp(m, j, dr)
    else:
        def displacement(m, j, _dr):
            """Use the natural centered displacement for trinomial node indexing."""
            # Trinomial displacement: (j - m) * dr
            # Model's fn expects (2*j - m)*dr, but for trinomial the center is
            # at j=m and spacing is dr (not 2*dr). Use direct formula.
            return (j - m) * dr

    # --- Pass 1: equal probabilities, calibrate ---
    if branching == 2:
        equal_probs = [0.5, 0.5]
    else:
        equal_probs = [1.0 / 6, 2.0 / 3, 1.0 / 6]

    for i in range(n_steps):
        n_nodes_i = lattice.n_nodes(i)
        probs[i, :n_nodes_i, :branching] = equal_probs

    phis = calibrate_lattice(
        lattice, discount_curve, displacement,
        rate_fn=model.rate_fn, discount_fn=model.discount_fn,
    )

    # --- Pass 2: model-specific probabilities ---
    _prob_fn = model.probability_fn
    state_metric_fn = getattr(model, "state_metric_fn", None)
    for i in range(n_steps):
        n_nodes_i = lattice.n_nodes(i)
        target_metric = float(phis[min(i + 1, n_steps)])
        for j in range(n_nodes_i):
            if state_metric_fn is not None:
                current_metric = float(state_metric_fn(lattice.get_state(i, j)))
                if branching == 2:
                    probs[i, j, :branching] = binomial_mean_reversion_probabilities_from_metric(
                        current_metric,
                        target_metric,
                        dt=dt,
                        a=a,
                        dr=dr,
                    )
                else:
                    probs[i, j, :branching] = trinomial_mean_reversion_probabilities_from_metric(
                        current_metric,
                        target_metric,
                        dt=dt,
                        a=a,
                        dr=dr,
                    )
            else:
                probs[i, j, :branching] = _prob_fn(lattice, i, j, phis, a, dr)

    # --- Pass 3: re-calibrate with final probabilities ---
    phis = calibrate_lattice(
        lattice, discount_curve, displacement,
        rate_fn=model.rate_fn, discount_fn=model.discount_fn,
    )

    return lattice


def build_lattice(topology, mesh, model, calibration_target=None, **params):
    """Build a lattice through the generalized lattice-algebra surface."""
    from trellis.models.trees.algebra import build_lattice as _build_lattice

    return _build_lattice(
        topology,
        mesh,
        model,
        calibration_target=calibration_target,
        **params,
    )


def price_on_lattice(lattice: RecombiningLattice, contract) -> float:
    """Price a generalized lattice contract on one built lattice."""
    from trellis.models.trees.algebra import price_on_lattice as _price_on_lattice

    return _price_on_lattice(lattice, contract)
