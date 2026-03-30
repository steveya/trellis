"""Backward induction: pricing options by working from expiry back to today.

Given a tree of possible future prices, backward induction computes the
option value at each node by discounting the expected value from the next
time step. Supports European, American, and Bermudan exercise styles.
"""

from __future__ import annotations

import numpy as raw_np

from trellis.core.differentiable import get_numpy
from trellis.models._numba import NUMBA_AVAILABLE, maybe_njit

np = get_numpy()


@maybe_njit(cache=False)
def _binomial_continuation_numba(values: raw_np.ndarray, p: float, df: float) -> raw_np.ndarray:
    """Return one discounted binomial rollback step."""
    out = raw_np.empty(len(values) - 1, dtype=values.dtype)
    q = 1.0 - p
    for j in range(len(out)):
        out[j] = df * (p * values[j + 1] + q * values[j])
    return out


@maybe_njit(cache=False)
def _trinomial_continuation_numba(
    values: raw_np.ndarray, pu: float, pm: float, pd: float, df: float,
) -> raw_np.ndarray:
    """Return one discounted trinomial rollback step."""
    out = raw_np.empty(len(values) - 2, dtype=values.dtype)
    for j in range(len(out)):
        out[j] = df * (pu * values[j + 2] + pm * values[j + 1] + pd * values[j])
    return out


def _node_values(count: int, generator) -> raw_np.ndarray:
    """Collect node values from a Python callback with minimal overhead."""
    return raw_np.fromiter(generator, dtype=float, count=count)


def _node_values_differentiable(count: int, generator):
    """Collect node values without scalarizing autograd-traceable outputs."""
    return np.array([value for value in generator])


def _binomial_continuation(values: raw_np.ndarray, p: float, df: float) -> raw_np.ndarray:
    """Return one discounted binomial rollback step."""
    if NUMBA_AVAILABLE:
        return _binomial_continuation_numba(values, p, df)
    return df * ((1.0 - p) * values[:-1] + p * values[1:])


def _trinomial_continuation(
    values: raw_np.ndarray, pu: float, pm: float, pd: float, df: float,
) -> raw_np.ndarray:
    """Return one discounted trinomial rollback step."""
    if NUMBA_AVAILABLE:
        return _trinomial_continuation_numba(values, pu, pm, pd, df)
    return df * (pd * values[:-2] + pm * values[1:-1] + pu * values[2:])


def _binomial_continuation_differentiable(values, p, df):
    """One discounted binomial rollback step (autograd-compatible, no numba)."""
    q = 1.0 - p
    return df * (q * values[:-1] + p * values[1:])


def _trinomial_continuation_differentiable(values, pu, pm, pd, df):
    """One discounted trinomial rollback step (autograd-compatible, no numba)."""
    return df * (pd * values[:-2] + pm * values[1:-1] + pu * values[2:])


def backward_induction(
    tree,
    payoff_at_node,
    discount_rate: float = 0.0,
    exercise_type: str = "european",
    exercise_steps: list[int] | None = None,
    exercise_value_fn=None,
    *,
    differentiable: bool = False,
) -> float:
    """Roll back through a tree to compute the present value.

    Parameters
    ----------
    tree : BinomialTree or TrinomialTree
        The lattice.
    payoff_at_node : callable(step, node) -> float
        Terminal (or intermediate) payoff at each node.
        For European options, only called at the final step.
    discount_rate : float
        Continuously compounded risk-free rate for discounting.
    exercise_type : str
        ``"european"``, ``"american"``, or ``"bermudan"``.
    exercise_steps : list[int] or None
        Step indices where exercise is allowed (Bermudan). Ignored for
        European/American.
    exercise_value_fn : callable(step, node, tree) -> float, optional
        Exercise value at a node. Defaults to ``payoff_at_node``.
    differentiable : bool, default ``False``
        When true, keep the rollback traceable by autograd and return the root
        value without scalarizing it.

    Returns
    -------
    float or scalar array
        Present value at node (0, 0). In differentiable mode the returned
        scalar remains in the autograd trace.
    """
    from trellis.models.trees.binomial import BinomialTree
    from trellis.models.trees.trinomial import TrinomialTree

    n = tree.n_steps
    df = np.exp(-discount_rate * tree.dt) if differentiable else raw_np.exp(-discount_rate * tree.dt)
    exercise_set = {int(step) for step in (exercise_steps or [])}

    if exercise_value_fn is None:
        def exercise_value_fn(step, node, current_tree):
            return payoff_at_node(step, node)

    if isinstance(tree, BinomialTree):
        return _backward_binomial(
            tree, n, df, payoff_at_node, exercise_type,
            exercise_set, exercise_value_fn, differentiable=differentiable,
        )
    if isinstance(tree, TrinomialTree):
        return _backward_trinomial(
            tree, n, df, payoff_at_node, exercise_type,
            exercise_set, exercise_value_fn, differentiable=differentiable,
        )
    raise TypeError(f"Unknown tree type: {type(tree)}")


def _backward_binomial(tree, n, df, payoff_fn, ex_type, ex_steps, ex_val_fn, *, differentiable: bool = False):
    """Backward induction on a binomial tree."""
    values = (
        _node_values_differentiable(n + 1, (payoff_fn(n, j) for j in range(n + 1)))
        if differentiable
        else _node_values(n + 1, (payoff_fn(n, j) for j in range(n + 1)))
    )

    for i in range(n - 1, -1, -1):
        continuation = (
            _binomial_continuation_differentiable(values, tree.p, df)
            if differentiable
            else _binomial_continuation(values, tree.p, df)
        )

        if ex_type == "american" or (ex_type == "bermudan" and i in ex_steps):
            exercise = (
                _node_values_differentiable(
                    i + 1,
                    (ex_val_fn(i, j, tree) for j in range(i + 1)),
                )
                if differentiable
                else _node_values(
                    i + 1,
                    (ex_val_fn(i, j, tree) for j in range(i + 1)),
                )
            )
            if differentiable:
                if ex_val_fn is max or ex_val_fn is raw_np.maximum or ex_val_fn is np.maximum:
                    values = np.maximum(continuation, exercise)
                elif ex_val_fn is min or ex_val_fn is raw_np.minimum or ex_val_fn is np.minimum:
                    values = np.minimum(continuation, exercise)
                else:
                    values = np.array([ex_val_fn(c, e) for c, e in zip(continuation, exercise)])
            else:
                values = raw_np.maximum(continuation, exercise)
        else:
            values = continuation

    return values[0] if differentiable else float(values[0])


def _backward_trinomial(tree, n, df, payoff_fn, ex_type, ex_steps, ex_val_fn, *, differentiable: bool = False):
    """Backward induction on a trinomial tree."""
    terminal_count = 2 * n + 1
    values = (
        _node_values_differentiable(
            terminal_count,
            (payoff_fn(n, j - n) for j in range(terminal_count)),
        )
        if differentiable
        else _node_values(
            terminal_count,
            (payoff_fn(n, j - n) for j in range(terminal_count)),
        )
    )

    for i in range(n - 1, -1, -1):
        continuation = (
            _trinomial_continuation_differentiable(values, tree.pu, tree.pm, tree.pd, df)
            if differentiable
            else _trinomial_continuation(values, tree.pu, tree.pm, tree.pd, df)
        )

        if ex_type == "american" or (ex_type == "bermudan" and i in ex_steps):
            n_nodes_i = 2 * i + 1
            exercise = (
                _node_values_differentiable(
                    n_nodes_i,
                    (ex_val_fn(i, j_idx - i, tree) for j_idx in range(n_nodes_i)),
                )
                if differentiable
                else _node_values(
                    n_nodes_i,
                    (ex_val_fn(i, j_idx - i, tree) for j_idx in range(n_nodes_i)),
                )
            )
            if differentiable:
                if ex_val_fn is max or ex_val_fn is raw_np.maximum or ex_val_fn is np.maximum:
                    values = np.maximum(continuation, exercise)
                elif ex_val_fn is min or ex_val_fn is raw_np.minimum or ex_val_fn is np.minimum:
                    values = np.minimum(continuation, exercise)
                else:
                    values = np.array([ex_val_fn(c, e) for c, e in zip(continuation, exercise)])
            else:
                values = raw_np.maximum(continuation, exercise)
        else:
            values = continuation

    return values[0] if differentiable else float(values[0])
