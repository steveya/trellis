"""Generic backward induction on binomial and trinomial trees."""

from __future__ import annotations

import numpy as raw_np

from trellis.models._numba import NUMBA_AVAILABLE, maybe_njit


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


def backward_induction(
    tree,
    payoff_at_node,
    discount_rate: float = 0.0,
    exercise_type: str = "european",
    exercise_steps: list[int] | None = None,
    exercise_value_fn=None,
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

    Returns
    -------
    float
        Present value at node (0, 0).
    """
    from trellis.models.trees.binomial import BinomialTree
    from trellis.models.trees.trinomial import TrinomialTree

    n = tree.n_steps
    df = raw_np.exp(-discount_rate * tree.dt)
    exercise_set = {int(step) for step in (exercise_steps or [])}

    if exercise_value_fn is None:
        def exercise_value_fn(step, node, current_tree):
            return payoff_at_node(step, node)

    if isinstance(tree, BinomialTree):
        return _backward_binomial(
            tree, n, df, payoff_at_node, exercise_type,
            exercise_set, exercise_value_fn,
        )
    if isinstance(tree, TrinomialTree):
        return _backward_trinomial(
            tree, n, df, payoff_at_node, exercise_type,
            exercise_set, exercise_value_fn,
        )
    raise TypeError(f"Unknown tree type: {type(tree)}")


def _backward_binomial(tree, n, df, payoff_fn, ex_type, ex_steps, ex_val_fn):
    """Backward induction on a binomial tree."""
    values = _node_values(n + 1, (payoff_fn(n, j) for j in range(n + 1)))

    for i in range(n - 1, -1, -1):
        continuation = _binomial_continuation(values, tree.p, df)

        if ex_type == "american" or (ex_type == "bermudan" and i in ex_steps):
            exercise = _node_values(
                i + 1,
                (ex_val_fn(i, j, tree) for j in range(i + 1)),
            )
            values = raw_np.maximum(continuation, exercise)
        else:
            values = continuation

    return float(values[0])


def _backward_trinomial(tree, n, df, payoff_fn, ex_type, ex_steps, ex_val_fn):
    """Backward induction on a trinomial tree."""
    terminal_count = 2 * n + 1
    values = _node_values(
        terminal_count,
        (payoff_fn(n, j - n) for j in range(terminal_count)),
    )

    for i in range(n - 1, -1, -1):
        continuation = _trinomial_continuation(values, tree.pu, tree.pm, tree.pd, df)

        if ex_type == "american" or (ex_type == "bermudan" and i in ex_steps):
            n_nodes_i = 2 * i + 1
            exercise = _node_values(
                n_nodes_i,
                (ex_val_fn(i, j_idx - i, tree) for j_idx in range(n_nodes_i)),
            )
            values = raw_np.maximum(continuation, exercise)
        else:
            values = continuation

    return float(values[0])
