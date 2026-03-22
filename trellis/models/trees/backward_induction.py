"""Generic backward induction on binomial and trinomial trees."""

from __future__ import annotations

import numpy as raw_np


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
    dt = tree.dt
    df = raw_np.exp(-discount_rate * dt)

    if exercise_value_fn is None:
        exercise_value_fn = lambda step, node, t: payoff_at_node(step, node)

    if isinstance(tree, BinomialTree):
        return _backward_binomial(
            tree, n, dt, df, payoff_at_node, exercise_type,
            exercise_steps, exercise_value_fn,
        )
    elif isinstance(tree, TrinomialTree):
        return _backward_trinomial(
            tree, n, dt, df, payoff_at_node, exercise_type,
            exercise_steps, exercise_value_fn,
        )
    else:
        raise TypeError(f"Unknown tree type: {type(tree)}")


def _backward_binomial(tree, n, dt, df, payoff_fn, ex_type, ex_steps, ex_val_fn):
    """Backward induction on a binomial tree."""
    p = tree.p

    # Terminal values
    values = raw_np.array([payoff_fn(n, j) for j in range(n + 1)])

    # Roll back
    for i in range(n - 1, -1, -1):
        continuation = raw_np.array([
            df * (p * values[j + 1] + (1 - p) * values[j])
            for j in range(i + 1)
        ])

        if ex_type == "american":
            exercise = raw_np.array([ex_val_fn(i, j, tree) for j in range(i + 1)])
            values = raw_np.maximum(continuation, exercise)
        elif ex_type == "bermudan" and ex_steps and i in ex_steps:
            exercise = raw_np.array([ex_val_fn(i, j, tree) for j in range(i + 1)])
            values = raw_np.maximum(continuation, exercise)
        else:
            values = continuation

    return float(values[0])


def _backward_trinomial(tree, n, dt, df, payoff_fn, ex_type, ex_steps, ex_val_fn):
    """Backward induction on a trinomial tree."""
    pu, pm, pd = tree.pu, tree.pm, tree.pd
    mid = n

    # Terminal values
    values = raw_np.array([payoff_fn(n, j - mid) for j in range(2 * n + 1)])

    for i in range(n - 1, -1, -1):
        new_values = raw_np.zeros(2 * i + 1)
        for j_idx in range(2 * i + 1):
            j = j_idx  # index in the new array
            # Map to parent indices in the larger array
            parent_mid = i + 1  # center of the previous step's array (2*(i+1)+1 wide)
            offset = j_idx - i  # offset from center at step i
            # Children indices in previous values array (2*(i+1)+1 wide)
            up_idx = (i + 1) + offset + 1
            mid_idx = (i + 1) + offset
            dn_idx = (i + 1) + offset - 1
            cont = df * (pu * values[up_idx] + pm * values[mid_idx] + pd * values[dn_idx])
            new_values[j_idx] = cont

        if ex_type == "american":
            for j_idx in range(2 * i + 1):
                ev = ex_val_fn(i, j_idx - i, tree)
                new_values[j_idx] = max(new_values[j_idx], ev)
        elif ex_type == "bermudan" and ex_steps and i in ex_steps:
            for j_idx in range(2 * i + 1):
                ev = ex_val_fn(i, j_idx - i, tree)
                new_values[j_idx] = max(new_values[j_idx], ev)

        values = new_values

    return float(values[0])
