"""Generic recombining lattice for tree-based pricing.

The lattice is a data structure. The agent (or hand-coded pricer) defines:
- What state each node holds (rate, price, (price, avg), ...)
- How states evolve (dynamics)
- Discount factors at each node
- Payoff and exercise logic

This separates the tree structure from the financial model.
"""

from __future__ import annotations

import numpy as raw_np


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
        return [float(self._probs[step, node, b]) for b in range(self.branching)]

    def set_discount(self, step: int, node: int, df: float):
        """Set the one-step discount factor at (step, node)."""
        self._discounts[step, node] = df

    def get_discount(self, step: int, node: int) -> float:
        return float(self._discounts[step, node])

    def child_indices(self, step: int, node: int) -> list[int]:
        """Return indices of child nodes at step+1."""
        if self.branching == 2:
            return [node, node + 1]  # down, up
        else:
            # Trinomial: down, mid, up relative to center
            return [node, node + 1, node + 2]


def lattice_backward_induction(
    lattice: RecombiningLattice,
    terminal_payoff,
    exercise_value=None,
    exercise_type: str = "european",
    exercise_steps: list[int] | None = None,
) -> float:
    """Generic backward induction on a RecombiningLattice.

    Parameters
    ----------
    lattice : RecombiningLattice
    terminal_payoff : callable(step, node, lattice) -> float
        Payoff at terminal nodes.
    exercise_value : callable(step, node, lattice) -> float, optional
        Exercise value at a node. Used for American/Bermudan.
    exercise_type : str
        "european", "american", or "bermudan".
    exercise_steps : list[int] or None
        Steps where exercise is allowed (Bermudan only).

    Returns
    -------
    float
        Price at root node (step=0, node=0).
    """
    n = lattice.n_steps

    # Terminal values
    n_terminal = lattice.n_nodes(n)
    values = raw_np.array([terminal_payoff(n, j, lattice) for j in range(n_terminal)])

    # Roll back
    for i in range(n - 1, -1, -1):
        n_nodes_i = lattice.n_nodes(i)
        new_values = raw_np.zeros(n_nodes_i)

        for j in range(n_nodes_i):
            df = lattice.get_discount(i, j)
            probs = lattice.get_probabilities(i, j)
            children = lattice.child_indices(i, j)

            # Continuation value
            cont = df * sum(p * values[c] for p, c in zip(probs, children))
            new_values[j] = cont

        # Exercise decisions
        if exercise_type == "american" and exercise_value is not None:
            for j in range(n_nodes_i):
                ev = exercise_value(i, j, lattice)
                new_values[j] = max(new_values[j], ev)
        elif exercise_type == "bermudan" and exercise_steps and i in exercise_steps:
            if exercise_value is not None:
                for j in range(n_nodes_i):
                    ev = exercise_value(i, j, lattice)
                    new_values[j] = max(new_values[j], ev)

        values = new_values

    return float(values[0])


# ---------------------------------------------------------------------------
# Convenience builders — the agent calls these to specialize the generic lattice
# ---------------------------------------------------------------------------

def build_rate_lattice(
    r0: float,
    sigma: float,
    a: float,
    T: float,
    n_steps: int,
    branching: int = 2,
) -> RecombiningLattice:
    """Build a mean-reverting short-rate lattice (Hull-White style).

    dr = a(theta - r)dt + sigma dW

    Calibrated so that the tree-implied discount factors match the
    input curve (simplified: uses r0 as the flat rate for theta).

    Parameters
    ----------
    r0 : float
        Initial short rate.
    sigma : float
        Rate volatility.
    a : float
        Mean reversion speed.
    T : float
        Time horizon.
    n_steps : int
        Number of steps.
    branching : int
        2 (binomial) or 3 (trinomial).
    """
    dt = T / n_steps
    lattice = RecombiningLattice(n_steps, dt, branching, state_dim=1)

    if branching == 2:
        # Binomial HW: up/down with mean reversion
        dr = sigma * raw_np.sqrt(dt)
        for i in range(n_steps + 1):
            for j in range(i + 1):
                # Rate at node: r0 + (2*j - i) * dr, adjusted for mean reversion
                r_node = r0 + (2 * j - i) * dr
                # Simple mean reversion adjustment
                reversion = a * (r0 - r_node) * dt * i / max(n_steps, 1)
                r_node += reversion
                lattice.set_state(i, j, r_node)
                lattice.set_discount(i, j, raw_np.exp(-r_node * dt))

        # Probabilities: adjusted for mean reversion
        for i in range(n_steps):
            for j in range(i + 1):
                r = lattice.get_state(i, j)
                # Risk-neutral probabilities with mean-reversion adjustment
                drift = a * (r0 - r) * dt
                p_up = 0.5 + drift / (2 * dr) if dr > 0 else 0.5
                p_up = max(0.01, min(0.99, p_up))
                lattice.set_probabilities(i, j, [1 - p_up, p_up])

    elif branching == 3:
        # Trinomial HW
        dr = sigma * raw_np.sqrt(3 * dt)
        for i in range(n_steps + 1):
            for j in range(2 * i + 1):
                offset = j - i
                r_node = r0 + offset * dr
                lattice.set_state(i, j, r_node)
                lattice.set_discount(i, j, raw_np.exp(-r_node * dt))

        for i in range(n_steps):
            for j in range(2 * i + 1):
                r = lattice.get_state(i, j)
                drift = a * (r0 - r) * dt
                # Standard trinomial probabilities
                p_up = 0.5 * (sigma ** 2 * dt + drift ** 2) / (dr ** 2) + drift / (2 * dr)
                p_dn = 0.5 * (sigma ** 2 * dt + drift ** 2) / (dr ** 2) - drift / (2 * dr)
                p_mid = 1.0 - p_up - p_dn
                p_up = max(0.01, min(0.98, p_up))
                p_dn = max(0.01, min(0.98, p_dn))
                p_mid = 1.0 - p_up - p_dn
                lattice.set_probabilities(i, j, [p_dn, p_mid, p_up])

    return lattice


def build_spot_lattice(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
) -> RecombiningLattice:
    """Build a CRR spot-price lattice (for equity options).

    dS/S = r dt + sigma dW
    """
    dt = T / n_steps
    u = raw_np.exp(sigma * raw_np.sqrt(dt))
    d = 1.0 / u
    p = (raw_np.exp(r * dt) - d) / (u - d)

    lattice = RecombiningLattice(n_steps, dt, branching=2, state_dim=1)

    for i in range(n_steps + 1):
        for j in range(i + 1):
            S = S0 * u ** j * d ** (i - j)
            lattice.set_state(i, j, S)
            lattice.set_discount(i, j, raw_np.exp(-r * dt))

    for i in range(n_steps):
        for j in range(i + 1):
            lattice.set_probabilities(i, j, [1 - p, p])

    return lattice
