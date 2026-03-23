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
    discount_curve=None,
    branching: int = 2,
) -> RecombiningLattice:
    """Build a calibrated mean-reverting short-rate lattice (Hull-White).

    dr = a(theta(t) - r)dt + sigma dW

    If ``discount_curve`` is provided, theta(t) is calibrated at each step
    so that the tree reprices zero-coupon bonds from the input curve.
    This is the standard Hull-White tree construction.

    Parameters
    ----------
    r0 : float
        Initial short rate (typically curve.zero_rate(dt)).
    sigma : float
        Hull-White rate volatility (absolute, NOT Black vol).
    a : float
        Mean reversion speed.
    T : float
        Time horizon.
    n_steps : int
        Number of steps.
    discount_curve : DiscountCurve or None
        If provided, calibrate theta(t) to reprice this curve.
        If None, use constant theta = a * r0 (uncalibrated — for testing only).
    branching : int
        2 (binomial) or 3 (trinomial).
    """
    dt = T / n_steps
    lattice = RecombiningLattice(n_steps, dt, branching, state_dim=1)
    dr = sigma * raw_np.sqrt(dt)

    if branching != 2:
        raise NotImplementedError("Calibrated trinomial tree not yet implemented")

    # Step 1: Build the tree structure (rates without theta calibration)
    # Base rates: r_node = alpha_i + (2*j - i) * dr
    # where alpha_i is a time-dependent shift calibrated to the curve.
    alphas = raw_np.zeros(n_steps + 1)
    alphas[0] = r0

    # Step 2: Calibrate alpha_i (which encodes theta) at each step
    # by matching the tree-implied ZCB price to the market ZCB price.
    for i in range(n_steps + 1):
        if i == 0:
            # Root: just r0
            lattice.set_state(0, 0, r0)
            lattice.set_discount(0, 0, raw_np.exp(-r0 * dt))
            lattice.set_probabilities(0, 0, [0.5, 0.5])  # initial guess
            continue

        if discount_curve is not None and i > 0:
            # Calibrate alpha_i so tree ZCB(i*dt) matches curve.discount(i*dt)
            market_df = float(discount_curve.discount(i * dt))

            # Arrow-Debreu prices at step i-1
            if i == 1:
                ad_prices = [1.0]  # single root node
            else:
                # Compute AD prices by forward induction
                ad_prices = _forward_induction_ad(lattice, i - 1)

            # Search for alpha_i that makes tree ZCB = market ZCB
            # ZCB(i*dt) = sum over nodes at step i-1: AD_j * exp(-r_j * dt)
            # where r_j depends on alpha_i
            def zcb_error(alpha):
                total = 0.0
                for j in range(i):
                    probs = lattice.get_probabilities(i - 1, j)
                    df_node = lattice.get_discount(i - 1, j)
                    for b, p in enumerate(probs):
                        child = j + b  # for binomial: [j, j+1]
                        r_child = alpha + (2 * child - i) * dr
                        child_df = raw_np.exp(-r_child * dt)
                        total += ad_prices[j] * p * df_node * child_df
                return total

            # Find alpha_i via bisection
            # Start with the forward rate as initial guess
            if discount_curve is not None:
                fwd = -raw_np.log(float(discount_curve.discount((i + 0.5) * dt)) /
                                   float(discount_curve.discount((i - 0.5) * dt))) / dt
            else:
                fwd = r0

            # Simple Newton iteration
            alpha = fwd
            for _ in range(20):
                # Current tree ZCB
                tree_zcb = 0.0
                d_tree_zcb = 0.0  # derivative w.r.t. alpha
                for j in range(i):
                    probs = lattice.get_probabilities(i - 1, j)
                    df_node = lattice.get_discount(i - 1, j)
                    for b, p in enumerate(probs):
                        child = j + b
                        r_child = alpha + (2 * child - i) * dr
                        child_df = raw_np.exp(-r_child * dt)
                        contrib = ad_prices[j] * p * df_node * child_df
                        tree_zcb += contrib
                        d_tree_zcb -= contrib * dt  # d/dalpha of exp(-r*dt)

                err = tree_zcb - market_df
                if abs(err) < 1e-12:
                    break
                if abs(d_tree_zcb) < 1e-15:
                    break
                alpha -= err / d_tree_zcb

            alphas[i] = alpha
        else:
            alphas[i] = r0  # uncalibrated fallback

        # Set states and discounts at step i
        for j in range(i + 1):
            r_node = alphas[i] + (2 * j - i) * dr
            lattice.set_state(i, j, r_node)
            lattice.set_discount(i, j, raw_np.exp(-r_node * dt))

    # Step 3: Set probabilities (mean-reversion adjusted)
    for i in range(n_steps):
        for j in range(i + 1):
            r = lattice.get_state(i, j)
            target = alphas[min(i + 1, n_steps)]
            drift = a * (target - r) * dt
            p_up = 0.5 + drift / (2 * dr) if dr > 0 else 0.5
            p_up = max(0.01, min(0.99, p_up))
            lattice.set_probabilities(i, j, [1 - p_up, p_up])

    return lattice


def _forward_induction_ad(lattice, target_step):
    """Compute Arrow-Debreu prices at a given step via forward induction."""
    # AD price at root = 1
    ad = {(0, 0): 1.0}

    for i in range(target_step):
        new_ad = {}
        for j in range(i + 1):
            if (i, j) not in ad:
                continue
            price = ad[(i, j)]
            df = lattice.get_discount(i, j)
            probs = lattice.get_probabilities(i, j)
            children = lattice.child_indices(i, j)
            for b, (p, c) in enumerate(zip(probs, children)):
                key = (i + 1, c)
                new_ad[key] = new_ad.get(key, 0.0) + price * p * df
        ad.update(new_ad)

    # Extract prices at target_step
    n_nodes = lattice.n_nodes(target_step)
    result = [ad.get((target_step, j), 0.0) for j in range(n_nodes)]
    return result


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
