Lattice Methods (Trees)
=======================

Binomial trees discretize the underlying process into up/down moves at each
time step. Combined with backward induction, they price European, American,
and Bermudan options.

For smooth payoffs, the simple spot-tree path can also be made autograd-safe:
``BinomialTree`` and ``TrinomialTree`` now build their node grids with
autograd-aware numpy, and ``backward_induction(..., differentiable=True)``
keeps the rollback traceable for gradient extraction.

Spot-Price Trees (CRR)
----------------------

Cox-Ross-Rubinstein (CRR)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   u = e^{\sigma\sqrt{\Delta t}}, \quad d = \frac{1}{u}, \quad
   p = \frac{e^{r \Delta t} - d}{u - d}

where :math:`\Delta t = T / n`. The tree is recombining: :math:`u \cdot d = 1`,
so there are :math:`n+1` terminal nodes.

Node value at step :math:`i`, node :math:`j`:

.. math::

   S_{i,j} = S_0 \cdot u^j \cdot d^{i-j}

Jarrow-Rudd (Equal Probability)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sets :math:`p = 0.5`:

.. math::

   u = e^{(r - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t}}, \quad
   d = e^{(r - \sigma^2/2)\Delta t - \sigma\sqrt{\Delta t}}

Trinomial Tree
~~~~~~~~~~~~~~

.. math::

   u = e^{\sigma\sqrt{2\Delta t}}, \quad d = 1/u

.. math::

   p_u + p_m + p_d = 1

Trinomial trees converge faster and handle barriers more naturally.


Short-Rate Trees (Hull-White)
-----------------------------

For interest rate derivatives with early exercise (callable bonds, Bermudan
swaptions), we need a *short-rate* tree calibrated to the yield curve.  The
Cox-Ross-Rubinstein construction is designed for spot prices and is
**not appropriate** for mean-reverting rate processes.

Hull-White Model
~~~~~~~~~~~~~~~~

The one-factor Hull-White model specifies:

.. math::

   dr(t) = a\bigl[\theta(t) - r(t)\bigr]\,dt + \sigma\,dW(t)

where:

- :math:`a` — mean reversion speed (typically 0.01–0.20)
- :math:`\sigma` — rate volatility (absolute, in rate units, typically 0.5%–2%)
- :math:`\theta(t)` — time-dependent drift chosen to fit the initial term structure

The function :math:`\theta(t)` ensures the model reprices all zero-coupon bonds
observed in the market, making it an *arbitrage-free* model.

.. important::

   **Volatility units matter.**  The Hull-White :math:`\sigma` is an *absolute*
   rate volatility (e.g., 0.01 = 1%).  It is **not** the same as the Black
   implied volatility typically quoted for caps/swaptions (e.g., 20%).  The
   conversion is:

   .. math::

      \sigma_{\text{HW}} = \sigma_{\text{Black}} \times F

   where :math:`F` is the relevant forward rate.  Passing Black vol directly to
   the tree produces catastrophically wrong prices.


Rate Decomposition
~~~~~~~~~~~~~~~~~~

Following Brigo & Mercurio (2006, Ch. 15) and Hull & White (1994), the short
rate at tree node :math:`(m, j)` decomposes as:

.. math::
   :label: rate-decomposition

   r(m, j) = \phi(m) + x(m, j)

where:

- :math:`\phi(m)` is a *time-dependent drift* (one value per time step)
- :math:`x(m, j)` is a *displacement* that depends only on the node index

For the standard binomial Hull-White tree:

.. math::

   x(m, j) = (2j - m) \cdot \sigma\sqrt{\Delta t}

This decomposition is the key insight that makes *analytical* calibration
possible: :math:`\phi(m)` absorbs the entire yield curve, while the
displacement :math:`x` captures the stochastic spread of rates.


Universal Analytical Calibration
--------------------------------

This is the central algorithm of the rate tree implementation. It calibrates
:math:`\phi(m)` at each step so that the tree exactly reprices zero-coupon bonds
from the input discount curve.

Arrow-Debreu State Prices
~~~~~~~~~~~~~~~~~~~~~~~~~

Define the **Arrow-Debreu state price** :math:`Q(m, j)` as the time-0 value of
a security that pays \$1 if and only if node :math:`(m, j)` is reached:

.. math::

   Q(0, 0) = 1

The state prices propagate forward via:

.. math::
   :label: ad-forward

   Q(m{+}1, k) = \sum_{j \to k} Q(m, j) \cdot p(j \to k) \cdot
                  e^{-r(m,j)\,\Delta t}

where the sum is over all parent nodes :math:`j` at step :math:`m` that
transition to child node :math:`k` at step :math:`m{+}1`, and :math:`p(j \to k)`
is the transition probability.

For a binomial tree, node :math:`j` at step :math:`m` connects to nodes
:math:`j` (down) and :math:`j{+}1` (up) at step :math:`m{+}1`.

Calibration Condition
~~~~~~~~~~~~~~~~~~~~~

The tree must reprice the market zero-coupon bond :math:`P(0, t)` at every time
step :math:`t = (m{+}1)\Delta t`.  The ZCB price is the sum of all Arrow-Debreu
prices discounted one more step:

.. math::
   :label: zcb-condition

   P\bigl(0,\,(m{+}1)\Delta t\bigr) = \sum_{j=0}^{n_m} Q(m, j) \cdot
   e^{-r(m,j)\,\Delta t}

Substituting the decomposition :eq:`rate-decomposition`:

.. math::

   P\bigl(0,\,(m{+}1)\Delta t\bigr) = e^{-\phi(m)\,\Delta t}
   \sum_{j=0}^{n_m} Q(m, j) \cdot e^{-x(m,j)\,\Delta t}

Solving for :math:`\phi(m)`:

.. math::
   :label: phi-formula

   \boxed{
   \phi(m) = \frac{1}{\Delta t} \ln\!\left(
       \frac{\displaystyle\sum_{j=0}^{n_m} Q(m,j)\,e^{-x(m,j)\,\Delta t}}
            {P\bigl(0,\,(m{+}1)\Delta t\bigr)}
   \right)
   }

This is the **analytical calibration formula**.  It is evaluated in closed form
at each step — no Newton iteration, no root-finding, no convergence tolerance.

Algorithm
~~~~~~~~~

.. code-block:: text

   Input: discount_curve P(0,t), displacement function x(m,j), probabilities p
   Output: calibrated φ(m) at each step, fully populated lattice

   1. Initialize Q(0,0) = 1
   2. For m = 0, 1, ..., n_steps:
      a. Compute φ(m) via Eq. (φ-formula)
      b. Set r(m,j) = φ(m) + x(m,j) for all nodes j at step m
      c. Set discount factors: df(m,j) = exp(-r(m,j) · Δt)
      d. Propagate Arrow-Debreu prices to step m+1 via Eq. (AD-forward)

The entire calibration is **O(n²)** where *n* is the number of time steps
(summing over all nodes at each step). There is no inner loop for numerical
convergence.

Two-Pass Calibration with Mean Reversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the Hull-White model, transition probabilities depend on the mean-reversion
drift, which in turn depends on the calibrated rates.  We resolve this circular
dependency with a two-pass approach:

**Pass 1:** Set equal probabilities :math:`p_{\text{up}} = p_{\text{down}} = 0.5`
and run the analytical calibration to obtain initial :math:`\phi(m)` values and
rates at each node.

**Pass 2:** Using the calibrated rates from Pass 1, compute mean-reversion-adjusted
probabilities:

.. math::

   p_{\text{up}}(m, j) = \frac{1}{2} + \frac{a\bigl[\phi(m{+}1) - r(m,j)\bigr]
   \Delta t}{2\,\sigma\sqrt{\Delta t}}

clamped to :math:`[0.01, 0.99]` for numerical stability.

**Pass 3:** Re-run the analytical calibration with the updated probabilities.
This is necessary because changing the probabilities changes the Arrow-Debreu
prices, which changes the calibration condition.

In practice, two passes suffice.  The probability adjustment is a second-order
effect on the calibrated :math:`\phi`, so the re-calibration in Pass 3 is a
small correction.

.. note::

   FinancePy uses a single-pass approach with equal probabilities (no mean
   reversion in the transition probs). QuantLib uses an iterative Newton
   approach that jointly solves for rates and probabilities.  Our two-pass
   analytical method achieves the same curve-fitting accuracy as both while
   incorporating mean reversion.


Comparison with Alternative Approaches
--------------------------------------

Newton Iteration (QuantLib Style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QuantLib's ``TreeLattice`` calibrates :math:`\alpha_i` (equivalent to our
:math:`\phi(m)`) via Newton's method at each step:

1. Compute Arrow-Debreu prices at step :math:`m-1`
2. Define :math:`f(\alpha) = \sum_j Q_{m-1,j} \cdot g(\alpha) - P(0, m\Delta t)`
3. Newton update: :math:`\alpha \leftarrow \alpha - f(\alpha)/f'(\alpha)`
4. Iterate until :math:`|f| < \varepsilon`

This approach converges in 3–5 iterations per step, but introduces two sources
of error:

1. **Finite convergence tolerance** — residual :math:`\varepsilon > 0`
2. **Error accumulation** — Arrow-Debreu prices at step *m* are computed from
   the (inexactly) calibrated rates at steps 0 through *m-1*, so small errors
   compound through forward induction

In practice these errors are tiny (< 1 bp for 200 steps), but they are
unnecessary — the analytical formula :eq:`phi-formula` is exact.

Single-Pass Analytical (FinancePy Style)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FinancePy's ``HWTree`` uses the same analytical formula :eq:`phi-formula` but
with a simplification: equal probabilities everywhere (no mean-reversion
adjustment).  Their calibration loop is:

.. code-block:: python

   # FinancePy HWTree._build_tree (simplified)
   for m in range(n_steps):
       sum_qz = sum(Q[m,j] * exp(-x[j] * dt) for j in nodes(m))
       alpha[m] = log(sum_qz / discount_factors[m+1]) / dt

This matches our :eq:`phi-formula` exactly.  The difference is that FinancePy
does not adjust probabilities for mean reversion, so the rate dynamics are
slightly different (the mean reversion affects option exercise decisions,
not the ZCB repricing).

Why QuantLib and FinancePy Differ Slightly
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Despite implementing the "same" Hull-White tree, QuantLib and FinancePy
produce slightly different callable bond prices (typically within 10–50 bps).
The sources of difference are:

1. **Probability adjustment.** QuantLib adjusts transition probabilities for
   mean reversion; FinancePy uses equal probabilities.  This changes the
   distribution of rates at each step, affecting exercise boundaries.

2. **Calibration method.** QuantLib uses Newton iteration with a finite
   tolerance (:math:`\sim 10^{-8}`); FinancePy uses the analytical formula
   (exact to machine precision :math:`\sim 10^{-16}`).

3. **Tree construction details.** QuantLib uses a trinomial tree by default
   (Hull-White 1994); FinancePy uses a binomial tree.  Trinomial trees
   converge faster but have different truncation behavior at extreme nodes.

4. **Day count and schedule.** Different conventions for mapping calendar dates
   to tree steps, coupon accrual, and call schedule interpolation.

5. **Discount factor computation.** QuantLib may use continuous or discrete
   compounding depending on the rate helper; FinancePy uses continuous.

For well-calibrated trees with sufficient steps (≥ 200), these differences
are small and both converge to the same continuous-time limit.


Backward Induction
------------------

Once the tree is calibrated, we price instruments via backward induction from
the terminal step to the root.

European
~~~~~~~~

.. math::

   V_{i,j} = e^{-r_{i,j}\Delta t}\!\left[p_u \cdot V_{i+1,j+1} + p_d \cdot V_{i+1,j}\right]

American
~~~~~~~~

At every step: :math:`V_{i,j} = f\bigl(C_{i,j},\; h_{i,j}\bigr)` where
:math:`C` is the continuation value, :math:`h` is the exercise value, and
:math:`f` is the exercise function.

Bermudan
~~~~~~~~

Exercise only at specified steps :math:`\mathcal{E}`:

.. math::

   V_{i,j} = \begin{cases}
   f\bigl(C_{i,j},\; h_{i,j}\bigr) & \text{if } i \in \mathcal{E} \\
   C_{i,j} & \text{otherwise}
   \end{cases}

.. important::

   **Exercise function matters.**  The choice of :math:`f` depends on who holds
   the exercise right:

   - **Callable bonds** (issuer calls): :math:`f = \min` — the issuer exercises
     to *minimize* their liability
   - **Puttable bonds** (holder puts): :math:`f = \max` — the holder exercises
     to *maximize* their value
   - **American options** (holder exercises): :math:`f = \max`

   Using the wrong exercise function is a common and severe modeling error.

Intermediate Cashflows
~~~~~~~~~~~~~~~~~~~~~~

For coupon-bearing instruments, intermediate cashflows (coupons) must be added
during backward induction:

.. math::

   C_{i,j} = e^{-r_{i,j}\Delta t} \cdot E[V_{i+1}] + \text{cashflow}(i, j)

Without intermediate cashflows, the tree only "sees" the terminal payoff and
dramatically undervalues coupon bonds. For example, a 10Y 5% bond with only
terminal payoff valued on a rate tree gives ~61 instead of ~100.


Implementation
--------------

Generic Lattice
~~~~~~~~~~~~~~~

.. autoclass:: trellis.models.trees.lattice.RecombiningLattice
   :members:

Calibration
~~~~~~~~~~~

.. autofunction:: trellis.models.trees.lattice.calibrate_lattice

.. autofunction:: trellis.models.trees.lattice.build_rate_lattice

.. autofunction:: trellis.models.trees.lattice.build_spot_lattice

Backward Induction
~~~~~~~~~~~~~~~~~~~

.. autofunction:: trellis.models.trees.lattice.lattice_backward_induction


References
----------

- Brigo, D. & Mercurio, F. (2006). *Interest Rate Models — Theory and Practice*,
  2nd ed. Springer. **Chapter 15: The Hull-White Model.** — The primary reference
  for the analytical calibration framework. Eq. 15.9 gives the drift-fitting
  formula; Section 15.3 covers the tree implementation.

- Hull, J. & White, A. (1994). "Numerical Procedures for Implementing Term
  Structure Models: Single-Factor Models." *Journal of Derivatives*, 2(1), 7–16.
  — Original trinomial tree construction with mean reversion. Introduces the
  :math:`r = \phi + x` decomposition.

- Hull, J. & White, A. (1990). "Pricing Interest Rate Derivative Securities."
  *Review of Financial Studies*, 3(4), 573–592. — The original Hull-White model
  paper.

- Cox, J., Ross, S. & Rubinstein, M. (1979). "Option Pricing: A Simplified
  Approach." *Journal of Financial Economics*, 7(3), 229–263. — CRR binomial
  tree for spot prices (not appropriate for mean-reverting rates).

- Hull, J. (2022). *Options, Futures, and Other Derivatives*, 11th ed. Ch. 13
  (binomial trees) and Ch. 32 (short-rate models).

- FinancePy source: ``financepy/models/hw_tree.py`` — Single-pass analytical
  calibration with equal probabilities.

- QuantLib source: ``ql/models/shortrate/onefactormodels/hullwhite.cpp`` and
  ``ql/methods/lattice/treelattice.hpp`` — Newton-based calibration with
  mean-reversion-adjusted trinomial probabilities.
