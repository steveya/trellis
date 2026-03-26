Monte Carlo Methods
===================

.. math::

   V_0 = e^{-rT} \cdot \frac{1}{N} \sum_{k=1}^{N} f(S_T^{(k)})

SDE Discretization
------------------

**Euler-Maruyama** (order 0.5):

.. math::

   X_{t+\Delta t} = X_t + \mu(X_t, t)\Delta t + \sigma(X_t, t)\sqrt{\Delta t} \, Z

**Milstein** (order 1.0):

.. math::

   X_{t+\Delta t} = X_t + \mu\Delta t + \sigma\sqrt{\Delta t} \, Z
   + \tfrac{1}{2}\sigma\sigma'(Z^2 - 1)\Delta t

**Exact simulation** for GBM:

.. math::

   S_{t+\Delta t} = S_t \exp\!\left[(\mu - \sigma^2/2)\Delta t + \sigma\sqrt{\Delta t}\,Z\right]

Brownian Bridge
---------------

.. math::

   W(t_m) \mid W(t_1), W(t_2) \sim N\!\left(\frac{(t_2 - t_m)W_1 + (t_m - t_1)W_2}{t_2 - t_1},\;
   \frac{(t_m - t_1)(t_2 - t_m)}{t_2 - t_1}\right)

Variance Reduction
------------------

**Antithetic**: average :math:`f(S^+)` and :math:`f(S^-)` with mirrored paths.

**Control variate**: :math:`\hat{V}_{\text{cv}} = \hat{V} - \beta^*(\hat{C} - \mathbb{E}[C])` with :math:`\beta^* = \text{Cov}(V,C)/\text{Var}(C)`.

**Quasi-random (Sobol)**: low-discrepancy sequences give nearly :math:`O(N^{-1})` convergence.

Canonical package surface: ``trellis.models.qmc`` re-exports Sobol normals and
Brownian-bridge helpers while the estimator logic remains in ``trellis.models.monte_carlo``.

Longstaff-Schwartz (LSM)
-------------------------

For American/Bermudan options: regress continuation value on basis functions at each exercise date, exercise if intrinsic > continuation estimate.

Standard error: :math:`\text{SE} = \hat\sigma / \sqrt{N}`, convergence :math:`O(N^{-1/2})`.

Implementation
--------------

.. autoclass:: trellis.models.monte_carlo.engine.MonteCarloEngine
   :members:

.. autofunction:: trellis.models.monte_carlo.lsm.longstaff_schwartz
   :no-index:
.. autofunction:: trellis.models.qmc.sobol_normals
   :no-index:
.. autofunction:: trellis.models.qmc.brownian_bridge
   :no-index:

References
----------

- Glasserman (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Longstaff & Schwartz (2001). *Review of Financial Studies*, 14(1), 113-147.
