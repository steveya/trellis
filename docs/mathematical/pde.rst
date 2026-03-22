PDE Methods
===========

Black-Scholes PDE:

.. math::

   \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
   + rS\frac{\partial V}{\partial S} - rV = 0

Finite Difference Coefficients
------------------------------

On uniform grid :math:`\Delta S`, central differences give tridiagonal coefficients:

.. math::

   \alpha_i = \frac{\Delta t}{2}\!\left(\frac{\sigma^2 S_i^2}{\Delta S^2} - \frac{rS_i}{2\Delta S}\right), \quad
   \beta_i = -\frac{\Delta t}{2}\!\left(\frac{2\sigma^2 S_i^2}{\Delta S^2} + r\right), \quad
   \gamma_i = \frac{\Delta t}{2}\!\left(\frac{\sigma^2 S_i^2}{\Delta S^2} + \frac{rS_i}{2\Delta S}\right)

**Crank-Nicolson** (:math:`\theta = 0.5`): :math:`(I - L/2)V^n = (I + L/2)V^{n+1}`. Second-order in time and space.

**Fully Implicit** (:math:`\theta = 1`): :math:`(I - L)V^n = V^{n+1}`. First-order, unconditionally stable.

Thomas Algorithm
----------------

Tridiagonal :math:`Ax = d` in :math:`O(n)` via forward sweep + back substitution.

PSOR for American Options
-------------------------

Linear complementarity :math:`V \geq g, \; \mathcal{L}V \leq 0, \; (V-g)\mathcal{L}V = 0`.

Projected SOR iterates and projects: :math:`V_i \leftarrow \max(V_i^{\text{SOR}}, g(S_i))`.

Implementation
--------------

.. autofunction:: trellis.models.pde.crank_nicolson.crank_nicolson_1d
.. autofunction:: trellis.models.pde.psor.psor_1d
.. autofunction:: trellis.models.pde.thomas.thomas_solve

References
----------

- Wilmott (2006). *Paul Wilmott on Quantitative Finance*, 2nd ed. Ch. 77-79.
- Tavella & Randall (2000). *Pricing Financial Instruments: The Finite Difference Method*. Wiley.
