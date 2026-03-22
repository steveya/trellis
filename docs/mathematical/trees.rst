Lattice Methods (Trees)
=======================

Binomial trees discretize the underlying process into up/down moves at each
time step. Combined with backward induction, they price European, American,
and Bermudan options.

Binomial Tree Construction
--------------------------

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
--------------

.. math::

   u = e^{\sigma\sqrt{2\Delta t}}, \quad d = 1/u

.. math::

   p_u + p_m + p_d = 1

Trinomial trees converge faster and handle barriers more naturally.

Backward Induction
------------------

**European:**

.. math::

   V_{i,j} = e^{-r\Delta t}\!\left[p \cdot V_{i+1,j+1} + (1-p) \cdot V_{i+1,j}\right]

**American:** :math:`V_{i,j} = \max(C_{i,j}, \, h_{i,j})` where :math:`h` is intrinsic value.

**Bermudan:** exercise only at specified steps :math:`\mathcal{E}`.

Convergence: CRR with 200 steps matches BS within 1%. Put-call parity holds on the tree.

Implementation
--------------

.. autoclass:: trellis.models.trees.binomial.BinomialTree
   :members:

.. autofunction:: trellis.models.trees.backward_induction.backward_induction

References
----------

- Cox, Ross & Rubinstein (1979). *Journal of Financial Economics*, 7(3), 229-263.
- Hull (2022). *Options, Futures, and Other Derivatives*, 11th ed. Ch. 13.
