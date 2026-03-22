Cash Flow Engine
================

The cash flow engine provides building blocks for structured product pricing:
waterfalls, prepayment models, and amortization schedules.

Waterfall (Priority of Payments)
--------------------------------

A waterfall distributes available cash to tranches in order of seniority.
Each period:

1. **Interest waterfall**: pay interest to each tranche (senior first)
2. **Principal waterfall**: distribute principal to each tranche (senior first)
3. **Residual**: any remaining cash after all tranches are paid

.. math::

   \text{Interest}_i = \min(\text{Balance}_i \times c_i \times \tau, \; \text{Remaining interest})

.. math::

   \text{Principal}_i = \min(\text{Balance}_i, \; \text{Remaining principal})

**Conservation**: total distributed = total available (no cash created or destroyed).

**Subordination**: junior tranches absorb losses first, protecting senior tranches.

Prepayment Models
-----------------

PSA (Public Securities Association)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The standard prepayment benchmark. CPR ramps linearly then stays constant:

.. math::

   \text{CPR}(t) = \begin{cases}
   0.06 \times \frac{t}{30} \times \text{speed} & t \leq 30 \\
   0.06 \times \text{speed} & t > 30
   \end{cases}

where :math:`t` is the seasoning month and speed = 1.0 is 100% PSA.

**Single Monthly Mortality** (SMM):

.. math::

   \text{SMM} = 1 - (1 - \text{CPR})^{1/12}

CPR (Constant Prepayment Rate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \text{CPR}(t) = c \quad \forall t

where :math:`c` is the annualized constant rate.

Rate-Dependent Prepayment
~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepayment increases when market rates fall below the coupon (refinancing incentive):

.. math::

   \text{CPR}(t, r) = \text{CPR}_{\text{base}} + \text{mult} \times \max(c - r, 0) \times e^{-\beta t}

where:

- :math:`c` — mortgage coupon rate
- :math:`r` — current market mortgage rate
- :math:`\beta` — burnout factor (seasoned pools prepay less)
- :math:`\text{mult}` — incentive multiplier

Amortization Schedules
----------------------

Level Pay (Fully Amortizing)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Constant periodic payment :math:`A`:

.. math::

   A = \frac{B \cdot r}{1 - (1+r)^{-n}}

where :math:`B` is the initial balance, :math:`r` is the periodic rate, and
:math:`n` is the number of periods.

Each period decomposes into interest and principal:

.. math::

   I_k = B_k \cdot r, \quad P_k = A - I_k, \quad B_{k+1} = B_k - P_k

**Properties:**

- Payments are constant: :math:`A_k = A \; \forall k`
- Interest decreases over time, principal increases
- :math:`\sum_k P_k = B_0` (entire principal repaid)
- Final balance :math:`B_n = 0`

OAS (Option-Adjusted Spread)
-----------------------------

For MBS/ABS with path-dependent prepayment:

1. Simulate :math:`N` interest rate paths
2. On each path, project prepayments and cash flows
3. Discount cash flows at path rates + OAS
4. Find OAS such that average PV = market price

.. math::

   \text{Price} = \frac{1}{N} \sum_{k=1}^{N} \sum_{t} \frac{CF_t^{(k)}}{(1 + r_t^{(k)} + \text{OAS})^t}

Implementation
--------------

.. autoclass:: trellis.models.cashflow_engine.waterfall.Waterfall
   :members:

.. autoclass:: trellis.models.cashflow_engine.waterfall.Tranche
   :members:

.. autoclass:: trellis.models.cashflow_engine.prepayment.PSA
   :members:

.. autofunction:: trellis.models.cashflow_engine.amortization.level_pay

Numerical Example
-----------------

$100M MBS, 6% coupon, 100% PSA, two tranches (80/20):

.. code-block:: python

   from trellis.models.cashflow_engine.waterfall import Waterfall, Tranche
   from trellis.models.cashflow_engine.prepayment import PSA
   from trellis.models.cashflow_engine.amortization import level_pay

   # Generate base amortization
   schedule = level_pay(100e6, 0.06/12, 360)

   # Apply prepayment
   psa = PSA(speed=1.0)
   # ... adjust each period by smm

   # Run waterfall
   wf = Waterfall([
       Tranche("A", 80e6, 0.04, subordination=0),
       Tranche("B", 20e6, 0.06, subordination=1),
   ])
   results = wf.run(schedule, period=1/12)

References
----------

- Fabozzi, F. J. (2006). *The Handbook of Mortgage-Backed Securities*, 6th ed. McGraw-Hill.
- Hayre, L. (2001). *Salomon Smith Barney Guide to Mortgage-Backed and
  Asset-Backed Securities*. Wiley.
