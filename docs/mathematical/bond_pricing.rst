Bond Pricing
============

Fixed-Rate Coupon Bond
----------------------

A fixed-rate bond pays periodic coupons and returns the face value at maturity.

Dirty Price
~~~~~~~~~~~

The dirty price (full price) is the present value of all future cashflows:

.. math::

   P_{\text{dirty}} = \sum_{i=1}^{n} \frac{C}{f} \cdot D(t_i) + F \cdot D(t_n)

where:

- :math:`C` — annual coupon rate (e.g. 0.05 for 5%)
- :math:`f` — coupon frequency (1=annual, 2=semi-annual, 4=quarterly)
- :math:`F` — face value (typically 100)
- :math:`D(t)` — discount factor at time :math:`t`
- :math:`t_i` — time to the :math:`i`-th coupon payment
- :math:`n` — number of remaining coupon periods

Under continuous compounding with rate :math:`r(t)`:

.. math::

   D(t) = e^{-r(t) \cdot t}

Clean Price
~~~~~~~~~~~

.. math::

   P_{\text{clean}} = P_{\text{dirty}} - AI

where :math:`AI` is the accrued interest:

.. math::

   AI = \frac{C}{f} \cdot \frac{\text{days since last coupon}}{\text{days in coupon period}}

Interest Rate Sensitivity (Greeks)
----------------------------------

DV01 (Dollar Value of a Basis Point)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The change in price for a 1 basis point parallel shift in yields:

.. math::

   \text{DV01} = -\frac{\partial P}{\partial y} \times 0.0001

In Trellis, DV01 is computed via automatic differentiation (autograd):

.. math::

   \text{DV01} = -\sum_i \frac{\partial P}{\partial r_i} \times 0.0001

where :math:`r_i` are the zero rates at each curve tenor.

Modified Duration
~~~~~~~~~~~~~~~~~

.. math::

   D_{\text{mod}} = -\frac{1}{P} \frac{\partial P}{\partial y}

For a zero-coupon bond: :math:`D_{\text{mod}} = T` (maturity).

For a coupon bond: :math:`D_{\text{mod}} < T`.

**Relationship to DV01:**

.. math::

   \text{DV01} = D_{\text{mod}} \times P \times 0.0001

Convexity
~~~~~~~~~

The second-order sensitivity to yield changes:

.. math::

   \text{Convexity} = \frac{1}{P} \frac{\partial^2 P}{\partial y^2}

Computed via autograd second derivatives when the payoff path is differentiable,
with finite-difference fallback only for nonsmooth routes:

.. math::

   \text{Convexity} \approx \frac{1}{P} \frac{\partial^2 P}{\partial y^2}

For forward-only routes, the fallback central-difference approximation uses
:math:`\Delta y = 10^{-4}`.

**Price approximation with convexity:**

.. math::

   \Delta P \approx -D_{\text{mod}} \cdot P \cdot \Delta y + \frac{1}{2} \cdot \text{Convexity} \cdot P \cdot (\Delta y)^2

Key Rate Durations
~~~~~~~~~~~~~~~~~~

Sensitivity to individual tenor points on the yield curve:

.. math::

   \text{KRD}_i = -\frac{1}{P} \frac{\partial P}{\partial r_i}

where :math:`r_i` is the zero rate at tenor :math:`i`. The sum of KRDs
approximates the total modified duration:

.. math::

   D_{\text{mod}} \approx \sum_i \text{KRD}_i

Implementation
--------------

.. autofunction:: trellis.engine.pricer.price_instrument
   :no-index:

.. autoclass:: trellis.instruments.bond.Bond
   :members:
   :no-index:

Numerical Example
-----------------

10-year semi-annual 5% coupon bond, flat curve at 5%:

.. code-block:: python

   from trellis import Bond, YieldCurve, price
   from datetime import date

   bond = Bond(face=100, coupon=0.05, maturity_date=date(2034, 11, 15),
               maturity=10, frequency=2)
   curve = YieldCurve.flat(0.05)
   result = price(bond, curve, date(2024, 11, 15))

   # Dirty price ≈ 100 (at par since coupon = yield)
   # DV01 ≈ 0.079
   # Duration ≈ 7.9 years

References
----------

- Fabozzi, F. J. (2007). *Fixed Income Analysis*, 2nd ed. Wiley.
- Tuckman, B. & Serrat, A. (2011). *Fixed Income Securities*, 3rd ed. Wiley.
