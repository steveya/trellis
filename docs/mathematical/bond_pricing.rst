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

   AI = \frac{C}{f} \cdot \frac{\alpha(t_{\text{last}}, t_{\text{settle}})}{\alpha(t_{\text{last}}, t_{\text{next}})}

where :math:`\alpha(\cdot, \cdot)` is the selected bond day-count fraction,
evaluated over both the elapsed accrual stub and the full coupon period. In
Trellis this is computed from the explicit coupon schedule, so regular and
short/long periods use the same convention-aware fractioning as the rest of
the dated cashflow machinery.

Yield To Maturity
~~~~~~~~~~~~~~~~~

Trellis reports ``ytm`` as a nominal annual yield compounded at the bond coupon
frequency. It is solved from the dirty price by finding :math:`y` such that:

.. math::

   P_{\text{dirty}} = \sum_{i=1}^{n} \frac{CF_i}{\left(1 + y / f\right)^{f t_i}}

where :math:`CF_i` are the remaining coupon/principal cashflows and
:math:`t_i` are the settlement-to-payment year fractions under the bond day
count. This keeps the reported ``ytm`` aligned with standard street-style
nominal bond yields even though the underlying curve object may use continuous
zero rates internally.

Current reporting intentionally stops short of full settlement-convention
coverage: ex-coupon windows, settlement lags, and other market-specific clean
price adjustments are not modeled in this surface yet.

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

The runtime KRD measure now evaluates these sensitivities on the user-requested
bucket grid itself. Trellis first re-expresses the base zero curve on that
piecewise-linear tenor grid, then applies symmetric bucket bumps there. That
preserves interpolation-aware off-grid buckets such as ``7Y`` on a base curve
whose original knots might only include ``5Y`` and ``10Y``. The shared
curve-shock substrate still reports explicit warnings when a requested bucket
sits outside curve support or spans an unusually wide interpolation interval.

For bootstrap-backed discount curves, Trellis also supports a rebuild-based
variant. Instead of shocking the final zero-rate nodes directly, it bumps the
quoted calibration instruments, rebuilds the zero curve, and then reprices on
that rebuilt surface. The runtime result discloses which methodology ran
through attached metadata so review tools can distinguish zero-curve KRD from
quote-space rebuild KRD.

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
