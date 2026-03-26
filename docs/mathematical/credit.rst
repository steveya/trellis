Credit Modeling
===============

Hazard Rate and Survival Probability
-------------------------------------

Under the reduced-form (intensity) framework, default is modeled as the
first jump of a Poisson process with intensity (hazard rate) :math:`\lambda(t)`.

Survival Probability
~~~~~~~~~~~~~~~~~~~~

The probability of surviving to time :math:`t`:

.. math::

   S(t) = \mathbb{P}[\tau > t] = e^{-\int_0^t \lambda(s) \, ds}

For a flat (constant) hazard rate :math:`\lambda`:

.. math::

   S(t) = e^{-\lambda t}

Risky Discount Factor
~~~~~~~~~~~~~~~~~~~~~

The risky discount factor combines credit risk and interest rate risk:

.. math::

   D_{\text{risky}}(t) = S(t) \cdot D(t) = e^{-\lambda t} \cdot e^{-r(t) \cdot t}

CDS Spread Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~

For a flat hazard rate and constant recovery :math:`R`:

.. math::

   \text{CDS spread} \approx \lambda \cdot (1 - R)

This is the first-order approximation. The exact relationship requires
solving the protection and premium leg equations.

From CDS Spreads to Hazard Rates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given CDS par spreads :math:`s_i` at tenors :math:`T_i`:

.. math::

   \lambda_i \approx \frac{s_i}{1 - R}

This is the bootstrap approximation used in ``CreditCurve.from_spreads()``.

Credit Curve Scenarios
~~~~~~~~~~~~~~~~~~~~~~

The ``shift(bps)`` method creates a parallel-shifted credit curve:

.. math::

   \lambda'(t) = \lambda(t) + \frac{\text{bps}}{10{,}000}

Implementation
--------------

.. autoclass:: trellis.curves.credit_curve.CreditCurve
   :members:
   :no-index:

Numerical Example
-----------------

Flat hazard rate λ = 200bp, recovery R = 40%:

.. math::

   S(5) = e^{-0.02 \times 5} = 0.9048

   \text{CDS spread} = 0.02 \times (1 - 0.4) = 0.012 = 120\text{bp}

References
----------

- Duffie, D. & Singleton, K. (2003). *Credit Risk*. Princeton University Press.
- O'Kane, D. (2008). *Modelling Single-name and Multi-name Credit Derivatives*. Wiley.
