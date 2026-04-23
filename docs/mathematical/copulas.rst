Copula Methods
==============

Copulas model the dependence structure between default events in a portfolio,
separating marginal default probabilities from their joint distribution.

Gaussian Copula
---------------

The standard model for CDO and basket credit pricing (Li, 2000).

For :math:`n` names with marginal default probabilities :math:`p_i`:

1. Generate correlated normals :math:`\mathbf{X} = L\mathbf{Z}` where :math:`L` is
   the Cholesky factor of the correlation matrix :math:`\Sigma` and
   :math:`\mathbf{Z} \sim N(\mathbf{0}, I)`.

2. Convert to uniform: :math:`U_i = \Phi(X_i)` where :math:`\Phi` is the standard
   normal CDF.

3. Convert to default times: :math:`\tau_i = -\ln(U_i) / \lambda_i` where
   :math:`\lambda_i` is the hazard rate.

A name :math:`i` defaults before time :math:`T` if :math:`\tau_i < T`, i.e.,
:math:`U_i < 1 - S_i(T)` where :math:`S_i(T) = e^{-\lambda_i T}`.

Student-t Copula
----------------

Replaces the Gaussian with a Student-t distribution for fatter tails:

1. Generate :math:`\mathbf{Z} \sim N(\mathbf{0}, I)` and :math:`\chi^2 \sim \chi^2_\nu`
2. Form :math:`\mathbf{T} = L\mathbf{Z} \cdot \sqrt{\nu / \chi^2}`
3. :math:`U_i = t_\nu(T_i)` where :math:`t_\nu` is the Student-t CDF

Lower degrees of freedom :math:`\nu` → more joint extreme events → fatter tails
in the loss distribution.

One-Factor Gaussian Copula
--------------------------

The standard CDO pricing model decomposes each name's latent variable:

.. math::

   X_i = \sqrt{\rho} \, M + \sqrt{1-\rho} \, Z_i

where :math:`M` is the common (systematic) factor and :math:`Z_i` are
idiosyncratic factors, all standard normal and independent.

**Conditional default probability** given :math:`M = m`:

.. math::

   p(m) = \Phi\!\left(\frac{\Phi^{-1}(p_i) - \sqrt{\rho}\,m}{\sqrt{1-\rho}}\right)

For a homogeneous portfolio (all :math:`p_i = p`), the number of defaults
conditional on :math:`M` is binomial:

.. math::

   \mathbb{P}[k \text{ defaults} \mid M = m] = \binom{n}{k} p(m)^k (1-p(m))^{n-k}

**Unconditional loss distribution** via numerical integration over :math:`M`:

.. math::

   \mathbb{P}[k \text{ defaults}] = \int_{-\infty}^{\infty} \binom{n}{k} p(m)^k (1-p(m))^{n-k} \, \phi(m) \, dm

Computed via Gauss-Hermite quadrature in our implementation.

**Key properties:**

- :math:`\mathbb{E}[\text{loss fraction}] = p` (independent of :math:`\rho`)
- Higher :math:`\rho` → heavier tails (more probability of extreme losses)
- At :math:`\rho = 0`: loss distribution = binomial :math:`B(n, p)` (independent defaults)
- At :math:`\rho = 1`: all default or none (Bernoulli)

CDO Tranche Pricing
~~~~~~~~~~~~~~~~~~~~

A CDO tranche :math:`[a, d]` (attachment :math:`a`, detachment :math:`d`) has
expected tranche loss:

.. math::

   \text{ETL} = \sum_{k=0}^{n} \mathbb{P}[k\text{ defaults}] \cdot \min\!\left(\max\!\left(\frac{k}{n} - a, 0\right), d - a\right)

The tranche PV = notional × ETL × discount factor.

Base Correlation
~~~~~~~~~~~~~~~~

In practice, the market quotes tranche spreads that imply different
correlations :math:`\rho` for each tranche. The **base correlation** framework
finds the :math:`\rho` that reprices each equity tranche :math:`[0, d_i]`.

The checked calibration support is narrower than a production
base-correlation bootstrap. ``trellis.models.calibration.basket_credit`` fits
exact-node tranche-implied correlations for homogeneous basket fixtures that
share one representative calibrated ``CreditCurve``. Inputs are normalized by
maturity, attachment, detachment, quote family, quote style, and quote value.
Supported quote families use the existing bounded calibration vocabulary:
``Price`` for tranche PV or expected-loss-fraction targets and ``Spread`` for
fair-spread-in-basis-points targets.

The fitted ``BasketCreditCorrelationSurface`` records quote residuals, root
failures for impossible quotes, simple tranche-bound warnings, and
monotonicity/smoothness warnings. It can be materialized on ``MarketState`` as
``correlation_surface`` so downstream basket-tranche helpers can consume it
when a contract does not carry an explicit correlation.

This does not yet support heterogeneous name-level credit curves, index-credit
conventions, bid/ask tranche surfaces, interpolated base-correlation term
structures, or arbitrage repair across a production tranche grid.

Implementation
--------------

.. autoclass:: trellis.models.copulas.gaussian.GaussianCopula
   :members:

.. autoclass:: trellis.models.copulas.student_t.StudentTCopula
   :members:

.. autoclass:: trellis.models.copulas.factor.FactorCopula
   :members:

Numerical Example
-----------------

100-name portfolio, marginal default prob 5%, correlation 30%:

.. code-block:: python

   from trellis.models.copulas.factor import FactorCopula
   fc = FactorCopula(n_names=100, correlation=0.3)
   losses, probs = fc.loss_distribution(0.05)
   # E[loss] ≈ 5%, but P(>20 defaults) is significant due to correlation

References
----------

- Li, D. (2000). "On default correlation: A copula function approach."
  *Journal of Fixed Income*, 9(4), 43-54.
- Schönbucher, P. (2003). *Credit Derivatives Pricing Models*. Wiley.
- O'Kane, D. (2008). *Modelling Single-name and Multi-name Credit Derivatives*. Wiley.
