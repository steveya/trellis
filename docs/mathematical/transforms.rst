Transform Methods
=================

Price European options via the characteristic function :math:`\varphi(u) = \mathbb{E}[e^{iu\ln S_T}]`.

FFT (Carr-Madan 1999)
---------------------

.. math::

   C(k) = \frac{e^{-\alpha k}}{\pi}\,\text{Re}\!\left[\text{FFT}\!\left(e^{ivb}\,\psi(v)\,\eta\,w\right)\right]

where :math:`\psi(v) = \frac{e^{-rT}\varphi(v-(1+\alpha)i)}{\alpha^2+\alpha-v^2+i(2\alpha+1)v}`.

Computed via a single FFT call: :math:`O(N\log N)`.

COS Method (Fang-Oosterlee 2008)
---------------------------------

Cosine expansion on :math:`[a,b]`:

.. math::

   C = Ke^{-rT}\sum_{k=0}^{N-1}{}'\,\text{Re}\!\left[\varphi\!\left(\frac{k\pi}{b-a}\right)e^{ik\pi\frac{x-a}{b-a}}\right]U_k

Payoff coefficients :math:`U_k` involve integrals :math:`\chi_k` and :math:`\psi_k`.
Exponential convergence for smooth densities.

Characteristic Functions
------------------------

**GBM**: :math:`\varphi(u) = \exp[iu(\ln S_0 + (r-\sigma^2/2)T) - \sigma^2Tu^2/2]`

**Heston**: closed-form via :math:`C, D` functions (see :doc:`processes`).
Pass ``log_spot=np.log(S0)`` when using with FFT/COS.

Heston Route Binding
--------------------

``trellis.models.transforms.heston`` provides the checked helper surface used
by task routes for Heston FFT and COS pricing. It resolves the runtime
``Heston`` process from ``market_state.model_parameters``, the underlier spot
from the spec or ``market_state.spot``, and the discount curve, then binds the
characteristic function to the existing FFT/COS kernels.
The helper does not read ``market_state.vol_surface``; Black volatility
surfaces are calibration targets or comparison evidence, not live substitutes
for Heston model parameters.

Gauss-Laguerre Heston transform targets remain fail-closed until a checked
quadrature kernel is added. The helper raises a typed repair packet instead of
falling back to a vanilla Black-vol adapter. The packet includes a
``quadrature_contract`` naming the Heston characteristic-function binding,
required model parameters, Gauss-Laguerre nodes/weights, damping or contour
policy, oscillatory-integrand stabilization, diagnostics, and the missing
``heston_gauss_laguerre_transform_kernel`` plus validation bundle.

Implementation
--------------

.. autofunction:: trellis.models.transforms.fft_pricer.fft_price
.. autofunction:: trellis.models.transforms.cos_method.cos_price
.. autofunction:: trellis.models.transforms.heston.price_heston_option_transform
.. autofunction:: trellis.models.transforms.heston.price_heston_option_transform_result

References
----------

- Carr & Madan (1999). *Journal of Computational Finance*, 2(4), 61-73.
- Fang & Oosterlee (2008). *SIAM J. Sci. Comput.*, 31(2), 826-848.
