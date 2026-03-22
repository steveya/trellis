Models API
==========

Black76
-------

.. autofunction:: trellis.models.black.black76_call
.. autofunction:: trellis.models.black.black76_put

Volatility Surface
------------------

.. autoclass:: trellis.models.vol_surface.FlatVol
   :members:

Trees
-----

.. autoclass:: trellis.models.trees.binomial.BinomialTree
   :members:

.. autoclass:: trellis.models.trees.trinomial.TrinomialTree
   :members:

.. autofunction:: trellis.models.trees.backward_induction.backward_induction

Monte Carlo
-----------

.. autoclass:: trellis.models.monte_carlo.engine.MonteCarloEngine
   :members:

.. autofunction:: trellis.models.monte_carlo.discretization.euler_maruyama
.. autofunction:: trellis.models.monte_carlo.discretization.milstein
.. autofunction:: trellis.models.monte_carlo.lsm.longstaff_schwartz

PDE Solvers
-----------

.. autofunction:: trellis.models.pde.crank_nicolson.crank_nicolson_1d
.. autofunction:: trellis.models.pde.implicit_fd.implicit_fd_1d
.. autofunction:: trellis.models.pde.psor.psor_1d
.. autofunction:: trellis.models.pde.thomas.thomas_solve
.. autoclass:: trellis.models.pde.grid.Grid
   :members:

Transforms
----------

.. autofunction:: trellis.models.transforms.fft_pricer.fft_price
.. autofunction:: trellis.models.transforms.cos_method.cos_price

Stochastic Processes
--------------------

.. autoclass:: trellis.models.processes.gbm.GBM
   :members:
.. autoclass:: trellis.models.processes.vasicek.Vasicek
   :members:
.. autoclass:: trellis.models.processes.cir.CIR
   :members:
.. autoclass:: trellis.models.processes.hull_white.HullWhite
   :members:
.. autoclass:: trellis.models.processes.heston.Heston
   :members:
.. autoclass:: trellis.models.processes.sabr.SABRProcess
   :members:
.. autoclass:: trellis.models.processes.local_vol.LocalVol
   :members:
.. autoclass:: trellis.models.processes.jump_diffusion.MertonJumpDiffusion
   :members:

Copulas
-------

.. autoclass:: trellis.models.copulas.gaussian.GaussianCopula
   :members:
.. autoclass:: trellis.models.copulas.student_t.StudentTCopula
   :members:
.. autoclass:: trellis.models.copulas.factor.FactorCopula
   :members:

Calibration
-----------

.. autofunction:: trellis.models.calibration.implied_vol.implied_vol
.. autofunction:: trellis.models.calibration.implied_vol.implied_vol_jaeckel
.. autofunction:: trellis.models.calibration.sabr_fit.calibrate_sabr
.. autofunction:: trellis.models.calibration.local_vol.dupire_local_vol

Cash Flow Engine
----------------

.. autoclass:: trellis.models.cashflow_engine.waterfall.Waterfall
   :members:
.. autoclass:: trellis.models.cashflow_engine.waterfall.Tranche
   :members:
.. autoclass:: trellis.models.cashflow_engine.prepayment.PSA
   :members:
.. autoclass:: trellis.models.cashflow_engine.prepayment.CPR
   :members:
.. autofunction:: trellis.models.cashflow_engine.amortization.level_pay
