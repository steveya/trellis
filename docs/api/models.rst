Models API
==========

Black76
-------

.. autofunction:: trellis.models.black76_call
.. autofunction:: trellis.models.black76_put

Volatility Surface
------------------

.. autoclass:: trellis.models.FlatVol
   :members:

.. autoclass:: trellis.models.GridVolSurface
   :members:

Trees
-----

.. autoclass:: trellis.models.trees.BinomialTree
   :members:

.. autoclass:: trellis.models.trees.TrinomialTree
   :members:

.. autofunction:: trellis.models.trees.backward_induction

Monte Carlo
-----------

.. autoclass:: trellis.models.monte_carlo.MonteCarloEngine
   :members:

.. autofunction:: trellis.models.monte_carlo.euler_maruyama
.. autofunction:: trellis.models.monte_carlo.milstein
.. autofunction:: trellis.models.monte_carlo.lsm.longstaff_schwartz

QMC
---

.. autofunction:: trellis.models.qmc.sobol_normals
.. autofunction:: trellis.models.qmc.brownian_bridge

PDE Solvers
-----------

.. autofunction:: trellis.models.pde.theta_method_1d
.. autofunction:: trellis.models.pde.crank_nicolson_1d
.. autofunction:: trellis.models.pde.implicit_fd_1d
.. autofunction:: trellis.models.pde.psor_1d
.. autofunction:: trellis.models.pde.thomas_solve
.. autoclass:: trellis.models.pde.Grid
   :members:

Transforms
----------

.. autofunction:: trellis.models.transforms.fft_price
.. autofunction:: trellis.models.transforms.cos_price

Stochastic Processes
--------------------

.. autoclass:: trellis.models.processes.GBM
   :members:
.. autoclass:: trellis.models.processes.Vasicek
   :members:
.. autoclass:: trellis.models.processes.CIR
   :members:
.. autoclass:: trellis.models.processes.HullWhite
   :members:
.. autoclass:: trellis.models.processes.Heston
   :members:
.. autoclass:: trellis.models.processes.SABRProcess
   :members:
.. autoclass:: trellis.models.processes.LocalVol
   :members:
.. autoclass:: trellis.models.processes.MertonJumpDiffusion
   :members:

Copulas
-------

.. autoclass:: trellis.models.copulas.GaussianCopula
   :members:
.. autoclass:: trellis.models.copulas.StudentTCopula
   :members:
.. autoclass:: trellis.models.copulas.FactorCopula
   :members:

Calibration
-----------

.. autofunction:: trellis.models.calibration.implied_vol
.. autofunction:: trellis.models.calibration.implied_vol_jaeckel
.. autofunction:: trellis.models.calibration.calibrate_sabr
.. autofunction:: trellis.models.calibration.dupire_local_vol

Cash Flow Engine
----------------

.. autoclass:: trellis.models.cashflow_engine.Waterfall
   :members:
.. autoclass:: trellis.models.cashflow_engine.Tranche
   :members:
.. autoclass:: trellis.models.cashflow_engine.PSA
   :members:
.. autoclass:: trellis.models.cashflow_engine.CPR
   :members:
.. autofunction:: trellis.models.cashflow_engine.level_pay
