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

Scalar Diffusion Market Resolution
----------------------------------

.. autoclass:: trellis.models.resolution.ResolvedScalarDiffusionMarketInputs
   :members:

.. autofunction:: trellis.models.resolution.resolve_scalar_diffusion_market_inputs

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

.. autoclass:: trellis.models.monte_carlo.RunningExtremumContract
   :members:

.. autoclass:: trellis.models.monte_carlo.SquaredLogReturnContract
   :members:

.. autofunction:: trellis.models.monte_carlo.discrete_path_extremum
.. autofunction:: trellis.models.monte_carlo.annualized_squared_log_return_sum
.. autofunction:: trellis.models.monte_carlo.build_running_extremum_reducer
.. autofunction:: trellis.models.monte_carlo.build_squared_log_return_reducer

.. autoclass:: trellis.models.monte_carlo.ScalarTransitionObservation
   :members:

.. autoclass:: trellis.models.monte_carlo.ScalarTransitionReducer
   :members:

.. autoclass:: trellis.models.monte_carlo.ConditionalBridgeExtremumContract
   :members:

.. autoclass:: trellis.models.monte_carlo.MonteCarloRandomInputs
   :members:

.. autofunction:: trellis.models.monte_carlo.conditional_log_bridge_extremum
.. autofunction:: trellis.models.monte_carlo.build_conditional_bridge_extremum_reducer
.. autofunction:: trellis.models.monte_carlo.replay_scalar_transition_reducers
.. autofunction:: trellis.models.monte_carlo.transition_state.coerce_transition_uniforms

QMC
---

.. autofunction:: trellis.models.qmc.sobol_normals
.. autofunction:: trellis.models.qmc.sobol_transition_inputs
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
.. autofunction:: trellis.models.transforms.price_heston_option_transform
.. autofunction:: trellis.models.transforms.price_heston_option_transform_result

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
