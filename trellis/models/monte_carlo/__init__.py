"""Monte Carlo simulation: path generation, discretization, variance reduction."""

from trellis.models.monte_carlo.early_exercise import (
    EarlyExerciseDiagnostics,
    EarlyExercisePolicyResult,
    FastLaguerreContinuationEstimator,
    FastPolynomialContinuationEstimator,
    LeastSquaresContinuationEstimator,
    default_continuation_estimator,
)
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.discretization import euler_maruyama, milstein
from trellis.models.monte_carlo.brownian_bridge import brownian_bridge
from trellis.models.monte_carlo.local_vol import (
    LocalVolMonteCarloResult,
    local_vol_european_vanilla_price,
    local_vol_european_vanilla_price_result,
)
from trellis.models.monte_carlo.lsm import longstaff_schwartz, longstaff_schwartz_result
from trellis.models.monte_carlo.path_state import (
    BarrierMonitor,
    MonteCarloPathRequirement,
    MonteCarloPathState,
    PathReducer,
    StateAwarePayoff,
    barrier_payoff,
    terminal_value_payoff,
)
from trellis.models.monte_carlo.primal_dual import primal_dual_mc, primal_dual_mc_result
from trellis.models.monte_carlo.stochastic_mesh import stochastic_mesh, stochastic_mesh_result
from trellis.models.monte_carlo.tv_regression import (
    tsitsiklis_van_roy,
    tsitsiklis_van_roy_result,
)
from trellis.models.monte_carlo.variance_reduction import (
    antithetic,
    antithetic_normals,
    brownian_bridge_increments,
    control_variate,
    sobol_normals,
)
