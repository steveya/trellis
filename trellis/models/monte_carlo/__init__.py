"""Monte Carlo simulation: path generation, discretization, variance reduction."""

from trellis.models.monte_carlo.early_exercise import (
    EarlyExerciseDiagnostics,
    EarlyExercisePolicyResult,
    FastLaguerreContinuationEstimator,
    FastPolynomialContinuationEstimator,
    LeastSquaresContinuationEstimator,
    default_continuation_estimator,
)
from trellis.models.monte_carlo.basket_state import (
    build_basket_path_requirement,
    evaluate_ranked_observation_basket_paths,
    evaluate_ranked_observation_basket_state,
    observation_step_indices,
)
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.monte_carlo.discretization import euler_maruyama, milstein
from trellis.models.monte_carlo.brownian_bridge import brownian_bridge
from trellis.models.monte_carlo.event_state import (
    PathEventRecord,
    PathEventSpec,
    PathEventState,
    PathEventTimeline,
    apply_path_event_spec,
    build_event_path_requirement,
    event_step_indices,
    replay_path_event_timeline,
)
from trellis.models.monte_carlo.local_vol import (
    LocalVolMonteCarloResult,
    local_vol_european_vanilla_price,
    local_vol_european_vanilla_price_result,
)
from trellis.models.monte_carlo.quanto import (
    build_quanto_mc_initial_state,
    build_quanto_mc_process,
    price_quanto_option_monte_carlo,
    recommended_quanto_mc_engine_kwargs,
    terminal_quanto_option_payoff,
)
from trellis.models.monte_carlo.ranked_observation_payoffs import (
    build_ranked_observation_basket_initial_state,
    build_ranked_observation_basket_process,
    build_ranked_observation_basket_state_payoff,
    price_ranked_observation_basket_monte_carlo,
    recommended_ranked_observation_basket_mc_engine_kwargs,
    terminal_ranked_observation_basket_payoff,
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
from trellis.models.monte_carlo.semantic_basket import (
    RankedObservationBasketMonteCarloPayoff,
    RankedObservationBasketSpec,
)
from trellis.models.monte_carlo.profiling import (
    MonteCarloPathKernelBenchmark,
    benchmark_path_kernel,
)
