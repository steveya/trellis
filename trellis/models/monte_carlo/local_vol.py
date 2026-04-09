"""Reusable local-volatility Monte Carlo helpers for vanilla equity options."""

from __future__ import annotations

from dataclasses import dataclass

from trellis.core.differentiable import get_numpy
from trellis.models.monte_carlo.event_aware import (
    EventAwareMonteCarloProblemSpec,
    EventAwareMonteCarloProcessSpec,
    build_event_aware_monte_carlo_problem,
    price_event_aware_monte_carlo,
)

np = get_numpy()


@dataclass(frozen=True)
class LocalVolMonteCarloResult:
    """Monte Carlo result for a vanilla local-vol option price."""

    price: float
    std_error: float
    n_paths: int


def local_vol_european_vanilla_price_result(
    *,
    spot: float,
    strike: float,
    maturity: float,
    discount_curve,
    local_vol_surface,
    option_type: str = "call",
    dividend_yield: float = 0.0,
    n_paths: int = 20_000,
    n_steps: int = 100,
    seed: int | None = None,
) -> LocalVolMonteCarloResult:
    """Price a European vanilla option under a supplied local-vol surface.

    Parameters
    ----------
    spot
        Current underlying spot.
    strike
        Option strike.
    maturity
        Time to expiry in years.
    discount_curve
        Discount curve supplying the risk-free zero rate.
    local_vol_surface
        Callable local-vol surface ``sigma(S, t)``.
    option_type
        ``"call"`` or ``"put"``.
    dividend_yield
        Continuous dividend yield. Defaults to zero when the task only supplies
        spot, discounting, and local volatility.
    """
    option = option_type.lower().strip()
    if option not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if maturity <= 0:
        intrinsic = max(spot - strike, 0.0) if option == "call" else max(strike - spot, 0.0)
        return LocalVolMonteCarloResult(price=float(intrinsic), std_error=0.0, n_paths=0)

    risk_free_rate = float(discount_curve.zero_rate(maturity))

    def terminal_payoff(terminal):
        if option == "call":
            return np.maximum(terminal - strike, 0.0)
        return np.maximum(strike - terminal, 0.0)

    problem = build_event_aware_monte_carlo_problem(
        EventAwareMonteCarloProblemSpec(
            process_spec=EventAwareMonteCarloProcessSpec(
                family="local_vol_1d",
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                local_vol_surface=local_vol_surface,
            ),
            initial_state=spot,
            maturity=maturity,
            n_steps=n_steps,
            discount_rate=risk_free_rate,
            path_requirement_kind="terminal_only",
            reducer_kind="terminal_payoff",
            terminal_payoff=terminal_payoff,
        )
    )

    result = price_event_aware_monte_carlo(
        problem,
        n_paths=n_paths,
        seed=seed,
        return_paths=False,
    )
    return LocalVolMonteCarloResult(
        price=float(result["price"]),
        std_error=float(result["std_error"]),
        n_paths=int(result["n_paths"]),
    )


def local_vol_european_vanilla_price(**kwargs) -> float:
    """Return the scalar present value of a vanilla local-vol MC price."""
    return local_vol_european_vanilla_price_result(**kwargs).price
