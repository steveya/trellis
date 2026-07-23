"""Compatibility wrappers for terminal basket pricing.

Generated code should compose the public resolver and raw model primitives
directly. These product-level functions remain for callers of the historical
API and for independent comparison evidence.
"""

from __future__ import annotations

from trellis.core.differentiable import get_numpy
from trellis.core.market_state import MarketState
from trellis.models.analytical.support import implied_zero_rate
from trellis.models.analytical.terminal_basket import (
    two_asset_extremum_option_stulz,
    two_asset_spread_option_kirk,
    two_asset_terminal_basket_gauss_hermite,
)
from trellis.models.monte_carlo.engine import MonteCarloEngine
from trellis.models.payoffs import terminal_basket_option_payoff
from trellis.models.processes.correlated_gbm import CorrelatedGBM
from trellis.models.resolution.basket_semantics import ResolvedBasketSemantics
from trellis.models.resolution.terminal_basket import (
    ResolvedTerminalBasketInputs as ResolvedBasketOptionInputs,
    TerminalBasketSpecLike as BasketOptionSpecLike,
    resolve_terminal_basket_inputs as resolve_basket_option_inputs,
)
from trellis.models.transforms.spread_option import (
    correlated_gbm_log_return_characteristic_function,
    hurd_zhou_spread_option_2d_fft,
)

np = get_numpy()


def price_basket_option_analytical(
    market_state: MarketState,
    spec: BasketOptionSpecLike,
    *,
    comparison_target: str | None = None,
) -> float:
    """Compatibility price through the explicit terminal-basket kernels."""
    resolved = resolve_basket_option_inputs(
        market_state,
        spec,
        comparison_target=comparison_target,
    )
    semantics = resolved.semantics
    if len(semantics.constituent_names) != 2:
        raise ValueError(
            "Analytical terminal-basket pricing supports exactly two underliers"
        )
    if semantics.T <= 0.0:
        intrinsic = terminal_basket_option_payoff(
            np.asarray([semantics.constituent_spots], dtype=float),
            weights=resolved.weights,
            basket_style=resolved.basket_style,
            strike=resolved.strike,
            option_type=resolved.option_type,
        )[0]
        return float(getattr(spec, "notional", 1.0)) * float(intrinsic)

    if resolved.basket_style == "spread":
        unit_price = two_asset_spread_option_kirk(
            forwards=_forwards_from_resolved(semantics),
            strike=resolved.strike,
            T=semantics.T,
            discount_factor=semantics.domestic_df,
            volatilities=resolved.vols,
            correlation=resolved.correlation_matrix[0][1],
            weights=resolved.weights,
            option_type=resolved.option_type,
        )
    elif resolved.basket_style in {"best_of", "worst_of"}:
        unit_price = two_asset_extremum_option_stulz(
            spots=resolved.notional_spots,
            strike=resolved.strike,
            T=semantics.T,
            discount_factor=semantics.domestic_df,
            dividend_yields=resolved.carry,
            volatilities=resolved.vols,
            correlation=resolved.correlation_matrix[0][1],
            basket_style=resolved.basket_style,
            option_type=resolved.option_type,
        )
    else:
        unit_price = two_asset_terminal_basket_gauss_hermite(
            spots=resolved.notional_spots,
            weights=resolved.weights,
            strike=resolved.strike,
            T=semantics.T,
            discount_factor=semantics.domestic_df,
            dividend_yields=resolved.carry,
            volatilities=resolved.vols,
            correlation=resolved.correlation_matrix[0][1],
            basket_style=resolved.basket_style,
            option_type=resolved.option_type,
        )
    return float(getattr(spec, "notional", 1.0)) * float(unit_price)


def price_basket_option_monte_carlo(
    market_state: MarketState,
    spec: BasketOptionSpecLike,
    *,
    comparison_target: str | None = None,
    n_paths: int | None = None,
    seed: int = 42,
) -> float:
    """Compatibility price through process, engine, and payoff primitives."""
    resolved = resolve_basket_option_inputs(
        market_state,
        spec,
        comparison_target=comparison_target,
    )
    semantics = resolved.semantics
    if semantics.T <= 0.0:
        intrinsic = terminal_basket_option_payoff(
            np.asarray([semantics.constituent_spots], dtype=float),
            weights=resolved.weights,
            basket_style=resolved.basket_style,
            strike=resolved.strike,
            option_type=resolved.option_type,
        )[0]
        return float(getattr(spec, "notional", 1.0)) * float(intrinsic)

    paths = int(
        n_paths
        or getattr(spec, "n_paths", None)
        or getattr(spec, "n_simulations", None)
        or 40_000
    )
    domestic_rate = implied_zero_rate(
        float(semantics.domestic_df),
        float(semantics.T),
    )
    process = CorrelatedGBM(
        mu=[domestic_rate for _ in semantics.constituent_names],
        sigma=list(resolved.vols),
        corr=[list(row) for row in resolved.correlation_matrix],
        dividend_yield=list(resolved.carry),
    )
    engine = MonteCarloEngine(
        process,
        n_paths=max(paths, 8_192),
        n_steps=1,
        seed=int(seed),
        method="exact",
    )

    def payoff_fn(simulated_paths):
        return terminal_basket_option_payoff(
            np.asarray(simulated_paths[:, -1, :], dtype=float),
            weights=resolved.weights,
            basket_style=resolved.basket_style,
            strike=resolved.strike,
            option_type=resolved.option_type,
        )

    result = engine.price(
        np.asarray(semantics.constituent_spots, dtype=float),
        float(semantics.T),
        payoff_fn,
        discount_rate=domestic_rate,
        return_paths=False,
    )
    return float(getattr(spec, "notional", 1.0)) * float(result["price"])


def price_basket_option_transform_proxy(
    market_state: MarketState,
    spec: BasketOptionSpecLike,
    *,
    comparison_target: str | None = None,
) -> float:
    """Compatibility price through the genuine Hurd-Zhou Fourier primitives."""
    resolved = resolve_basket_option_inputs(
        market_state,
        spec,
        comparison_target=comparison_target or "fft_spread_2d",
    )
    semantics = resolved.semantics
    if len(semantics.constituent_names) != 2:
        raise ValueError("Hurd-Zhou basket pricing supports exactly two underliers")
    if resolved.basket_style != "spread":
        raise ValueError("Hurd-Zhou basket pricing requires basket_style='spread'")
    if semantics.T <= 0.0:
        intrinsic = terminal_basket_option_payoff(
            np.asarray([semantics.constituent_spots], dtype=float),
            weights=resolved.weights,
            basket_style=resolved.basket_style,
            strike=resolved.strike,
            option_type=resolved.option_type,
        )[0]
        return float(getattr(spec, "notional", 1.0)) * float(intrinsic)

    rate = implied_zero_rate(semantics.domestic_df, semantics.T)

    def characteristic_function(u1, u2):
        return correlated_gbm_log_return_characteristic_function(
            u1,
            u2,
            T=semantics.T,
            rate=rate,
            dividend_yields=resolved.carry,
            volatilities=resolved.vols,
            correlation=resolved.correlation_matrix[0][1],
        )

    unit_price = hurd_zhou_spread_option_2d_fft(
        characteristic_function,
        spots=resolved.notional_spots,
        weights=resolved.weights,
        strike=resolved.strike,
        discount_factor=semantics.domestic_df,
        option_type=resolved.option_type,
    )
    return float(getattr(spec, "notional", 1.0)) * float(unit_price)


def _forwards_from_resolved(
    semantics: ResolvedBasketSemantics,
) -> tuple[float, float]:
    if len(semantics.constituent_spots) != 2:
        raise ValueError("Kirk pricing requires exactly two underliers")
    domestic_rate = implied_zero_rate(
        float(semantics.domestic_df),
        float(semantics.T),
    )
    forwards = tuple(
        float(spot)
        * float(
            np.exp(
                (domestic_rate - float(carry)) * float(semantics.T)
            )
        )
        for spot, carry in zip(
            semantics.constituent_spots,
            semantics.constituent_carry,
            strict=True,
        )
    )
    return float(forwards[0]), float(forwards[1])


__all__ = [
    "BasketOptionSpecLike",
    "ResolvedBasketOptionInputs",
    "price_basket_option_analytical",
    "price_basket_option_monte_carlo",
    "price_basket_option_transform_proxy",
    "resolve_basket_option_inputs",
]
