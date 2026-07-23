"""Engine-neutral payoff algebra for model composition."""

from __future__ import annotations

from trellis.core.differentiable import get_numpy

np = get_numpy()


def terminal_basket_option_payoff(
    terminal_values,
    *,
    weights: tuple[float, ...],
    basket_style: str,
    strike: float,
    option_type: str,
):
    """Return an undiscounted terminal basket call or put payoff.

    This primitive owns payoff algebra only. Market resolution, state
    propagation, discounting, and notional scaling remain explicit caller
    responsibilities, so analytical, transform, and simulation routes can
    share the same contract without inheriting one another's engine family.
    """
    terminal = np.asarray(terminal_values, dtype=float)
    normalized_weights = np.asarray(
        tuple(float(value) for value in weights),
        dtype=float,
    )
    if terminal.ndim < 1 or terminal.shape[-1] != normalized_weights.shape[0]:
        raise ValueError(
            "terminal_values last dimension must match the number of basket weights"
        )

    style = _normalized_basket_style(basket_style)
    if style == "best_of":
        basket_value = np.max(terminal, axis=-1)
    elif style == "worst_of":
        basket_value = np.min(terminal, axis=-1)
    else:
        basket_value = np.dot(terminal, normalized_weights)

    normalized_option_type = str(option_type or "").strip().lower()
    if normalized_option_type == "call":
        return np.maximum(basket_value - float(strike), 0.0)
    if normalized_option_type == "put":
        return np.maximum(float(strike) - basket_value, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")


def _normalized_basket_style(value: object) -> str:
    style = str(value or "weighted_sum").strip().lower()
    aliases = {
        "best": "best_of",
        "bestof": "best_of",
        "best_of_two": "best_of",
        "worst": "worst_of",
        "worstof": "worst_of",
    }
    normalized = aliases.get(style, style)
    if normalized not in {"weighted_sum", "spread", "best_of", "worst_of"}:
        raise ValueError(
            "basket_style must be weighted_sum, spread, best_of, or worst_of"
        )
    return normalized


__all__ = ["terminal_basket_option_payoff"]
