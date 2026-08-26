"""Product-neutral homogeneous pool-loss and bounded layer algebra."""

from __future__ import annotations

from operator import index

from trellis.core.differentiable import get_numpy


np = get_numpy()


def homogeneous_pool_loss_fraction(
    default_counts,
    *,
    pool_size: int,
    recovery: float,
):
    """Map homogeneous default counts to total-pool loss fractions.

    ``default_counts`` may be a scalar or any non-empty array shape. Fractional
    counts are accepted so callers can project expected event counts as well as
    discrete analytical or sampled counts.
    """
    count = _positive_integer(pool_size, name="pool_size")
    recovery_fraction = _bounded_scalar(
        recovery,
        name="recovery",
        upper_inclusive=False,
    )
    defaults = _fraction_array(
        default_counts,
        name="default_counts",
        upper=float(count),
    )
    return defaults * (1.0 - recovery_fraction) / float(count)


def bounded_layer_loss_fraction(
    portfolio_loss_fraction,
    *,
    attachment: float,
    detachment: float,
):
    """Project total-pool loss fractions onto one bounded loss layer.

    The result is measured as a fraction of total pool notional, capped at the
    layer width ``detachment - attachment`` rather than normalized by that
    width.
    """
    try:
        attachment_fraction = _bounded_scalar(
            attachment,
            name="attachment",
            upper_inclusive=True,
        )
        detachment_fraction = _bounded_scalar(
            detachment,
            name="detachment",
            upper_inclusive=True,
        )
    except ValueError:
        raise ValueError(
            "layer bounds must satisfy 0 <= attachment < detachment <= 1"
        ) from None
    if attachment_fraction >= detachment_fraction:
        raise ValueError(
            "layer bounds must satisfy 0 <= attachment < detachment <= 1"
        )
    losses = _fraction_array(
        portfolio_loss_fraction,
        name="portfolio_loss_fraction",
        upper=1.0,
    )
    return np.clip(
        losses - attachment_fraction,
        0.0,
        detachment_fraction - attachment_fraction,
    )


def _positive_integer(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized = index(value)
    except TypeError:
        raise ValueError(f"{name} must be a positive integer") from None
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _bounded_scalar(value, *, name: str, upper_inclusive: bool) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = float("nan")
    valid_upper = normalized <= 1.0 if upper_inclusive else normalized < 1.0
    if not bool(np.isfinite(normalized)) or normalized < 0.0 or not valid_upper:
        relation = "<=" if upper_inclusive else "<"
        raise ValueError(f"{name} must satisfy 0 <= {name} {relation} 1")
    return normalized


def _fraction_array(value, *, name: str, upper: float):
    try:
        normalized = value if hasattr(value, "_value") else np.asarray(value)
        validation_view = np.asarray(getattr(normalized, "_value", normalized))
        if int(np.size(validation_view)) == 0:
            raise ValueError
        if not bool(np.all(np.isfinite(validation_view))):
            raise ValueError
        if bool(np.any(validation_view < 0.0)) or bool(
            np.any(validation_view > upper)
        ):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be a non-empty finite value in [0, {upper:g}]"
        ) from None
    return normalized


__all__ = [
    "bounded_layer_loss_fraction",
    "homogeneous_pool_loss_fraction",
]
