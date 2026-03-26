"""Market-data resolution helpers."""

from __future__ import annotations

from datetime import date

from trellis.curves.yield_curve import YieldCurve
from trellis.data.schema import MarketSnapshot


def _normalize_as_of(as_of: date | str | None = None) -> date:
    """Normalize market-data date inputs."""
    if as_of is None or as_of == "latest":
        return date.today()
    if isinstance(as_of, str):
        return date.fromisoformat(as_of)
    return as_of


def _provider_for_source(source: str):
    """Instantiate the provider for a supported market-data source."""
    if source == "mock":
        from trellis.data.mock import MockDataProvider

        return MockDataProvider()
    if source == "fred":
        try:
            from trellis.data.fred import FredDataProvider

            return FredDataProvider()
        except ImportError:
            import warnings

            warnings.warn(
                "fredapi not installed; falling back to mock data",
                stacklevel=2,
            )
            from trellis.data.mock import MockDataProvider

            return MockDataProvider()
    if source == "treasury_gov":
        try:
            from trellis.data.treasury_gov import TreasuryGovDataProvider

            return TreasuryGovDataProvider()
        except ImportError:
            import warnings

            warnings.warn(
                "requests not installed; falling back to mock data",
                stacklevel=2,
            )
            from trellis.data.mock import MockDataProvider

            return MockDataProvider()
    raise ValueError(f"Unknown data source: {source!r}")


def resolve_market_snapshot(
    as_of: date | str | None = None,
    source: str = "treasury_gov",
    *,
    vol_surface=None,
    vol_surfaces: dict | None = None,
    default_vol_surface: str | None = None,
    forecast_curves: dict | None = None,
    credit_curve=None,
    fx_rates: dict | None = None,
    state_space=None,
    state_spaces: dict | None = None,
    default_state_space: str | None = None,
    underlier_spots: dict | None = None,
    default_underlier_spot: str | None = None,
    local_vol_surface=None,
    local_vol_surfaces: dict | None = None,
    default_local_vol_surface: str | None = None,
    jump_parameters: dict | None = None,
    jump_parameter_sets: dict | None = None,
    default_jump_parameters: str | None = None,
    model_parameters: dict | None = None,
    model_parameter_sets: dict | None = None,
    default_model_parameters: str | None = None,
    metadata: dict | None = None,
) -> MarketSnapshot:
    """Resolve a canonical market snapshot.

    Parameters
    ----------
    as_of : date, str, or None
        ``"latest"`` / ``None`` → today; string → parsed as ISO date.
    source : str
        ``"treasury_gov"``, ``"fred"``, or ``"mock"``.
    """
    resolved_date = _normalize_as_of(as_of)
    provider = _provider_for_source(source)

    if vol_surface is not None and vol_surfaces is not None:
        raise ValueError("Pass either vol_surface= or vol_surfaces=, not both")
    if state_space is not None and state_spaces is not None:
        raise ValueError("Pass either state_space= or state_spaces=, not both")
    if local_vol_surface is not None and local_vol_surfaces is not None:
        raise ValueError("Pass either local_vol_surface= or local_vol_surfaces=, not both")
    if jump_parameters is not None and jump_parameter_sets is not None:
        raise ValueError("Pass either jump_parameters= or jump_parameter_sets=, not both")
    if model_parameters is not None and model_parameter_sets is not None:
        raise ValueError("Pass either model_parameters= or model_parameter_sets=, not both")

    try:
        base_snapshot = provider.fetch_market_snapshot(resolved_date)
    except NotImplementedError:
        base_snapshot = None
    if not isinstance(base_snapshot, MarketSnapshot):
        base_snapshot = None

    if base_snapshot is None:
        yields = provider.fetch_yields(resolved_date)
        if not yields:
            raise RuntimeError(
                f"No yield data returned from {source!r} for as_of={resolved_date}"
            )
        discount_curve = YieldCurve.from_treasury_yields(yields)
        base_snapshot = MarketSnapshot(
            as_of=resolved_date,
            source=source,
            discount_curves={"discount": discount_curve},
            default_discount_curve="discount",
        )

    if not base_snapshot.discount_curves:
        raise RuntimeError(
            f"No yield data returned from {source!r} for as_of={resolved_date}"
        )

    resolved_forecast_curves = dict(base_snapshot.forecast_curves)
    resolved_fx_rates = dict(base_snapshot.fx_rates)
    resolved_state_spaces = dict(base_snapshot.state_spaces)
    resolved_metadata = dict(base_snapshot.metadata)
    resolved_vol_surfaces = dict(base_snapshot.vol_surfaces)
    resolved_credit_curves = dict(base_snapshot.credit_curves)
    resolved_underlier_spots = dict(base_snapshot.underlier_spots)
    resolved_local_vol_surfaces = dict(base_snapshot.local_vol_surfaces)
    resolved_jump_parameter_sets = {
        key: dict(value) for key, value in base_snapshot.jump_parameter_sets.items()
    }
    resolved_model_parameter_sets = {
        key: dict(value) for key, value in base_snapshot.model_parameter_sets.items()
    }
    resolved_default_vol_surface = base_snapshot.default_vol_surface
    resolved_default_credit_curve = base_snapshot.default_credit_curve
    resolved_default_state_space = base_snapshot.default_state_space
    resolved_default_underlier_spot = base_snapshot.default_underlier_spot
    resolved_default_local_vol_surface = base_snapshot.default_local_vol_surface
    resolved_default_jump_parameters = base_snapshot.default_jump_parameters
    resolved_default_model_parameters = base_snapshot.default_model_parameters

    if forecast_curves:
        resolved_forecast_curves.update(forecast_curves)
    if fx_rates:
        resolved_fx_rates.update(fx_rates)
    if state_spaces is not None:
        resolved_state_spaces.update(state_spaces)
        resolved_default_state_space = default_state_space or resolved_default_state_space
        if resolved_default_state_space is None and len(resolved_state_spaces) == 1:
            resolved_default_state_space = next(iter(resolved_state_spaces))
    elif state_space is not None:
        resolved_default_state_space = (
            default_state_space or resolved_default_state_space or "default"
        )
        resolved_state_spaces[resolved_default_state_space] = state_space
    if underlier_spots:
        resolved_underlier_spots.update(underlier_spots)
    if metadata:
        resolved_metadata.update(metadata)

    if vol_surfaces is not None:
        resolved_vol_surfaces.update(vol_surfaces)
        resolved_default_vol_surface = default_vol_surface or resolved_default_vol_surface
        if resolved_default_vol_surface is None and len(resolved_vol_surfaces) == 1:
            resolved_default_vol_surface = next(iter(resolved_vol_surfaces))
    elif vol_surface is not None:
        resolved_default_vol_surface = resolved_default_vol_surface or "default"
        resolved_vol_surfaces[resolved_default_vol_surface] = vol_surface

    if local_vol_surfaces is not None:
        resolved_local_vol_surfaces.update(local_vol_surfaces)
        resolved_default_local_vol_surface = (
            default_local_vol_surface or resolved_default_local_vol_surface
        )
        if (
            resolved_default_local_vol_surface is None
            and len(resolved_local_vol_surfaces) == 1
        ):
            resolved_default_local_vol_surface = next(iter(resolved_local_vol_surfaces))
    elif local_vol_surface is not None:
        resolved_default_local_vol_surface = (
            default_local_vol_surface or resolved_default_local_vol_surface or "default"
        )
        resolved_local_vol_surfaces[resolved_default_local_vol_surface] = local_vol_surface

    if credit_curve is not None:
        resolved_default_credit_curve = resolved_default_credit_curve or "default"
        resolved_credit_curves[resolved_default_credit_curve] = credit_curve
    elif resolved_default_credit_curve is None and len(resolved_credit_curves) == 1:
        resolved_default_credit_curve = next(iter(resolved_credit_curves))

    if resolved_default_underlier_spot is None and len(resolved_underlier_spots) == 1:
        resolved_default_underlier_spot = next(iter(resolved_underlier_spots))
    if default_underlier_spot is not None:
        resolved_default_underlier_spot = default_underlier_spot

    if jump_parameter_sets is not None:
        resolved_jump_parameter_sets.update({
            key: dict(value) for key, value in jump_parameter_sets.items()
        })
        resolved_default_jump_parameters = (
            default_jump_parameters or resolved_default_jump_parameters
        )
        if resolved_default_jump_parameters is None and len(resolved_jump_parameter_sets) == 1:
            resolved_default_jump_parameters = next(iter(resolved_jump_parameter_sets))
    elif jump_parameters is not None:
        resolved_default_jump_parameters = (
            default_jump_parameters or resolved_default_jump_parameters or "default"
        )
        resolved_jump_parameter_sets[resolved_default_jump_parameters] = dict(jump_parameters)

    if model_parameter_sets is not None:
        resolved_model_parameter_sets.update({
            key: dict(value) for key, value in model_parameter_sets.items()
        })
        resolved_default_model_parameters = (
            default_model_parameters or resolved_default_model_parameters
        )
        if resolved_default_model_parameters is None and len(resolved_model_parameter_sets) == 1:
            resolved_default_model_parameters = next(iter(resolved_model_parameter_sets))
    elif model_parameters is not None:
        resolved_default_model_parameters = (
            default_model_parameters or resolved_default_model_parameters or "default"
        )
        resolved_model_parameter_sets[resolved_default_model_parameters] = dict(model_parameters)

    return MarketSnapshot(
        as_of=base_snapshot.as_of,
        source=base_snapshot.source,
        discount_curves=dict(base_snapshot.discount_curves),
        forecast_curves=resolved_forecast_curves,
        vol_surfaces=resolved_vol_surfaces,
        credit_curves=resolved_credit_curves,
        fx_rates=resolved_fx_rates,
        state_spaces=resolved_state_spaces,
        underlier_spots=resolved_underlier_spots,
        local_vol_surfaces=resolved_local_vol_surfaces,
        jump_parameter_sets=resolved_jump_parameter_sets,
        model_parameter_sets=resolved_model_parameter_sets,
        metadata=resolved_metadata,
        default_discount_curve=base_snapshot.default_discount_curve,
        default_vol_surface=resolved_default_vol_surface,
        default_credit_curve=resolved_default_credit_curve,
        default_state_space=resolved_default_state_space,
        default_underlier_spot=resolved_default_underlier_spot,
        default_local_vol_surface=resolved_default_local_vol_surface,
        default_jump_parameters=resolved_default_jump_parameters,
        default_model_parameters=resolved_default_model_parameters,
    )


def resolve_curve(
    as_of: date | str | None = None,
    source: str = "treasury_gov",
) -> YieldCurve:
    """Fetch market data and build the default discount curve."""
    return resolve_market_snapshot(as_of=as_of, source=source).discount_curve()
