"""Curve resolution: normalize date + data source → YieldCurve."""

from __future__ import annotations

from datetime import date

from trellis.curves.yield_curve import YieldCurve


def resolve_curve(
    as_of: date | str | None = None,
    source: str = "treasury_gov",
) -> YieldCurve:
    """Fetch market data and build a YieldCurve.

    Parameters
    ----------
    as_of : date, str, or None
        ``"latest"`` / ``None`` → today; string → parsed as ISO date.
    source : str
        ``"treasury_gov"`` or ``"fred"``.
    """
    # Normalize date
    if as_of is None or as_of == "latest":
        resolved_date = date.today()
    elif isinstance(as_of, str):
        resolved_date = date.fromisoformat(as_of)
    else:
        resolved_date = as_of

    # Instantiate provider
    if source == "mock":
        from trellis.data.mock import MockDataProvider
        provider = MockDataProvider()
    elif source == "fred":
        try:
            from trellis.data.fred import FredDataProvider
            provider = FredDataProvider()
        except ImportError:
            import warnings
            warnings.warn(
                "fredapi not installed; falling back to mock data",
                stacklevel=2,
            )
            from trellis.data.mock import MockDataProvider
            provider = MockDataProvider()
    elif source == "treasury_gov":
        try:
            from trellis.data.treasury_gov import TreasuryGovDataProvider
            provider = TreasuryGovDataProvider()
        except ImportError:
            import warnings
            warnings.warn(
                "requests not installed; falling back to mock data",
                stacklevel=2,
            )
            from trellis.data.mock import MockDataProvider
            provider = MockDataProvider()
    else:
        raise ValueError(f"Unknown data source: {source!r}")

    yields = provider.fetch_yields(resolved_date)
    if not yields:
        raise RuntimeError(
            f"No yield data returned from {source!r} for as_of={resolved_date}"
        )

    return YieldCurve.from_treasury_yields(yields)
