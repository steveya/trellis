"""Bloomberg data provider (optional — requires blpapi)."""

from __future__ import annotations

from datetime import date

from trellis.data.base import BaseDataProvider


class BloombergDataProvider(BaseDataProvider):
    """Fetch Treasury yields via the Bloomberg API.

    Requires ``blpapi`` to be installed and a Bloomberg terminal session.
    """

    def __init__(self):
        """Validate that the optional Bloomberg SDK dependency is installed."""
        try:
            import blpapi  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "blpapi is required for BloombergDataProvider. "
                "Install with: pip install blpapi"
            )

    def fetch_yields(self, as_of: date | None = None) -> dict[float, float]:
        """Placeholder for Bloomberg yield retrieval; currently not implemented."""
        raise NotImplementedError(
            "Bloomberg integration is a placeholder — "
            "implement with your terminal's blpapi session."
        )
