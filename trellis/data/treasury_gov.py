"""Treasury.gov fallback data provider (no API key required)."""

from __future__ import annotations

from datetime import date, timedelta

from trellis.data.base import BaseDataProvider
from trellis.data.cache import DiskCache

# fiscaldata.treasury.gov field → tenor in years
FIELD_TENOR: dict[str, float] = {
    "t_bill_4_wk_rate": 1 / 12,
    "t_bill_13_wk_rate": 0.25,
    "t_bill_26_wk_rate": 0.5,
    "t_note_1_yr_rate": 1.0,
    "t_note_2_yr_rate": 2.0,
    "t_note_3_yr_rate": 3.0,
    "t_note_5_yr_rate": 5.0,
    "t_note_7_yr_rate": 7.0,
    "t_note_10_yr_rate": 10.0,
    "t_bond_20_yr_rate": 20.0,
    "t_bond_30_yr_rate": 30.0,
}

API_URL = (
    "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
    "v2/accounting/od/avg_interest_rates"
)


class TreasuryGovDataProvider(BaseDataProvider):
    """Fetch yields from fiscaldata.treasury.gov (no API key needed)."""

    def __init__(self):
        """Initialize the on-disk response cache for Treasury.gov data."""
        self._cache = DiskCache()

    def fetch_yields(self, as_of: date | None = None) -> dict[float, float]:
        """Fetch or cache Treasury average-interest-rate data from fiscaldata.gov."""
        as_of = as_of or date.today()
        cached = self._cache.get("treasury_gov_yields", as_of)
        if cached is not None:
            return cached

        import requests

        params = {
            "filter": f"record_date:lte:{as_of.isoformat()}",
            "sort": "-record_date",
            "page[size]": "1",
            "fields": ",".join(["record_date"] + list(FIELD_TENOR.keys())),
        }
        resp = requests.get(API_URL, params=params, timeout=15)
        resp.raise_for_status()
        records = resp.json().get("data", [])
        if not records:
            return {}

        row = records[0]
        yields: dict[float, float] = {}
        for field_name, tenor in FIELD_TENOR.items():
            val = row.get(field_name)
            if val is not None and val != "null":
                try:
                    yields[tenor] = float(val) / 100.0
                except (ValueError, TypeError):
                    continue

        if yields:
            self._cache.put("treasury_gov_yields", as_of, yields)
        return yields
