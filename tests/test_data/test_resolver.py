"""Tests for trellis.data.resolver."""

from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from trellis.data.resolver import resolve_curve


SAMPLE_YIELDS = {
    0.25: 0.045,
    0.5: 0.046,
    1.0: 0.047,
    2.0: 0.048,
    5.0: 0.045,
    10.0: 0.044,
    30.0: 0.046,
}


class TestResolveCurve:
    """resolve_curve with mocked providers."""

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_treasury_gov_returns_curve(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        curve = resolve_curve(as_of=date(2024, 11, 15), source="treasury_gov")
        assert len(curve.tenors) == len(SAMPLE_YIELDS)
        mock.fetch_yields.assert_called_once_with(date(2024, 11, 15))

    @patch("trellis.data.fred.FredDataProvider")
    def test_fred_returns_curve(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        curve = resolve_curve(as_of=date(2024, 11, 15), source="fred")
        assert len(curve.tenors) == len(SAMPLE_YIELDS)

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_latest_resolves_to_today(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        resolve_curve(as_of="latest")
        mock.fetch_yields.assert_called_once_with(date.today())

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_empty_yields_raises(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = {}
        with pytest.raises(RuntimeError, match="No yield data"):
            resolve_curve(as_of=date(2024, 11, 15))

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="Unknown data source"):
            resolve_curve(source="bloomberg")

    @patch("trellis.data.treasury_gov.TreasuryGovDataProvider")
    def test_string_date_parsed(self, MockProvider):
        mock = MockProvider.return_value
        mock.fetch_yields.return_value = SAMPLE_YIELDS
        resolve_curve(as_of="2024-11-15")
        mock.fetch_yields.assert_called_once_with(date(2024, 11, 15))
