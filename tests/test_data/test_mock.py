"""Tests for trellis.data.mock — MockDataProvider."""

from datetime import date

import pytest

from trellis.data.mock import MockDataProvider, SNAPSHOTS, _TENOR_GRID
from trellis.data.resolver import resolve_curve


class TestMockDataProvider:

    def test_fetch_yields_returns_dict(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields()
        assert isinstance(yields, dict)
        assert len(yields) == len(_TENOR_GRID)

    def test_all_yields_reasonable(self):
        provider = MockDataProvider()
        for d in provider.available_dates:
            yields = provider.fetch_yields(d)
            for tenor, rate in yields.items():
                assert 0 <= rate <= 0.10, f"rate {rate} at tenor {tenor} on {d}"

    def test_exact_date_match(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields(date(2024, 11, 15))
        assert yields == SNAPSHOTS[date(2024, 11, 15)]

    def test_nearest_prior_date(self):
        """A date between snapshots returns the most recent prior."""
        provider = MockDataProvider()
        # 2024-06-01 is between 2023-10-15 and 2024-11-15
        yields = provider.fetch_yields(date(2024, 6, 1))
        assert yields == SNAPSHOTS[date(2023, 10, 15)]

    def test_future_date_returns_latest(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields(date(2030, 1, 1))
        assert yields == SNAPSHOTS[date(2024, 11, 15)]

    def test_before_all_snapshots_returns_empty(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields(date(2000, 1, 1))
        assert yields == {}

    def test_none_returns_latest(self):
        provider = MockDataProvider()
        yields = provider.fetch_yields(None)
        assert yields == SNAPSHOTS[date(2024, 11, 15)]

    def test_custom_overrides(self):
        custom = {date(2025, 1, 1): {1.0: 0.05, 10.0: 0.06}}
        provider = MockDataProvider(overrides=custom)
        yields = provider.fetch_yields(date(2025, 1, 1))
        assert yields == {1.0: 0.05, 10.0: 0.06}
        # Built-in snapshots still accessible
        yields_old = provider.fetch_yields(date(2019, 9, 15))
        assert yields_old == SNAPSHOTS[date(2019, 9, 15)]

    def test_from_dict_no_builtins(self):
        custom = {date(2025, 6, 1): {5.0: 0.04}}
        provider = MockDataProvider.from_dict(custom)
        # Only the custom data exists
        assert provider.available_dates == [date(2025, 6, 1)]
        assert provider.fetch_yields(date(2025, 6, 1)) == {5.0: 0.04}
        # Date before the only snapshot returns empty
        assert provider.fetch_yields(date(2024, 11, 15)) == {}
        # Date after returns the snapshot
        assert provider.fetch_yields(date(2026, 1, 1)) == {5.0: 0.04}

    def test_returns_copy(self):
        """Mutations to returned dict don't affect internal state."""
        provider = MockDataProvider()
        y1 = provider.fetch_yields(date(2024, 11, 15))
        y1[999.0] = 0.99
        y2 = provider.fetch_yields(date(2024, 11, 15))
        assert 999.0 not in y2

    def test_available_dates(self):
        provider = MockDataProvider()
        dates = provider.available_dates
        assert len(dates) == len(SNAPSHOTS)
        assert dates == sorted(dates)


class TestResolverMockSource:
    """Integration: resolve_curve(source='mock') works without mocking."""

    def test_mock_source_returns_curve(self):
        curve = resolve_curve(as_of=date(2024, 11, 15), source="mock")
        assert len(curve.tenors) == len(_TENOR_GRID)
        # Rates should be continuously compounded (converted from BEY)
        assert all(r > 0 for r in curve.rates)

    def test_mock_source_latest(self):
        curve = resolve_curve(as_of="latest", source="mock")
        assert len(curve.tenors) == len(_TENOR_GRID)
