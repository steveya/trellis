"""Testing Schedule (date_utils) functions"""
import pytest
from trellis.core.types import Frequency
from trellis.core.date_utils import get_bracketing_dates, get_accrual_fraction
from datetime import datetime

def test_get_bracketing_dates():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    frequency = Frequency.QUARTERLY
    date = datetime(2020, 3, 1)
    assert get_bracketing_dates(start_date, end_date, frequency, date) == (datetime(2020, 1, 1).date(), datetime(2020, 4, 1).date())
    date = datetime(2020, 6, 1)
    assert get_bracketing_dates(start_date, end_date, frequency, date) == (datetime(2020, 4, 1).date(), datetime(2020, 7, 1).date())
    date = datetime(2020, 9, 1)
    assert get_bracketing_dates(start_date, end_date, frequency, date) == (datetime(2020, 7, 1).date(), datetime(2020, 10, 1).date())
    date = datetime(2020, 12, 31)
    assert get_bracketing_dates(start_date, end_date, frequency, date) == (datetime(2020, 10, 1).date(), datetime(2021, 1, 1).date())

def test_get_accrual_fraction():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    frequency = Frequency.QUARTERLY
    date = datetime(2020, 3, 1)
    # 60 days into a 91-day quarter (Jan 1 to Apr 1)
    assert get_accrual_fraction(start_date, end_date, frequency, date) == pytest.approx(60 / 91)
    date = datetime(2020, 6, 1)
    # 61 days into a 91-day quarter (Apr 1 to Jul 1)
    assert get_accrual_fraction(start_date, end_date, frequency, date) == pytest.approx(61 / 91)
    date = datetime(2020, 9, 1)
    # 62 days into a 92-day quarter (Jul 1 to Oct 1)
    assert get_accrual_fraction(start_date, end_date, frequency, date) == pytest.approx(62 / 92)
    date = datetime(2020, 12, 31)
    # 91 days into a 92-day quarter (Oct 1 to Jan 1)
    assert get_accrual_fraction(start_date, end_date, frequency, date) == pytest.approx(91 / 92)

def test_get_bracketing_dates_exception():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    frequency = Frequency.QUARTERLY
    date = datetime(2021, 3, 1)
    with pytest.raises(ValueError):
        get_bracketing_dates(start_date, end_date, frequency, date)

def test_get_accrual_fraction_exception():
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2020, 12, 31)
    frequency = Frequency.QUARTERLY
    date = datetime(2021, 3, 1)
    with pytest.raises(ValueError):
        get_accrual_fraction(start_date, end_date, frequency, date)
