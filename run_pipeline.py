from get_data import get_daily_form4_csv
from datetime import datetime, date
from dateutil.tz import gettz
from dateutil.relativedelta import relativedelta
import holidays
from zoneinfo import ZoneInfo

# cache the holiday calendar
US_HOLIDAYS = holidays.US(observed=True)

def is_sec_nonpublishing_day(dt_or_date):
    '''
    True if in US the given day is a weekend or US federal holiday
    Accepts a date or datetime
    '''
    if isinstance(dt_or_date, datetime):
        et = ZoneInfo("America/New_York")
        if dt_or_date.tzinfo is None:
            d = dt_or_date.replace(tzinfo=et).date()
        else:
            d = dt_or_date.astimezone(et).date()
    else:
        d = dt_or_date  # already a calendar date

    return d.weekday() >= 5 or d in US_HOLIDAYS

# get todays date and quarter and format them correctly
def today_date(tz_name: str = "America/New_York"):
    tz = gettz(tz_name)
    now = datetime.now(tz)
    if is_sec_nonpublishing_day(now):
        raise Exception("Today is not a SEC publishing day, either because it's a weekend or a federal holiday")
    date_str = now.strftime("%Y%m%d")
    quarter = f"QTR{(now.month - 1)//3 + 1}"
    return date_str, quarter

# get yesterdays date and quarter and format them correctly
def yesterday_date(tz_name: str = "America/New_York"):
    tz = gettz(tz_name)
    now = datetime.now(tz) - relativedelta(days = 1)
    if is_sec_nonpublishing_day(now):
        raise Exception("Yesterday was not a SEC publishing day, either because it's a weekend or a federal holiday")
    date_str = now.strftime("%Y%m%d")
    quarter = f"QTR{(now.month - 1)//3 + 1}"
    return date_str, quarter

# get specific date, checks if the date is valid if not raises error
def specific_date(day: int, month: int, year: int):
    if year < 1995 or year > datetime.now().year: # 1995 is the first year SEC has the full year data
        raise Exception("Invalid year, SEC has full data starting from 1995")
    if month < 1 or month > 12:
        raise Exception("Invalid month")
    try:
        date(year, month, day) 
    except ValueError:
        raise ValueError(f"Invalid date, this date does not exist for this specific month and year: {year}-{month:02d}-{day:02d}") from None
    date_str = f"{year:04d}{month:02d}{day:02d}"
    quarter = f"QTR{(month - 1)//3 + 1}"
    return date_str, quarter

date_str, quarter = specific_date(20, 8, 2025)
get_daily_form4_csv(date_str, quarter)