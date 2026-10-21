"""One place that turns timestamps into ERCOT (operating day, hour ending).

ERCOT, EIA-930 and Open-Meteo's period quantities all stamp an hour at its END. HE24
of 21 Oct is stamped 2026-10-22 00:00. Subtracting one second before reading the
local date and hour maps every such stamp to the right operating day and HE 1..24.

Doing the subtraction on tz-aware timestamps (absolute time) makes the DST days work
without special cases:
  * fall-back day: two distinct hours both map to HE2, so the day has 25 rows and
    any (day, HE) groupby averages the repeated hour;
  * spring-forward day: one HE is absent and the day has 23 rows.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from .config import CT, DAM_CLOSE_HOUR_CT


def to_ct(ts: pd.Series) -> pd.Series:
    """Coerce to tz-aware America/Chicago. Naive input is rejected, not guessed."""
    s = pd.to_datetime(ts)
    if getattr(s.dt, "tz", None) is None:
        raise ValueError("naive timestamps are ambiguous around DST; localise first")
    return s.dt.tz_convert(CT)


def hour_ending(ts: pd.Series) -> pd.DataFrame:
    """Return columns op_date (datetime.date) and he (int 1..24) for hour-ending stamps."""
    s = to_ct(ts) - pd.Timedelta(seconds=1)
    return pd.DataFrame({"op_date": s.dt.date.to_numpy(),
                         "he": (s.dt.hour + 1).to_numpy()}, index=ts.index)


def dam_close(target: dt.date) -> dt.datetime:
    """The moment the day-ahead market for `target` closes (10:00 CT the day before)."""
    d = target - dt.timedelta(days=1)
    return dt.datetime.combine(d, dt.time(DAM_CLOSE_HOUR_CT), CT)


def today_ct() -> dt.date:
    return dt.datetime.now(CT).date()
