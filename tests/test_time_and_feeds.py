"""Hour-ending conventions, DST, and parsing of the live ERCOT feed structures.

The JSON fixtures below copy the exact field names, timestamp formats and values of
the ERCOT dashboard feeds as fetched live on 2026-10-21 (first/last records shown;
the middle is filled with the same structure).
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src import sources
from src.config import CT
from src.timeutil import dam_close, hour_ending

HUB_KEYS = ["hbBusAvg", "hbHubAvg", "hbHouston", "hbNorth", "hbPan", "hbSouth", "hbWest",
            "lzAen", "lzCps", "lzHouston", "lzLcra", "lzNorth", "lzRaybn", "lzSouth", "lzWest"]

# Values copied from the live feed (2026-10-21, lastUpdated 09:02 CDT).
LIVE_DAM = {1: 28.86, 2: 29.79, 3: 29.98, 22: 63.14, 23: 48.17, 24: 39.48}
LIVE_RT_FIRST = [("00:15", 28.21), ("00:30", 28.26), ("00:45", 27.1)]


def live_like_prices(day=dt.date(2026, 10, 21)):
    dam = []
    for he in range(1, 25):
        ts = dt.datetime.combine(day, dt.time(0), CT) + dt.timedelta(hours=he)
        v = LIVE_DAM.get(he, 30.0 + he)
        rec = {"hourEnding": he, "dstFlag": "N", **{k: v for k in HUB_KEYS},
               "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
               "interval": int(ts.timestamp() * 1000)}
        dam.append(rec)
    rt = []
    for i in range(1, 37):                      # 00:15 .. 09:00, as in the live snapshot
        ts = dt.datetime.combine(day, dt.time(0), CT) + dt.timedelta(minutes=15 * i)
        v = LIVE_RT_FIRST[i - 1][1] if i <= 3 else 25.0
        rt.append({"intervalEnding": ts.strftime("%H:%M"), "dstFlag": "N",
                   **{k: v for k in HUB_KEYS},
                   "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z"),
                   "interval": int(ts.timestamp() * 1000)})
    return {"lastUpdated": "2026-10-21 09:02:00-0500", "rtSppData": rt, "damSppData": dam}


def test_timestamp_format_matches_live_feed():
    j = live_like_prices()
    assert j["damSppData"][0]["timestamp"] == "2026-10-21 01:00:00-0500"
    assert j["damSppData"][-1]["timestamp"] == "2026-10-22 00:00:00-0500"
    assert j["rtSppData"][0]["timestamp"] == "2026-10-21 00:15:00-0500"


def test_he24_belongs_to_the_previous_day():
    out = sources.parse_dashboard_prices(live_like_prices())
    dam = out["dam"]
    assert list(dam["he"]) == list(range(1, 25))
    assert set(dam["op_date"]) == {dt.date(2026, 10, 21)}           # HE24 not moved to 22nd
    assert dam.loc[dam["he"] == 24, "hbHubAvg"].item() == 39.48
    assert dam.loc[dam["he"] == 1, "hbHubAvg"].item() == 28.86


def test_rt_intervals_map_to_the_right_hour():
    rt = sources.parse_dashboard_prices(live_like_prices())["rt"]
    # intervals ending 00:15, 00:30, 00:45, 01:00 are all HE1
    assert list(rt["he"].iloc[:4]) == [1, 1, 1, 1]
    assert rt["he"].iloc[4] == 2


def test_dashboard_label_mismatch_is_caught():
    j = live_like_prices()
    j["damSppData"][5]["hourEnding"] = 99
    with pytest.raises(RuntimeError, match="hour-ending mismatch"):
        sources.parse_dashboard_prices(j)


def test_dashboard_long_form_has_all_hubs():
    dam = sources.parse_dashboard_prices(live_like_prices())["dam"]
    long = sources.dashboard_dam_long(dam)
    assert set(long["location"]) == {"HB_HUBAVG", "HB_BUSAVG", "HB_HOUSTON", "HB_NORTH",
                                     "HB_PAN", "HB_SOUTH", "HB_WEST"}
    assert len(long) == 7 * 24


def test_fall_back_day_has_25_hours_and_he2_repeats():
    # 2026-11-01: clocks go back at 02:00 CDT
    start = pd.Timestamp("2026-11-01 00:00", tz=CT)
    ts = pd.Series(pd.date_range(start + pd.Timedelta(hours=1), periods=25, freq="h"))
    he = hour_ending(ts)
    assert set(he["op_date"]) == {dt.date(2026, 11, 1)}
    assert list(he["he"]).count(2) == 2
    assert he["he"].nunique() == 24


def test_spring_forward_day_has_23_hours():
    start = pd.Timestamp("2026-03-08 00:00", tz=CT)
    ts = pd.Series(pd.date_range(start + pd.Timedelta(hours=1), periods=23, freq="h"))
    he = hour_ending(ts)
    assert set(he["op_date"]) == {dt.date(2026, 3, 8)}
    assert he["he"].nunique() == 23


def test_naive_timestamps_are_refused():
    with pytest.raises(ValueError):
        hour_ending(pd.Series(pd.date_range("2026-10-21", periods=3, freq="h")))


def test_dam_close_is_10am_ct_day_before():
    c = dam_close(dt.date(2026, 10, 22))
    assert c.isoformat() == "2026-10-21T10:00:00-05:00"
    assert dam_close(dt.date(2026, 12, 2)).isoformat() == "2026-12-01T10:00:00-06:00"


def test_fuel_mix_new_storage_fields_are_not_double_counted():
    # Structure of the live feed on 2026-10-21, including the new storage sub-fields.
    rec = {"Coal and Lignite": {"gen": 6071.02001953125}, "Hydro": {"gen": 5.59499979019165},
           "Nuclear": {"gen": 3834.794921875}, "Other": {"gen": 50.7000007629395},
           "Power Storage": {"gen": -2368.08276367187}, "Solar": {"gen": 0.0440958701074123},
           "Wind": {"gen": 17903.14453125}, "Natural Gas": {"gen": 20961.73222923278794},
           "Power Storage Charging": {"gen": -2368.08276367187},
           "Power Storage Discharging": {"gen": 0}}
    j = {"lastUpdated": "2026-10-21 09:16:00-0500",
         "monthlyCapacity": {"Power Storage": 20948, "Solar": 40226, "Wind": 40405,
                             "Hydro": 579, "Other": 0, "Natural Gas": 69569,
                             "Coal and Lignite": 13705, "Nuclear": 5268},
         "data": {"2026-10-20": {"2026-10-20 00:00:00-0500": rec}}}
    out = sources.parse_fuel_mix(j)
    g = out["gen"].iloc[0]
    expected = (6071.02001953125 + 5.59499979019165 + 3834.794921875 + 50.7000007629395
                - 2368.08276367187 + 0.0440958701074123 + 17903.14453125 + 20961.73222923278794)
    assert g["load"] == pytest.approx(expected)
    assert out["capacity"]["storage"] == 20948 and out["capacity"]["gas"] == 69569
    assert "power storage charging" not in out["gen"].columns


def test_weather_builder_uses_hour_ending_and_utc():
    ts = pd.Series(pd.date_range("2026-10-21 05:00", periods=48, freq="h", tz="UTC")
                   .tz_convert(CT))
    def mk(v):
        return pd.DataFrame({"ts": ts, "v": v, "site": "a", "w": 1.0})
    wx = sources.build_weather(mk(80.0), mk(8.0), mk(np.linspace(0, 900, 48)))
    assert {"op_date", "he", "cdd", "hdd", "ws_a", "cdd_24h"} <= set(wx.columns)
    assert wx["cdd"].iloc[0] == pytest.approx(15.0)
    first = wx.iloc[0]
    assert first["op_date"] == dt.date(2026, 10, 20) and first["he"] == 24   # 00:00 CDT = HE24


def test_secrets_never_reach_feed_logs(monkeypatch):
    monkeypatch.setattr(sources, "EIA_API_KEY", "SECRETKEY123")
    sources.log("x", "FAIL", "GET failed https://api.eia.gov/v2/x?api_key=SECRETKEY123&length=5: 500")
    assert "SECRETKEY123" not in sources.FEED["x"]["detail"]
    assert "api_key=***" in sources.FEED["x"]["detail"]


def test_spring_forward_dashboard_label_does_not_kill_the_feed():
    day = dt.date(2027, 3, 14)
    start = dt.datetime.combine(day, dt.time(0), CT)
    dam, labels = [], [1] + list(range(3, 25))       # ERCOT skips HE02
    for i, lab in enumerate(labels):
        ts = (start.astimezone(dt.timezone.utc) + dt.timedelta(hours=i + 1)).astimezone(CT)
        dam.append({"hourEnding": lab, "dstFlag": "N", "hbHubAvg": 30.0,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z")})
    out = sources.parse_dashboard_prices({"rtSppData": [], "damSppData": dam})
    assert len(out["dam"]) == 23
