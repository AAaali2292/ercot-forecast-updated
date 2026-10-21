"""End-to-end with every network call replaced by a deterministic simulated world.

Proves the CLI commands the scheduler runs (daily, dam, capture, publish, backtest)
work together and produce a coherent forecast, scorecard and site data file.
"""
import datetime as dt
import json
import math

import numpy as np
import pandas as pd
import pytest

from src import config, history, sources
from src import run as cli
from src.config import CT

HUB_KEYS = ["hbBusAvg", "hbHubAvg", "hbHouston", "hbNorth", "hbPan", "hbSouth", "hbWest"]


# ----------------------------------------------------------- simulated world
def temp_f(ts):           # ts: tz-aware
    c = ts.tz_convert(CT)
    day = c.toordinal() if hasattr(c, "toordinal") else c.date().toordinal()
    return 72 + 14 * math.sin(2 * math.pi * (c.hour - 9) / 24) + 6 * math.sin(day / 6.0)


def wind_ms(ts):
    c = ts.tz_convert(CT)
    return 8 + 3.5 * math.sin(c.date().toordinal() / 3.0 + c.hour / 6.0)


def ghi(ts):
    h = ts.tz_convert(CT).hour
    return max(0.0, 900 * math.sin(math.pi * (h - 6) / 13)) if 6 < h < 19 else 0.0


def price(day: dt.date, he: int) -> float:
    ts = pd.Timestamp(dt.datetime.combine(day, dt.time(0), CT)) + pd.Timedelta(hours=he)
    cdd = max(0.0, temp_f(ts) - 65)
    noise = math.sin(day.toordinal() * 12.9898 + he * 78.233) * 1.5
    return round(22 + 1.3 * cdd - 1.6 * (wind_ms(ts) - 8) - 0.012 * ghi(ts)
                 + (18 if he in (19, 20, 21) else 0) + noise, 2)


def om_site(url, lat, lon, var, params):
    now = pd.Timestamp.now(tz="UTC").floor("h")
    if "start_date" in params:
        a = pd.Timestamp(params["start_date"], tz="UTC")
        b = pd.Timestamp(params["end_date"], tz="UTC") + pd.Timedelta(hours=23)
    else:
        a = now.normalize() - pd.Timedelta(days=params.get("past_days", 0))
        b = now.normalize() + pd.Timedelta(days=params.get("forecast_days", 1)) - pd.Timedelta(hours=1)
    ts = pd.date_range(a, b, freq="h")
    f = temp_f if var.startswith("temperature") else wind_ms if var.startswith("wind") else ghi
    jitter = 0.4 if "previous_day1" in var else 0.0
    return pd.DataFrame({"ts": ts.tz_convert(CT), "v": [f(t) + jitter for t in ts]})


def dam_long(day, source):
    rows = [{"op_date": day, "he": he, "location": loc, "spp": price(day, he), "source": source}
            for he in range(1, 25) for loc in history.HUBS]
    return pd.DataFrame(rows)


def dashboard_json(now):
    today = now.date()
    mid = dt.datetime.combine(today, dt.time(0), CT)
    dam = [{"hourEnding": he, "dstFlag": "N", **{k: price(today, he) for k in HUB_KEYS},
            "timestamp": (mid + dt.timedelta(hours=he)).strftime("%Y-%m-%d %H:%M:%S%z")}
           for he in range(1, 25)]
    rt = []
    yday = mid - dt.timedelta(days=1)
    for i in range(1, 96 + 40):
        ts = yday + dt.timedelta(minutes=15 * i)
        rt.append({"intervalEnding": ts.strftime("%H:%M"), "dstFlag": "N",
                   **{k: 25.0 + (i % 7) for k in HUB_KEYS},
                   "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S%z")})
    return {"lastUpdated": now.strftime("%Y-%m-%d %H:%M:%S%z"), "rtSppData": rt, "damSppData": dam}


def fuelmix_json(now):
    data = {}
    start = now - dt.timedelta(hours=40)
    for i in range(40 * 12):
        ts = start + dt.timedelta(minutes=5 * i)
        h = ts.hour + ts.minute / 60
        solar = max(0.0, 25000 * math.sin(math.pi * (h - 6.5) / 12)) if 6.5 < h < 18.5 else 0.0
        rec = {"Coal and Lignite": {"gen": 8000.0}, "Hydro": {"gen": 50.0}, "Nuclear": {"gen": 4900.0},
               "Other": {"gen": 40.0}, "Power Storage": {"gen": 0.0}, "Solar": {"gen": solar},
               "Wind": {"gen": 14000.0}, "Natural Gas": {"gen": max(2000.0, 52000 - solar)},
               "Power Storage Charging": {"gen": 0.0}, "Power Storage Discharging": {"gen": 0.0}}
        data.setdefault(ts.strftime("%Y-%m-%d"), {})[ts.strftime("%Y-%m-%d %H:%M:%S%z")] = rec
    return {"lastUpdated": now.isoformat(),
            "monthlyCapacity": {"Power Storage": 20948, "Solar": 40226, "Wind": 40405, "Hydro": 579,
                                "Other": 0, "Natural Gas": 69569, "Coal and Lignite": 13705,
                                "Nuclear": 5268},
            "data": data}


class FakeResp:
    text = "Date,Open,High,Low,Close\n2026-10-20,3.0,3.2,2.9,3.11\n"

    def raise_for_status(self):
        pass


@pytest.fixture
def world(monkeypatch):
    state = {"tomorrow_published": False}

    def http_json(url, params=None, **kw):
        now = dt.datetime.now(CT)
        if url.endswith("fuel-mix.json"):
            return fuelmix_json(now)
        if url.endswith("system-wide-prices.json"):
            return dashboard_json(now)
        raise RuntimeError(f"unexpected URL in test: {url}")

    def mis_daily(day):
        today = dt.datetime.now(CT).date()
        if day > today and not state["tomorrow_published"]:
            raise RuntimeError("not published yet")
        return dam_long(day, "ercot-mis-daily")

    def mis_annual(year):
        today = dt.datetime.now(CT).date()
        days = [today - dt.timedelta(days=i) for i in range(32, 200)]
        return pd.concat([dam_long(d, "ercot-mis-annual") for d in days if d.year == year])

    def eia(days=120):
        end = pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=3)
        ts = pd.date_range(end - pd.Timedelta(days=min(days, 90)), end, freq="h").tz_convert(CT)
        dem = [48000 + 900 * max(0.0, temp_f(t) - 65) + 3000 * math.sin(2 * math.pi * (t.hour - 4) / 24)
               for t in ts]
        sources.log("EIA-930 ERCOT demand", "OK", f"{len(ts)} hours (simulated)")
        return pd.DataFrame({"ts": ts, "demand": dem})

    monkeypatch.setattr(sources, "fetch_eia930_demand", eia)
    monkeypatch.setattr(sources, "http_json", http_json)
    monkeypatch.setattr(sources, "_om_site", om_site)
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: FakeResp())
    monkeypatch.setattr(history, "fetch_mis_daily", mis_daily)
    monkeypatch.setattr(history, "fetch_mis_annual", mis_annual)
    return state


def test_full_cycle(world):
    assert cli.main(["daily"]) == 0
    tomorrow = dt.datetime.now(CT).date() + dt.timedelta(days=1)
    fc = json.loads((config.FORECASTS / f"{tomorrow}.json").read_text())
    assert fc["primary_model"] == "hybrid" and not fc["primary_is_fallback"]
    assert len(fc["hourly"]) == 24
    assert all(r["hybrid"] is not None for r in fc["hourly"])
    for m in ("hybrid", "stack", "persistence", "seasonal"):
        assert m in fc["summary"], m
    assert fc["intervals"]["atc_p10"] < fc["summary"]["hybrid"]["atc"] < fc["intervals"]["atc_p90"]
    assert fc["pre_close"] == (dt.datetime.now(CT) < dt.datetime.fromisoformat(fc["dam_close_ct"]))
    assert fc["hybrid"]["training_weather"].startswith("Open-Meteo previous-day")

    # The model should track the simulated world better than "today's price again".
    truth = np.array([price(tomorrow, h) for h in range(1, 25)])
    hyb = np.array([r["hybrid"] for r in fc["hourly"]])
    pers = np.array([r["persistence"] for r in fc["hourly"]])
    assert np.mean(np.abs(hyb - truth)) < np.mean(np.abs(pers - truth)) + 1.0

    # Afternoon: ERCOT publishes tomorrow's DAM -> the call is scored.
    world["tomorrow_published"] = True
    assert cli.main(["dam"]) == 0
    sc = pd.read_csv(config.SCORECARD)
    row = sc[sc["target_date"] == str(tomorrow)].iloc[0]
    assert row["act_atc"] == pytest.approx(truth.mean(), abs=0.01)
    assert row["err_atc"] == pytest.approx(row["fc_atc"] - row["act_atc"], abs=0.011)

    assert cli.main(["capture"]) == 0
    assert any(config.OUTCOMES.glob("*.json"))

    assert cli.main(["backtest", "--days", "15"]) == 0
    bt = json.loads((config.BACKTEST / "summary.json").read_text())
    assert bt["status"] == "ok" and bt["n_days"] == 15

    data = json.loads((config.DOCS / "data.json").read_text())
    assert data["latest_forecast"]["target_date"] == str(tomorrow)
    assert data["backtest"]["summary"]["status"] == "ok"
    assert str(tomorrow) in data["detail"]


def test_rerun_after_close_never_replaces_a_live_call(world):
    from src import forecast as fx
    tomorrow = dt.datetime.now(CT).date() + dt.timedelta(days=1)
    close = dt.datetime.combine(tomorrow - dt.timedelta(days=1), dt.time(10), CT).isoformat()
    live = {"target_date": str(tomorrow), "pre_close": True, "live": True, "dam_close_ct": close,
            "primary_is_fallback": False, "x": 1}
    (config.FORECASTS / f"{tomorrow}.json").write_text(json.dumps(live))
    late = {"target_date": str(tomorrow), "pre_close": False, "live": False, "dam_close_ct": close,
            "primary_is_fallback": False, "x": 2}
    _, written = fx.save_forecast(late, force=True)
    assert not written
    assert json.loads((config.FORECASTS / f"{tomorrow}.json").read_text())["x"] == 1
    # a live fallback CAN be upgraded to the primary model, but only before the close
    (config.FORECASTS / f"{tomorrow}.json").write_text(json.dumps({**live, "primary_is_fallback": True}))
    _, written = fx.save_forecast({**live, "x": 3})
    assert written == (dt.datetime.now(CT) < dt.datetime.fromisoformat(close))


def test_save_rechecks_the_clock():
    from src import forecast as fx
    t = dt.date(2020, 1, 2)                       # its close is long past
    fc = {"target_date": str(t), "pre_close": True, "live": True,
          "dam_close_ct": dt.datetime.combine(t - dt.timedelta(days=1), dt.time(10), CT).isoformat()}
    fx.save_forecast(fc)
    saved = json.loads((config.FORECASTS / f"{t}.json").read_text())
    assert saved["live"] is False and saved["pre_close"] is False


def test_backtest_refuses_observed_weather(world, monkeypatch):
    for i in range(1, 120):
        history.upsert(dam_long(dt.datetime.now(CT).date() - dt.timedelta(days=i), "ercot-mis-annual"))
    monkeypatch.setattr(sources, "fetch_weather_previous_day1", lambda a, b: pd.DataFrame())
    from src import backtest
    r = backtest.run(n_days=10)
    assert r["status"] == "unavailable" and "optimistic" in r["reason"]


def test_merge_after_push_collision_keeps_existing_live_call(tmp_path):
    t = dt.date(2030, 1, 2)
    close = dt.datetime.combine(t - dt.timedelta(days=1), dt.time(10), CT).isoformat()
    theirs = {"target_date": str(t), "live": True, "pre_close": True, "dam_close_ct": close, "x": "theirs"}
    (config.FORECASTS / f"{t}.json").write_text(json.dumps(theirs))
    (tmp_path / "forecasts").mkdir()
    (tmp_path / "forecasts" / f"{t}.json").write_text(json.dumps({**theirs, "x": "ours"}))
    (tmp_path / "forecasts" / "2030-01-03.json").write_text(json.dumps({**theirs, "target_date": "2030-01-03", "x": "new"}))
    assert cli.main(["merge", "--src", str(tmp_path)]) == 0
    assert json.loads((config.FORECASTS / f"{t}.json").read_text())["x"] == "theirs"
    assert json.loads((config.FORECASTS / "2030-01-03.json").read_text())["x"] == "new"
