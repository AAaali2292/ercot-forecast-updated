"""Live data sources. Every fetch is open data; the only key is a free EIA one.

Feed status is recorded in FEED so every forecast file and daily note states exactly
which inputs were live and which failed. Nothing here invents a number: a failed feed
returns an empty frame / None and is logged as FAIL, and the caller decides whether it
can still proceed honestly.
"""
from __future__ import annotations

import datetime as dt
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
import requests

from .config import (CDD_BASE_F, CT, DASH_HUB_KEYS, EIA_API_KEY, HDD_BASE_F,
                     LOAD_CITIES, SOLAR_SITES, WIND_SITES)
from .timeutil import hour_ending

HEADERS = {
    "User-Agent": "ercot-daily-forecast/2.0 (+https://github.com; public-data research)",
    "Accept": "application/json, text/plain, */*",
}
ERCOT_BASE = "https://www.ercot.com/api/1/services/read/dashboards"
OM_FORECAST = "https://api.open-meteo.com/v1/forecast"
OM_PREVIOUS = "https://previous-runs-api.open-meteo.com/v1/forecast"
EIA_V2 = "https://api.eia.gov/v2"

FEED: dict[str, dict] = {}


_SECRET = re.compile(r"(api_key=)[^&\s'\"]+", re.I)


def redact(msg) -> str:
    """Never let a key reach a committed file: feed details end up in public JSON."""
    s = _SECRET.sub(r"\1***", str(msg))
    if EIA_API_KEY:
        s = s.replace(EIA_API_KEY, "***")
    return s


def log(name: str, status: str, detail: str = "", rows: Optional[int] = None) -> None:
    FEED[name] = {"status": status, "detail": redact(detail)[:300], "rows": rows}


def _explain(e: Exception, url: str) -> str:
    msg = redact(e)
    if "ercot.com" in url and "403" in msg:
        msg += (" | ERCOT blocks requests from outside the USA (Imperva firewall). "
                "GitHub-hosted runners are US-based; a 403 here means the runner IP was "
                "refused - re-run the job.")
    return msg


def http_json(url, params=None, tries=3, timeout=25):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < tries - 1:
                time.sleep(2.0 * (i + 1))
    raise RuntimeError(f"GET failed {url}: {_explain(last, url)}")


# --------------------------------------------------------------------------- ERCOT
# The feed now carries "Power Storage Charging" / "Power Storage Discharging" next to
# the net "Power Storage" figure. Only the net figure is used, otherwise batteries
# would be counted twice.
FUEL_MAP = {
    "Natural Gas": "gas", "Coal and Lignite": "coal", "Nuclear": "nuclear",
    "Wind": "wind", "Solar": "solar", "Power Storage": "storage",
    "Hydro": "hydro", "Other": "other",
}
FUELS = list(FUEL_MAP.values())


def parse_fuel_mix(j: dict) -> dict:
    cap = {FUEL_MAP[k]: float(v) for k, v in (j.get("monthlyCapacity") or {}).items()
           if k in FUEL_MAP and v is not None}
    rows = []
    for _day, ivs in (j.get("data") or {}).items():
        for ts, fuels in ivs.items():
            rec = {"ts": ts}
            for f, d in fuels.items():
                if f in FUEL_MAP and isinstance(d, dict):
                    rec[FUEL_MAP[f]] = d.get("gen")
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        return {"gen": df, "capacity": cap, "last_updated": j.get("lastUpdated")}
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").dt.tz_convert(CT)
    df = df.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    for c in FUELS:
        if c not in df.columns:
            df[c] = 0.0
    df[FUELS] = df[FUELS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["load"] = df[FUELS].sum(axis=1)                 # storage enters net of charging
    df["renewables"] = df["wind"] + df["solar"]
    df["net_load"] = df["load"] - df["renewables"]
    return {"gen": df, "capacity": cap, "last_updated": j.get("lastUpdated")}


def fetch_fuel_mix() -> dict:
    """5-minute generation by fuel plus ERCOT's installed capacity by fuel."""
    try:
        out = parse_fuel_mix(http_json(f"{ERCOT_BASE}/fuel-mix.json"))
        df, cap = out["gen"], out["capacity"]
        if df.empty:
            raise RuntimeError("fuel-mix feed returned no intervals")
        log("ERCOT fuel mix", "OK",
            f"{len(df)} intervals {df['ts'].min():%m-%d %H:%M}-{df['ts'].max():%m-%d %H:%M}; "
            f"installed gas {cap.get('gas', 0):,.0f} / wind {cap.get('wind', 0):,.0f} / "
            f"solar {cap.get('solar', 0):,.0f} / storage {cap.get('storage', 0):,.0f} MW",
            len(df))
        return out
    except Exception as e:  # noqa: BLE001
        log("ERCOT fuel mix", "FAIL", e)
        return {"gen": pd.DataFrame(), "capacity": {}, "last_updated": None}


def parse_dashboard_prices(j: dict) -> dict:
    """Real-time 15-min and day-ahead hourly SPPs, in long form keyed by (op_date, he)."""
    def mk(key):
        d = pd.DataFrame(j.get(key) or [])
        if d.empty or "timestamp" not in d:
            return pd.DataFrame()
        d["ts"] = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dt.tz_convert(CT)
        d = d.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        d = pd.concat([d, hour_ending(d["ts"])], axis=1)
        for k in DASH_HUB_KEYS:
            if k in d.columns:
                d[k] = pd.to_numeric(d[k], errors="coerce")
        return d

    rt, dam = mk("rtSppData"), mk("damSppData")
    # Cross-check our hour-ending arithmetic against ERCOT's own label.
    if len(dam) and "hourEnding" in dam.columns:
        lab = pd.to_numeric(dam["hourEnding"], errors="coerce")
        normal = dam["dstFlag"].astype(str).str.upper().ne("Y") if "dstFlag" in dam else True
        # Spring-forward day: ERCOT skips HE02 and labels the next hour HE03, whereas we
        # label by elapsed hours. Compare only on ordinary 24/25-hour days.
        n_hours = dam.groupby("op_date")["he"].transform("size")
        bad = (lab != dam["he"]) & normal & (n_hours >= 24)
        if bool(np.any(bad)):
            raise RuntimeError(f"hour-ending mismatch in DAM feed: {int(np.sum(bad))} rows")
    return {"rt": rt, "dam": dam, "last_updated": j.get("lastUpdated")}


def fetch_prices() -> dict:
    try:
        out = parse_dashboard_prices(http_json(f"{ERCOT_BASE}/system-wide-prices.json"))
        rt, dam = out["rt"], out["dam"]
        rd = ", ".join(str(d) for d in sorted(set(rt["op_date"]))) if len(rt) else "none"
        dd = ", ".join(str(d) for d in sorted(set(dam["op_date"]))) if len(dam) else "none"
        log("ERCOT prices (dashboard)", "OK",
            f"RT {len(rt)} intervals [{rd}]; DAM {len(dam)} hours [{dd}]", len(rt))
        return out
    except Exception as e:  # noqa: BLE001
        log("ERCOT prices (dashboard)", "FAIL", e)
        return {"rt": pd.DataFrame(), "dam": pd.DataFrame(), "last_updated": None}


def dashboard_dam_long(dam: pd.DataFrame) -> pd.DataFrame:
    """Dashboard DAM frame -> long rows (op_date, he, location, spp, source)."""
    if dam is None or not len(dam):
        return pd.DataFrame(columns=["op_date", "he", "location", "spp", "source"])
    keys = [k for k in DASH_HUB_KEYS if k in dam.columns]
    m = dam.melt(id_vars=["op_date", "he"], value_vars=keys,
                 var_name="key", value_name="spp")
    m["location"] = m["key"].map(DASH_HUB_KEYS)
    m["source"] = "ercot-dashboard"
    m = m.dropna(subset=["spp"])
    return (m.groupby(["op_date", "he", "location", "source"], as_index=False)["spp"].mean()
             [["op_date", "he", "location", "spp", "source"]])


# ------------------------------------------------------------------------ EIA-930
def fetch_eia930_demand(days: int = 120) -> pd.DataFrame:
    """Hourly ERCOT demand history (hour-ending UTC). Trains the stack model's load model."""
    if not EIA_API_KEY:
        log("EIA-930 ERCOT demand", "WARN",
            "no EIA_API_KEY secret -> stack model's load model uses 2 days of ERCOT data")
        return pd.DataFrame(columns=["ts", "demand"])
    try:
        start = (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=days)).isoformat()
        rows, offset = [], 0
        while True:
            j = http_json(
                f"{EIA_V2}/electricity/rto/region-data/data/",
                params={
                    "api_key": EIA_API_KEY, "frequency": "hourly", "data[0]": "value",
                    "facets[respondent][]": "ERCO", "facets[type][]": "D",
                    "start": f"{start}T00", "length": 5000, "offset": offset,
                    "sort[0][column]": "period", "sort[0][direction]": "desc",
                })
            chunk = j.get("response", {}).get("data", [])
            rows += chunk
            if len(chunk) < 5000 or offset > 20000:
                break
            offset += 5000
        if not rows:
            raise RuntimeError("empty EIA-930 response")
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True,
                                  errors="coerce").dt.tz_convert(CT)
        df["demand"] = pd.to_numeric(df["value"], errors="coerce")
        df = (df.dropna(subset=["ts", "demand"])[["ts", "demand"]]
                .sort_values("ts").drop_duplicates("ts").reset_index(drop=True))
        df = df[df["demand"] > 10000]          # drop obvious EIA placeholders / zeros
        log("EIA-930 ERCOT demand", "OK",
            f"{len(df)} hours, {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d}", len(df))
        return df
    except Exception as e:  # noqa: BLE001
        log("EIA-930 ERCOT demand", "FAIL", e)
        return pd.DataFrame(columns=["ts", "demand"])


# ------------------------------------------------------------------------ weather
def _om_site(url: str, lat: float, lon: float, var: str, params: dict) -> pd.DataFrame:
    """One site, one variable, in UTC (unambiguous across DST), returned as CT."""
    p = {"latitude": lat, "longitude": lon, "hourly": var,
         "temperature_unit": "fahrenheit", "wind_speed_unit": "ms", "timezone": "GMT"}
    p.update(params)
    j = http_json(url, params=p)
    h = j["hourly"]
    df = pd.DataFrame({"ts": pd.to_datetime(h["time"], format="%Y-%m-%dT%H:%M")})
    df["v"] = pd.to_numeric(pd.Series(h[var]), errors="coerce")
    df["ts"] = df["ts"].dt.tz_localize("UTC").dt.tz_convert(CT)
    return df


def _weighted(frames: list[tuple[str, float, pd.DataFrame]]) -> pd.DataFrame:
    parts = []
    for name, w, d in frames:
        parts.append(d.assign(site=name, w=w))
    return pd.concat(parts, ignore_index=True)


def build_weather(temp: pd.DataFrame, wind: pd.DataFrame, solar: pd.DataFrame) -> pd.DataFrame:
    """Per-site long frames (ts, v, site, w) -> one hourly frame of model drivers."""
    def wavg(d, name):
        d = d.dropna(subset=["v"])
        g = d.assign(x=d["v"] * d["w"]).groupby("ts").agg(x=("x", "sum"), w=("w", "sum"))
        return (g["x"] / g["w"]).rename(name)

    out = pd.concat([wavg(temp, "temp_f"), wavg(wind, "wind_ms"), wavg(solar, "ghi")],
                    axis=1).reset_index().rename(columns={"index": "ts"})
    wsite = (wind.pivot_table(index="ts", columns="site", values="v")
                 .add_prefix("ws_").reset_index())
    out = out.merge(wsite, on="ts", how="left").sort_values("ts").reset_index(drop=True)
    out["hdd"] = (HDD_BASE_F - out["temp_f"]).clip(lower=0)
    out["cdd"] = (out["temp_f"] - CDD_BASE_F).clip(lower=0)
    # Thermal mass: after a run of hot hours the same temperature draws more load.
    out["cdd_24h"] = out["cdd"].rolling(24, min_periods=1).mean()
    out["cdd_72h"] = out["cdd"].rolling(72, min_periods=1).mean()
    out = pd.concat([out, hour_ending(out["ts"])], axis=1)
    return out


def _fetch_all_sites(url: str, params: dict, suffix: str = "") -> pd.DataFrame:
    temp = _weighted([(n, w, _om_site(url, la, lo, "temperature_2m" + suffix, params))
                      for n, la, lo, w in LOAD_CITIES])
    wind = _weighted([(n, w, _om_site(url, la, lo, "wind_speed_100m" + suffix, params))
                      for n, la, lo, w in WIND_SITES])
    solar = _weighted([(n, w, _om_site(url, la, lo, "shortwave_radiation" + suffix, params))
                       for n, la, lo, w in SOLAR_SITES])
    return build_weather(temp, wind, solar)


def fetch_weather_forecast(past_days: int = 92, forecast_days: int = 3) -> pd.DataFrame:
    """Live NWP for the forecast, plus recent history for the stack model's load fit.

    92 is Open-Meteo's maximum past window; the load model can only train on the
    overlap of demand history and weather history, so a short pull silently starves it.
    """
    try:
        out = _fetch_all_sites(OM_FORECAST, {"past_days": min(past_days, 92),
                                             "forecast_days": min(forecast_days, 16)})
        log("Open-Meteo NWP (live)", "OK",
            f"{len(out)} hours, {out['ts'].min():%m-%d} to {out['ts'].max():%m-%d}", len(out))
        return out
    except Exception as e:  # noqa: BLE001
        log("Open-Meteo NWP (live)", "FAIL", e)
        return pd.DataFrame()


def fetch_weather_previous_day1(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Archived forecasts as they stood ~24h before valid time (Open-Meteo Previous Runs).

    This is what makes the hybrid's training data and the backtest honest: each past
    day is described by the forecast that existed the day before, not by what the
    weather turned out to be.
    """
    try:
        out = _fetch_all_sites(OM_PREVIOUS, {"start_date": start.isoformat(),
                                             "end_date": end.isoformat()},
                               suffix="_previous_day1")
        good = out.dropna(subset=["temp_f", "wind_ms", "ghi"])
        if len(good) < 24:
            raise RuntimeError("previous-runs archive returned no usable hours")
        log("Open-Meteo previous-day forecasts", "OK",
            f"{len(good)} hours, {good['ts'].min():%Y-%m-%d} to {good['ts'].max():%Y-%m-%d}",
            len(good))
        return out
    except Exception as e:  # noqa: BLE001
        log("Open-Meteo previous-day forecasts", "FAIL", e)
        return pd.DataFrame()


# --------------------------------------------------------------------------- gas
def fetch_gas_price() -> Optional[dict]:
    """Henry Hub spot (EIA) first, NYMEX front month (Stooq) as backup."""
    if EIA_API_KEY:
        try:
            j = http_json(f"{EIA_V2}/seriesid/NG.RNGWHHD.D",
                          params={"api_key": EIA_API_KEY, "length": 5,
                                  "sort[0][column]": "period",
                                  "sort[0][direction]": "desc"})
            rows = j.get("response", {}).get("data", [])
            v, d = float(rows[0]["value"]), str(rows[0]["period"])
            log("Gas price", "OK", f"EIA Henry Hub spot ${v:.3f}/MMBtu ({d})")
            return {"price": v, "as_of": d, "source": "EIA Henry Hub spot"}
        except Exception as e:  # noqa: BLE001
            log("Gas price", "WARN", f"EIA failed ({e}); trying futures")
    try:
        import io
        r = requests.get("https://stooq.com/q/d/l/?s=ng.f&i=d", timeout=30, headers=HEADERS)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        last = d.dropna(subset=["Close"]).iloc[-1]
        v = float(last["Close"])
        log("Gas price", "OK", f"NYMEX NG front month ${v:.3f}/MMBtu ({last['Date']})")
        return {"price": v, "as_of": str(last["Date"]), "source": "NYMEX NG front month (Stooq)"}
    except Exception as e:  # noqa: BLE001
        log("Gas price", "FAIL", f"{e} -> stack model uses config default, flagged")
        return None


def feed_table() -> pd.DataFrame:
    return (pd.DataFrame(FEED).T.reset_index().rename(columns={"index": "feed"})
            if FEED else pd.DataFrame(columns=["feed", "status", "detail", "rows"]))
