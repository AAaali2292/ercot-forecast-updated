"""Official ERCOT day-ahead price history, stored in data/dam_history.csv.

Sources, in order of authority:
  1. ERCOT MIS daily DAM Settlement Point Prices (NP4-190-CD, report 12331). One file
     per operating day, published when the DAM clears (~13:30 CT the day before).
     ERCOT keeps the last 31 days online.
  2. ERCOT MIS Historical DAM Load Zone and Hub Prices (NP4-180-ER, report 13060). One
     file per year, republished weekly. Covers everything older than 31 days.
  3. The ERCOT public dashboard. Only ever holds the current day; used as a cross-check
     and as a last resort, and labelled as such.

Downloads go through the `gridstatus` library, which is widely used for exactly these
files and handles ERCOT's DST flags. Every batch is validated before it is stored; a
batch that fails validation is rejected and logged, never repaired by guesswork.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import sources
from .config import CT, DAM_HISTORY, HUB, HUBS
from .timeutil import hour_ending

COLS = ["op_date", "he", "location", "spp", "source"]
PRICE_MIN, PRICE_MAX = -300.0, 5100.0       # ERCOT floor -$251, offer cap $5,000


# ------------------------------------------------------------------- store
def load() -> pd.DataFrame:
    if not DAM_HISTORY.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(DAM_HISTORY)
    if not len(df):
        return pd.DataFrame(columns=COLS)
    df["op_date"] = pd.to_datetime(df["op_date"]).dt.date
    df["he"] = df["he"].astype(int)
    return df[COLS]


_RANK = {"ercot-mis-daily": 3, "ercot-mis-annual": 2, "ercot-dashboard": 1}


def upsert(new: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge rows in. A more authoritative source replaces a less authoritative one."""
    old = load()
    if new is None or not len(new):
        return old, 0
    new = new[COLS].copy()
    both = pd.concat([old, new], ignore_index=True) if len(old) else new.reset_index(drop=True)
    both["_rank"] = both["source"].map(_RANK).fillna(0)
    both = (both.sort_values(["op_date", "he", "location", "_rank"], kind="mergesort")
                .drop_duplicates(["op_date", "he", "location"], keep="last")
                .drop(columns="_rank")
                .sort_values(["op_date", "location", "he"]).reset_index(drop=True))
    both["spp"] = both["spp"].round(2)
    added = len(both) - len(old)
    both.to_csv(DAM_HISTORY, index=False)
    return both, added


def hub_hourly(df: pd.DataFrame | None = None, location: str = HUB) -> pd.DataFrame:
    """Wide-ish view for one location: op_date, he, spp (DST repeat hour averaged)."""
    df = load() if df is None else df
    d = df[df["location"] == location]
    return (d.groupby(["op_date", "he"], as_index=False)["spp"].mean()
             .sort_values(["op_date", "he"]).reset_index(drop=True))


def complete_days(h: pd.DataFrame) -> set:
    """Days with a full set of hours (23 on spring-forward, 24 normally)."""
    n = h.groupby("op_date")["he"].nunique()
    return set(n[n >= 23].index)


# -------------------------------------------------------------- validation
def validate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Reject anything that is not a clean, complete day of hub prices."""
    if df is None or not len(df):
        raise ValueError(f"{label}: no rows")
    bad = df["spp"].isna() | (df["spp"] < PRICE_MIN) | (df["spp"] > PRICE_MAX)
    if bad.any():
        raise ValueError(f"{label}: {int(bad.sum())} prices missing or outside "
                         f"[{PRICE_MIN}, {PRICE_MAX}]")
    if HUB not in set(df["location"]):
        raise ValueError(f"{label}: {HUB} missing")
    hub = df[df["location"] == HUB]
    per_day = hub.groupby("op_date")["he"].nunique()
    short = per_day[per_day < 23]
    if len(short):
        # keep complete days, drop partial ones explicitly
        df = df[~df["op_date"].isin(short.index)]
        sources.log(f"{label} validation", "WARN",
                    f"dropped {len(short)} partial day(s): "
                    + ", ".join(str(d) for d in short.index[:5]))
        if not len(df):
            raise ValueError(f"{label}: no complete days")
    return df


# ------------------------------------------------------------- gridstatus
def _from_gridstatus(gs: pd.DataFrame, source: str) -> pd.DataFrame:
    """gridstatus SPP frame (Interval End tz-aware, Location, SPP) -> our long form."""
    if gs is None or not len(gs):
        return pd.DataFrame(columns=COLS)
    d = gs[gs["Location"].isin(HUBS)].copy()
    he = hour_ending(pd.to_datetime(d["Interval End"]))
    out = pd.DataFrame({"op_date": he["op_date"].to_numpy(), "he": he["he"].to_numpy(),
                        "location": d["Location"].to_numpy(),
                        "spp": pd.to_numeric(d["SPP"], errors="coerce").to_numpy(),
                        "source": source})
    # ERCOT occasionally posts a corrected/retry file for the same day. If two files
    # disagree, say so loudly rather than quietly averaging them.
    spread = out.groupby(["op_date", "he", "location"])["spp"].agg(lambda s: s.max() - s.min())
    if len(spread) and spread.max() > 0.005:
        sources.log("MIS duplicate files disagree", "WARN",
                    f"{int((spread > 0.005).sum())} hub-hours differ by up to ${spread.max():.2f}; "
                    "averaged - check ERCOT notices")
    return (out.groupby(["op_date", "he", "location", "source"], as_index=False)["spp"].mean()
               [COLS])


def _ercot():
    import gridstatus  # imported lazily so tests and offline tools need not install it
    return gridstatus.Ercot()


def fetch_mis_daily(day: dt.date) -> pd.DataFrame:
    """Official DAM SPPs for one operating day (available from ~13:30 CT the day before)."""
    from gridstatus.base import Markets
    gs = _ercot().get_spp(date=pd.Timestamp(day), market=Markets.DAY_AHEAD_HOURLY,
                          location_type="Trading Hub")
    out = _from_gridstatus(gs, "ercot-mis-daily")
    out = out[out["op_date"] == day]            # a file can only vouch for its own day
    return validate(out, f"MIS DAM {day}")


def fetch_mis_annual(year: int) -> pd.DataFrame:
    gs = _ercot().get_dam_spp(year)
    return validate(_from_gridstatus(gs, "ercot-mis-annual"), f"MIS DAM annual {year}")


def _official_days() -> set:
    """Complete days whose hub prices come from an official ERCOT MIS file."""
    df = load()
    off = df[df["source"].astype(str).str.startswith("ercot-mis")]
    return complete_days(hub_hourly(off)) if len(off) else set()


# ----------------------------------------------------------------- jobs
def refresh(days_back: int = 31, include_tomorrow: bool = True,
            annual: bool = False, annual_years: list[int] | None = None) -> dict:
    """Pull official DAM prices into the store. Safe to run any number of times."""
    have = _official_days()
    today = dt.datetime.now(CT).date()
    targets = [today - dt.timedelta(days=i) for i in range(days_back, -1, -1)]
    if include_tomorrow:
        targets.append(today + dt.timedelta(days=1))
    got, failed, added_total = [], [], 0

    if annual:
        years = annual_years or sorted({today.year, (today - dt.timedelta(days=150)).year})
        for y in years:
            try:
                _, added = upsert(fetch_mis_annual(y))
                added_total += added
                got.append(f"annual {y}")
            except Exception as e:  # noqa: BLE001
                failed.append(f"annual {y}: {e}")
        have = _official_days()

    for d in targets:
        # Days held only from the dashboard are re-fetched until the official file lands.
        if d in have and d < today - dt.timedelta(days=2):
            continue
        try:
            _, added = upsert(fetch_mis_daily(d))
            added_total += added
            got.append(str(d))
        except Exception as e:  # noqa: BLE001
            # Tomorrow's file not being out yet is normal before ~13:30 CT.
            if d > today:
                continue
            failed.append(f"{d}: {str(e)[:120]}")

    status = "OK" if not failed else ("WARN" if got else "FAIL")
    sources.log("ERCOT MIS DAM prices", status,
                f"fetched {len(got)} file(s), +{added_total} rows"
                + (f"; failed: {'; '.join(failed[:4])}" if failed else ""), added_total)
    return {"fetched": got, "failed": failed, "added": added_total}


def ingest_dashboard(dash_dam: pd.DataFrame) -> dict:
    """Store dashboard DAM rows (lowest authority) and cross-check against MIS rows."""
    long = sources.dashboard_dam_long(dash_dam)
    if not len(long):
        return {"checked": 0, "max_abs_diff": None}
    try:
        long = validate(long, "dashboard DAM")
    except ValueError as e:
        sources.log("Dashboard DAM validation", "WARN", e)
        return {"checked": 0, "max_abs_diff": None}

    cur = load()
    m = long.merge(cur[cur["source"] != "ercot-dashboard"],
                   on=["op_date", "he", "location"], suffixes=("_dash", "_mis"))
    diff = (m["spp_dash"] - m["spp_mis"]).abs()
    res = {"checked": int(len(m)),
           "max_abs_diff": round(float(diff.max()), 4) if len(m) else None}
    if len(m):
        status = "OK" if diff.max() <= 0.011 else "WARN"
        sources.log("DAM cross-check dashboard vs MIS", status,
                    f"{len(m)} hub-hours compared, max |diff| ${diff.max():.4f}")
    upsert(long)
    return res
