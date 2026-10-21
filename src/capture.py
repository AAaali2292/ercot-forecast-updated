"""Snapshot what actually happened.

Day-ahead actuals come from ERCOT's official MIS settlement files (src.history), which
stay online for 31 days, so they cannot be lost by a late job.

Real-time prices and generation only exist on the public dashboard, which holds about
one day. Cloud cron drifts and ignores DST, so `capture_all` snapshots EVERY date the
feed holds and `save_outcome` never replaces a fuller snapshot with a thinner one.
Run it several times a night and the fullest capture wins.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import history, sources
from .config import CT, OUTCOMES, PEAK_HE
from .timeutil import hour_ending

RT_KEY = "hbHubAvg"


def hours_in_day(d: dt.date) -> int:
    a = dt.datetime.combine(d, dt.time(0), CT)
    b = dt.datetime.combine(d + dt.timedelta(days=1), dt.time(0), CT)
    return int(round((b.astimezone(dt.timezone.utc) - a.astimezone(dt.timezone.utc))
                     .total_seconds() / 3600))


def build_outcome(d: dt.date, rt: pd.DataFrame, gen: pd.DataFrame) -> dict | None:
    r = rt[rt["op_date"] == d] if len(rt) else rt
    if r is None or not len(r) or RT_KEY not in r:
        return None
    expected = 4 * hours_in_day(d)
    by_he = r.groupby("he")[RT_KEY].mean()
    pk = r[r["he"].isin(PEAK_HE)]
    out = {
        "target_date": d.isoformat(),
        "captured_at_ct": dt.datetime.now(CT).isoformat(timespec="seconds"),
        "intervals": int(len(r)),
        "expected_intervals": expected,
        "coverage_pct": round(100 * len(r) / expected, 1),
        "rt": {
            "atc": round(float(r[RT_KEY].mean()), 2),
            "peak": round(float(pk[RT_KEY].mean()), 2) if len(pk) == 4 * len(PEAK_HE) else None,
            "max": round(float(r[RT_KEY].max()), 2),
            "min": round(float(r[RT_KEY].min()), 2),
            "n_intervals_over_200": int((r[RT_KEY] > 200).sum()),
            "n_intervals_negative": int((r[RT_KEY] < 0).sum()),
        },
        "rt_by_he": {int(h): round(float(v), 2) for h, v in by_he.items()},
    }
    if gen is not None and len(gen):
        g = gen.copy()
        g = pd.concat([g, hour_ending(g["ts"])], axis=1) if "he" not in g else g
        g = g[g["op_date"] == d]
        if len(g):
            gp = g[g["he"].isin(PEAK_HE)]
            out["fundamentals"] = {
                "intervals": int(len(g)),
                "coverage_pct": round(100 * len(g) / (12 * hours_in_day(d)), 1),
                "peak_load_mw": round(float(g["load"].max()), 0),
                "peak_net_load_mw": round(float(g["net_load"].max()), 0),
                "min_net_load_mw": round(float(g["net_load"].min()), 0),
                "wind_peak_block_mw": round(float(gp["wind"].mean()), 0) if len(gp) else None,
                "solar_peak_block_mw": round(float(gp["solar"].mean()), 0) if len(gp) else None,
            }
    return out


def capture_all() -> list[dict]:
    px = sources.fetch_prices()
    mix = sources.fetch_fuel_mix()
    history.ingest_dashboard(px.get("dam"))
    history.refresh(days_back=3, include_tomorrow=True)
    rt = px.get("rt")
    if rt is None or not len(rt):
        return []
    out = []
    for d in sorted(set(rt["op_date"])):
        o = build_outcome(d, rt, mix.get("gen"))
        if o:
            out.append(o)
    return out


def save_outcome(o: dict) -> tuple[str, bool]:
    p = OUTCOMES / f"{o['target_date']}.json"
    if p.exists():
        try:
            prev = json.loads(p.read_text())
            if prev.get("intervals", 0) > o.get("intervals", 0):
                return str(p), False
            # Same RT coverage: keep the richer fundamentals snapshot.
            if (prev.get("intervals", 0) == o.get("intervals", 0)
                    and (prev.get("fundamentals") or {}).get("intervals", 0)
                    > (o.get("fundamentals") or {}).get("intervals", 0)):
                return str(p), False
        except Exception:  # noqa: BLE001
            pass
    p.write_text(json.dumps(o, indent=2))
    return str(p), True
