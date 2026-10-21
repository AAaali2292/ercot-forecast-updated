"""Walk-forward backtest of the hybrid model. SIMULATED - never mixed with the live record.

For each past day T:
  * the model is fitted only on target days <= T-1,
  * the features for T use DAM prices up to T-1 and the weather forecast for T as it
    stood ~24h earlier (Open-Meteo previous-runs archive),
  * the 80% interval comes from rolling-origin errors that also end at T-1.

That is the same information set the live model has at 06:00 CT. What a backtest
cannot prove is that the method was fixed before the data was seen - which is why the
live, timestamped, git-committed record exists and is reported separately.

If the previous-runs archive is unavailable the backtest is NOT run with observed
weather instead (that would be optimistic); it reports itself unavailable.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import history, sources
from .config import BACKTEST, CFG, CT, PEAK_HE, Config
from .hybrid import (HE_BLOCK, HybridModel, build_rows, interval_from_residuals,
                     rolling_residuals, weather_by_hour)


def run(n_days: int = 60, cfg: Config = CFG) -> dict:
    now = dt.datetime.now(CT)
    dam = history.hub_hourly()
    full = history.complete_days(dam)
    today = now.date()
    days = sorted(d for d in full if d < today)
    if len(days) < n_days + cfg.hybrid_min_train_days + 7:
        n_days = max(0, len(days) - cfg.hybrid_min_train_days - 7)
    test_days = days[-n_days:] if n_days else []
    meta = {"generated_at_ct": now.isoformat(timespec="seconds"), "label": "SIMULATED BACKTEST",
            "requested_days": n_days, "method": __doc__.strip().splitlines()[0]}
    if not test_days:
        meta.update({"status": "unavailable", "reason": "not enough DAM history"})
        (BACKTEST / "summary.json").write_text(json.dumps(meta, indent=2))
        return meta

    start = test_days[0] - dt.timedelta(days=cfg.hybrid_train_days + cfg.hybrid_cv_days + 10)
    wx = sources.fetch_weather_previous_day1(start, test_days[-1] + dt.timedelta(days=1))
    if not len(wx):
        meta.update({"status": "unavailable",
                     "reason": "Open-Meteo previous-runs archive unavailable; refusing to "
                               "backtest on observed weather (would be optimistic)",
                     "feeds": sources.FEED})
        (BACKTEST / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta

    wxh = weather_by_hour(wx)
    all_days = sorted(d for d in dam["op_date"].unique() if d >= start)
    rows = build_rows(all_days, dam, wxh)

    out = []
    for t in test_days:
        as_of = t - dt.timedelta(days=1)
        try:
            m = HybridModel(cfg).fit(rows, as_of=as_of)
        except ValueError:
            continue
        r = rows[rows["op_date"] == t]
        if r["he"].nunique() < 23:
            continue
        res = rolling_residuals(rows, as_of=as_of, n_days=cfg.hybrid_cv_days, cfg=cfg)
        iv = interval_from_residuals(res)
        pred = m.predict(r)
        o = r[["op_date", "he", "y", "lag1", "lag7"]].copy()
        o["hybrid"] = pred
        o["block"] = o["he"].map(HE_BLOCK)
        o["p10"] = o["hybrid"] + o["block"].map(lambda b: iv["hourly"][b][0]) if iv else np.nan
        o["p90"] = o["hybrid"] + o["block"].map(lambda b: iv["hourly"][b][1]) if iv else np.nan
        o["atc_lo"] = iv["daily"][0] if iv else np.nan
        o["atc_hi"] = iv["daily"][1] if iv else np.nan
        out.append(o)

    if not out:
        meta.update({"status": "unavailable", "reason": "no day could be fitted"})
        (BACKTEST / "summary.json").write_text(json.dumps(meta, indent=2))
        return meta

    hourly = (pd.concat(out, ignore_index=True)
                .rename(columns={"y": "actual", "lag1": "persistence", "lag7": "seasonal"})
                .groupby(["op_date", "he"], as_index=False)
                .agg(actual=("actual", "mean"), hybrid=("hybrid", "mean"),
                     persistence=("persistence", "mean"), seasonal=("seasonal", "mean"),
                     p10=("p10", "mean"), p90=("p90", "mean"),
                     atc_lo=("atc_lo", "first"), atc_hi=("atc_hi", "first")))
    hourly.round(2).to_csv(BACKTEST / "backtest_hourly.csv", index=False)

    daily = hourly.groupby("op_date").agg(
        actual=("actual", "mean"), hybrid=("hybrid", "mean"),
        persistence=("persistence", "mean"), seasonal=("seasonal", "mean"),
        atc_lo=("atc_lo", "first"), atc_hi=("atc_hi", "first")).reset_index()
    pk = (hourly[hourly["he"].isin(PEAK_HE)].groupby("op_date")
          [["actual", "hybrid", "persistence", "seasonal"]].mean().add_suffix("_peak").reset_index())
    daily = daily.merge(pk, on="op_date", how="left")
    daily["p10"] = daily["hybrid"] + daily["atc_lo"]
    daily["p90"] = daily["hybrid"] + daily["atc_hi"]
    daily["in_interval"] = ((daily["actual"] >= daily["p10"]) &
                            (daily["actual"] <= daily["p90"])).astype(int)
    mv = daily["hybrid"] - daily["persistence"]
    av = daily["actual"] - daily["persistence"]
    ok = (av.abs() >= 0.5) & (mv.abs() >= 0.01)
    daily["direction_hit"] = np.where(ok, (np.sign(mv) == np.sign(av)).astype(float), np.nan)
    daily.drop(columns=["atc_lo", "atc_hi"]).round(2).to_csv(
        BACKTEST / "backtest_daily.csv", index=False)

    def mae(a, b):
        return round(float((a - b).abs().mean()), 2)

    s = {
        "status": "ok", "n_days": int(len(daily)),
        "first_date": str(daily["op_date"].min()), "last_date": str(daily["op_date"].max()),
        "training_weather": "Open-Meteo previous-day forecasts",
        "mae_atc": {m: mae(daily[m], daily["actual"]) for m in ("hybrid", "persistence", "seasonal")},
        "mae_peak": {m: mae(daily[f"{m}_peak"], daily["actual_peak"])
                     for m in ("hybrid", "persistence", "seasonal")},
        "mae_hourly": {m: mae(hourly[m], hourly["actual"]) for m in ("hybrid", "persistence", "seasonal")},
        "bias_atc_hybrid": round(float((daily["hybrid"] - daily["actual"]).mean()), 2),
        "interval_coverage_pct": round(100 * float(daily["in_interval"].mean()), 1),
        "direction_hit_pct": (round(100 * float(np.nanmean(daily["direction_hit"])), 1)
                              if daily["direction_hit"].notna().any() else None),
        "days_beat_persistence": int(((daily["hybrid"] - daily["actual"]).abs()
                                      < (daily["persistence"] - daily["actual"]).abs()).sum()),
    }
    s["skill_vs_persistence_atc_pct"] = round(
        100 * (1 - s["mae_atc"]["hybrid"] / max(s["mae_atc"]["persistence"], 1e-9)), 1)
    s["skill_vs_persistence_hourly_pct"] = round(
        100 * (1 - s["mae_hourly"]["hybrid"] / max(s["mae_hourly"]["persistence"], 1e-9)), 1)
    meta.update(s)
    meta["feeds"] = sources.FEED
    (BACKTEST / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta
