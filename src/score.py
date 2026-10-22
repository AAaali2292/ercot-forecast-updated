"""Score day-ahead forecasts against the actual cleared day-ahead price.

Rules that make the track record worth showing:

1. Only LIVE calls count: generated for tomorrow, timestamped before the 10:00 CT
   market close. Anything else is shown but excluded from the headline numbers.
2. The actual is ERCOT's official DAM settlement price (MIS file), not a scrape.
3. Every model is scored against the same actual, beside two benchmarks the model has
   to beat to claim any skill: persistence (today's DAM, already public at 06:00) and
   seasonal-naive (same weekday last week). Persistence is a hard benchmark for a
   day-ahead price; a model that does not beat it is reported as not beating it.
4. Errors are forecast minus actual: positive means the call was too high.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import history
from .config import FORECASTS, ONPEAK_HE, OUTCOMES, PEAK_HE, SCORECARD

MODELS = ["hybrid", "stack", "persistence", "seasonal"]
COLS = (["target_date", "generated_at_ct", "saved_at_ct", "dam_close_ct", "live", "git_sha", "run_url",
         "primary_model", "primary_is_fallback", "training_weather_fallback",
         "actual_source", "actual_official", "n_hours", "n_filled", "status",
         "act_atc", "act_peak", "act_onpeak", "act_max",
         "fc_atc", "fc_peak", "fc_onpeak", "fc_atc_p10", "fc_atc_p90",
         "err_atc", "err_peak", "mae_hourly", "in_interval", "direction_hit"]
        + [f"{m}_{k}" for m in MODELS for k in ("atc", "err_atc", "err_peak", "mae_hourly")]
        + ["rt_atc", "da_minus_rt", "rt_coverage_pct"])


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def _avg(d: dict, hes) -> float | None:
    v = [d[h] for h in hes if h in d and d[h] is not None]
    return float(np.mean(v)) if len(v) == len(list(hes)) else None


def actual_by_he(dam: pd.DataFrame, day: dt.date) -> dict:
    a = dam[dam["op_date"] == day]
    return {int(h): float(v) for h, v in zip(a["he"], a["spp"])}


def is_live(fc: dict, target: dt.date) -> bool:
    """Recomputed from the timestamps; the file's own flag is not trusted."""
    try:
        close = dt.datetime.fromisoformat(fc["dam_close_ct"])
        gen = dt.datetime.fromisoformat(fc["generated_at_ct"])
        saved = dt.datetime.fromisoformat(fc.get("saved_at_ct") or fc["generated_at_ct"])
    except Exception:  # noqa: BLE001
        return False
    # Valid window: after DAM for T-1 is out (~13:30 CT on T-2) and before T's close.
    return bool(fc.get("live") and gen < close and saved < close
                and close.date() == target - dt.timedelta(days=1)
                and close - gen <= dt.timedelta(hours=24))


def score_one(target: str, dam: pd.DataFrame, sources_by_day: dict) -> dict | None:
    fc = _load(FORECASTS / f"{target}.json")
    if not fc or fc.get("schema", 1) < 2:
        return None
    d = dt.date.fromisoformat(target)
    act = actual_by_he(dam, d)
    if len(act) < 23:
        return None                                   # actual not published yet
    src = sources_by_day.get(d, "unknown")
    base = {"target_date": target, "generated_at_ct": fc.get("generated_at_ct"),
            "saved_at_ct": fc.get("saved_at_ct"), "dam_close_ct": fc.get("dam_close_ct"),
            "live": is_live(fc, d), "git_sha": fc.get("code_version") or fc.get("git_sha"),
            "run_url": fc.get("run_url"),
            "primary_model": fc.get("primary_model"),
            "primary_is_fallback": bool(fc.get("primary_is_fallback")),
            "training_weather_fallback": bool(fc.get("training_weather_fallback")),
            "actual_source": src, "actual_official": "ercot-mis" in src}
    hes_all = sorted(act)
    a_atc = float(np.mean([act[h] for h in hes_all]))
    a_peak = _avg(act, PEAK_HE)
    a_on = _avg(act, ONPEAK_HE)

    paths = {m: {} for m in MODELS}
    for row in fc.get("hourly", []):
        for m in MODELS:
            if row.get(m) is not None:
                paths[m][int(row["he"])] = float(row[m])

    # Every model is scored on the SAME hours: those the actual, the primary model and the
    # persistence benchmark all have. At most one hour may be missing (e.g. DST), recorded.
    prim_ = fc.get("primary_model")
    common = [h for h in hes_all if h in paths.get(prim_, {})
              and (h in paths["persistence"] or not paths["persistence"])]
    a_atc = float(np.mean([act[h] for h in common])) if common else a_atc

    def metrics(p: dict):
        hs = [h for h in common if h in p]
        if len(hs) < len(hes_all) - 1 or len(hs) < len(common):
            return None
        atc = float(np.mean([p[h] for h in hs]))
        a_atc_m = float(np.mean([act[h] for h in hs]))
        pk = _avg(p, PEAK_HE)
        return {"atc": round(atc, 2), "err_atc": round(atc - a_atc_m, 2),
                "err_peak": round(pk - a_peak, 2) if (pk is not None and a_peak is not None) else None,
                "mae_hourly": round(float(np.mean([abs(p[h] - act[h]) for h in hs])), 2),
                "onpeak": _avg(p, ONPEAK_HE), "peak": pk, "n": len(hs)}

    m_all = {m: metrics(p) for m, p in paths.items()}
    prim = fc.get("primary_model")
    pm = m_all.get(prim) if prim else None
    if pm is None:
        # Never drop a day silently: it stays in the scorecard, marked and explained.
        return {**base, "status": "unscorable: forecast missing more than one hour",
                "act_atc": round(a_atc, 2)}

    iv = fc.get("intervals") or {}
    lo, hi = iv.get("atc_p10"), iv.get("atc_p90")
    in_iv = int(lo <= a_atc <= hi) if (lo is not None and hi is not None
                                      and prim == "hybrid") else None
    pers = m_all.get("persistence")
    dir_hit = None
    if pers is not None:
        mv, av = pm["atc"] - pers["atc"], a_atc - pers["atc"]
        if abs(av) >= 0.50 and abs(mv) >= 0.01:      # ignore flat days: no direction to call
            dir_hit = int(np.sign(mv) == np.sign(av))

    oc = _load(OUTCOMES / f"{target}.json") or {}
    rt_atc = (oc.get("rt") or {}).get("atc")
    rt_cov = oc.get("coverage_pct")
    rt_ok = rt_atc is not None and rt_cov is not None and rt_cov >= 95.0

    row = {
        **base, "n_hours": pm["n"],
        "n_filled": len((fc.get("hybrid") or {}).get("hours_filled_from_persistence") or []),
        "status": "scored" if base["actual_official"] else "provisional: dashboard actual",
        "act_atc": round(a_atc, 2),
        "act_peak": round(a_peak, 2) if a_peak is not None else None,
        "act_onpeak": round(a_on, 2) if a_on is not None else None,
        "act_max": round(max(act.values()), 2),
        "fc_atc": pm["atc"], "fc_peak": round(pm["peak"], 2) if pm["peak"] is not None else None,
        "fc_onpeak": round(pm["onpeak"], 2) if pm["onpeak"] is not None else None,
        "fc_atc_p10": lo, "fc_atc_p90": hi,
        "err_atc": pm["err_atc"], "err_peak": pm["err_peak"], "mae_hourly": pm["mae_hourly"],
        "in_interval": in_iv, "direction_hit": dir_hit,
        "rt_atc": rt_atc if rt_ok else None,
        "da_minus_rt": round(a_atc - rt_atc, 2) if rt_ok else None,
        "rt_coverage_pct": rt_cov,
    }
    for m in MODELS:
        mm = m_all.get(m) or {}
        for k in ("atc", "err_atc", "err_peak", "mae_hourly"):
            row[f"{m}_{k}"] = mm.get(k)
    return row


def rebuild_scorecard() -> pd.DataFrame:
    full = history.load()
    dam = history.hub_hourly(full)
    src = (full[full["location"] == "HB_HUBAVG"].groupby("op_date")["source"]
           .agg(lambda s: "+".join(sorted(set(s)))).to_dict()) if len(full) else {}
    rows = [r for p in sorted(FORECASTS.glob("*.json"))
            if (r := score_one(p.stem, dam, src))]
    df = pd.DataFrame(rows, columns=COLS) if rows else pd.DataFrame(columns=COLS)
    df.to_csv(SCORECARD, index=False)
    return df


def summarise(df: pd.DataFrame, live_only: bool = True, window: int | None = None) -> dict:
    if not len(df):
        return {"n": 0}
    d = df[df["status"].astype(str) != ""] if "status" in df else df
    d = d[d["status"].astype(str).str.startswith(("scored", "provisional"))]
    if live_only:
        # Headline record: live calls scored against ERCOT's official settlement file.
        d = d[d["live"].astype(bool) & (d["status"] == "scored")]
    if window:
        d = d.tail(window)
    if not len(d):
        return {"n": 0}

    def mae(col):
        s = pd.to_numeric(d[col], errors="coerce").dropna()
        return round(float(s.abs().mean()), 2) if len(s) else None

    out = {"n": int(len(d)), "first_date": str(d["target_date"].iloc[0]),
           "last_date": str(d["target_date"].iloc[-1]),
           "n_fallback": int(d["primary_is_fallback"].astype(bool).sum()),
           "n_training_weather_fallback": int(d["training_weather_fallback"].astype(bool).sum())}
    allrows = df if "status" in df else pd.DataFrame()
    if len(allrows):
        out["n_provisional"] = int((allrows["status"] == "provisional: dashboard actual").sum())
        out["n_unscorable"] = int(allrows["status"].astype(str).str.startswith("unscorable").sum())
    for m in MODELS:
        out[f"{m}_mae_atc"] = mae(f"{m}_err_atc")
        out[f"{m}_mae_peak"] = mae(f"{m}_err_peak")
        out[f"{m}_mae_hourly"] = (round(float(pd.to_numeric(d[f"{m}_mae_hourly"],
                                                            errors="coerce").dropna().mean()), 2)
                                  if pd.to_numeric(d[f"{m}_mae_hourly"], errors="coerce").notna().any()
                                  else None)
        out[f"{m}_n"] = int(pd.to_numeric(d[f"{m}_err_atc"], errors="coerce").notna().sum())
    out["primary_mae_atc"] = mae("err_atc")
    pr = d[["err_atc", "persistence_err_atc"]].apply(pd.to_numeric, errors="coerce").dropna()
    out["paired_primary_mae_atc"] = round(float(pr["err_atc"].abs().mean()), 2) if len(pr) else None
    out["paired_persistence_mae_atc"] = (round(float(pr["persistence_err_atc"].abs().mean()), 2)
                                         if len(pr) else None)
    out["primary_mae_hourly"] = round(float(pd.to_numeric(d["mae_hourly"]).mean()), 2)
    out["primary_bias_atc"] = round(float(pd.to_numeric(d["err_atc"]).mean()), 2)
    out["primary_rmse_atc"] = round(float(np.sqrt((pd.to_numeric(d["err_atc"]) ** 2).mean())), 2)

    # Skill vs persistence on the SAME days (pairs where both exist).
    pair = d[["err_atc", "persistence_err_atc", "mae_hourly", "persistence_mae_hourly"]].apply(
        pd.to_numeric, errors="coerce").dropna()
    if len(pair):
        out["skill_vs_persistence_atc_pct"] = round(
            100 * (1 - pair["err_atc"].abs().mean() / max(pair["persistence_err_atc"].abs().mean(), 1e-9)), 1)
        out["skill_vs_persistence_hourly_pct"] = round(
            100 * (1 - pair["mae_hourly"].mean() / max(pair["persistence_mae_hourly"].mean(), 1e-9)), 1)
        out["days_beat_persistence"] = int((pair["err_atc"].abs() < pair["persistence_err_atc"].abs()).sum())
        out["paired_days"] = int(len(pair))
    iv = pd.to_numeric(d["in_interval"], errors="coerce").dropna()
    out["interval_coverage_pct"] = round(100 * float(iv.mean()), 1) if len(iv) else None
    dh = pd.to_numeric(d["direction_hit"], errors="coerce").dropna()
    out["direction_hit_pct"] = round(100 * float(dh.mean()), 1) if len(dh) else None
    out["direction_n"] = int(len(dh))
    prem = pd.to_numeric(d["da_minus_rt"], errors="coerce").dropna()
    if len(prem):
        out.update({"da_rt_mean": round(float(prem.mean()), 2),
                    "da_rt_median": round(float(prem.median()), 2),
                    "da_rt_days_da_higher_pct": round(100 * float((prem > 0).mean()), 1),
                    "da_rt_n": int(len(prem))})
    return out


def load_scorecard() -> pd.DataFrame:
    if SCORECARD.exists():
        try:
            return pd.read_csv(SCORECARD)
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame(columns=COLS)
