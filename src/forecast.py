"""Produce tomorrow's day-ahead price forecast.

Runs at ~06:00 CT. ERCOT's day-ahead market for tomorrow closes at 10:00 CT today, so
everything written here is a genuine ex-ante call. The file records the time it was
generated and whether that was before the close; only pre-close calls are scored.

Four hourly price paths are produced for tomorrow (HE1..HE24, ERCOT Hub Average):

  hybrid       PRIMARY. Market-anchored statistical model on forecast fundamentals.
  stack        The fundamental merit-order + ORDC model (the physics view).
  persistence  Benchmark: today's cleared DAM price, same hour.
  seasonal     Benchmark: DAM price for the same weekday last week, same hour.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
from typing import Optional

import numpy as np
import pandas as pd

from . import history, sources
from .config import (CFG, CT, FORECASTS, ONPEAK_HE, PEAK_HE, Config, config_dict)
from .hybrid import (HybridModel, build_rows, interval_from_residuals, rolling_residuals,
                     weather_by_hour, HE_BLOCK)
from .models import (LoadModel, apply_hourly_bias, build_load_history, calibrate_solar,
                     calibrate_wind, hourly_bias, solar_forecast, wind_forecast)
from .stack import (StackArrays, build_stack, clear, implied_heat_rate,
                    mmbtu_h_to_bcfd, spark_spread, synth_fleet)
from .timeutil import dam_close

SCHEMA = 2


def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def _block_avg(by_he: dict, hes) -> Optional[float]:
    v = [by_he[h] for h in hes if h in by_he and by_he[h] is not None
         and np.isfinite(by_he[h])]
    return round(float(np.mean(v)), 2) if len(v) else None


def summarise_path(by_he: dict) -> dict:
    return {"atc": _block_avg(by_he, range(1, 25)), "peak": _block_avg(by_he, PEAK_HE),
            "onpeak": _block_avg(by_he, ONPEAK_HE),
            "max": round(float(max(by_he.values())), 2) if by_he else None}


# ------------------------------------------------------------------ stack model
def stack_forecast(target: dt.date, wx: pd.DataFrame, cfg: Config) -> dict:
    """The fundamental view. Returns hourly price by HE plus the physical drivers."""
    mix = sources.fetch_fuel_mix()
    gen, cap = mix["gen"], mix["capacity"]
    eia = sources.fetch_eia930_demand()
    gas = sources.fetch_gas_price()
    if gas:
        cfg = Config(**{**cfg.__dict__, "henry_hub": float(gas["price"])})
    if not cap:
        raise RuntimeError("no ERCOT installed capacity (fuel-mix feed failed)")

    day = wx[wx["op_date"] == target].copy()
    if day["he"].nunique() < 23:
        raise RuntimeError(f"weather forecast covers only {day['he'].nunique()} hours of {target}")

    hist, hist_src = build_load_history(eia, wx, gen)
    lm = LoadModel().fit(hist, hist_src)
    day["load_mw"] = lm.predict(day)
    sources.log("Load model", "OK" if lm.n >= 720 else "WARN",
                f"fit on {lm.n} hours from {lm.source}; in-sample MAPE {lm.mape:.2f}%"
                + ("" if lm.n >= 720 else " <- thin training window"), lm.n)

    wind_cap = float(cap.get("wind", 0.0))
    solar_cap = float(cap.get("solar", 0.0))
    k_wind = calibrate_wind(wx, gen, wind_cap)
    k_solar = calibrate_solar(wx, gen, solar_cap)
    kh_wind = hourly_bias(wx, gen, wind_cap, "wind")
    kh_solar = hourly_bias(wx, gen, solar_cap, "solar")
    hrs = day["ts"].dt.hour.to_numpy()
    day["wind_mw"] = apply_hourly_bias(wind_forecast(day, wind_cap), hrs, kh_wind, k_wind)
    day["solar_mw"] = apply_hourly_bias(solar_forecast(day, solar_cap), hrs, kh_solar, k_solar)
    day["net_load_mw"] = (day["load_mw"] - day["wind_mw"] - day["solar_mw"]).clip(lower=0)
    sources.log("Renewable model", "OK",
                f"wind k={k_wind:.2f}, solar k={k_solar:.2f}; installed "
                f"{wind_cap:,.0f} MW wind / {solar_cap:,.0f} MW solar")

    fleet = synth_fleet(cap)
    rng = np.random.default_rng(int(target.strftime("%Y%m%d")))
    base_out = cfg.forced_outage_rate + cfg.planned_outage_rate
    grid = np.clip(np.linspace(base_out - 3 * cfg.outage_sigma,
                               base_out + 3 * cfg.outage_sigma, 13), 0.0, 0.6)
    stacks = [StackArrays(build_stack(fleet, cfg, outage=float(o))) for o in grid]
    load_v, wind_v, solar_v = (day[c].to_numpy(float) for c in ("load_mw", "wind_mw", "solar_mw"))
    n_h = len(day)
    sims = np.zeros((cfg.n_sims, n_h))
    for s in range(cfg.n_sims):
        l = load_v * (1 + rng.normal(0, cfg.load_sigma_frac * 0.7)
                      + rng.normal(0, cfg.load_sigma_frac * 0.5, n_h))
        w = np.clip(wind_v * (1 + rng.normal(0, cfg.wind_sigma_frac * 0.8)
                              + rng.normal(0, cfg.wind_sigma_frac * 0.6, n_h)), 0, None)
        so = np.clip(solar_v * (1 + rng.normal(0, cfg.solar_sigma_frac * 0.7)
                                + rng.normal(0, cfg.solar_sigma_frac * 0.6, n_h)), 0, None)
        o = float(np.clip(rng.normal(base_out, cfg.outage_sigma), 0.0, 0.6))
        sims[s] = clear(np.clip(l - w - so, 0, None),
                        stacks[int(np.argmin(np.abs(grid - o)))], cfg)["price"]

    central = clear(day["net_load_mw"].to_numpy(float), StackArrays(build_stack(fleet, cfg)), cfg)
    day["price"] = central["price"]
    day["marginal_cat"] = central["marginal_cat"]
    day["gas_burn_bcfd"] = mmbtu_h_to_bcfd(central["gas_burn_mmbtu_h"])
    day["sim_median"] = np.median(sims, axis=0)

    by = day.groupby("he").agg(price=("sim_median", "mean"), load=("load_mw", "mean"),
                               wind=("wind_mw", "mean"), solar=("solar_mw", "mean"),
                               net_load=("net_load_mw", "mean"), temp_f=("temp_f", "mean"))
    pk = day["he"].isin(PEAK_HE).to_numpy()
    sim_pk = sims[:, pk].mean(axis=1)
    gas_d = cfg.henry_hub + cfg.hsc_basis
    peak_px = float(np.median(sim_pk))
    return {
        "by_he": {int(h): round(float(v), 2) for h, v in by["price"].items()},
        "drivers": by.round(1).reset_index().to_dict(orient="records"),
        "summary": {
            "peak_p10": round(float(np.percentile(sim_pk, 10)), 2),
            "peak_p90": round(float(np.percentile(sim_pk, 90)), 2),
            "p_scarcity": round(float((sims.max(axis=1) > cfg.scarcity_threshold).mean()), 4),
            "peak_load_mw": round(float(day["load_mw"].max()), 0),
            "peak_net_load_mw": round(float(day["net_load_mw"].max()), 0),
            "wind_peak_block_mw": round(float(day.loc[pk, "wind_mw"].mean()), 0),
            "solar_peak_block_mw": round(float(day.loc[pk, "solar_mw"].mean()), 0),
            "gas_burn_bcfd": round(float(day["gas_burn_bcfd"].mean()), 2),
            "implied_heat_rate": round(float(implied_heat_rate(peak_px, gas_d)), 2),
            "spark_spread_7": round(float(spark_spread(peak_px, gas_d, 7.0, cfg.vom_ccgt)), 2),
            "marginal_unit_peak": str(pd.Series(day.loc[pk, "marginal_cat"]).mode().iloc[0]),
            "max_temp_f": round(float(day["temp_f"].max()), 1),
        },
        "inputs": {
            "gas_price": round(cfg.henry_hub, 3),
            "gas_source": (gas or {}).get("source", "CONFIG DEFAULT (live gas feed failed)"),
            "gas_as_of": (gas or {}).get("as_of"),
            "installed_mw": {k: round(v, 0) for k, v in cap.items()},
            "load_model_source": lm.source, "load_model_hours": lm.n,
            "load_model_mape_pct": round(lm.mape, 3),
            "wind_bias_k": round(k_wind, 3), "solar_bias_k": round(k_solar, 3),
            "n_sims": cfg.n_sims,
        },
    }


# ----------------------------------------------------------------- hybrid model
def hybrid_forecast(target: dt.date, today: dt.date, dam: pd.DataFrame,
                    wx_live: pd.DataFrame, cfg: Config) -> dict:
    """`today` is the as-of day (target - 1); `dam` must hold nothing dated after it."""
    start = today - dt.timedelta(days=cfg.hybrid_train_days + cfg.hybrid_cv_days + 10)
    # +1 day: the archive is in UTC days, and today's Central evening ends after
    # midnight UTC. Without it the last training day silently loses HE19-HE24.
    wx_train = sources.fetch_weather_previous_day1(start, today + dt.timedelta(days=1))
    train_src = "Open-Meteo previous-day forecasts (no look-ahead)"
    fallback = False
    if not len(wx_train):
        fallback = True
        # Fallback, stated loudly: analysis weather for past days. Slightly optimistic
        # training inputs; the live input is still a genuine forecast.
        wx_train = wx_live
        train_src = "FALLBACK: Open-Meteo recent analysis (previous-runs archive unavailable)"
    wxh_train = weather_by_hour(wx_train)
    wxh_live = weather_by_hour(wx_live)

    train_days = sorted(d for d in dam["op_date"].unique() if start < d <= today)
    rows = build_rows(train_days, dam, wxh_train)
    if not len(rows):
        raise RuntimeError("no hybrid training rows (DAM history or weather archive empty)")

    model = HybridModel(cfg).fit(rows, as_of=today)
    res = rolling_residuals(rows, as_of=today, n_days=cfg.hybrid_cv_days, cfg=cfg)
    iv = interval_from_residuals(res)

    # Live row. Training rows describe day T and day T-1 both with the forecast issued
    # ~24h ahead, so the change terms (d_dd, d_wind_cf, d_solar_cf) are "forecast minus
    # forecast". To keep that meaning live: tomorrow from the live run, today from the
    # same previous-day archive. Only if the archive failed are both taken from live.
    base_today = wxh_train[wxh_train["op_date"] == today]
    if not fallback and base_today["he"].nunique() >= 23:
        wxh_row = pd.concat([wxh_live[wxh_live["op_date"] == target], base_today])
        change_basis = "tomorrow: live NWP; today: previous-day archive (as in training)"
    else:
        wxh_row = wxh_live
        change_basis = "tomorrow and today: live NWP (archive unavailable)"
    live = build_rows([target], dam, wxh_row)
    if live["he"].nunique() < 23:
        raise RuntimeError(f"hybrid live row incomplete: {live['he'].nunique()} hours")
    clamps = model.clamp_report(live)
    live = live.assign(pred=model.predict(live))
    by = live.groupby("he")["pred"].mean()
    by_he = {int(h): round(float(v), 2) for h, v in by.items()}

    # A missing input hour (e.g. HE3 after a spring-forward day) must not silently knock
    # the whole day out of the record: fill it with persistence and say so.
    p_known = dam.set_index(["op_date", "he"])["spp"]
    filled = []
    for h in range(1, 25):
        if h not in by_he and (today, h) in p_known.index:
            by_he[h] = round(float(p_known[(today, h)]), 2)
            filled.append(h)
    by_he = dict(sorted(by_he.items()))
    lo = {h: round(v + iv["hourly"][HE_BLOCK[h]][0], 2) for h, v in by_he.items()} if iv else {}
    hi = {h: round(v + iv["hourly"][HE_BLOCK[h]][1], 2) for h, v in by_he.items()} if iv else {}
    s = summarise_path(by_he)

    cv = {}
    if len(res):
        e = res["y"] - res["pred"]
        d = res.groupby("op_date")[["y", "pred"]].mean()
        cv = {"days": int(len(d)), "hourly_mae": round(float(e.abs().mean()), 2),
              "daily_avg_mae": round(float((d["y"] - d["pred"]).abs().mean()), 2)}
    if clamps:
        sources.log("Hybrid inputs outside training range", "WARN",
                    "; ".join(f"{c['block']}:{c['feature']} x{c['hours']}" for c in clamps))
    sources.log("Hybrid model", "OK",
                f"trained on {model.n_days} days through {today}; training weather: "
                f"{train_src}; rolling CV {cv}")
    return {
        "by_he": by_he, "p10_by_he": lo, "p90_by_he": hi,
        "atc_p10": round(s["atc"] + iv["daily"][0], 2) if iv else None,
        "atc_p90": round(s["atc"] + iv["daily"][1], 2) if iv else None,
        "peak_p10": round(s["peak"] + iv["peak"][0], 2) if iv and iv.get("peak") else None,
        "peak_p90": round(s["peak"] + iv["peak"][1], 2) if iv and iv.get("peak") else None,
        "train_days": model.n_days, "training_weather": train_src,
        "training_weather_fallback": fallback, "change_basis": change_basis,
        "hours_filled_from_persistence": filled,
        "rolling_cv": cv, "coefficients": model.describe(), "clamped_inputs": clamps,
        "features_live": live.drop(columns=["y"]).round(4).astype({"op_date": str})
                             .to_dict(orient="records"),
    }


# --------------------------------------------------------------------- assemble
def make_forecast(target_date: Optional[dt.date] = None, cfg: Config = CFG) -> dict:
    now = dt.datetime.now(CT)
    today = now.date()
    target = target_date or (today + dt.timedelta(days=1))
    if target != today + dt.timedelta(days=1):
        # Anything other than "tomorrow, as of now" is not a live forecast. It is built
        # as of target-1 so that a past date is at least out-of-sample.
        sources.log("Target date", "WARN", f"{target} is not tomorrow; marked non-live")
    as_of = target - dt.timedelta(days=1)

    # 1. What the market has already told us: official DAM history + today's dashboard.
    history.refresh(days_back=10, include_tomorrow=False)
    px = sources.fetch_prices()
    history.ingest_dashboard(px.get("dam"))
    dam = history.hub_hourly()
    known = dam[dam["op_date"] <= min(today, as_of)]   # nothing dated after the as-of day

    # 2. Weather.
    wx = sources.fetch_weather_forecast()
    if not len(wx):
        raise RuntimeError("no live weather forecast: refusing to forecast")

    paths, errors = {}, {}
    hyb = stk = None
    try:
        hyb = hybrid_forecast(target, as_of, known, wx, cfg)
        paths["hybrid"] = hyb["by_he"]
    except Exception as e:  # noqa: BLE001
        errors["hybrid"] = str(e)
        sources.log("Hybrid model", "FAIL", e)
    try:
        stk = stack_forecast(target, wx, cfg)
        paths["stack"] = stk["by_he"]
    except Exception as e:  # noqa: BLE001
        errors["stack"] = str(e)
        sources.log("Stack model", "FAIL", e)

    k = known.set_index(["op_date", "he"])["spp"]
    d1, d7 = target - dt.timedelta(days=1), target - dt.timedelta(days=7)
    pers = {h: round(float(k[(d1, h)]), 2) for h in range(1, 25) if (d1, h) in k.index}
    seas = {h: round(float(k[(d7, h)]), 2) for h in range(1, 25) if (d7, h) in k.index}
    if len(pers) >= 23:
        paths["persistence"] = pers
    if len(seas) >= 23:
        paths["seasonal"] = seas

    primary = "hybrid" if "hybrid" in paths else ("stack" if "stack" in paths else None)
    if primary is None:
        raise RuntimeError(f"no model produced a forecast: {errors}")

    close = dam_close(target)
    fc = {
        "schema": SCHEMA,
        "target_date": target.isoformat(),
        "market": "ERCOT day-ahead market, Hub Average (HB_HUBAVG) settlement point price",
        "generated_at_ct": now.isoformat(timespec="seconds"),
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "dam_close_ct": close.isoformat(),
        "pre_close": bool(now < close),
        "live": bool(target == today + dt.timedelta(days=1) and now < close),
        "code_version": _git_sha(),
        "run_url": os.environ.get("GITHUB_RUN_URL"),
        "primary_model": primary,
        "primary_is_fallback": primary != "hybrid",
        "training_weather_fallback": bool((hyb or {}).get("training_weather_fallback")),
        "summary": {name: summarise_path(p) for name, p in paths.items()},
        "hourly": [
            {"he": h, **{name: p.get(h) for name, p in paths.items()},
             "hybrid_p10": (hyb or {}).get("p10_by_he", {}).get(h),
             "hybrid_p90": (hyb or {}).get("p90_by_he", {}).get(h)}
            for h in range(1, 25)
        ],
        "intervals": {k2: (hyb or {}).get(k2) for k2 in
                      ("atc_p10", "atc_p90", "peak_p10", "peak_p90")},
        "hybrid": {k2: v for k2, v in (hyb or {}).items()
                   if k2 not in ("by_he", "p10_by_he", "p90_by_he")},
        "stack": {k2: v for k2, v in (stk or {}).items() if k2 != "by_he"},
        "model_errors": errors,
        "dam_known_through": str(known["op_date"].max()) if len(known) else None,
        "assumptions": config_dict(),
        "feeds": sources.FEED,
    }
    return fc


def save_forecast(fc: dict, force: bool = False) -> tuple[str, bool]:
    """Write forecasts/<target>.json. Returns (path, written).

    Integrity rule: once a live (pre-close) forecast exists for a day it is THE call. A
    later run may only replace it if it is also live and upgrades a fallback forecast
    to the primary model. Post-close runs never touch a live file.

    The clock is re-read here, at the moment of writing: a run that started before the
    close but finished after it is stamped non-live.
    """
    now = dt.datetime.now(CT)
    close = dt.datetime.fromisoformat(fc["dam_close_ct"]) if fc.get("dam_close_ct") else None
    if close is not None and now >= close:
        fc["pre_close"] = False
        fc["live"] = False
    fc["saved_at_ct"] = now.isoformat(timespec="seconds")
    p = FORECASTS / f"{fc['target_date']}.json"
    if p.exists():
        try:
            old = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            old = {}
        # Hard rule, even with force: a post-close run never replaces a pre-close call.
        if old.get("live") and not fc.get("live"):
            return str(p), False
        if old.get("live") and not force:
            upgrade = (fc.get("live") and old.get("primary_is_fallback")
                       and not fc.get("primary_is_fallback"))
            if not upgrade:
                return str(p), False
    p.write_text(json.dumps(fc, indent=2, default=str))
    return str(p), True
