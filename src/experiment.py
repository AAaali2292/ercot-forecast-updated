"""Model v2 experiment: does the hybrid forecast cold days better if it is shown winter?

SIMULATED - this never touches the live record or the production model.

The problem
-----------
The production hybrid (v1) trains on the last 120 days. In November that window is
summer and early autumn, so the model has hardly seen a cold day, and heating and
cooling share one "degree-hours" input, so it cannot learn that a cold morning and a
hot afternoon move prices differently.

Candidates (all fitted on exactly the same rows, scored on exactly the same hours)
--------------------------------------------------------------------------------
  v1   production model, unchanged (src.hybrid.HybridModel)
  v2a  heating and cooling as separate inputs (hdd, cdd and their day-on-day change)
  v2b  v2a + a "winter bank": cold days from the previous ~14 months that fall outside
       the 120-day window, added to training at a fixed weight of 0.25
  v2c  as v2b with bank weight 0.5

Information set is identical to the live model: a row for day T uses DAM prices up to
T-1 and the weather forecast for T as it stood ~24h earlier (Open-Meteo previous runs).

Decision rule (written before the results are seen)
---------------------------------------------------
Pick the candidate with the lowest daily-average MAE on COLD days in the winter-onset
window (1 Nov 2025 - 31 Jan 2026), subject to: its daily MAE over the most recent 60
days is no worse than v1's by more than 2%. If no candidate passes, keep v1.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import history, sources
from .config import CFG, CT, DATA, HUB, PEAK_HE, ROOT
from .hybrid import BLOCKS, HE_BLOCK, HybridModel, build_rows, huber_ridge
from .models import solar_forecast, wind_forecast

OUT = ROOT / "backtest" / "v2_compare"
ARCHIVE = DATA / "dam_archive_hub.csv"          # HB_HUBAVG only, 2024-2025, for this test

V2_FEATURES = ["lag1", "lag1_mean", "lag7", "hdd", "cdd", "d_hdd", "d_cdd",
               "wind_cf", "solar_cf", "d_wind_cf", "d_solar_cf", "wknd_t", "wknd_d"]
V2_CLAMPED = ["lag1", "lag1_mean", "lag7", "hdd", "cdd", "d_hdd", "d_cdd",
              "wind_cf", "solar_cf", "d_wind_cf", "d_solar_cf"]
COLD_DAY_HDD = 5.0            # load-weighted mean degree-hours below 65F (~ a 60F day)
BANK_LOOKBACK_DAYS = 430
VARIANTS = {"v2a": 0.0, "v2b": 0.25, "v2c": 0.5}

WINDOWS = {
    "winter_onset": ("2025-11-01", "2026-01-31"),
    "late_winter": ("2026-02-01", "2026-03-31"),
    "spring_summer": ("2026-04-01", "2026-09-30"),
    "live_period": ("2026-10-24", "2099-12-31"),
}


# ------------------------------------------------------------------- data
def load_dam() -> pd.DataFrame:
    """Hub hourly DAM: production store (2026+) plus an archive of 2024-2025."""
    cur = history.hub_hourly()
    if ARCHIVE.exists():
        arc = pd.read_csv(ARCHIVE, parse_dates=["op_date"])
        arc["op_date"] = arc["op_date"].dt.date
    else:
        parts = []
        for y in (2024, 2025):
            try:
                a = history.fetch_mis_annual(y)
                parts.append(history.hub_hourly(a))
                sources.log(f"ERCOT MIS annual {y}", "OK", f"{len(parts[-1])} hub-hours")
            except Exception as e:  # noqa: BLE001
                sources.log(f"ERCOT MIS annual {y}", "FAIL", e)
        arc = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cur.columns)
        if len(arc):
            arc[["op_date", "he", "spp"]].to_csv(ARCHIVE, index=False)
    arc = arc[~arc["op_date"].isin(set(cur["op_date"]))]
    dam = pd.concat([arc[["op_date", "he", "spp"]], cur[["op_date", "he", "spp"]]],
                    ignore_index=True)
    return dam.drop_duplicates(["op_date", "he"], keep="last").sort_values(["op_date", "he"])


def load_weather(start: dt.date, end: dt.date) -> pd.DataFrame:
    """Previous-day forecasts in ~quarterly chunks (one long request is fragile)."""
    parts, d = [], start
    while d <= end:
        e = min(d + dt.timedelta(days=89), end)
        w = sources.fetch_weather_previous_day1(d, e)
        if len(w):
            parts.append(w)
        d = e + dt.timedelta(days=1)
    if not parts:
        return pd.DataFrame()
    w = pd.concat(parts, ignore_index=True).drop_duplicates("ts", keep="last")
    return w.dropna(subset=["temp_f", "wind_ms", "ghi"])


def weather_by_hour_v2(wx: pd.DataFrame) -> pd.DataFrame:
    w = wx.copy()
    w["dd"] = w["cdd"] + w["hdd"]
    w["wind_cf"] = wind_forecast(w, 1.0)
    w["solar_cf"] = solar_forecast(w, 1.0)
    return (w.groupby(["op_date", "he"], as_index=False)
             [["dd", "hdd", "cdd", "wind_cf", "solar_cf"]].mean())


def add_v2_columns(rows: pd.DataFrame, wxh: pd.DataFrame) -> pd.DataFrame:
    w = wxh.set_index(["op_date", "he"])[["hdd", "cdd"]]
    t = w.reindex(pd.MultiIndex.from_arrays([rows["op_date"], rows["he"]])).to_numpy()
    prev = [d - dt.timedelta(days=1) for d in rows["op_date"]]
    p = w.reindex(pd.MultiIndex.from_arrays([prev, rows["he"]])).to_numpy()
    out = rows.copy()
    out["hdd"], out["cdd"] = t[:, 0], t[:, 1]
    out["d_hdd"], out["d_cdd"] = t[:, 0] - p[:, 0], t[:, 1] - p[:, 1]
    return out.dropna(subset=["hdd", "cdd", "d_hdd", "d_cdd"])


# ------------------------------------------------------------------ model
class V2Model:
    """Same estimator as v1 (block-wise Huber ridge, 30-day half-life), v2 inputs,
    plus an optional fixed-weight winter bank."""

    def __init__(self, bank_weight: float, cold_days: set, cfg=CFG):
        self.bw, self.cold, self.cfg, self.fits = bank_weight, cold_days, cfg, {}
        self.n_bank_days = 0

    @staticmethod
    def _design(df, block):
        cols = [df[c].to_numpy(float) for c in V2_FEATURES]
        names = list(V2_FEATURES)
        for he in BLOCKS[block][1:]:
            cols.append((df["he"].to_numpy() == he).astype(float))
            names.append(f"he{he}")
        return np.column_stack(cols), names

    def fit(self, rows, as_of):
        lo = as_of - dt.timedelta(days=self.cfg.hybrid_train_days)
        main = rows[(rows["op_date"] <= as_of) & (rows["op_date"] > lo)].dropna(subset=["y"])
        age = np.array([(as_of - d).days for d in main["op_date"]], float)
        w_main = 0.5 ** (age / self.cfg.hybrid_halflife_days)
        tr, wts = main, w_main
        self.n_bank_days = 0
        if self.bw > 0:
            blo = as_of - dt.timedelta(days=BANK_LOOKBACK_DAYS)
            bank = rows[(rows["op_date"] <= lo) & (rows["op_date"] > blo)
                        & rows["op_date"].isin(self.cold)].dropna(subset=["y"])
            if len(bank):
                self.n_bank_days = int(bank["op_date"].nunique())
                tr = pd.concat([main, bank], ignore_index=True)
                wts = np.concatenate([w_main, np.full(len(bank), self.bw)])
        if main["op_date"].nunique() < self.cfg.hybrid_min_train_days:
            raise ValueError("not enough training days")
        self.fits = {}
        for b in BLOCKS:
            m = (tr["block"] == b).to_numpy()
            X, names = self._design(tr[m], b)
            mu, sd, beta, _ = huber_ridge(X, tr.loc[m, "y"].to_numpy(float), wts[m],
                                          self.cfg.hybrid_ridge, self.cfg.hybrid_huber_k)
            xlo, xhi = X.min(axis=0), X.max(axis=0)
            free = [i for i, n in enumerate(names) if n not in V2_CLAMPED]
            xlo[free], xhi[free] = -np.inf, np.inf
            self.fits[b] = (mu, sd, beta, xlo, xhi)
        return self

    def predict(self, rows):
        out = np.full(len(rows), np.nan)
        for b, (mu, sd, beta, xlo, xhi) in self.fits.items():
            m = (rows["block"] == b).to_numpy()
            if m.any():
                X, _ = self._design(rows[m], b)
                Z = (np.clip(X, xlo, xhi) - mu) / sd
                out[m] = beta[0] + Z @ beta[1:]
        return np.clip(out, self.cfg.price_floor, self.cfg.voll)


# ---------------------------------------------------------------- scoring
def _metrics(h: pd.DataFrame, models: list[str]) -> dict:
    d = h.groupby("op_date")[["actual", "persistence"] + models].mean()
    pk = h[h["he"].isin(PEAK_HE)].groupby("op_date")[["actual", "persistence"] + models].mean()
    res = {"n_days": int(len(d))}
    for m in ["persistence"] + models:
        mae = float((d[m] - d["actual"]).abs().mean())
        res[m] = {
            "mae_atc": round(mae, 2),
            "mae_hourly": round(float((h[m] - h["actual"]).abs().mean()), 2),
            "mae_peak": round(float((pk[m] - pk["actual"]).abs().mean()), 2),
            "bias_atc": round(float((d[m] - d["actual"]).mean()), 2),
        }
        if m != "persistence":
            mv, av = d[m] - d["persistence"], d["actual"] - d["persistence"]
            ok = (av.abs() >= 0.5) & (mv.abs() >= 0.01)
            res[m]["direction_pct"] = (round(100 * float((np.sign(mv[ok]) == np.sign(av[ok])).mean()), 1)
                                       if ok.any() else None)
            res[m]["skill_vs_persistence_pct"] = round(
                100 * (1 - mae / max(res["persistence"]["mae_atc"], 1e-9)), 1)
    return res


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(CT)
    meta = {"generated_at_ct": now.isoformat(timespec="seconds"),
            "label": "SIMULATED - model v2 experiment, not part of the live record",
            "method": __doc__.strip()}
    dam = load_dam()
    full = history.complete_days(dam)
    first_test = dt.date(2025, 11, 1)
    test_days = sorted(d for d in full if first_test <= d < now.date())
    if not test_days:
        meta.update(status="unavailable", reason="no DAM history for the test period")
        (OUT / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta

    wstart = first_test - dt.timedelta(days=BANK_LOOKBACK_DAYS + 10)
    wx = load_weather(wstart, test_days[-1])
    if not len(wx):
        meta.update(status="unavailable", reason="previous-runs weather archive unavailable")
        (OUT / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
        return meta
    wxh = weather_by_hour_v2(wx)
    days = sorted(d for d in dam["op_date"].unique() if d >= wstart)
    rows = add_v2_columns(build_rows(days, dam, wxh[["op_date", "he", "dd", "wind_cf", "solar_cf"]]), wxh)
    day_hdd = wxh.groupby("op_date")["hdd"].mean()
    cold = set(day_hdd[day_hdd >= COLD_DAY_HDD].index)

    meta["data"] = {
        "dam_first": str(min(dam["op_date"])), "dam_last": str(max(dam["op_date"])),
        "weather_first": str(min(wxh["op_date"])), "weather_last": str(max(wxh["op_date"])),
        "rows": int(len(rows)), "cold_days_total": len(cold), "test_days": len(test_days),
    }

    out, bank_used = [], []
    for t in test_days:
        r = rows[rows["op_date"] == t]
        if r["he"].nunique() < 23:
            continue
        as_of = t - dt.timedelta(days=1)
        try:
            v1 = HybridModel(CFG).fit(rows, as_of)
            preds = {"v1": v1.predict(r)}
            for name, bw in VARIANTS.items():
                m = V2Model(bw, cold).fit(rows, as_of)
                preds[name] = m.predict(r)
                if name == "v2b":
                    bank_used.append(m.n_bank_days)
        except ValueError:
            continue
        o = r[["op_date", "he", "y", "lag1"]].rename(columns={"y": "actual", "lag1": "persistence"})
        for k, v in preds.items():
            o[k] = v
        o["cold_day"] = t in cold
        out.append(o)

    h = pd.concat(out, ignore_index=True)
    h.round(2).to_csv(OUT / "hourly.csv", index=False)
    models = ["v1"] + list(VARIANTS)
    windows = {}
    last = max(h["op_date"])
    spans = dict(WINDOWS)
    spans["recent_60"] = (str(last - dt.timedelta(days=59)), str(last))
    for name, (a, b) in spans.items():
        a, b = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
        sub = h[(h["op_date"] >= a) & (h["op_date"] <= b)]
        if len(sub):
            windows[name] = {"all_days": _metrics(sub, models)}
            cs = sub[sub["cold_day"]]
            if cs["op_date"].nunique() >= 3:
                windows[name]["cold_days"] = _metrics(cs, models)
    windows["all_test_days"] = {"all_days": _metrics(h, models),
                                "cold_days": _metrics(h[h["cold_day"]], models)}

    # Decision rule (see module docstring).
    wo = windows.get("winter_onset", {}).get("cold_days") or windows.get("winter_onset", {}).get("all_days")
    rc = windows["recent_60"]["all_days"]
    passing = [v for v in VARIANTS if rc[v]["mae_atc"] <= 1.02 * rc["v1"]["mae_atc"]]
    choice = "v1"
    if wo and passing:
        best = min(passing, key=lambda v: wo[v]["mae_atc"])
        if wo[best]["mae_atc"] < wo["v1"]["mae_atc"]:
            choice = best
    meta.update(status="ok", windows=windows, decision={
        "chosen": choice, "passing_recent_check": passing,
        "rule": "lowest winter-onset cold-day ATC MAE among variants whose recent-60-day "
                "ATC MAE is within 2% of v1; keep v1 otherwise",
        "v2b_bank_days_median": float(np.median(bank_used)) if bank_used else 0.0})
    meta["feeds"] = sources.FEED
    (OUT / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta["decision"], indent=2))
    return meta


if __name__ == "__main__":
    run()
