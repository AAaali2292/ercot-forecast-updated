"""Hybrid day-ahead price model - the PRIMARY forecast.

Why a hybrid
------------
A pure merit-order stack gets the *physics* right (which unit is marginal, when
scarcity bites) but systematically misses the level of a market that prices risk,
congestion and battery opportunity cost into every hour. A pure statistical model gets
the level right but has no idea why tomorrow should differ from today.

So the hybrid anchors on what the market already said - today's cleared DAM price for
the same hour, which is public before our 06:00 CT call - and learns how much tomorrow
moves away from it given the *forecast change* in the fundamentals:

    DAM[T,h] ~ DAM[T-1,h], mean DAM[T-1], DAM[T-7,h],
               degree-hours |T-65F| [T,h] and the change vs T-1,
               wind and solar capacity factor[T,h] and their change vs T-1,
               weekend flags for T and T-1, hour-of-block dummies

Fitted separately for four blocks of the day (night, morning, evening peak, late
evening), because what moves a 3 a.m. price is not what moves a 7 p.m. price.

Estimation: ridge-penalised Huber regression by IRLS, recency-weighted (half-life 30
days). Huber because ERCOT prices have $1,000+ spikes that would otherwise own an OLS
fit. Everything is numpy; every coefficient can be printed and argued about.

No look-ahead: a training row for target day T uses only (a) DAM prices up to T-1 and
(b) the weather forecast for T issued ~24h earlier (Open-Meteo previous-runs archive).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import CFG, Config
from .models import solar_forecast, wind_forecast

BLOCKS = {"night": list(range(1, 7)), "morning": list(range(7, 15)),
          "peak": list(range(15, 21)), "evening": list(range(21, 25))}
HE_BLOCK = {he: b for b, hs in BLOCKS.items() for he in hs}

# Heating and cooling enter as ONE degree-hours driver (CDD + HDD). Separate columns
# looked better on paper but failed a test in a way that matters every autumn: in a
# summer-heavy training window HDD is almost always zero, so its coefficient is set by
# a handful of rows, and the first cool evening extrapolated the price by -$15/MWh.
BASE_FEATURES = ["lag1", "lag1_mean", "lag7", "dd", "d_dd",
                 "wind_cf", "solar_cf", "d_wind_cf", "d_solar_cf", "wknd_t", "wknd_d"]
# Live inputs are clamped to the range seen in training (per block). The model may not
# extrapolate; tails beyond its experience are the stack model's job. Clamps are logged.
CLAMPED = ["lag1", "lag1_mean", "lag7", "dd", "d_dd", "wind_cf", "solar_cf",
           "d_wind_cf", "d_solar_cf"]


# ------------------------------------------------------------------ features
def weather_by_hour(wx: pd.DataFrame) -> pd.DataFrame:
    """Hourly weather frame (with op_date, he) -> one row per (op_date, he)."""
    if wx is None or not len(wx):
        return pd.DataFrame(columns=["op_date", "he", "dd", "wind_cf", "solar_cf"])
    w = wx.dropna(subset=["temp_f", "wind_ms", "ghi"]).copy()
    w["dd"] = w["cdd"] + w["hdd"]
    w["wind_cf"] = wind_forecast(w, 1.0)
    w["solar_cf"] = solar_forecast(w, 1.0)
    return (w.groupby(["op_date", "he"], as_index=False)[["dd", "wind_cf", "solar_cf"]]
             .mean())


def build_rows(targets: list[dt.date], dam: pd.DataFrame, wxh: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for each target day. `dam` = (op_date, he, spp); `wxh` = weather_by_hour.

    A row is dropped (not imputed) if any input it needs is missing.
    """
    if not len(dam) or not len(wxh):
        return pd.DataFrame()
    p = dam.set_index(["op_date", "he"])["spp"]
    daily_mean = dam.groupby("op_date")["spp"].mean()
    w = wxh.set_index(["op_date", "he"])
    rows = []
    for t in targets:
        d1, d7 = t - dt.timedelta(days=1), t - dt.timedelta(days=7)
        if d1 not in daily_mean.index:
            continue
        for he in range(1, 25):
            try:
                lag1 = p.loc[(d1, he)]
                lag7 = p.loc[(d7, he)]
                wt = w.loc[(t, he)]
                wd = w.loc[(d1, he)]
            except KeyError:
                continue
            rows.append({
                "op_date": t, "he": he, "block": HE_BLOCK[he],
                "lag1": float(lag1), "lag1_mean": float(daily_mean.loc[d1]),
                "lag7": float(lag7),
                "dd": float(wt["dd"]), "d_dd": float(wt["dd"] - wd["dd"]),
                "wind_cf": float(wt["wind_cf"]), "solar_cf": float(wt["solar_cf"]),
                "d_wind_cf": float(wt["wind_cf"] - wd["wind_cf"]),
                "d_solar_cf": float(wt["solar_cf"] - wd["solar_cf"]),
                "wknd_t": float(t.weekday() >= 5), "wknd_d": float(d1.weekday() >= 5),
                "y": float(p.get((t, he), np.nan)),
            })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.dropna(subset=BASE_FEATURES)
    return df


def _design(df: pd.DataFrame, block: str) -> tuple[np.ndarray, list[str]]:
    hes = BLOCKS[block]
    cols = [df[c].to_numpy(float) for c in BASE_FEATURES]
    names = list(BASE_FEATURES)
    for he in hes[1:]:                       # first hour of the block is the reference
        cols.append((df["he"].to_numpy() == he).astype(float))
        names.append(f"he{he}")
    return np.column_stack(cols), names


# --------------------------------------------------------------- estimator
@dataclass
class BlockFit:
    names: list
    mu: np.ndarray
    sd: np.ndarray
    beta: np.ndarray          # [intercept, standardised coefs...]
    n: int
    scale: float
    lo: np.ndarray = None     # training range per column (for clamping)
    hi: np.ndarray = None

    def clamp(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xc = np.clip(X, self.lo, self.hi)
        return Xc, (Xc != X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xc, _ = self.clamp(X)
        Z = (Xc - self.mu) / self.sd
        return self.beta[0] + Z @ self.beta[1:]

    def coefficients(self) -> dict:
        """Coefficients back on the original scale ($/MWh per unit of feature)."""
        raw = self.beta[1:] / self.sd
        return {n: round(float(b), 4) for n, b in zip(self.names, raw)}


def huber_ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, k: float,
                iters: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu = np.average(X, axis=0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=w))
    # A near-constant column gets a neutral scale so the penalty still restrains it;
    # otherwise a feature active in 3 rows could earn an enormous raw coefficient.
    sd[sd < 1e-6] = 1.0
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    P = np.eye(Z.shape[1]) * lam
    P[0, 0] = 0.0                            # intercept not penalised
    r_w = w.copy()
    beta = np.zeros(Z.shape[1])
    scale = 1.0
    for _ in range(iters):
        A = Z.T @ (Z * r_w[:, None]) + P
        b = Z.T @ (r_w * y)
        new = np.linalg.solve(A, b)
        resid = y - Z @ new
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = max(1.4826 * mad, 1e-6)
        u = np.abs(resid) / (k * scale)
        hub = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-12))
        r_w = w * hub
        if np.max(np.abs(new - beta)) < 1e-6:
            beta = new
            break
        beta = new
    return mu, sd, beta, scale


@dataclass
class HybridModel:
    cfg: Config = field(default_factory=lambda: CFG)
    fits: dict = field(default_factory=dict)
    trained_through: dt.date | None = None
    n_days: int = 0

    def fit(self, rows: pd.DataFrame, as_of: dt.date) -> "HybridModel":
        """Fit on rows whose target day is <= as_of and inside the training window."""
        lo = as_of - dt.timedelta(days=self.cfg.hybrid_train_days)
        tr = rows[(rows["op_date"] <= as_of) & (rows["op_date"] > lo)].dropna(subset=["y"])
        self.n_days = int(tr["op_date"].nunique())
        if self.n_days < self.cfg.hybrid_min_train_days:
            raise ValueError(f"hybrid needs {self.cfg.hybrid_min_train_days} training days, "
                             f"has {self.n_days}")
        age = np.array([(as_of - d).days for d in tr["op_date"]], float)
        wts = 0.5 ** (age / self.cfg.hybrid_halflife_days)
        self.fits = {}
        for b in BLOCKS:
            m = (tr["block"] == b).to_numpy()
            X, names = _design(tr[m], b)
            mu, sd, beta, scale = huber_ridge(X, tr.loc[m, "y"].to_numpy(float), wts[m],
                                              self.cfg.hybrid_ridge, self.cfg.hybrid_huber_k)
            lo, hi = X.min(axis=0), X.max(axis=0)
            free = [i for i, n in enumerate(names) if n not in CLAMPED]
            lo[free], hi[free] = -np.inf, np.inf
            self.fits[b] = BlockFit(names, mu, sd, beta, int(m.sum()), scale, lo, hi)
        self.trained_through = as_of
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        out = np.full(len(rows), np.nan)
        for b, f in self.fits.items():
            m = (rows["block"] == b).to_numpy()
            if m.any():
                X, _ = _design(rows[m], b)
                out[m] = f.predict(X)
        return np.clip(out, self.cfg.price_floor, self.cfg.voll)

    def clamp_report(self, rows: pd.DataFrame) -> list[dict]:
        """Which live inputs were outside training experience (and so were clamped)."""
        out = []
        for b, f in self.fits.items():
            m = (rows["block"] == b).to_numpy()
            if not m.any():
                continue
            X, names = _design(rows[m], b)
            _, hit = f.clamp(X)
            for j in np.where(hit.any(axis=0))[0]:
                out.append({"block": b, "feature": names[j], "hours": int(hit[:, j].sum()),
                            "train_min": round(float(f.lo[j]), 4),
                            "train_max": round(float(f.hi[j]), 4),
                            "live_values": [round(float(v), 4) for v in X[hit[:, j], j]]})
        return out

    def describe(self) -> dict:
        return {b: {"n_rows": f.n, "robust_scale": round(f.scale, 3),
                    "intercept": round(float(f.beta[0]), 4),
                    "coef": f.coefficients()} for b, f in self.fits.items()}


def rolling_residuals(rows: pd.DataFrame, as_of: dt.date, n_days: int,
                      cfg: Config = CFG) -> pd.DataFrame:
    """Rolling-origin out-of-sample errors over the last n_days before as_of.

    For each day t: fit on targets <= t-1 only, predict t. Used to size the interval
    honestly (in-sample residuals understate the error).
    """
    out = []
    days = sorted(d for d in rows["op_date"].unique() if d <= as_of)[-n_days:]
    for t in days:
        try:
            m = HybridModel(cfg).fit(rows, t - dt.timedelta(days=1))
        except ValueError:
            continue
        r = rows[rows["op_date"] == t].dropna(subset=["y"])
        if not len(r):
            continue
        out.append(r.assign(pred=m.predict(r))[["op_date", "he", "y", "pred"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["op_date", "he", "y", "pred"])


def interval_from_residuals(res: pd.DataFrame) -> dict:
    """Empirical 10th/90th percentile errors: hourly by block, and for the daily average."""
    if not len(res):
        return {}
    res = res.assign(err=res["y"] - res["pred"], block=res["he"].map(HE_BLOCK))
    hourly = {b: (float(np.percentile(g["err"], 10)), float(np.percentile(g["err"], 90)))
              for b, g in res.groupby("block")}
    d = res.groupby("op_date")[["y", "pred"]].mean()
    de = (d["y"] - d["pred"]).to_numpy()
    pk = res[res["he"].between(15, 20)].groupby("op_date")[["y", "pred"]].mean()
    pe = (pk["y"] - pk["pred"]).to_numpy()
    return {"hourly": hourly,
            "daily": (float(np.percentile(de, 10)), float(np.percentile(de, 90))),
            "peak": (float(np.percentile(pe, 10)), float(np.percentile(pe, 90))) if len(pe) else None,
            "n_days": int(len(d))}
