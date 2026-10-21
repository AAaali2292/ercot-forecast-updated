"""Forecast models: load, wind, solar.

These are deliberately simple and fully inspectable. A model you can explain in an
interview beats a model with a better R-squared that you cannot defend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CDD_BASE_F, HDD_BASE_F, WIND_SITES

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
# ERCOT load is driven by temperature, but the shape of that response matters more than
# it looks. An earlier version of this model used a quadratic in CDD:
#
#     load = b0 + b1*CDD + b2*CDD^2 + ...
#
# On a holdout test that failed in a specific and instructive way. It over-forecast
# above ~98F by 2-5 GW and under-forecast the 94-98F band by 2-5 GW, ten days in a row.
# The mean error looked acceptable because the two halves cancelled; the model was
# actually wrong nearly every single day.
#
# The cause is that a parabola has one curvature everywhere. Forced to fit both the
# gentle warm-day slope and the steep heatwave slope, it splits the difference and
# then accelerates without limit past the top of the training range.
#
# So the temperature response is now PIECEWISE LINEAR - a series of hinge terms
# max(0, CDD - knot). Each segment gets its own slope, and past the last knot the
# response stays linear instead of curving off to infinity. Safer to extrapolate and
# far more faithful in the middle of the range.
#
# Three other terms earn their place:
#   * a time trend, because ERCOT demand is growing several GW a year (data centres,
#     electrification, oilfield load). Without it the model blames recent growth on
#     weather and systematically under-forecasts the most recent days.
#   * 24-hour average CDD, because thermal mass means day 5 of a heatwave draws more
#     load than day 1 at the same temperature.
#   * full day-of-week effects, not just a weekend flag.

CDD_KNOTS = (5.0, 12.0, 18.0, 24.0, 30.0)
HDD_KNOTS = (10.0, 20.0)
CDD24_KNOTS = (8.0, 16.0)
_EPOCH = pd.Timestamp("2020-01-01", tz="America/Chicago")


def _hinge(x: np.ndarray, knot: float) -> np.ndarray:
    return np.maximum(0.0, x - knot)


def _design(df: pd.DataFrame) -> np.ndarray:
    ts = df["ts"]
    hr = ts.dt.hour.to_numpy() + ts.dt.minute.to_numpy() / 60.0
    dow = ts.dt.dayofweek.to_numpy()
    cdd = df["cdd"].to_numpy(float)
    hdd = df["hdd"].to_numpy(float)
    cdd24 = (df["cdd_24h"].to_numpy(float)
             if "cdd_24h" in df.columns else cdd)
    years = ((ts - _EPOCH).dt.total_seconds().to_numpy() / (86400.0 * 365.25))

    cols = [np.ones(len(df)), years]

    # piecewise-linear temperature response
    cols.append(cdd)
    cols += [_hinge(cdd, k) for k in CDD_KNOTS]
    cols.append(hdd)
    cols += [_hinge(hdd, k) for k in HDD_KNOTS]

    # heat build over the preceding day
    cols.append(cdd24)
    cols += [_hinge(cdd24, k) for k in CDD24_KNOTS]

    # day of week (Monday is the reference level)
    for d in range(1, 7):
        cols.append((dow == d).astype(float))
    cols.append((dow >= 5).astype(float) * cdd)

    # hour-of-day shape, and how that shape steepens on hot days
    for k in (1, 2, 3, 4):
        cols.append(np.sin(2 * np.pi * k * hr / 24.0))
        cols.append(np.cos(2 * np.pi * k * hr / 24.0))
    for k in (1, 2):
        cols.append(cdd * np.sin(2 * np.pi * k * hr / 24.0))
        cols.append(cdd * np.cos(2 * np.pi * k * hr / 24.0))
    return np.column_stack(cols)


class LoadModel:
    def __init__(self):
        self.beta = None
        self.rmse = np.nan
        self.mape = np.nan
        self.n = 0
        self.source = "unfitted"

    def fit(self, hist: pd.DataFrame, source: str = "EIA-930") -> "LoadModel":
        """hist needs columns: ts, demand, cdd, hdd."""
        d = hist.dropna(subset=["demand", "cdd", "hdd"]).copy()
        if len(d) < 200:
            raise ValueError(f"not enough history to fit load model ({len(d)} rows)")
        X = _design(d)
        y = d["demand"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.beta = beta
        resid = y - X @ beta
        self.rmse = float(np.sqrt(np.mean(resid ** 2)))
        self.mape = float(np.mean(np.abs(resid / y)) * 100)
        self.n = len(d)
        self.source = source
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("LoadModel not fitted")
        return _design(df) @ self.beta


def build_load_history(eia: pd.DataFrame, wx: pd.DataFrame,
                       fuelmix: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Join demand history to weather. Prefers EIA-930; falls back to ERCOT fuel mix."""
    if len(eia):
        hist, src = eia.copy(), "EIA-930"
    elif len(fuelmix):
        h = fuelmix[["ts", "load"]].rename(columns={"load": "demand"})
        hist = (h.set_index("ts").resample("1h", label="right", closed="right").mean().dropna().reset_index())
        src = "ERCOT fuel mix (short history)"
    else:
        return pd.DataFrame(), "none"

    keep = [c for c in ("ts", "cdd", "hdd", "temp_f", "cdd_24h", "cdd_72h")
            if c in wx.columns]
    w = wx[keep].copy()
    hist = hist.sort_values("ts")
    w = w.sort_values("ts")
    merged = pd.merge_asof(hist, w, on="ts", direction="nearest",
                           tolerance=pd.Timedelta("90min")).dropna(subset=["cdd", "hdd"])
    return merged, src


# ---------------------------------------------------------------------------
# WIND
# ---------------------------------------------------------------------------
# Standard turbine power curve, normalised. Cut-in ~3 m/s, rated ~12 m/s,
# cut-out 25 m/s. Between cut-in and rated, output goes as roughly v^3 (kinetic
# energy of the air column), which is why wind forecasts are so error-prone:
# a 10% miss on wind speed is a ~30% miss on power.
CUT_IN, RATED, CUT_OUT = 3.0, 12.0, 25.0


def turbine_cf(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    cf = np.zeros_like(v)
    ramp = (v >= CUT_IN) & (v < RATED)
    cf[ramp] = ((v[ramp] ** 3 - CUT_IN ** 3) / (RATED ** 3 - CUT_IN ** 3))
    cf[(v >= RATED) & (v < CUT_OUT)] = 1.0
    return np.clip(cf, 0.0, 1.0)


def wind_forecast(wx: pd.DataFrame, installed_mw: float,
                  availability: float = 0.93, losses: float = 0.12) -> np.ndarray:
    """Convert per-site 100m wind speed to fleet MW.

    Averaging speeds across sites before applying the power curve would understate
    output because the curve is convex; we apply the curve per site, then weight.
    """
    site_cols = [f"ws_{n}" for n, *_ in WIND_SITES if f"ws_{n}" in wx.columns]
    if site_cols:
        weights = np.array([w for n, _, _, w in WIND_SITES if f"ws_{n}" in wx.columns])
        weights = weights / weights.sum()
        cf = np.zeros(len(wx))
        for col, w in zip(site_cols, weights):
            cf += w * turbine_cf(wx[col].to_numpy(float))
    else:
        cf = turbine_cf(wx["wind_ms"].to_numpy(float))
    return np.clip(cf * (1 - losses) * availability, 0, 1) * installed_mw


def calibrate_wind(wx: pd.DataFrame, fuelmix: pd.DataFrame,
                   installed_mw: float) -> float:
    """Scale factor so modelled wind matches recent actuals. Corrects fleet-wide bias."""
    if not len(fuelmix) or not len(wx) or installed_mw <= 0:
        return 1.0
    act = (fuelmix.set_index("ts")["wind"].resample("1h", label="right", closed="right").mean().dropna().reset_index())
    if len(act) < 12:
        return 1.0
    m = pd.merge_asof(act.sort_values("ts"), wx.sort_values("ts"), on="ts",
                      direction="nearest", tolerance=pd.Timedelta("40min")).dropna(
                          subset=["wind_ms"])
    if len(m) < 12:
        return 1.0
    model = wind_forecast(m, installed_mw)
    if model.sum() <= 0:
        return 1.0
    k = float(m["wind"].sum() / model.sum())
    return float(np.clip(k, 0.5, 2.0))


# ---------------------------------------------------------------------------
# SOLAR
# ---------------------------------------------------------------------------
# PV output is close to linear in plane-of-array irradiance, with a small negative
# temperature coefficient (panels lose ~0.4%/C above 25C cell temperature).
STC_IRRADIANCE = 1000.0  # W/m2


def solar_forecast(wx: pd.DataFrame, installed_mw: float,
                   dc_ac_ratio: float = 1.25, losses: float = 0.14,
                   temp_coeff: float = -0.004) -> np.ndarray:
    ghi = wx["ghi"].to_numpy(float)
    tf = wx["temp_f"].to_numpy(float)
    tc = (tf - 32) * 5 / 9 + 0.030 * ghi          # cell temp approx (NOCT-style)
    derate = 1 + temp_coeff * np.clip(tc - 25.0, 0, None)
    cf = np.clip(ghi / STC_IRRADIANCE * dc_ac_ratio * (1 - losses) * derate, 0, 1)
    return cf * installed_mw


def calibrate_solar(wx: pd.DataFrame, fuelmix: pd.DataFrame,
                    installed_mw: float) -> float:
    if not len(fuelmix) or not len(wx) or installed_mw <= 0:
        return 1.0
    act = (fuelmix.set_index("ts")["solar"].resample("1h", label="right", closed="right").mean().dropna().reset_index())
    if len(act) < 12:
        return 1.0
    m = pd.merge_asof(act.sort_values("ts"), wx.sort_values("ts"), on="ts",
                      direction="nearest", tolerance=pd.Timedelta("40min")).dropna(
                          subset=["ghi"])
    m = m[m["ghi"] > 50]
    if len(m) < 8:
        return 1.0
    model = solar_forecast(m, installed_mw)
    if model.sum() <= 0:
        return 1.0
    return float(np.clip(m["solar"].sum() / model.sum(), 0.5, 2.0))


# ---------------------------------------------------------------------------
# HOUR-OF-DAY BIAS CORRECTION
# ---------------------------------------------------------------------------
# A single scalar cannot fix a shape error. If the solar model keeps the sun up an hour
# too long, one multiplier either crushes midday output or leaves the evening too high -
# and the evening is exactly when net load peaks and price is set.
#
# So we fit a correction per hour, shrunk toward the global factor in proportion to how
# little data that hour has. With a couple of days of feed history the hourly estimates
# are noisy, and shrinkage stops us over-fitting three observations.

def hourly_bias(wx: pd.DataFrame, fuelmix: pd.DataFrame, installed_mw: float,
                which: str = "solar", shrink: float = 6.0) -> dict:
    global_k = (calibrate_solar(wx, fuelmix, installed_mw) if which == "solar"
                else calibrate_wind(wx, fuelmix, installed_mw))
    out = {h: global_k for h in range(24)}
    if not len(fuelmix) or not len(wx) or installed_mw <= 0:
        return out

    act = (fuelmix.set_index("ts")[which].resample("1h", label="right", closed="right").mean().dropna().reset_index())
    if len(act) < 12:
        return out
    m = pd.merge_asof(act.sort_values("ts"), wx.sort_values("ts"), on="ts",
                      direction="nearest", tolerance=pd.Timedelta("40min"))
    need = "ghi" if which == "solar" else "wind_ms"
    m = m.dropna(subset=[need])
    if len(m) < 12:
        return out

    m["model"] = (solar_forecast(m, installed_mw) if which == "solar"
                  else wind_forecast(m, installed_mw))
    m["hour"] = m["ts"].dt.hour
    for h, d in m.groupby("hour"):
        denom = float(d["model"].sum())
        # An hour with essentially no modelled output tells us nothing about bias.
        if denom < 0.005 * installed_mw * max(len(d), 1):
            continue
        k_h = float(d[which].sum()) / denom
        n = len(d)
        w = n / (n + shrink)                      # more data -> trust the hour more
        out[int(h)] = float(np.clip(w * k_h + (1 - w) * global_k, 0.3, 2.5))
    return out


def apply_hourly_bias(values: np.ndarray, hours: np.ndarray, factors: dict,
                      fallback: float = 1.0) -> np.ndarray:
    k = np.array([factors.get(int(h), fallback) for h in hours], dtype=float)
    return np.asarray(values, dtype=float) * k
