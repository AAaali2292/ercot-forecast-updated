"""Model behaviour tests. No network - everything here is deterministic."""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CFG, Config  # noqa: E402
from src.models import (LoadModel, solar_forecast, turbine_cf,  # noqa: E402
                        wind_forecast)
from src.stack import (StackArrays, build_stack, clear,  # noqa: E402
                       mmbtu_h_to_bcfd, synth_fleet)

CAPACITY = {"gas": 69756, "coal": 13705, "nuclear": 5268,
            "wind": 40586, "solar": 39926, "storage": 21149,
            "hydro": 579, "other": 663}


@pytest.fixture(scope="module")
def fleet():
    return synth_fleet(CAPACITY)


@pytest.fixture(scope="module")
def stack(fleet):
    return build_stack(fleet, CFG)


@pytest.fixture(scope="module")
def sa(stack):
    return StackArrays(stack)


# ------------------------------------------------------------------ the stack
def test_fleet_matches_installed_capacity(fleet):
    for cat, key in [("COAL", "coal"), ("NUCLEAR", "nuclear")]:
        assert fleet[fleet.cat == cat]["mw"].sum() == pytest.approx(CAPACITY[key], rel=0.02)
    gas = fleet[fleet.cat.isin(["CCGT", "CT", "STEAM_GAS", "ICE_GAS"])]["mw"].sum()
    assert gas == pytest.approx(CAPACITY["gas"], rel=0.02)


def test_stack_is_merit_ordered(stack):
    assert stack["mc"].is_monotonic_increasing
    assert stack["cum_mw"].is_monotonic_increasing


def test_nuclear_is_cheapest_and_peakers_dearest(stack):
    assert stack["cat"].iloc[0] == "NUCLEAR"
    assert stack[stack.cat == "CT"]["mc"].mean() > stack[stack.cat == "CCGT"]["mc"].mean()


def test_heat_rates_are_physical(stack):
    ccgt = stack[stack.cat == "CCGT"]["heat_rate"]
    assert 6.0 < ccgt.mean() < 8.5
    thermal = stack[stack.fuel_type != "storage"]["heat_rate"]
    assert thermal.min() >= 6.0
    assert thermal.max() <= 15.0
    # storage burns nothing
    assert (stack[stack.fuel_type == "storage"]["heat_rate"] == 0).all()


def test_available_capacity_is_derated(fleet, stack):
    assert stack["avail_mw"].sum() < fleet["mw"].sum()
    assert 70_000 < stack["avail_mw"].sum() < 110_000


def test_storage_sits_above_ccgt_below_the_cap(stack):
    st = stack[stack.cat == "STORAGE"]
    assert len(st) > 0
    assert st["mc"].iloc[0] > stack[stack.cat == "CCGT"]["mc"].max()
    assert st["mc"].iloc[0] < CFG.voll
    # ~21 GW installed, duration-limited
    assert 10_000 < st["avail_mw"].sum() < 20_000


def test_storage_suppresses_spurious_scarcity(fleet):
    """Without batteries the stack runs short and over-fires ORDC on tight days."""
    with_st = Config(**{**CFG.__dict__, "include_storage": True})
    without = Config(**{**CFG.__dict__, "include_storage": False})
    sa_w = StackArrays(build_stack(fleet, with_st))
    sa_o = StackArrays(build_stack(fleet, without))
    net_load = 72_000
    p_w = clear([net_load], sa_w, with_st)["price"][0]
    p_o = clear([net_load], sa_o, without)["price"][0]
    assert p_o > p_w
    assert p_w < 400, f"battery-inclusive price should stay off the cap, got {p_w}"


# -------------------------------------------------------------- price forming
def test_price_is_non_decreasing_in_net_load(sa):
    loads = np.arange(15_000, 80_001, 2_500)
    p = clear(loads, sa, CFG)["energy_price"]
    assert np.all(np.diff(p) >= -1e-9)


def test_higher_gas_gives_higher_power_price(fleet):
    lo, hi = Config(**{**CFG.__dict__}), Config(**{**CFG.__dict__})
    lo.henry_hub, hi.henry_hub = 2.0, 6.0
    p_lo = clear([55_000], StackArrays(build_stack(fleet, lo)), lo)["energy_price"][0]
    p_hi = clear([55_000], StackArrays(build_stack(fleet, hi)), hi)["energy_price"][0]
    assert p_hi > p_lo


def test_ordc_fires_only_when_reserves_tighten(sa):
    slack = clear([35_000], sa, CFG)
    tight = clear([sa.total - 400], sa, CFG)
    assert slack["ordc"][0] < 1.0
    assert tight["ordc"][0] > slack["ordc"][0]


def test_shortage_pins_at_offer_cap(sa):
    r = clear([sa.total + 5_000], sa, CFG)
    assert r["price"][0] == pytest.approx(CFG.voll)
    assert r["marginal_cat"][0] == "SCARCITY"


def test_renewable_oversupply_prints_negative(sa):
    r = clear([-2_000], sa, CFG)
    assert r["price"][0] < 0
    assert r["price"][0] >= CFG.price_floor
    assert r["marginal_cat"][0] == "RENEWABLE_OVERSUPPLY"


def test_price_never_breaches_the_caps(sa):
    r = clear(np.arange(-5_000, 120_000, 1_000), sa, CFG)
    assert r["price"].min() >= CFG.price_floor
    assert r["price"].max() <= CFG.voll


def test_gas_burn_is_physically_sensible(sa):
    burn = mmbtu_h_to_bcfd(clear([55_000], sa, CFG)["gas_burn_mmbtu_h"])[0]
    assert 2.0 < burn < 9.0
    # burn must rise with net load
    b = mmbtu_h_to_bcfd(clear([30_000, 45_000, 60_000], sa, CFG)["gas_burn_mmbtu_h"])
    assert np.all(np.diff(b) > 0)


def test_gas_burn_consistent_with_dispatched_mw(sa):
    r = clear([50_000], sa, CFG)
    implied_hr = r["gas_burn_mmbtu_h"][0] / max(r["gas_mw"][0], 1)
    assert 6.0 < implied_hr < 12.0


# -------------------------------------------------------------------- weather
def test_turbine_power_curve_shape():
    assert turbine_cf([0, 2.9])[0] == 0 and turbine_cf([2.9])[0] == 0
    assert turbine_cf([12.0])[0] == pytest.approx(1.0)
    assert turbine_cf([20.0])[0] == pytest.approx(1.0)
    assert turbine_cf([26.0])[0] == 0.0          # cut-out
    v = np.linspace(3, 12, 30)
    assert np.all(np.diff(turbine_cf(v)) > 0)    # monotonic on the ramp


def test_wind_forecast_bounded_by_installed():
    wx = pd.DataFrame({"ws_Panhandle": [15.0] * 5, "ws_West TX": [15.0] * 5,
                       "ws_Permian": [15.0] * 5, "ws_South TX": [15.0] * 5,
                       "ws_Gulf Coast": [15.0] * 5})
    mw = wind_forecast(wx, 40_000)
    assert np.all(mw >= 0) and np.all(mw <= 40_000)
    assert mw.mean() == pytest.approx(40_000 * 0.93 * 0.88, rel=0.02)


def test_solar_zero_at_night_and_derates_when_hot():
    night = pd.DataFrame({"ghi": [0.0], "temp_f": [78.0]})
    assert solar_forecast(night, 39_000)[0] == 0.0
    cool = pd.DataFrame({"ghi": [900.0], "temp_f": [70.0]})
    hot = pd.DataFrame({"ghi": [900.0], "temp_f": [105.0]})
    assert solar_forecast(hot, 39_000)[0] < solar_forecast(cool, 39_000)[0]


# ----------------------------------------------------------------- load model
def _synthetic_load_history(n_days=90, growth_mw_per_year=9_000.0,
                            kink_slope=1_400.0, weather_noise=0.0, seed=3):
    """Synthetic ERCOT-like demand.

    Deliberately contains the two things that broke the original model on live data:

      * a demand growth trend (ERCOT is adding several GW a year), and
      * a temperature response that is much steeper above ~18 CDD than below it.

    A single quadratic cannot represent two different slopes, so it splits the
    difference: over-forecasting the hot end and under-forecasting the warm middle.
    Set ``weather_noise`` to give days realistic day-to-day variation, which is what
    makes an error-vs-temperature correlation visible.
    """
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-04-01", periods=n_days * 24, freq="h", tz="America/Chicago")
    hr = ts.hour.to_numpy()
    doy = ts.dayofyear.to_numpy()

    temp = (74 + 14 * np.sin(2 * np.pi * (doy - 105) / 365)
            + 11 * np.sin(2 * np.pi * (hr - 9) / 24))
    if weather_noise:
        days = (ts.normalize() - ts.normalize().min()).days.to_numpy()
        anom = rng.normal(0, weather_noise, days.max() + 1)
        anom = pd.Series(anom).rolling(3, min_periods=1).mean().to_numpy()
        temp = temp + anom[days]

    cdd = np.clip(temp - 65, 0, None)
    hdd = np.clip(65 - temp, 0, None)
    years = (ts - pd.Timestamp("2026-04-01", tz="America/Chicago")).total_seconds() \
        / (86400 * 365.25)
    load = (40_000 + 850 * cdd + np.clip(cdd - 18, 0, None) * kink_slope + 350 * hdd
            + growth_mw_per_year * years
            + 6_500 * np.sin(2 * np.pi * (hr - 8) / 24)
            + rng.normal(0, 400, len(ts)))
    df = pd.DataFrame({"ts": ts, "demand": load, "cdd": cdd, "hdd": hdd})
    df["cdd_24h"] = df["cdd"].rolling(24, min_periods=1).mean()
    return df


def _daily_peak_errors(d, hold_days=14):
    cut = d["ts"].max() - pd.Timedelta(days=hold_days)
    lm = LoadModel().fit(d[d["ts"] <= cut])
    test = d[d["ts"] > cut].copy()
    test["pred"] = lm.predict(test)
    return (test.groupby(test["ts"].dt.date)
            .agg(actual=("demand", "max"), pred=("pred", "max"), temp=("cdd", "max")))


def test_load_model_fits_and_generalises():
    d = _synthetic_load_history()
    train, test = d.iloc[:-336], d.iloc[-336:]
    lm = LoadModel().fit(train)
    assert lm.mape < 4.0
    pred = lm.predict(test)
    oos_mape = float(np.mean(np.abs((pred - test["demand"]) / test["demand"])) * 100)
    assert oos_mape < 5.0, f"out-of-sample MAPE {oos_mape:.2f}%"


def test_load_model_rejects_short_history():
    d = _synthetic_load_history(2)
    with pytest.raises(ValueError):
        LoadModel().fit(d)


def test_hotter_weather_implies_more_load():
    d = _synthetic_load_history()
    lm = LoadModel().fit(d)
    base = d.tail(24).copy()
    hot = base.copy()
    hot["cdd"] = hot["cdd"] + 8
    hot["cdd_24h"] = hot["cdd_24h"] + 8
    assert lm.predict(hot).max() > lm.predict(base).max()


# --- regression guards for the bias found on live data ----------------------
# Each of these FAILS against the original quadratic model. Verified, not assumed.


def test_error_is_not_correlated_with_temperature():
    """The bug that showed up on live ERCOT data.

    The old model over-forecast above ~98F and under-forecast the 94-98F band. Mean
    bias looked acceptable because the two halves cancelled, but the model was wrong
    nearly every day. Error correlating with temperature is what exposes it.
    Old model scored r = +0.94 to +0.96 here.
    """
    for seed in (5, 11, 23):
        d = _synthetic_load_history(120, weather_noise=5.0, seed=seed)
        daily = _daily_peak_errors(d, hold_days=21)
        err = daily["pred"] - daily["actual"]
        r = float(np.corrcoef(daily["temp"], err)[0, 1])
        assert abs(r) < 0.75, (
            f"seed {seed}: error tracks temperature (r={r:+.2f}) - "
            "the shape of the temperature response is wrong")


def test_daily_peak_error_is_small():
    """Old model: MAE ~1,200 MW. New model: ~300 MW."""
    for seed in (5, 11, 23):
        d = _synthetic_load_history(120, weather_noise=5.0, seed=seed)
        daily = _daily_peak_errors(d, hold_days=21)
        mae = float((daily["pred"] - daily["actual"]).abs().mean())
        assert mae < 700, f"seed {seed}: daily peak MAE {mae:,.0f} MW"


def test_no_long_runs_of_same_signed_error():
    """Ten consecutive same-signed errors is not noise, it is bias.

    Old model produced a run of 14 out of 14 on this fixture.
    """
    d = _synthetic_load_history(120)
    err = (_daily_peak_errors(d)["pred"] - _daily_peak_errors(d)["actual"]).to_numpy()
    longest, run = 0, 0
    for i in range(len(err)):
        run = run + 1 if i and np.sign(err[i]) == np.sign(err[i - 1]) else 1
        longest = max(longest, run)
    assert longest <= 8, f"{longest} consecutive same-signed errors: systematic bias"


def test_model_captures_demand_growth():
    """Without a trend term, recent growth gets misattributed to weather.

    Old model bias: +1,246 MW. New model: -40 MW.
    """
    d = _synthetic_load_history(150, growth_mw_per_year=12_000.0)
    daily = _daily_peak_errors(d)
    bias = float((daily["pred"] - daily["actual"]).mean())
    assert abs(bias) < 600, f"bias {bias:,.0f} MW on unseen recent days"


def test_extrapolation_stays_linear_beyond_training_range():
    """Piecewise-linear must not accelerate past the hottest day it has seen."""
    d = _synthetic_load_history()
    lm = LoadModel().fit(d)
    hi = float(d["cdd"].max())
    row = d.tail(1).copy()

    def at(c):
        r = row.copy()
        r["cdd"] = c
        r["cdd_24h"] = c * 0.7
        return float(lm.predict(r)[0])

    step_inside = at(hi) - at(hi - 3)
    step_outside = at(hi + 6) - at(hi + 3)
    assert step_outside < 2.0 * abs(step_inside) + 1_500, (
        f"extrapolation accelerating: {step_inside:,.0f} -> {step_outside:,.0f} MW "
        "per 3 CDD")


# --------------------------------------------------------------------- scoring
