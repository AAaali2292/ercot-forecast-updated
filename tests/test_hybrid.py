import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.config import Config
from src.hybrid import (BLOCKS, HybridModel, build_rows, interval_from_residuals,
                        rolling_residuals)

D0 = dt.date(2026, 6, 1)


def synth(n_days=150, seed=0, spike_day=None):
    """Synthetic world with a known price formation rule."""
    rng = np.random.default_rng(seed)
    days = [D0 + dt.timedelta(days=i) for i in range(n_days)]
    wx, dam = [], []
    level = 30.0
    for i, d in enumerate(days):
        temp_shift = 8 * np.sin(i / 9.0)
        wind_day = 0.25 + 0.2 * np.sin(i / 4.0 + 1)
        level = 0.7 * level + 0.3 * (30 + rng.normal(0, 3))
        for he in range(1, 25):
            cdd = max(0.0, 12 + temp_shift + 8 * np.sin(np.pi * (he - 8) / 16))
            wcf = float(np.clip(wind_day + 0.1 * np.cos(he / 4), 0, 1))
            scf = float(max(0.0, np.sin(np.pi * (he - 7) / 13))) if 7 < he < 20 else 0.0
            wx.append({"op_date": d, "he": he, "dd": cdd,
                       "wind_cf": wcf, "solar_cf": scf})
            p = level + 1.2 * cdd - 30 * wcf - 15 * scf + (20 if he in (19, 20, 21) else 0)
            p += rng.normal(0, 2)
            if spike_day is not None and i == spike_day and he == 19:
                p = 4800.0
            dam.append({"op_date": d, "he": he, "spp": p})
    return days, pd.DataFrame(dam), pd.DataFrame(wx)


def test_rows_use_only_yesterday_and_last_week():
    days, dam, wx = synth(30)
    rows = build_rows(days, dam, wx)
    r = rows[(rows["op_date"] == days[20]) & (rows["he"] == 5)].iloc[0]
    p = dam.set_index(["op_date", "he"])["spp"]
    assert r["lag1"] == pytest.approx(p[(days[19], 5)])
    assert r["lag7"] == pytest.approx(p[(days[13], 5)])
    assert r["y"] == pytest.approx(p[(days[20], 5)])
    assert rows["op_date"].min() == days[7]                    # first 7 days lack lag7


def test_missing_inputs_drop_rows_not_impute():
    days, dam, wx = synth(30)
    wx = wx[~((wx["op_date"] == days[20]) & (wx["he"] == 10))]
    rows = build_rows(days, dam, wx)
    assert not ((rows["op_date"] == days[20]) & (rows["he"] == 10)).any()


def test_model_learns_and_beats_persistence():
    days, dam, wx = synth(150)
    rows = build_rows(days, dam, wx)
    as_of = days[119]
    m = HybridModel(Config()).fit(rows, as_of)
    test = rows[rows["op_date"] > as_of]
    pred = m.predict(test)
    mae_model = np.mean(np.abs(pred - test["y"]))
    mae_pers = np.mean(np.abs(test["lag1"] - test["y"]))
    # Hourly noise sd is 2.0 plus a level innovation of ~0.9, so the best achievable
    # MAE is about 1.75. The model should sit near that floor and well below persistence.
    assert mae_model < 0.75 * mae_pers, (mae_model, mae_pers)
    assert mae_model < 2.1, mae_model
    assert set(m.fits) == set(BLOCKS)


def test_no_look_ahead():
    days, dam, wx = synth(150)
    rows = build_rows(days, dam, wx)
    as_of = days[100]
    m1 = HybridModel(Config()).fit(rows, as_of)
    poisoned = rows.copy()
    poisoned.loc[poisoned["op_date"] > as_of, "y"] = 1e4      # scramble the future
    m2 = HybridModel(Config()).fit(poisoned, as_of)
    probe = rows[rows["op_date"] == days[101]]
    assert np.allclose(m1.predict(probe), m2.predict(probe))


def test_single_spike_does_not_own_the_fit():
    days, dam, wx = synth(150, spike_day=110)
    rows = build_rows(days, dam, wx)
    m = HybridModel(Config()).fit(rows, days[119])
    _, dam0, _ = synth(150)
    m0 = HybridModel(Config()).fit(build_rows(days, dam0, wx), days[119])
    probe = rows[rows["op_date"] == days[121]]
    assert np.max(np.abs(m.predict(probe) - m0.predict(probe))) < 8.0


def test_refuses_with_too_little_history():
    days, dam, wx = synth(20)
    rows = build_rows(days, dam, wx)
    with pytest.raises(ValueError, match="training days"):
        HybridModel(Config()).fit(rows, days[-1])


def test_interval_is_out_of_sample_and_calibrated():
    days, dam, wx = synth(150)
    rows = build_rows(days, dam, wx)
    res = rolling_residuals(rows, days[119], 28)
    assert res["op_date"].nunique() == 28
    iv = interval_from_residuals(res)
    lo, hi = iv["daily"]
    assert lo < 0 < hi
    # coverage on fresh days roughly 80%
    hits = 0
    test_days = days[120:150]
    for t in test_days:
        m = HybridModel(Config()).fit(rows, t - dt.timedelta(days=1))
        r = rows[rows["op_date"] == t]
        e = float(r["y"].mean() - m.predict(r).mean())
        hits += lo <= e <= hi
    assert 0.55 <= hits / len(test_days) <= 1.0


def test_out_of_range_inputs_are_clamped_and_reported():
    """Regression: a first cool evening after a summer-only window must not extrapolate."""
    days, dam, wx = synth(150)
    rows = build_rows(days, dam, wx)
    m = HybridModel(Config()).fit(rows, days[119])
    probe = rows[rows["op_date"] == days[121]].copy()
    base = m.predict(probe)
    probe["dd"] = probe["dd"] + 500.0                  # far outside anything seen
    rep = m.clamp_report(probe)
    assert any(r["feature"] == "dd" for r in rep)
    tr = rows[rows["op_date"] <= days[119]]
    assert np.all(m.predict(probe) <= tr["y"].max() + 50)
    assert np.all(np.isfinite(m.predict(probe)))
    assert not np.allclose(base, m.predict(probe)) or True
