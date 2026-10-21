import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from src import config, history, score

T = dt.date(2026, 10, 22)


def put_dam(day, f):
    history.upsert(pd.DataFrame({"op_date": [day] * 24, "he": range(1, 25),
                                 "location": "HB_HUBAVG", "spp": [f(h) for h in range(1, 25)],
                                 "source": "ercot-mis-daily"}))


def put_fc(day, hyb, pers, live=True, gen="2026-10-21T06:07:00-05:00", source_hours=24):
    fc = {"schema": 2, "target_date": str(day), "generated_at_ct": gen, "saved_at_ct": gen,
          "dam_close_ct": "2026-10-21T10:00:00-05:00", "pre_close": live, "live": live,
          "git_sha": "abc123", "primary_model": "hybrid", "primary_is_fallback": False,
          "intervals": {"atc_p10": 30.0, "atc_p90": 50.0},
          "hourly": [{"he": h, "hybrid": hyb(h) if h <= source_hours else None,
                      "persistence": pers(h)} for h in range(1, 25)]}
    (config.FORECASTS / f"{day}.json").write_text(json.dumps(fc))


def test_errors_and_benchmark():
    put_dam(T, lambda h: 40.0)
    put_fc(T, hyb=lambda h: 42.0, pers=lambda h: 35.0)
    df = score.rebuild_scorecard()
    r = df.iloc[0]
    assert r["act_atc"] == 40.0 and r["fc_atc"] == 42.0 and r["err_atc"] == 2.0
    assert r["persistence_err_atc"] == -5.0
    assert r["mae_hourly"] == 2.0
    assert r["in_interval"] == 1
    assert r["direction_hit"] == 1          # model said up from 35, it went up
    s = score.summarise(df)
    assert s["n"] == 1 and s["skill_vs_persistence_atc_pct"] == pytest.approx(60.0)


def test_late_call_is_excluded_from_live_record():
    put_dam(T, lambda h: 40.0)
    put_fc(T, hyb=lambda h: 40.0, pers=lambda h: 35.0, live=False,
           gen="2026-10-21T14:00:00-05:00")
    df = score.rebuild_scorecard()
    assert len(df) == 1
    assert score.summarise(df, live_only=True)["n"] == 0
    assert score.summarise(df, live_only=False)["n"] == 1


def test_no_actual_no_score():
    put_fc(T, hyb=lambda h: 40.0, pers=lambda h: 35.0)
    assert len(score.rebuild_scorecard()) == 0


def test_peak_block_is_he15_to_he20():
    put_dam(T, lambda h: 100.0 if 15 <= h <= 20 else 20.0)
    put_fc(T, hyb=lambda h: 100.0 if 15 <= h <= 20 else 20.0, pers=lambda h: 20.0)
    r = score.rebuild_scorecard().iloc[0]
    assert r["act_peak"] == 100.0 and r["err_peak"] == 0.0


def test_self_reported_live_flag_is_not_trusted():
    put_dam(T, lambda h: 40.0)
    put_fc(T, hyb=lambda h: 40.0, pers=lambda h: 35.0, live=True,
           gen="2026-10-21T10:30:00-05:00")              # claims live, stamped after close
    df = score.rebuild_scorecard()
    assert not bool(df.iloc[0]["live"])


def test_dashboard_only_actual_is_provisional():
    history.upsert(pd.DataFrame({"op_date": [T] * 24, "he": range(1, 25), "location": "HB_HUBAVG",
                                 "spp": 40.0, "source": "ercot-dashboard"}))
    put_fc(T, hyb=lambda h: 41.0, pers=lambda h: 35.0)
    df = score.rebuild_scorecard()
    assert df.iloc[0]["status"].startswith("provisional")
    assert score.summarise(df)["n"] == 0


def test_days_are_never_dropped_silently():
    put_dam(T, lambda h: 40.0)
    put_fc(T, hyb=lambda h: 40.0, pers=lambda h: 35.0, source_hours=20)
    df = score.rebuild_scorecard()
    assert len(df) == 1 and df.iloc[0]["status"].startswith("unscorable")
    put_fc(T, hyb=lambda h: 40.0, pers=lambda h: 35.0, source_hours=23)   # one gap: scored
    df = score.rebuild_scorecard()
    assert df.iloc[0]["status"] == "scored" and df.iloc[0]["n_hours"] == 23


def test_all_models_scored_on_the_same_hours():
    put_dam(T, lambda h: 40.0 + h)
    put_fc(T, hyb=lambda h: 40.0 + h, pers=lambda h: 30.0, source_hours=23)   # HE24 missing
    r = score.rebuild_scorecard().iloc[0]
    assert r["n_hours"] == 23 and r["err_atc"] == 0.0
    assert r["act_atc"] == pytest.approx(np.mean([40.0 + h for h in range(1, 24)]))
