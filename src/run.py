"""CLI entry point (what the GitHub Actions schedule calls).

    python -m src.run status            feed health: is every live source reachable?
    python -m src.run backfill          pull official DAM history (31 days daily + annual file)
    python -m src.run forecast          tomorrow's call (adds --if-missing for the backup run)
    python -m src.run daily             06:00 CT: DAM refresh, forecast, score, publish
    python -m src.run dam               afternoon: fetch tomorrow's cleared DAM, score, publish
    python -m src.run capture           overnight: snapshot real-time prices + generation
    python -m src.run score             rebuild scorecard.csv from files
    python -m src.run publish           regenerate README scorecard + docs/data.json
    python -m src.run backtest          simulated walk-forward backtest (kept separate)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback

from . import backtest as bt
from . import capture as cap
from . import forecast as fx
from . import history
from . import publish as pub
from . import score as sc
from . import sources
from .config import CT, FORECASTS


def _print_feeds():
    t = sources.feed_table()
    if not len(t):
        return
    print("\nData feeds")
    print("-" * 78)
    for _, r in t.iterrows():
        print(f"  {r['status']:<5} {r['feed']:<34} {r['detail']}")


def _ensure_history():
    n = len(history.complete_days(history.hub_hourly()))
    if n < 45:
        print(f"DAM history has {n} complete days -> full backfill")
        history.refresh(days_back=31, include_tomorrow=False, annual=True)


def _score_publish():
    df = sc.rebuild_scorecard()
    s = sc.summarise(df)
    pub.update_readme(df)
    pub.build_site()
    print(f"scored {s.get('n', 0)} live day(s)"
          + (f"; primary MAE (daily avg) ${s['primary_mae_atc']}, persistence "
             f"${s.get('persistence_mae_atc')}" if s.get("n") else ""))


def cmd_status(args) -> int:
    sources.fetch_fuel_mix()
    px = sources.fetch_prices()
    sources.fetch_weather_forecast(past_days=1, forecast_days=2)
    today = dt.datetime.now(CT).date()
    sources.fetch_weather_previous_day1(today - dt.timedelta(days=3), today)
    sources.fetch_eia930_demand(days=3)
    sources.fetch_gas_price()
    try:
        d = history.fetch_mis_daily(today)
        sources.log("ERCOT MIS DAM (today's file)", "OK",
                    f"{len(d)} rows; HB_HUBAVG daily avg "
                    f"${d[d['location'] == 'HB_HUBAVG']['spp'].mean():.2f}")
        dash = sources.dashboard_dam_long(px.get("dam"))
        if len(dash):
            m = dash.merge(d, on=["op_date", "he", "location"], suffixes=("_dash", "_mis"))
            if len(m):
                diff = (m["spp_dash"] - m["spp_mis"]).abs().max()
                sources.log("DAM cross-check dashboard vs MIS", "OK" if diff <= 0.011 else "WARN",
                            f"{len(m)} hub-hours, max |diff| ${diff:.4f}")
    except Exception as e:  # noqa: BLE001
        sources.log("ERCOT MIS DAM (today's file)", "FAIL", e)
    _print_feeds()
    bad = [k for k, v in sources.FEED.items() if v["status"] == "FAIL"]
    print("\nALL FEEDS OK" if not bad else f"\nFAILED: {', '.join(bad)}")
    return 0 if not bad else 1


def cmd_backfill(args) -> int:
    r = history.refresh(days_back=31, include_tomorrow=True, annual=True)
    n = len(history.complete_days(history.hub_hourly()))
    print(f"DAM history: {n} complete days; fetched {len(r['fetched'])}, failed {len(r['failed'])}")
    for f in r["failed"][:10]:
        print("  failed:", f)
    _print_feeds()
    _score_publish()
    return 0 if n >= 21 else 1


def _do_forecast(date: str | None, if_missing: bool) -> int:
    target = dt.date.fromisoformat(date) if date else None
    tgt = target or (dt.datetime.now(CT).date() + dt.timedelta(days=1))
    if if_missing:
        p = FORECASTS / f"{tgt.isoformat()}.json"
        if p.exists():
            old = json.loads(p.read_text())
            if old.get("pre_close") and not old.get("primary_is_fallback"):
                print(f"forecast for {tgt} already committed pre-close; nothing to do")
                return 0
    _ensure_history()
    f = fx.make_forecast(target)
    path, written = fx.save_forecast(f)
    if not written:
        print(f"kept existing pre-close forecast for {f['target_date']} (integrity rule)")
    else:
        pub.save_note(f)
    p = f["summary"][f["primary_model"]]
    iv = f.get("intervals") or {}
    print(f"forecast for {f['target_date']} (primary: {f['primary_model']}, "
          f"{'LIVE pre-close' if f['live'] else 'NOT live'})")
    print(f"  daily avg  ${p['atc']}  [80%: {iv.get('atc_p10')} - {iv.get('atc_p90')}]")
    print(f"  peak HE15-20 ${p['peak']}  [80%: {iv.get('peak_p10')} - {iv.get('peak_p90')}]")
    for m, sm in f["summary"].items():
        print(f"    {m:<12} daily avg ${sm['atc']}  peak ${sm['peak']}")
    if f.get("model_errors"):
        print("  model errors:", f["model_errors"])
    print(f"  -> {path}")
    return 0


def cmd_forecast(args) -> int:
    rc = _do_forecast(args.date, args.if_missing)
    _print_feeds()
    _score_publish()
    return rc


def cmd_daily(args) -> int:
    rc = 0
    try:
        rc = _do_forecast(None, if_missing=False)
    except Exception as e:  # noqa: BLE001
        print(f"FORECAST FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        rc = 1
    _print_feeds()
    _score_publish()
    return rc


def cmd_dam(args) -> int:
    history.refresh(days_back=3, include_tomorrow=True)
    _print_feeds()
    _score_publish()
    return 0


def cmd_capture(args) -> int:
    outs = cap.capture_all()
    if not outs:
        print("nothing to capture: the real-time feed returned no intervals", file=sys.stderr)
        _print_feeds()
        return 1
    for o in outs:
        p, written = cap.save_outcome(o)
        print(f"captured {o['target_date']}: {o['coverage_pct']}% of RT intervals"
              + ("" if written else " (kept existing fuller snapshot)")
              + f"; RT daily avg ${o['rt']['atc']}")
    _print_feeds()
    _score_publish()
    return 0


def cmd_score(args) -> int:
    _score_publish()
    return 0


def cmd_publish(args) -> int:
    _score_publish()
    return 0


def cmd_merge(args) -> int:
    """Re-apply this run's outputs (saved in --src) onto a fresh checkout of main.

    Used only when two scheduled runs collide on push. The same integrity rules apply:
    an existing live forecast is never replaced, the fuller outcome snapshot wins, and
    price history is merged by source authority.
    """
    import shutil
    from pathlib import Path
    from .config import BACKTEST, DAM_HISTORY, NOTES, OUTCOMES
    src = Path(args.src)
    n = 0
    for p in sorted((src / "forecasts").glob("*.json")):
        # Only files main does not have yet: existing evidence files are never rewritten.
        if not (FORECASTS / p.name).exists():
            shutil.copy(p, FORECASTS / p.name)
            n += 1
    for p in sorted((src / "outcomes").glob("*.json")):
        cap.save_outcome(json.loads(p.read_text()))      # fuller snapshot wins
    for p in sorted((src / "notes").glob("*.md")):
        if not (NOTES / p.name).exists():
            shutil.copy(p, NOTES / p.name)
    hist = src / "dam_history.csv"
    if hist.exists():
        import pandas as pd
        h = pd.read_csv(hist)
        h["op_date"] = pd.to_datetime(h["op_date"]).dt.date
        history.upsert(h)
    # Backtest: keep whichever run is newer.
    def _gen(d):
        try:
            return json.loads((d / "summary.json").read_text()).get("generated_at_ct", "")
        except Exception:  # noqa: BLE001
            return ""
    if (src / "backtest").exists() and _gen(src / "backtest") > _gen(BACKTEST):
        for p in sorted((src / "backtest").glob("*")):
            shutil.copy(p, BACKTEST / p.name)
    print(f"merged {n} forecast file(s) from {src}")
    _score_publish()
    return 0


def cmd_backtest(args) -> int:
    _ensure_history()
    r = bt.run(n_days=args.days)
    print(json.dumps({k: v for k, v in r.items() if k != "feeds"}, indent=2, default=str))
    _print_feeds()
    _score_publish()
    return 0 if r.get("status") == "ok" else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ercot-daily")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, hlp in [
        ("status", cmd_status, "feed health"),
        ("backfill", cmd_backfill, "pull official DAM history"),
        ("forecast", cmd_forecast, "tomorrow's call"),
        ("daily", cmd_daily, "forecast + score + publish"),
        ("dam", cmd_dam, "fetch cleared DAM, score, publish"),
        ("capture", cmd_capture, "snapshot real-time prices"),
        ("score", cmd_score, "rebuild scorecard"),
        ("publish", cmd_publish, "regenerate site + README"),
        ("backtest", cmd_backtest, "simulated walk-forward backtest"),
        ("merge", cmd_merge, "re-apply outputs after a push collision"),
    ]:
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--date", default=None, help="YYYY-MM-DD (forecast only; non-live)")
        p.add_argument("--if-missing", action="store_true", dest="if_missing")
        p.add_argument("--days", type=int, default=60)
        p.add_argument("--src", default=None)
        p.set_defaults(func=fn)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
