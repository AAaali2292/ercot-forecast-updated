"""Publish: docs/data.json (read by the GitHub Pages dashboard), README scorecard, daily note."""
from __future__ import annotations

import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from . import history
from .config import BACKTEST, CT, DOCS, FORECASTS, NOTES, ROOT
from .score import MODELS, load_scorecard, summarise

README = ROOT / "README.md"
MARK_START = "<!-- SCORECARD:START -->"
MARK_END = "<!-- SCORECARD:END -->"


def _clean(o):
    """JSON-safe: NaN/inf -> None, numpy -> python."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    return o


def _fmt(v, dp=2, dash="–"):
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
            return dash
    except TypeError:
        pass
    return f"{float(v):,.{dp}f}"


def _repo_url() -> str | None:
    repo = os.environ.get("GITHUB_REPOSITORY")
    return f"https://github.com/{repo}" if repo else None


def _load_fc(name: str):
    p = FORECASTS / f"{name}.json"
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def scored_day_detail(df: pd.DataFrame) -> dict:
    """Hourly actual vs every model, for each scored day (feeds the day picker)."""
    dam = history.hub_hourly()
    out = {}
    for t in df["target_date"].astype(str):
        fc = _load_fc(t)
        if not fc:
            continue
        a = dam[dam["op_date"] == dt.date.fromisoformat(t)]
        act = {int(h): round(float(v), 2) for h, v in zip(a["he"], a["spp"])}
        out[t] = [{"he": r["he"], "actual": act.get(int(r["he"])),
                   **{m: r.get(m) for m in MODELS},
                   "p10": r.get("hybrid_p10"), "p90": r.get("hybrid_p90")}
                  for r in fc.get("hourly", [])]
    return out


def latest_forecast() -> dict | None:
    files = sorted(FORECASTS.glob("*.json"))
    for p in reversed(files):
        try:
            fc = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if fc.get("schema", 1) >= 2:
            return fc
    return None


def build_site() -> str:
    df = load_scorecard()
    s_live = summarise(df, live_only=True)
    s_all = summarise(df, live_only=False)
    fc = latest_forecast()
    lf = None
    if fc:
        lf = {k: fc.get(k) for k in ("target_date", "generated_at_ct", "saved_at_ct",
                                     "dam_close_ct", "pre_close", "live", "code_version",
                                     "run_url", "training_weather_fallback", "primary_model",
                                     "primary_is_fallback", "summary", "hourly", "intervals",
                                     "dam_known_through", "model_errors", "market")}
        lf["stack"] = {k: (fc.get("stack") or {}).get(k) for k in ("summary", "inputs")}
        h = fc.get("hybrid") or {}
        lf["hybrid_meta"] = {k: h.get(k) for k in ("train_days", "training_weather", "rolling_cv",
                                                    "change_basis", "hours_filled_from_persistence",
                                                    "clamped_inputs")}
        lf["feeds"] = {k: {"status": v.get("status"), "detail": v.get("detail")}
                       for k, v in (fc.get("feeds") or {}).items()}

    bt = {}
    try:
        bt["summary"] = json.loads((BACKTEST / "summary.json").read_text())
        bt["summary"].pop("feeds", None)
        bd = BACKTEST / "backtest_daily.csv"
        if bd.exists():
            bt["daily"] = pd.read_csv(bd).to_dict(orient="records")
    except Exception:  # noqa: BLE001
        bt = {}

    data = {
        "generated_at_ct": dt.datetime.now(CT).isoformat(timespec="seconds"),
        "repo_url": _repo_url(),
        "latest_forecast": lf,
        "summary_live": s_live,
        "summary_all": s_all,
        "scorecard": df.to_dict(orient="records"),
        "detail": scored_day_detail(df) if len(df) else {},
        "backtest": bt,
        "history_days": len(history.complete_days(history.hub_hourly())),
    }
    p = DOCS / "data.json"
    p.write_text(json.dumps(_clean(data), indent=1, default=str))
    return str(p)


# ------------------------------------------------------------------ README
def scorecard_markdown(df: pd.DataFrame) -> str:
    s = summarise(df, live_only=True)
    if not s.get("n"):
        n_fc = len([p for p in FORECASTS.glob("*.json")])
        return (f"_No live day scored yet ({n_fc} forecast file(s) committed). A forecast is "
                "scored once ERCOT publishes that day's official DAM prices (~13:30 CT the "
                "day before delivery)._")
    lines = [
        f"**{s['n']} live days scored** ({s['first_date']} → {s['last_date']}). "
        "Target: ERCOT DAM Hub Average, $/MWh. Error = forecast − actual.",
        "",
        "| Model | MAE daily avg | MAE peak HE15-20 | MAE hourly |",
        "|---|---|---|---|",
    ]
    names = {"hybrid": "**Hybrid (primary)**", "stack": "Fundamental stack",
             "persistence": "Benchmark: today's DAM", "seasonal": "Benchmark: same day last week"}
    for m in MODELS:
        if s.get(f"{m}_n"):
            lines.append(f"| {names[m]} | ${_fmt(s.get(f'{m}_mae_atc'))} | "
                         f"${_fmt(s.get(f'{m}_mae_peak'))} | ${_fmt(s.get(f'{m}_mae_hourly'))} |")
    lines += ["",
              f"Skill vs persistence (daily avg): **{_fmt(s.get('skill_vs_persistence_atc_pct'), 1)}%** · "
              f"beat persistence on {s.get('days_beat_persistence', '–')}/{s.get('paired_days', '–')} days · "
              f"direction of change called {_fmt(s.get('direction_hit_pct'), 1)}% "
              f"({s.get('direction_n', 0)} days) · 80% interval coverage "
              f"{_fmt(s.get('interval_coverage_pct'), 1)}%"]
    if s.get("n_fallback"):
        lines.append(f"\n_{s['n_fallback']} day(s) used the fallback stack model because the "
                     "hybrid could not run; they are included above._")
    lines += ["", "| Day | Published (CT) | Forecast | Actual | Error | Today's DAM (benchmark) |",
              "|---|---|---|---|---|---|"]
    d = df[df["live"].astype(bool) & (df["status"] == "scored")].tail(14)
    for _, r in d.iloc[::-1].iterrows():
        lines.append(f"| {r['target_date']} | {str(r['generated_at_ct'])[:16].replace('T', ' ')} | "
                     f"${_fmt(r['fc_atc'])} | ${_fmt(r['act_atc'])} | {_fmt(r['err_atc'])} | "
                     f"${_fmt(r['persistence_atc'])} |")
    return "\n".join(lines)


def update_readme(df: pd.DataFrame | None = None) -> None:
    if not README.exists():
        return
    df = load_scorecard() if df is None else df
    txt = README.read_text()
    if MARK_START not in txt or MARK_END not in txt:
        return
    a, rest = txt.split(MARK_START, 1)
    _, b = rest.split(MARK_END, 1)
    README.write_text(a + MARK_START + "\n" + scorecard_markdown(df) + "\n" + MARK_END + b)


# -------------------------------------------------------------------- note
def render_note(fc: dict) -> str:
    s = fc.get("summary", {})
    prim = fc["primary_model"]
    p = s.get(prim, {})
    iv = fc.get("intervals") or {}
    pers = s.get("persistence", {})
    st = (fc.get("stack") or {}).get("summary") or {}
    lines = [
        f"# ERCOT day-ahead call for {fc['target_date']}",
        "",
        f"Published {fc['generated_at_ct'][:16].replace('T', ' ')} CT · market closes "
        f"{fc['dam_close_ct'][:16].replace('T', ' ')} CT · "
        f"{'**pre-close (live)**' if fc.get('live') else '**NOT a live call**'} · "
        f"primary model: `{prim}`{' (FALLBACK)' if fc.get('primary_is_fallback') else ''}",
        "",
        "| ERCOT DAM Hub Average | Forecast | 80% range | Today's DAM |",
        "|---|---|---|---|",
        f"| Daily average (24h) | **${_fmt(p.get('atc'))}** | ${_fmt(iv.get('atc_p10'))} – "
        f"${_fmt(iv.get('atc_p90'))} | ${_fmt(pers.get('atc'))} |",
        f"| Peak HE15-20 | **${_fmt(p.get('peak'))}** | ${_fmt(iv.get('peak_p10'))} – "
        f"${_fmt(iv.get('peak_p90'))} | ${_fmt(pers.get('peak'))} |",
        f"| On-peak HE7-22 | ${_fmt(p.get('onpeak'))} | | ${_fmt(pers.get('onpeak'))} |",
        "",
    ]
    if st:
        lines += ["## Fundamentals (stack model)", "",
                  f"Peak load {_fmt(st.get('peak_load_mw'), 0)} MW · peak net load "
                  f"{_fmt(st.get('peak_net_load_mw'), 0)} MW · wind (peak block) "
                  f"{_fmt(st.get('wind_peak_block_mw'), 0)} MW · solar (peak block) "
                  f"{_fmt(st.get('solar_peak_block_mw'), 0)} MW · high {_fmt(st.get('max_temp_f'), 0)}°F "
                  f"· marginal unit at peak {st.get('marginal_unit_peak')} · stack peak-block price "
                  f"${_fmt((s.get('stack') or {}).get('peak'))}", ""]
    lines += ["## Feeds", "", "| feed | status | detail |", "|---|---|---|"]
    for k, v in (fc.get("feeds") or {}).items():
        lines.append(f"| {k} | {v.get('status')} | {str(v.get('detail')).replace('|', '/')} |")
    if fc.get("model_errors"):
        lines += ["", "Model errors: " + json.dumps(fc["model_errors"])]
    return "\n".join(lines) + "\n"


def save_note(fc: dict) -> str:
    p = NOTES / f"{fc['target_date']}.md"
    p.write_text(render_note(fc))
    return str(p)
