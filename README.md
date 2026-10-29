# ERCOT Day-Ahead Price Forecast

**Every morning, before ERCOT's day-ahead market closes, this repo publishes tomorrow's hourly Hub Average price — then scores itself against the price that actually cleared.**

A call for delivery day T can be made from the moment ERCOT publishes day-ahead prices for T-1 (about 13:30 CT on T-2) until T's market closes (10:00 CT on T-1). Every scheduled GitHub Action, about 11 a day, makes the next call if it is missing, so a late-starting job cannot miss the window. On 22 Oct 2026 GitHub started jobs 4-5 hours late, which is why the design changed. Once T's price clears, the call is scored against ERCOT's official settlement file. The evidence sits outside the code. Each forecast is produced inside a GitHub Actions run, and GitHub records that run's start time and logs (the run link is saved in every forecast file). The file is pushed to this public repo minutes later. Commit dates alone could be faked; GitHub writes the run records and push times, and the author cannot edit them. A repo admin could delete a run, and GitHub keeps run logs for 90 days by default, so the run link is also copied into each forecast file.

**[Live dashboard →](https://aaaali2292.github.io/ercot-forecast-updated/)**

---

## Live track record

<!-- SCORECARD:START -->
**7 live days scored** (2026-10-24 → 2026-10-30). Target: ERCOT DAM Hub Average, $/MWh. Error = forecast − actual.

| Model | MAE daily avg | MAE peak HE15-20 | MAE hourly |
|---|---|---|---|
| **Hybrid (primary)** | $3.33 | $6.16 | $6.22 |
| Fundamental stack | $7.37 | $13.30 | $10.13 |
| Benchmark: today's DAM | $4.73 | $8.38 | $8.39 |
| Benchmark: same day last week | $8.35 | $11.51 | $12.35 |

Skill vs persistence (daily avg): **29.6%** · beat persistence on 3/7 days · direction of change called 66.7% (6 days) · 80% interval coverage 100.0%

| Day | Published (CT) | Forecast | Actual | Error | Today's DAM (benchmark) |
|---|---|---|---|---|---|
| 2026-10-30 | 2026-10-28 13:29 | $34.97 | $39.92 | -4.95 | $38.30 |
| 2026-10-29 | 2026-10-27 15:44 | $34.12 | $38.30 | -4.19 | $26.94 |
| 2026-10-28 | 2026-10-26 13:51 | $31.57 | $26.94 | 4.64 | $31.96 |
| 2026-10-27 | 2026-10-25 14:05 | $35.13 | $31.96 | 3.17 | $34.27 |
| 2026-10-26 | 2026-10-24 13:49 | $32.49 | $34.27 | -1.78 | $22.27 |
| 2026-10-25 | 2026-10-23 14:58 | $23.59 | $22.27 | 1.32 | $21.60 |
| 2026-10-24 | 2026-10-22 17:48 | $18.33 | $21.60 | -3.28 | $21.78 |
<!-- SCORECARD:END -->

*Updated automatically. Only calls stamped before the 10:00 CT market close are counted. Error = forecast − actual, $/MWh.*

---

## What it forecasts

**ERCOT day-ahead market, Hub Average (HB_HUBAVG) settlement point price, $/MWh, for each hour HE1–HE24 of tomorrow.** Headline numbers: the 24-hour average, the HE15–20 evening peak block, and an 80% range.

## The models

| | What it is | Role |
|---|---|---|
| **Hybrid** | Starts from today's cleared DAM price (public before the call), then adjusts for the *forecast change* in temperature, wind and solar from today to tomorrow. Robust (Huber) ridge regression, fitted separately for 4 blocks of the day on the last 120 days, refitted every morning. | **Primary — this is the call that is scored** |
| **Fundamental stack** | Weather → load, wind and solar → net load → merit-order supply stack (heat rates × Henry Hub, ~21 GW of batteries, ERCOT's ORDC scarcity adder), with a 400-path Monte Carlo. | Explains *why* the price is where it is |
| Persistence | Today's DAM price, same hour | Benchmark to beat |
| Seasonal | Same hour, same weekday last week | Benchmark |

Persistence is a hard benchmark for a day-ahead price, and the scorecard reports honestly whether the model beats it.

### Design choices worth defending

1. **Anchor on the market, model the change.** A pure stack model gets the physics right but misses the level of a market that prices in risk, congestion and battery opportunity cost. The hybrid takes the level from today's market and uses the fundamentals to explain the *difference* between today and tomorrow.
2. **No look-ahead, by construction.** Training rows use the weather forecast as it stood about 24 hours before each day (Open-Meteo's previous-runs archive), not the weather that actually happened. A test proves the fit cannot change when future data is scrambled.
3. **Robust loss.** ERCOT prints $1,000+ hours. Under ordinary least squares one spike would dominate the fit; the Huber loss caps its influence. A test checks this.
4. **No extrapolation.** Live inputs are clamped to the range seen in training, and every clamp is logged in the forecast file. This caught a real failure in testing: with separate heating and cooling terms, the first cool October evening after a summer training window moved the price by −$15/MWh.
5. **The interval is out-of-sample.** The 80% range comes from 28 rolling refits, each predicting a day it had not seen.

## Integrity rules

- A forecast stamped after 10:00 CT on the day before delivery is never scored as live.
- Once a live forecast exists, no later run can overwrite it. The only exception: before the close, a fallback forecast can be upgraded to the primary model.
- "Live" is recomputed at scoring time from the timestamps, not trusted from the file, and the clock is re-checked when the file is saved.
- A day whose actual came only from the public dashboard is shown as *provisional*. It stays out of the headline numbers until ERCOT's official file replaces it.
- No day is dropped silently. A day that cannot be scored stays in the scorecard with the reason.
- Actuals come from ERCOT's official MIS settlement files (report NP4-190-CD) and are cross-checked against the public dashboard.
- Every data feed's status is written into each forecast file. A failed feed is reported as failed, never filled in by guesswork.
- The backtest is labelled as simulated and kept separate from the live record.

## Data sources (all public)

| Source | Use | Key |
|---|---|---|
| ERCOT MIS — DAM Settlement Point Prices (NP4-190-CD), Historical DAM Hub Prices (NP4-180-ER) | Actuals and price history, via [`gridstatus`](https://github.com/gridstatus/gridstatus) | none |
| ERCOT public dashboards | Today's DAM, real-time prices, 5-minute fuel mix, installed capacity | none |
| Open-Meteo forecast and previous-runs APIs | Temperature (8 load-weighted metros), 100 m wind (5 wind regions), solar irradiance (4 solar regions) | none |
| EIA-930 (API v2) | Hourly ERCOT demand, which trains the stack model's load model | free |
| EIA Henry Hub spot / NYMEX front month | Gas price for the stack | free |

ERCOT blocks requests from outside the USA. GitHub's runners are US-based, so the scheduled job can reach it, but a laptop outside the US cannot.

## Schedule

About 11 scheduled runs a day (UTC times in `.github/workflows/daily.yml`). Each does its own job, then makes the next call if it is missing:

| Job | What it does |
|---|---|
| `daily`, `backup` (several slots) | make the next live call |
| `dam` (afternoon CT) | fetch the newly cleared DAM prices, score, publish, then make the next call |
| `capture` (around CT midnight) | snapshot real-time prices and generation, then make the next call if missing |
| `weekly` (Sunday) | refresh the annual price file and re-run the backtest |

## Repo layout

```
src/config.py     every assumption, with a comment
src/timeutil.py   hour-ending / DST handling (one place, tested)
src/sources.py    live fetchers with per-feed status logging
src/history.py    official DAM price store (data/dam_history.csv)
src/hybrid.py     the primary model
src/models.py     load, wind, solar models (stack inputs)
src/stack.py      merit order + ORDC
src/forecast.py   assembles the daily call; integrity rule on saving
src/score.py      scoring vs actual and benchmarks
src/backtest.py   simulated walk-forward (kept separate)
src/publish.py    README scorecard, docs/data.json, daily note
forecasts/        one JSON per target day  <- the evidence
outcomes/         real-time snapshot per day
notes/            human-readable daily note
docs/             the dashboard (GitHub Pages)
```

## Known limitations

- **Hub average only.** ERCOT is nodal; individual hubs and nodes separate under congestion.
- **No outage data.** Thermal outages are a statistical derate in the stack model, and the hybrid only sees them through yesterday's price.
- **Weather is the only forward-looking driver in the hybrid.** Gas price moves, outages and transmission events enter only through the lagged price.
- **Short-history fallback.** Without an `EIA_API_KEY` the stack model cannot fit its load model and is skipped. The hybrid does not need the key.

*Public data only. A forecasting exercise, not investment advice.*
