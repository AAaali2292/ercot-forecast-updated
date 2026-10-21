# ERCOT Day-Ahead Price Forecast

**Every morning, before ERCOT's day-ahead market closes, this repo publishes tomorrow's hourly Hub Average price — then scores itself against the price that actually cleared.**

The forecast is generated at about 06:00 CT by a scheduled GitHub Action and committed to this repo. The market closes at 10:00 CT. ERCOT publishes the cleared prices at about 13:30 CT, and an afternoon run scores the call against ERCOT's official settlement file. The evidence sits outside the code. Each forecast is produced inside a GitHub Actions run, and GitHub records that run's start time and logs (the run link is saved in every forecast file). The file is pushed to this public repo minutes later. Commit dates alone could be faked; GitHub writes the run records and push times, and the author cannot edit them. A repo admin could delete a run, and GitHub keeps run logs for 90 days by default, so the run link is also copied into each forecast file.

**[Live dashboard →](https://AAaali2292.github.io/ercot-forecast/)**

---

## Live track record

<!-- SCORECARD:START -->
_No live day scored yet (1 forecast file(s) committed). A forecast is scored once ERCOT publishes that day's official DAM prices (~13:30 CT the day before delivery)._
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

| UTC | Central | Job |
|---|---|---|
| 11:07 | 06:07 CDT / 05:07 CST | `daily` — refresh prices, forecast tomorrow, score, publish |
| 12:37 | 07:37 CDT / 06:37 CST | backup forecast, runs only if the first one is missing |
| 19:53, 22:23 | afternoon | `dam` — fetch tomorrow's cleared prices, score, publish |
| 04:47, 05:47, 06:47 | around midnight | `capture` — real-time prices and generation |
| Sun 09:17 | | `weekly` — refresh annual history file, re-run backtest |

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
