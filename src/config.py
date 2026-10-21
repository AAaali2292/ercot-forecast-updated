"""Configuration and physical assumptions.

Everything a trader would argue about lives here. Nothing is hidden in the model code.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

# ERCOT_ROOT lets the test-suite run against a scratch directory.
ROOT = Path(os.environ.get("ERCOT_ROOT") or Path(__file__).resolve().parent.parent)
FORECASTS = ROOT / "forecasts"     # one JSON per target day, committed before the market closes
OUTCOMES = ROOT / "outcomes"       # one JSON per settled day (real-time + dashboard DAM snapshot)
NOTES = ROOT / "notes"             # the daily written note
DOCS = ROOT / "docs"               # GitHub Pages site
DATA = ROOT / "data"               # official DAM price history (ERCOT MIS settlement files)
BACKTEST = ROOT / "backtest"       # simulated walk-forward results - kept apart from the live record
CACHE = ROOT / ".cache"
for _d in (FORECASTS, OUTCOMES, NOTES, DOCS, DATA, BACKTEST, CACHE):
    _d.mkdir(parents=True, exist_ok=True)

SCORECARD = ROOT / "scorecard.csv"
DAM_HISTORY = DATA / "dam_history.csv"
CT = ZoneInfo("America/Chicago")

EIA_API_KEY = os.environ.get("EIA_API_KEY", "").strip()

# ---------------------------------------------------------------------------
# TIME CONVENTION - read this before touching any hourly join.
#
# ERCOT labels hours by HOUR ENDING. HE1 is 00:00-01:00 CT and is stamped 01:00.
# HE24 is 23:00-24:00 and is stamped 00:00 of the NEXT calendar day. Reading those
# stamps with .dt.hour / .dt.date (as the first version of this repo did) shifts every
# hour by one and files HE24 under the wrong day.
#
# Everything in this repo is therefore keyed by (operating day, HE 1..24), computed by
# src.timeutil.hour_ending() from tz-aware timestamps. Weather, EIA demand and prices
# all go through that one function, so they line up by construction.
# ---------------------------------------------------------------------------
HUB = "HB_HUBAVG"                      # the forecast target: ERCOT Hub Average DAM SPP
DASH_HUB_KEYS = {                      # dashboard JSON key -> official settlement point
    "hbHubAvg": "HB_HUBAVG", "hbBusAvg": "HB_BUSAVG", "hbHouston": "HB_HOUSTON",
    "hbNorth": "HB_NORTH", "hbPan": "HB_PAN", "hbSouth": "HB_SOUTH", "hbWest": "HB_WEST",
}
HUBS = list(DASH_HUB_KEYS.values())

PEAK_HE = list(range(15, 21))          # HE15-HE20: the ERCOT evening peak block
ONPEAK_HE = list(range(7, 23))         # HE7-HE22: standard on-peak (16 hours)
# Kept for the fundamental stack model, which works on clock hours (hour-beginning).
PEAK_HOURS = [h - 1 for h in PEAK_HE]

# The day-ahead market for day T closes at 10:00 CT on T-1. A forecast stamped after
# that could have seen the market's answer, so it is never scored as a live call.
DAM_CLOSE_HOUR_CT = 10


@dataclass
class Config:
    # ---- fuel -------------------------------------------------------------
    henry_hub: float = 3.20          # $/MMBtu, overwritten by live price each run
    hsc_basis: float = -0.05         # Houston Ship Channel vs HH
    coal_prb: float = 2.10
    nuclear_fuel: float = 0.75

    # ---- VOM $/MWh --------------------------------------------------------
    vom_ccgt: float = 2.50
    vom_ct: float = 5.00
    vom_steam: float = 4.00
    vom_coal: float = 4.50
    vom_nuclear: float = 2.20
    vom_ice: float = 6.00

    # ---- renewable offers (why ERCOT prints negative prices) --------------
    wind_offer: float = -22.0
    solar_offer: float = -5.0

    # ---- ERCOT scarcity pricing ------------------------------------------
    voll: float = 5000.0             # system-wide offer cap
    price_floor: float = -251.0
    ordc_min_contingency: float = 3000.0
    ordc_sigma: float = 1100.0

    # ---- availability -----------------------------------------------------
    forced_outage_rate: float = 0.05
    planned_outage_rate: float = 0.06
    ambient_derate_gas: float = 0.06

    # ---- storage ----------------------------------------------------------
    # ~21 GW of batteries. They offer at opportunity cost and are duration-limited.
    # Leaving them out makes the stack run short and fires ORDC on ordinary days.
    include_storage: bool = True
    storage_offer: float = 75.0
    storage_peak_availability: float = 0.70

    # ---- stack-model uncertainty (Monte Carlo) ----------------------------
    n_sims: int = 400
    load_sigma_frac: float = 0.035
    wind_sigma_frac: float = 0.22
    solar_sigma_frac: float = 0.10
    outage_sigma: float = 0.035

    # ---- hybrid DAM model -------------------------------------------------
    hybrid_train_days: int = 120       # trailing window of target days used to fit
    hybrid_min_train_days: int = 21    # below this the hybrid refuses to forecast
    hybrid_halflife_days: float = 30.0 # recency weighting inside the window
    hybrid_ridge: float = 1.0          # L2 penalty on standardised coefficients
    hybrid_huber_k: float = 1.345      # robust loss: spikes cannot dominate the fit
    hybrid_cv_days: int = 28           # rolling-origin refits used to size the interval

    # ---- thresholds -------------------------------------------------------
    scarcity_threshold: float = 200.0  # $/MWh

    # Do not score an outcome we only partially captured.
    min_coverage_pct: float = 80.0


CFG = Config()

# Heat-rate model. base = a unit built in REF_YEAR; per_year = efficiency drift.
REF_YEAR = 2000
HR_MODEL = {
    "CCGT":      dict(base=7.85,  per_year=-0.032, lo=6.10, hi=9.60,  vom="vom_ccgt",    fuel="gas"),
    "CT":        dict(base=11.30, per_year=-0.035, lo=8.40, hi=15.00, vom="vom_ct",      fuel="gas"),
    "STEAM_GAS": dict(base=10.60, per_year=-0.012, lo=9.20, hi=13.50, vom="vom_steam",   fuel="gas"),
    "ICE_GAS":   dict(base=9.30,  per_year=-0.010, lo=8.20, hi=10.50, vom="vom_ice",     fuel="gas"),
    "COAL":      dict(base=10.45, per_year=-0.008, lo=9.20, hi=12.50, vom="vom_coal",    fuel="coal"),
    "NUCLEAR":   dict(base=10.45, per_year=0.0,    lo=10.0, hi=11.00, vom="vom_nuclear", fuel="nuclear"),
}

# Load-weighted ERCOT metros (weights approximate metro load share).
LOAD_CITIES = [
    ("Houston",     29.76,  -95.37, 0.26),
    ("Dallas",      32.78,  -96.80, 0.24),
    ("San Antonio", 29.42,  -98.49, 0.12),
    ("Austin",      30.27,  -97.74, 0.11),
    ("Midland",     31.99, -102.08, 0.10),
    ("Corpus",      27.80,  -97.40, 0.05),
    ("McAllen",     26.20,  -98.23, 0.06),
    ("Waco",        31.55,  -97.15, 0.06),
]

# Wind sites, weighted by roughly where ERCOT wind capacity sits.
WIND_SITES = [
    ("Panhandle",   35.22, -101.83, 0.34),
    ("West TX",     32.45, -101.48, 0.31),
    ("Permian",     31.45, -102.35, 0.12),
    ("South TX",    27.55,  -97.85, 0.18),
    ("Gulf Coast",  28.80,  -96.60, 0.05),
]

# Solar sites: West Texas dominates utility-scale PV.
SOLAR_SITES = [
    ("West TX",     31.85, -102.40, 0.42),
    ("Central TX",  31.10,  -98.20, 0.25),
    ("North TX",    33.20,  -97.60, 0.18),
    ("South TX",    28.40,  -98.30, 0.15),
]

HDD_BASE_F = 65.0
CDD_BASE_F = 65.0


def config_dict() -> dict:
    return asdict(CFG)
