"""ERCOT supply stack: merit-order dispatch with ORDC scarcity pricing.

Price is set where the vertical net-load line crosses the marginal-cost staircase.
In an energy-only market that is not the whole story: when reserves tighten, ERCOT
adds an administrative adder equal to the expected value of lost load. Most of the
annual P&L in ERCOT lives in the few hundred hours where that adder fires.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import CFG, HR_MODEL, REF_YEAR, Config

# Approximate ERCOT gas fleet composition when unit-level data is unavailable.
# Shares are of installed gas capacity; vintages set the heat-rate spread.
FLEET_MIX = {
    "CCGT":      (0.635, 2007, 11, 260),
    "CT":        (0.160, 2004, 14, 85),
    "STEAM_GAS": (0.180, 1975, 9, 300),
    "ICE_GAS":   (0.025, 2018, 4, 18),
}


def synth_fleet(capacity: dict, seed: int = 7) -> pd.DataFrame:
    """Build a unit-level fleet anchored to ERCOT's published installed capacity."""
    rng = np.random.default_rng(seed)
    gas_mw = float(capacity.get("gas", 69000.0))
    rows = []
    for cat, (share, vy, vsd, unit_mw) in FLEET_MIX.items():
        tot = gas_mw * share
        n = max(1, int(round(tot / unit_mw)))
        yrs = np.clip(rng.normal(vy, vsd, n).round(), 1958, 2026).astype(int)
        rows += [dict(cat=cat, mw=tot / n, year=int(y)) for y in yrs]
    for cat, key, unit_mw, vy, vsd in [("COAL", "coal", 600, 1980, 8),
                                       ("NUCLEAR", "nuclear", 1250, 1990, 3)]:
        tot = float(capacity.get(key, 0.0))
        if tot <= 0:
            continue
        n = max(1, int(round(tot / unit_mw)))
        yrs = np.clip(rng.normal(vy, vsd, n).round(), 1965, 2015).astype(int)
        rows += [dict(cat=cat, mw=tot / n, year=int(y)) for y in yrs]

    stor = float(capacity.get("storage", 0.0))
    if stor > 0:
        n = max(1, int(round(stor / 100.0)))
        rows += [dict(cat="STORAGE", mw=stor / n, year=2023) for _ in range(n)]

    f = pd.DataFrame(rows)
    f["unit"] = [f"{c}_{i}" for i, c in enumerate(f["cat"])]
    return f


def fuel_price(fuel_type: str, cfg: Config) -> float:
    return {"gas": cfg.henry_hub + cfg.hsc_basis,
            "coal": cfg.coal_prb,
            "nuclear": cfg.nuclear_fuel}.get(fuel_type, float("nan"))


def build_stack(fleet: pd.DataFrame, cfg: Config = CFG,
                outage: float | None = None, ambient: float | None = None) -> pd.DataFrame:
    """Assign heat rates and marginal costs, apply derates, sort into merit order."""
    out = cfg.forced_outage_rate + cfg.planned_outage_rate if outage is None else outage
    amb = cfg.ambient_derate_gas if ambient is None else ambient

    f = fleet.copy()
    storage = f[f["cat"] == "STORAGE"].copy()
    f = f[f["cat"] != "STORAGE"].copy()
    cat = f["cat"].astype(str)
    yr = pd.to_numeric(f["year"], errors="coerce").fillna(REF_YEAR)
    base = cat.map(lambda c: HR_MODEL.get(c, {}).get("base", np.nan)).astype(float)
    slope = cat.map(lambda c: HR_MODEL.get(c, {}).get("per_year", 0.0)).astype(float)
    lo = cat.map(lambda c: HR_MODEL.get(c, {}).get("lo", -np.inf)).astype(float)
    hi = cat.map(lambda c: HR_MODEL.get(c, {}).get("hi", np.inf)).astype(float)
    f["heat_rate"] = np.clip(base + slope * (yr - REF_YEAR), lo, hi)
    f["fuel_type"] = cat.map(lambda c: HR_MODEL.get(c, {}).get("fuel"))
    f["vom"] = cat.map(lambda c: getattr(cfg, HR_MODEL.get(c, {}).get("vom", ""), 0.0)
                       if HR_MODEL.get(c) else 0.0)
    f = f.dropna(subset=["heat_rate", "fuel_type"]).copy()

    f["fuel_price"] = f["fuel_type"].map(lambda x: fuel_price(x, cfg))
    f["mc"] = f["heat_rate"] * f["fuel_price"] + f["vom"]

    f["avail_mw"] = f["mw"] * (1 - np.clip(out, 0, 0.9))
    f.loc[f["fuel_type"] == "gas", "avail_mw"] *= (1 - amb)
    # Nuclear runs baseload regardless of the economic derate.
    f.loc[f["cat"] == "NUCLEAR", "avail_mw"] = f.loc[f["cat"] == "NUCLEAR", "mw"] * 0.94

    # Batteries are not a fuel-cost resource: they offer at opportunity cost and are
    # limited by duration, so only part of the fleet is available in any one hour.
    if cfg.include_storage and len(storage):
        storage["heat_rate"] = 0.0
        storage["fuel_type"] = "storage"
        storage["vom"] = 0.0
        storage["fuel_price"] = 0.0
        storage["mc"] = cfg.storage_offer
        storage["avail_mw"] = storage["mw"] * cfg.storage_peak_availability
        f = pd.concat([f, storage], ignore_index=True)

    f = f.sort_values("mc").reset_index(drop=True)
    f["cum_mw"] = f["avail_mw"].cumsum()
    return f


def _norm_cdf(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


class StackArrays:
    """Numpy view of a stack so thousands of intervals clear at once."""

    def __init__(self, stack: pd.DataFrame):
        self.mc = stack["mc"].to_numpy(float)
        self.cum = stack["cum_mw"].to_numpy(float)
        self.avail = stack["avail_mw"].to_numpy(float)
        self.hr = stack["heat_rate"].to_numpy(float)
        self.cat = stack["cat"].to_numpy(object)
        is_gas = (stack["fuel_type"] == "gas").to_numpy()
        self.is_gas = is_gas
        self.cum_gas_btu = np.cumsum(self.avail * self.hr * is_gas)
        self.cum_gas_mw = np.cumsum(self.avail * is_gas)
        self.total = float(self.avail.sum())


def clear(net_loads, sa: StackArrays, cfg: Config = CFG, reserves=None) -> dict:
    """Vectorised merit-order clearing with the ORDC adder."""
    L = np.asarray(net_loads, dtype=float)
    n = len(sa.cum)
    idx = np.searchsorted(sa.cum, L, side="left")
    idx_c = np.clip(idx, 0, n - 1)
    short = idx >= n
    zero = L <= 0
    served = np.minimum(L, sa.total)

    energy = np.where(short, sa.mc[-1], sa.mc[idx_c])
    energy = np.where(zero, cfg.wind_offer, energy)

    over = np.maximum(0.0, sa.cum[idx_c] - served)
    gas_btu = sa.cum_gas_btu[idx_c] - np.where(sa.is_gas[idx_c], over * sa.hr[idx_c], 0.0)
    gas_mw = sa.cum_gas_mw[idx_c] - np.where(sa.is_gas[idx_c], over, 0.0)
    gas_btu = np.where(zero, 0.0, gas_btu)
    gas_mw = np.where(zero, 0.0, gas_mw)

    res = (sa.total - L) if reserves is None else np.asarray(reserves, dtype=float)
    lolp = _norm_cdf((cfg.ordc_min_contingency - res) / max(cfg.ordc_sigma, 1.0))
    ordc = np.maximum(0.0, lolp * (cfg.voll - energy))
    ordc = np.where(short, cfg.voll - energy, ordc)
    ordc = np.where(zero, 0.0, ordc)

    price = np.clip(energy + ordc, cfg.price_floor, cfg.voll)
    marg_hr = np.where(short | zero, np.nan, sa.hr[idx_c])
    cats = np.where(short, "SCARCITY",
                    np.where(zero, "RENEWABLE_OVERSUPPLY", sa.cat[idx_c]))
    return {"price": price, "energy_price": energy, "ordc": ordc, "reserves": res,
            "marginal_hr": marg_hr, "marginal_cat": cats,
            "gas_mw": gas_mw, "gas_burn_mmbtu_h": gas_btu}


def mmbtu_h_to_bcfd(x):
    return np.asarray(x, dtype=float) * 24.0 / 1_000_000.0


def implied_heat_rate(power_price, gas_price):
    return np.asarray(power_price, float) / gas_price if gas_price else np.nan


def spark_spread(power_price, gas_price, hr=7.0, vom=2.5):
    return np.asarray(power_price, float) - hr * gas_price - vom
