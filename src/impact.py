"""
VaayuMitra impact calculator: litres -> rupees, payback, CO2, fleet scale-up.

Assumptions (user-provided local figures, 2026-11-15):
  sail kit cost : FRP boat Rs 12,000 ; trawler Rs 65,000  (soft sail; wing = 3x, flagged as estimate)
  fishing days  : 200 per year
  diesel price  : Rs 100 per litre (sensitivity Rs 121, Mar-2026 AP price)
  CO2           : 2.68 kg per litre diesel
Fleet impact is reported per 10,000 boats (no verified national craft count used).

Two separate claims (kept apart for honest attribution):
  A. Sail only    : same speed, with vs without sail
  B. Package      : advisor (6 kn + sail) vs current practice (8 kn, no sail); includes slow steaming
"""
import glob
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DAYS = 200
PRICE = 100.0
PRICE_HI = 121.0
CO2 = 2.68
KIT = {("frp", "soft"): 12_000, ("trawler", "soft"): 65_000,
       ("frp", "wing"): 36_000, ("trawler", "wing"): 195_000}  # wing = 3x soft (estimate)
FLEET = 10_000


def main():
    d = pd.concat(pd.read_parquet(f) for f in glob.glob(str(RES / "trips_*_v*.parquet")))
    d = d[d.dep_hour == 4]  # current-practice departure for the headline numbers
    rows = []
    for (boat, v), g in d.groupby(["boat", "speed_kn"]):
        for s in ("soft", "wing"):
            save = g[f"save_{s}_L"].mean()
            rows.append(dict(boat=boat, speed_kn=v, sail=s, claim="A_sail_only",
                             fuel_before_L=g.fuel_base_L.mean(), fuel_after_L=g[f"fuel_{s}_L"].mean(),
                             save_L_trip=save))
    for boat in d.boat.unique():
        hab = d[(d.boat == boat) & (d.speed_kn == 8)].fuel_base_L.mean()
        slow = d[(d.boat == boat) & (d.speed_kn == 6)]
        for s in ("soft", "wing"):
            rows.append(dict(boat=boat, speed_kn=6, sail=s, claim="B_package_6kn_plus_sail",
                             fuel_before_L=hab, fuel_after_L=slow[f"fuel_{s}_L"].mean(),
                             save_L_trip=hab - slow[f"fuel_{s}_L"].mean()))
    t = pd.DataFrame(rows)
    t["save_pct"] = 100 * t.save_L_trip / t.fuel_before_L
    t["save_L_yr"] = t.save_L_trip * DAYS
    t["inr_yr"] = t.save_L_yr * PRICE
    t["inr_yr_at121"] = t.save_L_yr * PRICE_HI
    t["kit_inr"] = [KIT[(b, s)] for b, s in zip(t.boat, t.sail)]
    # payback only meaningful for the sail investment itself (claim A); package needs no extra capex
    t["payback_months"] = 12 * t.kit_inr / t.inr_yr
    t["co2_t_yr"] = t.save_L_yr * CO2 / 1000
    t[f"fleet{FLEET}_ML_diesel_yr"] = t.save_L_yr * FLEET / 1e6
    t[f"fleet{FLEET}_kt_co2_yr"] = t.co2_t_yr * FLEET / 1000
    t[f"fleet{FLEET}_crore_inr_yr"] = t.inr_yr * FLEET / 1e7
    t = t.round(2)
    t.to_csv(RES / "impact_table.csv", index=False)
    (RES / "impact_assumptions.json").write_text(json.dumps(
        dict(days=DAYS, price=PRICE, price_hi=PRICE_HI, co2_kg_L=CO2, kit_inr={f"{a}_{b}": c for (a, b), c in KIT.items()},
             fleet=FLEET, departure="04:00 IST"), indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t[["boat", "speed_kn", "sail", "claim", "fuel_before_L", "fuel_after_L", "save_pct", "inr_yr",
                 "payback_months", "co2_t_yr", f"fleet{FLEET}_crore_inr_yr", f"fleet{FLEET}_kt_co2_yr"]])


if __name__ == "__main__":
    main()
