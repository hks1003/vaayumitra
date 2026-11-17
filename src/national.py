"""
Objectives 3 + 5 - validation against published trials and national / state scale-up.

National scale-up uses the CMFRI Marine Fisheries Census 2016 count of motorised craft (97,659;
66,250 outboard + 31,409 inboard, as cited in Indian J. Fisheries) with the SAIL-ONLY saving of the
generic 8.5 m FRP boat at 6 kn (claim A) - a screening estimate, since not every motorised craft is
an 8.5 m FRP boat. State lines use the coast-map regional mean saving per trip.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DAYS, PRICE, CO2 = 200, 100.0, 2.68
CRAFT = {"India": 97_659, "Andhra Pradesh": 12_078, "Kerala": 13_868}  # CMFRI census 2016 (motorised)

imp = pd.read_csv(RES / "impact_table.csv")
a6 = imp.query("boat == 'frp' and speed_kn == 6 and sail == 'soft' and claim == 'A_sail_only'").iloc[0]
L_yr = a6.save_L_trip * DAYS
cm = pd.read_parquet(RES / "coast_map.parquet").query("month == 0")
reg = {"Andhra Pradesh": cm[(cm.lat >= 14.0) & (cm.lat < 18.9) & (cm.lon > 79.5)].save_L_trip.mean(),
       "Kerala": cm[(cm.lon < 77.5) & (cm.lat >= 8.5) & (cm.lat < 12.8)].save_L_trip.mean()}
out = {"sail_only_L_per_boat_yr": L_yr, "sail_only_inr_per_boat_yr": L_yr * PRICE,
       "sail_only_t_co2_per_boat_yr": L_yr * CO2 / 1000}
n = CRAFT["India"]
out["india"] = dict(craft=n, ML_diesel=L_yr * n / 1e6, crore=L_yr * n * PRICE / 1e7, kt_co2=L_yr * n * CO2 / 1e6)
for s, Lt in reg.items():
    c = CRAFT[s]
    out[s] = dict(craft=c, L_trip=Lt, crore=Lt * DAYS * c * PRICE / 1e7, kt_co2=Lt * DAYS * c * CO2 / 1e6)
out["validation"] = {
    "sail_trials_pct": {"FAO range (variable wind)": [5, 80], "Baltic retrofit (Germany)": 19,
                        "33 ft vessel, Norway, 10-20 kn wind": [10, 15], "47 ft vessel, US appraisal": 37},
    "model_sail_pct": {"FRP 6 kn mean": float(a6.save_pct), "FRP 8 kn mean": float(
        imp.query("boat == 'frp' and speed_kn == 8 and sail == 'soft' and claim == 'A_sail_only'").save_pct.iloc[0])},
    "speed_cut_fuel_per_nm": {"FAO 10 m trawler trial": {"-10%": 28, "-20%": 51},
                              "model FRP": {"-10%": 18, "-20%": 34}, "model trawler": {"-10%": 19, "-20%": 35}}}
(RES / "national_validation.json").write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=1, default=lambda x: round(float(x), 2)))
