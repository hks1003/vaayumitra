"""Robustness: recompute poster headline numbers for sub-periods of the 10-yr ERA5 simulation."""
import glob
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"
DAYS, PRICE, CO2, KIT, FLEET = 200, 100.0, 2.68, 12_000, 10_000
P = {"Draft (Jan15-Jun16)": ("2015-01-01", "2016-06-30"), "5y 2015-19": ("2015", "2019"),
     "5y 2020-24": ("2020", "2024"), "7y 2015-21": ("2015", "2021"), "7y 2018-24": ("2018", "2024"),
     "9y 2016-24": ("2016", "2024"), "Final 10y 2015-24": ("2015", "2024")}
d = pd.concat(pd.read_parquet(f, columns=["date", "boat", "speed_kn", "dep_hour", "harbour", "fuel_base_L",
              "fuel_soft_L", "save_soft_L", "save_soft_pct"]) for f in glob.glob(str(RES / "trips_*_v*.parquet")))
d = d[d.dep_hour == 4]; d["date"] = pd.to_datetime(d.date)
rows = []
for name, (a, b) in P.items():
    x = d[(d.date >= pd.Timestamp(a)) & (d.date <= pd.Timestamp(b) + pd.offsets.YearEnd(0) if len(b) == 4 else d.date <= b)]
    x = d[(d.date >= pd.Timestamp(a)) & (d.date <= (pd.Timestamp(b) + pd.offsets.YearEnd(0) if len(b) == 4 else pd.Timestamp(b)))]
    f6 = x[(x.boat == "frp") & (x.speed_kn == 6)]
    hab = x[(x.boat == "frp") & (x.speed_kn == 8)].fuel_base_L.mean()
    pkgL = (hab - f6.fuel_soft_L.mean()) * DAYS
    mon = f6.assign(m=f6.date.dt.month).groupby(["harbour", "m"]).save_soft_pct.mean()
    t6 = x[(x.boat == "trawler") & (x.speed_kn == 6)]
    rows.append({"period": name, "years": round((x.date.max() - x.date.min()).days / 365.25, 1),
                 "trips(all boats/speeds)": len(x) * 1,
                 "FRP sail-only saving %": f6.save_soft_pct.mean(),
                 "best month %": mon.max(), "best": f"{mon.idxmax()[0][:5]}-{mon.idxmax()[1]}",
                 "Chennai %": f6[f6.harbour == "Chennai"].save_soft_pct.mean(),
                 "Kochi %": f6[f6.harbour == "Kochi"].save_soft_pct.mean(),
                 "Vizag %": f6[f6.harbour == "Visakhapatnam"].save_soft_pct.mean(),
                 "payback months": 12 * KIT / (f6.save_soft_L.mean() * DAYS * PRICE),
                 "FRP 8kn->14? L/trip": f6.fuel_soft_L.mean(),
                 "Rs lakh/boat/yr": pkgL * PRICE / 1e5, "t CO2/boat/yr": pkgL * CO2 / 1000,
                 "Rs cr /10k boats": pkgL * PRICE * FLEET / 1e7, "kt CO2 /10k boats": pkgL * CO2 * FLEET / 1e6,
                 "trawler sail-only %": t6.save_soft_pct.mean()})
t = pd.DataFrame(rows).rename(columns={"FRP 8kn->14? L/trip": "FRP 6kn+sail L/trip"})
t.to_csv(RES / "period_comparison.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(t.round(2).T)
