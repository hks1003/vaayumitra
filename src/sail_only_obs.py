"""Sail-only (claim A) observations, final 10-yr ERA5 period 2015-2024."""
import glob
from pathlib import Path
import pandas as pd, numpy as np
RES = Path(__file__).resolve().parents[1] / "results"
d = pd.concat(pd.read_parquet(f) for f in glob.glob(str(RES / "trips_*_v*.parquet")))
d["date"] = pd.to_datetime(d.date); d["m"] = d.date.dt.month; d["y"] = d.date.dt.year
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
print("== boat x speed x sail: mean %, mean L, sail hours, trip_h")
for s in ("soft", "wing"):
    print(s); print(d.groupby(["boat", "speed_kn"]).agg(pct=(f"save_{s}_pct", "mean"), L=(f"save_{s}_L", "mean"),
          sailh=(f"sail_hours_{s}", "mean"), trip=("trip_h", "mean")).round(2))
f = d[(d.boat == "frp") & (d.speed_kn == 6)]
print("== FRP 6kn soft: distribution"); q = f.save_soft_pct
print(q.describe(percentiles=[.1, .25, .5, .75, .9]).round(1), " zero-saving share", (q < 0.5).mean().round(3))
print("== by departure hour"); print(f.groupby("dep_hour").save_soft_pct.mean().round(2))
print("== harbour x bearing"); print(f.groupby(["harbour", "bearing"]).save_soft_pct.mean().unstack().round(1))
print("== harbour x dist"); print(f.groupby(["harbour", "dist_nm"]).save_soft_pct.mean().unstack().round(1))
print("== harbour x month"); print(f.groupby(["harbour", "m"]).save_soft_pct.mean().unstack().round(1))
print("== wind bins"); f = f.assign(wb=pd.cut(f.tws_mean, [0, 3, 5, 7, 9, 11, 30]))
print(f.groupby("wb", observed=True).agg(pct=("save_soft_pct", "mean"), n=("save_soft_pct", "size"),
      sailh=("sail_hours_soft", "mean")).round(2))
print("== Hs>2.5 or tws>14 trips share", ((f.hs_max > 2.5) | (f.tws_mean > 14)).mean().round(4))
print("== year to year (harbour)"); yy = f.groupby(["harbour", "y"]).save_soft_pct.mean().unstack(0)
print(yy.round(1)); print("CV%", (100 * yy.std() / yy.mean()).round(1).to_dict())
print("== wing vs soft per harbour (FRP 6)"); print(f.groupby("harbour")[["save_soft_pct", "save_wing_pct"]].mean().round(1))
