"""
Objective 2 - Sensitivity of the headline sail-only result (FRP boat, 6 kn, soft sail, 04:00, 10-yr ERA5).

Physics knobs (re-simulated): sail area +-20 %, sail CLmax +-20 %, sail parasitic drag CD0 +-20 %,
boat calibration fuel rate +-20 %.  Economic knobs (analytic): diesel Rs 90-121/L, kit Rs 8k-20k,
fishing days 150-250.  Output: payback months + Rs/yr per knob -> results/sensitivity.csv,
figures/F6_sensitivity.png (tornado, one axis).
"""
import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
import simulate as S  # noqa: E402
from physics import BOATS, SAIL_SETS, Boat  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
DAYS, PRICE, KIT = 200, 100.0, 12_000
S.DEP_HOURS_IST = [4]
TMP = Path(tempfile.mkdtemp())
S.OUT = TMP / "trips_{boat}_v{v}.parquet"
BASE_SAIL = SAIL_SETS["small"]["soft"]


def sim(sail_kw=None, boat_kw=None):
    """Mean sail-only saving (L/trip, %) for FRP 6 kn soft sail under modified parameters."""
    SAIL_SETS["small"]["soft"] = dataclasses.replace(BASE_SAIL, **(sail_kw or {}))
    boat = Boat(**{**BOATS["frp"], **(boat_kw or {})}, V_run_kn=6.0)
    S.run(S.build_routes(boat), boat, "frp")
    d = pd.read_parquet(TMP / "trips_frp_v6.parquet")
    SAIL_SETS["small"]["soft"] = BASE_SAIL
    return d.save_soft_L.mean(), d.save_soft_pct.mean()


def econ(L, price=PRICE, kit=KIT, days=DAYS):
    inr = L * days * price
    return 12 * kit / inr, inr


def main():
    L0, p0 = sim()
    rows = [dict(knob="Baseline", side="", L=L0, pct=p0)]
    B = BASE_SAIL
    phys = {"Sail area ±20 %": ("sail", "area", B.area), "Sail max lift CLmax ±20 %": ("sail", "CLmax", B.CLmax),
            "Sail drag CD0 ±20 %": ("sail", "CD0", B.CD0),
            "Boat fuel calibration ±20 %": ("boat", "fuel_steam_Lph", BOATS["frp"]["fuel_steam_Lph"])}
    for knob, (kind, field, v) in phys.items():
        for side, f in (("low", 0.8), ("high", 1.2)):
            kw = {field: v * f}
            L, p = sim(sail_kw=kw) if kind == "sail" else sim(boat_kw=kw)
            rows.append(dict(knob=knob, side=side, L=L, pct=p))
    t = pd.DataFrame(rows)
    t["payback_m"], t["inr_yr"] = zip(*[econ(L) for L in t.L])
    for knob, kw in {"Diesel price ₹90–121/L": [dict(price=90), dict(price=121)],
                     "Sail kit cost ₹8k–20k": [dict(kit=20_000), dict(kit=8_000)],
                     "Fishing days 150–250/yr": [dict(days=150), dict(days=250)]}.items():
        for side, k in zip(("low", "high"), kw):
            pb, inr = econ(L0, **k)
            t.loc[len(t)] = dict(knob=knob, side=side, L=L0, pct=p0, payback_m=pb, inr_yr=inr)
    t.round(3).to_csv(RES / "sensitivity.csv", index=False)
    print(t.round(2).to_string())
    (RES / "sensitivity_summary.json").write_text(json.dumps(
        dict(base_payback=float(t.payback_m[0]), worst_payback=float(t.payback_m.max()),
             best_payback=float(t.payback_m.min()), base_pct=float(p0),
             pct_range=[float(t.pct.min()), float(t.pct.max())]), indent=2))
    plot(t)


def plot(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figures  # noqa: F401  (applies the poster rcParams / palette)
    base = t.payback_m[0]
    g = t[t.knob != "Baseline"].groupby("knob").payback_m.agg(["min", "max"])
    g = g.assign(span=g["max"] - g["min"]).sort_values("span")
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    for i, (k, r) in enumerate(g.iterrows()):
        ax.barh(i, r["max"] - r["min"], left=r["min"], color=figures.CAT[0], height=0.55)
        ax.text(r["max"] + 0.08, i, f"{r['min']:.1f}–{r['max']:.1f}", va="center", fontsize=14, color=figures.INK2)
    ax.axvline(base, color=figures.INK2, lw=1)
    ax.set_yticks(range(len(g)), g.index, fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, g["max"].max() * 1.25)
    ax.set_xlabel("Payback of ₹12k sail (months)", fontsize=14)
    ax.set_title(f"Payback under {g['max'].max():.0f} months in every case",
                 loc="left", color=figures.INK, fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG / "F6_sensitivity.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
