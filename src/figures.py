"""
Poster figures (300 dpi PNG, light theme) for VaayuMitra.

F1 monthly sail-only saving by harbour (seasonality / monsoon)
F2 speed vs fuel trade-off, FRP boat and trawler as separate panels (no dual axis)
F3 SHAP global importance (what drives the saving)
F4 surrogate parity plot with split-conformal 90 % band
Palette: validated default categorical order (blue, orange, aqua) + neutral inks.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
for _f in (Path(__file__).resolve().parents[1] / "latex" / "fonts").glob("texgyretermes-*.otf"):
    font_manager.fontManager.addfont(str(_f))  # same font files as the LaTeX poster
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

CAT = ["#2a78d6", "#eb6834", "#1baf7a"]          # v3: three greens (dark/mid/light), told apart by
LS = ["-", "--", ":"]                            # ...line style, marker shape and hatch as well
MK = ["o", "s", "^"]
HATCH = ["", "///", "..."]
SEQ = "YlGnBu"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
HARB = ["Chennai", "Kochi", "Visakhapatnam"]
plt.rcParams.update({"font.size": 13, "font.family": ["TeX Gyre Termes", "Times New Roman"], "mathtext.fontset": "custom",
                     "mathtext.rm": "TeX Gyre Termes", "mathtext.it": "TeX Gyre Termes:italic",
                     "mathtext.bf": "TeX Gyre Termes:bold", "axes.edgecolor": AXIS,
                     "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

# Poster readability: draw every figure at 75 % of its nominal size with unchanged font sizes,
# so text is ~1.33x larger relative to the plot once placed on the A1 poster.
FIG_SCALE = 0.75
_subplots = plt.subplots


def _scaled_subplots(*a, figsize=None, **k):
    if figsize is not None:
        figsize = (figsize[0] * FIG_SCALE, figsize[1] * FIG_SCALE)
    return _subplots(*a, figsize=figsize, **k)


plt.subplots = _scaled_subplots


def load():
    return pd.concat(pd.read_parquet(f) for f in glob.glob(str(RES / "trips_*_v*.parquet")))


def f1_monthly(d):
    x = d[(d.boat == "frp") & (d.speed_kn == 6) & (d.dep_hour == 4)].copy()
    x["month"] = pd.to_datetime(x.date).dt.month
    x["cut"] = 100 * ((x.hs_max > 2.5) | (x.tws_mean > 14))
    m = x.groupby(["harbour", "month"]).save_soft_pct.mean().unstack(0)
    c = x.groupby(["harbour", "month"]).cut.mean().unstack(0)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.1]})
    for i, h in enumerate(HARB):
        ax.plot(m.index, m[h], color=CAT[i], lw=2.4, ls=LS[i], marker=MK[i], ms=8, mec=SURF, mew=1.5, label=h)
        bx.bar(c.index + (i - 1) * 0.27, c[h], width=0.25, color=CAT[i], hatch=HATCH[i], edgecolor="#333333", lw=0.5)
    ax.set_ylabel("Fuel saved by sail (%)")
    ax.set_title("Sail saving by month - 8.5 m FRP boat, 6 kn", loc="left", color=INK, fontsize=14)
    for a_ in (ax, bx):
        a_.axvspan(5.5, 9.5, color=GRID, alpha=0.5, lw=0, zorder=0)
    ax.text(7.5, 1.0, "SW monsoon", ha="center", va="bottom", color=MUTED, fontsize=11)
    ax.legend(frameon=False, loc="upper right", fontsize=10, ncol=3, bbox_to_anchor=(1, 1.02))
    ax.set_ylim(0, m.values.max() * 1.25)
    bx.set_ylabel("Sail lowered\nfor safety (%)", fontsize=10)
    bx.set_ylim(0, max(c.values.max() * 1.25, 1))
    bx.grid(axis="x", visible=False)
    bx.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    fig.tight_layout()
    fig.savefig(FIG / "F1_monthly_saving.png", dpi=300)
    plt.close(fig)
    return m


def f2_speed(d):
    x = d[d.dep_hour == 4].groupby(["boat", "speed_kn"])[["fuel_base_L", "fuel_soft_L", "trip_h"]].mean()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, b, ttl in zip(axs, ["frp", "trawler"], ["8.5 m FRP boat", "15 m trawler"]):
        g = x.loc[b].sort_index(ascending=False)
        ax.plot(g.trip_h, g.fuel_base_L, color=CAT[0], lw=2, marker="o", ms=8, mec=SURF, mew=2, label="No sail")
        ax.plot(g.trip_h, g.fuel_soft_L, color=CAT[1], lw=2, marker="o", ms=8, mec=SURF, mew=2, label="With sail")
        for v, r in g.iterrows():
            ax.annotate(f"{v:g} kn", (r.trip_h, r.fuel_base_L), xytext=(0, 9), textcoords="offset points",
                        ha="center", color=MUTED, fontsize=10)
        ax.set_title(ttl, loc="left", color=INK, fontsize=14)
        ax.set_xlabel("Trip duration (h)")
        ax.set_ylabel("Fuel per trip (L)")
        ax.set_ylim(0, g.fuel_base_L.max() * 1.18)  # headroom so speed labels clear the title
    axs[0].legend(frameon=False, loc="lower left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "F2_speed_tradeoff.png", dpi=300)
    plt.close(fig)
    return x


def f3_shap():
    r = json.loads((RES / "ml_results.json").read_text())
    s = pd.Series(r["shap_mean_abs"]).sort_values().tail(8)
    names = {"tws_mean": "Mean wind speed", "boat": "Boat type", "dist_nm": "Distance to ground",
             "sail": "Sail type", "bearing": "Ground direction", "u_dep": "E-W wind at departure",
             "v_ret": "N-S wind on return", "u_mid": "E-W wind mid-trip", "u_ret": "E-W wind on return",
             "speed_kn": "Steaming speed", "hs_mean": "Wave height", "v_dep": "N-S wind at departure",
             "harbour": "Harbour", "v_mid": "N-S wind mid-trip", "dep_hour": "Departure hour"}
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh([names.get(i, i) for i in s.index], s.values, color=CAT[0], height=0.6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP| (litres per trip)")
    ax.set_title("What drives the saving", loc="left", color=INK, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "F3_shap.png", dpi=300)
    plt.close(fig)


def f4_parity():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ml import MLP, design, load_long, predict, split  # noqa: E402
    ck = torch.load(RES / "surrogate.pt", weights_only=False)
    df = load_long()
    _, _, te = split(df)
    X, _ = design(df[te], ck["stats"])
    models = []
    for sd in ck["state"]:
        m = MLP(X.shape[1]).to("cuda" if torch.cuda.is_available() else "cpu")
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    p, _ = predict(models, X)
    y = df[te].save_L.values
    r = json.loads((RES / "ml_results.json").read_text())
    q = ck["q90"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    hb = ax.hexbin(y, p, gridsize=60, bins="log", cmap=SEQ, mincnt=1, linewidths=0)
    lim = [min(y.min(), p.min()), np.percentile(y, 99.5)]
    xs = np.linspace(*lim, 50)
    ax.plot(xs, xs, color=INK2, lw=1)
    ax.plot(xs, xs - q, color=CAT[0], lw=1.2, ls="--", label=f"90% conformal band (±{q:.1f} L)")
    ax.plot(xs, xs + q, color=CAT[0], lw=1.2, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Simulated saving (L/trip)")
    ax.set_ylabel("ML predicted saving (L/trip)")
    mm = r["mlp_ensemble"]
    ax.set_title(f"Surrogate: R² {mm['r2']:.3f}, coverage {100 * r['conformal_coverage_test']:.0f}%",
                 loc="left", color=INK, fontsize=13)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.colorbar(hb, ax=ax, label="Trips (log)")
    fig.tight_layout()
    fig.savefig(FIG / "F4_parity.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    print(f1_monthly(d).round(1))
    print(f2_speed(d).round(1))
    f3_shap()
    f4_parity()
    print("figures ->", FIG)
