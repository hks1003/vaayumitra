"""
Extra poster figures for the LaTeX poster.
F7: sail-only saving vs mean wind speed (FRP 6 kn, 04:00 departures) with share of trips; shows safety
    reefing above ~11 m/s.
F8: like-for-like comparison of measured whole-trip savings in published trials vs our model (technical potential).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import figures as F  # noqa: E402  (shared rcParams + palette)

RES, FIG = F.RES, F.FIG


def f7_wind():
    d = pd.read_parquet(RES / "trips_frp_v6.parquet", columns=["dep_hour", "tws_mean", "save_soft_pct"])
    d = d[d.dep_hour == 4]
    lab = ["<3", "3–5", "5–7", "7–9", "9–11", ">11"]
    g = d.groupby(pd.cut(d.tws_mean, [0, 3, 5, 7, 9, 11, 30], labels=lab), observed=True).agg(
        p=("save_soft_pct", "mean"), n=("save_soft_pct", "size"))
    g["share"] = 100 * g.n / g.n.sum()
    xl = [f"{c}\n{s:.0f}%" for c, s in zip(lab, g.share)]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    bars = ax.bar(xl, g.p, color=[F.CAT[0]] * 5 + [F.CAT[1]], width=0.62, edgecolor="#333333", lw=0.5)
    bars[-1].set_hatch("///")
    for b, p in zip(bars, g.p):
        ax.text(b.get_x() + b.get_width() / 2, p + 0.8, f"{p:.0f} %", ha="center", fontsize=12, color=F.INK)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Mean trip wind (m/s); share of trips below")
    ax.set_ylabel("Fuel saved by sail (%)")
    ax.set_ylim(0, g.p.max() * 1.45)
    ax.grid(axis="x", visible=False)
    ax.annotate("sail reefed or\nfurled for safety", xy=(5, g.p.iloc[-1] + 3.5), xytext=(4.75, g.p.max() * 1.22),
                fontsize=10, color=F.CAT[1], ha="center", arrowprops=dict(arrowstyle="->", color=F.CAT[1]))
    ax.set_title("Saving grows with wind until safety limits bite", loc="left", fontsize=13, color=F.INK)
    fig.tight_layout()
    fig.savefig(FIG / "F7_wind_bins.png", dpi=300)
    plt.close(fig)
    return g


def f8_validation():
    rows = [("Norfolk Rebel 1983 (US)", 6, "measured"),
            ("Kerala BOBP 1981–83", 10, "measured"),
            ("Ours: Kochi", 8, "model"), ("Ours: Chennai", 15, "model"), ("Ours: Visakhapatnam", 20, "model")]
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    for i, (n, v, k) in enumerate(rows):
        ax.barh(i, v, color=F.CAT[1] if k == "measured" else F.CAT[0], height=0.6,
                hatch="" if k == "measured" else "//", edgecolor="#333333", lw=0.5)
        ax.text(v + 0.3, i, f"{v} %", va="center", fontsize=12, color=F.INK)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 31)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Whole-trip fuel saved by sail (%)")
    ax.legend(handles=[Patch(facecolor=F.CAT[1], edgecolor="#333", label="measured trials"),
                       Patch(facecolor=F.CAT[0], hatch="//", edgecolor="#333", label="our model (potential)")],
              frameon=False, loc="upper right", fontsize=10)
    ax.set_title("Whole trip, same speed, all weather", loc="left", fontsize=13, color=F.INK)
    fig.tight_layout()
    fig.savefig(FIG / "F8_validation.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    print(f7_wind().round(1))
    f8_validation()
