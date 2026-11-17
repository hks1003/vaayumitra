"""
Poster v4 figures (LaTeX poster): larger text, fewer elements, readable at A2.
F3b  SHAP: top-6 drivers of the sail saving
F4b  surrogate parity with conformal band (square, big labels)
F9   live forecast day: diesel for three trip plans (replaces the app screenshot)
Same font (TeX Gyre Termes) and style as src/figures.py.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import figures as F  # noqa: E402  (sets fonts, rcParams, FIG_SCALE)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

plt.rcParams.update({"font.size": 15})
G = ["#0B4D1C", "#2E8B3E", "#8CC596"]       # dark / mid / light green


def f3b():
    r = json.loads((F.RES / "ml_results.json").read_text())
    s = pd.Series(r["shap_mean_abs"]).sort_values().tail(6)
    names = {"tws_mean": "Mean wind speed", "boat": "Boat type", "dist_nm": "Distance to ground",
             "sail": "Sail type", "bearing": "Ground direction", "u_dep": "E-W wind, departure",
             "v_ret": "N-S wind, return", "u_mid": "E-W wind, mid-trip", "u_ret": "E-W wind, return",
             "speed_kn": "Steaming speed", "hs_mean": "Wave height", "v_dep": "N-S wind, departure"}
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    cols = [G[2]] * (len(s) - 2) + [G[1], G[0]]
    ax.barh([names.get(i, i) for i in s.index], s.values, color=cols, height=0.62)
    for y, v in enumerate(s.values):
        ax.text(v + s.max() * 0.02, y, f"{v:.2f}", va="center", fontsize=13, color=F.INK2)
    ax.set_xlim(0, s.max() * 1.22)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean |SHAP| (litres per trip)")
    ax.set_title("What drives the saving", loc="left", color=F.INK, fontsize=16)
    fig.tight_layout()
    fig.savefig(F.FIG / "F3b_shap.png", dpi=300)
    plt.close(fig)


def f4b():
    from ml import MLP, design, load_long, predict, split  # noqa: E402
    ck = torch.load(F.RES / "surrogate.pt", weights_only=False)
    df = load_long()
    _, _, te = split(df)
    X, _ = design(df[te], ck["stats"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = []
    for sd in ck["state"]:
        m = MLP(X.shape[1]).to(dev)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    p, _ = predict(models, X)
    y = df[te].save_L.values
    q = ck["q90"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.hexbin(y, p, gridsize=50, bins="log", cmap="Greens", mincnt=1, linewidths=0)
    lim = [min(y.min(), p.min()), np.percentile(y, 99.5)]
    xs = np.linspace(*lim, 50)
    ax.plot(xs, xs, color=F.INK, lw=1.2, label="perfect")
    ax.plot(xs, xs - q, color="#C0392B", lw=1.6, ls="--", label=f"90% band ±{q:.1f} L")
    ax.plot(xs, xs + q, color="#C0392B", lw=1.6, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Physics saving (L/trip)")
    ax.set_ylabel("ML saving (L/trip)")
    ax.set_title("Unseen 2023–24 trips", loc="left", color=F.INK, fontsize=16)
    ax.legend(frameon=False, loc="upper left", fontsize=13)
    fig.tight_layout()
    fig.savefig(F.FIG / "F4b_parity.png", dpi=300)
    plt.close(fig)


def f9():
    e = json.loads((F.RES / "live_example.json").read_text())
    rows = [("Habit\n04:00, 8 kn, no sail", e["habit"]["fuel"], G[2], ""),
            (f"Same trip,\nsail up {e['sail_only']['sail']:.1f} h", e["sail_only"]["fuel"], G[1], "//"),
            (f"VaayuMitra\n{e['best']['dep']:02d}:00, {e['best']['speed']} kn, sail {e['best']['sail']:.1f} h",
             e["best"]["fuel"], G[0], "")]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    base = rows[0][1]
    for i, (lab, v, c, h) in enumerate(rows[::-1]):
        ax.barh(i, v, color=c, hatch=h, edgecolor="white", height=0.62)
        txt = f"{v:.1f} L" + ("" if v == base else f"  (−{100 * (1 - v / base):.0f}%)")
        ax.text(v + 0.4, i, txt, va="center", fontsize=15, color=F.INK, fontweight="bold" if i == 0 else None)
    ax.set_yticks(range(3), [r[0] for r in rows[::-1]], fontsize=14)
    ax.set_xlim(0, base * 1.45)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Diesel for the trip (litres)")
    ax.set_title("Chennai, 16 Nov 2026 forecast", loc="left", color=F.INK, fontsize=16)
    fig.tight_layout()
    fig.savefig(F.FIG / "F9_live_plan.png", dpi=300)
    plt.close(fig)


def f8b():
    from matplotlib.patches import Patch
    rows = [("Trial: Norfolk Rebel, US", 6, "m"), ("Trial: Kerala BOBP", 10, "m"),
            ("Ours: Kochi", 8, "o"), ("Ours: Chennai", 15, "o"), ("Ours: Vizag", 20, "o")]
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for i, (n, v, k) in enumerate(rows):
        ax.barh(i, v, color="#C9A227" if k == "m" else G[1], height=0.62,
                hatch="" if k == "m" else "//", edgecolor="white")
        ax.text(v + 0.4, i, f"{v} %", va="center", fontsize=15, color=F.INK, fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, 30)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Whole-trip fuel saved by the sail (%)")
    # bar labels already say trial vs ours: no legend needed
    ax.set_title("Measured vs modelled saving", loc="left", color=F.INK, fontsize=16)
    fig.tight_layout()
    fig.savefig(F.FIG / "F8b_validation.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    f8b()
    f3b()
    f9()
    if "noparity" not in sys.argv:
        f4b()
    print("ok")
