"""
VaayuMitra ML layer.

1. Surrogate: predicts litres saved by the sail for a planned trip from FORECAST-type inputs
   (winds/waves at departure, mid-trip, return; route; boat; sail; month; departure hour).
   - main model : GPU MLP ensemble (5 members), PyTorch
   - baseline   : LightGBM
   - uncertainty: split-conformal 90 % intervals (calibration year held out)
2. Advisor: for each planned trip, pick the departure hour with the highest predicted saving.
   Evaluated against (a) habit = always leave 04:00 and (b) oracle = best hour in hindsight,
   with perfect forecasts and with realistic forecast error (u,v noise sigma = 1.5 m/s).
3. Explainability: SHAP (TreeExplainer on LightGBM) -> simple rules for fishers.

Temporal split: train <= 2021, calibration 2022, test 2023-2024 (falls back to day-based
split when fewer years are available, e.g. smoke tests).

Run: .venv/Scripts/python.exe src/ml.py
"""
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
logging.basicConfig(filename=ROOT / "logs" / "ml.log", level=logging.INFO, format="%(asctime)s %(message)s")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

WIND = ["u_dep", "v_dep", "u_mid", "v_mid", "u_ret", "v_ret"]
NUM = WIND + ["tws_mean", "hs_mean", "hs_max", "bearing", "dist_nm", "dep_hour", "speed_kn", "m_sin", "m_cos"]
CAT = ["harbour", "boat", "sail"]
TARGET = "save_L"


def load_long():
    """trips_<boat>.parquet (wide over sails) -> long table, one row per trip x sail."""
    parts = []
    for f in sorted(RES.glob("trips_*.parquet")):
        d = pd.read_parquet(f)
        sails = sorted(c[5:-2] for c in d.columns if c.startswith("save_") and c.endswith("_L"))
        for s in sails:
            x = d[["date", "harbour", "boat", "speed_kn", "trip_h", "bearing", "dist_nm", "dep_hour", "fuel_base_L",
                   "tws_mean", "hs_mean", "hs_max"] + WIND].copy()
            x["sail"] = s
            x["save_L"] = d[f"save_{s}_L"]
            parts.append(x)
    df = pd.concat(parts, ignore_index=True)
    m = pd.to_datetime(df["date"]).dt.month
    df["m_sin"], df["m_cos"] = np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    return df


def split(df):
    yrs = sorted(df.year.unique())
    if len(yrs) >= 4:
        tr = df.year <= yrs[-4] if yrs[-1] - yrs[0] < 6 else df.year <= 2021
        ca = df.year == (2022 if yrs[-1] >= 2024 else yrs[-3])
        te = ~(tr | ca)
    else:  # smoke-test fallback: split by day of month
        dd = pd.to_datetime(df["date"]).dt.day
        tr, ca, te = dd <= 18, (dd > 18) & (dd <= 24), dd > 24
    return tr.values, ca.values, te.values


def design(df, stats=None):
    X = df[NUM].astype(np.float32).copy()
    for c in CAT:
        for lv in sorted(df[c].unique()) if stats is None else stats["levels"][c]:
            X[f"{c}={lv}"] = (df[c] == lv).astype(np.float32)
    if stats is None:
        stats = {"mu": X.mean().to_dict(), "sd": (X.std() + 1e-6).to_dict(),
                 "levels": {c: sorted(df[c].unique()) for c in CAT}, "cols": list(X.columns)}
    X = X.reindex(columns=stats["cols"], fill_value=0.0)
    Xn = (X - pd.Series(stats["mu"])) / pd.Series(stats["sd"])
    return Xn.values.astype(np.float32), stats


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, Xva, yva, member, epochs=40, bs=4096, writer=None):
    torch.manual_seed(SEED + member)
    m = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, yt = torch.tensor(Xtr, device=DEV), torch.tensor(ytr, device=DEV)
    Xv, yv = torch.tensor(Xva, device=DEV), torch.tensor(yva, device=DEV)
    best, state = 1e9, None
    for ep in tqdm(range(epochs), desc=f"MLP member {member}", leave=False):
        m.train()
        perm = torch.randperm(len(Xt), device=DEV)
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = nn.functional.huber_loss(m(Xt[idx]), yt[idx], delta=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            vl = torch.sqrt(((m(Xv) - yv) ** 2).mean()).item()
        if writer:
            writer.add_scalar(f"val_rmse/member{member}", vl, ep)
        if vl < best:
            best, state = vl, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state)
    return m


def predict(models, X):
    with torch.no_grad():
        Xt = torch.tensor(X, device=DEV)
        P = torch.stack([m(Xt) for m in models]).cpu().numpy()
    return P.mean(0), P.std(0)


def metrics(y, p):
    return {"rmse_L": float(np.sqrt(np.mean((y - p) ** 2))), "mae_L": float(np.mean(np.abs(y - p))),
            "r2": float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))}


def advisor_eval(te_df, pred_col):
    """Choose dep hour per (date, harbour, bearing, dist, boat, sail) by predicted saving."""
    g = ["date", "harbour", "bearing", "dist_nm", "boat", "speed_kn", "sail"]
    pick = te_df.loc[te_df.groupby(g)[pred_col].idxmax()].set_index(g)["save_L"]
    habit = te_df[te_df.dep_hour == 4].set_index(g)["save_L"]
    oracle = te_df.groupby(g)["save_L"].max()
    base = te_df[te_df.dep_hour == 4].set_index(g)["fuel_base_L"]
    out = pd.DataFrame({"advisor": pick, "habit": habit, "oracle": oracle, "base": base}).dropna()
    return out


def main():
    df = load_long()
    tr, ca, te = split(df)
    print(f"rows {len(df):,}  train {tr.sum():,}  calib {ca.sum():,}  test {te.sum():,}  device {DEV}")
    Xtr, st = design(df[tr])
    Xca, _ = design(df[ca], st)
    Xte, _ = design(df[te], st)
    y = df[TARGET].values.astype(np.float32)
    writer = SummaryWriter(ROOT / "logs" / "tb")

    # --- GPU MLP ensemble ---
    models = [train_mlp(Xtr, y[tr], Xca, y[ca], k, writer=writer) for k in range(5)]
    p_ca, _ = predict(models, Xca)
    p_te, sd_te = predict(models, Xte)
    # split-conformal 90 % interval (absolute residual score)
    q90 = float(np.quantile(np.abs(y[ca] - p_ca), 0.9 * (1 + 1 / ca.sum())))
    cover = float(np.mean(np.abs(y[te] - p_te) <= q90))

    # --- LightGBM baseline + SHAP ---
    Xl = df[NUM + CAT].copy()
    for c in CAT:
        Xl[c] = Xl[c].astype("category")
    gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=SEED)
    gbm.fit(Xl[tr], y[tr], eval_set=[(Xl[ca], y[ca])], callbacks=[lgb.early_stopping(50, verbose=False)])
    g_te = gbm.predict(Xl[te])

    res = {"mlp_ensemble": metrics(y[te], p_te), "lightgbm": metrics(y[te], g_te),
           "conformal_q90_L": q90, "conformal_coverage_test": cover,
           "mean_save_L_test": float(y[te].mean())}

    # --- advisor evaluation: perfect forecast and noisy forecast ---
    tdf = df[te].copy()
    tdf["pred"] = p_te
    adv = advisor_eval(tdf, "pred")
    rng = np.random.default_rng(SEED)
    noisy = df[te].copy()
    noisy[WIND] = noisy[WIND] + rng.normal(0, 1.5, size=noisy[WIND].shape)  # ~24 h forecast error
    noisy["tws_mean"] = noisy["tws_mean"] + rng.normal(0, 1.0, size=len(noisy))
    Xno, _ = design(noisy, st)
    tdf["pred_noisy"], _ = predict(models, Xno)
    adv_n = advisor_eval(tdf, "pred_noisy")
    for name, a in (("advisor_perfect_forecast", adv), ("advisor_noisy_forecast", adv_n)):
        res[name] = {k: float(a[k].mean()) for k in ("advisor", "habit", "oracle", "base")}
        res[name]["gain_vs_habit_pct"] = float(100 * (a.advisor.mean() / a.habit.mean() - 1))
        res[name]["share_of_oracle_pct"] = float(100 * (a.advisor.mean() - a.habit.mean()) /
                                                 max(a.oracle.mean() - a.habit.mean(), 1e-9))

    # --- SHAP on LightGBM (sample for speed) ---
    import shap
    samp = Xl[te].sample(min(5000, int(te.sum())), random_state=SEED)
    sv = shap.TreeExplainer(gbm).shap_values(samp)
    imp = pd.Series(np.abs(sv).mean(0), index=samp.columns).sort_values(ascending=False)
    res["shap_mean_abs"] = imp.round(4).to_dict()
    np.save(RES / "shap_values.npy", sv)
    samp.to_parquet(RES / "shap_sample.parquet")

    torch.save({"state": [m.state_dict() for m in models], "stats": st, "q90": q90}, RES / "surrogate.pt")
    gbm.booster_.save_model(str(RES / "lightgbm.txt"))
    (RES / "ml_results.json").write_text(json.dumps(res, indent=2))
    logging.info(json.dumps(res))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
