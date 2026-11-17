"""
Objective 1 - Coast-wide sail-saving potential map for India (10-yr ERA5, 2015-2024).

Each ERA5 wave-grid (0.5 deg) sea cell that touches land along the Indian mainland coast gets a
generic FRP trip: steam 25 nm straight offshore (perpendicular to the local coast, from the land
mask) at 6 kn -> drift 6 h -> steam back; leave 04:00 IST every day. Wind/wave from that cell
(uniform along the route: a screening simplification, stated on the poster).
Outputs: results/coast_map.parquet (per-cell, per-month mean saving), figures/F5_coast_map.png
"""
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from physics import BOATS, KN, SAIL_SETS, Boat, apparent_wind, fuel_rate, sail_net_saving  # noqa: E402

ERA, RES = ROOT / "data" / "era5", ROOT / "results"
logging.basicConfig(filename=ROOT / "logs" / "coast_map.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
DIST, V_KN, DRIFT_H, DEP_IST = 25.0, 6.0, 6, 4


def month_files():
    return sorted(Path(p) for p in glob.glob(str(ERA / "era5_*_*")) if Path(p).is_dir())


def coastal_cells():
    """Sea cells (valid swh) with >=1 land neighbour (NaN swh), Indian mainland only; offshore bearing."""
    ds = xr.open_dataset(next(month_files()[0].glob("*wave*.nc")))
    swh = ds["swh"].isel(valid_time=0)
    la, lo = swh.latitude.values, swh.longitude.values
    sea = np.isfinite(swh.values)
    out = []
    for i in range(2, len(la) - 2):
        for j in range(2, len(lo) - 2):
            if not sea[i, j]:
                continue
            n_e = n_n = 0.0
            nland = 0
            for di in range(-2, 3):          # 5x5 window smooths the coastline direction
                for dj in range(-2, 3):
                    if (di or dj) and not sea[i + di, j + dj]:
                        if abs(di) <= 1 and abs(dj) <= 1:
                            nland += 1           # coastal = land in the 3x3 neighbourhood
                        r2 = di * di + dj * dj
                        n_n -= (la[i + di] - la[i]) / r2   # vector pointing away from land
                        n_e -= (lo[j + dj] - lo[j]) / r2
            if nland == 0:
                continue
            lat, lon = float(la[i]), float(lo[j])
            # mainland India only: drop Sri Lanka, Maldives, Bangladesh/Myanmar, Pakistan
            if (lat < 10.0 and lon > 79.6) or lat < 7.6 or lon > 89.0 or (lat > 21.5 and lon < 69.0):
                continue
            out.append((lat, lon, np.degrees(np.arctan2(n_e, n_n)) % 360, nland))
    ds.close()
    c = pd.DataFrame(out, columns=["lat", "lon", "bearing", "nland"])
    return c[c.nland <= 6].reset_index(drop=True)   # drop cells almost enclosed by land (creeks)


def extract(c):
    lats, lons = xr.DataArray(c.lat.values, dims="cell"), xr.DataArray(c.lon.values, dims="cell")
    T, U, V, H = [], [], [], []
    for d in tqdm(month_files(), desc="extract coast"):
        w = xr.open_dataset(next(d.glob("*oper*.nc")))
        s = xr.open_dataset(next(d.glob("*wave*.nc")))
        T.append(w.valid_time.values)
        U.append(w.u10.sel(latitude=lats, longitude=lons, method="nearest").transpose("valid_time", "cell").values)
        V.append(w.v10.sel(latitude=lats, longitude=lons, method="nearest").transpose("valid_time", "cell").values)
        H.append(s.swh.sel(latitude=lats, longitude=lons, method="nearest").transpose("valid_time", "cell").values)
        w.close(); s.close()
    return pd.to_datetime(np.concatenate(T)), *(np.concatenate(a).astype(np.float32) for a in (U, V, H))


def route(boat, brg):
    t_leg = DIST / V_KN
    n = int(np.ceil(t_leg))
    hrs, hdg, mode, V = [], [], [], []
    for k in range(n):
        hrs.append(min(1.0, t_leg - k)); hdg.append(brg); mode.append("steam"); V.append(V_KN * KN)
    for k in range(DRIFT_H):
        hrs.append(1.0); hdg.append((brg + (90 if k % 2 == 0 else -90)) % 360); mode.append(boat.op_mode)
        V.append(boat.V_trawl_kn * KN)
    for k in range(n):
        hrs.append(min(1.0, t_leg - k)); hdg.append((brg + 180) % 360); mode.append("steam"); V.append(V_KN * KN)
    return np.array(hrs), np.radians(hdg), np.array(mode), np.array(V)


def main():
    boat = Boat(**BOATS["frp"], V_run_kn=V_KN)
    sail = SAIL_SETS[boat.sail_set]["soft"]
    c = coastal_cells()
    logging.info("coastal cells %d", len(c))
    t, U, Vv, H = extract(c)
    H = np.where(np.isnan(H), np.nanmean(H, axis=1, keepdims=True), H)
    t0 = t[0]
    days = pd.date_range(t0.normalize() + pd.Timedelta(days=1), t[-1].normalize() - pd.Timedelta(days=1), freq="D")
    base = ((days + pd.Timedelta(hours=DEP_IST - 5.5) - t0) / pd.Timedelta(hours=1)).values
    rows = []
    for k, r in tqdm(c.iterrows(), total=len(c), desc="cells"):
        hrs, hdg, mode, Vb1 = route(boat, r.bearing)
        offs = np.concatenate([[0], np.cumsum(hrs)[:-1]])
        ti = np.clip(np.rint(base[:, None] + offs[None, :]).astype(int), 0, len(t) - 1)
        u, v, hs = U[ti, k], Vv[ti, k], H[ti, k]
        tws, twd = np.hypot(u, v), np.arctan2(-u, -v)
        D, S = tws.shape
        Vb, hd = np.broadcast_to(Vb1, (D, S)), np.broadcast_to(hdg, (D, S))
        aws, awa = apparent_wind(tws, twd, hd, Vb)
        _, _, _, net = sail_net_saving(boat, sail, aws.ravel(), awa.ravel(), Vb.ravel(), hs.ravel(), tws.ravel())
        net = net.reshape(D, S)
        net[:, mode == boat.op_mode] = 0.0
        f0, f1 = np.zeros((D, S)), np.zeros((D, S))
        for m in ("steam", boat.op_mode):
            s_ = mode == m
            f0[:, s_] = fuel_rate(boat, m, Vb[:, s_].ravel(), hs[:, s_].ravel()).reshape(D, -1)
            f1[:, s_] = fuel_rate(boat, m, Vb[:, s_].ravel(), hs[:, s_].ravel(), net[:, s_].ravel()).reshape(D, -1)
        F0, F1 = (f0 * hrs).sum(1), (f1 * hrs).sum(1)
        df = pd.DataFrame(dict(m=days.month, F0=F0, F1=F1))
        g = df.groupby("m").sum()
        for mth, gg in g.iterrows():
            rows.append(dict(lat=r.lat, lon=r.lon, bearing=r.bearing, month=mth,
                             save_pct=100 * (gg.F0 - gg.F1) / gg.F0, save_L_trip=(gg.F0 - gg.F1) / (days.month == mth).sum()))
        rows.append(dict(lat=r.lat, lon=r.lon, bearing=r.bearing, month=0,
                         save_pct=100 * (F0 - F1).sum() / F0.sum(), save_L_trip=(F0 - F1).mean()))
    out = pd.DataFrame(rows)
    out.to_parquet(RES / "coast_map.parquet")
    a = out[out.month == 0]
    logging.info("done; annual saving %% range %.1f-%.1f", a.save_pct.min(), a.save_pct.max())
    print(a.describe().round(2))




def plot():
    """F5: annual sail-only saving at every coastal cell; land drawn from the ERA5 wave land mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import figures as F
    a = pd.read_parquet(RES / "coast_map.parquet").query("month == 0")
    ds = xr.open_dataset(next(month_files()[0].glob("*wave*.nc")))
    sw = ds.swh.isel(valid_time=0)
    land = np.where(np.isfinite(sw.values), np.nan, 1.0)
    LA, LO = np.meshgrid(sw.latitude.values, sw.longitude.values, indexing="ij")
    land[(LA < 10.0) & (LO > 79.6)] = np.nan          # hide Sri Lanka (not in study)
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.pcolormesh(sw.longitude, sw.latitude, land, cmap=LinearSegmentedColormap.from_list("l", [F.GRID, F.GRID]),
                  shading="nearest", zorder=0)
    ds.close()
    cmap = LinearSegmentedColormap.from_list("seq", ["#ffffd9", "#41b6c4", "#081d58"])
    sc = ax.scatter(a.lon, a.lat, c=a.save_pct, cmap=cmap, s=70, vmin=0, vmax=np.ceil(a.save_pct.max() / 5) * 5,
                    edgecolors=F.SURF, linewidths=1.2, zorder=3)
    for name, (la, lo, _) in {"Chennai": (13.12, 80.30, 0), "Kochi": (10.18, 76.17, 0),
                              "Visakhapatnam": (17.69, 83.30, 0)}.items():
        ax.plot(lo, la, marker="*", ms=14, color="#111111", mec=F.SURF, mew=1, zorder=4)
        ax.annotate(name, (lo, la), xytext=(-8, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=9, color=F.INK, zorder=5)
    ax.set_xlim(68.6, 90.2); ax.set_ylim(7.4, 22.2); ax.set_aspect(1.03)
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.01)
    cb.set_label("Annual fuel saved by sail (%)")
    ax.set_title("Where sails pay most", loc="left", color=F.INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "F5_coast_map.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


if __name__ == "__main__":
    main() if "plot" not in sys.argv else None
    plot()
