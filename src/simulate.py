"""
VaayuMitra trip simulator: replay single-day fishing trips over 10 years of ERA5 weather.

Trip = steam out (8 kn) -> trawl TRAWL_H hours (3 kn, tows along the depth contour, alternating
direction) -> steam home. For every day, harbour, ground (bearing x distance), departure hour and
sail type we compute fuel with and without the sail. Route geometry is fixed, so only the ERA5
time series at the grid cells the routes touch are extracted (keeps memory small).

Step 1  python src/simulate.py extract   -> data/cells_timeseries.npz
Step 2  python src/simulate.py run       -> results/trips.parquet
"""
import argparse
import glob
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from physics import KN, CO2_KG_PER_L, SAIL_SETS, BOATS, Boat, apparent_wind, fuel_rate, sail_net_saving  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ERA = ROOT / "data" / "era5"
CELLS = ROOT / "data" / "cells_timeseries.npz"
OUT = ROOT / "results" / "trips_{boat}_v{v}.parquet"
logging.basicConfig(filename=ROOT / "logs" / "simulate.log", level=logging.INFO,
                    format="%(asctime)s %(message)s")

# Harbour (lat, lon) and seaward bearing sector (deg true)
HARBOURS = {
    "Chennai":       (13.12, 80.30, [60, 90, 120]),
    "Kochi":         (10.18, 76.17, [225, 255, 285]),
    "Visakhapatnam": (17.69, 83.30, [100, 130, 160]),
}
DISTS_NM = [15, 25, 35]
DEP_HOURS_IST = [3, 4, 5, 6]
TRAWL_H = 6
GRID = 0.25


def destination(lat, lon, brg_deg, dist_nm):
    """Flat-earth step (fine for <60 nm)."""
    d = dist_nm * 1852.0
    b = np.radians(brg_deg)
    return lat + d * np.cos(b) / 111_320.0, lon + d * np.sin(b) / (111_320.0 * np.cos(np.radians(lat)))


def build_routes(boat: Boat):
    """List of routes; each has per-step arrays: lat, lon, heading(rad), mode, hours, speed(m/s)."""
    routes = []
    for hb, (la, lo, brgs) in HARBOURS.items():
        for brg in brgs:
            for dist in DISTS_NM:
                steps = []
                # outbound: one step per hour of steaming, last step partial
                t_leg = dist / boat.V_run_kn
                n = int(np.ceil(t_leg))
                for k in range(n):
                    h = min(1.0, t_leg - k)
                    mid = (k + 0.5 * h) * boat.V_run_kn
                    steps.append((*destination(la, lo, brg, mid), brg, "steam", h, boat.V_run_kn * KN))
                gla, glo = destination(la, lo, brg, dist)
                for k in range(TRAWL_H):  # tows along contour, alternating direction
                    hd = (brg + (90 if k % 2 == 0 else -90)) % 360
                    steps.append((gla, glo, hd, boat.op_mode, 1.0, boat.V_trawl_kn * KN))
                back = (brg + 180) % 360
                for k in range(n):
                    h = min(1.0, t_leg - k)
                    mid = dist - (k + 0.5 * h) * boat.V_run_kn
                    steps.append((*destination(la, lo, brg, mid), back, "steam", h, boat.V_run_kn * KN))
                arr = list(zip(*steps))
                routes.append(dict(harbour=hb, bearing=brg, dist_nm=dist,
                                   lat=np.array(arr[0]), lon=np.array(arr[1]),
                                   hdg=np.radians(np.array(arr[2])), mode=np.array(arr[3]),
                                   hours=np.array(arr[4]), V=np.array(arr[5])))
    return routes


def cell_key(lat, lon):
    return (round(round(lat / GRID) * GRID, 2), round(round(lon / GRID) * GRID, 2))


def _open_month(path: Path):
    """Return list of datasets for one month (new CDS may split wind/wave streams into a zip)."""
    if path.suffix == ".nc" and path.read_bytes()[:14] == b"zip-extracted:":
        files = glob.glob(str(path.with_suffix("")) + "/*.nc")
    else:
        files = [str(path)]
    return [xr.open_dataset(f) for f in files]


def extract(routes):
    cells = sorted({cell_key(a, b) for r in routes for a, b in zip(r["lat"], r["lon"])})
    lats = xr.DataArray([c[0] for c in cells], dims="cell")
    lons = xr.DataArray([c[1] for c in cells], dims="cell")
    months = sorted(ERA.glob("era5_*_*.nc"))
    if not months:
        raise SystemExit("no ERA5 files yet")
    series = {"u10": [], "v10": [], "swh": []}
    times = []
    for f in tqdm(months, desc="extract cells"):
        dss = _open_month(f)
        got = {}
        for ds in dss:
            tname = "valid_time" if "valid_time" in ds.coords else "time"
            for v in ("u10", "v10", "swh"):
                if v in ds:
                    da = ds[v].sel(latitude=lats, longitude=lons, method="nearest")
                    got[v] = (ds[tname].values, da.transpose(tname, "cell").values.astype(np.float32))
        t = got["u10"][0]
        times.append(t)
        for v in series:
            # waves are on a coarser grid; reindex in time if needed and fill land cells later
            series[v].append(got[v][1] if v in got else np.full((len(t), len(cells)), np.nan, np.float32))
        for ds in dss:
            ds.close()
    t = np.concatenate(times)
    data = {v: np.concatenate(series[v]) for v in series}
    # coastal cells can be NaN for waves: fill with the seaward-most valid mean of that time step
    swh = data["swh"]
    row_mean = np.nanmean(swh, axis=1, keepdims=True)
    data["swh"] = np.where(np.isnan(swh), row_mean, swh)
    np.savez_compressed(CELLS, time=t, cells=np.array(cells), **data)
    logging.info("extracted %d cells x %d hours", len(cells), len(t))
    print(f"saved {CELLS}  cells={len(cells)} hours={len(t)}")


def run(routes, boat: Boat, boat_key: str):
    SAILS = SAIL_SETS[boat.sail_set]
    out = Path(str(OUT).format(boat=boat_key, v=f"{boat.V_run_kn:g}"))
    z = np.load(CELLS)
    t = pd.to_datetime(z["time"])
    cells = [tuple(c) for c in z["cells"]]
    cidx = {c: i for i, c in enumerate(cells)}
    U, V, H = z["u10"], z["v10"], z["swh"]
    # hourly index from first timestamp
    t0 = t[0]
    hrs = ((t - t0) / pd.Timedelta(hours=1)).astype(int)
    assert np.all(np.diff(hrs) == 1), "ERA5 series must be continuous hourly"
    days = pd.date_range(t0.normalize() + pd.Timedelta(days=1), t[-1].normalize() - pd.Timedelta(days=1), freq="D")
    rows = []
    for r in tqdm(routes, desc="routes"):
        ci = np.array([cidx[cell_key(a, b)] for a, b in zip(r["lat"], r["lon"])])
        offs = np.concatenate([[0], np.cumsum(r["hours"])[:-1]])
        for dep in DEP_HOURS_IST:
            # IST = UTC + 5.5 h -> departure UTC hour index for each day
            dep_utc = days + pd.Timedelta(hours=dep - 5.5)
            base = ((dep_utc - t0) / pd.Timedelta(hours=1)).values  # float hours
            ti = np.clip(np.rint(base[:, None] + offs[None, :]).astype(int), 0, len(t) - 1)  # (D,S)
            u, v, hs = U[ti, ci[None, :]], V[ti, ci[None, :]], H[ti, ci[None, :]]
            tws = np.hypot(u, v)
            twd = np.arctan2(-u, -v)  # direction wind comes FROM (rad, met convention)
            D, S = tws.shape
            Vb = np.broadcast_to(r["V"], (D, S))
            hdg = np.broadcast_to(r["hdg"], (D, S))
            aws, awa = apparent_wind(tws, twd, hdg, Vb)
            base_f = np.zeros((D, S))
            modes = ("steam", boat.op_mode)
            for mode in modes:
                m = r["mode"] == mode
                base_f[:, m] = fuel_rate(boat, mode, Vb[:, m].ravel(), hs[:, m].ravel()).reshape(D, -1)
            fuel_base = (base_f * r["hours"]).sum(1)
            rec = dict(date=days, trip_h=r["hours"].sum(), harbour=r["harbour"], bearing=r["bearing"], dist_nm=r["dist_nm"],
                       dep_hour=dep, fuel_base_L=fuel_base,
                       tws_mean=tws.mean(1), hs_mean=hs.mean(1), hs_max=hs.max(1),
                       u_dep=u[:, 0], v_dep=v[:, 0], u_mid=u[:, S // 2], v_mid=v[:, S // 2],
                       u_ret=u[:, -1], v_ret=v[:, -1])
            for sk, sail in SAILS.items():
                _, _, _, net = sail_net_saving(boat, sail, aws.ravel(), awa.ravel(), Vb.ravel(), hs.ravel(), tws.ravel())
                net = net.reshape(D, S)
                if boat.op_mode == "drift":
                    net[:, r["mode"] == "drift"] = 0.0  # engine off while drifting -> no saving
                f = np.zeros((D, S))
                for mode in modes:
                    m = r["mode"] == mode
                    f[:, m] = fuel_rate(boat, mode, Vb[:, m].ravel(), hs[:, m].ravel(), net[:, m].ravel()).reshape(D, -1)
                fs = (f * r["hours"]).sum(1)
                rec[f"fuel_{sk}_L"] = fs
                rec[f"save_{sk}_L"] = fuel_base - fs
                rec[f"sail_hours_{sk}"] = ((net > 0) * r["hours"]).sum(1)
            rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True)
    for sk in SAILS:
        df[f"save_{sk}_pct"] = 100 * df[f"save_{sk}_L"] / df["fuel_base_L"]
        df[f"co2_{sk}_kg"] = df[f"save_{sk}_L"] * CO2_KG_PER_L
    df["boat"] = boat_key
    df["speed_kn"] = boat.V_run_kn

    out.parent.mkdir(exist_ok=True)
    df.to_parquet(out)
    logging.info("trips %d", len(df))
    print(f"saved {out}  trips={len(df):,}")
    print(df.groupby("harbour")[[f"save_{s}_pct" for s in SAILS] + ["fuel_base_L"]].mean().round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "run"])
    ap.add_argument("--boat", default="trawler", choices=list(BOATS))
    ap.add_argument("--speed", type=float, default=0.0, help="steaming speed kn (0 = calibration speed)")
    a = ap.parse_args()
    boat = Boat(**BOATS[a.boat], V_run_kn=a.speed)
    routes = build_routes(boat)
    print(f"{len(routes)} routes; steps per route ~{np.mean([len(r['lat']) for r in routes]):.0f}")
    extract(routes) if a.step == "extract" else run(routes, boat, a.boat)
