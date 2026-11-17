"""
Download ERA5 hourly winds + waves over Indian coastal waters for VaayuMitra.

Variables : 10 m u/v wind, significant wave height (combined wind sea + swell)
Region    : 5-22 N, 68-90 E (both Indian coasts, Arabian Sea + Bay of Bengal)
Period    : 2015-2024, hourly (hourly needed to resolve land/sea breeze cycle)
Output    : data/era5/era5_YYYY_MM.nc  (one file per month; skips existing files)

Usage     : python src/download_era5.py [--years 2015 2024]
Requires  : ~/.cdsapirc with your CDS API key, and the ERA5 licence accepted on
            https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
"""
import argparse
import logging
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cdsapi
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "era5"
LOG = ROOT / "logs" / "download_era5.log"

AREA = [22, 68, 5, 90]  # N, W, S, E
VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind",
        "significant_height_of_combined_wind_waves_and_swell"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=[2015, 2024])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=LOG, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    jobs = [(y, m) for y in range(args.years[0], args.years[1] + 1) for m in range(1, 13)]
    jobs = [j for j in jobs if not (OUT / f"era5_{j[0]}_{j[1]:02d}.nc").exists()]  # resume-safe

    def fetch(y, m):
        """One month; own client per thread (requests sessions are not thread-safe)."""
        client = cdsapi.Client(quiet=True, progress=False)
        target = OUT / f"era5_{y}_{m:02d}.nc"
        tmp = target.with_suffix(".download")
        client.retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"], "variable": VARS,
            "year": [str(y)], "month": [f"{m:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": AREA, "data_format": "netcdf", "download_format": "unarchived",
        }, str(tmp))
        # New CDS returns a zip when wind (oper) and wave streams are separate files
        if zipfile.is_zipfile(tmp):
            sub = OUT / f"era5_{y}_{m:02d}"
            sub.mkdir(exist_ok=True)
            with zipfile.ZipFile(tmp) as z:
                z.extractall(sub)
            tmp.unlink()
            target.write_text("zip-extracted:" + str(sub))  # marker so resume skips it
        else:
            tmp.rename(target)
        return y, m

    # several requests queue in parallel on CDS -> much faster than one at a time
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, y, m): (y, m) for y, m in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc="ERA5 months"):
            y, m = futs[f]
            try:
                f.result()
                logging.info("ok %d-%02d", y, m)
            except Exception as e:  # failed months are retried on the next run
                logging.error("fail %d-%02d: %s", y, m, e)
                tqdm.write(f"FAILED {y}-{m:02d}: {e}")

if __name__ == "__main__":
    main()
