"""
Check that the browser advisor (docs/physics.js) gives the same answers as the Python model
(src/physics.py + src/simulate.py, the logic of app/app.py::evaluate).

Both sides use the same synthetic weather field (a deterministic function of forecast hour and
position, with wave heights crossing the 2.5 m furling limit) for every boat, harbour, bearing,
distance and sail. Run from the repo root:  python tests/check_js_port.py   (needs node on PATH)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physics import BOATS, SAIL_SETS, Boat, apparent_wind, fuel_rate, sail_net_saving  # noqa: E402
from simulate import HARBOURS, build_routes  # noqa: E402


def wx(lat, lon, idx):
    tws = 2.0 + 7.0 * np.abs(np.sin(0.3 * idx + lat))
    twd = np.mod(0.4 * idx + lon, 2 * np.pi)
    hs = 0.5 + 2.4 * np.abs(np.cos(0.2 * idx + lat))
    return tws, twd, hs


def evaluate_py(boat_key, harbour, bearing, dist, sail_key, day_start=0):
    rows = []
    for v in (8.0, 7.0, 6.0):
        boat = Boat(**BOATS[boat_key], V_run_kn=v)
        r = [x for x in build_routes(boat) if x["harbour"] == harbour and x["bearing"] == bearing
             and x["dist_nm"] == dist][0]
        offs = np.concatenate([[0], np.cumsum(r["hours"])[:-1]])
        sail = SAIL_SETS[boat.sail_set][sail_key]
        for dep in range(2, 9):
            idx = day_start + dep + np.rint(offs)
            tws, twd, h = wx(r["lat"], r["lon"], idx)
            aws, awa = apparent_wind(tws, twd, r["hdg"], r["V"])
            _, _, _, net = sail_net_saving(boat, sail, aws, awa, r["V"], h, tws)
            if boat.op_mode == "drift":
                net = np.where(r["mode"] == "drift", 0.0, net)
            f0, f1 = np.zeros(len(offs)), np.zeros(len(offs))
            for mode in ("steam", boat.op_mode):
                k = r["mode"] == mode
                f0[k] = fuel_rate(boat, mode, r["V"][k], h[k])
                f1[k] = fuel_rate(boat, mode, r["V"][k], h[k], net[k])
            rows.append([dep, v, (f0 * r["hours"]).sum(), (f1 * r["hours"]).sum(), ((net > 0) * r["hours"]).sum()])
    return rows


JS = r"""
const VM = require(process.argv[2]);
const cases = JSON.parse(process.argv[3]);
const wx = (lat, lon, idx) => [2 + 7 * Math.abs(Math.sin(0.3 * idx + lat)),
  ((0.4 * idx + lon) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI), 0.5 + 2.4 * Math.abs(Math.cos(0.2 * idx + lat))];
const out = cases.map(([b, h, br, d, s]) => VM.evaluate(b, h, br, d, s, (route, idx) => {
  const r = { tws: [], twd: [], hs: [] };
  route.forEach((st, i) => { const [a, c, e] = wx(st.lat, st.lon, idx[i]); r.tws.push(a); r.twd.push(c); r.hs.push(e); });
  return r;
}, 0, 1e9).map((x) => [x.dep, x.speed, x.fuel_nosail, x.fuel_sail, x.sail_hours]));
console.log(JSON.stringify(out));
"""

if __name__ == "__main__":
    cases = [(b, h, br, d, s) for b in BOATS for h, (_, _, brs) in HARBOURS.items() for br in brs
             for d in (15, 25, 35) for s in ("soft", "wing")]
    py = [evaluate_py(*c) for c in cases]
    tmp = ROOT / "tests" / "_check.js"
    tmp.write_text(JS)
    js = json.loads(subprocess.run(["node", str(tmp), str(ROOT / "docs" / "physics.js"), json.dumps(cases)],
                                   capture_output=True, text=True, check=True).stdout)
    tmp.unlink()
    err = max(abs(a - b) for P, J in zip(py, js) for pr, jr in zip(P, J) for a, b in zip(pr, jr))
    n = sum(len(P) for P in py)
    assert all(len(P) == len(J) for P, J in zip(py, js))
    print(f"{len(cases)} trips x {n // len(cases)} options = {n} plans compared; max |difference| = {err:.2e}")
    assert err < 0.01, "JS port disagrees with the Python model"
    print("OK: browser model matches the Python model")
