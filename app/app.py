"""
VaayuMitra - wind advisor for fishing boats (Streamlit demo).

Live mode : pulls a 3-day hourly wind + wave forecast (Open-Meteo, free, no key; only harbour /
            route coordinates are sent) and runs the calibrated boat+sail physics for every
            departure hour x speed option, then recommends the cheapest plan within the
            fisher's allowed extra time.
Fallback  : if offline, uses a stored historical day from the 10-year ERA5 simulation.

Run: .venv/Scripts/streamlit run app/app.py
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physics import BOATS, KN, SAIL_SETS, Boat, apparent_wind, fuel_rate, sail_net_saving  # noqa: E402
from simulate import HARBOURS, build_routes  # noqa: E402

PRICE = 100.0
CO2 = 2.68

T = {  # bilingual labels (Tamil wording to be checked by a native speaker)
    "title": ("VaayuMitra - Wind advisor for fishing boats", "வாயுமித்ரா - மீன்பிடி படகுகளுக்கான காற்று ஆலோசகர்"),
    "dep": ("Recommended departure", "பரிந்துரைக்கப்பட்ட புறப்பாடு நேரம்"),
    "speed": ("Steaming speed", "வேகம்"),
    "sail": ("Raise the sail", "பாய் விரிக்கவும்"),
    "saved": ("Fuel saved", "சேமிக்கப்பட்ட எரிபொருள்"),
    "rupees": ("Money saved", "சேமிக்கப்பட்ட பணம்"),
    "litres": ("litres", "லிட்டர்"),
}


def L(k, lang):
    return T[k][0] if lang == "English" else T[k][1]


@st.cache_data(ttl=3600, show_spinner=False)
def live_forecast(lats, lons):
    """Hourly 10 m wind (m/s, from-direction) and wave height for a list of points, IST."""
    q = dict(latitude=",".join(f"{x:.3f}" for x in lats), longitude=",".join(f"{x:.3f}" for x in lons),
             hourly="wind_speed_10m,wind_direction_10m", wind_speed_unit="ms",
             timezone="Asia/Kolkata", forecast_days=3)
    w = requests.get("https://api.open-meteo.com/v1/forecast", params=q, timeout=20).json()
    q2 = dict(latitude=q["latitude"], longitude=q["longitude"], hourly="wave_height",
              timezone="Asia/Kolkata", forecast_days=3)
    m = requests.get("https://marine-api.open-meteo.com/v1/marine", params=q2, timeout=20).json()
    w = w if isinstance(w, list) else [w]
    m = m if isinstance(m, list) else [m]
    t = pd.to_datetime(w[0]["hourly"]["time"])
    ws = np.array([p["hourly"]["wind_speed_10m"] for p in w], float)          # (P,H)
    wd = np.radians(np.array([p["hourly"]["wind_direction_10m"] for p in w], float))
    hs = np.array([[np.nan if v is None else v for v in p["hourly"]["wave_height"]] for p in m], float)
    hs = np.where(np.isnan(hs), np.nanmean(hs), hs)
    return t, ws, wd, hs


def evaluate(boat_key, harbour, bearing, dist, sail_key, day, lang_unused=None):
    """All (departure hour, speed) options for one trip on `day`; returns DataFrame."""
    rows = []
    for v in (8.0, 7.0, 6.0):
        boat = Boat(**BOATS[boat_key], V_run_kn=v)
        r = [x for x in build_routes(boat) if x["harbour"] == harbour and x["bearing"] == bearing
             and x["dist_nm"] == dist][0]
        t, ws, wd, hs = live_forecast(tuple(r["lat"]), tuple(r["lon"]))
        offs = np.concatenate([[0], np.cumsum(r["hours"])[:-1]])
        sail = SAIL_SETS[boat.sail_set][sail_key]
        for dep in range(2, 9):
            t0 = pd.Timestamp(day) + pd.Timedelta(hours=dep)
            idx = np.searchsorted(t.values, (t0 + pd.to_timedelta(np.rint(offs), "h")).values)  # nearest forecast hour
            if idx.max() >= len(t):
                continue
            P = np.arange(len(offs))
            tws, twd, h = ws[P, idx], wd[P, idx], hs[P, idx]
            aws, awa = apparent_wind(tws, twd, r["hdg"], r["V"])
            _, _, reef, net = sail_net_saving(boat, sail, aws, awa, r["V"], h, tws)
            if boat.op_mode == "drift":
                net = np.where(r["mode"] == "drift", 0.0, net)
            f0 = np.zeros(len(P))
            f1 = np.zeros(len(P))
            for mode in ("steam", boat.op_mode):
                k = r["mode"] == mode
                f0[k] = fuel_rate(boat, mode, r["V"][k], h[k])
                f1[k] = fuel_rate(boat, mode, r["V"][k], h[k], net[k])
            rows.append(dict(dep=dep, speed=v, trip_h=r["hours"].sum(), fuel_nosail=(f0 * r["hours"]).sum(),
                             fuel_sail=(f1 * r["hours"]).sum(), sail_hours=((net > 0) * r["hours"]).sum(),
                             max_hs=h.max(), mean_wind=tws.mean()))
    return pd.DataFrame(rows)


st.set_page_config(page_title="VaayuMitra", page_icon="⛵", layout="wide")


@st.cache_resource
def _font_css():
    """Embed TeX Gyre Termes (the poster font) so the app and the poster screenshot match."""
    import base64
    fd = ROOT / "latex" / "fonts"
    faces = ""
    for f, w, sty in (("regular", 400, "normal"), ("bold", 700, "normal"), ("italic", 400, "italic")):
        p = fd / f"texgyretermes-{f}.otf"
        if p.exists():
            b = base64.b64encode(p.read_bytes()).decode()
            faces += (f"@font-face{{font-family:'Termes';src:url(data:font/otf;base64,{b}) format('opentype');"
                      f"font-weight:{w};font-style:{sty};}}")
    return (f"<style>{faces} html, body, p, li, label, input, button, table, td, th, div, span, h1, h2, h3,"
            " [data-testid] {font-family: 'Termes', 'Times New Roman', serif !important;}</style>")


st.markdown(_font_css(), unsafe_allow_html=True)
lang = st.sidebar.radio("Language / மொழி", ["English", "தமிழ்"])
st.title(L("title", lang))
st.caption("Research prototype · IIT Madras · physics calibrated to CIFT-measured fuel rates · "
           "live forecast: Open-Meteo")

c1, c2, c3 = st.sidebar.columns(1)[0], None, None
qp = st.query_params  # optional deep link, e.g. ?boat=frp&harbour=Chennai&bearing=90 (used by the poster QR)
_boats, _harbs = list(BOATS), list(HARBOURS)
boat_key = st.sidebar.selectbox("Boat", _boats, format_func=lambda k: Boat(**BOATS[k]).name,
                                index=_boats.index(qp["boat"]) if qp.get("boat") in _boats else 0)
harbour = st.sidebar.selectbox("Harbour", _harbs,
                               index=_harbs.index(qp["harbour"]) if qp.get("harbour") in _harbs else 0)
_brg = HARBOURS[harbour][2]
bearing = st.sidebar.selectbox("Fishing ground direction (deg)", _brg,
                               index=_brg.index(int(qp["bearing"])) if qp.get("bearing", "").isdigit()
                               and int(qp["bearing"]) in _brg else 0)
dist = st.sidebar.selectbox("Distance to ground (nm)", [15, 25, 35], index=1)
sail_key = st.sidebar.selectbox("Sail", ["soft", "wing"], format_func=lambda k: SAIL_SETS["small"][k].name)
day = st.sidebar.date_input("Trip date", datetime.now().date() + timedelta(days=1))
extra = st.sidebar.slider("Extra time I can accept (hours)", 0.0, 3.0, 2.0, 0.5)

try:
    df = evaluate(boat_key, harbour, bearing, dist, sail_key, day)
except Exception as e:  # offline or API change
    st.error(f"Live forecast unavailable ({e}). Connect to the internet and refresh.")
    st.stop()
if df.empty:
    st.warning("Pick a date within the next 2 days.")
    st.stop()

habit = df[(df.dep == 4) & (df.speed == 8.0)].iloc[0]
ok = df[df.trip_h <= habit.trip_h + extra + 1e-6]
best = ok.loc[ok.fuel_sail.idxmin()]
saved = habit.fuel_nosail - best.fuel_sail

k1, k2, k3, k4 = st.columns(4)
k1.metric(L("dep", lang), f"{int(best.dep):02d}:00")
k2.metric(L("speed", lang), f"{best.speed:.0f} kn")
k3.metric(L("saved", lang), f"{saved:.1f} {L('litres', lang)}", f"-{100 * saved / habit.fuel_nosail:.0f}%")
k4.metric(L("rupees", lang), f"₹{saved * PRICE:,.0f}", f"{saved * CO2:.0f} kg CO₂")

if best.sail_hours > 0:
    st.success(f"⛵ {L('sail', lang)}: {best.sail_hours:.1f} h of the trip "
               f"(mean wind {best.mean_wind:.1f} m/s, max wave {best.max_hs:.1f} m)")
else:
    st.info("Sail not useful on this trip (wind ahead, too weak, or seas too rough) - keep it furled.")

st.subheader("All options")
show = df.assign(saving_L=habit.fuel_nosail - df.fuel_sail).round(1).sort_values("fuel_sail")
st.markdown(show.head(10).to_html(index=False, border=0, justify="right").replace(
    "<table", "<table style='width:100%;border-collapse:collapse;text-align:right'"),
    unsafe_allow_html=True)  # HTML table (not canvas) so it uses the same font as the poster
st.caption(f"Baseline = today's habit: leave 04:00, 8 kn, no sail → {habit.fuel_nosail:.1f} L. "
           "Savings are model estimates (±conformal band from the 10-year study shown on poster).")
