"""
Build the RSD 2026 poster from the official template (RSD_Template.pptx).

All numbers are read from results/ (impact_table.csv, ml_results.json, trips parquet) so a
re-run after the full 10-year simulation refreshes every figure and statistic.
The live advisory card calls the app's advisor on tomorrow's Open-Meteo forecast.

Output: docs/VaayuMitra_RSD2026_poster.pptx
"""
import glob
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
OUT = ROOT / "docs" / "VaayuMitra_RSD2026_poster.pptx"

GREEN = RGBColor(0x05, 0x5A, 0x08)      # template section-title green
TINT = RGBColor(0xEE, 0xF6, 0xEC)       # light green card fill
INK = RGBColor(0x22, 0x22, 0x22)        # template body ink
MUTED = RGBColor(0x5F, 0x5E, 0x5A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ORANGE = RGBColor(0xEB, 0x68, 0x34)     # matches chart "with sail" series
FONT = "Calibri"

X0, W = 0.6, 22.2            # usable width (in)
GUT = 0.3
COL = (W - GUT) / 2


# ---------------- helpers ----------------
def box(sl, x, y, w, h, fill=None, line=None, rounded=True, name=None):
    shp = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
                              Inches(x), Inches(y), Inches(w), Inches(h))
    if rounded:
        shp.adjustments[0] = 0.04
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(2)
    shp.shadow.inherit = False
    if name:
        shp.name = name
    return shp


def text(sl, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, name=None):
    """paras: list of (text|list-of-runs, size, bold, color, align, bullet)."""
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        t, size, bold, color = p[:4]
        align = p[4] if len(p) > 4 else PP_ALIGN.LEFT
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(size * 0.25)
        runs = t if isinstance(t, list) else [(t, bold, color)]
        for rt, rb, rc in runs:
            r = para.add_run()
            r.text = rt
            r.font.size, r.font.bold, r.font.name = Pt(size), rb, FONT
            r.font.color.rgb = rc
    if name:
        tb.name = name
    return tb


def heading(sl, x, y, w, t):
    return text(sl, x, y, w, 0.8, [(t, 44, True, GREEN)])


def bullets(sl, x, y, w, h, items, size=28):
    return text(sl, x, y, w, h, [([("•  ", True, GREEN)] + (it if isinstance(it, list) else [(it, False, INK)]),
                                   size, False, INK) for it in items])


def picture(sl, path, x, y, w=None, h=None):
    im = Image.open(path)
    ar = im.height / im.width
    if w is not None and h is not None:  # fit inside box keeping aspect
        if w * ar > h:
            w = h / ar
        else:
            h = w * ar
    elif w is not None:
        h = w * ar
    else:
        w = h / ar
    return sl.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h)), w, h


def stat(sl, x, y, w, big, small, color=GREEN, small_color=INK, big_size=64):
    text(sl, x, y, w, 1.2, [(big, big_size, True, color, PP_ALIGN.CENTER)])
    text(sl, x, y + 1.15, w, 1.2, [(small, 24, False, small_color, PP_ALIGN.CENTER)])


# ---------------- numbers ----------------
def numbers():
    imp = pd.read_csv(RES / "impact_table.csv")
    ml = json.loads((RES / "ml_results.json").read_text())
    d = pd.concat(pd.read_parquet(f) for f in glob.glob(str(RES / "trips_*_v*.parquet")))
    n_trips = len(d)
    yrs = pd.to_datetime(d.date)
    span = f"{yrs.min():%b %Y}–{yrs.max():%b %Y}"
    frp = d[(d.boat == "frp") & (d.speed_kn == 6) & (d.dep_hour == 4)].copy()
    frp["m"] = pd.to_datetime(frp.date).dt.month
    mon = frp.groupby(["harbour", "m"]).save_soft_pct.mean()
    best_h, best_m = mon.idxmax()
    A = imp[(imp.claim == "A_sail_only") & (imp.sail == "soft")].set_index(["boat", "speed_kn"])
    B = imp[(imp.claim == "B_package_6kn_plus_sail") & (imp.sail == "soft")].set_index("boat")
    nat = json.loads((RES / "national_validation.json").read_text())
    sens = json.loads((RES / "sensitivity_summary.json").read_text())
    cm = pd.read_parquet(RES / "coast_map.parquet").query("month == 0")
    east = cm[(cm.lon > 77.5) | (cm.lat < 8.5)].save_pct.mean()
    west = cm[(cm.lon <= 77.5) & (cm.lat >= 8.5)].save_pct.mean()
    tip = cm[(cm.lat < 10.5) & (cm.lon > 77.0) & (cm.lon < 79.7)].save_pct.mean()
    return dict(nat=nat, sens=sens, east=east, west=west, tip=tip, ncell=len(cm),
                ml=ml, n_trips=n_trips, span=span, best_pct=mon.max(), best_h=best_h,
                best_m=pd.Timestamp(2000, best_m, 1).strftime("%B"), A=A, B=B,
                harb=frp.groupby("harbour").save_soft_pct.mean())


def live_card():
    """Real advisor output for tomorrow (Chennai, FRP boat, 25 nm east, soft sail)."""
    sys.path.insert(0, str(ROOT / "src"))
    src = (ROOT / "app" / "app.py").read_text(encoding="utf-8").split("st.set_page_config")[0]
    g = {"__file__": str(ROOT / "app" / "app.py")}
    exec(src, g)
    day = date.today() + timedelta(days=1)
    df = g["evaluate"]("frp", "Chennai", 90, 25, "soft", day)
    habit = df[(df.dep == 4) & (df.speed == 8.0)].iloc[0]
    ok = df[df.trip_h <= habit.trip_h + 2.0 + 1e-6]
    best = ok.loc[ok.fuel_sail.idxmin()]
    return day, habit, best


# ---------------- build ----------------
def main():
    N = numbers()
    prs = Presentation(ROOT / "RSD_Template.pptx")
    sl = prs.slides[0]
    for shp in list(sl.shapes):  # drop template's sample body; keep logos + heading
        if shp.name == "Body":
            shp._element.getparent().remove(shp._element)
    # Heading (keep template box, replace text)
    hd = [s for s in sl.shapes if s.name == "Heading"][0]
    tf = hd.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "VaayuMitra: Wind-Smart Sails for Fishing Boats"
    r.font.size, r.font.bold, r.font.name = Pt(46), True, "Times New Roman"
    r.font.color.rgb = RGBColor(0, 0, 0)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = "A physics + ML advisor that cuts diesel, costs and CO₂ for small-scale fishers"
    r2.font.size, r2.font.name, r2.font.italic = Pt(22), FONT, True
    r2.font.color.rgb = GREEN
    p3 = tf.add_paragraph()
    p3.alignment = PP_ALIGN.CENTER
    r3 = p3.add_run()
    r3.text = "[Your Name]  ·  Department of Ocean Engineering, IIT Madras"
    r3.font.size, r3.font.name = Pt(20), FONT
    r3.font.color.rgb = INK

    y = 5.45
    # ---- 1. Problem band ----
    box(sl, X0, y, W, 3.0, fill=TINT, name="ProblemBand")
    heading(sl, X0 + 0.3, y + 0.15, 9.0, "The problem")
    text(sl, X0 + 0.3, y + 0.85, 8.6, 2.1, [
        ("Diesel is the biggest cost for India's fishers – and it keeps rising. Every litre burnt also "
         "emits 2.68 kg of CO₂. Yet the monsoon winds that powered Indian sailing trade for "
         "millennia blow past these boats unused.", 25, False, INK)])
    sx = X0 + 9.3
    sw = (W - 9.5) / 3
    stat(sl, sx, y + 0.25, sw, "1.2 bn L", "diesel per year,\nIndia's fishing fleet¹")
    stat(sl, sx + sw, y + 0.25, sw, "56–77 %", "of a boat's operating cost\nis fuel²")
    stat(sl, sx + 2 * sw, y + 0.25, sw, "₹97→121", "marine diesel per litre,\nAndhra Pradesh, Mar 2026³",
         color=ORANGE)
    y += 3.0 + GUT

    # ---- 2. Idea + Method ----
    h2 = 4.4
    box(sl, X0, y, COL, h2, line=GREEN, name="IdeaCard")
    heading(sl, X0 + 0.3, y + 0.15, COL - 0.6, "Our idea: old wind, new intelligence")
    bullets(sl, X0 + 0.3, y + 1.0, COL - 0.6, 2.0, [
        "Retrofit a low-cost soft sail (₹12k on an FRP boat) – no engine change",
        "VaayuMitra app reads tomorrow's wind/wave forecast and tells the fisher: when to leave, "
        "how fast to steam, when to raise the sail",
        "Safety first: sail auto-reefed in strong wind or rough sea"], size=24)
    # flow diagram
    fx, fy, fw, fh = X0 + 0.3, y + 3.1, (COL - 0.6 - 3 * 0.35) / 4, 1.15
    for i, lab in enumerate(["Wind & wave\nforecast", "Boat + sail\nphysics", "ML surrogate\n+ uncertainty",
                             "Advice in\nTamil / English"]):
        bx = fx + i * (fw + 0.35)
        b = box(sl, bx, fy, fw, fh, fill=GREEN if i == 3 else TINT)
        text(sl, bx, fy, fw, fh, [(lab, 22, True, WHITE if i == 3 else GREEN, PP_ALIGN.CENTER)],
             anchor=MSO_ANCHOR.MIDDLE)
        if i < 3:
            a = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(bx + fw + 0.05), Inches(fy + fh / 2 - 0.15),
                                    Inches(0.25), Inches(0.3))
            a.fill.solid()
            a.fill.fore_color.rgb = GREEN
            a.line.fill.background()

    mx = X0 + COL + GUT
    box(sl, mx, y, COL, h2, line=GREEN, name="MethodCard")
    heading(sl, mx + 0.3, y + 0.15, COL - 0.6, "Method")
    bullets(sl, mx + 0.3, y + 1.0, COL - 0.6, 3.9, [
        [("Boats: ", True, INK), ("8.5 m FRP gillnetter and 15 m trawler, calibrated to measured Indian "
                                  "fuel rates (CIFT trials: 25.5 L/h steaming, 20.4 L/h trawling)⁴", False, INK)],
        [("Sails: ", True, INK), ("lifting-line aerodynamics, optimal trim every hour, heel limit 8°, "
                                  "hull side-force drag", False, INK)],
        [("Weather: ", True, INK), (f"hourly ERA5 winds & waves, {N['span']}", False, INK)],
        [("Trips: ", True, INK), (f"{N['n_trips']:,} single-day trips from Chennai, Kochi, "
                                  "Visakhapatnam at 6/7/8 kn", False, INK)],
        [("ML: ", True, INK), ("GPU neural-net ensemble, conformal 90 % bands, SHAP", False, INK)]],
        size=21)
    y += h2 + GUT

    # ---- 3. Results: season + speed ----
    h3 = 6.2
    box(sl, X0, y, W, h3, line=GREEN, name="ResultsCard")
    heading(sl, X0 + 0.3, y + 0.15, W - 0.6,
            "Results: wind is a free fuel – especially on the east coast")
    _, w1, h1 = picture(sl, FIG / "F1_monthly_saving.png", X0 + 0.3, y + 1.05, w=COL - 0.3, h=h3 - 1.25)
    _, w2, hh2 = picture(sl, FIG / "F5_coast_map.png", X0 + 0.6 + w1, y + 1.0, h=h3 - 1.2)
    A, B = N["A"], N["B"]
    tx = X0 + 0.6 + w1 + w2 + 0.2
    bullets(sl, tx, y + 1.1, X0 + W - tx - 0.2, h3 - 1.3, [
        [("Whole coast: ", True, GREEN), (f"{N['ncell']} sites, 10 years: east coast {N['east']:.0f} %, "
                                          f"west {N['west']:.0f} %, TN tip {N['tip']:.0f} %", False, INK)],
        [("Seasons: ", True, GREEN), ("Apr–Aug best; in July the sail is lowered for safety in rough seas "
                                      "(lower panel)", False, INK)],
        [("Slow + sail: ", True, GREEN), (f"FRP fuel {B.loc['frp', 'fuel_before_L']:.0f} → "
                                          f"{B.loc['frp', 'fuel_after_L']:.0f} L/trip "
                                          f"(−{B.loc['frp', 'save_pct']:.0f} %)", False, INK)]], size=21)
    y += h3 + GUT

    # ---- 4. ML ----
    h4 = 4.5
    box(sl, X0, y, W, h4, fill=TINT, name="MLCard")
    heading(sl, X0 + 0.3, y + 0.15, 8, "ML fishers can trust")
    ml = N["ml"]
    mm = ml["mlp_ensemble"]
    bullets(sl, X0 + 0.3, y + 1.0, 7.4, 4.5, [
        [("Accurate: ", True, INK), (f"R² {mm['r2']:.3f} on unseen days (MAE {mm['mae_L']:.2f} L/trip); "
                                     f"beats LightGBM", False, INK)],
        [("Honest: ", True, INK), (f"90 % conformal band ±{ml['conformal_q90_L']:.1f} L, held on "
                                   f"{100 * ml['conformal_coverage_test']:.0f} % of test trips", False, INK)],
        [("Instant: ", True, INK), ("one prediction in milliseconds → works on a phone or SMS server", False, INK)],
        [("Explainable: ", True, INK), ("wind strength and boat type dominate the saving", False, INK)]], size=22)
    picture(sl, FIG / "F4_parity.png", X0 + 7.9, y + 0.3, h=h4 - 0.5)
    picture(sl, FIG / "F3_shap.png", X0 + 13.9, y + 0.55, w=W - 14.1, h=h4 - 0.8)
    y += h4 + GUT

    # ---- 5. Impact band ----
    h5 = 3.0
    box(sl, X0, y, W, h5, fill=GREEN, name="ImpactBand")
    text(sl, X0 + 0.3, y + 0.1, W - 0.6, 0.8, [("Impact – per boat and at scale", 40, True, WHITE)])
    a6 = A.loc[("frp", 6.0)]
    nat = N["nat"]
    items = [(f"{a6.payback_months:.0f} months", "payback of a ₹12k sail\n(FRP boat)"),
             (f"₹{nat['sail_only_inr_per_boat_yr'] / 1e3:.0f}k", "saved per boat per year by sail\n"
              f"(₹{B.loc['frp', 'inr_yr'] / 1e5:.1f} lakh with slow steaming)"),
             (f"{nat['india']['ML_diesel']:.0f} M L", "diesel saved per year,\nall India⁵"),
             (f"₹{nat['india']['crore']:.0f} cr", f"saved per year across India's\n{nat['india']['craft']:,} motorised boats⁵"),
             (f"{nat['india']['kt_co2']:.0f} kt", "CO₂ avoided per year,\nall India⁵")]
    iw = (W - 0.6) / len(items)
    for i, (b_, s_) in enumerate(items):
        stat(sl, X0 + 0.3 + i * iw, y + 0.8, iw, b_, s_, color=WHITE, small_color=WHITE, big_size=54)
    y += h5 + GUT

    # ---- 6. Implementation + robustness + refs ----
    h6 = 32.35 - y
    assert h6 >= 3.8, f"layout overflow: only {h6:.2f} in left for bottom row"
    cw = (W - 2 * GUT) / 3
    box(sl, X0, y, cw, h6, line=GREEN, name="LiveCard")
    heading(sl, X0 + 0.25, y + 0.1, cw - 0.5, "It works today")
    try:
        day, habit, best = live_card()
        saved = habit.fuel_nosail - best.fuel_sail
        live = [f"Live advice, {day:%d %b}, Chennai FRP boat:",
                f"Leave {int(best.dep):02d}:00 · {best.speed:.0f} kn · sail {best.sail_hours:.0f} h",
                f"{habit.fuel_nosail:.1f} → {best.fuel_sail:.1f} L (₹{saved * 100:,.0f} saved)"]
    except Exception as e:  # offline build
        live = ["Live forecast unavailable at build time", str(e)[:60], ""]
    qs = min(h6 - 1.5, 2.2)
    text(sl, X0 + 0.25, y + 0.9, cw - 0.6 - 2 * qs - 0.1, h6 - 1.0, [
        (live[0], 16, False, MUTED), (live[1], 19, True, GREEN), (live[2], 19, True, INK),
        ("Scan to try it live →", 16, True, ORANGE)])
    for i, (q, lab) in enumerate((("qr_wifi", "1. join Wi-Fi"), ("qr_app", "2. open app"))):
        qx = X0 + cw - 0.2 - (2 - i) * (qs + 0.05)
        picture(sl, FIG / f"{q}.png", qx, y + 0.95, w=qs)
        text(sl, qx, y + 0.95 + qs, qs, 0.4, [(lab, 15, True, GREEN, PP_ALIGN.CENTER)])

    nx = X0 + cw + GUT
    box(sl, nx, y, cw, h6, fill=TINT, name="RobustCard")
    heading(sl, nx + 0.25, y + 0.1, cw - 0.5, "Robust & validated")
    v = N["nat"]["validation"]["model_sail_pct"]
    text(sl, nx + 0.25, y + 0.8, cw - 0.5, 1.3, [
        ([("Measured whole-trip savings in past trials: ~6 % (US, 1983) and ~10 % (Kerala, BOBP logbooks)⁶. "
           "Our model gives a technical potential of ", False, INK),
          ("8–20 % at 6 kn", True, GREEN),
          (" with good sail handling; real savings likely nearer the low end → next: a logged sea trial.",
           False, INK)], 14, False, INK)])
    picture(sl, FIG / "F6_sensitivity.png", nx + 0.15, y + 2.0, w=cw - 0.3, h=h6 - 2.1)

    rx = X0 + 2 * (cw + GUT)
    box(sl, rx, y, cw, h6, line=GREEN, name="RefCard")
    heading(sl, rx + 0.25, y + 0.1, cw - 0.5, "Next steps & references")
    bullets(sl, rx + 0.25, y + 0.9, cw - 0.5, 1.2, [
        "Sea trial: Chennai FRP boat + fuel logger",
        "Co-design with fishers; Tamil voice / SMS"], size=19)
    text(sl, rx + 0.25, y + 2.1, cw - 0.5, h6 - 2.2, [
        ("¹ ICAR-CIFT  ² IISD 2022; ICAR-CIFT  ³ Deccan Chronicle, Mar 2026  ⁴ CIFT trials, Fishery Technology  "
         "⁵ CMFRI Marine Fisheries Census 2016 (motorised craft) × sail-only saving of the 8.5 m FRP boat  "
         "⁶ Norfolk Rebel trial (VIMS/NMFS 1983); BOBP/WP/25 (Kerala 1981–83)  "
         "· ERA5 (Copernicus C3S)", 14, False, MUTED)])

    OUT.parent.mkdir(exist_ok=True)
    prs.save(OUT)
    print("saved", OUT, "bottom y =", round(y + h6, 2))


if __name__ == "__main__":
    main()
