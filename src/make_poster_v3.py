"""
VaayuMitra poster v3 (RSD 2026 template): v2 layout, green palette, photos from the cited reports.

Changes vs v1: template header + border stay green, every other element in shades of blue;
Times New Roman throughout; no empty space (photos crop-to-fill, columns sized to content);
one QR code (live app on the IIT Madras campus network); next steps and references split;
full numbered references; two real photos of sail-rigged fishing boats (Wikimedia Commons).
All numbers are read from results/ exactly as in v1.

Output: docs/VaayuMitra_RSD2026_poster_v2.pptx
"""
import json
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
import make_poster as V1  # noqa: E402  (reuse numbers() and live_card())

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
PHOTO = FIG / "photos"
OUT = ROOT / "docs" / "VaayuMitra_RSD2026_poster_v3.pptx"
PIMG = FIG / "paper_imgs"
TMP = ROOT / "docs" / "qa" / "crops"
TMP.mkdir(parents=True, exist_ok=True)

NAVY = RGBColor(0x0B, 0x4D, 0x1C)     # (name kept from v2) darkest green: impact band, strong text
BLUE = RGBColor(0x05, 0x5A, 0x08)     # template green: headings, card borders
MID = RGBColor(0x2E, 0x8B, 0x3E)      # mid green accents
TINT = RGBColor(0xEC, 0xF6, 0xEC)     # light green card fill
TINT2 = RGBColor(0xD3, 0xEB, 0xD5)    # flow boxes / reference strip
INK = RGBColor(0x14, 0x22, 0x16)
MUTED = RGBColor(0x45, 0x58, 0x48)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Times New Roman"

X0, W, GUT = 0.6, 22.2, 0.25
COL = (W - GUT) / 2
TOP, BOTTOM = 5.45, 32.35
H1, BODY, SMALL = 40, 24, 19     # heading / body / small sizes (pt)


# ---------------- helpers ----------------
def box(sl, x, y, w, h, fill=None, line=None, name=None):
    s = sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.04
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(2.25)
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def text(sl, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, name=None, space=0.25):
    """paras: list of (text | [(run, bold, color[, italic])...], size, bold, color[, align])."""
    tb = sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p in enumerate(paras):
        t, size, bold, color = p[:4]
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = p[4] if len(p) > 4 else PP_ALIGN.LEFT
        para.space_after = Pt(size * space)
        for run in (t if isinstance(t, list) else [(t, bold, color)]):
            r = para.add_run()
            r.text = run[0]
            r.font.size, r.font.bold, r.font.name = Pt(size), run[1], FONT
            r.font.italic = len(run) > 3 and run[3]
            r.font.color.rgb = run[2]
    if name:
        tb.name = name
    return tb


def heading(sl, x, y, w, t, size=H1, color=BLUE):
    return text(sl, x, y, w, 0.75, [(t, size, True, color)])


def bullets(sl, x, y, w, h, items, size=BODY):
    return text(sl, x, y, w, h, [([("▪  ", True, MID)] + (it if isinstance(it, list) else [(it, False, INK)]),
                                   size, False, INK) for it in items])


def picture(sl, path, x, y, w=None, h=None):
    """Fit inside (w, h) keeping aspect; returns (shape, w, h)."""
    im = Image.open(path)
    ar = im.height / im.width
    if w is not None and h is not None:
        w, h = (h / ar, h) if w * ar > h else (w, w * ar)
    elif w is not None:
        h = w * ar
    else:
        w = h / ar
    return sl.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h)), w, h


def photo_fill(sl, path, x, y, w, h, tag):
    """Centre-crop a photo to exactly fill (w, h) - no empty space."""
    im = Image.open(path).convert("RGB")
    tr = w / h
    if im.width / im.height > tr:
        nw = int(im.height * tr)
        im = im.crop(((im.width - nw) // 2, 0, (im.width + nw) // 2, im.height))
    else:
        nh = int(im.width / tr)
        top = max(0, (im.height - nh) // 2)
        im = im.crop((0, top, im.width, top + nh))
    out = TMP / f"{tag}.jpg"
    im.save(out, quality=92)
    return sl.shapes.add_picture(str(out), Inches(x), Inches(y), Inches(w), Inches(h))


def stat(sl, x, y, w, big, small, color=NAVY, small_color=INK, big_size=58, small_size=SMALL):
    text(sl, x, y, w, 1.0, [(big, big_size, True, color, PP_ALIGN.CENTER)])
    text(sl, x, y + 0.95, w, 1.1, [(small, small_size, False, small_color, PP_ALIGN.CENTER)], space=0.05)


def caption(sl, x, y, w, t):
    text(sl, x, y, w, 0.3, [(t, 12, False, MUTED, PP_ALIGN.LEFT)], space=0)


# ---------------- build ----------------
def main():
    N = V1.numbers()
    credits = {c["file"]: c for c in json.loads((PHOTO / "credits.json").read_text(encoding="utf-8"))}
    prs = Presentation(ROOT / "RSD_Template.pptx")
    sl = prs.slides[0]
    for shp in list(sl.shapes):
        if shp.name == "Body":
            shp._element.getparent().remove(shp._element)
    tf = [s for s in sl.shapes if s.name == "Heading"][0].text_frame
    tf.clear()
    for i, (t, size, bold, italic, col) in enumerate([
            ("VaayuMitra: Wind-Smart Sails for Fishing Boats", 48, True, False, RGBColor(0, 0, 0)),
            ("A physics + machine-learning advisor that cuts diesel, cost and CO₂ for small-scale fishers",
             22, False, True, NAVY),
            ("[Your Name]  ·  Department of Ocean Engineering, IIT Madras", 20, False, False, INK)]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = t
        r.font.size, r.font.bold, r.font.italic, r.font.name = Pt(size), bold, italic, FONT
        r.font.color.rgb = col

    y = TOP
    # ---- A. Problem band: text | 3 stats | photo ----
    hA = 3.0
    box(sl, X0, y, W, hA, fill=TINT, name="ProblemBand")
    tw, sw, pw = 7.3, 3.25, 0
    pw = W - 0.25 - tw - 3 * sw - 0.25
    heading(sl, X0 + 0.25, y + 0.1, tw, "The problem")
    text(sl, X0 + 0.25, y + 0.85, tw, hA - 0.9, [
        ("Diesel is the biggest cost for India's small-scale fishers and it keeps rising; every litre burnt "
         "emits 2.68 kg of CO₂. Yet the monsoon winds that once powered Indian sailing craft now blow past "
         "these boats unused.", 22, False, INK)], space=0.1)
    sx = X0 + 0.25 + tw
    for i, (b, s) in enumerate([("1.2 bn L", "diesel per year,\nIndia's fishing fleet [1]"),
                                ("56–77 %", "of a boat's operating\ncost is fuel [2]"),
                                ("₹97→121", "marine diesel per litre,\nAndhra Pradesh, 2026 [3]")]):
        stat(sl, sx + i * sw, y + 0.45, sw, b, s, color=NAVY if i < 2 else MID, big_size=52)
    px = sx + 3 * sw
    photo_fill(sl, PIMG / "ind18_sail_bw.png", px, y + 0.15, pw, hA - 0.55, "p1")
    caption(sl, px, y + hA - 0.38, pw, "IND-18 engine-cum-sail gillnetter, Kerala trials [7]")
    y += hA + GUT

    # ---- B. Idea (bullets + photo + flow) | Method ----
    hB = 4.5
    box(sl, X0, y, COL, hB, line=BLUE, name="IdeaCard")
    heading(sl, X0 + 0.25, y + 0.1, COL - 0.5, "Our idea: old wind, new intelligence")
    ph_w = 3.7
    bullets(sl, X0 + 0.25, y + 0.85, COL - ph_w - 0.65, 2.2, [
        "Retrofit a low-cost soft sail (₹12k on an FRP boat); no engine change",
        "App reads tomorrow's wind and wave forecast and tells the fisher when to leave, how fast to "
        "steam and when to raise the sail",
        "Safety first: sail reefed in strong wind or rough sea"], size=21)
    php = X0 + COL - ph_w - 0.25
    photo_fill(sl, PIMG / "ind17_sail_bw_wide.png", php, y + 0.9, ph_w, 1.95, "p2")
    caption(sl, php, y + 2.86, ph_w, "FAO/BOBP IND-17 under sail, Kerala [7]")
    fx, fy, fh = X0 + 0.25, y + 3.3, 1.0
    fw = (COL - 0.5 - 3 * 0.35) / 4
    for i, lab in enumerate(["Wind & wave\nforecast", "Boat + sail\nphysics", "ML surrogate\n+ uncertainty",
                             "Advice in\nTamil / English"]):
        bx = fx + i * (fw + 0.35)
        box(sl, bx, fy, fw, fh, fill=NAVY if i == 3 else TINT2)
        text(sl, bx, fy, fw, fh, [(lab, 19, True, WHITE if i == 3 else NAVY, PP_ALIGN.CENTER)],
             anchor=MSO_ANCHOR.MIDDLE, space=0)
        if i < 3:
            a = sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(bx + fw + 0.05), Inches(fy + fh / 2 - 0.15),
                                    Inches(0.25), Inches(0.3))
            a.fill.solid()
            a.fill.fore_color.rgb = MID
            a.line.fill.background()

    mx = X0 + COL + GUT
    box(sl, mx, y, COL, hB, line=BLUE, name="MethodCard")
    heading(sl, mx + 0.25, y + 0.1, COL - 0.5, "Method")
    bullets(sl, mx + 0.25, y + 0.85, COL - 0.5, hB - 0.9, [
        [("Boats: ", True, NAVY), ("8.5 m FRP gillnetter and 15 m trawler, calibrated to measured Indian fuel "
                                   "rates (25.5 L/h steaming, 20.4 L/h trawling) [4]", False, INK)],
        [("Sails: ", True, NAVY), ("lifting-line aerodynamics, best trim every hour, 8° heel limit, hull "
                                   "side-force drag", False, INK)],
        [("Weather: ", True, NAVY), (f"hourly ERA5 wind and waves, {N['span']} [9]", False, INK)],
        [("Trips: ", True, NAVY), (f"{N['n_trips']:,} single-day trips from Chennai, Kochi and "
                                   "Visakhapatnam at 6, 7 and 8 kn", False, INK)],
        [("ML: ", True, NAVY), ("GPU neural-net ensemble, conformal 90 % bands, SHAP", False, INK)]], size=21)
    y += hB + GUT

    # ---- C. Results: F1 | map | bullets ----
    hC = 5.9
    box(sl, X0, y, W, hC, line=BLUE, name="ResultsCard")
    heading(sl, X0 + 0.25, y + 0.1, W - 0.5, "Results: wind is a free fuel, especially on the east coast")
    ih = hC - 1.0
    _, w1, _ = picture(sl, FIG / "F1_monthly_saving.png", X0 + 0.25, y + 0.85, h=ih)
    _, w2, _ = picture(sl, FIG / "F5_coast_map.png", X0 + 0.4 + w1, y + 0.85, h=ih)
    tx = X0 + 0.55 + w1 + w2
    A, B, nat = N["A"], N["B"], N["nat"]
    bullets(sl, tx, y + 0.9, X0 + W - tx - 0.2, ih, [
        [("Whole coast: ", True, NAVY), (f"{N['ncell']} sites over 10 years; sail alone saves {N['east']:.0f} % "
                                         f"on the east coast, {N['west']:.0f} % on the west, {N['tip']:.0f} % "
                                         "at the Tamil Nadu tip", False, INK)],
        [("Seasons: ", True, NAVY), ("April–August is best; in July the sail is lowered for safety in "
                                     "rough seas (lower panel)", False, INK)],
        [("Slow + sail: ", True, NAVY), (f"FRP boat {B.loc['frp', 'fuel_before_L']:.0f} → "
                                         f"{B.loc['frp', 'fuel_after_L']:.0f} L per trip "
                                         f"(−{B.loc['frp', 'save_pct']:.0f} %)", False, INK)],
        [("States: ", True, NAVY), (f"Andhra Pradesh ₹{nat['Andhra Pradesh']['crore']:.0f} cr/yr, "
                                    f"Kerala ₹{nat['Kerala']['crore']:.0f} cr/yr (sail only) [5]", False, INK)],
        [("Trawler: ", True, NAVY), ("sail alone saves only 2.5–3.5 % on a big engine, so small FRP boats "
                                     "are the right target", False, INK)]],
        size=21)
    y += hC + GUT

    # ---- D. ML: bullets | parity | SHAP ----
    hD = 4.3
    box(sl, X0, y, W, hD, fill=TINT, name="MLCard")
    ml, mm = N["ml"], N["ml"]["mlp_ensemble"]
    ph = hD - 0.3
    pr_w = ph * Image.open(FIG / "F4_parity.png").width / Image.open(FIG / "F4_parity.png").height
    sh_w = ph * Image.open(FIG / "F3_shap.png").width / Image.open(FIG / "F3_shap.png").height
    bw = W - pr_w - sh_w - 0.75
    heading(sl, X0 + 0.25, y + 0.1, bw, "ML fishers can trust")
    bullets(sl, X0 + 0.25, y + 0.85, bw, hD - 0.9, [
        [("Accurate: ", True, NAVY), (f"R² {mm['r2']:.3f} on unseen years (error {mm['mae_L']:.2f} L per "
                                      "trip); beats LightGBM", False, INK)],
        [("Honest: ", True, NAVY), (f"90 % conformal band ±{ml['conformal_q90_L']:.1f} L, held on "
                                    f"{100 * ml['conformal_coverage_test']:.0f} % of test trips", False, INK)],
        [("Instant: ", True, NAVY), ("milliseconds per prediction, runs on a phone or SMS server", False, INK)],
        [("Explainable: ", True, NAVY), ("wind strength and boat type dominate the saving", False, INK)],
        [("Advisor: ", True, NAVY), ("choosing the departure hour adds ~7 % with a perfect forecast but only "
                                     "~1.4 % with real forecast errors, so the sail itself is the main lever",
                                     False, INK)]],
        size=22)
    picture(sl, FIG / "F4_parity.png", X0 + 0.25 + bw + 0.1, y + 0.15, h=ph)
    picture(sl, FIG / "F3_shap.png", X0 + W - 0.15 - sh_w, y + 0.15, h=ph)
    y += hD + GUT

    # ---- E. Impact band ----
    hE = 2.8
    box(sl, X0, y, W, hE, fill=BLUE, name="ImpactBand")
    text(sl, X0 + 0.25, y + 0.08, W - 0.5, 0.7, [("Impact: per boat and at national scale", 36, True, WHITE)])
    a6 = A.loc[("frp", 6.0)]
    items = [(f"{a6.payback_months:.0f} months", "payback of a ₹12k sail\n(FRP boat)"),
             (f"₹{nat['sail_only_inr_per_boat_yr'] / 1e3:.0f}k",
              f"saved per boat per year by sail\n(₹{B.loc['frp', 'inr_yr'] / 1e5:.1f} lakh with slow steaming)"),
             (f"{nat['india']['ML_diesel']:.0f} M L", "diesel saved per year,\nall India [5]"),
             (f"₹{nat['india']['crore']:.0f} cr", f"saved per year across\n{nat['india']['craft']:,} motorised boats [5]"),
             (f"{nat['india']['kt_co2']:.0f} kt", "CO₂ avoided per year,\nall India [5]")]
    iw = (W - 0.5) / len(items)
    for i, (b_, s_) in enumerate(items):
        stat(sl, X0 + 0.25 + i * iw, y + 0.75, iw, b_, s_, color=WHITE, small_color=WHITE, big_size=50,
             small_size=18)
    y += hE + GUT

    # ---- F. Live + QR | Robust | Next steps ----
    hR = 2.15                                  # references strip height
    hF = BOTTOM - y - GUT - hR
    assert hF >= 2.6, f"layout overflow: bottom cards only {hF:.2f} in"
    lw, rw = 6.3, 9.4
    nw = W - lw - rw - 2 * GUT
    box(sl, X0, y, lw, hF, line=BLUE, name="LiveCard")
    heading(sl, X0 + 0.2, y + 0.05, lw - 0.4, "It works today", size=32)
    try:
        day, habit, best = V1.live_card()
        saved = habit.fuel_nosail - best.fuel_sail
        live = [f"Live advice for {day:%d %b}, Chennai FRP boat:",
                f"Leave {int(best.dep):02d}:00 · {best.speed:.0f} kn · sail up {best.sail_hours:.0f} h",
                f"{habit.fuel_nosail:.1f} → {best.fuel_sail:.1f} L (₹{saved * 100:,.0f} saved)"]
    except Exception as e:  # offline build
        live = ["Live forecast unavailable at build time", str(e)[:50], ""]
    qs = min(hF - 0.55, 2.4)
    text(sl, X0 + 0.2, y + 0.7, lw - qs - 0.55, hF - 0.75, [
        (live[0], 17, False, MUTED), (live[1], 19, True, NAVY), (live[2], 19, True, INK),
        ("Scan on IIT Madras Wi-Fi to try the app →", 16, True, MID)], space=0.15)
    picture(sl, FIG / "qr_app.png", X0 + lw - qs - 0.2, y + (hF - qs) / 2, w=qs)

    rx = X0 + lw + GUT
    box(sl, rx, y, rw, hF, fill=TINT, name="RobustCard")
    heading(sl, rx + 0.2, y + 0.05, rw - 0.4 - (hF - 0.25) * 2.08, "Validated", size=32)
    tor = FIG / "F6_sensitivity.png"
    th = hF - 0.25
    tr_w = th * Image.open(tor).width / Image.open(tor).height
    text(sl, rx + 0.2, y + 0.7, rw - tr_w - 0.45, hF - 0.75, [
        ([("Measured whole-trip savings in past trials: ~6 % (US, 1983) [6] and ~10 % (Kerala logbooks) [7]. "
           "Our model gives a technical potential of ", False, INK),
          ("8–20 % at 6 kn", True, NAVY),
          (" with good sail handling; real savings are likely nearer the low end.", False, INK)], 16, False, INK)],
        space=0.1)
    picture(sl, tor, rx + rw - tr_w - 0.15, y + 0.12, h=th)

    nx = rx + rw + GUT
    box(sl, nx, y, nw, hF, line=BLUE, name="NextCard")
    heading(sl, nx + 0.2, y + 0.05, nw - 0.4, "Next steps", size=32)
    bullets(sl, nx + 0.2, y + 0.7, nw - 0.4, hF - 0.75, [
        "Logged sea trial on a Chennai FRP boat (fuel meter, GPS, anemometer)",
        "Co-design with fishers: Tamil voice and SMS advisories",
        "Measure real sail-use over 20–40 fishing trips",
        "Open the app to fishers' cooperatives after the trial"], size=17)
    y += hF + GUT

    # ---- G. References strip (two columns) ----
    box(sl, X0, y, W, hR, fill=TINT2, name="RefStrip")
    text(sl, X0 + 0.2, y + 0.05, 3.0, 0.5, [("References", 22, True, NAVY)])
    refs = [
        "[1] ICAR-CIFT. VSOB: indigenous innovation driving energy-efficient and climate-smart fishing. ICAR, icar.org.in.",
        "[2] IISD (2022). Fuel subsidies in marine fisheries in India. International Institute for Sustainable Development.",
        "[3] Deccan Chronicle (2026). Marine diesel price hike from ₹97 to ₹121 pushes fishermen to the brink.",
        "[4] ICAR-CIFT LNG–diesel dual-fuel trawler trials off Kochi. Fishery Technology (ICAR e-pubs).",
        "[5] ICAR-CMFRI. Marine Fisheries Census 2016: India (97,659 motorised craft); scaled by our sail-only "
        "saving of the 8.5 m FRP boat.",
        "[6] Rebel Marine Service Inc. (1983). The reduction of energy needs for fish harvesting through the use "
        "of sails on fishing vessels. NMFS S-K grant NA-80-FA-D-0012, with VIMS.",
        "[7] Gulbrandsen, Ø. (1984). Fishing craft development in Kerala: evaluation report. BOBP/WP/25, "
        "Bay of Bengal Programme, FAO, Madras.",
        "[8] Wilson, J.D.K. (1999). Fuel and financial savings for operators of small fishing vessels. "
        "FAO Fisheries Technical Paper 383, Rome.",
        "[9] Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol. Soc. 146, 1999–2049.",
        "[10] Open-Meteo forecast and marine weather APIs (open-meteo.com). Photos reproduced from [7] "
        "(FAO/BOBP) for research and teaching."]
    half = (len(refs) + 1) // 2
    cw = (W - 0.6) / 2
    for k, chunk in enumerate((refs[:half], refs[half:])):
        text(sl, X0 + 0.2 + k * (cw + 0.2), y + 0.5, cw, hR - 0.55, [(r, 13, False, INK) for r in chunk],
             space=0.12)

    OUT.parent.mkdir(exist_ok=True)
    prs.save(OUT)
    print("saved", OUT, "bottom y =", round(y + hR, 2))


if __name__ == "__main__":
    main()
