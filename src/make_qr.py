"""
Poster QR code -> the live advisor on GitHub Pages (opens on any phone, any network).

Usage (from the repo root), after enabling GitHub Pages on the /docs folder:
    python src/make_qr.py https://<username>.github.io/<repo>/
Writes figures/qr_app.png and latex/figs/qr_app.png, then rebuild the poster
(Overleaf with XeLaTeX, or: tectonic -X compile latex/poster.tex).
The deep link opens the Chennai FRP-boat example shown on the poster.
"""
import shutil
import sys
from pathlib import Path

import qrcode

ROOT = Path(__file__).resolve().parents[1]
DEEP_LINK = "?boat=frp&harbour=Chennai&bearing=90"

if len(sys.argv) != 2 or not sys.argv[1].startswith("http"):
    sys.exit("usage: python src/make_qr.py https://<username>.github.io/<repo>/")
url = sys.argv[1].rstrip("/") + "/" + DEEP_LINK
q = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=20, border=2)
q.add_data(url)
q.make(fit=True)
out = ROOT / "figures" / "qr_app.png"
q.make_image(fill_color="#055A08", back_color="white").save(out)
shutil.copy2(out, ROOT / "latex" / "figs" / "qr_app.png")
print("QR ->", url)
