# VaayuMitra: wind-smart sails for small fishing boats

**Live demo:** https://hks1003.github.io/vaayumitra/ (works on any phone; runs in the browser on today's forecast)
**Poster:** [docs/poster.pdf](docs/poster.pdf), IIT Madras Research Scholars' Day 2026, sustainability theme

Diesel is the largest cost for India's small-scale fishers. A low-cost sail (about ₹12,000) on an
existing 8.5 m FRP boat can carry part of the load, but only if the fisher knows **when** the wind
will help. VaayuMitra turns the wind and wave forecast into a simple plan: when to leave, how fast
to go and when to raise the sail.

| Result (2015–2024, ERA5) | Value |
|---|---|
| Simulated single-day trips | 2.37 million (3 harbours, 2 boats, 2 sails, 6/7/8 kn) |
| Sail-only diesel saving, FRP boat at 6 kn | 8 % (Kochi) to 20 % (Visakhapatnam) |
| Payback of the sail | about 3 months as modelled; 7 months at the 6 % measured in 1983 trials |
| Slow steaming + sail | 25 → 14 L per trip (−44 %) |
| ML surrogate (unseen 2023–24) | R² 0.988, error 0.33 L/trip, 90 % band ±0.7 L held on 89 % |
| Scaled to India's 97,659 motorised boats | 51 Mn L diesel, ₹508 cr and 136 kt CO₂ a year |

Savings are a **technical potential** with good sail handling. Past whole-trip trials measured
6–10 % (Norfolk Rebel, US, 1983; FAO/BOBP Kerala, 1981–83), which still pays back within a year.

## How it works

1. **Weather:** hourly ERA5 10 m wind and significant wave height, 2015–2024, along each route.
2. **Physics** (`src/physics.py`): lifting-line sail with the best trim every hour, heel limit of 8°,
   furling above Hs 2.5 m or wind 14 m/s, hull side-force drag, and a Willans-line engine calibrated
   to fuel rates measured by ICAR-CIFT.
3. **Simulation** (`src/simulate.py`): every day, departure hour, speed, boat and sail, with fuel
   compared at the same speed with and without the sail.
4. **ML** (`src/ml.py`): a 5-member neural-network surrogate with a split-conformal 90 % band and
   SHAP explanations, tested on unseen 2023–24 trips.
5. **Advisor:** the same physics on the live Open-Meteo forecast. It runs as a Streamlit app
   (`app/app.py`) and as a static web page (`docs/`, served by GitHub Pages) that needs no server.

## Repository layout

```
docs/       GitHub Pages site: index.html + physics.js (browser advisor), poster.pdf
app/        Streamlit version of the advisor
src/        data download, physics, simulation, ML, impact, figures, poster numbers, QR code
tests/      check_js_port.py: the browser model must match the Python model
results/    small outputs (JSON/CSV, trained surrogate); large trip tables are rebuilt
figures/    poster figures and credited photos
latex/      poster source (XeLaTeX), official template background, fonts, figures
run_all.sh  full rebuild
```

## Run it

```bash
pip install -r requirements.txt
streamlit run app/app.py                  # local advisor
python tests/check_js_port.py             # check the browser model (needs Node.js)
./run_all.sh                              # full rebuild (needs a Copernicus CDS key in ~/.cdsapirc)
```

To try the web page locally: `cd docs && python -m http.server`, then open http://localhost:8000.

The poster compiles with XeLaTeX (Overleaf: Menu → Compiler → XeLaTeX) or
`tectonic -X compile latex/poster.tex`.

## Publish the demo on GitHub Pages

1. Push this folder to a new GitHub repository.
2. Repository **Settings → Pages → Build and deployment**: Source *Deploy from a branch*,
   Branch `main`, folder `/docs`, then **Save**.
3. After a minute the advisor is live at `https://<username>.github.io/<repo>/`.
4. Point the poster QR code at it and rebuild the poster:
   `python src/make_qr.py https://<username>.github.io/<repo>/`
5. Replace `USERNAME.github.io/REPO` at the top of this README.

## Data and credits

ERA5 (Copernicus Climate Change Service); live forecast from Open-Meteo (free, no key; only route
coordinates are sent). Fuel calibration from ICAR-CIFT; fleet size from the CMFRI Marine Fisheries
Census 2016. Photos from Wikimedia Commons (see `figures/photos/credits.json`). Code under the MIT
licence; third-party material keeps its own licence (see `LICENSE`). The Tamil wording in the
advisor still needs a check by a native speaker.
