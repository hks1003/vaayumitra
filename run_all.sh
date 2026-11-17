#!/usr/bin/env bash
# Rebuild everything: ERA5 download -> trip simulation -> ML surrogate -> impact -> figures -> poster numbers.
# Needs ~/.cdsapirc (Copernicus CDS key) for the download step and ~3 GB of disk.
# Set PYTHON to your interpreter if it is not "python" (e.g. PYTHON=.venv/Scripts/python.exe ./run_all.sh).
set -e
cd "$(dirname "$0")"
P=${PYTHON:-python}
mkdir -p data logs
$P src/download_era5.py                      # ERA5 hourly wind + waves, 2015-2024
$P src/simulate.py extract
for b in trawler frp; do for v in 8 7 6; do $P src/simulate.py run --boat $b --speed $v; done; done
$P src/ml.py                                 # surrogate ensemble, conformal band, SHAP
$P src/impact.py
$P src/coast_map.py                          # 100 coastal sites
$P src/sensitivity.py
$P src/national.py
$P src/figures.py
$P src/figures_extra.py
$P src/figures_v4.py
$P src/make_latex_numbers.py                 # latex/numbers.tex + latex/figs
$P tests/check_js_port.py                    # browser demo == Python model
echo "ALL DONE - now compile latex/poster.tex with XeLaTeX"
