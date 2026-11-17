# VaayuMitra poster (LaTeX)

Compile with **XeLaTeX** (or LuaLaTeX), twice is not needed:

    xelatex poster.tex

Overleaf: upload this whole folder (or `VaayuMitra_poster_latex.zip`), set
Menu → Compiler → XeLaTeX. On Overleaf the font falls back to TeX Gyre Termes (a Times clone).

- `template_bg.pdf` – the official RSD 2026 template (logos, green wave, border), used as the page background.
- `numbers.tex` – every number on the poster; regenerate with `python src/make_latex_numbers.py`.
- `figs/` – figures copied from `../figures` by the same script.
- Page size: 594 × 841 mm (the template's size, ISO A1); prints proportionally on A2.
