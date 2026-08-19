# Figures

`make_figures.py` renders the paper figures used on the About page, dark-themed
on the site's palette, and writes them to `../images/paper/`.

    python make_figures.py

It reads its inputs from `data/`, which is not part of this repo — the rendered
PNGs are what the site ships. The SR-DiT convergence values are inlined in the
script, taken from the plotting script in that paper's own repo.
