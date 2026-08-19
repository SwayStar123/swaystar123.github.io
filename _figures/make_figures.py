"""
Render the paper figures used on the About page, dark-themed.
Inputs live in ./data (not in the repo — see README.md).

    python make_figures.py

Writes into ../images/paper/.

Styling follows the site, not the papers: the site's own accent colours from
_sass/_variables.scss instead of matplotlib defaults, stand-ins for the site's
Inter + JetBrains Mono pairing, and a transparent background so each figure
sits on the page's wash exactly the way the body copy does.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.abspath(os.path.join(HERE, "..", "images", "paper"))

# ---- site palette, straight from _sass/_variables.scss ----------------------
FG = "#e9ecf1"        # headings / high emphasis
MUTED = "#a2aab8"     # body copy
FAINT = "#5c6470"     # de-emphasised meta
GRID = "#22262e"      # hairline / border
SURFACE = "#191c22"   # raised surface, for legend fills
EDGE = "#0f1116"      # marker outline, just under the page ground
ACCENT = "#7aa2f7"    # link blue
CYAN = "#7dcfff"
PURPLE = "#bb9af7"
PINK = "#f7768e"
ORANGE = "#ff9e64"
AMBER = "#e0af68"
GREEN = "#9ece6a"
GRAY = "#868e9c"     # site $gray, mid step between MUTED and FAINT

SANS = ["Segoe UI", "DejaVu Sans"]                      # stands in for Inter
MONO = ["Cascadia Mono", "Consolas", "DejaVu Sans Mono"]  # for JetBrains Mono

SIZES = ["60m", "120m", "250m", "500m", "1b", "2b"]
# six site accents ordered cool -> warm, so model size still reads as a
# progression while staying distinguishable at 4px marker size
SIZE_COLOR = dict(zip(SIZES, [CYAN, ACCENT, PURPLE, PINK, ORANGE, AMBER]))


def style():
    plt.rcParams.update({
        "font.family": SANS,
        "font.size": 11,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.transparent": True,
        "text.color": MUTED,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "xtick.color": FAINT,
        "ytick.color": FAINT,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "grid.color": GRID,
        "axes.grid": True,
        "grid.linewidth": 0.7,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.facecolor": SURFACE,
        "legend.edgecolor": GRID,
        "legend.framealpha": 0.92,
        "legend.fancybox": True,
        "legend.labelcolor": MUTED,
    })


def mono_ticks(ax):
    for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lbl.set_fontfamily(MONO)
        lbl.set_fontsize(9.5)


def mono_legend(leg, title_size=9.5):
    for t in leg.get_texts():
        t.set_fontfamily(MONO)
    if leg.get_title().get_text():
        leg.get_title().set_color(FG)
        leg.get_title().set_fontsize(title_size)


def read(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- Abra fig 1
def fig1():
    rows = read("abra_fig1_scaling.csv")
    curves = {s: [] for s in SIZES}
    for r in rows:
        curves[r["size"]].append((float(r["flops"]), float(r["loss"])))
    for s in curves:
        curves[s].sort()

    # Compute-optimal frontier by tangency, iterated to a fixed point: fit a
    # power law through the current per-size optima, re-pick each size's optimum
    # as its point of closest approach to that fit, repeat.
    star_idx = {s: len(curves[s]) - 1 for s in SIZES}
    for _ in range(40):
        pts = [curves[s][star_idx[s]] for s in SIZES]
        b, a = np.polyfit(np.log10([p[0] for p in pts]), np.log10([p[1] for p in pts]), 1)
        new = {}
        for s in SIZES:
            resid = [np.log10(l) - (a + b * np.log10(c)) for c, l in curves[s]]
            new[s] = int(np.argmin(resid))
        if new == star_idx:
            break
        star_idx = new

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for s in SIZES:
        c = np.array([p[0] for p in curves[s]])
        l = np.array([p[1] for p in curves[s]])
        col = SIZE_COLOR[s]
        ax.plot(c, l, color=col, lw=2.2, alpha=0.95, solid_capstyle="round", zorder=2)
        ax.plot(c, l, "o", color=col, ms=3.6, mec=EDGE, mew=0.6, zorder=3)

    grid = np.logspace(17.9, 22.4, 200)
    ax.plot(grid, 10 ** (a + b * np.log10(grid)), color=MUTED, lw=1.5,
            dashes=(5, 4), zorder=4, alpha=0.8)

    for s in SIZES:
        cc, ll = curves[s][star_idx[s]]
        ax.plot(cc, ll, "*", ms=19, color=FG, mec=EDGE, mew=0.9, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(1e18, 2.2e22)
    ax.set_ylim(0.593, 0.686)
    ax.set_xlabel("$C$ (FLOPs)")
    ax.set_ylabel("EMA averaged loss  $L$", color=FG)
    ax.grid(which="major", alpha=0.85)
    ax.grid(which="minor", alpha=0.35)
    mono_ticks(ax)

    handles = [Line2D([], [], color=SIZE_COLOR[s], lw=3, solid_capstyle="round", label=s)
               for s in SIZES]
    handles.append(Line2D([], [], color=FG, marker="*", ms=11, ls="none",
                          mec=EDGE, mew=0.8, label="optimum"))
    leg = ax.legend(handles=handles, title="model size", ncol=4, loc="lower left",
                    fontsize=8.5, borderpad=0.6, labelspacing=0.45,
                    columnspacing=1.1, handlelength=1.6)
    mono_legend(leg)

    fig.tight_layout()
    p = os.path.join(OUT, "abra_fig1_scaling.png")
    fig.savefig(p, dpi=200, transparent=True)
    plt.close(fig)
    print(f"wrote {p}\n   frontier fit: L = {10**a:.4g} * C^{b:.4f}")
    for s in SIZES:
        cc, ll = curves[s][star_idx[s]]
        print(f"   optimum {s:5s}  C={cc:.3g}  L={ll:.4f}")


# ------------------------------------------------------------- Abra fig 8(c)
def fig8c():
    rows = read("abra_fig8c_probe.csv")
    by_itpp = {200: [], 400: []}
    for r in rows:
        by_itpp[int(r["itpp"])].append((r["size"], float(r["flops"]), float(r["best_top1"])))

    def fit(pts):
        """acc = M_F - k * C^-alpha, ceiling capped at 1.0 (as the paper's does)."""
        C = np.array([p[1] for p in pts])
        y = np.array([p[2] for p in pts])
        best = None
        for M in np.arange(0.60, 1.0001, 0.001):
            for al in np.arange(0.005, 0.30, 0.001):
                X = C ** -al
                k = float(np.sum(X * (M - y)) / np.sum(X * X))
                pred = M - k * X
                r2 = 1 - float(np.sum((pred - y) ** 2)) / float(np.sum((y - y.mean()) ** 2))
                if best is None or r2 > best[0]:
                    best = (r2, M, k, al)
        return best

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    styles = {200: ("o", (0, ()), "iTPP 200"), 400: ("s", (0, (5, 4)), "iTPP 400")}
    grid = np.logspace(18.4, 22.1, 200)
    fit_handles = []
    for itpp in (200, 400):
        pts = sorted(by_itpp[itpp], key=lambda p: p[1])
        r2, M, k, al = fit(pts)
        marker, dash, name = styles[itpp]
        lbl = f"{name}:  M_F={M:.3f}  α={al:.3f}  (R²={r2:.3f})"
        ax.plot(grid, M - k * grid ** -al, color=FAINT, lw=1.5, ls=dash, zorder=2)
        fit_handles.append(Line2D([], [], color=FAINT, lw=1.5, ls=dash, marker=marker,
                                  ms=7, mfc=MUTED, mec=EDGE, mew=0.8, label=lbl))
        for size, c, y in pts:
            ax.plot(c, y, marker, color=SIZE_COLOR[size], ms=9.5, mec=EDGE, mew=1.1, zorder=4)
        print(f"   iTPP {itpp}: M_F={M:.3f} alpha={al:.3f} k={k:.4g} R2={r2:.4f}")

    ax.set_xscale("log")
    ax.set_xlabel("image-token training FLOPs   $C = 6\\,\\mathrm{iTPP}\\,N^2$")
    ax.set_ylabel("best linear-probe top-1 accuracy", color=FG)
    ax.grid(which="major", alpha=0.85)
    ax.grid(which="minor", alpha=0.35)
    mono_ticks(ax)

    size_handles = [Line2D([], [], color=SIZE_COLOR[s], marker="o", ls="none", ms=8,
                           mec=EDGE, mew=0.8, label=s) for s in SIZES]
    leg1 = ax.legend(handles=fit_handles, loc="lower right", fontsize=9, borderpad=0.7)
    mono_legend(leg1)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=size_handles, title="model size", loc="upper left",
                     ncol=3, fontsize=9, borderpad=0.6, columnspacing=1.0)
    mono_legend(leg2)

    fig.tight_layout()
    p = os.path.join(OUT, "abra_fig8c_probe.png")
    fig.savefig(p, dpi=200, transparent=True)
    plt.close(fig)
    print(f"wrote {p}")


# ----------------------------------------------- SR-DiT convergence (fig. 2)
# Values as published, copied from the paper repo's own generate_plots.py.
# Baselines in muted site accents; ours carries the site's link blue and the
# only solid line, so it reads first.
# Baselines are greys, lightening as they get stronger, so the eye reads a
# ladder up to ours rather than four competing hues; ours is the only colour.
SRDIT = [
    ("SiT-B/2",           [200, 400],          [46, 34.8],                FAINT,  "o", (0, (5, 4)), 1.5, 6.5),
    ("SiT-B/2 + REPA",    [50, 100, 200, 400], [78.2, 49.5, 33.2, 24.4],  GRAY,   "s", (0, (5, 4)), 1.5, 6.5),
    ("SiT-B/2 + REG",     [50, 100, 200, 400], [64.7, 36.1, 22.1, 15.2],  MUTED,  "^", (0, (5, 4)), 1.5, 6.5),
    ("SR-DiT-B/1 (Ours)", [50, 100, 200, 400], [21.55, 9.17, 4.91, 3.49], ACCENT, "*", (0, ()),     2.8, 16),
]


def srdit():
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for name, it, fid, col, mk, ls, lw, ms in SRDIT:
        alpha = 1.0 if "Ours" in name else 0.85
        ax.plot(it, fid, label=name, color=col, marker=mk, linestyle=ls, alpha=alpha,
                linewidth=lw, markersize=ms, markeredgewidth=0.9, markeredgecolor=EDGE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training iterations")
    ax.set_ylabel("FID $\\downarrow$", color=FG)
    # no in-plot title: the page carries it as the figure caption
    ax.grid(True, alpha=0.85, which="major")
    ax.grid(True, alpha=0.35, which="minor")
    ax.xaxis.set_major_locator(FixedLocator([50, 100, 200, 400]))
    ax.xaxis.set_major_formatter(FixedFormatter(["50K", "100K", "200K", "400K"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim([49, 410])
    ax.set_yticks([3, 4, 5, 7, 10, 15, 20, 25, 35, 50, 70])
    ax.set_yticklabels(["3", "4", "5", "7", "10", "15", "20", "25", "35", "50", "70"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim([2.5, 90])
    mono_ticks(ax)
    leg = ax.legend(loc="lower left", fontsize=9.5, borderpad=0.7)
    mono_legend(leg)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_convergence.png")
    fig.savefig(p, dpi=200, transparent=True)
    plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    style()
    os.makedirs(OUT, exist_ok=True)
    fig1()
    fig8c()
    srdit()
