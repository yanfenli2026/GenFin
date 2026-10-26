"""Shared figure style for the main figures.

- sans-serif (Arial, fallback DejaVu Sans), 7 pt text, 8 pt bold lower-case panel letters
- white background, no titles, line weights >= 1 pt
- Okabe-Ito colour-blind-safe palette
- fixed physical widths: single column 88 mm, double column 180 mm
- figures are saved WITHOUT bbox cropping so the physical width is exact
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM = 1 / 25.4
W_SINGLE = 88 * MM   # 3.465 in
W_DOUBLE = 180 * MM  # 7.087 in

FS = 7          # all text
FS_LETTER = 8   # panel letters
LW = 1.0        # minimum line weight (pt)

# Okabe & Ito (2008)
OI = {
    "orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
    "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7", "black": "#000000",
    "grey": "#999999",
}


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.default": "regular",
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "legend.fontsize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS,
        "axes.linewidth": LW, "lines.linewidth": LW, "patch.linewidth": LW,
        "xtick.major.width": LW, "ytick.major.width": LW, "xtick.minor.width": LW, "ytick.minor.width": LW,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "errorbar.capsize": 2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.unicode_minus": True,
    })
    return plt


def panel_letter(ax, letter, x=None, y=1.0, fig=None):
    """Bold lower-case letter at the top-left of an axes (outside the plotting area)."""
    if x is None:
        x = -0.08
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=FS_LETTER, fontweight="bold",
            ha="left", va="bottom")


def fig_letter(fig, letter, x, y):
    """Bold letter positioned in figure coordinates (top-left anchored)."""
    fig.text(x, y, letter, fontsize=FS_LETTER, fontweight="bold", ha="left", va="top")


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p + ".pdf", facecolor="white")
    fig.savefig(p + ".png", dpi=300, facecolor="white")
    plt.close(fig)
    return p
