"""
plot_style.py
-------------
One shared Matplotlib style so every figure in the report looks consistent:
white background, light grid, thin lines, no top/right borders.
"""

import matplotlib

matplotlib.use("Agg")  # render to files only (no window needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import COLORS, FIG_DIR


def apply_style():
    """Set global Matplotlib defaults used by all charts."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["grid"],
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": COLORS["text"],
        "axes.labelcolor": COLORS["text_muted"],
        "axes.labelsize": 10,
        "xtick.color": COLORS["text_muted"],
        "ytick.color": COLORS["text_muted"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "font.family": "DejaVu Sans",
    })


def millions(ax, axis="y"):
    """Format an axis as $ millions, e.g. 1.5M."""
    fmt = mticker.FuncFormatter(lambda v, _: f"${v / 1e6:,.1f}M")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def thousands(ax, axis="y"):
    """Format an axis as $ thousands, e.g. 850K."""
    fmt = mticker.FuncFormatter(lambda v, _: f"${v / 1e3:,.0f}K")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def save(fig, name):
    """Save a figure into outputs/figures and close it to free memory."""
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved figure -> outputs/figures/{name}")
    return path
