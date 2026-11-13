"""Shared figure style.

One look for every figure in the project: a light surface, recessive axes and
grid, and a small fixed palette in which a colour always means the same thing.
Every figure carries a provenance line naming its dataset, split and units.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # figures are written to files; no display is needed

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3de"
NEUTRAL_FILL = "#d9d8d2"

# Colour follows the entity: the same band is the same colour in every figure.
COLOR_LFP = "#2a78d6"  # blue
COLOR_SPIKE = "#eb6834"  # orange
COLOR_ACCENT = "#1baf7a"  # aqua, for a third series when one is needed

DPI = 200


@contextmanager
def figure_style() -> Iterator[None]:
    """Apply the project style inside a ``with`` block."""
    params: dict[Any, Any] = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
    }
    with plt.rc_context(params):
        yield


def add_headline(fig: Figure, headline: str, subtitle: str) -> None:
    """Put a finding-first headline and a one-line subtitle at the top left."""
    fig.text(0.012, 0.985, headline, ha="left", va="top", fontsize=13.5, fontweight="bold")
    fig.text(0.012, 0.948, subtitle, ha="left", va="top", fontsize=9.5, color=INK_SECONDARY)


def add_provenance(fig: Figure, *, dataset: str, split: str, units: str) -> None:
    """Write the dataset, split and units along the bottom edge of a figure."""
    text = f"Dataset: {dataset}\nSplit: {split}\nUnits: {units}"
    fig.text(
        0.012,
        0.012,
        text,
        ha="left",
        va="bottom",
        fontsize=7.8,
        color=INK_SECONDARY,
        linespacing=1.5,
    )


def panel_label(ax: Axes, text: str) -> None:
    """Set a left-aligned panel title."""
    ax.set_title(text)


def save_figure(fig: Figure, path: Path) -> Path:
    """Save a figure as PNG at the project resolution and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path
