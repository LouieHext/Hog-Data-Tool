import os
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def set_hog_time_axis(
    ax: Axes,
) -> None:
    """Set consistent time axis formatting for HOG plots."""
    ax.set_yticks([0, 30, 60, 90, 120, 180, 240, 300])
    ax.set_ylim(0, 300)


def style_axis(
    ax: Axes,
    *,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    rotation: int = 45,
    grid: bool = True,
) -> None:
    """Apply consistent styling to an axis."""
    if title:
        ax.set_title(title, fontsize=20, pad=10, weight="bold")
    if x_label:
        ax.set_xlabel(x_label, fontsize=20, labelpad=8, weight="bold")
    if y_label:
        ax.set_ylabel(y_label, fontsize=20, labelpad=8, weight="bold")

    for lbl in ax.get_xticklabels():
        lbl.set_rotation(rotation)

    ax.tick_params(axis="both", which="major", labelsize=16, width=2, length=8)

    if grid:
        ax.grid(linestyle="--", alpha=0.7, color="black")

    for axis in ["bottom", "left"]:
        ax.spines[axis].set_linewidth(2.5)
        ax.spines[axis].set_color("0.2")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def create_figure(
    figsize: tuple[int, int] = (12, 6),
    facecolor: str = "white",
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_facecolor(facecolor)
    return fig, ax


def save_figure(
    fig: Figure,
    output_path: Path | None = None,
) -> None:
    """Save a figure to the specified path, ensuring the directory exists."""
    if output_path is None:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
