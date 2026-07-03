import os
import matplotlib.pyplot as plt

from models.empirical import EMPIRICAL_RHO, EMPIRICAL_V
from analysis.config import OUT_DIR, PALETTE

def save_figure(fig, filename: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, filename))
    plt.close(fig)


def setup_fd_axes(ax, title: str) -> None:
    ax.set(
        xlim=(0, 3.2),
        ylim=(0, 1.5),
        xlabel="Density ρ [1/m]",
        ylabel="Mean velocity v̄ [m/s]",
        title=title,
    )


def plot_empirical(ax, zorder: int = 2) -> None:
    ax.scatter(
        EMPIRICAL_RHO,
        EMPIRICAL_V,
        s=15,
        color=PALETTE["empirical"],
        marker="D",
        linewidths=0,
        zorder=zorder,
        label="Empirical (Seyfried 2005)",
    )

def plot_result_curves(ax, results: dict, colors: dict, markers: dict, label_fmt) -> None:
    for key, (rho, velocity) in sorted(results.items()):
        ax.plot(
            rho,
            velocity,
            "-",
            color=colors.get(key, "#000000"),
            marker=markers.get(key, "x"),
            ms=5,
            lw=1.8,
            label=label_fmt(key),
        )
