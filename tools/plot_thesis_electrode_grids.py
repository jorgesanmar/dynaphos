"""Create thesis-ready electrode-grid and phosphene-location figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from dynaphos.electrodes.visualization import (
    HEMISPHERE_STYLES,
    GridVisualization,
    load_grid_visualization,
    resolve_default_grids,
)
from dynaphos.paths import package_file
from dynaphos.simulation import utils


FONT_SIZE = 14
TICK_SIZE = 12
LEGEND_SIZE = 13
RIGHT_COLOR = HEMISPHERE_STYLES["right"]["color"]
LEFT_COLOR = HEMISPHERE_STYLES["left"]["color"]
HEMISPHERE_DISPLAY_OFFSET_MM = 3.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "axes.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def style_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, labelpad=7)
    ax.set_ylabel(ylabel, labelpad=7)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.7, zorder=0)
    ax.tick_params(direction="out", length=4.5, width=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_electrodes(
    ax: plt.Axes,
    item: GridVisualization,
    limits: tuple[float, float],
) -> None:
    right = item.vis_data["setup_right"]
    left = item.vis_data["setup_left"]
    x_right, y_right = right.cartesian

    x_right_display = np.asarray(x_right) + HEMISPHERE_DISPLAY_OFFSET_MM
    ax.scatter(
        x_right_display,
        y_right,
        s=7.0,
        c=RIGHT_COLOR,
        marker="o",
        edgecolors="none",
        alpha=0.9,
        zorder=3,
    )
    if left is not None:
        x_left, y_left = left.cartesian
        x_left_display = np.asarray(x_left) - HEMISPHERE_DISPLAY_OFFSET_MM
        ax.scatter(
            x_left_display,
            y_left,
            s=8.0,
            c=LEFT_COLOR,
            marker="^",
            edgecolors="none",
            alpha=0.9,
            zorder=4,
        )

    display_limits = (
        limits[0] - HEMISPHERE_DISPLAY_OFFSET_MM,
        limits[1] + HEMISPHERE_DISPLAY_OFFSET_MM,
    )
    physical_ticks = np.arange(-15.0, 15.1, 5.0)
    display_ticks = physical_ticks.copy()
    display_ticks[physical_ticks < 0.0] -= HEMISPHERE_DISPLAY_OFFSET_MM
    display_ticks[physical_ticks > 0.0] += HEMISPHERE_DISPLAY_OFFSET_MM
    display_ticks = np.insert(
        display_ticks[physical_ticks != 0.0],
        3,
        [
            -HEMISPHERE_DISPLAY_OFFSET_MM,
            HEMISPHERE_DISPLAY_OFFSET_MM,
        ],
    )
    tick_labels = ["-15", "-10", "-5", "0", "0", "5", "10", "15"]

    ax.set_xlim(*display_limits)
    ax.set_ylim(*limits)
    ax.set_xticks(display_ticks)
    ax.set_xticklabels(tick_labels)
    style_axis(ax, "Cortical x-position (mm)", "Cortical y-position (mm)")

    break_size = 0.012
    break_positions = (
        -HEMISPHERE_DISPLAY_OFFSET_MM,
        HEMISPHERE_DISPLAY_OFFSET_MM,
    )
    x_span = display_limits[1] - display_limits[0]
    for position in break_positions:
        x_fraction = (position - display_limits[0]) / x_span
        ax.plot(
            [x_fraction - break_size, x_fraction + break_size],
            [-break_size, break_size],
            transform=ax.transAxes,
            color="black",
            linewidth=1.1,
            clip_on=False,
        )


def plot_phosphenes(
    ax: plt.Axes,
    item: GridVisualization,
    limits: tuple[float, float],
    view_angle_deg: float,
) -> None:
    phosphene_map = item.vis_data["final_phosphene"]
    grid_ids = np.asarray(item.vis_data["grid_ids"], dtype=np.int64)
    x_vis, y_vis = phosphene_map.cartesian
    x_vis = np.asarray(x_vis, dtype=float)
    y_vis = np.asarray(y_vis, dtype=float)

    half_view = view_angle_deg / 2.0
    ax.add_patch(
        Rectangle(
            (-half_view, -half_view),
            view_angle_deg,
            view_angle_deg,
            facecolor="#F5F5F5",
            edgecolor="#555555",
            linewidth=1.1,
            linestyle="--",
            zorder=1,
        )
    )
    ax.axhline(0.0, color="#AAAAAA", linewidth=0.7, zorder=1)
    ax.axvline(0.0, color="#AAAAAA", linewidth=0.7, zorder=1)

    left_mask = grid_ids == 1
    right_mask = grid_ids == 0
    ax.scatter(
        x_vis[left_mask],
        y_vis[left_mask],
        s=8.0,
        c=LEFT_COLOR,
        marker="^",
        edgecolors="none",
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        x_vis[right_mask],
        y_vis[right_mask],
        s=7.0,
        c=RIGHT_COLOR,
        marker="o",
        edgecolors="none",
        alpha=0.9,
        zorder=4,
    )

    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    style_axis(ax, "Horizontal visual angle (deg)", "Vertical visual angle (deg)")


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor=LEFT_COLOR,
            markeredgecolor="none",
            markersize=8,
            label="Left hemisphere",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=RIGHT_COLOR,
            markeredgecolor="none",
            markersize=8,
            label="Right hemisphere",
        ),
    ]


def save_all_formats(fig: plt.Figure, path_stem: Path, dpi: int) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"dpi": dpi} if suffix == ".png" else {}
        fig.savefig(
            path_stem.with_suffix(suffix),
            bbox_inches="tight",
            facecolor="white",
            **kwargs,
        )


def save_standalone_figures(
    items: list[GridVisualization],
    out_dir: Path,
    electrode_limits: tuple[float, float],
    phosphene_limits: tuple[float, float],
    view_angle_deg: float,
    dpi: int,
) -> None:
    for item in items:
        spacing = "".join(character for character in item.path.stem if character.isdigit())

        fig_e, ax_e = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        plot_electrodes(ax_e, item, electrode_limits)
        save_all_formats(fig_e, out_dir / f"electrode_grid_{spacing}um", dpi)
        plt.close(fig_e)

        fig_p, ax_p = plt.subplots(figsize=(6.4, 6.0), constrained_layout=True)
        plot_phosphenes(ax_p, item, phosphene_limits, view_angle_deg)
        save_all_formats(fig_p, out_dir / f"phosphene_locations_{spacing}um", dpi)
        plt.close(fig_p)


def save_comparison_figure(
    items: list[GridVisualization],
    out_dir: Path,
    electrode_limits: tuple[float, float],
    phosphene_limits: tuple[float, float],
    view_angle_deg: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.8, 10.2))

    for column, item in enumerate(items):
        plot_electrodes(axes[0, column], item, electrode_limits)
        plot_phosphenes(
            axes[1, column],
            item,
            phosphene_limits,
            view_angle_deg,
        )

    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
        handletextpad=0.6,
        columnspacing=2.4,
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        top=0.985,
        bottom=0.105,
        wspace=0.34,
        hspace=0.34,
    )
    save_all_formats(
        fig,
        out_dir / "electrode_grids_and_phosphene_locations",
        dpi,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        type=Path,
        default=package_file("params"),
        help="DynaPhos parameter YAML used for the cortex mapping.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/thesis_figures/electrode_grids"),
        help="Output directory.",
    )
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()

    configure_style()
    params = utils.load_params(str(args.params.resolve()))
    seed = params.get("run", {}).get("seed", 42)
    view_angle_deg = float(params["run"]["view_angle"])
    items = [
        load_grid_visualization(path, params, seed=seed)
        for path in resolve_default_grids()
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    electrode_limits = (-16.8, 16.8)
    phosphene_extent = max(
        view_angle_deg / 2.0,
        max(
            float(np.max(np.abs(item.vis_data["final_phosphene"].complex)))
            for item in items
        ),
    )
    phosphene_extent *= 1.08
    phosphene_limits = (-phosphene_extent, phosphene_extent)

    save_standalone_figures(
        items,
        args.out,
        electrode_limits,
        phosphene_limits,
        view_angle_deg,
        args.dpi,
    )
    save_comparison_figure(
        items,
        args.out,
        electrode_limits,
        phosphene_limits,
        view_angle_deg,
        args.dpi,
    )

    print(f"Saved thesis figures to {args.out.resolve()}")
    for item in items:
        spacing = "".join(character for character in item.path.stem if character.isdigit())
        phosphene_count = len(item.vis_data["final_phosphene"].complex)
        print(
            f"{spacing} um: {len(item.x_cortex)} physical electrodes, "
            f"{phosphene_count} mapped phosphene locations"
        )


if __name__ == "__main__":
    main()
