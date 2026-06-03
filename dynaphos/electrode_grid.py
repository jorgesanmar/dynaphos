import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos import cortex_models, utils
from dynaphos.utils import Map


RIGHT_HEMISPHERE_COLOR = "#D62728"
LEFT_HEMISPHERE_COLOR = "#1F77B4"
RIGHT_HEMISPHERE_FILL = "#FCE8E6"
LEFT_HEMISPHERE_FILL = "#E7F0FA"
HEMISPHERE_EDGE_COLOR = "white"
HEMISPHERE_MARKER_EDGE_WIDTH = 0.35
HEMISPHERE_STYLES = {
    "right": {
        "label": "Right hemisphere",
        "color": RIGHT_HEMISPHERE_COLOR,
        "marker": "o",
    },
    "left": {
        "label": "Left hemisphere",
        "color": LEFT_HEMISPHERE_COLOR,
        "marker": "^",
    },
}
REGION_GRAY = "#D9D9D9"
PHOSPHENE_PLOT_VIEW_ANGLE_DEG = 3.0
GRID_COLORS = {
    400: "#B91C1C",
    800: "#1F4E79",
    1200: "#2E8B57",
}
FALLBACK_GRID_COLOR = "#6B7280"


@dataclass
class GridVisualization:
    path: Path
    name: str
    x_cortex: np.ndarray
    y_cortex: np.ndarray
    vis_data: dict
    area_mm2: float | None


def resolve_default_grid() -> Path:
    preferred = PROJECT_ROOT / "config" / "coords_800um.yaml"
    if preferred.exists():
        return preferred.resolve()
    fallback = PROJECT_ROOT / "config" / "grid_coords_dipole.yaml"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError("Could not find a default dipole grid YAML file in config/.")


def resolve_default_grids() -> list[Path]:
    preferred = [
        PROJECT_ROOT / "config" / "coords_400um.yaml",
        PROJECT_ROOT / "config" / "coords_800um.yaml",
        PROJECT_ROOT / "config" / "coords_1200um.yaml",
    ]
    existing = [path.resolve() for path in preferred if path.exists()]
    return existing or [resolve_default_grid()]


def resolve_grid_paths(grid_args: list[str] | None) -> list[Path]:
    if not grid_args:
        return resolve_default_grids()

    paths = []
    for grid_arg in grid_args:
        path = Path(grid_arg)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Grid file not found: {path}")
        paths.append(path)
    return paths


def pretty_grid_name(path: Path) -> str:
    name = path.stem
    return name.replace("full_fov_", "").replace("coords_", "").replace("_", " ")


def grid_spacing_um(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def grid_color(path: Path) -> str:
    spacing = grid_spacing_um(path)
    return GRID_COLORS.get(spacing, FALLBACK_GRID_COLOR)


def convex_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    points = np.column_stack([x, y]).astype(float)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in points[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.array(lower[:-1] + upper[:-1])


def calculate_polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the area of a polygon using the shoelace formula."""
    # Ensure the polygon is closed
    x_closed = np.append(x, x[0])
    y_closed = np.append(y, y[0])
    # Shoelace formula
    area = 0.5 * np.abs(np.sum(x_closed[:-1] * y_closed[1:] - x_closed[1:] * y_closed[:-1]))
    return area


def smooth_expand_outline(points: np.ndarray, pad_ratio: float = 0.08, n_samples: int = 400) -> np.ndarray:
    if len(points) < 3:
        return points

    center = points.mean(axis=0)
    vectors = points - center
    expanded = center + vectors * (1.0 + pad_ratio)

    closed = np.vstack([expanded, expanded[0]])
    segments = np.diff(closed, axis=0)
    segment_lengths = np.sqrt((segments ** 2).sum(axis=1))
    cumulative = np.hstack([[0.0], np.cumsum(segment_lengths)])
    total = cumulative[-1]
    if total <= 0:
        return expanded

    samples = np.linspace(0.0, total, n_samples, endpoint=False)
    xi = np.interp(samples, cumulative, closed[:, 0])
    yi = np.interp(samples, cumulative, closed[:, 1])

    kernel_size = 13
    pad = kernel_size // 2
    xpad = np.r_[xi[-pad:], xi, xi[:pad]]
    ypad = np.r_[yi[-pad:], yi, yi[:pad]]
    kernel = np.ones(kernel_size) / kernel_size
    xs = np.convolve(xpad, kernel, mode="valid")
    ys = np.convolve(ypad, kernel, mode="valid")
    return np.column_stack([xs, ys])


def get_phosphene_coordinates(
    params: dict,
    x_cortex: np.ndarray,
    y_cortex: np.ndarray,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates_cortex = Map(x=x_cortex, y=y_cortex)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    if hasattr(cortex_models, "get_visual_field_coordinates_from_cortex_full"):
        mapped = cortex_models.get_visual_field_coordinates_from_cortex_full(
            params["cortex_model"],
            coordinates_cortex,
            rng=rng,
        )
    else:
        mapped = cortex_models.get_visual_field_coordinates_from_cortex(
            params["cortex_model"],
            coordinates_cortex,
            rng=rng,
        )
    phosphene_map = mapped[0] if isinstance(mapped, tuple) else mapped
    return phosphene_map.cartesian


def build_full_field_visualization_data(
    params: dict,
    x_cortex: np.ndarray,
    y_cortex: np.ndarray,
    seed: Optional[int] = None,
) -> dict:
    coordinates_cortex = Map(x=np.asarray(x_cortex, dtype=float), y=np.asarray(y_cortex, dtype=float))
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    if hasattr(cortex_models, "get_full_field_mapping_from_cortex"):
        x_sorted, y_sorted, _ = cortex_models._sort_cortical_coordinates(coordinates_cortex)
        right_setup = Map(x=np.abs(x_sorted), y=y_sorted)
        right_visual = cortex_models._map_cortical_coordinates_to_displayed_visual_field(
            params["cortex_model"],
            "right",
            x_sorted,
            y_sorted,
        )
        left_visual = cortex_models.mirror_map_along_y_axis(right_visual)
        left_setup = cortex_models._map_displayed_visual_field_to_cortex(
            params["cortex_model"],
            left_visual,
            "left",
        )
        full_mapping = cortex_models.get_full_field_mapping_from_cortex(
            params["cortex_model"],
            coordinates_cortex=coordinates_cortex,
            rng=rng,
        )

        grid_ids = np.asarray(full_mapping.grid_ids, dtype=np.int64)
        return {
            "is_full_field": True,
            "setup_right": right_setup,
            "setup_left": left_setup,
            "final_cortical": full_mapping.cortical_coordinates,
            "final_phosphene": full_mapping.phosphene_map,
            "grid_ids": grid_ids,
        }

    x_vis, y_vis = get_phosphene_coordinates(params, x_cortex, y_cortex, seed=seed)
    return {
        "is_full_field": False,
        "setup_right": coordinates_cortex,
        "setup_left": None,
        "final_cortical": coordinates_cortex,
        "final_phosphene": Map(x=x_vis, y=y_vis),
        "grid_ids": np.zeros(len(x_vis), dtype=np.int64),
    }


def draw_region(ax, x, y, region_color="#D3D3D3", outline_color="#000000", outline_lw=2.0):
    hull = convex_hull(np.asarray(x), np.asarray(y))
    if len(hull) >= 3:
        smooth = smooth_expand_outline(hull, pad_ratio=0.08, n_samples=500)
        ax.fill(smooth[:, 0], smooth[:, 1], color=region_color, zorder=1)
        hx = np.r_[smooth[:, 0], smooth[0, 0]]
        hy = np.r_[smooth[:, 1], smooth[0, 1]]
        ax.plot(hx, hy, color=outline_color, linewidth=outline_lw, zorder=2)


def style_ax_clean(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")


def style_ax_academic(ax, *, xlabel: str, ylabel: str, title: str) -> None:
    style_ax_clean(ax)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(labelsize=9, width=0.8)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.axhline(0.0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.5, zorder=0)
    ax.axvline(0.0, color="#666666", linewidth=0.9, linestyle="--", alpha=0.5, zorder=0)


def shade_cortical_hemispheres(ax) -> None:
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()

    if x_limits[0] < 0.0:
        ax.axvspan(
            x_limits[0],
            min(0.0, x_limits[1]),
            facecolor=LEFT_HEMISPHERE_FILL,
            linewidth=0,
            zorder=-1,
        )
    if x_limits[1] > 0.0:
        ax.axvspan(
            max(0.0, x_limits[0]),
            x_limits[1],
            facecolor=RIGHT_HEMISPHERE_FILL,
            linewidth=0,
            zorder=-1,
        )

    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)


def padded_limits(values: np.ndarray, *, pad_fraction: float = 0.08, symmetric: bool = False) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0)

    if symmetric:
        half = float(np.max(np.abs(values)))
        if half <= 0:
            half = 1.0
        half *= 1.0 + pad_fraction
        return (-half, half)

    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    if span <= 0:
        span = max(abs(upper), 1.0)
    pad = span * pad_fraction
    return (lower - pad, upper + pad)


def combined_limits(
    items: list[GridVisualization],
    getter,
    *,
    pad_fraction: float = 0.08,
    symmetric: bool = False,
) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = []
    ys = []
    for item in items:
        x_values, y_values = getter(item)
        xs.append(np.asarray(x_values, dtype=float).ravel())
        ys.append(np.asarray(y_values, dtype=float).ravel())
    return (
        padded_limits(np.concatenate(xs), pad_fraction=pad_fraction, symmetric=symmetric),
        padded_limits(np.concatenate(ys), pad_fraction=pad_fraction, symmetric=symmetric),
    )


def set_axis_limits(ax, limits: tuple[tuple[float, float], tuple[float, float]] | None) -> None:
    if limits is None:
        ax.margins(0.08)
        return
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])


def full_visual_field_limits(
    limits: tuple[tuple[float, float], tuple[float, float]] | None,
    *,
    view_angle_deg: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if limits is not None:
        x_limits, y_limits = limits
        values = np.asarray([*x_limits, *y_limits], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            half = float(np.max(np.abs(values)))
            if half > 0:
                return ((-half, half), (-half, half))

    half = float(view_angle_deg) / 2.0
    return ((-half, half), (-half, half))


def plot_electrode_panel(
    ax,
    x,
    y,
    dot_size=8.0,
    dot_color="#D62728",
    *,
    title: str = "Right Cortical Electrode Grid",
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
):
    ax.scatter(x, y, s=dot_size, c=dot_color, linewidths=0, zorder=3)
    style_ax_academic(ax, xlabel="x (mm)", ylabel="y (mm)", title=title)
    set_axis_limits(ax, limits)


def plot_dual_electrode_panel(
    ax,
    right_coords: Map,
    left_coords: Optional[Map],
    *,
    dot_size: float,
    dot_color: str | None = None,
    title: str = "Cortical Electrode Grid Coordinates",
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    show_legend: bool = True,
) -> None:
    x_right, y_right = right_coords.cartesian
    right_style = HEMISPHERE_STYLES["right"]
    left_style = HEMISPHERE_STYLES["left"]
    right_color = dot_color if left_coords is None and dot_color is not None else right_style["color"]
    ax.scatter(
        x_right,
        y_right,
        s=dot_size,
        c=right_color,
        marker=right_style["marker"],
        edgecolors=HEMISPHERE_EDGE_COLOR,
        linewidths=HEMISPHERE_MARKER_EDGE_WIDTH,
        zorder=3,
        label=right_style["label"],
    )

    if left_coords is not None:
        x_left, y_left = left_coords.cartesian
        ax.scatter(
            x_left,
            y_left,
            s=dot_size,
            c=left_style["color"],
            marker=left_style["marker"],
            edgecolors=HEMISPHERE_EDGE_COLOR,
            linewidths=HEMISPHERE_MARKER_EDGE_WIDTH,
            zorder=4,
            label=left_style["label"],
        )

    style_ax_academic(
        ax,
        xlabel="x (mm)",
        ylabel="y (mm)",
        title=title,
    )
    set_axis_limits(ax, limits)
    if left_coords is not None:
        shade_cortical_hemispheres(ax)
    handles, labels = ax.get_legend_handles_labels()
    if not show_legend:
        return
    if left_coords is not None:
        handles = handles[::-1]
        labels = labels[::-1]
    ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )


def draw_visual_field_reference(
    ax,
    *,
    limits: tuple[tuple[float, float], tuple[float, float]] | None,
    view_angle_deg: float,
) -> None:
    half = float(view_angle_deg) / 2.0
    if limits is not None:
        x_limits, y_limits = limits
        values = np.asarray([*x_limits, *y_limits], dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            half = max(half, float(np.max(np.abs(values))))

    center = (0.0, 0.0)

    outer = plt.Circle(center, half, facecolor=REGION_GRAY, edgecolor="black", linewidth=2.0, zorder=1, clip_on=False)
    ax.add_patch(outer)

    for frac in (0.25, 0.5, 0.75):
        ring = plt.Circle(center, half * frac, facecolor="none", edgecolor="black", linewidth=1.0, zorder=2)
        ax.add_patch(ring)

    angles = np.deg2rad([0, 45, 90, 135])
    for angle in angles:
        dx = np.cos(angle) * half
        dy = np.sin(angle) * half
        ax.plot([-dx, dx], [-dy, dy], color="black", linewidth=1.4, zorder=2)


def plot_phosphene_panel(
    ax,
    x,
    y,
    view_angle_deg=PHOSPHENE_PLOT_VIEW_ANGLE_DEG,
    dot_size=10.0,
    dot_color="#D62728",
    *,
    title: str = "Phosphene Map",
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
):
    limits = full_visual_field_limits(limits, view_angle_deg=view_angle_deg)
    ax.scatter(x, y, s=dot_size, c=dot_color, linewidths=0, zorder=3)
    style_ax_academic(ax, xlabel="x (deg)", ylabel="y (deg)", title=title)
    set_axis_limits(ax, limits)


def plot_grid_colored_phosphene_panel(
    ax,
    phosphene_coords: Map,
    grid_ids: np.ndarray,
    *,
    view_angle_deg: float,
    dot_size: float,
    dot_color: str | None = None,
    title: str = "Full-Field Phosphene Map",
    limits: tuple[tuple[float, float], tuple[float, float]] | None = None,
    show_legend: bool = True,
) -> None:
    x_vis, y_vis = phosphene_coords.cartesian
    x_vis = np.asarray(x_vis, dtype=float)
    y_vis = np.asarray(y_vis, dtype=float)
    grid_ids = np.asarray(grid_ids, dtype=np.int64)

    limits = full_visual_field_limits(limits, view_angle_deg=view_angle_deg)

    masks = [
        ("Phosphenes from left hemisphere", grid_ids == 1, HEMISPHERE_STYLES["left"]),
        ("Phosphenes from right hemisphere", grid_ids == 0, HEMISPHERE_STYLES["right"]),
    ]
    handles = []
    for label, mask, style in masks:
        if np.any(mask):
            ax.scatter(
                x_vis[mask],
                y_vis[mask],
                s=dot_size,
                c=style["color"],
                marker=style["marker"],
                edgecolors=HEMISPHERE_EDGE_COLOR,
                linewidths=HEMISPHERE_MARKER_EDGE_WIDTH,
                zorder=3,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=style["marker"],
                    color="none",
                    markerfacecolor=style["color"],
                    markeredgecolor=HEMISPHERE_EDGE_COLOR,
                    markeredgewidth=HEMISPHERE_MARKER_EDGE_WIDTH,
                    markersize=7,
                    label=label,
                )
            )

    if not handles:
        ax.scatter(x_vis, y_vis, s=dot_size, c=dot_color or FALLBACK_GRID_COLOR, linewidths=0, zorder=3)

    style_ax_academic(ax, xlabel="x (deg)", ylabel="y (deg)", title=title)
    set_axis_limits(ax, limits)
    if handles and show_legend:
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)


def load_grid_visualization(path: Path, params: dict, seed: Optional[int]) -> GridVisualization:
    x_cortex, y_cortex = utils.load_coordinates_from_yaml(str(path))
    hull = convex_hull(x_cortex, y_cortex)
    area = calculate_polygon_area(hull[:, 0], hull[:, 1]) if len(hull) >= 3 else None
    vis_data = build_full_field_visualization_data(params, x_cortex, y_cortex, seed=seed)
    return GridVisualization(
        path=path,
        name=pretty_grid_name(path),
        x_cortex=np.asarray(x_cortex, dtype=float),
        y_cortex=np.asarray(y_cortex, dtype=float),
        vis_data=vis_data,
        area_mm2=area,
    )


def setup_coordinates(item: GridVisualization) -> tuple[np.ndarray, np.ndarray]:
    setup_right = item.vis_data["setup_right"]
    setup_left = item.vis_data["setup_left"]
    x_parts = []
    y_parts = []
    for coords in (setup_right, setup_left):
        if coords is None:
            continue
        x_values, y_values = coords.cartesian
        x_parts.append(np.asarray(x_values, dtype=float))
        y_parts.append(np.asarray(y_values, dtype=float))
    return np.concatenate(x_parts), np.concatenate(y_parts)


def phosphene_coordinates(item: GridVisualization) -> tuple[np.ndarray, np.ndarray]:
    return item.vis_data["final_phosphene"].cartesian


def save_comparison_figures(
    items: list[GridVisualization],
    out_dir: Path,
    *,
    dot_size: float,
    view_angle: float,
) -> None:
    if len(items) < 2:
        return

    electrode_limits = combined_limits(items, setup_coordinates, pad_fraction=0.08, symmetric=False)
    phosphene_limits = combined_limits(items, phosphene_coordinates, pad_fraction=0.08, symmetric=True)
    n_grids = len(items)

    fig_e, axes_e = plt.subplots(1, n_grids, figsize=(4.2 * n_grids, 4.2), dpi=400, squeeze=False)
    fig_e.patch.set_facecolor("white")
    for ax, item in zip(axes_e[0], items):
        color = grid_color(item.path)
        plot_dual_electrode_panel(
            ax,
            item.vis_data["setup_right"],
            item.vis_data["setup_left"],
            dot_size=dot_size * 0.5,
            dot_color=color,
            title=f"{item.name}\n{len(item.x_cortex)} electrodes",
            limits=electrode_limits,
            show_legend=False,
        )
    fig_e.savefig(out_dir / "electrode_grid_comparison.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig_e)
    print(f"saved: {out_dir / 'electrode_grid_comparison.png'}")

    fig_p, axes_p = plt.subplots(1, n_grids, figsize=(4.2 * n_grids, 4.2), dpi=400, squeeze=False)
    fig_p.patch.set_facecolor("white")
    for ax, item in zip(axes_p[0], items):
        color = grid_color(item.path)
        if item.vis_data["is_full_field"]:
            plot_grid_colored_phosphene_panel(
                ax,
                item.vis_data["final_phosphene"],
                item.vis_data["grid_ids"],
                view_angle_deg=view_angle,
                dot_size=dot_size * 0.5,
                dot_color=color,
                title=f"{item.name}\nphosphene map",
                limits=phosphene_limits,
                show_legend=False,
            )
        else:
            x_vis, y_vis = phosphene_coordinates(item)
            plot_phosphene_panel(
                ax,
                x_vis,
                y_vis,
                view_angle_deg=view_angle,
                dot_size=dot_size * 0.5,
                dot_color=color,
                title=f"{item.name}\nphosphene map",
                limits=phosphene_limits,
            )
    fig_p.savefig(out_dir / "phosphene_map_comparison.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig_p)
    print(f"saved: {out_dir / 'phosphene_map_comparison.png'}")

    fig_m, axes_m = plt.subplots(2, n_grids, figsize=(4.2 * n_grids, 8.2), dpi=400, squeeze=False)
    fig_m.patch.set_facecolor("white")
    for col, item in enumerate(items):
        color = grid_color(item.path)
        plot_dual_electrode_panel(
            axes_m[0, col],
            item.vis_data["setup_right"],
            item.vis_data["setup_left"],
            dot_size=dot_size * 0.45,
            dot_color=color,
            title=f"{item.name}\n{len(item.x_cortex)} electrodes",
            limits=electrode_limits,
            show_legend=False,
        )
        axes_m[0, col].set_xlabel("")
        if item.vis_data["is_full_field"]:
            plot_grid_colored_phosphene_panel(
                axes_m[1, col],
                item.vis_data["final_phosphene"],
                item.vis_data["grid_ids"],
                view_angle_deg=view_angle,
                dot_size=dot_size * 0.45,
                dot_color=color,
                title="Phosphene map",
                limits=phosphene_limits,
                show_legend=False,
            )
        else:
            x_vis, y_vis = phosphene_coordinates(item)
            plot_phosphene_panel(
                axes_m[1, col],
                x_vis,
                y_vis,
                view_angle_deg=view_angle,
                dot_size=dot_size * 0.45,
                dot_color=color,
                title="Phosphene map",
                limits=phosphene_limits,
            )
    fig_m.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08, wspace=0.32, hspace=0.52)
    fig_m.savefig(out_dir / "electrode_and_phosphene_comparison.png", dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig_m)
    print(f"saved: {out_dir / 'electrode_and_phosphene_comparison.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize electrode coordinates in cortex space and receptive-field space."
    )
    parser.add_argument(
        "--grid",
        "--coords-yaml",
        dest="grids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more coordinate YAML files. Defaults to config/coords_400um.yaml, "
            "config/coords_800um.yaml, and config/coords_1200um.yaml when present."
        ),
    )
    parser.add_argument("--params", type=str, default=str(PROJECT_ROOT / "config" / "params.yaml"))
    parser.add_argument(
        "--out",
        type=str,
        default=str(PROJECT_ROOT / "results" / "electrode_grid" / "visuals"),
        help="Output directory for visuals.",
    )
    parser.add_argument("--dot-size", type=float, default=8.0, help="Marker size.")
    parser.add_argument("--save-combined", action="store_true", help="Also save a side-by-side combined figure.")
    args = parser.parse_args()

    grid_paths = resolve_grid_paths(args.grids)

    params_path = Path(args.params)
    if not params_path.is_absolute():
        params_path = (PROJECT_ROOT / params_path).resolve()
    if not params_path.exists():
        raise FileNotFoundError(f"Params file not found: {params_path}")

    params = utils.load_params(str(params_path))
    seed = params.get("run", {}).get("seed", None)
    view_angle = PHOSPHENE_PLOT_VIEW_ANGLE_DEG
    dropout_rate = params.get("cortex_model", {}).get("dropout_rate", 0.0)
    grid_items = [load_grid_visualization(path, params, seed=seed) for path in grid_paths]

    for item in grid_items:
        if item.area_mm2 is None:
            print(f"{item.path.name}: {len(item.x_cortex)} electrodes | area unavailable")
        else:
            print(f"{item.path.name}: {len(item.x_cortex)} electrodes | area {item.area_mm2:.2f} mm^2")

    shared_setup_limits = combined_limits(grid_items, setup_coordinates, pad_fraction=0.08, symmetric=False)
    shared_phosphene_limits = combined_limits(grid_items, phosphene_coordinates, pad_fraction=0.1, symmetric=True)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()

    for item in grid_items:
        color = grid_color(item.path)
        grid_out_dir = out_dir / item.path.stem / f"{dropout_rate:.1f}"
        grid_out_dir.mkdir(parents=True, exist_ok=True)

        out_prefix = grid_out_dir / "electrode_grid"
        stem = out_prefix.stem
        suffix = out_prefix.suffix if out_prefix.suffix else ".png"
        electrodes_out = out_prefix.with_name(f"{stem}_electrodes{suffix}")
        phosphenes_out = out_prefix.with_name(f"{stem}_phosphenes{suffix}")
        full_setup_out = out_prefix.with_name(f"{stem}_full_setup{suffix}")
        mapping_overview_out = out_prefix.with_name(f"{stem}_mapping_overview{suffix}")

        vis_data = item.vis_data
        setup_right = vis_data["setup_right"]
        setup_left = vis_data["setup_left"]
        final_phosphene = vis_data["final_phosphene"]
        grid_ids = vis_data["grid_ids"]
        right_limits = combined_limits([item], lambda grid: (grid.x_cortex, grid.y_cortex), pad_fraction=0.08)

        fig_e, ax_e = plt.subplots(figsize=(5, 5), dpi=400)
        fig_e.patch.set_facecolor("white")
        plot_electrode_panel(
            ax_e,
            item.x_cortex,
            item.y_cortex,
            dot_size=args.dot_size,
            dot_color=color,
            title=f"{item.name} Right Electrode Grid",
            limits=right_limits,
        )
        fig_e.savefig(electrodes_out, dpi=400, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig_e)
        print(f"saved: {electrodes_out}")

        fig_m, ax_m = plt.subplots(figsize=(5, 5), dpi=400)
        fig_m.patch.set_facecolor("white")
        plot_dual_electrode_panel(
            ax_m,
            setup_right,
            setup_left,
            dot_size=args.dot_size * 0.6,
            dot_color=color,
            title=f"{item.name} Cortical Electrode Grids",
            limits=shared_setup_limits,
        )
        fig_m.savefig(full_setup_out, dpi=400, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig_m)
        print(f"saved: {full_setup_out}")

        fig_p, ax_p = plt.subplots(figsize=(5, 5), dpi=400)
        fig_p.patch.set_facecolor("white")
        if vis_data["is_full_field"]:
            plot_grid_colored_phosphene_panel(
                ax_p,
                final_phosphene,
                grid_ids,
                view_angle_deg=view_angle,
                dot_size=args.dot_size,
                dot_color=color,
                title=f"{item.name} Phosphene Map",
                limits=shared_phosphene_limits,
            )
        else:
            x_vis, y_vis = final_phosphene.cartesian
            plot_phosphene_panel(
                ax_p,
                x_vis,
                y_vis,
                view_angle_deg=view_angle,
                dot_size=args.dot_size,
                dot_color=color,
                title=f"{item.name} Phosphene Map",
                limits=shared_phosphene_limits,
            )
        fig_p.savefig(phosphenes_out, dpi=400, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig_p)
        print(f"saved: {phosphenes_out}")

        fig_o, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=400)
        fig_o.patch.set_facecolor("white")
        plot_dual_electrode_panel(
            axes[0],
            setup_right,
            setup_left,
            dot_size=args.dot_size,
            dot_color=color,
            title="Cortical electrodes",
            limits=shared_setup_limits,
        )
        if vis_data["is_full_field"]:
            plot_grid_colored_phosphene_panel(
                axes[1],
                final_phosphene,
                grid_ids,
                view_angle_deg=view_angle,
                dot_size=args.dot_size,
                dot_color=color,
                title="Phosphene map",
                limits=shared_phosphene_limits,
            )
        else:
            x_vis, y_vis = final_phosphene.cartesian
            plot_phosphene_panel(
                axes[1],
                x_vis,
                y_vis,
                view_angle_deg=view_angle,
                dot_size=args.dot_size,
                dot_color=color,
                title="Phosphene map",
                limits=shared_phosphene_limits,
            )
        fig_o.suptitle(item.name, fontsize=13)
        plt.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18, wspace=0.35)
        fig_o.savefig(mapping_overview_out, dpi=400, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig_o)
        print(f"saved: {mapping_overview_out}")

        if args.save_combined:
            combined_out = out_prefix.with_name(f"{stem}_combined{suffix}")
            fig_c, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=400)
            fig_c.patch.set_facecolor("white")
            plot_electrode_panel(
                axes[0],
                item.x_cortex,
                item.y_cortex,
                dot_size=args.dot_size,
                dot_color=color,
                title="Right electrodes",
                limits=right_limits,
            )
            if vis_data["is_full_field"]:
                plot_grid_colored_phosphene_panel(
                    axes[1],
                    final_phosphene,
                    grid_ids,
                    view_angle_deg=view_angle,
                    dot_size=args.dot_size,
                    dot_color=color,
                    title="Phosphene map",
                    limits=shared_phosphene_limits,
                )
            else:
                x_vis, y_vis = final_phosphene.cartesian
                plot_phosphene_panel(
                    axes[1],
                    x_vis,
                    y_vis,
                    view_angle_deg=view_angle,
                    dot_size=args.dot_size,
                    dot_color=color,
                    title="Phosphene map",
                    limits=shared_phosphene_limits,
                )
            plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.26)
            fig_c.savefig(combined_out, dpi=400, bbox_inches="tight", pad_inches=0, facecolor="white")
            plt.close(fig_c)
            print(f"saved: {combined_out}")

    comparison_out_dir = out_dir / "_comparisons" / f"{dropout_rate:.1f}"
    comparison_out_dir.mkdir(parents=True, exist_ok=True)
    save_comparison_figures(
        grid_items,
        comparison_out_dir,
        dot_size=args.dot_size,
        view_angle=view_angle,
    )
    return

if __name__ == "__main__":
    main()
