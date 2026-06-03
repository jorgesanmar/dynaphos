"""
Visualize the two heat-source locations used by the Bioheat2D model.

The plot shows:
- electrode Joule heating as one red point at each electrode location
- internal-circuit heating as a uniform blue footprint over the implant area
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos import utils
from dynaphos.simulator import Bioheat2D


DEFAULT_PARAMS = PROJECT_ROOT / "config" / "params.yaml"
DEFAULT_COORDS = PROJECT_ROOT / "config" / "coords_800um.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "safety" / "visuals" / "heat_sources.png"

JOULE_COLOR = "#DC2626"
JOULE_EDGE_COLOR = "white"
IC_COLOR = "#2563EB"
IC_EDGE_COLOR = "#1D4ED8"
AXIS_COLOR = "#374151"
GRID_COLOR = "#AEB7C2"


@dataclass(frozen=True)
class HeatSourceFootprint:
    coords_path: Path
    x_mm: np.ndarray
    y_mm: np.ndarray
    mask: np.ndarray
    extent_mm: tuple[float, float, float, float]
    pixel_area_mm2: float
    internal_circuit_power_density_w_m3: float

    @property
    def electrode_count(self) -> int:
        return int(self.x_mm.size)

    @property
    def internal_circuit_area_mm2(self) -> float:
        return float(np.count_nonzero(self.mask) * self.pixel_area_mm2)


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def load_heat_source_footprint(
    coords_path: Path,
    params_path: Path,
    *,
    internal_circuit_power_mw: float,
) -> HeatSourceFootprint:
    params = utils.load_params(str(params_path))
    params = copy.deepcopy(params)
    params.setdefault("bioheat", {})["internal_circuit_power_mw"] = float(internal_circuit_power_mw)

    x_mm, y_mm = utils.load_coordinates_from_yaml(str(coords_path))
    x_mm = np.asarray(x_mm, dtype=np.float64)
    y_mm = np.asarray(y_mm, dtype=np.float64)
    if x_mm.size == 0:
        raise ValueError(f"No electrode coordinates found in {coords_path}")
    if x_mm.shape != y_mm.shape:
        raise ValueError(f"Coordinate arrays in {coords_path} have different lengths.")

    elec_xy_mm = np.column_stack([x_mm, y_mm])
    bioheat = Bioheat2D(params=params, elec_xy_mm=elec_xy_mm, device="cpu")
    mask = bioheat.ic_footprint_mask.detach().cpu().numpy().astype(bool)
    pixel_area_mm2 = float(bioheat.dx * bioheat.dy) * 1e6

    return HeatSourceFootprint(
        coords_path=coords_path,
        x_mm=x_mm,
        y_mm=y_mm,
        mask=mask,
        extent_mm=tuple(float(v) for v in bioheat.extent_mm),
        pixel_area_mm2=pixel_area_mm2,
        internal_circuit_power_density_w_m3=float(bioheat.ic_power_density_W_m3),
    )


def mask_coordinate_limits(footprint: HeatSourceFootprint) -> tuple[tuple[float, float], tuple[float, float]]:
    xmin, xmax, ymin, ymax = footprint.extent_mm
    mask_y, mask_x = np.nonzero(footprint.mask)
    x_values = [footprint.x_mm]
    y_values = [footprint.y_mm]

    if mask_x.size > 0:
        dx_mm = (xmax - xmin) / max(footprint.mask.shape[1] - 1, 1)
        dy_mm = (ymax - ymin) / max(footprint.mask.shape[0] - 1, 1)
        x_values.append(xmin + mask_x.astype(np.float64) * dx_mm)
        y_values.append(ymin + mask_y.astype(np.float64) * dy_mm)

    x_all = np.concatenate(x_values)
    y_all = np.concatenate(y_values)
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    span = max(x_max - x_min, y_max - y_min, 1.0)
    pad = span * 0.08
    return ((x_min - pad, x_max + pad), (y_min - pad, y_max + pad))


def source_rgba(mask: np.ndarray, color: str, alpha: float) -> np.ndarray:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=np.float32)
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(mask, float(alpha), 0.0)
    return rgba


def style_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    ax.tick_params(labelsize=9, width=0.8, colors=AXIS_COLOR)
    ax.grid(True, alpha=0.25, linewidth=0.6, color=GRID_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["left"].set_color(AXIS_COLOR)
    ax.spines["bottom"].set_color(AXIS_COLOR)


def draw_heat_sources(
    footprint: HeatSourceFootprint,
    out_path: Path,
    *,
    dot_size: float,
    internal_circuit_alpha: float,
    show_contour: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
    fig.patch.set_facecolor("white")

    ax.imshow(
        source_rgba(footprint.mask, IC_COLOR, internal_circuit_alpha),
        extent=footprint.extent_mm,
        origin="lower",
        interpolation="nearest",
        zorder=1,
    )

    if show_contour and np.any(footprint.mask):
        xmin, xmax, ymin, ymax = footprint.extent_mm
        x_centers = np.linspace(xmin, xmax, footprint.mask.shape[1])
        y_centers = np.linspace(ymin, ymax, footprint.mask.shape[0])
        ax.contour(
            x_centers,
            y_centers,
            footprint.mask.astype(float),
            levels=[0.5],
            colors=[IC_EDGE_COLOR],
            linewidths=1.2,
            zorder=2,
        )

    ax.scatter(
        footprint.x_mm,
        footprint.y_mm,
        s=float(dot_size),
        c=JOULE_COLOR,
        edgecolors=JOULE_EDGE_COLOR,
        linewidths=0.35,
        zorder=3,
    )

    x_limits, y_limits = mask_coordinate_limits(footprint)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    style_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=JOULE_COLOR,
            markeredgecolor=JOULE_EDGE_COLOR,
            markeredgewidth=0.5,
            markersize=7,
            label="Electrode Joule heating sites",
        ),
        Patch(
            facecolor=mcolors.to_rgba(IC_COLOR, internal_circuit_alpha),
            edgecolor=IC_EDGE_COLOR,
            label="Internal-circuit heat footprint",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Bioheat2D heat-source locations: electrode Joule sites and the internal-circuit footprint."
    )
    parser.add_argument(
        "--coords-yaml",
        "--grid",
        dest="coords_yaml",
        default=str(DEFAULT_COORDS),
        help="Electrode coordinate YAML file.",
    )
    parser.add_argument(
        "--params",
        default=str(DEFAULT_PARAMS),
        help="Parameter YAML containing bioheat settings.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output image path. The extension controls the saved format.",
    )
    parser.add_argument(
        "--internal-circuit-power-mw",
        type=float,
        default=13.0,
        help="Internal-circuit power used only to report model power density.",
    )
    parser.add_argument("--dot-size", type=float, default=8.0, help="Electrode marker size.")
    parser.add_argument(
        "--internal-circuit-alpha",
        type=float,
        default=0.32,
        help="Opacity of the blue internal-circuit footprint.",
    )
    parser.add_argument(
        "--no-contour",
        action="store_true",
        help="Do not draw an outline around the internal-circuit footprint.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coords_path = resolve_repo_path(args.coords_yaml)
    params_path = resolve_repo_path(args.params)
    out_path = resolve_repo_path(args.out)

    if not coords_path.exists():
        raise FileNotFoundError(f"Coordinate YAML not found: {coords_path}")
    if not params_path.exists():
        raise FileNotFoundError(f"Params YAML not found: {params_path}")
    if not 0.0 < float(args.internal_circuit_alpha) <= 1.0:
        raise ValueError("--internal-circuit-alpha must be in the range (0, 1].")

    footprint = load_heat_source_footprint(
        coords_path,
        params_path,
        internal_circuit_power_mw=float(args.internal_circuit_power_mw),
    )
    draw_heat_sources(
        footprint,
        out_path,
        dot_size=float(args.dot_size),
        internal_circuit_alpha=float(args.internal_circuit_alpha),
        show_contour=not bool(args.no_contour),
    )

    print(f"Saved heat-source visual to: {out_path}")
    print(f"Coordinates: {coords_path}")
    print(f"Electrode Joule heating sites: {footprint.electrode_count}")
    print(f"Internal-circuit footprint area: {footprint.internal_circuit_area_mm2:.3f} mm^2")
    print(
        "Internal-circuit power density: "
        f"{footprint.internal_circuit_power_density_w_m3:.6g} W/m^3 "
        f"for {float(args.internal_circuit_power_mw):g} mW"
    )

    if args.show:
        image = plt.imread(out_path)
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        ax.imshow(image)
        ax.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
