"""
Plot the Shannon K relationship between charge per phase and charge density.

The Shannon model used in this repository is:

    K = log10(Q_µC) + log10(D_µC_cm²)

where Q is charge per phase in microcoulombs and D is charge density in
microcoulombs per square centimeter. The plot x-axis uses nC/phase because
config/safety.yaml stores charge-per-phase limits in nC.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAFETY_YAML = PROJECT_ROOT / "config" / "safety.yaml"
DEFAULT_OUT = PROJECT_ROOT / "results" / "safety" / "visuals" / "shannon_k_boundary.png"

SAFE_FILL = "#CFEBD6"
UNSAFE_FILL = "#F4C7C3"
SAFE_EDGE = "#1F7A3A"
UNSAFE_EDGE = "#B91C1C"
BOUNDARY_COLOR = "#111827"
LIMIT_LINE_COLOR = "#6B7280"


@dataclass(frozen=True)
class SafetyDefaults:
    shannon_k_limit: float
    charge_per_phase_max_nC: float
    charge_density_max_uC_cm2: float


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def optional_float(value: object, fallback: float | None = None) -> float | None:
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def read_safety_defaults(path: Path) -> SafetyDefaults:
    cfg = load_yaml(path)
    thresholds = cfg.get("thresholds", {}) or {}
    guidelines = cfg.get("stimulation_safety_guidelines", {}) or {}
    charge_per_phase = guidelines.get("charge_per_phase", {}) or {}
    charge_density = guidelines.get("charge_density", {}) or {}
    shannon_k = guidelines.get("shannon_k", {}) or {}

    return SafetyDefaults(
        shannon_k_limit=float(
            optional_float(
                thresholds.get("shannon_k_limit"),
                optional_float(shannon_k.get("value"), 1.85),
            )
        ),
        charge_per_phase_max_nC=float(
            optional_float(
                thresholds.get("charge_per_phase_max_nc"),
                optional_float(charge_per_phase.get("value"), 100.0),
            )
        ),
        charge_density_max_uC_cm2=float(
            optional_float(
                thresholds.get("charge_density_max_uc_per_cm2"),
                optional_float(charge_density.get("value"), 4000.0),
            )
        ),
    )


def shannon_k(charge_per_phase_nC: np.ndarray, charge_density_uC_cm2: np.ndarray) -> np.ndarray:
    charge_per_phase_uC = np.asarray(charge_per_phase_nC, dtype=np.float64) / 1e3
    charge_density_uC_cm2 = np.asarray(charge_density_uC_cm2, dtype=np.float64)
    result = np.full_like(charge_density_uC_cm2, -np.inf, dtype=np.float64)
    valid = (charge_per_phase_uC > 0.0) & (charge_density_uC_cm2 > 0.0)
    result[valid] = np.log10(charge_per_phase_uC[valid]) + np.log10(charge_density_uC_cm2[valid])
    return result


def shannon_boundary_density(charge_per_phase_nC: np.ndarray, k_limit: float) -> np.ndarray:
    charge_per_phase_uC = np.asarray(charge_per_phase_nC, dtype=np.float64) / 1e3
    density = np.full_like(charge_per_phase_uC, np.nan, dtype=np.float64)
    valid = charge_per_phase_uC > 0.0
    density[valid] = (10.0 ** float(k_limit)) / charge_per_phase_uC[valid]
    return density


def validate_positive_range(name: str, lower: float, upper: float) -> None:
    if lower <= 0.0 or upper <= 0.0:
        raise ValueError(f"{name} bounds must be positive for the logarithmic Shannon plot.")
    if lower >= upper:
        raise ValueError(f"{name} minimum must be smaller than maximum.")


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(labelsize=9, width=0.8)
    ax.grid(True, which="major", alpha=0.28, linewidth=0.7, color="#AEB7C2")
    ax.grid(True, which="minor", alpha=0.14, linewidth=0.45, color="#AEB7C2")


def plot_shannon_boundary(
    out_path: Path,
    *,
    q_min_nC: float,
    q_max_nC: float,
    d_min_uC_cm2: float,
    d_max_uC_cm2: float,
    k_limit: float,
    points: int,
    configured_q_max_nC: float,
    configured_d_max_uC_cm2: float,
    title: str,
) -> None:
    validate_positive_range("Charge per phase", q_min_nC, q_max_nC)
    validate_positive_range("Charge density", d_min_uC_cm2, d_max_uC_cm2)
    if points < 50:
        raise ValueError("--points must be at least 50.")

    charge_per_phase = np.logspace(np.log10(q_min_nC), np.log10(q_max_nC), int(points))
    charge_density = np.logspace(np.log10(d_min_uC_cm2), np.log10(d_max_uC_cm2), int(points))
    q_grid, d_grid = np.meshgrid(charge_per_phase, charge_density)
    k_grid = shannon_k(q_grid, d_grid)
    region = np.where(k_grid <= k_limit, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    fig.patch.set_facecolor("white")
    ax.contourf(
        q_grid,
        d_grid,
        region,
        levels=[-0.5, 0.5, 1.5],
        colors=[SAFE_FILL, UNSAFE_FILL],
        alpha=0.96,
    )

    boundary = shannon_boundary_density(charge_per_phase, k_limit)
    ax.plot(
        charge_per_phase,
        boundary,
        color=BOUNDARY_COLOR,
        linewidth=2.2,
        label="_nolegend_",
        zorder=5,
    )

    if q_min_nC < configured_q_max_nC < q_max_nC:
        ax.axvline(
            configured_q_max_nC,
            color=LIMIT_LINE_COLOR,
            linewidth=1.2,
            linestyle=":",
            label=f"Configured Q max = {configured_q_max_nC:g} nC",
        )
    if d_min_uC_cm2 < configured_d_max_uC_cm2 < d_max_uC_cm2:
        ax.axhline(
            configured_d_max_uC_cm2,
            color=LIMIT_LINE_COLOR,
            linewidth=1.2,
            linestyle="-.",
            label=f"Configured D max = {configured_d_max_uC_cm2:g} µC/cm²",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(q_min_nC, q_max_nC)
    ax.set_ylim(d_min_uC_cm2, d_max_uC_cm2)
    ax.set_xlabel("Charge per phase Q (nC/phase)", fontsize=11)
    ax.set_ylabel("Charge density D (µC/cm²/phase)", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    style_axes(ax)

    handles = [
        Patch(facecolor=SAFE_FILL, edgecolor=SAFE_EDGE, label=f"Safe: K <= {k_limit:g}"),
        Patch(facecolor=UNSAFE_FILL, edgecolor=UNSAFE_EDGE, label=f"Unsafe: K > {k_limit:g}"),
        Line2D([0], [0], color=BOUNDARY_COLOR, linewidth=2.2, label=f"K = {k_limit:g} boundary"),
    ]
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    for handle, label in zip(existing_handles, existing_labels):
        if label and not label.startswith("_") and label not in {item.get_label() for item in handles}:
            handles.append(handle)
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left")

    ax.text(
        0.99,
        0.02,
        "K = log10(Q[µC]) + log10(D[µC/cm²])",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#374151",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 4},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot safe and unsafe charge-per-phase/charge-density regions under the Shannon K model."
    )
    parser.add_argument(
        "--safety-yaml",
        default=str(DEFAULT_SAFETY_YAML),
        help="Safety YAML containing Shannon K and charge limits.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Output image path. The extension controls the saved format.",
    )
    parser.add_argument("--charge-per-phase-min-nc", type=float, default=10.0)
    parser.add_argument("--charge-per-phase-max-nc", type=float, default=100.0)
    parser.add_argument("--charge-density-min-uc-cm2", type=float, default=1.0)
    parser.add_argument("--charge-density-max-uc-cm2", type=float, default=None)
    parser.add_argument("--k-limit", type=float, default=1.5, help="Shannon K limit used for the boundary.")
    parser.add_argument("--points", type=int, default=600, help="Grid resolution for the filled regions.")
    parser.add_argument(
        "--title",
        default="Shannon K Safety Boundary",
        help="Figure title.",
    )
    parser.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    safety_yaml = resolve_repo_path(args.safety_yaml)
    if not safety_yaml.exists():
        raise FileNotFoundError(f"Safety YAML not found: {safety_yaml}")

    defaults = read_safety_defaults(safety_yaml)
    q_max_nC = float(args.charge_per_phase_max_nc)
    d_max_uC_cm2 = (
        float(args.charge_density_max_uc_cm2)
        if args.charge_density_max_uc_cm2 is not None
        else defaults.charge_density_max_uC_cm2
    )
    k_limit = float(args.k_limit)
    out_path = resolve_repo_path(args.out)

    plot_shannon_boundary(
        out_path,
        q_min_nC=float(args.charge_per_phase_min_nc),
        q_max_nC=q_max_nC,
        d_min_uC_cm2=float(args.charge_density_min_uc_cm2),
        d_max_uC_cm2=d_max_uC_cm2,
        k_limit=k_limit,
        points=int(args.points),
        configured_q_max_nC=defaults.charge_per_phase_max_nC,
        configured_d_max_uC_cm2=defaults.charge_density_max_uC_cm2,
        title=str(args.title),
    )
    print(f"Saved Shannon K boundary plot to: {out_path}")
    print(f"Safe region: K <= {k_limit:g}; unsafe region: K > {k_limit:g}")

    if args.show:
        image = plt.imread(out_path)
        fig, ax = plt.subplots(figsize=(8.4, 6.2))
        ax.imshow(image)
        ax.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
