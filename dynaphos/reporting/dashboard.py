"""
Create a compact overview plot for one DynaPhos experiment.

The input can be either a run directory containing metrics.npz or the
NPZ file itself. The figure is meant as a quick preflight check before running
or reviewing a larger experiment matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from dynaphos.experiment.manifest import load_manifest as load_canonical_manifest


ACADEMIC_BLUE = "#1F4E79"
ACADEMIC_RED = "#8B1E3F"
ACADEMIC_ORANGE = "#C05621"
ACADEMIC_GRAY = "#6B7280"
GRID_COLOR = "#D7DBE0"
TIME_AXIS_LABEL = "Time (min)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a one-page dashboard for one canonical DynaPhos run."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Run directory, metrics.npz file, or a directory containing exactly one run.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output figure path. Defaults to <case_dir>/single_case_overview.png.",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=("png", "pdf", "svg"),
        help="Override output format. If omitted, the output file suffix is used.",
    )
    parser.add_argument("--show", action="store_true", help="Open the figure after writing it.")
    return parser.parse_args()


def resolve_repo_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def resolve_npz_path(input_path: str | Path) -> Path:
    path = resolve_repo_path(input_path)
    if path.is_file():
        if path.name != "metrics.npz":
            raise ValueError(f"Expected metrics.npz, got: {path}")
        return path

    direct = path / "metrics.npz"
    if direct.exists():
        return direct

    matches = sorted(path.rglob("metrics.npz")) if path.exists() else []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Input directory contains {len(matches)} metrics.npz files. "
            "Point --input at one case directory or one NPZ file."
        )
    raise FileNotFoundError(f"No metrics.npz found for: {path}")


def load_npz_dict(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_manifest(npz_path: Path) -> dict[str, Any]:
    return load_canonical_manifest(npz_path.parent)


@dataclass(frozen=True)
class HeatmapSnapshot:
    heatmap: np.ndarray
    grid_name: str | None
    index: int


def data_keys(data: Mapping[str, Any] | Any) -> list[str]:
    if hasattr(data, "files"):
        return list(data.files)
    if hasattr(data, "keys"):
        return list(data.keys())
    return []


def has_data_key(data: Mapping[str, Any] | Any, key: str) -> bool:
    if hasattr(data, "files"):
        return key in data.files
    return key in data


def get_array(data: Mapping[str, Any] | Any, key: str) -> np.ndarray | None:
    if not has_data_key(data, key):
        return None
    return np.asarray(data[key])


def as_float(value: Any, default: float = float("nan")) -> float:
    if value is None:
        return default
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return default
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return default


def display_preprocessing_label(value: object) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"dog", "difference_of_gaussians", "difference-of-gaussians"}:
        return "DoG"
    if normalized in {"gt", "groundtruth", "sanpo_groundtruth", "sanpo_gt"}:
        return "Hand Segmented"
    return str(value)


def add_video_end_line(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    video_end_s = as_float(data.get("cooldown_start_s"), float("nan"))
    if not np.isfinite(video_end_s):
        for key in ("electrode_time_s", "time_s"):
            values = series_1d(data, key)
            if values is not None:
                finite = values[np.isfinite(values)]
                if finite.size:
                    video_end_s = float(finite[-1])
                    break
    if np.isfinite(video_end_s):
        ax.axvline(
            video_end_s / 60.0,
            color="#4B5563",
            linestyle="--",
            linewidth=1.1,
            alpha=0.85,
            label="video end",
        )


def metric_float(
    data: dict[str, np.ndarray],
    manifest: dict[str, Any],
    key: str,
    default: float = float("nan"),
) -> float:
    if key in manifest:
        return as_float(manifest.get(key), default)
    if key in data:
        return as_float(data[key], default)
    return default


def series_1d(data: dict[str, np.ndarray], key: str) -> np.ndarray | None:
    arr = get_array(data, key)
    if arr is None:
        return None
    try:
        return np.asarray(arr, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None


def time_axis(
    data: dict[str, np.ndarray],
    length: int,
    *,
    prefer_electrode: bool = False,
) -> tuple[np.ndarray, str]:
    candidates = (
        ("electrode_time_s", "time_s")
        if prefer_electrode
        else ("time_s", "thermal_time_s", "device_power_time_s", "electrode_time_s")
    )
    for key in candidates:
        values = series_1d(data, key)
        if values is not None and values.size == length:
            return values / 60.0, TIME_AXIS_LABEL
    return np.arange(length, dtype=np.float64) / 60.0, TIME_AXIS_LABEL


def safe_nanmax(values: Any, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return default
    if arr.size == 0:
        return default
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.max(finite))


def safe_nanmean(values: Any, default: float = float("nan")) -> float:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return default
    if arr.size == 0:
        return default
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.mean(finite))


def sanitize_data_key_part(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return safe.strip("._") or "item"


def string_scalar(value: Any) -> str | None:
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    raw = arr.reshape(-1)[0]
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    return text if text else None


def heatmap_stack_candidates(
    data: Mapping[str, Any] | Any,
    grid_name: str | None = None,
) -> list[tuple[str | None, str]]:
    candidates: list[tuple[str | None, str]] = []
    seen: set[str] = set()

    def add_candidate(name: str | None, key: str) -> None:
        if key in seen or not has_data_key(data, key):
            return
        seen.add(key)
        candidates.append((name, key))

    if grid_name:
        add_candidate(grid_name, f"dT_heatmaps_{sanitize_data_key_part(grid_name)}")
        return candidates

    reference_grid = string_scalar(get_array(data, "dT_final_reference_grid"))
    if reference_grid:
        add_candidate(reference_grid, f"dT_heatmaps_{sanitize_data_key_part(reference_grid)}")

    grid_names = get_array(data, "heatmap_grid_names")
    if grid_names is not None:
        for raw_name in grid_names.reshape(-1):
            name = string_scalar(raw_name)
            if name:
                add_candidate(name, f"dT_heatmaps_{sanitize_data_key_part(name)}")

    for key in data_keys(data):
        if key.startswith("dT_heatmaps_"):
            add_candidate(key.removeprefix("dT_heatmaps_"), key)
    return candidates


def middle_snapshot_index(data: Mapping[str, Any] | Any, length: int) -> int:
    if length <= 1:
        return 0
    for key in ("heatmap_actual_fractions", "heatmap_target_fractions"):
        fractions = series_1d(data, key)
        if fractions is None or fractions.size != length:
            continue
        finite = np.isfinite(fractions)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            nearest = np.argmin(np.abs(fractions[finite] - 0.5))
            return int(finite_indices[int(nearest)])
    return int((length - 1) // 2)


def middle_heatmap_snapshot(
    data: Mapping[str, Any] | Any,
    *,
    grid_name: str | None = None,
) -> HeatmapSnapshot | None:
    best: HeatmapSnapshot | None = None
    best_peak = -np.inf
    for candidate_grid, key in heatmap_stack_candidates(data, grid_name):
        try:
            stack = np.asarray(data[key], dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if stack.ndim != 3 or stack.shape[0] == 0:
            continue
        index = middle_snapshot_index(data, int(stack.shape[0]))
        heatmap = np.asarray(stack[index], dtype=np.float64)
        if heatmap.ndim != 2 or heatmap.size == 0:
            continue
        snapshot = HeatmapSnapshot(heatmap=heatmap, grid_name=candidate_grid, index=index)
        if grid_name is not None:
            return snapshot
        peak = safe_nanmax(heatmap, -np.inf)
        if best is None or peak > best_peak:
            best = snapshot
            best_peak = peak
    return best


def heatmap_snapshot_detail(data: Mapping[str, Any] | Any, snapshot: HeatmapSnapshot) -> str:
    parts: list[str] = []
    fractions = series_1d(data, "heatmap_actual_fractions")
    if fractions is None or fractions.size <= snapshot.index:
        fractions = series_1d(data, "heatmap_target_fractions")
    if fractions is not None and fractions.size > snapshot.index and np.isfinite(fractions[snapshot.index]):
        parts.append(f"{100.0 * float(fractions[snapshot.index]):.0f}%")

    times_s = series_1d(data, "heatmap_times_s")
    if times_s is not None and times_s.size > snapshot.index and np.isfinite(times_s[snapshot.index]):
        parts.append(f"{float(times_s[snapshot.index]) / 60.0:.1f} min")

    if snapshot.grid_name:
        parts.append(str(snapshot.grid_name))
    return " | ".join(parts)


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=9, width=0.8)
    ax.grid(axis=grid_axis, alpha=0.28, linewidth=0.6, color=GRID_COLOR)


def add_no_data(ax: plt.Axes, message: str = "No data") -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=ACADEMIC_GRAY,
        fontsize=10,
    )
    style_axes(ax)


def plot_line(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    values: np.ndarray,
    *,
    label: str,
    color: str,
    prefer_electrode_time: bool = False,
    linewidth: float = 1.8,
) -> tuple[np.ndarray, str]:
    y = np.asarray(values, dtype=np.float64).reshape(-1)
    x, xlabel = time_axis(data, y.size, prefer_electrode=prefer_electrode_time)
    ax.plot(x, y, label=label, color=color, linewidth=linewidth)
    return x, xlabel


def current_summaries(data: dict[str, np.ndarray]) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    current = get_array(data, "amplitude_per_electrode_uA")
    if current is None:
        current = get_array(data, "current_amplitude_per_electrode_uA")
    if current is None:
        return None, None, None

    arr = np.asarray(current, dtype=np.float64)
    if arr.size == 0:
        return None, None, None
    if arr.ndim == 1:
        y = arr.reshape(-1)
        active_count = np.asarray((np.isfinite(y) & (y > 0.0)), dtype=np.float64)
        return y, np.where(active_count > 0, y, 0.0), active_count

    rows = arr.reshape(arr.shape[0], -1)
    finite = np.isfinite(rows)
    positive = finite & (rows > 0.0)
    max_current = np.max(np.where(finite, rows, -np.inf), axis=1)
    max_current = np.where(np.isfinite(max_current), max_current, 0.0)
    positive_sum = np.sum(np.where(positive, rows, 0.0), axis=1)
    positive_count = np.sum(positive, axis=1).astype(np.float64)
    mean_active = np.divide(
        positive_sum,
        positive_count,
        out=np.zeros_like(positive_sum, dtype=np.float64),
        where=positive_count > 0,
    )
    return max_current, mean_active, positive_count


def format_number(value: float, suffix: str = "", *, precision: int = 3) -> str:
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{precision}g}{suffix}"


def manifest_text(manifest: dict[str, Any], data: dict[str, np.ndarray], npz_path: Path) -> tuple[str, str]:
    block = str(manifest.get("block", npz_path.parent.parent.name if npz_path.parent.parent else "case"))
    run_id = str(manifest.get("run_id", npz_path.parent.name))
    coords = str(manifest.get("coords_yaml", ""))
    grid = Path(coords).stem if coords else "unknown_grid"
    source = display_preprocessing_label(
        manifest.get(
            "source_input_label",
            manifest.get("preprocessing_method", manifest.get("video", "unknown_input")),
        )
    )
    raster = str(manifest.get("raster_mode", "none"))
    raster_normalized = str(manifest.get("raster_mode_normalized", raster))
    if raster_normalized != raster:
        raster = f"{raster} -> {raster_normalized}"

    amp = metric_float(data, manifest, "amplitude_uA")
    threshold = metric_float(data, manifest, "threshold_uA")
    if not np.isfinite(threshold):
        threshold = metric_float(data, manifest, "appearance_threshold_uA")
    freq = metric_float(data, manifest, "frequency_hz")
    pulse_width = metric_float(data, manifest, "pulse_width_us")
    ic_power = metric_float(data, manifest, "internal_circuit_power_mw", 0.0)

    title = f"{block} / {run_id}"
    subtitle = (
        f"{source} | {grid} | amp {format_number(amp, ' µA')} | "
        f"threshold {format_number(threshold, ' µA')} | "
        f"{format_number(freq, ' Hz')} / {format_number(pulse_width, ' µs')} | "
        f"raster {raster} | IC {format_number(ic_power, ' mW')}"
    )
    return title, subtitle


def plot_current_panel(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    _manifest: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    max_current, mean_active, active_count_from_current = current_summaries(data)
    if max_current is None or mean_active is None:
        add_no_data(ax, "No current trace")
        ax.set_title("Delivered Current", fontsize=11)
        return max_current, mean_active, active_count_from_current

    _, xlabel = plot_line(
        ax,
        data,
        max_current,
        label="max",
        color=ACADEMIC_BLUE,
        prefer_electrode_time=True,
    )
    ax.set_title("Delivered Current", fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Current (µA)")
    style_axes(ax)
    return max_current, mean_active, active_count_from_current


def plot_count_panel(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    active_count_from_current: np.ndarray | None,
) -> None:
    active_count = series_1d(data, "active_count")
    stimulated_count = series_1d(data, "stimulated_electrode_count")
    if active_count is None and active_count_from_current is not None:
        active_count = active_count_from_current

    plotted = False
    if active_count is not None:
        _, xlabel = plot_line(ax, data, active_count, label="active", color=ACADEMIC_BLUE)
        plotted = True
    else:
        xlabel = TIME_AXIS_LABEL
    if stimulated_count is not None:
        _, xlabel = plot_line(ax, data, stimulated_count, label="stimulated", color=ACADEMIC_ORANGE)
        plotted = True

    ax.set_title("Active Electrode Counts", fontsize=11)
    if plotted:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Electrodes")
        style_axes(ax)
    else:
        add_no_data(ax, "No electrode counts")


def plot_charge_panel(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    charge_rate_total = None
    charge_rate = get_array(data, "charge_per_second_per_electrode_nC_s")
    if charge_rate is not None:
        arr = np.asarray(charge_rate, dtype=np.float64)
        if arr.ndim == 2:
            charge_rate_total = np.sum(arr, axis=1)
        elif arr.size:
            charge_rate_total = arr.reshape(-1)
    if charge_rate_total is None:
        charge_rate_total = series_1d(data, "charge_per_second_total_nC_s")

    plotted = False
    xlabel = TIME_AXIS_LABEL
    if charge_rate_total is not None:
        rate_uC_s = np.asarray(charge_rate_total, dtype=np.float64) / 1e3
        x, _ = time_axis(data, rate_uC_s.size)
        ax.plot(x, rate_uC_s, label="total charge rate", color=ACADEMIC_RED, linewidth=1.6)
        plotted = True

    ax.set_title("Total Charge Rate", fontsize=11)
    if plotted:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Total charge rate (µC/s)")
        ax.legend(
            fontsize=8,
            frameon=False,
            loc="upper right",
        )
        style_axes(ax)
    else:
        add_no_data(ax, "No charge metrics")


def plot_temperature_panel(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    max_dt = series_1d(data, "max_dT")
    mean_dt = series_1d(data, "mean_dT")
    plotted = False
    xlabel = TIME_AXIS_LABEL
    if max_dt is not None:
        _, xlabel = plot_line(ax, data, max_dt, label="max ΔT", color=ACADEMIC_RED)
        plotted = True
    if mean_dt is not None:
        _, xlabel = plot_line(ax, data, mean_dt, label="mean ΔT", color=ACADEMIC_BLUE)
        plotted = True

    ax.set_title("Temperature Rise", fontsize=11)
    if plotted:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("ΔT (°C)")
        ax.legend(fontsize=8, frameon=False, loc="best")
        style_axes(ax)
    else:
        add_no_data(ax, "No thermal trace")


def plot_hotspot_area_panel(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    series = [
        ("area_gt1_mm2", ">1 °C", ACADEMIC_BLUE),
        ("area_gt2_mm2", ">2 °C", ACADEMIC_ORANGE),
        ("area_gt3_mm2", ">3 °C", ACADEMIC_RED),
    ]
    plotted = False
    xlabel = TIME_AXIS_LABEL
    for key, label, color in series:
        values = series_1d(data, key)
        if values is None:
            continue
        _, xlabel = plot_line(ax, data, values, label=label, color=color)
        plotted = True

    ax.set_title("Hotspot Area", fontsize=11)
    if plotted:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Area (mm²)")
        ax.legend(fontsize=8, frameon=False, loc="best")
        style_axes(ax)
    else:
        add_no_data(ax, "No hotspot areas")


def plot_heatmap_panel(ax: plt.Axes, fig: plt.Figure, data: dict[str, np.ndarray]) -> None:
    snapshot = middle_heatmap_snapshot(data)
    if snapshot is None:
        add_no_data(ax, "No middle heatmap")
        ax.set_title("Mid-Simulation ΔT Map", fontsize=11)
        return

    heatmap = np.asarray(snapshot.heatmap, dtype=np.float64)
    if heatmap.ndim != 2 or heatmap.size == 0:
        add_no_data(ax, "No middle heatmap")
        ax.set_title("Mid-Simulation ΔT Map", fontsize=11)
        return

    extent = get_array(data, "extent_mm")
    extent_list = None
    if extent is not None:
        extent_values = np.asarray(extent, dtype=np.float64).reshape(-1)
        if extent_values.size >= 4 and np.all(np.isfinite(extent_values[:4])):
            extent_list = [
                float(extent_values[0]),
                float(extent_values[1]),
                float(extent_values[2]),
                float(extent_values[3]),
            ]

    peak_dt = safe_nanmax(heatmap, 0.0)
    image = ax.imshow(
        heatmap,
        origin="lower",
        extent=extent_list,
        aspect="equal",
        cmap="inferno",
        vmin=0.0,
        vmax=peak_dt if peak_dt > 0.0 else 1.0,
    )
    detail = heatmap_snapshot_detail(data, snapshot)
    title = "Mid-Simulation ΔT Map"
    if detail:
        title = f"{title}\n{detail}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (mm)" if extent_list else "x pixel")
    ax.set_ylabel("y (mm)" if extent_list else "y pixel")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ΔT (°C)")
    cbar.ax.tick_params(labelsize=8)


def summary_text(
    data: dict[str, np.ndarray],
    max_current: np.ndarray | None,
    active_count: np.ndarray | None,
) -> str:
    peak_current = safe_nanmax(get_array(data, "amplitude_per_electrode_uA"))
    if not np.isfinite(peak_current):
        peak_current = metric_or_array_peak(data, "peak_current_amplitude_uA_exact", max_current)
    peak_phase_charge = safe_nanmax(get_array(data, "charge_per_phase_per_electrode_nC"))
    if not np.isfinite(peak_phase_charge):
        peak_phase_charge = metric_or_array_peak(data, "peak_charge_per_phase_nC_exact", "charge_per_phase_mean_nC")
    peak_shannon = safe_nanmax(get_array(data, "shannon_k_per_electrode"))
    if not np.isfinite(peak_shannon):
        peak_shannon = metric_or_array_peak(data, "peak_shannon_k_exact", "shannon_k_mean")
    peak_dt = metric_or_array_peak(data, "peak_max_dT_C", "max_dT")
    peak_area = metric_or_array_peak(data, "peak_area_gt1_mm2", "area_gt1_mm2")
    peak_active = safe_nanmax(active_count) if active_count is not None else safe_nanmax(series_1d(data, "active_count"))

    return (
        f"peak current {format_number(peak_current, ' µA')} | "
        f"peak phase charge {format_number(peak_phase_charge, ' nC')} | "
        f"peak Shannon k {format_number(peak_shannon)} | "
        f"peak max ΔT {format_number(peak_dt, ' °C')} | "
        f"peak area >1 °C {format_number(peak_area, ' mm²')} | "
        f"peak active electrodes {format_number(peak_active, precision=0)}"
    )


def metric_or_array_peak(
    data: dict[str, np.ndarray],
    scalar_key: str,
    fallback: str | np.ndarray | None,
) -> float:
    if scalar_key in data:
        value = as_float(data[scalar_key])
        if np.isfinite(value):
            return value
    if isinstance(fallback, str):
        return safe_nanmax(series_1d(data, fallback))
    if fallback is None:
        return float("nan")
    return safe_nanmax(fallback)


def build_figure(npz_path: Path, data: dict[str, np.ndarray], manifest: dict[str, Any]) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.5))
    title, subtitle = manifest_text(manifest, data, npz_path)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.947, subtitle, ha="center", va="top", fontsize=9.5, color="#374151")

    max_current, _mean_active, active_count_from_current = plot_current_panel(axes[0, 0], data, manifest)
    plot_count_panel(axes[0, 1], data, active_count_from_current)
    plot_charge_panel(axes[0, 2], data)
    plot_temperature_panel(axes[1, 0], data)
    plot_hotspot_area_panel(axes[1, 1], data)
    plot_heatmap_panel(axes[1, 2], fig, data)
    for ax in (*axes[0, :], *axes[1, :2]):
        add_video_end_line(ax, data)

    active_count = series_1d(data, "active_count")
    if active_count is None:
        active_count = active_count_from_current
    fig.text(
        0.5,
        0.018,
        summary_text(data, max_current, active_count),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=[0.02, 0.05, 0.98, 0.92])
    return fig


def resolve_output_path(npz_path: Path, out_arg: str | None, fmt: str | None) -> Path:
    if out_arg is None:
        suffix = f".{fmt}" if fmt else ".png"
        return npz_path.parent / f"single_case_overview{suffix}"
    out_path = resolve_repo_path(out_arg)
    if not out_path.suffix:
        out_path = out_path.with_suffix(f".{fmt or 'png'}")
    return out_path


def main() -> None:
    args = parse_args()
    npz_path = resolve_npz_path(args.input)
    data = load_npz_dict(npz_path)
    manifest = load_manifest(npz_path)
    out_path = resolve_output_path(npz_path, args.out, args.format)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = build_figure(npz_path, data, manifest)
    save_kwargs: dict[str, Any] = {"dpi": 220, "bbox_inches": "tight"}
    if args.format:
        save_kwargs["format"] = args.format
    fig.savefig(out_path, **save_kwargs)
    print(f"Wrote {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
