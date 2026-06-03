"""
Visualize the basic stimulation sweep outputs.

The script reads safety_metrics.npz files created by
tools/safety/run_basic_stimulation_sweep.py and writes grouped figures for the
amplitude, pulse-width, and frequency sweeps.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.safety.common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAFETY,
    DEFAULT_VISUALS_ROOT,
    STANDARD_AMPLITUDE_UA,
    STANDARD_FREQUENCY_HZ,
    STANDARD_PULSE_WIDTH_US,
    load_safety_limits,
    resolve_repo_path,
)


BASIC_STIMULATION_BLOCK = "basic_stimulation_v3"
BASIC_STIMULATION_BLOCK_PREFIX = "basic_stimulation"
DEFAULT_INPUT_ROOT = Path(DEFAULT_OUTPUT_ROOT) / BASIC_STIMULATION_BLOCK
DEFAULT_OUTPUT_DIR = Path(DEFAULT_VISUALS_ROOT) / f"{BASIC_STIMULATION_BLOCK}_summary"
SHANNON_K_LIMIT_FALLBACK = 1.85
CPS_NC_PER_UC = 1e3
CPS_NC_PER_MC = 1e6
PARAMETER_ORDER = ("amplitude", "pulse_width", "frequency", "worst_case")
INCLUDED_GRIDS_UM = (400, 800, 1200)
GRID_COLORS = {
    400: "#B91C1C",
    800: "#1F4E79",
    1200: "#2E8B57",
}
FALLBACK_COLORS = ["#1F4E79", "#2E8B57", "#8B1E3F", "#C05621", "#5B5EA6", "#6B7280"]
PLOT_CHOICES = (
    "summary",
    "shannon_k",
    "mean_charge",
    "max_charge",
    "total_mean_charge",
    "total_max_charge",
    "final_charge",
    "active_electrodes",
    "amplitudes",
    "temperature",
    "heatmaps",
)
PLOT_ALIASES = {
    "all": set(PLOT_CHOICES),
    "charge": {"mean_charge", "max_charge", "total_mean_charge", "total_max_charge"},
    "mean_charge": {"mean_charge"},
    "max_charge": {"max_charge"},
    "total_charge": {"total_mean_charge", "total_max_charge"},
    "total_charge_per_second": {"total_mean_charge", "total_max_charge"},
    "charge_per_second": {"mean_charge", "max_charge", "total_mean_charge", "total_max_charge"},
    "final_total_charge": {"final_charge"},
    "temperature": {"temperature"},
}


@dataclass(frozen=True)
class RunRecord:
    npz_path: Path
    run_id: str
    grid_um: int
    sweep: str
    sweep_value: float
    amplitude_uA: float
    frequency_hz: float
    pulse_width_us: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary plots for the basic stimulation safety sweep."
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Directory containing basic_stimulation run folders.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where summary plots will be written.",
    )
    parser.add_argument(
        "--safety-yaml",
        default=DEFAULT_SAFETY,
        help="Safety YAML used for limits.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=("png", "pdf", "svg"),
        help="Output image format.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing figures. Existing figures are skipped by default.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=["all"],
        metavar="RESULT",
        help=(
            "Only generate selected result groups. Values can be separated by spaces or commas. "
            "Common values: mean_charge, max_charge, total_charge, charge_per_second, "
            "final_charge, shannon_k, active_electrodes, amplitudes, temperature, heatmaps, summary, all."
        ),
    )
    return parser.parse_args()


def selected_results(raw_values: Iterable[str]) -> set[str]:
    selected: set[str] = set()
    valid_values = set(PLOT_CHOICES) | set(PLOT_ALIASES)
    for raw_value in raw_values:
        for value in str(raw_value).split(","):
            value = value.strip().lower().replace("-", "_")
            if not value:
                continue
            if value not in valid_values:
                allowed = ", ".join(sorted(valid_values))
                raise ValueError(f"Unsupported --only result '{value}'. Supported values: {allowed}")
            selected.update(PLOT_ALIASES.get(value, {value}))
    return selected or set(PLOT_CHOICES)


def safe_nanmean(values: object, default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return default
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.mean(finite))


def safe_nanmax(values: object, default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return default
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.max(finite))


def safe_last(values: object, default: float = float("nan")) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return default
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(finite[-1])


def finite_flat(values: object, *, positive_only: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if positive_only:
        arr = arr[arr > 0.0]
    return arr


def get_raw_metric_array(record: RunRecord, *keys: str) -> np.ndarray | None:
    with np.load(record.npz_path, allow_pickle=True) as data:
        for key in keys:
            if key in data.files:
                return np.asarray(data[key])
    return None


def metric_scalar(record: RunRecord, key: str, default: float = float("nan")) -> float:
    value = get_raw_metric_array(record, key)
    if value is None:
        return default
    return scalar_float(value, default=default)


def raw_time_series(record: RunRecord) -> np.ndarray:
    values = get_raw_metric_array(record, "time_s", "electrode_time_s")
    return finite_flat(values if values is not None else [])


def derived_frame_charge_per_electrode_nC(record: RunRecord) -> np.ndarray | None:
    amplitude = get_raw_metric_array(record, "current_amplitude_per_electrode_uA")
    pulse_width = get_raw_metric_array(record, "pulse_width_s")
    frequency = get_raw_metric_array(record, "pulse_frequency_hz")
    if amplitude is None or pulse_width is None or frequency is None:
        return None

    amplitude = np.asarray(amplitude, dtype=np.float64)
    pulse_width = np.asarray(pulse_width, dtype=np.float64).reshape(-1)
    frequency = np.asarray(frequency, dtype=np.float64).reshape(-1)
    if amplitude.ndim != 2 or amplitude.shape[0] == 0 or amplitude.shape[1] == 0:
        return None

    fps = metric_scalar(record, "fps", default=float("nan"))
    if not np.isfinite(fps) or fps <= 0.0:
        times = raw_time_series(record)
        finite_dt = np.diff(np.concatenate([[0.0], times]))
        finite_dt = finite_dt[np.isfinite(finite_dt) & (finite_dt > 0.0)]
        dt = float(np.median(finite_dt)) if finite_dt.size else float("nan")
    else:
        dt = 1.0 / fps
    if not np.isfinite(dt) or dt <= 0.0:
        return None

    relative_stim_duration = metric_scalar(record, "relative_stim_duration", default=1.0)
    if not np.isfinite(relative_stim_duration):
        relative_stim_duration = 1.0
    return (
        2.0
        * amplitude
        * pulse_width.reshape(1, -1)
        * frequency.reshape(1, -1)
        * dt
        * relative_stim_duration
        * 1e3
    )


def derived_charge_per_phase_nC(record: RunRecord) -> np.ndarray | None:
    amplitude = get_raw_metric_array(record, "current_amplitude_per_electrode_uA")
    pulse_width = get_raw_metric_array(record, "pulse_width_s")
    if amplitude is None or pulse_width is None:
        return None
    amplitude = np.asarray(amplitude, dtype=np.float64)
    pulse_width = np.asarray(pulse_width, dtype=np.float64).reshape(-1)
    if amplitude.ndim != 2 or pulse_width.size != amplitude.shape[1]:
        return None
    return amplitude * pulse_width.reshape(1, -1) * 1e3


def derived_charge_density_uc_cm2(record: RunRecord) -> np.ndarray | None:
    charge_per_phase = derived_charge_per_phase_nC(record)
    area_cm2 = metric_scalar(record, "electrode_surface_area_cm2", default=float("nan"))
    if charge_per_phase is None or not np.isfinite(area_cm2) or area_cm2 <= 0.0:
        return None
    return charge_per_phase / 1e3 / area_cm2


def derived_shannon_k(record: RunRecord) -> np.ndarray | None:
    charge_per_phase = derived_charge_per_phase_nC(record)
    charge_density = derived_charge_density_uc_cm2(record)
    if charge_per_phase is None or charge_density is None:
        return None
    result = np.full_like(charge_per_phase, -np.inf, dtype=np.float64)
    charge_per_phase_uC = charge_per_phase / 1e3
    valid = (charge_per_phase_uC > 0.0) & (charge_density > 0.0)
    result[valid] = np.log10(charge_per_phase_uC[valid]) + np.log10(charge_density[valid])
    return result


def rolling_charge_sum(frame_charge: np.ndarray, times: np.ndarray, window_s: float) -> np.ndarray | None:
    frame_charge = np.asarray(frame_charge, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    if frame_charge.ndim != 2 or frame_charge.shape[0] == 0 or times.size == 0:
        return None
    frame_count = min(frame_charge.shape[0], times.size)
    frame_charge = frame_charge[:frame_count]
    times = times[:frame_count]
    if not np.isfinite(window_s) or window_s <= 0.0:
        return None

    result = np.zeros_like(frame_charge, dtype=np.float64)
    entries: list[list[object]] = []
    window_time_s = 0.0
    window_sum = np.zeros(frame_charge.shape[1], dtype=np.float64)
    prev_time = 0.0
    finite_dt = np.diff(np.concatenate([[0.0], times]))
    finite_dt = finite_dt[np.isfinite(finite_dt) & (finite_dt > 0.0)]
    fallback_dt = float(np.median(finite_dt)) if finite_dt.size else 0.0

    for idx, (time_value, charge_row) in enumerate(zip(times, frame_charge)):
        dt_s = float(time_value - prev_time)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            dt_s = fallback_dt
        prev_time = float(time_value)
        entries.append([dt_s, charge_row])
        window_time_s += dt_s
        window_sum = window_sum + charge_row

        while window_time_s > window_s + 1e-12 and entries:
            excess_s = window_time_s - window_s
            head_dt_s = float(entries[0][0])
            head_charge = np.asarray(entries[0][1], dtype=np.float64)
            if head_dt_s <= excess_s + 1e-12:
                entries.pop(0)
                window_time_s -= head_dt_s
                window_sum = window_sum - head_charge
            else:
                fraction = excess_s / head_dt_s
                trim_charge = head_charge * fraction
                entries[0] = [head_dt_s - excess_s, head_charge - trim_charge]
                window_time_s -= excess_s
                window_sum = window_sum - trim_charge
        result[idx] = window_sum
    return result


def derive_metric_array(record: RunRecord, key: str) -> np.ndarray | None:
    if key in ("frame_charge_per_electrode_nC",):
        return derived_frame_charge_per_electrode_nC(record)
    if key in ("charge_per_phase_nC",):
        return derived_charge_per_phase_nC(record)
    if key in ("charge_density_uc_cm2",):
        return derived_charge_density_uc_cm2(record)
    if key in ("shannon_k",):
        return derived_shannon_k(record)

    frame_charge = None
    if key in (
        "protocol_charge_per_electrode_nC",
        "protocol_charge_nC",
        "window_charge_per_electrode_nC",
        "window_charge_nC",
        "charge_per_second_per_electrode_nC_s",
    ):
        frame_charge = derived_frame_charge_per_electrode_nC(record)
        if frame_charge is None:
            return None

    if key in ("protocol_charge_per_electrode_nC", "protocol_charge_nC"):
        return np.cumsum(frame_charge, axis=0)
    if key in ("window_charge_per_electrode_nC", "window_charge_nC"):
        return rolling_charge_sum(
            frame_charge,
            raw_time_series(record),
            metric_scalar(record, "charge_window_s", default=5.0),
        )
    if key == "charge_per_second_per_electrode_nC_s":
        return rolling_charge_sum(frame_charge, raw_time_series(record), 1.0)
    return None


def get_metric_array(record: RunRecord, *keys: str) -> np.ndarray | None:
    with np.load(record.npz_path, allow_pickle=True) as data:
        files = set(data.files)
        for key in keys:
            if key in files:
                return np.asarray(data[key])
    for key in keys:
        derived = derive_metric_array(record, key)
        if derived is not None:
            return np.asarray(derived)
    return None


def load_yaml_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_total_session_charge_limit_mC(safety_yaml: Path) -> float:
    cfg = load_yaml_file(safety_yaml)
    value_c = ((cfg.get("chronic", {}) or {}).get("session_charge_limit_c", None))
    if value_c is None:
        return float("nan")
    return float(value_c) * 1e3


def scalar_float(value: object, default: float = float("nan")) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    try:
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return default


def infer_grid_um(run_id: str, manifest: dict) -> int:
    coords_yaml = Path(str(manifest.get("coords_yaml", ""))).stem
    candidates = [run_id, coords_yaml]
    for candidate in candidates:
        parts = str(candidate).split("_")
        for part in parts:
            if part.endswith("um"):
                number = part[:-2]
                if number.isdigit():
                    return int(number)
    raise ValueError(f"Could not infer grid spacing from run '{run_id}'.")


def infer_sweep_value(sweep: str, manifest: dict) -> float:
    if sweep == "amplitude":
        return float(manifest["amplitude_uA"])
    if sweep == "frequency":
        return float(manifest["frequency_hz"])
    if sweep == "pulse_width":
        return float(manifest["pulse_width_us"])
    if sweep == "worst_case":
        return 1.0
    raise ValueError(f"Unsupported sweep parameter: {sweep}")


def discover_runs(input_root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for npz_path in sorted(input_root.rglob("safety_metrics.npz")):
        manifest_path = npz_path.parent / "run_manifest.yaml"
        if not manifest_path.exists():
            print(f"Skipping {npz_path}: missing run_manifest.yaml")
            continue
        manifest = load_yaml_file(manifest_path)
        block = str(manifest.get("block", ""))
        if block != BASIC_STIMULATION_BLOCK_PREFIX and not block.startswith(f"{BASIC_STIMULATION_BLOCK_PREFIX}_"):
            continue

        metadata = manifest.get("metadata", {}) or {}
        sweep = str(metadata.get("sweep", ""))
        if sweep not in PARAMETER_ORDER:
            print(f"Skipping {npz_path}: unsupported or missing sweep '{sweep}'")
            continue

        run_id = str(manifest.get("run_id", npz_path.parent.name))
        records.append(
            RunRecord(
                npz_path=npz_path,
                run_id=run_id,
                grid_um=infer_grid_um(run_id, manifest),
                sweep=sweep,
                sweep_value=infer_sweep_value(sweep, manifest),
                amplitude_uA=float(manifest["amplitude_uA"]),
                frequency_hz=float(manifest["frequency_hz"]),
                pulse_width_us=float(manifest["pulse_width_us"]),
            )
        )
    return sorted(records, key=lambda r: (PARAMETER_ORDER.index(r.sweep), r.sweep_value, r.grid_um))


def ensure_out(path: Path, *, overwrite: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    return overwrite or not path.exists()


def save_figure(fig: plt.Figure, path: Path, *, overwrite: bool) -> None:
    if ensure_out(path, overwrite=overwrite):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def outputs_older_than(path: Path, dependency: Path) -> bool:
    if not path.exists() or not dependency.exists():
        return False
    dependency_mtime = dependency.stat().st_mtime
    files = [path] if path.is_file() else [item for item in path.rglob("*") if item.is_file()]
    return any(item.stat().st_mtime < dependency_mtime for item in files)


def style_axes(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9, width=0.8)
    ax.grid(axis=grid_axis, alpha=0.25, linewidth=0.6, color="#D7DBE0")


def add_limit_line(ax: plt.Axes, limit: float, *, label: str | None = None) -> None:
    ax.axhline(limit, color="#B91C1C", linewidth=1.5, linestyle=":", label=label or "Safety limit")
    ax.annotate(
        f"{limit:g}",
        xy=(0.99, limit),
        xycoords=("axes fraction", "data"),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#B91C1C",
        clip_on=False,
    )


def place_legend_outside(ax: plt.Axes, *, title: str | None = None) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label and not label.startswith("_") and label not in unique:
            unique[label] = handle
    if not unique:
        return
    ax.legend(
        unique.values(),
        unique.keys(),
        title=title,
        frameon=False,
        fontsize=9,
        title_fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )


def finish_figure_with_outside_legend(fig: plt.Figure) -> None:
    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])


def parameter_label(sweep: str) -> str:
    return {
        "amplitude": "Amplitude (uA)",
        "frequency": "Frequency (Hz)",
        "pulse_width": "Pulse width (us)",
        "worst_case": "Worst case",
    }[sweep]


def value_label(sweep: str, value: float) -> str:
    if sweep == "worst_case":
        return "120 uA, 400 Hz, 800 us"
    suffix = {"amplitude": "uA", "frequency": "Hz", "pulse_width": "us"}[sweep]
    return f"{value:g} {suffix}"


def grid_label(grid_um: int) -> str:
    return f"{grid_um} um"


def grid_color(grid_um: int, index: int = 0) -> str:
    return GRID_COLORS.get(grid_um, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def shade_color(color: str, fraction: float) -> tuple[float, float, float]:
    rgb = np.asarray(to_rgb(color), dtype=np.float64)
    fraction = float(np.clip(fraction, 0.0, 1.0))
    return tuple((rgb * fraction + np.ones(3) * (1.0 - fraction)).tolist())


def parameter_context(records: list[RunRecord], sweep: str) -> str:
    if not records:
        return ""
    record = records[0]
    parts = []
    if sweep != "amplitude":
        parts.append(f"amplitude {record.amplitude_uA:g} uA")
    if sweep != "frequency":
        parts.append(f"frequency {record.frequency_hz:g} Hz")
    if sweep != "pulse_width":
        parts.append(f"pulse width {record.pulse_width_us:g} us")
    return ", ".join(parts)


def records_for(records: Iterable[RunRecord], sweep: str, *, grid_um: int | None = None) -> list[RunRecord]:
    selected = [record for record in records if record.sweep == sweep and (grid_um is None or record.grid_um == grid_um)]
    return sorted(selected, key=lambda r: (r.sweep_value, r.grid_um))


def records_by_value_and_grid(records: list[RunRecord]) -> tuple[list[float], list[int], dict[tuple[float, int], RunRecord]]:
    values = sorted({record.sweep_value for record in records})
    grids = sorted({record.grid_um for record in records})
    lookup = {(record.sweep_value, record.grid_um): record for record in records}
    return values, grids, lookup


def grouped_bar_positions(values: list[float], grids: list[int]) -> tuple[dict[tuple[float, int], float], list[float]]:
    positions: dict[tuple[float, int], float] = {}
    centers: list[float] = []
    group_gap = 0.75
    bar_gap = 0.12
    bar_width = 0.72
    group_width = len(grids) * bar_width + max(0, len(grids) - 1) * bar_gap
    cursor = 0.0
    for value in values:
        start = cursor
        for grid_index, grid_um in enumerate(grids):
            positions[(value, grid_um)] = start + grid_index * (bar_width + bar_gap)
        centers.append(start + (group_width - bar_width) / 2.0)
        cursor += group_width + group_gap
    return positions, centers


def shannon_mean(record: RunRecord) -> float:
    values = get_metric_array(record, "shannon_k_mean", "shannon_k")
    if values is None:
        return float("nan")
    return safe_nanmean(values)


def select_one_record_per_value(records: list[RunRecord], *, preferred_grid_um: int = 800) -> list[RunRecord]:
    selected = []
    for value in sorted({record.sweep_value for record in records}):
        matching = [record for record in records if record.sweep_value == value]
        preferred = [record for record in matching if record.grid_um == preferred_grid_um]
        selected.append((preferred or sorted(matching, key=lambda r: r.grid_um))[0])
    return selected


def derive_charge_per_second_from_frame_charge(record: RunRecord) -> tuple[np.ndarray | None, np.ndarray | None]:
    frame_charge = get_metric_array(record, "frame_charge_per_electrode_nC")
    if frame_charge is None:
        return None, None
    frame_charge = np.asarray(frame_charge, dtype=np.float64)
    if frame_charge.ndim != 2 or frame_charge.shape[0] == 0:
        return None, None

    times = np.asarray(time_series(record), dtype=np.float64).reshape(-1)
    if times.size == 0:
        return None, None
    frame_count = min(frame_charge.shape[0], times.size)
    frame_charge = frame_charge[:frame_count]
    times = times[:frame_count]

    window_entries: list[list[object]] = []
    window_time_s = 0.0
    window_sum = np.zeros(frame_charge.shape[1], dtype=np.float64)
    mean_per_electrode = np.zeros(frame_count, dtype=np.float64)
    total = np.zeros(frame_count, dtype=np.float64)
    prev_time = 0.0

    for idx, (time_s_value, charge_row) in enumerate(zip(times, frame_charge)):
        dt_s = float(time_s_value - prev_time)
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            dt_s = float(np.nanmedian(np.diff(times))) if times.size > 1 else 0.0
        prev_time = float(time_s_value)

        charge_row = np.asarray(charge_row, dtype=np.float64)
        window_entries.append([dt_s, charge_row])
        window_time_s += dt_s
        window_sum = window_sum + charge_row

        while window_time_s > 1.0 + 1e-12 and window_entries:
            excess_s = window_time_s - 1.0
            head_dt_s = float(window_entries[0][0])
            head_charge = np.asarray(window_entries[0][1], dtype=np.float64)
            if head_dt_s <= excess_s + 1e-12:
                window_entries.pop(0)
                window_time_s -= head_dt_s
                window_sum = window_sum - head_charge
            else:
                fraction = excess_s / head_dt_s
                trim_charge = head_charge * fraction
                window_entries[0] = [head_dt_s - excess_s, head_charge - trim_charge]
                window_time_s -= excess_s
                window_sum = window_sum - trim_charge

        active = window_sum > 0.0
        mean_per_electrode[idx] = float(np.mean(window_sum[active])) if np.any(active) else 0.0
        total[idx] = float(np.sum(window_sum))

    return mean_per_electrode, total


def per_second_charge_per_electrode(record: RunRecord) -> np.ndarray | None:
    frame_charge = get_metric_array(record, "frame_charge_per_electrode_nC")
    if frame_charge is None:
        return None
    frame_charge = np.asarray(frame_charge, dtype=np.float64)
    if frame_charge.ndim != 2 or frame_charge.shape[0] == 0 or frame_charge.shape[1] == 0:
        return None

    times = np.asarray(time_series(record), dtype=np.float64).reshape(-1)
    frame_count = min(times.size, frame_charge.shape[0])
    if frame_count == 0:
        return None
    frame_charge = frame_charge[:frame_count]
    times = times[:frame_count]

    if frame_count == 1:
        fps = scalar_float(get_metric_array(record, "fps"), default=float("nan"))
        dt_s = 1.0 / fps if np.isfinite(fps) and fps > 0 else 1.0
        return frame_charge / dt_s

    prev_times = np.concatenate([[0.0], times[:-1]])
    dt = times - prev_times
    finite_dt = dt[np.isfinite(dt) & (dt > 0)]
    fallback_dt = float(np.median(finite_dt)) if finite_dt.size else 1.0
    dt = np.where(np.isfinite(dt) & (dt > 0), dt, fallback_dt)

    second_index = np.floor(np.maximum(times - 1e-12, 0.0)).astype(np.int64)
    n_seconds = int(second_index.max()) + 1
    per_second_charge = np.zeros((n_seconds, frame_charge.shape[1]), dtype=np.float64)
    per_second_duration = np.zeros(n_seconds, dtype=np.float64)
    for idx, second in enumerate(second_index):
        per_second_charge[second] += frame_charge[idx]
        per_second_duration[second] += dt[idx]

    valid = per_second_duration > 0
    if not np.any(valid):
        return None
    per_second = per_second_charge[valid] / per_second_duration[valid, None]
    return per_second


def mean_charge_per_second_distribution(record: RunRecord) -> np.ndarray:
    per_second = per_second_charge_per_electrode(record)
    if per_second is not None:
        active = per_second > 0.0
        active_counts = np.sum(active, axis=1)
        means = np.divide(
            np.sum(per_second, axis=1),
            active_counts,
            out=np.zeros(per_second.shape[0], dtype=np.float64),
            where=active_counts > 0,
        )
        return finite_flat(means, positive_only=True)
    derived_mean, _ = derive_charge_per_second_from_frame_charge(record)
    if derived_mean is not None:
        return finite_flat(derived_mean, positive_only=True)
    values = get_metric_array(
        record,
        "charge_per_second_mean_per_electrode_nC_s",
        "mean_charge_per_second_per_electrode_nC_s",
    )
    if values is None:
        values = get_metric_array(record, "charge_per_second_per_electrode_nC_s")
    if values is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim <= 1:
        return finite_flat(arr, positive_only=True)
    active = arr > 0.0
    active_counts = np.sum(active, axis=1)
    means = np.divide(
        np.sum(arr, axis=1),
        active_counts,
        out=np.zeros(arr.shape[0], dtype=np.float64),
        where=active_counts > 0,
    )
    return finite_flat(means, positive_only=True)


def active_charge_per_second_per_electrode_distribution(record: RunRecord) -> np.ndarray:
    per_second = per_second_charge_per_electrode(record)
    if per_second is not None:
        return finite_flat(per_second[per_second > 0.0], positive_only=True)
    values = get_metric_array(record, "charge_per_second_per_electrode_nC_s")
    if values is None:
        return np.asarray([], dtype=np.float64)
    return finite_flat(values, positive_only=True)


def active_charge_per_second_per_electrode_uC_s(record: RunRecord) -> np.ndarray:
    return active_charge_per_second_per_electrode_distribution(record) / CPS_NC_PER_UC


def max_charge_per_second_distribution(record: RunRecord) -> np.ndarray:
    per_second = per_second_charge_per_electrode(record)
    if per_second is not None:
        return finite_flat(np.max(per_second, axis=1), positive_only=True)
    values = get_metric_array(record, "charge_per_second_per_electrode_nC_s")
    if values is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return finite_flat(arr, positive_only=True)
    return finite_flat(np.max(arr, axis=1), positive_only=True)


def max_charge_per_second_uC_s(record: RunRecord) -> np.ndarray:
    return max_charge_per_second_distribution(record) / CPS_NC_PER_UC


def total_charge_per_second_distribution(record: RunRecord) -> np.ndarray:
    _, derived_total = derive_charge_per_second_from_frame_charge(record)
    if derived_total is not None:
        return finite_flat(derived_total, positive_only=True)
    values = get_metric_array(
        record,
        "charge_per_second_total_nC_s",
        "total_charge_per_second_nC",
        "total_charge_per_second_nC_s",
    )
    return finite_flat(values if values is not None else [], positive_only=True)


def mean_total_charge_per_second(record: RunRecord) -> float:
    return safe_nanmean(total_charge_per_second_distribution(record))


def peak_total_charge_per_second(record: RunRecord) -> float:
    return safe_nanmax(total_charge_per_second_distribution(record))


def mean_total_charge_per_second_mC_s(record: RunRecord) -> float:
    return mean_total_charge_per_second(record) / CPS_NC_PER_MC


def peak_total_charge_per_second_mC_s(record: RunRecord) -> float:
    return peak_total_charge_per_second(record) / CPS_NC_PER_MC


def final_protocol_charge_per_electrode_mC(record: RunRecord) -> np.ndarray:
    frame_charge = get_metric_array(record, "frame_charge_per_electrode_nC")
    if frame_charge is not None:
        final_from_frames = np.asarray(frame_charge, dtype=np.float64).sum(axis=0) / 1e6
        return finite_flat(final_from_frames, positive_only=True)
    values = get_metric_array(
        record,
        "final_protocol_charge_per_electrode_nC",
        "protocol_charge_per_electrode_nC",
        "protocol_charge_nC",
    )
    if values is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        final = arr
    else:
        final = arr[-1, :]
    return finite_flat(final / 1e6, positive_only=True)


def final_protocol_total_mC(record: RunRecord) -> float:
    frame_charge_total = get_metric_array(record, "frame_charge_total_nC")
    if frame_charge_total is not None:
        return float(np.asarray(frame_charge_total, dtype=np.float64).sum() / 1e6)
    frame_charge = get_metric_array(record, "frame_charge_per_electrode_nC")
    if frame_charge is not None:
        return float(np.asarray(frame_charge, dtype=np.float64).sum() / 1e6)
    values = get_metric_array(record, "protocol_charge_total_nC", "total_protocol_nC")
    if values is None:
        return float("nan")
    return safe_last(values) / 1e6


def active_distribution(record: RunRecord) -> np.ndarray:
    values = get_metric_array(record, "active_count")
    return finite_flat(values if values is not None else [], positive_only=True)


def mean_active_amplitude_series(record: RunRecord) -> tuple[np.ndarray, np.ndarray]:
    amplitude = get_metric_array(record, "current_amplitude_per_electrode_uA")
    if amplitude is None:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    amplitude = np.asarray(amplitude, dtype=np.float64)
    if amplitude.ndim != 2 or amplitude.shape[0] == 0 or amplitude.shape[1] == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    times = np.asarray(time_series(record), dtype=np.float64).reshape(-1)
    frame_count = min(times.size, amplitude.shape[0])
    if frame_count == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    times = times[:frame_count]
    amplitude = amplitude[:frame_count]

    active = amplitude > 0.0
    values = np.zeros(frame_count, dtype=np.float64)
    active_frames = np.any(active, axis=1)
    if np.any(active_frames):
        values[active_frames] = [
            float(np.mean(row[mask]))
            for row, mask in zip(amplitude[active_frames], active[active_frames])
        ]
    return times, values


def mean_dT_series(record: RunRecord) -> np.ndarray:
    values = get_metric_array(record, "mean_dT")
    return finite_flat(values if values is not None else [])


def max_dT_series(record: RunRecord) -> np.ndarray:
    values = get_metric_array(record, "max_dT")
    return finite_flat(values if values is not None else [])


def time_series(record: RunRecord, *, prefer_electrode_time: bool = False) -> np.ndarray:
    key_order = ("electrode_time_s", "time_s") if prefer_electrode_time else ("time_s", "electrode_time_s")
    values = get_metric_array(record, *key_order)
    return finite_flat(values if values is not None else [])


def make_grouped_boxplot(
    records: list[RunRecord],
    distribution_getter,
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    overwrite: bool,
    limit: float | None = None,
    limit_label: str | None = None,
    show_legend: bool = True,
) -> None:
    values, grids, _ = records_by_value_and_grid(records)
    positions_by_key, centers = grouped_bar_positions(values, grids)
    distributions = []
    positions = []
    colors = []
    legend_grids = set()
    for idx, record in enumerate(records):
        distribution = np.asarray(distribution_getter(record), dtype=np.float64).reshape(-1)
        distribution = distribution[np.isfinite(distribution)]
        if distribution.size == 0:
            continue
        distributions.append(distribution)
        positions.append(positions_by_key[(record.sweep_value, record.grid_um)])
        colors.append(grid_color(record.grid_um, idx))
        legend_grids.add(record.grid_um)
    if not distributions:
        return

    fig_width = max(9.0, 1.05 * len(distributions) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    boxplot = ax.boxplot(distributions, positions=positions, patch_artist=True, showfliers=False, widths=0.62)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_edgecolor("#374151")
    for item_name in ("whiskers", "caps"):
        for item in boxplot[item_name]:
            item.set_color("#4B5563")
    for median in boxplot["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.3)
    if limit is not None and np.isfinite(limit):
        add_limit_line(ax, limit, label=limit_label)
    legend_handles = [
        Patch(facecolor=grid_color(grid_um), edgecolor="#374151", alpha=0.68, label=grid_label(grid_um))
        for grid_um in sorted(legend_grids)
    ]
    if limit is not None and np.isfinite(limit):
        legend_handles.append(plt.Line2D([0], [0], color="#B91C1C", linewidth=1.5, linestyle=":", label=limit_label or "Safety limit"))
    ax.set_xticks(centers)
    ax.set_xticklabels([value_label(records[0].sweep, value) for value in values], rotation=0, ha="center")
    ax.set_xlabel(parameter_label(records[0].sweep))
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(min(positions) - 0.75, max(positions) + 0.75)
    style_axes(ax)
    if show_legend and legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            fontsize=9,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0.0,
        )
    finish_figure_with_outside_legend(fig)
    save_figure(fig, out_path, overwrite=overwrite)


def make_parameter_boxplot(
    records: list[RunRecord],
    distribution_getter,
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    overwrite: bool,
    limit: float | None = None,
    limit_label: str | None = None,
    show_legend: bool = True,
) -> None:
    rows = []
    for record in sorted(records, key=lambda r: r.sweep_value):
        distribution = np.asarray(distribution_getter(record), dtype=np.float64).reshape(-1)
        distribution = distribution[np.isfinite(distribution)]
        if distribution.size:
            rows.append((record, distribution))
    if not rows:
        return

    x = np.arange(1, len(rows) + 1, dtype=np.float64)
    fig_width = max(7.0, 0.9 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    boxplot = ax.boxplot([values for _, values in rows], positions=x, patch_artist=True, showfliers=False, widths=0.62)
    color = grid_color(rows[0][0].grid_um)
    for patch in boxplot["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_edgecolor("#374151")
    for item_name in ("whiskers", "caps"):
        for item in boxplot[item_name]:
            item.set_color("#4B5563")
    for median in boxplot["medians"]:
        median.set_color("#111827")
        median.set_linewidth(1.3)
    if limit is not None and np.isfinite(limit):
        add_limit_line(ax, limit, label=(limit_label or "Safety limit") if show_legend else "_nolegend_")
    ax.set_xticks(x)
    ax.set_xticklabels([value_label(record.sweep, record.sweep_value) for record, _ in rows])
    ax.set_xlabel(parameter_label(rows[0][0].sweep))
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.set_ylim(bottom=0.0)
    style_axes(ax)
    if show_legend:
        place_legend_outside(ax)
    finish_figure_with_outside_legend(fig)
    save_figure(fig, out_path, overwrite=overwrite)


def make_grouped_barplot(
    records: list[RunRecord],
    value_getter,
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    overwrite: bool,
    limit: float | None = None,
    limit_label: str | None = None,
) -> None:
    rows = []
    for record in records:
        value = float(value_getter(record))
        if np.isfinite(value):
            rows.append((record, value))
    if not rows:
        return

    sweep_values, grids, _ = records_by_value_and_grid([record for record, _ in rows])
    positions_by_key, centers = grouped_bar_positions(sweep_values, grids)
    positions = [positions_by_key[(record.sweep_value, record.grid_um)] for record, _ in rows]
    plot_values = [value for _, value in rows]
    colors = [grid_color(record.grid_um, idx) for idx, (record, _) in enumerate(rows)]
    fig_width = max(8.0, 1.05 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    for idx, ((record, value), position, color) in enumerate(zip(rows, positions, colors)):
        label = grid_label(record.grid_um) if record.grid_um not in [r.grid_um for r, _ in rows[:idx]] else "_nolegend_"
        ax.bar(position, value, width=0.68, color=color, edgecolor="#374151", linewidth=0.8, alpha=0.86, label=label)
    if limit is not None and np.isfinite(limit):
        add_limit_line(ax, limit, label=limit_label)
    ax.set_xticks(centers)
    ax.set_xticklabels([value_label(rows[0][0].sweep, value) for value in sweep_values], rotation=0, ha="center")
    ax.set_xlabel(parameter_label(rows[0][0].sweep))
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.set_xlim(min(positions) - 0.75, max(positions) + 0.75)
    style_axes(ax)
    place_legend_outside(ax, title="Grid" if len(grids) > 1 else None)
    finish_figure_with_outside_legend(fig)
    save_figure(fig, out_path, overwrite=overwrite)


def make_parameter_barplot(
    records: list[RunRecord],
    value_getter,
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    overwrite: bool,
    limit: float | None = None,
    limit_label: str | None = None,
    show_legend: bool = True,
) -> None:
    rows = []
    for record in records:
        value = float(value_getter(record))
        if np.isfinite(value):
            rows.append((record, value))
    if not rows:
        return

    rows = sorted(rows, key=lambda item: item[0].sweep_value)
    x = np.arange(len(rows), dtype=np.float64)
    values = [value for _, value in rows]
    fig_width = max(7.0, 0.9 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(
        x,
        values,
        width=0.68,
        color=grid_color(rows[0][0].grid_um),
        edgecolor="#374151",
        linewidth=0.8,
        alpha=0.86,
    )
    if limit is not None and np.isfinite(limit):
        add_limit_line(ax, limit, label=(limit_label or "Safety limit") if show_legend else "_nolegend_")
    ax.set_xticks(x)
    ax.set_xticklabels([value_label(record.sweep, record.sweep_value) for record, _ in rows])
    ax.set_xlabel(parameter_label(rows[0][0].sweep))
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    style_axes(ax)
    if show_legend:
        place_legend_outside(ax)
    finish_figure_with_outside_legend(fig)
    save_figure(fig, out_path, overwrite=overwrite)


def plot_shannon_k(records: list[RunRecord], out_root: Path, image_format: str, *, overwrite: bool) -> None:
    shannon_dir = out_root / "shannon_k"
    for sweep in PARAMETER_ORDER:
        selected = records_for(records, sweep)
        if not selected:
            continue
        context = parameter_context(selected, sweep)
        title = f"Mean Shannon K by {sweep.replace('_', ' ')}"
        if context:
            title = f"{title} ({context})"
        make_grouped_barplot(
            selected,
            shannon_mean,
            shannon_dir / f"shannon_k_mean_by_{sweep}.{image_format}",
            title=title,
            ylabel="Mean Shannon K",
            overwrite=overwrite,
            limit=SHANNON_K_LIMIT_FALLBACK,
            limit_label="Shannon K limit (1.85)",
        )


def plot_charge_per_second(
    records: list[RunRecord],
    out_root: Path,
    image_format: str,
    *,
    limits: dict[str, float],
    overwrite: bool,
    include: set[str] | None = None,
) -> None:
    if include is None:
        include = {"mean_charge", "max_charge", "total_mean_charge", "total_max_charge"}
    out_dir = out_root / "charge_per_second"
    per_electrode_limit_uC_s = limits["window_charge_per_electrode_nC"] / CPS_NC_PER_UC
    total_limit_mC_s = limits["window_charge_total_nC"] / CPS_NC_PER_MC
    for sweep in PARAMETER_ORDER:
        selected_all_grids = records_for(records, sweep)
        if not selected_all_grids:
            continue
        if "mean_charge" in include:
            make_grouped_boxplot(
                selected_all_grids,
                active_charge_per_second_per_electrode_uC_s,
                out_dir / f"mean_per_electrode_by_{sweep}.{image_format}",
                title=f"Active electrode charge per second by {sweep.replace('_', ' ')}",
                ylabel="Charge per active electrode per second (uC/s)",
                overwrite=overwrite,
                limit=per_electrode_limit_uC_s,
                limit_label="Per-electrode limit",
            )
        if "max_charge" in include:
            make_grouped_boxplot(
                selected_all_grids,
                max_charge_per_second_uC_s,
                out_dir / f"max_per_electrode_by_{sweep}.{image_format}",
                title=f"Maximum charge per second per electrode by {sweep.replace('_', ' ')}",
                ylabel="Maximum charge per electrode per second (uC/s)",
                overwrite=overwrite,
                limit=per_electrode_limit_uC_s,
                limit_label="Per-electrode limit",
            )
        if "total_mean_charge" in include:
            make_grouped_barplot(
                selected_all_grids,
                mean_total_charge_per_second_mC_s,
                out_dir / f"mean_total_across_electrodes_by_{sweep}.{image_format}",
                title=f"Mean summed charge per second across all electrodes by {sweep.replace('_', ' ')}",
                ylabel="Mean summed charge per second (mC/s)",
                overwrite=overwrite,
                limit=total_limit_mC_s,
                limit_label="Total-array limit",
            )
        if "total_max_charge" in include:
            make_grouped_barplot(
                selected_all_grids,
                peak_total_charge_per_second_mC_s,
                out_dir / f"max_total_across_electrodes_by_{sweep}.{image_format}",
                title=f"Maximum summed charge per second across all electrodes by {sweep.replace('_', ' ')}",
                ylabel="Maximum summed charge per second (mC/s)",
                overwrite=overwrite,
                limit=total_limit_mC_s,
                limit_label="Total-array limit",
            )


def plot_total_charge(
    records: list[RunRecord],
    out_root: Path,
    image_format: str,
    *,
    limits: dict[str, float],
    total_session_limit_mC: float,
    overwrite: bool,
) -> None:
    out_dir = out_root / "total_charge"
    for sweep in PARAMETER_ORDER:
        selected = records_for(records, sweep)
        if not selected:
            continue
        make_grouped_boxplot(
            selected,
            final_protocol_charge_per_electrode_mC,
            out_dir / f"final_per_electrode_by_{sweep}.{image_format}",
            title=f"Final accumulated charge per electrode by {sweep.replace('_', ' ')}",
            ylabel="Final charge per electrode (mC)",
            overwrite=overwrite,
        )
        make_grouped_barplot(
            selected,
            final_protocol_total_mC,
            out_dir / f"final_total_across_electrodes_by_{sweep}.{image_format}",
            title=f"Final accumulated total charge by {sweep.replace('_', ' ')}",
            ylabel="Final total charge (mC)",
            overwrite=overwrite,
            limit=total_session_limit_mC,
            limit_label="Session total limit",
        )


def is_standard_case(record: RunRecord) -> bool:
    return (
        math.isclose(record.amplitude_uA, STANDARD_AMPLITUDE_UA, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(record.frequency_hz, STANDARD_FREQUENCY_HZ, rel_tol=0.0, abs_tol=1e-6)
        and math.isclose(record.pulse_width_us, STANDARD_PULSE_WIDTH_US, rel_tol=0.0, abs_tol=1e-6)
    )


def plot_active_electrodes(records: list[RunRecord], out_root: Path, image_format: str, *, overwrite: bool) -> None:
    standard_by_grid: dict[int, RunRecord] = {}
    for record in records:
        if is_standard_case(record):
            standard_by_grid.setdefault(record.grid_um, record)
    selected = [standard_by_grid[grid] for grid in sorted(standard_by_grid)]
    if not selected:
        return

    out_path = out_root / "active_electrodes" / f"standard_case_active_electrodes.{image_format}"
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    plotted = False
    for idx, record in enumerate(selected):
        y_values = get_metric_array(record, "active_count")
        if y_values is None:
            continue
        y_values = np.asarray(y_values, dtype=np.float64).reshape(-1)
        x_values = np.asarray(time_series(record), dtype=np.float64).reshape(-1) / 60.0
        count = min(x_values.size, y_values.size)
        if count == 0:
            continue
        ax.plot(
            x_values[:count],
            y_values[:count],
            linewidth=1.8,
            color=grid_color(record.grid_um, idx),
            label=grid_label(record.grid_um),
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Active electrodes per frame")
    ax.set_title("Active electrodes over time, standard stimulation case", pad=10)
    ax.set_ylim(bottom=0.0)
    style_axes(ax)
    place_legend_outside(ax, title="Grid")
    finish_figure_with_outside_legend(fig)
    save_figure(fig, out_path, overwrite=overwrite)

    bar_rows = []
    for record in selected:
        mean_active = safe_nanmean(active_distribution(record))
        if np.isfinite(mean_active):
            bar_rows.append((record, mean_active))
    if not bar_rows:
        return

    mean_out_path = out_root / "active_electrodes" / f"standard_case_mean_active_electrodes.{image_format}"
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    x = np.arange(len(bar_rows), dtype=np.float64)
    ax.bar(
        x,
        [value for _, value in bar_rows],
        width=0.62,
        color=[grid_color(record.grid_um, idx) for idx, (record, _) in enumerate(bar_rows)],
        edgecolor="#374151",
        linewidth=0.8,
        alpha=0.86,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([grid_label(record.grid_um) for record, _ in bar_rows])
    ax.set_ylabel("Mean active electrodes per active frame")
    ax.set_title("Mean active electrodes, standard stimulation case", pad=10)
    ax.set_ylim(bottom=0.0)
    style_axes(ax)
    fig.tight_layout()
    save_figure(fig, mean_out_path, overwrite=overwrite)

    box_rows = []
    for record in selected:
        distribution = active_distribution(record)
        if distribution.size:
            box_rows.append((record, distribution))
    if box_rows:
        box_out_path = out_root / "active_electrodes" / f"standard_case_active_electrodes_boxplot.{image_format}"
        fig, ax = plt.subplots(figsize=(6.8, 4.4))
        x = np.arange(1, len(box_rows) + 1, dtype=np.float64)
        boxplot = ax.boxplot(
            [distribution for _, distribution in box_rows],
            positions=x,
            patch_artist=True,
            showfliers=False,
            widths=0.62,
        )
        for idx, patch in enumerate(boxplot["boxes"]):
            patch.set_facecolor(grid_color(box_rows[idx][0].grid_um, idx))
            patch.set_alpha(0.68)
            patch.set_edgecolor("#374151")
        for item_name in ("whiskers", "caps"):
            for item in boxplot[item_name]:
                item.set_color("#4B5563")
        for median in boxplot["medians"]:
            median.set_color("#111827")
            median.set_linewidth(1.3)
        ax.set_xticks(x)
        ax.set_xticklabels([grid_label(record.grid_um) for record, _ in box_rows])
        ax.set_ylabel("Active electrodes per frame")
        ax.set_title("Active electrode count distribution, standard case", pad=10)
        ax.set_ylim(bottom=0.0)
        style_axes(ax)
        fig.tight_layout()
        save_figure(fig, box_out_path, overwrite=overwrite)


def plot_amplitudes(
    records: list[RunRecord],
    out_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    out_dir = out_root / "amplitudes"
    for sweep in PARAMETER_ORDER:
        selected = records_for(records, sweep)
        if not selected:
            continue

        for grid_um in sorted({record.grid_um for record in selected}):
            grid_records = sorted(
                [record for record in selected if record.grid_um == grid_um],
                key=lambda r: r.sweep_value,
            )
            if not grid_records:
                continue
            fig, ax = plt.subplots(figsize=(9.2, 4.8))
            plotted = False
            base_color = grid_color(grid_um)
            for idx, record in enumerate(grid_records):
                x_values, y_values = mean_active_amplitude_series(record)
                count = min(x_values.size, y_values.size)
                if count == 0:
                    continue
                shade = 0.42 + 0.48 * (idx / max(1, len(grid_records) - 1))
                ax.plot(
                    x_values[:count] / 60.0,
                    y_values[:count],
                    linewidth=1.8,
                    label=value_label(record.sweep, record.sweep_value),
                    color=shade_color(base_color, shade),
                )
                plotted = True
            if not plotted:
                plt.close(fig)
                continue

            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Mean active electrode amplitude (uA)")
            ax.set_title(f"Stimulation amplitudes by {sweep.replace('_', ' ')}, {grid_label(grid_um)}", pad=10)
            ax.set_ylim(bottom=0.0)
            style_axes(ax)
            place_legend_outside(ax)
            finish_figure_with_outside_legend(fig)
            save_figure(
                fig,
                out_dir / f"amplitudes_evolution_by_{sweep}_{grid_um}um.{image_format}",
                overwrite=overwrite,
            )


def temperature_summary_values(record: RunRecord) -> tuple[float, float, float, float]:
    mean_series = mean_dT_series(record)
    max_series = max_dT_series(record)
    return (
        safe_last(mean_series),
        safe_nanmax(mean_series),
        safe_last(max_series),
        safe_nanmax(max_series),
    )


def plot_temperature_bars(records: list[RunRecord], out_root: Path, image_format: str, *, overwrite: bool) -> None:
    out_dir = out_root / "temperature"
    specs = [
        (0, "final_mean_dT", "Final mean temperature rise"),
        (1, "max_mean_dT", "Maximum mean temperature rise"),
        (2, "final_max_dT", "Final maximum temperature rise"),
        (3, "max_max_dT", "Maximum temperature rise"),
    ]
    for sweep in PARAMETER_ORDER:
        selected = records_for(records, sweep)
        if not selected:
            continue
        for value_index, filename, title in specs:
            make_grouped_barplot(
                selected,
                lambda record, idx=value_index: temperature_summary_values(record)[idx],
                out_dir / f"{filename}_by_{sweep}.{image_format}",
                title=f"{title} by {sweep.replace('_', ' ')}",
                ylabel="Temperature rise (deg C)",
                overwrite=overwrite,
            )


def plot_temperature_evolution(records: list[RunRecord], out_root: Path, image_format: str, *, overwrite: bool) -> None:
    out_dir = out_root / "temperature"
    for sweep in PARAMETER_ORDER:
        selected = records_for(records, sweep)
        if not selected:
            continue
        values = sorted({record.sweep_value for record in selected})
        for grid_um in sorted({record.grid_um for record in selected}):
            fig, ax = plt.subplots(figsize=(9.2, 5.0))
            grid_records = [record for record in selected if record.grid_um == grid_um]
            base_color = grid_color(grid_um)
            for idx, record in enumerate(sorted(grid_records, key=lambda r: r.sweep_value)):
                y_values = np.asarray(max_dT_series(record), dtype=np.float64)
                x_values = np.asarray(time_series(record), dtype=np.float64) / 60.0
                count = min(x_values.size, y_values.size)
                if count == 0:
                    continue
                shade = 0.42 + 0.48 * (idx / max(1, len(grid_records) - 1))
                ax.plot(
                    x_values[:count],
                    y_values[:count],
                    linewidth=1.8,
                    label=value_label(record.sweep, record.sweep_value),
                    color=shade_color(base_color, shade),
                )
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Maximum temperature rise (deg C)")
            ax.set_title(f"Maximum dT over time by {sweep.replace('_', ' ')}, {grid_label(grid_um)}", pad=10)
            style_axes(ax)
            place_legend_outside(ax)
            finish_figure_with_outside_legend(fig)
            save_figure(fig, out_dir / f"max_dT_evolution_by_{sweep}_{grid_um}um.{image_format}", overwrite=overwrite)

        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        for idx, grid_um in enumerate(sorted({record.grid_um for record in selected})):
            y_values = []
            x_values = []
            for value in values:
                matching = [record for record in selected if record.grid_um == grid_um and record.sweep_value == value]
                if not matching:
                    continue
                worst = safe_nanmax(max_dT_series(matching[0]))
                if np.isfinite(worst):
                    x_values.append(value)
                    y_values.append(worst)
            if y_values:
                ax.plot(
                    x_values,
                    y_values,
                    marker="o",
                    linewidth=2.0,
                    color=grid_color(grid_um, idx),
                    label=grid_label(grid_um),
                )
        ax.set_xlabel(parameter_label(sweep))
        ax.set_ylabel("Worst maximum temperature rise (deg C)")
        ax.set_title(f"Worst maximum dT by {sweep.replace('_', ' ')}", pad=10)
        style_axes(ax)
        place_legend_outside(ax, title="Grid")
        finish_figure_with_outside_legend(fig)
        save_figure(fig, out_dir / f"max_dT_worst_case_by_{sweep}.{image_format}", overwrite=overwrite)


def decode_string_values(values: np.ndarray | None) -> list[str]:
    if values is None:
        return []
    arr = np.asarray(values).reshape(-1)
    decoded = []
    for value in arr:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def sanitize_path_part(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return safe.strip("._") or "item"


def logged_temperature_heatmaps(record: RunRecord) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    with np.load(record.npz_path, allow_pickle=True) as data:
        files = set(data.files)
        grid_names = decode_string_values(np.asarray(data["heatmap_grid_names"]) if "heatmap_grid_names" in files else None)
        times_s = np.asarray(data["heatmap_times_s"], dtype=np.float64).reshape(-1) if "heatmap_times_s" in files else np.asarray([])
        panels = []
        for grid_name in grid_names:
            suffix = sanitize_path_part(grid_name)
            heatmap_key = f"dT_heatmaps_{suffix}"
            if heatmap_key not in files:
                continue
            extent_key = f"extent_mm_{suffix}" if f"extent_mm_{suffix}" in files else "extent_mm"
            if extent_key not in files:
                continue
            panels.append(
                (
                    grid_name,
                    np.asarray(data[heatmap_key]),
                    np.asarray(data[extent_key], dtype=np.float64).reshape(-1),
                    times_s,
                )
            )
        if panels:
            return panels
    return []


def plot_heatmaps(records: list[RunRecord], out_root: Path, image_format: str, *, overwrite: bool) -> None:
    out_dir = out_root / "heatmaps"
    for record in records:
        panels = logged_temperature_heatmaps(record)
        if not panels:
            continue
        for panel_name, heatmaps, extent_mm, times_s in panels:
            if heatmaps.ndim != 3 or heatmaps.shape[0] == 0:
                continue
            out_path = (
                out_dir
                / record.sweep
                / f"{record.grid_um}um_{record.sweep}_{record.sweep_value:g}_{sanitize_path_part(panel_name)}.{image_format}"
            )
            if not ensure_out(out_path, overwrite=overwrite):
                continue
            finite_heatmaps = heatmaps[np.isfinite(heatmaps)]
            vmax = float(np.max(finite_heatmaps)) if finite_heatmaps.size else 0.0
            ncols = min(5, heatmaps.shape[0])
            nrows = int(math.ceil(float(heatmaps.shape[0]) / float(ncols)))
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(3.0 * ncols, 2.8 * nrows),
                squeeze=False,
            )
            for ax in axes.ravel():
                ax.axis("off")
            image = None
            for idx, heatmap in enumerate(heatmaps):
                ax = axes.ravel()[idx]
                image = ax.imshow(
                    heatmap,
                    origin="lower",
                    extent=[extent_mm[0], extent_mm[1], extent_mm[2], extent_mm[3]],
                    aspect="equal",
                    cmap="inferno",
                    vmin=0.0,
                    vmax=vmax if vmax > 0.0 else None,
                )
                time_label = f"{times_s[idx]:.1f} s" if idx < times_s.size and np.isfinite(times_s[idx]) else f"#{idx + 1}"
                ax.set_title(time_label, fontsize=9)
                ax.tick_params(labelsize=7, width=0.6)
                ax.axis("on")
                ax.grid(False)
            fig.suptitle(
                f"dT heatmaps over logged timestamps | {grid_label(record.grid_um)} | "
                f"{value_label(record.sweep, record.sweep_value)} | {panel_name}",
                fontsize=12,
                y=0.995,
            )
            fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.97])
            if image is not None:
                cax = fig.add_axes([0.94, 0.12, 0.018, 0.76])
                fig.colorbar(image, cax=cax, label="dT (deg C)")
            save_figure(fig, out_path, overwrite=True)


def build_summary_row(record: RunRecord) -> dict[str, object]:
    mean_cps = mean_charge_per_second_distribution(record)
    total_cps = total_charge_per_second_distribution(record)
    active = active_distribution(record)
    mean_dT = mean_dT_series(record)
    max_dT = max_dT_series(record)
    return {
        "run_id": record.run_id,
        "grid_um": record.grid_um,
        "sweep": record.sweep,
        "sweep_value": record.sweep_value,
        "amplitude_uA": record.amplitude_uA,
        "frequency_hz": record.frequency_hz,
        "pulse_width_us": record.pulse_width_us,
        "mean_shannon_k": shannon_mean(record),
        "mean_charge_per_second_per_electrode_nC_s": safe_nanmean(mean_cps),
        "peak_charge_per_second_per_electrode_nC_s": safe_nanmax(mean_cps),
        "mean_total_charge_per_second_nC_s": safe_nanmean(total_cps),
        "peak_total_charge_per_second_nC_s": safe_nanmax(total_cps),
        "final_total_charge_mC": final_protocol_total_mC(record),
        "mean_final_charge_per_electrode_mC": safe_nanmean(final_protocol_charge_per_electrode_mC(record)),
        "mean_active_electrodes": safe_nanmean(active),
        "max_active_electrodes": safe_nanmax(active),
        "final_mean_dT_C": safe_last(mean_dT),
        "peak_mean_dT_C": safe_nanmax(mean_dT),
        "final_max_dT_C": safe_last(max_dT),
        "peak_max_dT_C": safe_nanmax(max_dT),
        "npz_path": str(record.npz_path),
    }


def write_summary_csv(records: list[RunRecord], out_path: Path, *, overwrite: bool) -> None:
    if not ensure_out(out_path, overwrite=overwrite):
        return
    rows = [build_summary_row(record) for record in records]
    if not rows:
        return
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clean_generated_outputs(output_root: Path) -> None:
    generated_paths = [
        output_root / "summary.csv",
        output_root / "shannon_k",
        output_root / "charge_per_second",
        output_root / "total_charge",
        output_root / "active_electrodes",
        output_root / "amplitudes",
        output_root / "amplitude_clouds",
        output_root / "temperature",
        output_root / "heatmaps",
    ]
    for path in generated_paths:
        if path.is_file():
            path.unlink()
            continue
        if not path.is_dir():
            continue
        for file_path in sorted((p for p in path.rglob("*") if p.is_file()), reverse=True):
            file_path.unlink()


def read_shannon_limit(safety_yaml: Path, limits: dict[str, float]) -> float:
    try:
        cfg = load_yaml_file(safety_yaml)
    except FileNotFoundError:
        return SHANNON_K_LIMIT_FALLBACK
    threshold_value = ((cfg.get("thresholds", {}) or {}).get("shannon_k_limit", None))
    if threshold_value is not None:
        return float(threshold_value)
    guideline_value = ((cfg.get("stimulation_safety_guidelines", {}) or {}).get("shannon_k", {}) or {}).get("value", None)
    if guideline_value is not None:
        return float(guideline_value)
    _ = limits
    return SHANNON_K_LIMIT_FALLBACK


def main() -> None:
    args = parse_args()
    selected = selected_results(args.only)
    input_root = resolve_repo_path(args.input_root)
    output_root = resolve_repo_path(args.output_root)
    safety_yaml = resolve_repo_path(args.safety_yaml)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    limits = load_safety_limits(safety_yaml)
    total_session_limit_mC = load_total_session_charge_limit_mC(safety_yaml)
    global SHANNON_K_LIMIT_FALLBACK
    SHANNON_K_LIMIT_FALLBACK = read_shannon_limit(safety_yaml, limits)

    records = discover_runs(input_root)
    records = [record for record in records if record.grid_um in INCLUDED_GRIDS_UM]
    if not records:
        raise RuntimeError(f"No basic stimulation safety_metrics.npz files found under: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    overwrite_outputs = bool(args.overwrite)
    if not overwrite_outputs and outputs_older_than(output_root, safety_yaml):
        overwrite_outputs = True
        print(f"Safety config is newer than existing visuals; refreshing plots from: {safety_yaml}")
    if overwrite_outputs and selected == set(PLOT_CHOICES):
        clean_generated_outputs(output_root)
    if "summary" in selected:
        write_summary_csv(records, output_root / "summary.csv", overwrite=overwrite_outputs)
    if "shannon_k" in selected:
        plot_shannon_k(records, output_root, args.format, overwrite=overwrite_outputs)
    charge_selection = selected & {"mean_charge", "max_charge", "total_mean_charge", "total_max_charge"}
    if charge_selection:
        plot_charge_per_second(
            records,
            output_root,
            args.format,
            limits=limits,
            overwrite=overwrite_outputs,
            include=charge_selection,
        )
    if "final_charge" in selected:
        plot_total_charge(
            records,
            output_root,
            args.format,
            limits=limits,
            total_session_limit_mC=total_session_limit_mC,
            overwrite=overwrite_outputs,
        )
    if "active_electrodes" in selected:
        plot_active_electrodes(records, output_root, args.format, overwrite=overwrite_outputs)
    if "amplitudes" in selected:
        plot_amplitudes(records, output_root, args.format, overwrite=overwrite_outputs)
    if "temperature" in selected:
        plot_temperature_bars(records, output_root, args.format, overwrite=overwrite_outputs)
        plot_temperature_evolution(records, output_root, args.format, overwrite=overwrite_outputs)
    if "heatmaps" in selected:
        plot_heatmaps(records, output_root, args.format, overwrite=overwrite_outputs)

    grids = ", ".join(str(grid) for grid in sorted({record.grid_um for record in records}))
    print(f"Loaded {len(records)} basic stimulation runs across grids: {grids} um")
    print(f"Generated result groups: {', '.join(result for result in PLOT_CHOICES if result in selected)}")
    print(
        "Charge limits: "
        f"per-electrode={limits['window_charge_per_electrode_nC']:g} nC/s, "
        f"total-array={limits['window_charge_total_nC']:g} nC/s"
    )
    print(f"Saved basic stimulation visuals to: {output_root}")


if __name__ == "__main__":
    main()
