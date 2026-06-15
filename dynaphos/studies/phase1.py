"""
Generate comparison plots for canonical DynaPhos phase 1 results.
"""

from __future__ import annotations

import argparse
import csv
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml


from dynaphos.experiment.artifacts import discover_completed_runs
from dynaphos.paths import package_file
from dynaphos.reporting import dashboard as single_case
from dynaphos.safety.limits import load_safety_limits
from dynaphos.studies.common import resolve_path as resolve_repo_path
from dynaphos.studies.common import sanitize_path_part


DEFAULT_INPUT_ROOT = Path("results/safety/simulation_pipeline/amplitude_grid_preprocessing")
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "comparative_visuals"
DEFAULT_SAFETY = package_file("safety")
STANDARD_AMPLITUDE_UA = 60.0

MATRIX_BLOCK = "amplitude_grid_preprocessing"
BLOCK_ORDER = ("amplitude", "electrode_density", "preprocessing", "internal_circuit", "rastering", MATRIX_BLOCK)
THERMAL_CMAP = "hot"
COLORS = {
    "amplitude": "#1F4E79",
    "electrode_density": "#2E8B57",
    "preprocessing": "#8B1E3F",
    "internal_circuit": "#C05621",
    "rastering": "#5B5EA6",
}
FALLBACK_COLOR = "#6B7280"
GRID_COLOR = "#D7DBE0"
TIME_AXIS_LABEL = "Time (min)"
MAX_SERIES_POINTS = 1600
MAX_CLOUD_POINTS = 900
PER_PROTOCOL_SERIES_POINTS = 360
MAX_HEATMAP_PANELS = 20
THERMAL_BLOCK = "internal_circuit"
ELECTRICAL_COMPARISON_BLOCKS = ("amplitude", "electrode_density", "preprocessing", "rastering")
CLOUD_PREPROCESSING_TOKENS = ("dog", "difference_of_gaussians", "difference-of-gaussians")

LINE_COLORS = (
    "#1F4E79",
    "#2E8B57",
    "#8B1E3F",
    "#C05621",
    "#5B5EA6",
    "#2563EB",
    "#7C2D12",
    "#047857",
    "#9333EA",
    "#4B5563",
)
MATRIX_MARKERS = ("o", "s", "^", "D", "v", "P")
MATRIX_LINESTYLES = ("-", "--", "-.", ":")

FACTOR_ORDER = ("amplitude", "grid", "preprocessing")
PAIRWISE_SPECS = (
    ("amplitude", "grid", "preprocessing", "amplitude_vs_electrode_grid"),
)

_CASE_VISUALS_ROOT: Path | None = None
_PLOT_SAFETY_LIMITS: dict[str, float] = {}


@dataclass(frozen=True)
class MatrixRecord:
    npz_path: Path
    manifest_path: Path
    block: str
    run_id: str
    label: str
    sort_value: float
    metrics: dict[str, float]
    ratios: dict[str, float]


@dataclass(frozen=True)
class MatrixFactors:
    amplitude: float
    amplitude_key: str
    amplitude_label: str
    grid_um: int
    grid_key: str
    grid_label: str
    preprocessing_key: str
    preprocessing_label: str
    electrode_count: int


@dataclass(frozen=True)
class FactorValue:
    key: str
    label: str
    sort_value: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for run_phase1.py outputs."
    )
    parser.add_argument(
        "--input-root",
        default=str(DEFAULT_INPUT_ROOT),
        help="Directory containing matrix block folders.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where comparison plots and CSV files are written.",
    )
    parser.add_argument(
        "--safety-yaml",
        default=DEFAULT_SAFETY,
        help="Safety YAML used for normalization limits.",
    )
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing figures.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def scalar(data: np.lib.npyio.NpzFile, key: str, default: float = math.nan) -> float:
    if key not in data.files:
        return default
    values = np.asarray(data[key])
    if values.size == 0:
        return default
    try:
        return float(values.reshape(-1)[0])
    except (TypeError, ValueError):
        return default


def max_value(data: np.lib.npyio.NpzFile, key: str, default: float = 0.0) -> float:
    if key not in data.files:
        return default
    values = np.asarray(data[key], dtype=np.float64)
    if values.size == 0:
        return default
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.max(finite))


def peak_value_and_time(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    key: str,
) -> tuple[float, float, int]:
    values = get_data_array(data, key, dtype=np.float64)
    if values is None or values.size == 0:
        return math.nan, math.nan, -1
    series = values.reshape(-1)
    finite_indices = np.flatnonzero(np.isfinite(series))
    if finite_indices.size == 0:
        return math.nan, math.nan, -1
    peak_index = int(finite_indices[int(np.argmax(series[finite_indices]))])
    peak_time_s = math.nan
    for time_key in ("thermal_time_s", "time_s"):
        times = get_data_array(data, time_key, dtype=np.float64)
        if times is None:
            continue
        time_values = times.reshape(-1)
        if time_values.size == series.size and np.isfinite(time_values[peak_index]):
            peak_time_s = float(time_values[peak_index])
            break
    return float(series[peak_index]), peak_time_s, peak_index


def metric_or_peak(data: np.lib.npyio.NpzFile, scalar_key: str, series_key: str, default: float = 0.0) -> float:
    value = scalar(data, scalar_key, math.nan)
    if np.isfinite(value):
        return value
    return max_value(data, series_key, default)


def safe_ratio(value: float, limit: float) -> float:
    if not np.isfinite(value) or not np.isfinite(limit) or limit <= 0.0:
        return math.nan
    return value / limit


def set_case_visuals_root(output_root: Path | None) -> None:
    global _CASE_VISUALS_ROOT
    _CASE_VISUALS_ROOT = output_root


def set_plot_safety_limits(limits: Mapping[str, float]) -> None:
    global _PLOT_SAFETY_LIMITS
    _PLOT_SAFETY_LIMITS = {str(key): float(value) for key, value in limits.items()}


def plot_safety_limit(key: str) -> float:
    return float(_PLOT_SAFETY_LIMITS.get(key, math.nan))


def add_limit_line(
    ax: plt.Axes,
    value: float,
    *,
    label: str = "safety limit",
    color: str = "#8B1E3F",
    linestyle: str = "--",
) -> bool:
    if not np.isfinite(value) or value <= 0.0:
        return False
    ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.15, label=label)
    return True


def block_label_and_sort(block: str, manifest: dict) -> tuple[str, float]:
    if block == MATRIX_BLOCK:
        amp = float(manifest.get("amplitude_uA", math.nan))
        coords = Path(str(manifest.get("coords_yaml", ""))).stem
        grid_digits = digits_from_text(coords)
        grid = f"{grid_digits}um" if grid_digits else coords
        source = normalize_preprocessing_label(
            manifest.get("source_input_label", manifest.get("preprocessing_method", ""))
        )
        amp_text = f"{amp:g}uA" if np.isfinite(amp) else "amp?"
        return f"{grid}\n{display_preprocessing_label(source)}\n{amp_text}", amp
    if block == "amplitude":
        amp = float(manifest["amplitude_uA"])
        threshold = float(manifest["appearance_threshold_uA"])
        return f"{amp:g} uA\nthr {threshold:g}", amp
    if block == "electrode_density":
        coords = Path(str(manifest["coords_yaml"])).stem
        digits = "".join(ch for ch in coords if ch.isdigit())
        pitch = float(digits) if digits else math.nan
        return coords.replace("coords_", ""), pitch
    if block == "preprocessing":
        source = manifest.get("source_input_label", manifest.get("preprocessing_method", ""))
        return display_preprocessing_label(source), 0.0
    if block == "internal_circuit":
        power = float(manifest["internal_circuit_power_mw"])
        return f"{power:g} mW", power
    if block == "rastering":
        mode = str(manifest.get("raster_mode_normalized", manifest.get("raster_mode", "")))
        order = {"none": 0.0, "checkerboard": 1.0, "random": 2.0, "pseudo_random": 2.0}
        return mode.replace("_", " "), order.get(mode, 99.0)
    return str(manifest.get("run_id", "")), 0.0


def collect_metrics(npz_path: Path, limits: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    with np.load(npz_path, allow_pickle=True) as data:
        electrode_count = max(1, int(np.asarray(data["electrode_grid_ids"]).size)) if "electrode_grid_ids" in data.files else 1
        amplitude_key = "amplitude_per_electrode_uA" if "amplitude_per_electrode_uA" in data.files else "current_amplitude_per_electrode_uA"
        peak_charge = max_value(data, "charge_per_phase_per_electrode_nC", math.nan)
        if not np.isfinite(peak_charge):
            peak_charge = metric_or_peak(data, "peak_charge_per_phase_nC_exact", "charge_per_phase_mean_nC")
        peak_shannon = max_value(data, "shannon_k_per_electrode", math.nan)
        if not np.isfinite(peak_shannon):
            peak_shannon = metric_or_peak(data, "peak_shannon_k_exact", "shannon_k_mean")
        peak_current = max_value(data, amplitude_key, math.nan)
        if not np.isfinite(peak_current):
            peak_current = metric_or_peak(data, "peak_current_amplitude_uA_exact", "current_amplitude_per_electrode_uA")
        max_dT = metric_or_peak(data, "peak_max_dT_C", "max_dT")
        window_charge_electrode = max_value(data, "window_charge_per_electrode_nC")
        window_charge_total = max_value(data, "window_charge_total_nC")
        if amplitude_key in data.files:
            amplitude = np.asarray(data[amplitude_key], dtype=np.float64)
            if amplitude.ndim == 2:
                active_count = float(np.nanmax(np.sum(amplitude > 0.0, axis=1))) if amplitude.size else 0.0
            else:
                active_count = float(np.count_nonzero(amplitude > 0.0))
        else:
            active_count = max_value(data, "active_count")
        active_pct = 100.0 * active_count / float(electrode_count)
        if "charge_per_second_per_electrode_nC_s" in data.files:
            charge_rate = np.asarray(data["charge_per_second_per_electrode_nC_s"], dtype=np.float64)
            total_charge_rate = float(np.nanmax(np.sum(charge_rate, axis=1))) / 1e6 if charge_rate.ndim == 2 else max_value(data, "charge_per_second_per_electrode_nC_s") / 1e6
        else:
            total_charge_rate = max_value(data, "charge_per_second_total_nC_s") / 1e6
        final_charge = final_protocol_charge_nC(data)
        if final_charge is not None:
            final_charge_total = float(np.nansum(final_charge)) / 1e6
        else:
            final_charge_total = max_value(data, "protocol_charge_total_nC") / 1e6
        mean_shannon_series = mean_metric_series(data, "shannon", max_points=MAX_SERIES_POINTS)
        mean_shannon = finite_nanmean(mean_shannon_series[1]) if mean_shannon_series is not None else math.nan
        charge_rate_per_electrode, charge_rate_total = mean_charge_rate_values(data)
        max_mean_dT, max_mean_dT_time_s, _ = peak_value_and_time(data, "mean_dT")
        max_max_dT, max_focal_dT_time_s, _ = peak_value_and_time(data, "max_dT")
        cem43 = max_value(data, "max_cem43", 0.0)

    ratios = {
        "charge_per_phase": safe_ratio(peak_charge, limits["charge_per_phase_nC"]),
        "window_charge_per_electrode": safe_ratio(window_charge_electrode, limits["window_charge_per_electrode_nC"]),
        "window_charge_total": safe_ratio(window_charge_total, limits["window_charge_total_nC"]),
        "active_percentage": safe_ratio(active_pct, limits["simultaneous_activation_pct"]),
        "temperature": safe_ratio(max_dT, limits["temperature_increase_C"]),
        "cem43": safe_ratio(cem43, limits["cem43_min"]),
    }
    finite_ratios = [value for value in ratios.values() if np.isfinite(value)]
    safety_margin = max(finite_ratios) if finite_ratios else math.nan
    metrics = {
        "safety_margin": safety_margin,
        "peak_current_uA": peak_current,
        "peak_charge_per_phase_nC": peak_charge,
        "peak_shannon_k": peak_shannon,
        "peak_active_electrodes": active_count,
        "max_active_pct": active_pct,
        "max_total_charge_rate_mC_s": total_charge_rate,
        "final_total_charge_mC": final_charge_total,
        "max_window_charge_per_electrode_nC": window_charge_electrode,
        "max_window_charge_total_nC": window_charge_total,
        "max_dT_C": max_dT,
        "max_mean_dT_C": max_mean_dT,
        "max_mean_dT_time_s": max_mean_dT_time_s,
        "max_max_dT_C": max_max_dT,
        "max_focal_dT_time_s": max_focal_dT_time_s,
        "max_cem43_min": cem43,
        "mean_shannon_k": mean_shannon,
        "mean_charge_per_second_per_electrode_uC_s": charge_rate_per_electrode,
        "mean_charge_per_second_total_mC_s": charge_rate_total,
    }
    return metrics, ratios


def discover_records(input_root: Path, safety_yaml: Path) -> list[MatrixRecord]:
    limits = load_safety_limits(safety_yaml)
    set_plot_safety_limits(limits)
    records: list[MatrixRecord] = []
    for run in discover_completed_runs(input_root):
        npz_path = run.metrics_path
        manifest_path = run.manifest_path
        manifest = run.manifest
        block = str(manifest.get("block", npz_path.parent.parent.name))
        label, sort_value = block_label_and_sort(block, manifest)
        metrics, ratios = collect_metrics(npz_path, limits)
        records.append(
            MatrixRecord(
                npz_path=npz_path,
                manifest_path=manifest_path,
                block=block,
                run_id=str(manifest.get("run_id", npz_path.parent.name)),
                label=label,
                sort_value=sort_value,
                metrics=metrics,
                ratios=ratios,
            )
        )
    return sorted(records, key=lambda r: (block_index(r.block), r.sort_value, r.run_id))


def run_phase1_analysis(
    results_root: str | Path,
    output_root: str | Path | None = None,
    *,
    safety_yaml: str | Path | None = None,
    image_format: str = "png",
    overwrite: bool = True,
) -> list[Path]:
    input_root = resolve_repo_path(results_root)
    output = (
        resolve_repo_path(output_root)
        if output_root is not None
        else input_root / "comparative_visuals"
    )
    limits_path = package_file("safety") if safety_yaml is None else resolve_repo_path(safety_yaml)
    records = [
        record
        for record in discover_records(input_root, limits_path)
        if record.block == MATRIX_BLOCK
    ]
    if not records:
        raise RuntimeError(f"No completed phase 1 metrics.npz runs found under: {input_root}")
    output.mkdir(parents=True, exist_ok=True)
    before = set(output.rglob("*"))
    write_summary_csv(records, output)
    plot_comparative_suites(
        records,
        output,
        image_format,
        overwrite=overwrite,
    )
    return sorted(path for path in output.rglob("*") if path.is_file() and path not in before)


def block_index(block: str) -> int:
    try:
        return BLOCK_ORDER.index(block)
    except ValueError:
        return len(BLOCK_ORDER)


def ensure_output(path: Path, *, overwrite: bool) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return False
    return True


def save_figure(fig: plt.Figure, path: Path, *, overwrite: bool) -> None:
    if ensure_output(path, overwrite=overwrite):
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"wrote: {path}")
    plt.close(fig)


def data_files(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> tuple[str, ...]:
    if hasattr(data, "files"):
        return tuple(str(key) for key in data.files)
    return tuple(str(key) for key in data.keys())


def has_data_key(data: np.lib.npyio.NpzFile | Mapping[str, Any], key: str) -> bool:
    return key in data_files(data)


def get_data_array(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    key: str,
    *,
    dtype: Any | None = None,
) -> np.ndarray | None:
    if not has_data_key(data, key):
        return None
    arr = np.asarray(data[key])
    if dtype is not None and np.issubdtype(arr.dtype, np.number):
        arr = arr.astype(dtype, copy=False)
    return arr


def scalar_from_data(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    key: str,
    default: float = math.nan,
) -> float:
    arr = get_data_array(data, key)
    if arr is None or arr.size == 0:
        return float(default)
    try:
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return float(default)


def text_from_data(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    key: str,
    default: str = "",
) -> str:
    arr = get_data_array(data, key)
    if arr is None or arr.size == 0:
        return default
    value = arr.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def digits_from_text(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def normalize_preprocessing_label(value: object) -> str:
    text = str(value).strip().lower()
    aliases = {
        "groundtruth": "gt",
        "sanpo_groundtruth": "gt",
        "sanpo_gt": "gt",
        "difference_of_gaussians": "dog",
        "difference-of-gaussians": "dog",
    }
    return aliases.get(text, text or "unknown")


def display_preprocessing_label(value: object) -> str:
    normalized = normalize_preprocessing_label(value)
    return {
        "dog": "DoG",
        "gt": "Hand Segmented",
    }.get(normalized, normalized.capitalize())


def video_end_time_minutes(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    cooldown_start_s = scalar_from_data(data, "cooldown_start_s", math.nan)
    if np.isfinite(cooldown_start_s) and cooldown_start_s >= 0.0:
        return cooldown_start_s / 60.0
    for key in ("electrode_time_s", "time_s"):
        values = get_data_array(data, key, dtype=np.float64)
        if values is None:
            continue
        finite = values.reshape(-1)
        finite = finite[np.isfinite(finite)]
        if finite.size:
            return float(finite[-1]) / 60.0
    return math.nan


def add_video_end_line(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    *,
    label: str = "video end",
) -> bool:
    video_end_min = video_end_time_minutes(data)
    if not np.isfinite(video_end_min):
        return False
    ax.axvline(
        video_end_min,
        color="#4B5563",
        linestyle="--",
        linewidth=1.1,
        alpha=0.85,
        label=label,
    )
    return True


def electrode_count_from_data(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> int:
    for key in ("electrode_grid_ids", "electrode_xy_mm", "protocol_charge_per_electrode_nC"):
        arr = get_data_array(data, key)
        if arr is None or arr.size == 0:
            continue
        if key == "electrode_xy_mm":
            return int(np.asarray(arr).reshape(-1, 2).shape[0])
        if key == "protocol_charge_per_electrode_nC" and np.asarray(arr).ndim >= 2:
            return int(np.asarray(arr).reshape(np.asarray(arr).shape[0], -1).shape[1])
        return int(np.asarray(arr).reshape(-1).shape[0])
    amp = amplitude_matrix(data)
    if amp is not None and amp.ndim == 2:
        return int(amp.shape[1])
    return 0


def factors_from_manifest_and_data(
    manifest: dict,
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
) -> MatrixFactors:
    amplitude = float(manifest.get("amplitude_uA", math.nan))
    amplitude_key = f"amp_{amplitude:g}uA" if np.isfinite(amplitude) else "amp_unknown"
    amplitude_label = f"{amplitude:g} uA" if np.isfinite(amplitude) else "unknown amp"

    coords = Path(str(manifest.get("coords_yaml", ""))).stem
    grid_digits = digits_from_text(coords)
    grid_um = int(grid_digits) if grid_digits else 0
    grid_key = f"{grid_um}um" if grid_um else sanitize_path_part(coords or "unknown_grid")
    count = electrode_count_from_data(data)
    grid_label = f"{grid_um} um (N={count})" if grid_um and count else f"{grid_key} (N={count})"

    source = manifest.get("source_input_label", manifest.get("preprocessing_method", "unknown"))
    preprocessing_key = normalize_preprocessing_label(source)
    preprocessing_label = display_preprocessing_label(preprocessing_key)

    return MatrixFactors(
        amplitude=amplitude,
        amplitude_key=amplitude_key,
        amplitude_label=amplitude_label,
        grid_um=grid_um,
        grid_key=grid_key,
        grid_label=grid_label,
        preprocessing_key=preprocessing_key,
        preprocessing_label=preprocessing_label,
        electrode_count=count,
    )


def matrix_factors_for_record(record: MatrixRecord) -> MatrixFactors:
    manifest = load_yaml(record.manifest_path) if record.manifest_path.exists() else {}
    with np.load(record.npz_path, allow_pickle=True) as data:
        return factors_from_manifest_and_data(manifest, data)


def factor_value(factors: MatrixFactors, factor: str) -> FactorValue:
    if factor == "amplitude":
        return FactorValue(factors.amplitude_key, factors.amplitude_label, factors.amplitude)
    if factor == "grid":
        return FactorValue(factors.grid_key, factors.grid_label, float(factors.grid_um))
    if factor == "preprocessing":
        order = {"dog": 0.0, "canny": 1.0, "gt": 2.0}
        return FactorValue(
            factors.preprocessing_key,
            factors.preprocessing_label,
            order.get(factors.preprocessing_key, 99.0),
        )
    raise ValueError(f"Unknown matrix factor: {factor}")


def matrix_factor_maps(records: list[MatrixRecord]) -> tuple[dict[str, MatrixFactors], dict[tuple[str, str, str], MatrixRecord]]:
    factors_by_run_id: dict[str, MatrixFactors] = {}
    records_by_key: dict[tuple[str, str, str], MatrixRecord] = {}
    for record in records:
        factors = matrix_factors_for_record(record)
        factors_by_run_id[record.run_id] = factors
        records_by_key[(factors.amplitude_key, factors.grid_key, factors.preprocessing_key)] = record
    return factors_by_run_id, records_by_key


def sorted_factor_values(factors_by_run_id: Mapping[str, MatrixFactors], factor: str) -> list[FactorValue]:
    values: dict[str, FactorValue] = {}
    for factors in factors_by_run_id.values():
        value = factor_value(factors, factor)
        values.setdefault(value.key, value)
    return sorted(values.values(), key=lambda item: (item.sort_value, item.label))


def record_key_for_factors(values: Mapping[str, FactorValue]) -> tuple[str, str, str]:
    return (
        values["amplitude"].key,
        values["grid"].key,
        values["preprocessing"].key,
    )


def finite_nanmax(values: Any, default: float = math.nan) -> float:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return float(default)
    if arr.size == 0:
        return float(default)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.max(finite))


def finite_nanmean(values: Any, default: float = math.nan) -> float:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        return float(default)
    if arr.size == 0:
        return float(default)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def sample_indices(length: int, max_points: int = MAX_SERIES_POINTS) -> np.ndarray:
    length = int(length)
    if length <= 0:
        return np.asarray([], dtype=np.int64)
    if length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, int(max_points)).round().astype(np.int64))


def time_window_indices(length: int, max_points: int) -> list[np.ndarray]:
    length = int(length)
    if length <= 0:
        return []
    if length <= int(max_points):
        return [np.asarray([idx], dtype=np.int64) for idx in range(length)]
    return [
        chunk.astype(np.int64, copy=False)
        for chunk in np.array_split(np.arange(length, dtype=np.int64), int(max_points))
        if chunk.size > 0
    ]


def finite_mean(values: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    sums = np.sum(np.where(finite, arr, 0.0), axis=axis)
    counts = np.sum(finite, axis=axis)
    return np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def downsample_xy(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int = MAX_SERIES_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    length = min(x.size, y.size)
    if length <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    chunks = time_window_indices(length, max_points)
    if not chunks:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    x_out = np.asarray([finite_nanmean(x[:length][idx]) for idx in chunks], dtype=np.float64)
    y_out = np.asarray([finite_nanmean(y[:length][idx]) for idx in chunks], dtype=np.float64)
    return x_out, y_out


def downsample_time_rows(
    x: np.ndarray,
    rows: np.ndarray,
    max_points: int = MAX_CLOUD_POINTS,
    *,
    reducer: str = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    length = min(x.size, values.shape[0])
    if length <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float32)
    if reducer == "sample":
        idx = sample_indices(length, max_points=max_points)
        return x[:length][idx], values[:length][idx].astype(np.float32, copy=False)
    if reducer != "mean":
        raise ValueError(f"Unsupported row downsample reducer: {reducer!r}")
    chunks = time_window_indices(length, max_points)
    x_out = np.asarray([finite_nanmean(x[:length][idx]) for idx in chunks], dtype=np.float64)
    row_out = np.vstack([finite_mean(values[:length][idx], axis=0) for idx in chunks])
    return x_out, row_out.astype(np.float32, copy=False)


def raw_time_axis_seconds_for_length(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    length: int,
    *,
    prefer_electrode: bool = False,
) -> np.ndarray:
    candidates = (
        ("electrode_time_s", "time_s")
        if prefer_electrode
        else ("time_s", "thermal_time_s", "device_power_time_s", "electrode_time_s")
    )
    for key in candidates:
        arr = get_data_array(data, key, dtype=np.float64)
        if arr is not None:
            values = arr.reshape(-1)
            if values.size == length:
                return values
    return np.arange(int(length), dtype=np.float64)


def time_axis_for_length(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    length: int,
    *,
    prefer_electrode: bool = False,
) -> np.ndarray:
    return raw_time_axis_seconds_for_length(
        data,
        length,
        prefer_electrode=prefer_electrode,
    ) / 60.0


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
        color=FALLBACK_COLOR,
        fontsize=10,
    )
    style_axes(ax)


def as_time_electrode_matrix(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None or arr.size == 0:
        return None
    values = np.asarray(arr)
    if values.ndim == 1:
        return values.reshape(values.shape[0], 1)
    if values.ndim == 2:
        return values
    return values.reshape(values.shape[0], -1)


def first_matrix(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    keys: tuple[str, ...],
) -> np.ndarray | None:
    for key in keys:
        arr = as_time_electrode_matrix(get_data_array(data, key, dtype=np.float32))
        if arr is not None:
            return arr
    return None


def amplitude_matrix(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> np.ndarray | None:
    return first_matrix(
        data,
        (
            "amplitude_per_electrode_uA",
            "current_amplitude_per_electrode_uA",
        ),
    )


def electrode_values(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    key: str,
    n_electrodes: int,
    default: float,
) -> np.ndarray:
    arr = get_data_array(data, key, dtype=np.float32)
    if arr is None or arr.size == 0:
        return np.full(n_electrodes, float(default), dtype=np.float32)
    values = arr.reshape(-1).astype(np.float32, copy=False)
    if values.size == 1:
        return np.full(n_electrodes, float(values[0]), dtype=np.float32)
    if values.size == n_electrodes:
        return values
    return np.resize(values, n_electrodes).astype(np.float32, copy=False)


def charge_phase_from_amplitude(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    amplitude_uA: np.ndarray,
) -> np.ndarray:
    n_electrodes = amplitude_uA.shape[1]
    pulse_width_s = electrode_values(data, "pulse_width_s", n_electrodes, 170e-6)
    return amplitude_uA.astype(np.float32, copy=False) * pulse_width_s.reshape(1, -1) * 1e3


def charge_density_from_phase(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    charge_phase_nC: np.ndarray,
) -> np.ndarray:
    area_cm2 = scalar_from_data(data, "electrode_surface_area_cm2", math.nan)
    if not np.isfinite(area_cm2) or area_cm2 <= 0.0:
        return np.zeros_like(charge_phase_nC, dtype=np.float32)
    return charge_phase_nC.astype(np.float32, copy=False) / (1e3 * float(area_cm2))


def shannon_from_phase(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    charge_phase_nC: np.ndarray,
) -> np.ndarray:
    density = charge_density_from_phase(data, charge_phase_nC)
    charge_phase_uC = charge_phase_nC.astype(np.float32, copy=False) / 1e3
    out = np.full_like(charge_phase_uC, -np.inf, dtype=np.float32)
    valid = (charge_phase_uC > 0.0) & (density > 0.0)
    out[valid] = np.log10(charge_phase_uC[valid]) + np.log10(density[valid])
    return out


def charge_rate_from_amplitude(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    amplitude_uA: np.ndarray,
) -> np.ndarray:
    n_electrodes = amplitude_uA.shape[1]
    pulse_width_s = electrode_values(data, "pulse_width_s", n_electrodes, 170e-6)
    frequency_hz = electrode_values(data, "pulse_frequency_hz", n_electrodes, 300.0)
    rel = scalar_from_data(data, "relative_stim_duration", 1.0)
    if not np.isfinite(rel):
        rel = 1.0
    return (
        2.0
        * amplitude_uA.astype(np.float32, copy=False)
        * pulse_width_s.reshape(1, -1)
        * frequency_hz.reshape(1, -1)
        * float(rel)
        * 1e3
    )


METRIC_DIRECT_KEYS = {
    "amplitude": ("amplitude_per_electrode_uA", "current_amplitude_per_electrode_uA"),
    "charge_phase": ("charge_per_phase_per_electrode_nC", "charge_per_phase_nC"),
    "charge_density": ("charge_density_per_electrode_uc_cm2",),
    "shannon": ("shannon_k_per_electrode",),
    "charge_rate": ("charge_per_second_per_electrode_nC_s",),
}


def full_metric_rows(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    metric: str,
) -> np.ndarray | None:
    direct = first_matrix(data, METRIC_DIRECT_KEYS.get(metric, ()))
    if direct is not None:
        return direct

    amplitude = amplitude_matrix(data)
    if amplitude is None or metric == "amplitude":
        return amplitude
    charge_phase = charge_phase_from_amplitude(data, amplitude)
    if metric == "charge_phase":
        return charge_phase
    if metric == "charge_density":
        return charge_density_from_phase(data, charge_phase)
    if metric == "shannon":
        return shannon_from_phase(data, charge_phase)
    if metric == "charge_rate":
        return charge_rate_from_amplitude(data, amplitude)
    return None


def active_electrode_mean_series(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    metric: str,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    rows = full_metric_rows(data, metric)
    if rows is None:
        return None

    values = np.asarray(rows, dtype=np.float64)
    amplitude = amplitude_matrix(data)
    if amplitude is not None and amplitude.shape == values.shape:
        active = np.isfinite(amplitude) & (amplitude > 0.0)
    elif metric == "shannon":
        active = np.isfinite(values)
    else:
        active = np.isfinite(values) & (values > 0.0)
    valid = np.isfinite(values) & active
    count = np.sum(valid, axis=1)
    mean = np.divide(
        np.sum(np.where(valid, values, 0.0), axis=1),
        count,
        out=np.full(values.shape[0], np.nan, dtype=np.float64),
        where=count > 0,
    )
    x = time_axis_for_length(data, values.shape[0], prefer_electrode=True)
    return downsample_xy(x, mean, max_points=max_points)


def metric_rows(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    metric: str,
    *,
    max_points: int = MAX_CLOUD_POINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    rows = full_metric_rows(data, metric)
    if rows is not None:
        x_full = time_axis_for_length(data, rows.shape[0], prefer_electrode=True)
        reducer = "sample" if metric == "amplitude" else "mean"
        return downsample_time_rows(x_full, rows, max_points=max_points, reducer=reducer)
    return None


def series_from_key(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    key: str,
    *,
    max_points: int = MAX_SERIES_POINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    arr = get_data_array(data, key, dtype=np.float64)
    if arr is None or arr.size == 0:
        return None
    values = arr.reshape(-1)
    x = time_axis_for_length(data, values.size)
    return downsample_xy(x, values, max_points=max_points)


def distribution_stats(
    rows: np.ndarray,
    *,
    positive_only: bool = True,
    empty_value: float | None = None,
) -> dict[str, np.ndarray]:
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    n_rows = values.shape[0]
    stats = {
        "mean": np.full(n_rows, np.nan, dtype=np.float64),
        "p05": np.full(n_rows, np.nan, dtype=np.float64),
        "p25": np.full(n_rows, np.nan, dtype=np.float64),
        "p50": np.full(n_rows, np.nan, dtype=np.float64),
        "p75": np.full(n_rows, np.nan, dtype=np.float64),
        "p95": np.full(n_rows, np.nan, dtype=np.float64),
        "count": np.zeros(n_rows, dtype=np.float64),
    }
    for idx, row in enumerate(values):
        mask = np.isfinite(row)
        if positive_only:
            mask &= row > 0.0
        valid = row[mask]
        if valid.size == 0:
            if empty_value is not None:
                value = float(empty_value)
                stats["mean"][idx] = value
                stats["p05"][idx] = value
                stats["p25"][idx] = value
                stats["p50"][idx] = value
                stats["p75"][idx] = value
                stats["p95"][idx] = value
            continue
        stats["mean"][idx] = float(np.mean(valid))
        p05, p25, p50, p75, p95 = np.percentile(valid, [5, 25, 50, 75, 95])
        stats["p05"][idx] = p05
        stats["p25"][idx] = p25
        stats["p50"][idx] = p50
        stats["p75"][idx] = p75
        stats["p95"][idx] = p95
        stats["count"][idx] = float(valid.size)
    return stats


def plot_metric_cloud(
    ax: plt.Axes,
    x: np.ndarray,
    rows: np.ndarray,
    *,
    label: str,
    color: str,
    positive_only: bool,
    use_cloud: bool,
    empty_value: float | None = None,
) -> np.ndarray:
    stats = distribution_stats(rows, positive_only=positive_only, empty_value=empty_value)
    mean = stats["mean"]
    if use_cloud:
        ax.fill_between(x, stats["p05"], stats["p95"], color=color, alpha=0.14, linewidth=0)
        ax.fill_between(x, stats["p25"], stats["p75"], color=color, alpha=0.24, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=1.9, label=f"{label} mean")
    else:
        ax.plot(x, mean, color=color, linewidth=1.9, label=label)
    return mean


def mean_metric_series(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    metric: str,
    *,
    max_points: int = MAX_SERIES_POINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    if metric in METRIC_DIRECT_KEYS:
        active_mean = active_electrode_mean_series(
            data,
            metric,
            max_points=max_points,
        )
        if active_mean is not None:
            return active_mean

    mean_keys = {
        "charge_phase": ("charge_per_phase_mean_nC",),
        "charge_density": ("charge_density_mean_uc_cm2",),
        "shannon": ("shannon_k_mean",),
        "charge_rate": ("charge_per_second_mean_per_electrode_nC_s",),
        "mean_dT": ("mean_dT",),
        "max_dT": ("max_dT",),
        "area_gt1": ("area_gt1_mm2",),
        "area_gt2": ("area_gt2_mm2",),
        "area_gt3": ("area_gt3_mm2",),
    }
    for key in mean_keys.get(metric, ()):
        series = series_from_key(data, key, max_points=max_points)
        if series is not None:
            return series

    rows_result = metric_rows(data, metric, max_points=max_points)
    if rows_result is None:
        return None
    x, rows = rows_result
    positive_only = metric not in {"shannon"}
    empty_value = 0.0 if metric == "amplitude" and positive_only else None
    stats = distribution_stats(rows, positive_only=positive_only, empty_value=empty_value)
    return x, stats["mean"]


def active_count_series(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    *,
    max_points: int = MAX_SERIES_POINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    for key in ("active_count", "stimulated_electrode_count"):
        series = series_from_key(data, key, max_points=max_points)
        if series is not None:
            return series

    amp = amplitude_matrix(data)
    if amp is None:
        return None
    x_full = time_axis_for_length(data, amp.shape[0], prefer_electrode=True)
    x, rows = downsample_time_rows(x_full, amp, max_points=max_points)
    count = np.sum(np.isfinite(rows) & (rows > 0.0), axis=1).astype(np.float64)
    return x, count


def total_metric_series(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    metric: str,
    *,
    max_points: int = MAX_SERIES_POINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    if metric == "charge_rate":
        for key in ("charge_per_second_total_nC_s",):
            series = series_from_key(data, key, max_points=max_points)
            if series is not None:
                return series

    rows_result = metric_rows(data, metric, max_points=max_points)
    if rows_result is not None:
        x, rows = rows_result
        mask = np.isfinite(rows)
        if metric != "shannon":
            mask &= rows > 0.0
        totals = np.sum(np.where(mask, rows, 0.0), axis=1, dtype=np.float64)
        return x, totals

    mean_series_result = mean_metric_series(data, metric, max_points=max_points)
    counts_result = active_count_series(data, max_points=max_points)
    if mean_series_result is None or counts_result is None:
        return None
    x_mean, y_mean = mean_series_result
    x_count, counts = counts_result
    length = min(y_mean.size, counts.size)
    if length == 0:
        return None
    return x_mean[:length], y_mean[:length] * np.interp(x_mean[:length], x_count, counts)


def frame_durations_s(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    length: int,
) -> np.ndarray:
    time_s = raw_time_axis_seconds_for_length(data, length)
    diffs = np.diff(time_s)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    fps = scalar_from_data(data, "fps", math.nan)
    fallback_dt = (1.0 / fps) if np.isfinite(fps) and fps > 0.0 else 1.0
    if positive_diffs.size:
        fallback_dt = float(np.median(positive_diffs))
    durations = np.full(length, fallback_dt, dtype=np.float64)
    if length > 1:
        durations[1:] = np.where(np.isfinite(diffs) & (diffs > 0.0), diffs, fallback_dt)
    if length > 0 and np.isfinite(time_s[0]) and time_s[0] > 0.0:
        durations[0] = float(time_s[0])
    return durations


def cumulative_charge_summaries(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    *,
    max_points: int = MAX_CLOUD_POINTS,
) -> dict[str, np.ndarray] | None:
    protocol_arr = get_data_array(data, "protocol_charge_per_electrode_nC", dtype=np.float32)
    protocol = as_time_electrode_matrix(protocol_arr) if protocol_arr is not None and protocol_arr.ndim >= 2 else None
    if protocol is not None and protocol.shape[0] > 1:
        idx = sample_indices(protocol.shape[0], max_points=max_points)
        x = time_axis_for_length(data, protocol.shape[0], prefer_electrode=True)[idx]
        rows = protocol[idx]
        stats = distribution_stats(rows, positive_only=True)
        stats["time_s"] = x
        stats["total"] = np.sum(np.where(np.isfinite(rows) & (rows > 0.0), rows, 0.0), axis=1)
        stats["final_per_electrode"] = protocol[-1].astype(np.float64, copy=False)
        return stats

    rate = first_matrix(data, ("charge_per_second_per_electrode_nC_s",))
    derive_from_amplitude = False
    if rate is None:
        rate = amplitude_matrix(data)
        derive_from_amplitude = rate is not None
    if rate is None:
        return None

    n_time, n_electrodes = rate.shape
    idx = sample_indices(n_time, max_points=max_points)
    time_s = time_axis_for_length(data, n_time)
    durations = frame_durations_s(data, n_time)
    cumulative = np.zeros(n_electrodes, dtype=np.float64)
    out_rows = []
    totals = []
    prev = 0
    for target in idx:
        block = rate[prev : target + 1].astype(np.float32, copy=False)
        if derive_from_amplitude:
            block = charge_rate_from_amplitude(data, block)
        dt_block = durations[prev : target + 1].reshape(-1, 1)
        cumulative += np.sum(block.astype(np.float64, copy=False) * dt_block, axis=0)
        out_rows.append(cumulative.copy())
        totals.append(float(np.sum(cumulative[np.isfinite(cumulative) & (cumulative > 0.0)])))
        prev = int(target) + 1

    rows = np.asarray(out_rows, dtype=np.float64)
    stats = distribution_stats(rows, positive_only=True)
    stats["time_s"] = time_s[idx]
    stats["total"] = np.asarray(totals, dtype=np.float64)
    stats["final_per_electrode"] = cumulative
    return stats


def final_protocol_charge_nC(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> np.ndarray | None:
    for key in ("protocol_charge_per_electrode_nC", "final_protocol_charge_per_electrode_nC"):
        arr = get_data_array(data, key, dtype=np.float64)
        if arr is None or arr.size == 0:
            continue
        if arr.ndim == 1:
            return arr.reshape(-1)
        return arr.reshape(arr.shape[0], -1)[-1]
    summaries = cumulative_charge_summaries(data, max_points=2)
    if summaries is None:
        return None
    return summaries["final_per_electrode"]


def protocol_total_charge_series_nC(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    *,
    max_points: int = MAX_SERIES_POINTS,
) -> tuple[np.ndarray, np.ndarray] | None:
    series = series_from_key(data, "protocol_charge_total_nC", max_points=max_points)
    if series is not None:
        return series
    summaries = cumulative_charge_summaries(data, max_points=max_points)
    if summaries is None:
        return None
    return summaries["time_s"], summaries["total"]


def manifest_for_record(record: MatrixRecord) -> dict:
    return load_yaml(record.manifest_path) if record.manifest_path.exists() else {}


def should_use_cloud_for_case(manifest: dict) -> bool:
    text = " ".join(
        str(manifest.get(key, ""))
        for key in ("preprocessing_method", "source_input_label", "video")
    ).lower()
    return any(token in text for token in CLOUD_PREPROCESSING_TOKENS)


def case_visuals_dir(record: MatrixRecord) -> Path:
    return record.npz_path.parent


def protocol_color(record: MatrixRecord) -> str:
    return COLORS.get(record.block, COLORS["amplitude"])


def electrode_xy_for_charge_map(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    manifest: Mapping[str, Any] | None,
    n_electrodes: int,
) -> np.ndarray | None:
    xy = get_data_array(data, "electrode_xy_mm", dtype=np.float64)
    if xy is None:
        return None
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] != n_electrodes:
        return None
    return xy


def format_mC(value_nC: float) -> str:
    value_mC = float(value_nC) / 1e6
    return f"{value_mC:.3g} mC"


def format_uC(value_nC: float) -> str:
    value_uC = float(value_nC) / 1e3
    if abs(value_uC) >= 1e3:
        return f"{value_uC / 1e3:.3g} mC"
    return f"{value_uC:.3g} uC"


def format_summary_number(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    abs_value = abs(float(value))
    if abs_value >= 1000.0:
        return f"{value:.3g}"
    if abs_value >= 100.0:
        return f"{value:.1f}"
    if abs_value >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3g}"


def finite_summary_values(values: Any, *, positive_only: bool = False) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=np.float64)
    mask = np.isfinite(arr)
    if positive_only:
        mask &= arr > 0.0
    return arr[mask]


def annotate_mean_max(
    ax: plt.Axes,
    *,
    mean_value: float,
    max_value: float,
    mean_label: str = "mean",
    max_label: str = "max",
    unit: str = "",
    above_axes: bool = False,
) -> None:
    suffix = f" {unit}" if unit else ""
    ax.text(
        0.02,
        1.02 if above_axes else 0.98,
        f"{mean_label} {format_summary_number(mean_value)}{suffix}\n"
        f"{max_label} {format_summary_number(max_value)}{suffix}",
        transform=ax.transAxes,
        ha="left",
        va="bottom" if above_axes else "top",
        fontsize=8.5,
        color="#111827",
        clip_on=False,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )


def records_by_block(records: Iterable[MatrixRecord], block: str) -> list[MatrixRecord]:
    return sorted((record for record in records if record.block == block), key=lambda r: (r.sort_value, r.run_id))


def plot_protocol_metric_distribution(
    record: MatrixRecord,
    *,
    metric: str,
    title: str,
    ylabel: str,
    filename: str,
    image_format: str,
    overwrite: bool,
    positive_only: bool = True,
) -> None:
    manifest = manifest_for_record(record)
    out_path = case_visuals_dir(record) / f"{filename}.{image_format}"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing: {out_path}")
        return

    with np.load(record.npz_path, allow_pickle=True) as data:
        color = protocol_color(record)
        rows_result = metric_rows(data, metric, max_points=PER_PROTOCOL_SERIES_POINTS)
        fig, ax = plt.subplots(figsize=(8.8, 4.4))
        summary_mean = math.nan
        summary_max = math.nan
        if rows_result is None:
            series = mean_metric_series(data, metric, max_points=PER_PROTOCOL_SERIES_POINTS)
            if series is None:
                add_no_data(ax, f"No {title.lower()}")
            else:
                x, y = series
                ax.plot(x, y, color=color, linewidth=1.9, label=record.label)
                if metric not in {"amplitude", "shannon"}:
                    ax.legend(frameon=False, fontsize=8)
                style_axes(ax)
                valid = finite_summary_values(y, positive_only=False)
                summary_mean = finite_nanmean(valid)
                summary_max = finite_nanmax(valid)
        else:
            x, rows = rows_result
            mean_trace = plot_metric_cloud(
                ax,
                x,
                rows,
                label=record.label.replace("\n", " "),
                color=color,
                positive_only=positive_only,
                use_cloud=should_use_cloud_for_case(manifest),
                empty_value=0.0 if metric == "amplitude" and positive_only else None,
            )
            if metric not in {"amplitude", "shannon"}:
                ax.legend(frameon=False, fontsize=8)
            style_axes(ax)
            row_values = finite_summary_values(rows, positive_only=positive_only)
            summary_mean = finite_nanmean(mean_trace)
            summary_max = finite_nanmax(row_values)
        if np.isfinite(summary_mean) or np.isfinite(summary_max):
            if metric == "shannon":
                annotate_mean_max(ax, mean_value=summary_mean, max_value=summary_max, mean_label="mean K", max_label="max K")
            elif metric == "amplitude":
                annotate_mean_max(ax, mean_value=summary_mean, max_value=summary_max, mean_label="mean", max_label="max", unit="uA")
            else:
                annotate_mean_max(ax, mean_value=summary_mean, max_value=summary_max)
        add_video_end_line(ax, data)
        ax.set_title(title)
        ax.set_xlabel(TIME_AXIS_LABEL)
        ax.set_ylabel(ylabel)
        save_figure(fig, out_path, overwrite=overwrite)


def plot_protocol_activated_electrodes(
    record: MatrixRecord,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    out_path = case_visuals_dir(record) / f"activated_electrodes_over_time.{image_format}"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing: {out_path}")
        return
    with np.load(record.npz_path, allow_pickle=True) as data:
        fig, ax = plt.subplots(figsize=(8.8, 4.4))
        plotted = False
        active = series_from_key(data, "active_count", max_points=PER_PROTOCOL_SERIES_POINTS)
        if active is None:
            active = active_count_series(data, max_points=PER_PROTOCOL_SERIES_POINTS)
        if active is not None:
            ax.plot(active[0], active[1], color=protocol_color(record), linewidth=1.9, label="active")
            annotate_mean_max(
                ax,
                mean_value=finite_nanmean(active[1]),
                max_value=finite_nanmax(active[1]),
                mean_label="mean active",
                max_label="max active",
                above_axes=True,
            )
            plotted = True
        stimulated = series_from_key(data, "stimulated_electrode_count", max_points=PER_PROTOCOL_SERIES_POINTS)
        if stimulated is not None:
            ax.plot(stimulated[0], stimulated[1], color="#C05621", linewidth=1.4, label="stimulated")
            plotted = True
        add_video_end_line(ax, data)
        if plotted:
            style_axes(ax)
        else:
            add_no_data(ax, "No electrode counts")
        ax.set_title("Activated Electrodes Over Time")
        ax.set_xlabel(TIME_AXIS_LABEL)
        ax.set_ylabel("Electrodes")
        save_figure(fig, out_path, overwrite=overwrite)


def plot_protocol_phase_or_density(
    record: MatrixRecord,
    *,
    metric: str,
    title: str,
    per_electrode_ylabel: str,
    total_ylabel: str,
    filename: str,
    image_format: str,
    overwrite: bool,
) -> None:
    manifest = manifest_for_record(record)
    out_path = case_visuals_dir(record) / f"{filename}.{image_format}"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing: {out_path}")
        return
    with np.load(record.npz_path, allow_pickle=True) as data:
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
        rows_result = metric_rows(data, metric, max_points=MAX_CLOUD_POINTS)
        if rows_result is None:
            series = mean_metric_series(data, metric, max_points=MAX_SERIES_POINTS)
            if series is None:
                add_no_data(axes[0], f"No {title.lower()}")
            else:
                axes[0].plot(series[0], series[1], color=COLORS.get(record.block, FALLBACK_COLOR), linewidth=1.9)
                style_axes(axes[0])
        else:
            x, rows = rows_result
            plot_metric_cloud(
                axes[0],
                x,
                rows,
                label="active electrodes",
                color=COLORS.get(record.block, FALLBACK_COLOR),
                positive_only=True,
                use_cloud=should_use_cloud_for_case(manifest),
            )
            axes[0].legend(frameon=False, fontsize=8)
            style_axes(axes[0])
        axes[0].set_title("Per-electrode")
        axes[0].set_xlabel(TIME_AXIS_LABEL)
        axes[0].set_ylabel(per_electrode_ylabel)

        total = total_metric_series(data, metric, max_points=MAX_SERIES_POINTS)
        if total is None:
            add_no_data(axes[1], f"No total {title.lower()}")
        else:
            axes[1].plot(total[0], total[1], color="#8B1E3F", linewidth=1.9)
            style_axes(axes[1])
        axes[1].set_title("Summed across electrodes")
        add_video_end_line(axes[0], data)
        add_video_end_line(axes[1], data)
        axes[1].set_xlabel(TIME_AXIS_LABEL)
        axes[1].set_ylabel(total_ylabel)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.01, 1, 0.94])
        save_figure(fig, out_path, overwrite=overwrite)


def plot_protocol_charge_per_second(
    record: MatrixRecord,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    manifest = manifest_for_record(record)
    out_path = case_visuals_dir(record) / f"charge_per_second_over_time.{image_format}"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing: {out_path}")
        return
    with np.load(record.npz_path, allow_pickle=True) as data:
        fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.6))
        color = protocol_color(record)

        rows_result = metric_rows(data, "charge_rate", max_points=PER_PROTOCOL_SERIES_POINTS)
        per_e_limit = plot_safety_limit("window_charge_per_electrode_nC") / 1e3
        total_limit = plot_safety_limit("window_charge_total_nC") / 1e6
        if rows_result is None:
            series = mean_metric_series(data, "charge_rate", max_points=PER_PROTOCOL_SERIES_POINTS)
            if series is None:
                add_no_data(axes[0], "No charge rate")
            else:
                y_uC_s = series[1] / 1e3
                axes[0].plot(series[0], y_uC_s, color=color, linewidth=1.9)
                annotate_mean_max(
                    axes[0],
                    mean_value=finite_nanmean(y_uC_s),
                    max_value=finite_nanmax(y_uC_s),
                    mean_label="mean",
                    max_label="max",
                    unit="uC/s",
                    above_axes=True,
                )
                style_axes(axes[0])
        else:
            x, rows = rows_result
            rows_uC_s = rows / 1e3
            mean_trace = plot_metric_cloud(
                axes[0],
                x,
                rows_uC_s,
                label="_nolegend_",
                color=color,
                positive_only=True,
                use_cloud=should_use_cloud_for_case(manifest),
            )
            valid = finite_summary_values(rows_uC_s, positive_only=True)
            annotate_mean_max(
                axes[0],
                mean_value=finite_nanmean(mean_trace),
                max_value=finite_nanmax(valid),
                mean_label="mean",
                max_label="max",
                unit="uC/s",
                above_axes=True,
            )
            style_axes(axes[0])
        if add_limit_line(axes[0], per_e_limit, label="safety limit", linestyle=":"):
            axes[0].legend(frameon=False, fontsize=8)
        axes[0].set_title("Mean Across Electrodes", pad=42)
        axes[0].set_xlabel(TIME_AXIS_LABEL)
        axes[0].set_ylabel("Charge rate (uC/s/electrode)")

        total = total_metric_series(data, "charge_rate", max_points=PER_PROTOCOL_SERIES_POINTS)
        if total is None:
            add_no_data(axes[1], "No summed charge rate")
        else:
            x, y_nC_s = total
            y_mC_s = y_nC_s / 1e6
            axes[1].plot(x, y_mC_s, color="#8B1E3F", linewidth=1.9)
            annotate_mean_max(
                axes[1],
                mean_value=finite_nanmean(y_mC_s),
                max_value=finite_nanmax(y_mC_s),
                mean_label="mean",
                max_label="max",
                unit="mC/s",
                above_axes=True,
            )
            style_axes(axes[1])
        if add_limit_line(axes[1], total_limit, label="safety limit", linestyle=":"):
            axes[1].legend(frameon=False, fontsize=8)
        add_video_end_line(axes[0], data)
        add_video_end_line(axes[1], data)
        axes[1].set_title("Summed Across Electrodes", pad=42)
        axes[1].set_xlabel(TIME_AXIS_LABEL)
        axes[1].set_ylabel("Charge rate (mC/s)")

        fig.suptitle("Charge Per Second Over Time", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.01, 1, 0.94])
        save_figure(fig, out_path, overwrite=overwrite)


def plot_electrode_charge_heatmap(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    charge_nC: np.ndarray,
    *,
    manifest: Mapping[str, Any] | None = None,
    title: str,
    vmax_mC: float | None = None,
) -> Any | None:
    values_mC = np.asarray(charge_nC, dtype=np.float64).reshape(-1) / 1e6
    xy = electrode_xy_for_charge_map(data, manifest, values_mC.size)
    if xy is None:
        x = np.arange(values_mC.size, dtype=np.float64)
        y = np.zeros_like(x)
        xlabel = "electrode index"
        ylabel = ""
    else:
        x = xy[:, 0]
        y = xy[:, 1]
        xlabel = "x (mm)"
        ylabel = "y (mm)"

    finite = np.isfinite(values_mC)
    if not np.any(finite):
        add_no_data(ax, "No cumulative charge")
        ax.set_title(title)
        return None

    marker_size = float(np.clip(6000.0 / max(values_mC.size, 1), 1.5, 18.0))
    image = ax.scatter(
        x[finite],
        y[finite],
        c=values_mC[finite],
        s=marker_size,
        cmap="viridis",
        vmin=0.0,
        vmax=vmax_mC,
        linewidths=0,
    )
    total_nC = float(np.nansum(np.where(np.isfinite(charge_nC), charge_nC, 0.0)))
    ax.text(
        0.02,
        0.98,
        f"total {format_mC(total_nC)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#111827",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xy is not None and xy.shape[0] == values_mC.size:
        ax.set_aspect("equal", adjustable="datalim")
    style_axes(ax, grid_axis="both")
    return image


def plot_protocol_charge_over_protocol(
    record: MatrixRecord,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    manifest = manifest_for_record(record)
    out_path = case_visuals_dir(record) / f"charge_over_whole_protocol.{image_format}"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing: {out_path}")
        return
    with np.load(record.npz_path, allow_pickle=True) as data:
        final_charge = final_protocol_charge_nC(data)
        cumulative = cumulative_charge_summaries(data, max_points=MAX_CLOUD_POINTS)
        total_series = protocol_total_charge_series_nC(data, max_points=MAX_SERIES_POINTS)
        fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8))

        image = None
        if final_charge is None:
            add_no_data(axes[0], "No electrode charge")
            axes[0].set_title("Cumulative Electrode Charge")
        else:
            image = plot_electrode_charge_heatmap(
                axes[0],
                data,
                final_charge,
                manifest=manifest,
                title="Cumulative Electrode Charge",
            )
        if image is not None:
            cbar = fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.04)
            cbar.set_label("charge (mC)")
            cbar.ax.tick_params(labelsize=8)

        if total_series is None:
            add_no_data(axes[1], "No accumulated charge trace")
        else:
            axes[1].plot(total_series[0], total_series[1] / 1e6, color="#8B1E3F", linewidth=1.9)
            add_limit_line(axes[1], plot_safety_limit("session_charge_limit_mC"), label="_nolegend_")
            style_axes(axes[1])
        axes[1].set_title("Accumulated Charge")
        axes[1].set_xlabel(TIME_AXIS_LABEL)
        axes[1].set_ylabel("Total charge (mC)")

        if cumulative is None:
            add_no_data(axes[2], "No per-electrode accumulated charge")
        else:
            x = cumulative["time_s"]
            axes[2].fill_between(x, cumulative["p05"] / 1e6, cumulative["p95"] / 1e6, color="#1F4E79", alpha=0.14, linewidth=0)
            axes[2].fill_between(x, cumulative["p25"] / 1e6, cumulative["p75"] / 1e6, color="#1F4E79", alpha=0.24, linewidth=0)
            axes[2].plot(x, cumulative["mean"] / 1e6, color="#1F4E79", linewidth=1.9)
            style_axes(axes[2])
        axes[2].set_title("Accumulated Charge Per Electrode")
        add_video_end_line(axes[1], data)
        add_video_end_line(axes[2], data)
        axes[2].set_xlabel(TIME_AXIS_LABEL)
        axes[2].set_ylabel("Charge (mC/electrode)")
        fig.suptitle(f"Charge Over Whole Protocol - {record.label.replace(chr(10), ' ')}", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0.01, 1, 0.93])
        save_figure(fig, out_path, overwrite=overwrite)


def downsample_image(image: np.ndarray, *, max_dim: int = 350) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    step = max(1, int(math.ceil(max(arr.shape) / float(max_dim))))
    return arr[::step, ::step]


def thermal_snapshot_keys(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> list[tuple[str, str]]:
    names = get_data_array(data, "heatmap_grid_names")
    if names is not None and names.size:
        out: list[tuple[str, str]] = []
        for raw in names.reshape(-1):
            name = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            suffix = sanitize_path_part(name)
            key = f"dT_heatmaps_{suffix}"
            if has_data_key(data, key):
                out.append((name, key))
        if out:
            return out
    out = []
    for key in data_files(data):
        if key.startswith("dT_heatmaps_"):
            out.append((key.removeprefix("dT_heatmaps_"), key))
    return sorted(out)


def extent_for_grid(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    grid_name: str,
) -> list[float] | None:
    suffix = sanitize_path_part(grid_name)
    for key in (f"extent_mm_{suffix}", "extent_mm"):
        extent = get_data_array(data, key, dtype=np.float64)
        if extent is None:
            continue
        values = extent.reshape(-1)
        if values.size >= 4 and np.all(np.isfinite(values[:4])):
            return [float(values[0]), float(values[1]), float(values[2]), float(values[3])]
    return None


def peak_heatmap_grid_names(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
) -> list[str]:
    names: list[str] = []
    stored_names = get_data_array(data, "peak_heatmap_grid_names")
    if stored_names is not None:
        for raw_name in stored_names.reshape(-1):
            name = (
                raw_name.decode("utf-8", errors="replace")
                if isinstance(raw_name, bytes)
                else str(raw_name)
            )
            if name and name not in names:
                names.append(name)
    for grid_name, _ in thermal_snapshot_keys(data):
        if grid_name not in names:
            names.append(grid_name)
    if names:
        return names
    for key in data_files(data):
        for prefix in ("dT_peak_focal_", "dT_peak_mean_"):
            if key.startswith(prefix):
                name = key.removeprefix(prefix)
                if name and name not in names:
                    names.append(name)
    return names


def hemisphere_grid_sort_key(grid_name: str) -> tuple[int, str]:
    normalized = str(grid_name).strip().casefold()
    if "left" in normalized:
        return 0, normalized
    if "right" in normalized:
        return 1, normalized
    return 2, normalized


def peak_heatmap_for_grid(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    grid_name: str,
    peak_kind: str,
    peak_time_s: float,
) -> tuple[np.ndarray | None, float, bool]:
    suffix = sanitize_path_part(grid_name)
    exact = get_data_array(data, f"dT_peak_{peak_kind}_{suffix}", dtype=np.float32)
    if exact is not None and exact.ndim == 2 and exact.size:
        return (
            np.asarray(exact, dtype=np.float32),
            scalar_from_data(data, f"peak_{peak_kind}_time_s", peak_time_s),
            True,
        )

    stack = get_data_array(data, f"dT_heatmaps_{suffix}", dtype=np.float32)
    if stack is None or stack.ndim != 3 or stack.shape[0] == 0:
        return None, math.nan, False
    times = get_data_array(data, "heatmap_times_s", dtype=np.float64)
    snapshot_times = times.reshape(-1) if times is not None else np.asarray([])
    if (
        snapshot_times.size == stack.shape[0]
        and np.isfinite(peak_time_s)
        and np.any(np.isfinite(snapshot_times))
    ):
        finite_indices = np.flatnonzero(np.isfinite(snapshot_times))
        nearest = int(np.argmin(np.abs(snapshot_times[finite_indices] - peak_time_s)))
        index = int(finite_indices[nearest])
        map_time_s = float(snapshot_times[index])
    else:
        reducer = finite_nanmax if peak_kind == "focal" else finite_nanmean
        values = np.asarray(
            [reducer(heatmap, default=math.nan) for heatmap in stack],
            dtype=np.float64,
        )
        finite_indices = np.flatnonzero(np.isfinite(values))
        index = (
            int(finite_indices[int(np.argmax(values[finite_indices]))])
            if finite_indices.size
            else 0
        )
        map_time_s = (
            float(snapshot_times[index])
            if snapshot_times.size > index and np.isfinite(snapshot_times[index])
            else math.nan
        )
    return np.asarray(stack[index], dtype=np.float32), map_time_s, False


def format_peak_time(time_s: float) -> str:
    if not np.isfinite(time_s):
        return "time unavailable"
    return f"{time_s / 60.0:.2f} min ({time_s:.1f} s)"


def plot_peak_temperature_heatmaps(
    npz_path: str | Path,
    output_path: str | Path,
    *,
    title: str,
    overwrite: bool,
) -> Path | None:
    source = Path(npz_path)
    destination = Path(output_path)
    with np.load(source, allow_pickle=True) as data:
        grid_names = sorted(
            peak_heatmap_grid_names(data),
            key=hemisphere_grid_sort_key,
        )
        if not grid_names:
            return None

        peaks = {}
        for peak_kind, series_key in (("focal", "max_dT"), ("mean", "mean_dT")):
            value, time_s, _ = peak_value_and_time(data, series_key)
            if not np.isfinite(value):
                value = scalar_from_data(data, f"peak_{peak_kind}_dT_C")
            if not np.isfinite(time_s):
                time_s = scalar_from_data(data, f"peak_{peak_kind}_time_s")
            peaks[peak_kind] = (value, time_s)

        panels: dict[tuple[str, str], tuple[np.ndarray, float, bool]] = {}
        for grid_name in grid_names:
            for peak_kind in ("focal", "mean"):
                heatmap, map_time_s, exact = peak_heatmap_for_grid(
                    data,
                    grid_name,
                    peak_kind,
                    peaks[peak_kind][1],
                )
                if heatmap is not None:
                    panels[(grid_name, peak_kind)] = (heatmap, map_time_s, exact)
        if not panels:
            return None

        extents = {
            grid_name: extent_for_grid(data, grid_name)
            for grid_name in grid_names
        }

    vmax = finite_nanmax(
        np.concatenate([panel[0].reshape(-1) for panel in panels.values()]),
        default=1.0,
    )
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(
        2,
        len(grid_names),
        figsize=(max(5.6, 5.2 * len(grid_names)), 8.2),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for row, peak_kind in enumerate(("focal", "mean")):
        for col, grid_name in enumerate(grid_names):
            ax = axes[row, col]
            panel = panels.get((grid_name, peak_kind))
            if panel is None:
                add_no_data(ax, f"No {peak_kind} peak map")
                continue
            heatmap, map_time_s, exact = panel
            extent = extents[grid_name]
            displayed = downsample_image(heatmap)
            image = ax.imshow(
                displayed,
                origin="lower",
                extent=extent,
                cmap=THERMAL_CMAP,
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
            )
            value, peak_time_s = peaks[peak_kind]
            metric_name = "Max focal dT" if peak_kind == "focal" else "Max spatial mean dT"
            panel_title = (
                f"{metric_name}: {value:.4g} C\n"
                f"Peak time: {format_peak_time(peak_time_s)}"
            )
            map_matches_peak = (
                exact
                or (
                    np.isfinite(map_time_s)
                    and np.isfinite(peak_time_s)
                    and np.isclose(map_time_s, peak_time_s, rtol=0.0, atol=1e-4)
                )
            )
            if not map_matches_peak:
                panel_title += f"\nNearest saved map: {format_peak_time(map_time_s)}"
            if row == 0:
                panel_title = f"{grid_name}\n{panel_title}"
            ax.set_title(panel_title, fontsize=9)
            ax.set_xlabel("x (mm)" if extent is not None else "x pixel")
            ax.set_ylabel("y (mm)" if extent is not None else "y pixel")
            if peak_kind == "focal" and displayed.size:
                finite = np.isfinite(displayed)
                if np.any(finite):
                    peak_row, peak_col = np.unravel_index(
                        int(np.nanargmax(displayed)),
                        displayed.shape,
                    )
                    if extent is None:
                        marker_x, marker_y = float(peak_col), float(peak_row)
                    else:
                        xmin, xmax, ymin, ymax = extent
                        marker_x = xmin + (peak_col + 0.5) * (xmax - xmin) / displayed.shape[1]
                        marker_y = ymin + (peak_row + 0.5) * (ymax - ymin) / displayed.shape[0]
                    ax.plot(marker_x, marker_y, marker="x", color="cyan", markersize=7, mew=1.5)
            ax.tick_params(labelsize=8)

    if image is not None:
        cbar = fig.colorbar(image, ax=axes.reshape(-1).tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("dT (C)")
    fig.suptitle(
        textwrap.fill(title, width=88),
        fontsize=11.5,
        fontweight="bold",
    )
    save_figure(fig, destination, overwrite=overwrite)
    return destination


def write_peak_temperature_heatmaps(
    records: Iterable[Any],
    output_root: str | Path,
    *,
    image_format: str,
    overwrite: bool,
) -> list[Path]:
    out_dir = Path(output_root) / "peak_temperature_heatmaps"
    paths: list[Path] = []
    for record in records:
        run_id = str(record.run_id)
        path = out_dir / f"{sanitize_path_part(run_id)}.{image_format}"
        written = plot_peak_temperature_heatmaps(
            record.npz_path,
            path,
            title=f"Peak Temperature Maps - {run_id}",
            overwrite=overwrite,
        )
        if written is not None:
            paths.append(written)
    return paths


def thermal_snapshot_positions(length: int) -> list[tuple[str, int]]:
    length = int(length)
    if length <= 0:
        return []
    positions = [
        ("beginning", 0),
        ("middle", (length - 1) // 2),
        ("end", length - 1),
    ]
    unique: list[tuple[str, int]] = []
    seen: set[int] = set()
    for label, index in positions:
        index = max(0, min(index, length - 1))
        if index in seen:
            continue
        seen.add(index)
        unique.append((label, index))
    return unique


def middle_snapshot_index(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    length: int,
    *,
    fraction_keys: tuple[str, ...] = ("heatmap_actual_fractions", "heatmap_target_fractions"),
    time_key: str = "heatmap_times_s",
) -> int:
    length = int(length)
    if length <= 1:
        return 0

    for key in fraction_keys:
        fractions = get_data_array(data, key, dtype=np.float64)
        if fractions is None:
            continue
        values = fractions.reshape(-1)
        if values.size != length:
            continue
        finite = np.isfinite(values)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            nearest = int(np.argmin(np.abs(values[finite] - 0.5)))
            return int(finite_indices[nearest])

    times = get_data_array(data, time_key, dtype=np.float64)
    if times is not None:
        values = times.reshape(-1)
        if values.size == length:
            finite = np.isfinite(values)
            if np.any(finite):
                finite_indices = np.flatnonzero(finite)
                target = 0.5 * finite_nanmax(values[finite], default=0.0)
                nearest = int(np.argmin(np.abs(values[finite] - target)))
                return int(finite_indices[nearest])

    return int((length - 1) // 2)


def middle_snapshot_detail(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    index: int,
    *,
    fraction_keys: tuple[str, ...] = ("heatmap_actual_fractions", "heatmap_target_fractions"),
    time_key: str = "heatmap_times_s",
) -> str:
    parts: list[str] = []
    for key in fraction_keys:
        fractions = get_data_array(data, key, dtype=np.float64)
        if fractions is None:
            continue
        values = fractions.reshape(-1)
        if values.size > index and np.isfinite(values[index]):
            parts.append(f"{100.0 * float(values[index]):.0f}%")
            break

    times = get_data_array(data, time_key, dtype=np.float64)
    if times is not None:
        values = times.reshape(-1)
        if values.size > index and np.isfinite(values[index]):
            parts.append(f"{float(values[index]) / 60.0:.1f} min")
    return " | ".join(parts)


def thermal_volume_keys(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> list[tuple[str, str]]:
    names = get_data_array(data, "heatmap_grid_names")
    if names is not None and names.size:
        out: list[tuple[str, str]] = []
        for raw in names.reshape(-1):
            name = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            suffix = sanitize_path_part(name)
            key = f"dT_volume_final_{suffix}"
            if has_data_key(data, key):
                out.append((name, key))
        if out:
            return out

    out = []
    for key in data_files(data):
        if key.startswith("dT_volume_final_"):
            out.append((key.removeprefix("dT_volume_final_"), key))
    if out:
        return sorted(out)
    if has_data_key(data, "dT_volume_final"):
        return [("final", "dT_volume_final")]
    return []


def volume_extent_for_grid(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    grid_name: str,
) -> tuple[float, float, float, float, float, float] | None:
    suffix = sanitize_path_part(grid_name)
    for key in (f"volume_extent_mm_{suffix}", "volume_extent_mm"):
        extent = get_data_array(data, key, dtype=np.float64)
        if extent is None:
            continue
        values = extent.reshape(-1)
        if values.size >= 6 and np.all(np.isfinite(values[:6])):
            return tuple(float(value) for value in values[:6])
    return None


def volume_extent_from_shape(volume: np.ndarray) -> tuple[float, float, float, float, float, float]:
    z_size, y_size, x_size = volume.shape
    return (0.0, float(x_size), 0.0, float(y_size), 0.0, float(z_size))


def bioheat_layers_from_manifest(manifest: Mapping[str, Any]) -> list[tuple[float, float, str]]:
    # Study outputs must remain reproducible without reopening source configs.
    return []


def annotate_bioheat_layers(
    ax: plt.Axes,
    layers: list[tuple[float, float, str]],
    *,
    zmin: float,
    zmax: float,
) -> None:
    if not layers:
        return
    for start, end, label in layers:
        if zmin < start < zmax:
            ax.axhline(start, color="white", linewidth=1.0, linestyle=":", alpha=0.9)
        if zmin < end < zmax:
            ax.axhline(end, color="white", linewidth=1.0, linestyle=":", alpha=0.9)
        mid = 0.5 * (start + end)
        if zmin <= mid <= zmax and label in {"skull", "scalp"}:
            ax.text(
                0.03,
                mid,
                label,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=8,
                color="white",
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "#111827", "edgecolor": "none", "alpha": 0.55},
            )


def source_plane_for_grid(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    grid_name: str,
) -> np.ndarray | None:
    suffix = sanitize_path_part(grid_name)
    for key in (f"dT_source_plane_{suffix}", "dT_source_plane"):
        plane = get_data_array(data, key, dtype=np.float32)
        if plane is not None and plane.ndim == 2:
            return plane
    return None


def source_z_index_for_volume(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    grid_name: str,
    volume: np.ndarray,
) -> int:
    suffix = sanitize_path_part(grid_name)
    for key in (f"source_z_index_{suffix}", "source_z_index"):
        value = scalar_from_data(data, key, math.nan)
        if np.isfinite(value):
            return int(np.clip(round(value), 0, volume.shape[0] - 1))

    source_plane = source_plane_for_grid(data, grid_name)
    if source_plane is not None and source_plane.shape == volume.shape[1:]:
        diffs = []
        for z_index in range(volume.shape[0]):
            diff = finite_nanmean(np.abs(volume[z_index].astype(np.float64) - source_plane.astype(np.float64)))
            diffs.append(diff if np.isfinite(diff) else math.inf)
        if diffs and np.isfinite(min(diffs)):
            return int(np.argmin(diffs))

    finite_volume = np.where(np.isfinite(volume), volume, -np.inf)
    if np.all(np.isneginf(finite_volume)):
        return 0
    max_z, _max_y, _max_x = np.unravel_index(int(np.argmax(finite_volume)), volume.shape)
    return int(max_z)


def plot_temperature_volume_sections(
    axes: np.ndarray,
    volume: np.ndarray,
    *,
    extent: tuple[float, float, float, float, float, float],
    layers: list[tuple[float, float, str]],
    source_z: int,
    title_prefix: str,
) -> Any | None:
    volume = np.asarray(volume, dtype=np.float32)
    if volume.ndim != 3 or volume.size == 0:
        for ax in axes.reshape(-1):
            add_no_data(ax, "No 3D temperature volume")
        return None

    finite_volume = np.where(np.isfinite(volume), volume, -np.inf)
    if np.all(np.isneginf(finite_volume)):
        for ax in axes.reshape(-1):
            add_no_data(ax, "No finite temperature values")
        return None

    max_z, max_y, max_x = np.unravel_index(int(np.argmax(finite_volume)), volume.shape)
    z_size, y_size, x_size = volume.shape
    xmin, xmax, ymin, ymax, zmin, zmax = extent
    dx = (xmax - xmin) / max(x_size, 1)
    dy = (ymax - ymin) / max(y_size, 1)
    dz = (zmax - zmin) / max(z_size, 1)
    source_z = int(np.clip(source_z, 0, z_size - 1))
    x_hot_mm = xmin + (max_x + 0.5) * dx
    y_hot_mm = ymin + (max_y + 0.5) * dy
    z_hot_mm = zmin + (max_z + 0.5) * dz
    source_z_mm = zmin + (source_z + 0.5) * dz

    vmax = finite_nanmax(volume, default=0.0)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1e-6

    ax_xy, ax_xz, ax_yz = axes.reshape(-1)[:3]
    image = ax_xy.imshow(
        downsample_image(volume[source_z]),
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap=THERMAL_CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    ax_xy.axhline(y_hot_mm, color="cyan", linewidth=1.0, linestyle="--")
    ax_xy.axvline(x_hot_mm, color="cyan", linewidth=1.0, linestyle="--")
    ax_xy.plot(x_hot_mm, y_hot_mm, marker="x", color="cyan", markersize=6, mew=1.4)
    ax_xy.set_title(f"{title_prefix} x-y\nz={source_z_mm:.2f} mm", fontsize=9)
    ax_xy.set_xlabel("x (mm)")
    ax_xy.set_ylabel("y (mm)")

    ax_xz.imshow(
        downsample_image(volume[:, max_y, :]),
        origin="lower",
        extent=(xmin, xmax, zmin, zmax),
        cmap=THERMAL_CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax_xz.axvline(x_hot_mm, color="cyan", linewidth=1.0, linestyle="--")
    ax_xz.axhline(z_hot_mm, color="cyan", linewidth=1.0, linestyle="--")
    ax_xz.plot(x_hot_mm, z_hot_mm, marker="x", color="cyan", markersize=6, mew=1.4)
    annotate_bioheat_layers(ax_xz, layers, zmin=zmin, zmax=zmax)
    ax_xz.set_title(f"{title_prefix} x-z\ny={y_hot_mm:.2f} mm", fontsize=9)
    ax_xz.set_xlabel("x (mm)")
    ax_xz.set_ylabel("z (mm)")

    ax_yz.imshow(
        downsample_image(volume[:, :, max_x]),
        origin="lower",
        extent=(ymin, ymax, zmin, zmax),
        cmap=THERMAL_CMAP,
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="auto",
    )
    ax_yz.axvline(y_hot_mm, color="cyan", linewidth=1.0, linestyle="--")
    ax_yz.axhline(z_hot_mm, color="cyan", linewidth=1.0, linestyle="--")
    ax_yz.plot(y_hot_mm, z_hot_mm, marker="x", color="cyan", markersize=6, mew=1.4)
    annotate_bioheat_layers(ax_yz, layers, zmin=zmin, zmax=zmax)
    ax_yz.set_title(f"{title_prefix} y-z\nx={x_hot_mm:.2f} mm", fontsize=9)
    ax_yz.set_xlabel("y (mm)")
    ax_yz.set_ylabel("z (mm)")

    for ax in (ax_xy, ax_xz, ax_yz):
        ax.tick_params(labelsize=8)
    return image


def plot_protocol_temperature_heatmaps(
    record: MatrixRecord,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    with np.load(record.npz_path, allow_pickle=True) as data:
        snapshot_keys = thermal_snapshot_keys(data)
        if not snapshot_keys:
            return
        for grid_name, key in snapshot_keys:
            stack = get_data_array(data, key, dtype=np.float32)
            if stack is None or stack.ndim != 3 or stack.shape[0] == 0:
                continue
            stack = np.asarray(stack, dtype=np.float32)
            if stack.shape[1] == 0 or stack.shape[2] == 0:
                continue
            extent = extent_for_grid(data, grid_name)
            vmax = finite_nanmax(stack, default=0.0)
            if not np.isfinite(vmax) or vmax <= 0.0:
                vmax = 1.0

            safe_grid = sanitize_path_part(grid_name)
            sheet_path = case_visuals_dir(record) / f"temperature_heatmaps_{safe_grid}.{image_format}"
            if not sheet_path.exists() or overwrite:
                n_snapshots = int(stack.shape[0])
                ncols = min(5, n_snapshots)
                nrows = int(math.ceil(n_snapshots / float(ncols)))
                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(3.1 * ncols, 2.85 * nrows),
                    squeeze=False,
                )
                image = None
                for idx, ax in enumerate(axes.reshape(-1)):
                    if idx >= n_snapshots:
                        ax.axis("off")
                        continue
                    image = ax.imshow(
                        downsample_image(stack[idx]),
                        origin="lower",
                        extent=extent,
                        cmap=THERMAL_CMAP,
                        vmin=0.0,
                        vmax=vmax,
                        interpolation="nearest",
                        aspect="equal",
                    )
                    detail = middle_snapshot_detail(data, idx)
                    ax.set_title(detail or f"snapshot {idx + 1}", fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                if image is not None:
                    fig.subplots_adjust(top=0.90, right=0.88, hspace=0.28, wspace=0.10)
                    cbar_ax = fig.add_axes([0.91, 0.14, 0.015, 0.70])
                    cbar = fig.colorbar(image, cax=cbar_ax)
                    cbar.set_label("ΔT (°C)")
                    cbar.ax.tick_params(labelsize=8)
                fig.suptitle(f"Temperature Heatmaps - {grid_name}", fontsize=12, fontweight="bold")
                if image is None:
                    fig.subplots_adjust(top=0.90, right=0.96, hspace=0.28, wspace=0.10)
                save_figure(fig, sheet_path, overwrite=overwrite)
            else:
                print(f"Skipping existing: {sheet_path}")

            snapshot_dir = case_visuals_dir(record) / f"temperature_heatmaps_{safe_grid}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            for idx, heatmap in enumerate(stack):
                detail = middle_snapshot_detail(data, idx)
                label = sanitize_path_part(detail.replace(" | ", "_")) if detail else f"snapshot_{idx + 1:02d}"
                out_path = snapshot_dir / f"{idx + 1:02d}_{label}.{image_format}"
                if out_path.exists() and not overwrite:
                    print(f"Skipping existing: {out_path}")
                    continue
                fig, ax = plt.subplots(figsize=(6.4, 5.4))
                image = ax.imshow(
                    downsample_image(heatmap),
                    origin="lower",
                    extent=extent,
                    cmap=THERMAL_CMAP,
                    vmin=0.0,
                    vmax=vmax,
                    interpolation="nearest",
                    aspect="equal",
                )
                title = f"Temperature Map - {grid_name}"
                if detail:
                    title = f"{title}\n{detail}"
                ax.set_title(title, fontsize=12, fontweight="bold")
                ax.set_xlabel("x (mm)" if extent is not None else "x pixel")
                ax.set_ylabel("y (mm)" if extent is not None else "y pixel")
                ax.tick_params(labelsize=8)
                cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("ΔT (°C)")
                cbar.ax.tick_params(labelsize=8)
                fig.tight_layout()
                save_figure(fig, out_path, overwrite=overwrite)


def plot_protocol_temperature_series(
    record: MatrixRecord,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    for key, title, ylabel, filename, color in (
        ("mean_dT", "Mean ΔT Over Time", "mean ΔT (°C)", "mean_dT_over_time", "#1F4E79"),
        ("max_dT", "Max Focal ΔT Over Time", "max focal ΔT (°C)", "max_focal_dT_over_time", "#8B1E3F"),
    ):
        out_path = case_visuals_dir(record) / f"{filename}.{image_format}"
        if out_path.exists() and not overwrite:
            print(f"Skipping existing: {out_path}")
            continue
        with np.load(record.npz_path, allow_pickle=True) as data:
            series = series_from_key(data, key, max_points=MAX_SERIES_POINTS)
            fig, ax = plt.subplots(figsize=(8.8, 4.4))
            if series is None:
                add_no_data(ax, f"No {title.lower()}")
            else:
                ax.plot(series[0], series[1], color=color, linewidth=1.9)
                annotate_mean_max(
                    ax,
                    mean_value=finite_nanmean(series[1]),
                    max_value=finite_nanmax(series[1]),
                    mean_label="mean",
                    max_label="max",
                    unit="°C",
                )
                style_axes(ax)
            add_video_end_line(ax, data)
            ax.set_title(title)
            ax.set_xlabel(TIME_AXIS_LABEL)
            ax.set_ylabel(ylabel)
            save_figure(fig, out_path, overwrite=overwrite)


def plot_protocol_temperature_areas(
    record: MatrixRecord,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    out_path = case_visuals_dir(record) / f"temperature_area_thresholds_over_time.{image_format}"
    if out_path.exists() and not overwrite:
        print(f"Skipping existing: {out_path}")
        return
    with np.load(record.npz_path, allow_pickle=True) as data:
        fig, ax = plt.subplots(figsize=(8.8, 4.4))
        plotted = False
        for key, label, color in (
            ("area_gt1_mm2", ">1 °C", "#1F4E79"),
            ("area_gt2_mm2", ">2 °C", "#C05621"),
            ("area_gt3_mm2", ">3 °C", "#8B1E3F"),
        ):
            series = series_from_key(data, key, max_points=MAX_SERIES_POINTS)
            if series is None:
                continue
            ax.plot(series[0], series[1], color=color, linewidth=1.8, label=label)
            plotted = True
        if plotted:
            ax.legend(frameon=False, fontsize=8)
            style_axes(ax)
        else:
            add_no_data(ax, "No hotspot areas")
        add_video_end_line(ax, data)
        ax.set_title("Areas Above Temperature Thresholds")
        ax.set_xlabel(TIME_AXIS_LABEL)
        ax.set_ylabel("Area (mm2)")
        save_figure(fig, out_path, overwrite=overwrite)


def write_per_protocol_visuals(
    records: list[MatrixRecord],
    image_format: str,
    *,
    overwrite: bool,
    output_root: Path | None = None,
) -> None:
    previous_root = _CASE_VISUALS_ROOT
    if output_root is not None:
        set_case_visuals_root(output_root)
    try:
        for record in records:
            plot_protocol_metric_distribution(
                record,
                metric="amplitude",
                title="Current Amplitudes Over Time",
                ylabel="current (uA)",
                filename="current_amplitudes_over_time",
                image_format=image_format,
                overwrite=overwrite,
                positive_only=True,
            )
            plot_protocol_activated_electrodes(record, image_format, overwrite=overwrite)
            plot_protocol_metric_distribution(
                record,
                metric="shannon",
                title="Shannon K Over Time",
                ylabel="Shannon K",
                filename="shannon_k_over_time",
                image_format=image_format,
                overwrite=overwrite,
                positive_only=False,
            )
            plot_protocol_charge_per_second(record, image_format, overwrite=overwrite)
            plot_protocol_charge_over_protocol(record, image_format, overwrite=overwrite)
            plot_protocol_temperature_heatmaps(record, image_format, overwrite=overwrite)
            plot_protocol_temperature_series(record, image_format, overwrite=overwrite)
    finally:
        set_case_visuals_root(previous_root)


def comparison_output_dir(output_root: Path, block: str) -> Path:
    return output_root


def record_color(index: int, block: str | None = None) -> str:
    if block in COLORS:
        return COLORS[str(block)]
    return LINE_COLORS[index % len(LINE_COLORS)]


def plot_records_series(
    records: list[MatrixRecord],
    *,
    output_path: Path,
    image_format: str,
    overwrite: bool,
    title: str,
    ylabel: str,
    getter,
    show_legend: bool = True,
) -> None:
    if not records:
        return
    path = output_path.with_suffix(f".{image_format}")
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return
    fig, ax = plt.subplots(figsize=(max(8.5, 0.45 * len(records) + 7.0), 4.8))
    plotted = False
    for idx, record in enumerate(records):
        with np.load(record.npz_path, allow_pickle=True) as data:
            series = getter(data)
            if series is None:
                continue
            x, y = series
            if x.size == 0 or y.size == 0:
                continue
            ax.plot(
                x,
                y,
                linewidth=1.6,
                color=LINE_COLORS[idx % len(LINE_COLORS)],
                label=record.label.replace("\n", " "),
            )
            add_video_end_line(ax, data, label="video end" if not plotted else "_nolegend_")
            plotted = True
    if plotted:
        if show_legend:
            ax.legend(frameon=False, fontsize=8, ncol=2)
        style_axes(ax)
    else:
        add_no_data(ax, "No comparable series")
    ax.set_title(title)
    ax.set_xlabel(TIME_AXIS_LABEL)
    ax.set_ylabel(ylabel)
    save_figure(fig, path, overwrite=overwrite)


def plot_records_bar(
    records: list[MatrixRecord],
    *,
    output_path: Path,
    image_format: str,
    overwrite: bool,
    title: str,
    ylabel: str,
    getter,
    color: str | None = None,
) -> None:
    if not records:
        return
    path = output_path.with_suffix(f".{image_format}")
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return
    values = []
    labels = []
    for record in records:
        with np.load(record.npz_path, allow_pickle=True) as data:
            values.append(getter(data))
        labels.append(record.label)
    fig, ax = plt.subplots(figsize=(max(7.5, 0.75 * len(records) + 2.5), 4.6))
    bar_colors = color or COLORS.get(records[0].block, FALLBACK_COLOR)
    ax.bar(range(len(records)), values, color=bar_colors, alpha=0.9)
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    style_axes(ax)
    save_figure(fig, path, overwrite=overwrite)


def mean_of_series_key(data: np.lib.npyio.NpzFile | Mapping[str, Any], metric: str) -> float:
    series = mean_metric_series(data, metric, max_points=MAX_SERIES_POINTS)
    if series is None:
        return math.nan
    return finite_nanmean(series[1])


def max_of_series_key(data: np.lib.npyio.NpzFile | Mapping[str, Any], key: str) -> float:
    series = series_from_key(data, key, max_points=MAX_SERIES_POINTS)
    if series is None:
        return math.nan
    return finite_nanmax(series[1])


def mean_charge_rate_values(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
) -> tuple[float, float]:
    per_e_series = mean_metric_series(data, "charge_rate", max_points=MAX_SERIES_POINTS)
    total_series = series_from_key(data, "charge_per_second_total_nC_s")
    per_e = finite_nanmean(per_e_series[1]) if per_e_series is not None else math.nan
    total_nC_s = finite_nanmean(total_series[1]) if total_series is not None else math.nan
    if np.isfinite(per_e) and np.isfinite(total_nC_s):
        return per_e / 1e3, total_nC_s / 1e6

    rows_result = metric_rows(data, "charge_rate", max_points=MAX_CLOUD_POINTS)
    if rows_result is None:
        return (
            per_e / 1e3 if np.isfinite(per_e) else math.nan,
            total_nC_s / 1e6 if np.isfinite(total_nC_s) else math.nan,
        )
    _x, rows = rows_result
    stats = distribution_stats(rows, positive_only=True)
    totals = np.sum(np.where(np.isfinite(rows) & (rows > 0.0), rows, 0.0), axis=1)
    if not np.isfinite(per_e):
        per_e = finite_nanmean(stats["mean"])
    if not np.isfinite(total_nC_s):
        total_nC_s = finite_nanmean(totals)
    return per_e / 1e3, total_nC_s / 1e6


def plot_charge_rate_bars(
    records: list[MatrixRecord],
    output_root: Path,
    block: str,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    if not records:
        return
    path = comparison_output_dir(output_root, block) / f"mean_charge_per_second_bars.{image_format}"
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return
    labels = [record.label for record in records]
    per_e = []
    total = []
    for record in records:
        with np.load(record.npz_path, allow_pickle=True) as data:
            per_value, total_value = mean_charge_rate_values(data)
        per_e.append(per_value)
        total.append(total_value)
    fig, axes = plt.subplots(1, 2, figsize=(max(11.0, 0.8 * len(records) + 4.5), 4.8))
    for ax, values, title, ylabel, limit in (
        (
            axes[0],
            per_e,
            "Mean Per-Electrode Charge Rate",
            "uC/s/electrode",
            plot_safety_limit("window_charge_per_electrode_nC") / 1e3,
        ),
        (
            axes[1],
            total,
            "Mean Summed Charge Rate",
            "mC/s",
            plot_safety_limit("window_charge_total_nC") / 1e6,
        ),
    ):
        ax.bar(range(len(records)), values, color=COLORS.get(block, FALLBACK_COLOR), alpha=0.9)
        if add_limit_line(ax, limit):
            ax.legend(frameon=False, fontsize=8)
        ax.set_xticks(range(len(records)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        style_axes(ax)
    fig.suptitle(f"Mean Charge Per Second - {block.replace('_', ' ')}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    save_figure(fig, path, overwrite=overwrite)


def total_protocol_charge_mC(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    final = final_protocol_charge_nC(data)
    if final is not None:
        return float(np.nansum(final)) / 1e6
    series = protocol_total_charge_series_nC(data, max_points=MAX_SERIES_POINTS)
    if series is None or series[1].size == 0:
        return math.nan
    return float(series[1][-1]) / 1e6


def total_protocol_charge_uC(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    return total_protocol_charge_mC(data) * 1e3


def final_charge_distribution_mC(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> np.ndarray:
    final = final_protocol_charge_nC(data)
    if final is None:
        return np.asarray([], dtype=np.float64)
    values = np.asarray(final, dtype=np.float64).reshape(-1) / 1e6
    return values[np.isfinite(values)]


def final_charge_distribution_uC(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> np.ndarray:
    return final_charge_distribution_mC(data) * 1e3


def plot_total_charge_boxplot(
    records: list[MatrixRecord],
    output_root: Path,
    block: str,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    if not records:
        return
    path = comparison_output_dir(output_root, block) / f"total_charge_electrode_distribution_boxplot.{image_format}"
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return
    distributions = []
    labels = []
    for record in records:
        with np.load(record.npz_path, allow_pickle=True) as data:
            values = final_charge_distribution_mC(data)
        if values.size == 0:
            values = np.asarray([math.nan])
        distributions.append(values)
        labels.append(record.label)
    fig, ax = plt.subplots(figsize=(max(7.5, 0.72 * len(records) + 2.5), 4.8))
    try:
        ax.boxplot(distributions, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(distributions, labels=labels, showfliers=False)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(f"Protocol Charge Distribution - {block.replace('_', ' ')}")
    ax.set_ylabel("Final charge (mC/electrode)")
    style_axes(ax)
    save_figure(fig, path, overwrite=overwrite)


def plot_charge_heatmap_sheet(
    records: list[MatrixRecord],
    output_root: Path,
    block: str,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    if not records:
        return
    path = comparison_output_dir(output_root, block) / f"protocol_charge_heatmaps.{image_format}"
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return
    final_by_record: list[np.ndarray | None] = []
    vmax = 0.0
    for record in records:
        with np.load(record.npz_path, allow_pickle=True) as data:
            final = final_protocol_charge_nC(data)
        final_by_record.append(final)
        if final is not None:
            vmax = max(vmax, finite_nanmax(final / 1e6, default=0.0))
    cols = min(4, len(records))
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.6 * rows), squeeze=False)
    image = None
    for idx, (record, final) in enumerate(zip(records, final_by_record)):
        ax = axes.reshape(-1)[idx]
        manifest = manifest_for_record(record)
        with np.load(record.npz_path, allow_pickle=True) as data:
            if final is None:
                add_no_data(ax, "No charge map")
                ax.set_title(record.label.replace("\n", " "))
            else:
                image = plot_electrode_charge_heatmap(
                    ax,
                    data,
                    final,
                    manifest=manifest,
                    title=record.label.replace("\n", " "),
                    vmax_mC=vmax if vmax > 0.0 else None,
                )
    for ax in axes.reshape(-1)[len(records):]:
        ax.axis("off")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.reshape(-1).tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("charge (mC)")
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(f"Protocol Charge Heatmaps - {block.replace('_', ' ')}", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, right=0.92, hspace=0.35, wspace=0.25)
    save_figure(fig, path, overwrite=overwrite)


def pairwise_path(
    output_root: Path,
    metric: str,
    pair_name: str,
    fixed_factor: str,
    fixed_value: FactorValue,
    image_format: str,
    *,
    suffix: str = "",
) -> Path:
    filename = f"fixed_{fixed_factor}_{sanitize_path_part(fixed_value.key)}"
    if suffix:
        filename = f"{filename}_{sanitize_path_part(suffix)}"
    return output_root / "pairwise" / metric / pair_name / f"{filename}.{image_format}"


def fixed_factor_caption(fixed_factor: str, fixed_value: FactorValue) -> str:
    if fixed_factor == "preprocessing":
        return fixed_value.label
    return f"Fixed {factor_display_name(fixed_factor)}: {fixed_value.label}"


def is_thermal_metric(metric_name: str) -> bool:
    normalized = metric_name.strip().lower()
    return "temperature" in normalized or normalized.endswith("_dt")


def contrasting_cell_text_color(
    value: float,
    *,
    vmin: float,
    vmax: float,
    cmap: str = THERMAL_CMAP,
) -> str:
    if not np.isfinite(value):
        return FALLBACK_COLOR
    if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
        normalized = 0.5
    else:
        normalized = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    red, green, blue, _alpha = plt.get_cmap(cmap)(normalized)
    srgb = np.asarray([red, green, blue], dtype=np.float64)
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    luminance = float(np.dot(linear, np.asarray([0.2126, 0.7152, 0.0722])))
    white_contrast = 1.05 / (luminance + 0.05)
    black_contrast = (luminance + 0.05) / 0.05
    return "white" if white_contrast >= black_contrast else "black"


def matrix_record_for_values(
    records_by_key: Mapping[tuple[str, str, str], MatrixRecord],
    values: Mapping[str, FactorValue],
) -> MatrixRecord | None:
    return records_by_key.get(record_key_for_factors(values))


def pairwise_sheet_values(
    factors_by_run_id: Mapping[str, MatrixFactors],
    x_factor: str,
    y_factor: str,
    fixed_factor: str,
) -> tuple[list[FactorValue], list[FactorValue], list[FactorValue]]:
    return (
        sorted_factor_values(factors_by_run_id, x_factor),
        sorted_factor_values(factors_by_run_id, y_factor),
        sorted_factor_values(factors_by_run_id, fixed_factor),
    )


def scalar_peak_active_electrodes(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    series = active_count_series(data, max_points=MAX_SERIES_POINTS)
    if series is None:
        return math.nan
    return finite_nanmax(series[1])


def scalar_max_shannon_k(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    value = finite_nanmax(get_data_array(data, "shannon_k_per_electrode"), math.nan)
    if np.isfinite(value):
        return value
    return max_of_series_key(data, "shannon_k_mean")


def scalar_final_total_charge_mC(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    return total_protocol_charge_mC(data)


def scalar_max_mean_dT(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    return max_of_series_key(data, "mean_dT")


def scalar_max_dT(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    return max_of_series_key(data, "max_dT")


SCALAR_PAIRWISE_METRICS: tuple[tuple[str, str, str, Callable[[np.lib.npyio.NpzFile | Mapping[str, Any]], float]], ...] = (
    ("max_shannon_k", "Max Shannon K", "K", scalar_max_shannon_k),
    ("peak_activated_electrodes", "Peak Activated Electrodes", "electrodes", scalar_peak_active_electrodes),
    ("final_total_accumulated_charge", "Final Total Accumulated Charge", "mC", scalar_final_total_charge_mC),
    ("max_mean_dT", "Max Mean ΔT", "°C", scalar_max_mean_dT),
    ("max_dT", "Max Focal ΔT", "°C", scalar_max_dT),
)


def format_scalar_value(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if abs(value) >= 1000.0:
        return f"{value:.3g}"
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    if abs(value) >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3g}"


def plot_pairwise_scalar_heatmap(
    records_by_key: Mapping[tuple[str, str, str], MatrixRecord],
    factors_by_run_id: Mapping[str, MatrixFactors],
    output_root: Path,
    image_format: str,
    *,
    metric_name: str,
    title: str,
    unit: str,
    getter: Callable[[np.lib.npyio.NpzFile | Mapping[str, Any]], float],
    x_factor: str,
    y_factor: str,
    fixed_factor: str,
    pair_name: str,
    fixed_value: FactorValue,
    overwrite: bool,
) -> None:
    x_values, y_values, _fixed_values = pairwise_sheet_values(factors_by_run_id, x_factor, y_factor, fixed_factor)
    if not x_values or not y_values:
        return
    path = pairwise_path(output_root, metric_name, pair_name, fixed_factor, fixed_value, image_format)
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return

    values = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            factor_values = {x_factor: x_value, y_factor: y_value, fixed_factor: fixed_value}
            record = matrix_record_for_values(records_by_key, factor_values)
            if record is None:
                continue
            with np.load(record.npz_path, allow_pickle=True) as data:
                values[row, col] = getter(data)

    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    masked = np.ma.masked_invalid(values)
    cmap = THERMAL_CMAP if is_thermal_metric(metric_name) else "viridis"
    image = ax.imshow(masked, cmap=cmap, aspect="auto")
    finite = values[np.isfinite(values)]
    text_vmin = float(np.min(finite)) if finite.size else 0.0
    text_vmax = float(np.max(finite)) if finite.size else 1.0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                format_scalar_value(value),
                ha="center",
                va="center",
                color=contrasting_cell_text_color(
                    value,
                    vmin=text_vmin,
                    vmax=text_vmax,
                    cmap=cmap,
                ),
                fontsize=9,
                fontweight="bold" if np.isfinite(value) else "normal",
            )
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([value.label for value in x_values], rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([value.label for value in y_values], fontsize=11)
    ax.set_xlabel(factor_display_name(x_factor), fontsize=12)
    ax.set_ylabel(factor_display_name(y_factor), fontsize=12)
    ax.set_title(f"{title}\n{fixed_factor_caption(fixed_factor, fixed_value)}")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(unit)
    fig.tight_layout()
    save_figure(fig, path, overwrite=overwrite)


def draw_line_series(
    ax: plt.Axes,
    series: tuple[np.ndarray, np.ndarray] | None,
    *,
    color: str,
    label: str,
) -> bool:
    if series is None:
        return False
    x, y = series
    if x.size == 0 or y.size == 0:
        return False
    ax.plot(x, y, color=color, linewidth=1.7, label=label)
    return True


def draw_mean_metric_on_axis(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    metric: str,
    *,
    color: str,
    label: str,
) -> bool:
    return draw_line_series(
        ax,
        mean_metric_series(data, metric, max_points=MAX_SERIES_POINTS),
        color=color,
        label=label,
    )


def draw_data_series_on_axis(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    key: str,
    *,
    color: str,
    label: str,
) -> bool:
    return draw_line_series(
        ax,
        series_from_key(data, key, max_points=MAX_SERIES_POINTS),
        color=color,
        label=label,
    )


def draw_active_count_on_axis(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    *,
    color: str,
    label: str,
) -> bool:
    return draw_line_series(
        ax,
        active_count_series(data, max_points=MAX_SERIES_POINTS),
        color=color,
        label=label,
    )


def draw_amplitude_on_axis(
    ax: plt.Axes,
    record: MatrixRecord,
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    *,
    color: str,
    label: str,
) -> bool:
    manifest = manifest_for_record(record)
    rows_result = metric_rows(data, "amplitude", max_points=MAX_CLOUD_POINTS)
    if rows_result is not None and should_use_cloud_for_case(manifest):
        x, rows = rows_result
        plot_metric_cloud(
            ax,
            x,
            rows,
            label=label,
            color=color,
            positive_only=True,
            use_cloud=True,
        )
        return True
    return draw_mean_metric_on_axis(ax, data, "amplitude", color=color, label=label)


def draw_total_charge_on_axis(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    *,
    color: str,
    label: str,
) -> bool:
    series = protocol_total_charge_series_nC(data, max_points=MAX_SERIES_POINTS)
    if series is None:
        return False
    x, y = series
    return draw_line_series(ax, (x, y / 1e6), color=color, label=label)


def draw_per_electrode_charge_cloud_on_axis(
    ax: plt.Axes,
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    *,
    color: str,
    label: str,
) -> bool:
    cumulative = cumulative_charge_summaries(data, max_points=MAX_CLOUD_POINTS)
    if cumulative is None:
        return False
    x = cumulative["time_s"]
    ax.fill_between(x, cumulative["p05"] / 1e6, cumulative["p95"] / 1e6, color=color, alpha=0.14, linewidth=0)
    ax.fill_between(x, cumulative["p25"] / 1e6, cumulative["p75"] / 1e6, color=color, alpha=0.24, linewidth=0)
    ax.plot(x, cumulative["mean"] / 1e6, color=color, linewidth=1.7, label=label)
    return True


TIME_PAIRWISE_METRICS: tuple[
    tuple[
        str,
        str,
        str,
        Callable[[plt.Axes, MatrixRecord, np.lib.npyio.NpzFile | Mapping[str, Any], str, str], bool],
    ],
    ...,
] = (
    (
        "mean_shannon_k_over_time",
        "Mean Shannon K Over Time",
        "mean Shannon K",
        lambda ax, record, data, color, label: draw_mean_metric_on_axis(ax, data, "shannon", color=color, label=label),
    ),
    (
        "activated_electrodes_over_time",
        "Activated Electrodes Over Time",
        "active electrodes",
        lambda ax, record, data, color, label: draw_active_count_on_axis(ax, data, color=color, label=label),
    ),
    (
        "amplitude_over_time",
        "Amplitude Over Time",
        "current (uA)",
        lambda ax, record, data, color, label: draw_amplitude_on_axis(ax, record, data, color=color, label=label),
    ),
    (
        "summed_accumulated_charge_over_time",
        "Summed Accumulated Charge Over Time",
        "total charge (mC)",
        lambda ax, record, data, color, label: draw_total_charge_on_axis(ax, data, color=color, label=label),
    ),
    (
        "per_electrode_accumulated_charge_over_time",
        "Per-Electrode Accumulated Charge Over Time",
        "charge (mC/electrode)",
        lambda ax, record, data, color, label: draw_per_electrode_charge_cloud_on_axis(ax, data, color=color, label=label),
    ),
    (
        "mean_dT_over_time",
        "Mean ΔT Over Time",
        "mean ΔT (°C)",
        lambda ax, record, data, color, label: draw_data_series_on_axis(ax, data, "mean_dT", color=color, label=label),
    ),
    (
        "max_focal_dT_over_time",
        "Max Focal ΔT Over Time",
        "max focal ΔT (°C)",
        lambda ax, record, data, color, label: draw_data_series_on_axis(ax, data, "max_dT", color=color, label=label),
    ),
)


def plot_pairwise_time_sheet(
    records_by_key: Mapping[tuple[str, str, str], MatrixRecord],
    factors_by_run_id: Mapping[str, MatrixFactors],
    output_root: Path,
    image_format: str,
    *,
    metric_name: str,
    title: str,
    ylabel: str,
    drawer: Callable[[plt.Axes, MatrixRecord, np.lib.npyio.NpzFile | Mapping[str, Any], str, str], bool],
    x_factor: str,
    y_factor: str,
    fixed_factor: str,
    pair_name: str,
    fixed_value: FactorValue,
    overwrite: bool,
) -> None:
    x_values, y_values, _fixed_values = pairwise_sheet_values(factors_by_run_id, x_factor, y_factor, fixed_factor)
    if not x_values or not y_values:
        return
    path = pairwise_path(output_root, metric_name, pair_name, fixed_factor, fixed_value, image_format)
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return

    fig, axes = plt.subplots(
        len(y_values),
        len(x_values),
        figsize=(4.3 * len(x_values), 3.2 * len(y_values)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            ax = axes[row, col]
            factor_values = {x_factor: x_value, y_factor: y_value, fixed_factor: fixed_value}
            record = matrix_record_for_values(records_by_key, factor_values)
            if record is None:
                add_no_data(ax, "missing")
            else:
                color = LINE_COLORS[(row * len(x_values) + col) % len(LINE_COLORS)]
                with np.load(record.npz_path, allow_pickle=True) as data:
                    plotted = drawer(ax, record, data, color, record.label.replace("\n", " "))
                    add_video_end_line(ax, data)
                if plotted:
                    style_axes(ax)
                else:
                    add_no_data(ax, "no data")
            if row == 0:
                ax.set_title(x_value.label, fontsize=12)
            if col == 0:
                ax.set_ylabel(f"{y_value.label}\n{ylabel}", fontsize=11)
            else:
                ax.set_ylabel("")
            if row == len(y_values) - 1:
                ax.set_xlabel(TIME_AXIS_LABEL)
    fig.suptitle(
        f"{title}\n{pair_name.replace('_', ' ')} | {fixed_factor_caption(fixed_factor, fixed_value)}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    save_figure(fig, path, overwrite=overwrite)


def plot_pairwise_charge_heatmap_sheet(
    records_by_key: Mapping[tuple[str, str, str], MatrixRecord],
    factors_by_run_id: Mapping[str, MatrixFactors],
    output_root: Path,
    image_format: str,
    *,
    x_factor: str,
    y_factor: str,
    fixed_factor: str,
    pair_name: str,
    fixed_value: FactorValue,
    overwrite: bool,
) -> None:
    x_values, y_values, _fixed_values = pairwise_sheet_values(factors_by_run_id, x_factor, y_factor, fixed_factor)
    if not x_values or not y_values:
        return
    path = pairwise_path(
        output_root,
        "final_accumulated_charge_heatmap",
        pair_name,
        fixed_factor,
        fixed_value,
        image_format,
    )
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return

    final_by_cell: dict[tuple[int, int], np.ndarray | None] = {}
    vmax = 0.0
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            factor_values = {x_factor: x_value, y_factor: y_value, fixed_factor: fixed_value}
            record = matrix_record_for_values(records_by_key, factor_values)
            final = None
            if record is not None:
                with np.load(record.npz_path, allow_pickle=True) as data:
                    final = final_protocol_charge_nC(data)
            final_by_cell[(row, col)] = final
            if final is not None:
                vmax = max(vmax, finite_nanmax(final / 1e6, default=0.0))

    fig, axes = plt.subplots(
        len(y_values),
        len(x_values),
        figsize=(4.2 * len(x_values), 3.8 * len(y_values)),
        squeeze=False,
    )
    image = None
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            ax = axes[row, col]
            factor_values = {x_factor: x_value, y_factor: y_value, fixed_factor: fixed_value}
            record = matrix_record_for_values(records_by_key, factor_values)
            final = final_by_cell[(row, col)]
            if record is None or final is None:
                add_no_data(ax, "No charge map")
            else:
                manifest = manifest_for_record(record)
                with np.load(record.npz_path, allow_pickle=True) as data:
                    image = plot_electrode_charge_heatmap(
                        ax,
                        data,
                        final,
                        manifest=manifest,
                        title=x_value.label if row == 0 else "",
                        vmax_mC=vmax if vmax > 0.0 else None,
                    )
            if row == 0:
                ax.set_title(x_value.label, fontsize=10)
            if col == 0:
                ax.set_ylabel(y_value.label, fontsize=9)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.reshape(-1).tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("charge (mC)")
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Final Accumulated Charge Heatmap\n{pair_name.replace('_', ' ')} | "
        f"{fixed_factor_caption(fixed_factor, fixed_value)}",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.90, right=0.92, hspace=0.28, wspace=0.22)
    save_figure(fig, path, overwrite=overwrite)


def thermal_stack_for_grid(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    grid_name: str,
) -> np.ndarray | None:
    suffix = sanitize_path_part(grid_name)
    return get_data_array(data, f"dT_heatmaps_{suffix}", dtype=np.float32)


def plot_pairwise_temperature_heatmap_sheet(
    records_by_key: Mapping[tuple[str, str, str], MatrixRecord],
    factors_by_run_id: Mapping[str, MatrixFactors],
    output_root: Path,
    image_format: str,
    *,
    x_factor: str,
    y_factor: str,
    fixed_factor: str,
    pair_name: str,
    fixed_value: FactorValue,
    grid_name: str,
    position_label: str,
    overwrite: bool,
) -> None:
    x_values, y_values, _fixed_values = pairwise_sheet_values(factors_by_run_id, x_factor, y_factor, fixed_factor)
    if not x_values or not y_values:
        return
    path = pairwise_path(
        output_root,
        f"temperature_heatmaps_{position_label}",
        pair_name,
        fixed_factor,
        fixed_value,
        image_format,
        suffix=grid_name,
    )
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return

    images: dict[tuple[int, int], np.ndarray | None] = {}
    vmax = 0.0
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            factor_values = {x_factor: x_value, y_factor: y_value, fixed_factor: fixed_value}
            record = matrix_record_for_values(records_by_key, factor_values)
            heatmap = None
            if record is not None:
                with np.load(record.npz_path, allow_pickle=True) as data:
                    stack = thermal_stack_for_grid(data, grid_name)
                    if stack is not None and stack.ndim == 3 and stack.shape[0] > 0:
                        positions = dict(thermal_snapshot_positions(stack.shape[0]))
                        snap_idx = positions.get(position_label)
                        if snap_idx is not None:
                            heatmap = stack[snap_idx]
            images[(row, col)] = heatmap
            if heatmap is not None:
                vmax = max(vmax, finite_nanmax(heatmap, default=0.0))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(
        len(y_values),
        len(x_values),
        figsize=(3.5 * len(x_values), 3.25 * len(y_values)),
        squeeze=False,
    )
    image = None
    for row, y_value in enumerate(y_values):
        for col, x_value in enumerate(x_values):
            ax = axes[row, col]
            heatmap = images[(row, col)]
            if heatmap is None:
                add_no_data(ax, "No ΔT map")
            else:
                image = ax.imshow(
                    downsample_image(heatmap),
                    origin="lower",
                    aspect="equal",
                    cmap=THERMAL_CMAP,
                    vmin=0.0,
                    vmax=vmax,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            if row == 0:
                ax.set_title(x_value.label, fontsize=10)
            if col == 0:
                ax.set_ylabel(y_value.label, fontsize=9)
    if image is not None:
        fig.subplots_adjust(top=0.88, right=0.87, hspace=0.25, wspace=0.12)
        cbar_ax = fig.add_axes([0.90, 0.14, 0.015, 0.68])
        cbar = fig.colorbar(image, cax=cbar_ax)
        cbar.set_label("ΔT (°C)")
        cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        f"Temperature Heatmaps - {grid_name} - {position_label}\n"
        f"{pair_name.replace('_', ' ')} | {fixed_factor_caption(fixed_factor, fixed_value)}",
        fontsize=13,
        fontweight="bold",
    )
    if image is None:
        fig.subplots_adjust(top=0.88, right=0.96, hspace=0.25, wspace=0.12)
    save_figure(fig, path, overwrite=overwrite)


def thermal_grid_names_for_records(records: Iterable[MatrixRecord]) -> list[str]:
    names: dict[str, str] = {}
    for record in records:
        with np.load(record.npz_path, allow_pickle=True) as data:
            for grid_name, _key in thermal_snapshot_keys(data):
                names.setdefault(sanitize_path_part(grid_name), grid_name)
    return [names[key] for key in sorted(names)]


def factor_display_name(factor: str) -> str:
    return {
        "amplitude": "Amplitude",
        "grid": "Electrode Grid",
        "preprocessing": "Preprocessing",
    }.get(factor, factor.replace("_", " ").title())


def summary_grid_path(
    output_root: Path,
    metric_name: str,
    stat_name: str,
    pair_name: str,
    image_format: str,
) -> Path:
    return (
        output_root
        / "summary_grids"
        / sanitize_path_part(metric_name)
        / f"{sanitize_path_part(stat_name)}_{sanitize_path_part(pair_name)}.{image_format}"
    )


def title_with_charge_limit(title: str, metric_name: str, unit: str) -> str:
    limit = math.nan
    limit_label = "limit"
    if metric_name == "charge_per_second":
        limit = plot_safety_limit("window_charge_per_electrode_nC") / 1e3
        limit_label = "per-electrode limit"
    elif metric_name == "summed_charge_per_second":
        limit = plot_safety_limit("window_charge_total_nC") / 1e6
        limit_label = "total-array limit"
    elif metric_name == "total_charge":
        limit = plot_safety_limit("session_charge_limit_mC")
        limit_label = "session limit"
    if not np.isfinite(limit) or limit <= 0.0:
        return title
    return f"{title} ({limit_label}: {format_summary_number(limit)} {unit})"


def matrix_summary_values(
    records: Iterable[MatrixRecord],
    getter: Callable[[np.lib.npyio.NpzFile | Mapping[str, Any]], float],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for record in records:
        with np.load(record.npz_path, allow_pickle=True) as data:
            values[record.run_id] = getter(data)
    return values


def summary_active_electrodes_mean(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    series = active_count_series(data, max_points=MAX_SERIES_POINTS)
    if series is None:
        return math.nan
    return finite_nanmean(series[1])


def summary_active_electrodes_max(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    series = active_count_series(data, max_points=MAX_SERIES_POINTS)
    if series is None:
        return math.nan
    return finite_nanmax(series[1])


def summary_shannon_mean(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    series = mean_metric_series(data, "shannon", max_points=MAX_SERIES_POINTS)
    if series is not None:
        return finite_nanmean(series[1])
    rows_result = metric_rows(data, "shannon", max_points=MAX_CLOUD_POINTS)
    if rows_result is None:
        return math.nan
    _x, rows = rows_result
    return finite_nanmean(distribution_stats(rows, positive_only=False)["mean"])


def summary_shannon_max(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    return scalar_max_shannon_k(data)


def summary_charge_rate_mean(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    series = mean_metric_series(data, "charge_rate", max_points=MAX_SERIES_POINTS)
    if series is not None:
        return finite_nanmean(series[1] / 1e3)
    rows_result = metric_rows(data, "charge_rate", max_points=MAX_CLOUD_POINTS)
    if rows_result is None:
        return math.nan
    _x, rows = rows_result
    return finite_nanmean(distribution_stats(rows / 1e3, positive_only=True)["mean"])


def summary_charge_rate_max(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    rows_result = metric_rows(data, "charge_rate", max_points=MAX_CLOUD_POINTS)
    if rows_result is not None:
        _x, rows = rows_result
        return finite_nanmax(finite_summary_values(rows / 1e3, positive_only=True))
    series = mean_metric_series(data, "charge_rate", max_points=MAX_SERIES_POINTS)
    if series is None:
        return math.nan
    return finite_nanmax(series[1] / 1e3)


def summary_total_charge_rate_mean_mC_s(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    series = total_metric_series(data, "charge_rate", max_points=MAX_SERIES_POINTS)
    if series is None:
        return math.nan
    return finite_nanmean(series[1] / 1e6)


def summary_total_charge_rate_max_mC_s(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    series = total_metric_series(data, "charge_rate", max_points=MAX_SERIES_POINTS)
    if series is None:
        return math.nan
    return finite_nanmax(series[1] / 1e6)


def final_peak_temperature_scalars(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> list[float]:
    values: list[float] = []
    for key in data_files(data):
        if (
            key == "final_peak_temperature_C"
            or key.startswith("final_peak_temperature_C_")
            or key == "stationary_temperature_C"
            or key.startswith("stationary_temperature_C_")
        ):
            value = scalar_from_data(data, key, math.nan)
            if np.isfinite(value):
                values.append(value)
    return values


def temperature_baseline_C(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    for key in ("baseline_temperature_C", "body_temperature_C", "blood_temperature_C", "T_b"):
        value = scalar_from_data(data, key, math.nan)
        if np.isfinite(value):
            return value
    estimates: list[float] = []
    for key in data_files(data):
        if key == "final_peak_temperature_C":
            dT_key = "final_peak_dT_C"
        elif key.startswith("final_peak_temperature_C_"):
            dT_key = key.replace("final_peak_temperature_C_", "final_peak_dT_C_", 1)
        elif key == "stationary_temperature_C":
            dT_key = "stationary_dT_C"
        elif key.startswith("stationary_temperature_C_"):
            dT_key = key.replace("stationary_temperature_C_", "stationary_dT_C_", 1)
        else:
            continue
        temp = scalar_from_data(data, key, math.nan)
        dT = scalar_from_data(data, dT_key, math.nan)
        if np.isfinite(temp) and np.isfinite(dT):
            estimates.append(temp - dT)
    if estimates:
        return float(np.median(estimates))
    return 37.0


def middle_series_index(data: np.lib.npyio.NpzFile | Mapping[str, Any], length: int) -> int:
    length = int(length)
    if length <= 1:
        return 0
    times = raw_time_axis_seconds_for_length(data, length)
    if times.size == length:
        finite = np.isfinite(times)
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            target = 0.5 * finite_nanmax(times[finite], default=0.0)
            nearest = int(np.argmin(np.abs(times[finite] - target)))
            return int(finite_indices[nearest])
    return int((length - 1) // 2)


def middle_of_series_key(data: np.lib.npyio.NpzFile | Mapping[str, Any], key: str) -> float:
    arr = get_data_array(data, key, dtype=np.float64)
    if arr is None or arr.size == 0:
        return math.nan
    values = arr.reshape(-1)
    index = middle_series_index(data, int(values.size))
    value = values[index]
    return float(value) if np.isfinite(value) else math.nan


def middle_heatmap_dT_values(
    data: np.lib.npyio.NpzFile | Mapping[str, Any],
    reducer: Callable[[Any], float],
) -> list[float]:
    values: list[float] = []
    for _grid_name, key in thermal_snapshot_keys(data):
        stack = get_data_array(data, key, dtype=np.float64)
        if stack is None or stack.ndim != 3 or stack.shape[0] == 0:
            continue
        snapshot_index = middle_snapshot_index(data, int(stack.shape[0]))
        value = reducer(stack[snapshot_index])
        if np.isfinite(value):
            values.append(float(value))
    return values


def summary_stationary_temperature_mean(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    baseline = temperature_baseline_C(data)
    middle_means = middle_heatmap_dT_values(data, finite_nanmean)
    if middle_means:
        return baseline + float(np.mean(middle_means))
    mean_dT = middle_of_series_key(data, "mean_dT")
    if np.isfinite(mean_dT):
        return baseline + mean_dT
    means: list[float] = []
    for grid_name, key in thermal_volume_keys(data):
        volume = get_data_array(data, key, dtype=np.float64)
        if volume is None:
            continue
        mean_dT = finite_nanmean(volume)
        if np.isfinite(mean_dT):
            means.append(baseline + mean_dT)
    if means:
        return float(np.mean(means))
    temps = final_peak_temperature_scalars(data)
    if temps:
        return float(np.mean(temps))
    mean_dT = max_of_series_key(data, "mean_dT")
    return baseline + mean_dT if np.isfinite(mean_dT) else math.nan


def summary_stationary_temperature_max(data: np.lib.npyio.NpzFile | Mapping[str, Any]) -> float:
    baseline = temperature_baseline_C(data)
    middle_maxima = middle_heatmap_dT_values(data, finite_nanmax)
    if middle_maxima:
        return baseline + float(np.max(middle_maxima))
    max_dT = middle_of_series_key(data, "max_dT")
    if np.isfinite(max_dT):
        return baseline + max_dT
    temps = final_peak_temperature_scalars(data)
    if temps:
        return float(np.max(temps))
    max_values: list[float] = []
    for _grid_name, key in thermal_volume_keys(data):
        volume = get_data_array(data, key, dtype=np.float64)
        if volume is None:
            continue
        max_dT = finite_nanmax(volume)
        if np.isfinite(max_dT):
            max_values.append(baseline + max_dT)
    if max_values:
        return float(np.max(max_values))
    max_dT = max_of_series_key(data, "max_dT")
    return baseline + max_dT if np.isfinite(max_dT) else math.nan


def plot_matrix_summary_grid(
    records_by_key: Mapping[tuple[str, str, str], MatrixRecord],
    factors_by_run_id: Mapping[str, MatrixFactors],
    values_by_run_id: Mapping[str, float],
    output_root: Path,
    image_format: str,
    *,
    metric_name: str,
    stat_name: str,
    title: str,
    unit: str,
    x_factor: str,
    y_factor: str,
    fixed_factor: str,
    pair_name: str,
    overwrite: bool,
) -> None:
    x_values, y_values, fixed_values = pairwise_sheet_values(factors_by_run_id, x_factor, y_factor, fixed_factor)
    if not x_values or not y_values or not fixed_values:
        return
    path = summary_grid_path(output_root, metric_name, stat_name, pair_name, image_format)
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return

    grids: list[np.ndarray] = []
    finite_values: list[float] = []
    for fixed_value in fixed_values:
        values = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
        for row, y_value in enumerate(y_values):
            for col, x_value in enumerate(x_values):
                record = matrix_record_for_values(
                    records_by_key,
                    {x_factor: x_value, y_factor: y_value, fixed_factor: fixed_value},
                )
                if record is None:
                    continue
                value = values_by_run_id.get(record.run_id, math.nan)
                values[row, col] = value
                if np.isfinite(value):
                    finite_values.append(float(value))
        grids.append(values)

    vmin = min(finite_values) if finite_values else 0.0
    vmax = max(finite_values) if finite_values else 1.0
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.05, 1e-6)
        vmin -= delta
        vmax += delta

    fig, axes = plt.subplots(1, len(fixed_values), figsize=(5.8 * len(fixed_values), 4.7), squeeze=False)
    image = None
    cmap = THERMAL_CMAP if is_thermal_metric(metric_name) else "viridis"
    for idx, (ax, fixed_value, values) in enumerate(zip(axes.reshape(-1), fixed_values, grids)):
        image = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                ax.text(
                    col,
                    row,
                    format_summary_number(value),
                    ha="center",
                    va="center",
                    color=contrasting_cell_text_color(
                        value,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap,
                    ),
                    fontsize=8.5,
                    fontweight="bold" if np.isfinite(value) else "normal",
                )
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([value.label for value in x_values], rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([value.label for value in y_values], fontsize=10)
        ax.set_xlabel(factor_display_name(x_factor), fontsize=12)
        if idx == 0:
            ax.set_ylabel(factor_display_name(y_factor), fontsize=12)
        else:
            ax.set_ylabel("")
        if fixed_factor == "preprocessing":
            ax.set_title(fixed_value.label, fontsize=12)
        else:
            ax.set_title(f"Fixed {factor_display_name(fixed_factor)}\n{fixed_value.label}", fontsize=12)

    fig.suptitle(
        f"{title_with_charge_limit(title, metric_name, unit)}\n"
        f"{pair_name.replace('_', ' ')}",
        fontsize=13,
        fontweight="bold",
    )
    top = 0.72 if is_thermal_metric(metric_name) else 0.76
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.22, top=top, wspace=0.55)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.reshape(-1).tolist(), fraction=0.025, pad=0.05)
        cbar.set_label(unit)
        cbar.ax.tick_params(labelsize=8)
    save_figure(fig, path, overwrite=overwrite)


def plot_matrix_active_electrode_errorbars(
    records_by_key: Mapping[tuple[str, str, str], MatrixRecord],
    factors_by_run_id: Mapping[str, MatrixFactors],
    mean_values_by_run_id: Mapping[str, float],
    max_values_by_run_id: Mapping[str, float],
    output_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    amplitude_values = sorted_factor_values(factors_by_run_id, "amplitude")
    grid_values = sorted_factor_values(factors_by_run_id, "grid")
    preprocessing_values = sorted_factor_values(factors_by_run_id, "preprocessing")
    if not amplitude_values or not grid_values or not preprocessing_values:
        return
    amplitude_value = next(
        (
            value
            for value in amplitude_values
            if math.isclose(value.sort_value, STANDARD_AMPLITUDE_UA, rel_tol=0.0, abs_tol=1e-6)
        ),
        None,
    )
    if amplitude_value is None:
        return
    for grid_value in grid_values:
        path = (
            output_root
            / "active_electrodes"
            / f"active_electrodes_{sanitize_path_part(grid_value.key)}.{image_format}"
        )
        if path.exists() and not overwrite:
            print(f"Skipping existing: {path}")
            continue

        means: list[float] = []
        upper_errors: list[float] = []
        for preprocessing_value in preprocessing_values:
            record = matrix_record_for_values(
                records_by_key,
                {
                    "amplitude": amplitude_value,
                    "grid": grid_value,
                    "preprocessing": preprocessing_value,
                },
            )
            mean_value = mean_values_by_run_id.get(record.run_id, math.nan) if record is not None else math.nan
            max_value = max_values_by_run_id.get(record.run_id, math.nan) if record is not None else math.nan
            means.append(mean_value)
            upper_errors.append(max(0.0, max_value - mean_value) if np.isfinite(mean_value) and np.isfinite(max_value) else math.nan)

        x = np.arange(len(preprocessing_values), dtype=np.float64)
        means_array = np.asarray(means, dtype=np.float64)
        upper_errors_array = np.asarray(upper_errors, dtype=np.float64)
        plotted = bool(np.any(np.isfinite(means_array)))
        fig, ax = plt.subplots(figsize=(6.8, 4.9))
        if plotted:
            valid = np.isfinite(means_array)
            bar_colors = [LINE_COLORS[index % len(LINE_COLORS)] for index in range(len(preprocessing_values))]
            ax.bar(
                x[valid],
                means_array[valid],
                width=0.68,
                color=np.asarray(bar_colors, dtype=object)[valid].tolist(),
                edgecolor="#111827",
                linewidth=0.8,
                zorder=3,
            )
            error_valid = valid & np.isfinite(upper_errors_array)
            error_x = x[error_valid]
            error_bottom = means_array[error_valid]
            error_top = error_bottom + upper_errors_array[error_valid]
            ax.vlines(error_x, error_bottom, error_top, color="#111827", linewidth=1.4, zorder=4)
            ax.hlines(error_top, error_x - 0.08, error_x + 0.08, color="#111827", linewidth=1.2, zorder=4)
            style_axes(ax)
        else:
            add_no_data(ax, "No active electrode data")
        ax.set_xticks(x)
        ax.set_xticklabels([value.label for value in preprocessing_values], rotation=20, ha="right")
        ax.set_ylabel("Number of Electrodes")
        ax.set_xlabel("Preprocessing")
        fig.suptitle(f"Activated Electrodes - {grid_value.label}", fontsize=13, fontweight="bold")
        fig.subplots_adjust(bottom=0.20, left=0.12, right=0.97, top=0.88)
        save_figure(fig, path, overwrite=overwrite)


def plot_matrix_shannon_errorbars(
    records_by_key: Mapping[tuple[str, str, str], MatrixRecord],
    factors_by_run_id: Mapping[str, MatrixFactors],
    mean_values_by_run_id: Mapping[str, float],
    max_values_by_run_id: Mapping[str, float],
    output_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    amplitude_values = sorted_factor_values(factors_by_run_id, "amplitude")
    grid_values = sorted_factor_values(factors_by_run_id, "grid")
    preprocessing_values = sorted_factor_values(factors_by_run_id, "preprocessing")
    if not amplitude_values or not grid_values or not preprocessing_values:
        return

    path = output_root / "shannon_k" / f"shannon_k_by_amplitude.{image_format}"
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return
    if overwrite:
        for stale_name in ("mean_shannon_k_by_amplitude", "max_shannon_k_by_amplitude"):
            stale_path = output_root / "shannon_k" / f"{stale_name}.{image_format}"
            if stale_path.exists():
                stale_path.unlink()

    x = np.asarray([value.sort_value for value in amplitude_values], dtype=np.float64)
    fig, axes = plt.subplots(1, len(grid_values), figsize=(4.9 * len(grid_values), 4.4), squeeze=False, sharey=True)
    legend_handles: dict[str, Any] = {}
    for ax, grid_value in zip(axes.reshape(-1), grid_values):
        plotted = False
        for index, preprocessing_value in enumerate(preprocessing_values):
            means = []
            upper_errors = []
            for amplitude_value in amplitude_values:
                record = matrix_record_for_values(
                    records_by_key,
                    {
                        "amplitude": amplitude_value,
                        "grid": grid_value,
                        "preprocessing": preprocessing_value,
                    },
                )
                mean_value = (
                    mean_values_by_run_id.get(record.run_id, math.nan)
                    if record is not None
                    else math.nan
                )
                max_value = (
                    max_values_by_run_id.get(record.run_id, math.nan)
                    if record is not None
                    else math.nan
                )
                means.append(mean_value)
                upper_errors.append(
                    max(0.0, max_value - mean_value)
                    if np.isfinite(mean_value) and np.isfinite(max_value)
                    else math.nan
                )
            means_array = np.asarray(means, dtype=np.float64)
            upper_array = np.asarray(upper_errors, dtype=np.float64)
            if np.any(np.isfinite(means_array)):
                line = ax.plot(
                    x,
                    means_array,
                    marker=MATRIX_MARKERS[index % len(MATRIX_MARKERS)],
                    linestyle=MATRIX_LINESTYLES[index % len(MATRIX_LINESTYLES)],
                    linewidth=1.8,
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgewidth=1.1,
                    color=LINE_COLORS[index % len(LINE_COLORS)],
                    label=preprocessing_value.label,
                )[0]
                error_valid = np.isfinite(means_array) & np.isfinite(upper_array)
                error_x = x[error_valid]
                error_bottom = means_array[error_valid]
                error_top = error_bottom + upper_array[error_valid]
                cap_half_width = max(float(np.ptp(x)) * 0.015, 0.5)
                ax.vlines(
                    error_x,
                    error_bottom,
                    error_top,
                    color=LINE_COLORS[index % len(LINE_COLORS)],
                    linewidth=1.2,
                )
                ax.hlines(
                    error_top,
                    error_x - cap_half_width,
                    error_x + cap_half_width,
                    color=LINE_COLORS[index % len(LINE_COLORS)],
                    linewidth=1.2,
                )
                legend_handles.setdefault(preprocessing_value.label, line)
                plotted = True
        if plotted:
            style_axes(ax)
        else:
            add_no_data(ax, "No Shannon K data")
        ax.set_title(grid_value.label)
        ax.set_xlabel("Amplitude (uA)")
        ax.set_xticks(x)
        ax.set_xticklabels([value.label.replace(" uA", "") for value in amplitude_values])
    axes.reshape(-1)[0].set_ylabel("Shannon K")
    if legend_handles:
        fig.legend(
            list(legend_handles.values()),
            list(legend_handles.keys()),
            loc="lower center",
            ncol=len(legend_handles),
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
    fig.suptitle("Shannon K by Amplitude (mean with upper error to maximum)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.14, 1, 0.91])
    save_figure(fig, path, overwrite=overwrite)


def plot_matrix_charge_distribution_boxplots(
    records_by_key: Mapping[tuple[str, str, str], MatrixRecord],
    factors_by_run_id: Mapping[str, MatrixFactors],
    output_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    amplitude_values = sorted_factor_values(factors_by_run_id, "amplitude")
    grid_values = sorted_factor_values(factors_by_run_id, "grid")
    preprocessing_values = sorted_factor_values(factors_by_run_id, "preprocessing")
    if not amplitude_values or not grid_values or not preprocessing_values:
        return
    path = output_root / "total_charge" / f"electrode_distribution_boxplots.{image_format}"
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        return

    distributions: list[np.ndarray] = []
    positions: list[float] = []
    labels: list[str] = []
    colors: list[str] = []
    grid_centers: list[tuple[float, str]] = []
    prep_centers: list[tuple[float, str]] = []
    separators: list[float] = []

    position = 1.0
    for preprocessing_value in preprocessing_values:
        prep_start = position
        for grid_value in grid_values:
            grid_start = position
            for amp_index, amplitude_value in enumerate(amplitude_values):
                record = matrix_record_for_values(
                    records_by_key,
                    {
                        "amplitude": amplitude_value,
                        "grid": grid_value,
                        "preprocessing": preprocessing_value,
                    },
                )
                if record is None:
                    values = np.asarray([math.nan], dtype=np.float64)
                else:
                    with np.load(record.npz_path, allow_pickle=True) as data:
                        values = final_charge_distribution_mC(data)
                    if values.size == 0:
                        values = np.asarray([math.nan], dtype=np.float64)
                distributions.append(values)
                positions.append(position)
                labels.append(amplitude_value.label.replace(" uA", ""))
                colors.append(LINE_COLORS[amp_index % len(LINE_COLORS)])
                position += 1.0
            grid_centers.append(((grid_start + position - 1.0) / 2.0, grid_value.label))
            position += 0.55
        prep_centers.append(((prep_start + position - 1.55) / 2.0, preprocessing_value.label))
        separators.append(position - 0.25)
        position += 1.0

    fig, ax = plt.subplots(figsize=(max(13.0, 0.48 * len(distributions) + 5.0), 5.6))
    try:
        box = ax.boxplot(distributions, positions=positions, widths=0.62, tick_labels=labels, showfliers=False, patch_artist=True)
    except TypeError:
        box = ax.boxplot(distributions, positions=positions, widths=0.62, labels=labels, showfliers=False, patch_artist=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, alpha=0.55, edgecolor="#111827", linewidth=0.8)
    for median in box["medians"]:
        median.set(color="#111827", linewidth=1.2)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Final charge (mC/electrode)")
    ax.set_title("Total Charge Electrode Distributions")
    for separator in separators[:-1]:
        ax.axvline(separator, color=GRID_COLOR, linewidth=1.0)
    for center, label in grid_centers:
        ax.text(center, -0.12, label.split(" (")[0], transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
    for center, label in prep_centers:
        ax.text(center, -0.27, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9, fontweight="bold")
    legend_handles = [
        plt.Line2D([0], [0], color=LINE_COLORS[index % len(LINE_COLORS)], linewidth=7, alpha=0.55, label=value.label)
        for index, value in enumerate(amplitude_values)
    ]
    ax.legend(handles=legend_handles, title="amplitude", frameon=False, fontsize=8, title_fontsize=8, loc="upper right")
    style_axes(ax)
    fig.subplots_adjust(bottom=0.34, left=0.07, right=0.98, top=0.88)
    save_figure(fig, path, overwrite=overwrite)


def plot_matrix_pairwise_comparisons(
    records: list[MatrixRecord],
    output_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    selected = [record for record in records if record.block == MATRIX_BLOCK]
    if not selected:
        return
    factors_by_run_id, records_by_key = matrix_factor_maps(selected)
    selected_by_run = {record.run_id: record for record in selected}

    active_mean = matrix_summary_values(selected_by_run.values(), summary_active_electrodes_mean)
    active_max = matrix_summary_values(selected_by_run.values(), summary_active_electrodes_max)
    plot_matrix_active_electrode_errorbars(
        records_by_key,
        factors_by_run_id,
        active_mean,
        active_max,
        output_root,
        image_format,
        overwrite=overwrite,
    )
    shannon_mean = matrix_summary_values(selected_by_run.values(), summary_shannon_mean)
    shannon_max = matrix_summary_values(selected_by_run.values(), summary_shannon_max)
    plot_matrix_shannon_errorbars(
        records_by_key,
        factors_by_run_id,
        shannon_mean,
        shannon_max,
        output_root,
        image_format,
        overwrite=overwrite,
    )

    summary_specs: tuple[
        tuple[str, str, str, str, Callable[[np.lib.npyio.NpzFile | Mapping[str, Any]], float]],
        ...,
    ] = (
        ("charge_per_second", "mean", "Mean Charge Per Second Per Electrode", "uC/s/electrode", summary_charge_rate_mean),
        ("charge_per_second", "max", "Max Charge Per Second Per Electrode", "uC/s/electrode", summary_charge_rate_max),
        ("summed_charge_per_second", "mean", "Mean Summed Charge Per Second", "mC/s", summary_total_charge_rate_mean_mC_s),
        ("summed_charge_per_second", "max", "Max Summed Charge Per Second", "mC/s", summary_total_charge_rate_max_mC_s),
        ("total_charge", "summed", "Summed Total Charge", "mC", total_protocol_charge_mC),
        ("max_mean_temperature", "max", "Max Mean Temperature Rise", "C", lambda data: max_of_series_key(data, "mean_dT")),
        ("max_focal_temperature", "max", "Max Focal Temperature Rise", "C", scalar_max_dT),
    )
    for metric_name, stat_name, title, unit, getter in summary_specs:
        values_by_run_id = matrix_summary_values(selected_by_run.values(), getter)
        for x_factor, y_factor, fixed_factor, pair_name in PAIRWISE_SPECS:
            plot_matrix_summary_grid(
                records_by_key,
                factors_by_run_id,
                values_by_run_id,
                output_root,
                image_format,
                metric_name=metric_name,
                stat_name=stat_name,
                title=title,
                unit=unit,
                x_factor=x_factor,
                y_factor=y_factor,
                fixed_factor=fixed_factor,
                pair_name=pair_name,
                overwrite=overwrite,
            )

    plot_matrix_charge_distribution_boxplots(
        records_by_key,
        factors_by_run_id,
        output_root,
        image_format,
        overwrite=overwrite,
    )


def raster_protocol_grid_position(manifest: Mapping[str, Any]) -> tuple[int, int]:
    mode = str(manifest.get("raster_mode", "none")).strip().lower().replace("-", "_")
    if mode in {"none", "off"}:
        return 0, 0
    mode_row = {
        "checkerboard": 1,
        "random": 2,
        "pseudo_random": 2,
        "pseudorandom": 2,
    }.get(mode)
    groups = int(manifest.get("raster_groups", 0) or 0)
    group_col = {3: 1, 4: 2, 5: 3}.get(groups)
    if mode_row is None or group_col is None:
        raise ValueError(f"Unsupported raster protocol: mode={mode!r}, groups={groups!r}")
    return mode_row, group_col


def plot_raster_protocol_metric_grid(
    records: list[MatrixRecord],
    output_root: Path,
    image_format: str,
    *,
    filename: str,
    title: str,
    unit: str,
    getter: Callable[[np.lib.npyio.NpzFile | Mapping[str, Any]], float],
    ic_power_mw: float | None = None,
    overwrite: bool,
) -> None:
    selected = records_by_block(records, "raster_protocols")
    if ic_power_mw is not None:
        selected = [
            record
            for record in selected
            if np.isclose(
                float(manifest_for_record(record).get("internal_circuit_power_mw", 0.0)),
                float(ic_power_mw),
            )
        ]
    if not selected:
        return
    values = np.full((3, 4), np.nan, dtype=np.float64)
    for record in selected:
        row, col = raster_protocol_grid_position(manifest_for_record(record))
        if np.isfinite(values[row, col]):
            raise ValueError(f"Duplicate raster protocol grid cell for: {record.run_id}")
        with np.load(record.npz_path, allow_pickle=True) as data:
            values[row, col] = getter(data)

    finite = values[np.isfinite(values)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.05, 1e-6)
        vmin -= delta
        vmax += delta

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    image = ax.imshow(np.ma.masked_invalid(values), cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            ax.text(
                col,
                row,
                format_summary_number(value) if np.isfinite(value) else "",
                ha="center",
                va="center",
                color=contrasting_cell_text_color(
                    value,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                ),
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xticks(range(4))
    ax.set_xticklabels(("off", "3 groups\n5 Hz", "4 groups\n3.75 Hz", "5 groups\n3 Hz"))
    ax.set_yticks(range(3))
    ax.set_yticklabels(("raster off", "checkerboard", "pseudo random"))
    ax.set_xlabel("Protocol Grouping")
    ax.set_ylabel("Raster Mode")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(unit)
    save_figure(
        fig,
        output_root / f"{sanitize_path_part(filename)}.{image_format}",
        overwrite=overwrite,
    )


def raster_protocol_metric_values(
    records: list[MatrixRecord],
    getter: Callable[[np.lib.npyio.NpzFile | Mapping[str, Any]], float],
    *,
    ic_power_mw: float,
) -> np.ndarray:
    values = np.full((3, 4), np.nan, dtype=np.float64)
    for record in records_by_block(records, "raster_protocols"):
        manifest = manifest_for_record(record)
        power = float(manifest.get("internal_circuit_power_mw", 0.0))
        if not np.isclose(power, float(ic_power_mw)):
            continue
        row, col = raster_protocol_grid_position(manifest)
        if np.isfinite(values[row, col]):
            raise ValueError(
                f"Duplicate raster protocol grid cell at {power:g} mW for: {record.run_id}"
            )
        with np.load(record.npz_path, allow_pickle=True) as data:
            values[row, col] = getter(data)
    return values


def plot_raster_protocol_temperature_grid(
    records: list[MatrixRecord],
    output_root: Path,
    image_format: str,
    *,
    filename: str,
    title: str,
    getter: Callable[[np.lib.npyio.NpzFile | Mapping[str, Any]], float],
    overwrite: bool,
) -> None:
    powers = sorted(
        {
            float(manifest_for_record(record).get("internal_circuit_power_mw", 0.0))
            for record in records_by_block(records, "raster_protocols")
        }
    )
    if not powers:
        return
    grids = [
        raster_protocol_metric_values(records, getter, ic_power_mw=power)
        for power in powers
    ]
    finite = np.concatenate([grid[np.isfinite(grid)] for grid in grids])
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.05, 1e-6)
        vmin -= delta
        vmax += delta

    fig, axes = plt.subplots(
        1,
        len(powers),
        figsize=(4.7 * len(powers), 4.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    image = None
    for index, (ax, power, values) in enumerate(zip(axes.reshape(-1), powers, grids)):
        image = ax.imshow(
            np.ma.masked_invalid(values),
            cmap=THERMAL_CMAP,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                if np.isfinite(value):
                    ax.text(
                        col,
                        row,
                        format_summary_number(value),
                        ha="center",
                        va="center",
                        color=contrasting_cell_text_color(
                            value,
                            vmin=vmin,
                            vmax=vmax,
                            cmap=THERMAL_CMAP,
                        ),
                        fontsize=8.5,
                        fontweight="bold",
                    )
        ax.set_xticks(range(4))
        ax.set_xticklabels(("off", "3 groups\n5 Hz", "4 groups\n3.75 Hz", "5 groups\n3 Hz"))
        ax.set_yticks(range(3))
        ax.set_yticklabels(("raster off", "checkerboard", "pseudo random"))
        ax.set_xlabel("Protocol Grouping")
        if index == 0:
            ax.set_ylabel("Raster Mode")
        ax.set_title(f"{power:g} mW IC power")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.18, top=0.80, wspace=0.12)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.reshape(-1).tolist(), fraction=0.018, pad=0.02)
        cbar.set_label("C")
    save_figure(
        fig,
        output_root / f"{sanitize_path_part(filename)}.{image_format}",
        overwrite=overwrite,
    )


def plot_raster_protocol_comparison_grids(
    records: list[MatrixRecord],
    output_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    specs = (
        (
            "summed_charge_per_second_grid",
            "Mean Summed Charge Per Second Across Raster Protocols",
            "mC/s",
            summary_total_charge_rate_mean_mC_s,
        ),
        (
            "summed_total_protocol_charge_grid",
            "Summed Total Protocol Charge Across Raster Protocols",
            "mC",
            total_protocol_charge_mC,
        ),
    )
    for filename, title, unit, getter in specs:
        plot_raster_protocol_metric_grid(
            records,
            output_root,
            image_format,
            filename=filename,
            title=title,
            unit=unit,
            getter=getter,
            ic_power_mw=0.0,
            overwrite=overwrite,
        )
    plot_raster_protocol_temperature_grid(
        records,
        output_root,
        image_format,
        filename="max_mean_temperature_grid",
        title="Maximum Mean Temperature Rise Across Raster Protocols and IC Power",
        getter=lambda data: max_of_series_key(data, "mean_dT"),
        overwrite=overwrite,
    )
    plot_raster_protocol_temperature_grid(
        records,
        output_root,
        image_format,
        filename="max_focal_temperature_grid",
        title="Maximum Focal Temperature Rise Across Raster Protocols and IC Power",
        getter=scalar_max_dT,
        overwrite=overwrite,
    )


def plot_block_comparative_suite(
    records: list[MatrixRecord],
    block: str,
    output_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    selected = records_by_block(records, block)
    if not selected:
        return

    out_dir = comparison_output_dir(output_root, block)
    if block != THERMAL_BLOCK:
        plot_records_series(
            selected,
            output_path=out_dir / "current_amplitude_over_time",
            image_format=image_format,
            overwrite=overwrite,
            title=f"Current Amplitude Over Time - {block.replace('_', ' ')}",
            ylabel="mean active current (uA)",
            getter=lambda data: mean_metric_series(data, "amplitude", max_points=MAX_SERIES_POINTS),
            show_legend=False,
        )
        plot_records_series(
            selected,
            output_path=out_dir / "activated_electrodes_over_time",
            image_format=image_format,
            overwrite=overwrite,
            title=f"Activated Electrodes Over Time - {block.replace('_', ' ')}",
            ylabel="active electrodes",
            getter=lambda data: active_count_series(data, max_points=MAX_SERIES_POINTS),
            show_legend=False,
        )
        plot_records_series(
            selected,
            output_path=out_dir / "shannon_k_over_time",
            image_format=image_format,
            overwrite=overwrite,
            title=f"Shannon K Over Time - {block.replace('_', ' ')}",
            ylabel="mean active Shannon K",
            getter=lambda data: mean_metric_series(data, "shannon", max_points=MAX_SERIES_POINTS),
            show_legend=False,
        )
        plot_records_bar(
            selected,
            output_path=out_dir / "mean_shannon_k_bar",
            image_format=image_format,
            overwrite=overwrite,
            title=f"Mean Shannon K - {block.replace('_', ' ')}",
            ylabel="mean Shannon K",
            getter=lambda data: mean_of_series_key(data, "shannon"),
            color=COLORS.get(block, FALLBACK_COLOR),
        )
        plot_charge_rate_bars(selected, output_root, block, image_format, overwrite=overwrite)
        plot_records_bar(
            selected,
            output_path=out_dir / "summed_total_charge_bar",
            image_format=image_format,
            overwrite=overwrite,
            title=f"Summed Total Charge - {block.replace('_', ' ')}",
            ylabel="final total charge (mC)",
            getter=total_protocol_charge_mC,
            color=COLORS.get(block, FALLBACK_COLOR),
        )
        plot_total_charge_boxplot(selected, output_root, block, image_format, overwrite=overwrite)
        plot_charge_heatmap_sheet(selected, output_root, block, image_format, overwrite=overwrite)

    plot_records_bar(
        selected,
        output_path=out_dir / "maximum_mean_temperature_bar",
        image_format=image_format,
        overwrite=overwrite,
        title=f"Maximum Mean Temperature Rise - {block.replace('_', ' ')}",
        ylabel="max mean ΔT (°C)",
        getter=lambda data: max_of_series_key(data, "mean_dT"),
        color=COLORS.get(block, FALLBACK_COLOR),
    )
    plot_records_series(
        selected,
        output_path=out_dir / "mean_temperature_evolution",
        image_format=image_format,
        overwrite=overwrite,
        title=f"Mean Temperature Evolution - {block.replace('_', ' ')}",
        ylabel="mean ΔT (°C)",
        getter=lambda data: series_from_key(data, "mean_dT", max_points=MAX_SERIES_POINTS),
    )
    plot_records_series(
        selected,
        output_path=out_dir / "max_focal_temperature_evolution",
        image_format=image_format,
        overwrite=overwrite,
        title=f"Max Focal Temperature Evolution - {block.replace('_', ' ')}",
        ylabel="max focal ΔT (°C)",
        getter=lambda data: series_from_key(data, "max_dT", max_points=MAX_SERIES_POINTS),
    )
    if block == THERMAL_BLOCK:
        plot_records_bar(
            selected,
            output_path=out_dir / "maximum_focal_temperature_bar",
            image_format=image_format,
            overwrite=overwrite,
            title="Maximum Focal Temperature Rise - internal circuit",
            ylabel="max focal ΔT (°C)",
            getter=lambda data: max_of_series_key(data, "max_dT"),
            color=COLORS.get(block, FALLBACK_COLOR),
        )


def plot_amplitude_grid_double_comparisons(
    records: list[MatrixRecord],
    output_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    selected = sorted(
        [
            record for record in records
            if record.block in {"amplitude", "electrode_density"}
        ],
        key=lambda record: (record.block, record.sort_value, record.run_id),
    )
    if not selected:
        return
    out_dir = output_root / "amplitude_vs_electrode_density"
    plot_records_bar(
        selected,
        output_path=out_dir / "charge_over_whole_protocol",
        image_format=image_format,
        overwrite=overwrite,
        title="Charge Over Whole Protocol - amplitude and electrode grids",
        ylabel="final total charge (mC)",
        getter=total_protocol_charge_mC,
        color="#1F4E79",
    )
    plot_records_bar(
        selected,
        output_path=out_dir / "maximum_mean_temperature_bar",
        image_format=image_format,
        overwrite=overwrite,
        title="Maximum Mean Temperature - amplitude and electrode grids",
        ylabel="max mean ΔT (°C)",
        getter=lambda data: max_of_series_key(data, "mean_dT"),
        color="#2E8B57",
    )
    plot_records_bar(
        selected,
        output_path=out_dir / "maximum_focal_temperature_bar",
        image_format=image_format,
        overwrite=overwrite,
        title="Maximum Focal Temperature - amplitude and electrode grids",
        ylabel="max focal ΔT (°C)",
        getter=lambda data: max_of_series_key(data, "max_dT"),
        color="#8B1E3F",
    )
    plot_records_series(
        selected,
        output_path=out_dir / "mean_temperature_evolution",
        image_format=image_format,
        overwrite=overwrite,
        title="Mean Temperature Evolution - amplitude and electrode grids",
        ylabel="mean ΔT (°C)",
        getter=lambda data: series_from_key(data, "mean_dT", max_points=MAX_SERIES_POINTS),
    )
    plot_records_series(
        selected,
        output_path=out_dir / "max_focal_temperature_evolution",
        image_format=image_format,
        overwrite=overwrite,
        title="Max Focal Temperature Evolution - amplitude and electrode grids",
        ylabel="max focal ΔT (°C)",
        getter=lambda data: series_from_key(data, "max_dT", max_points=MAX_SERIES_POINTS),
    )


def plot_comparative_suites(
    records: list[MatrixRecord],
    output_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> None:
    comparison_root = output_root
    for block in sorted({record.block for record in records}, key=block_index):
        if block == MATRIX_BLOCK:
            plot_matrix_pairwise_comparisons(records, comparison_root, image_format, overwrite=overwrite)
            continue
        plot_block_comparative_suite(records, block, comparison_root, image_format, overwrite=overwrite)
    plot_amplitude_grid_double_comparisons(records, comparison_root, image_format, overwrite=overwrite)
    write_peak_temperature_heatmaps(
        records,
        comparison_root,
        image_format=image_format,
        overwrite=overwrite,
    )


def plot_overall_safety(records: list[MatrixRecord], output_root: Path, image_format: str, *, overwrite: bool) -> None:
    if not records:
        return
    labels = [f"{record.block}\n{record.label}" for record in records]
    values = [record.metrics.get("safety_margin", math.nan) for record in records]
    colors = [COLORS.get(record.block, FALLBACK_COLOR) for record in records]

    fig, ax = plt.subplots(figsize=(max(10.0, 0.55 * len(records)), 5.2))
    ax.bar(range(len(records)), values, color=colors, alpha=0.9)
    ax.axhline(1.0, color="#8B1E3F", linestyle="--", linewidth=1.2, label="safety limit")
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Worst safety-limit ratio")
    ax.set_title("Safety margin across experiment matrix")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    save_figure(fig, output_root / f"overall_safety_margin.{image_format}", overwrite=overwrite)


def write_summary_csv(records: list[MatrixRecord], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "matrix_summary.csv"
    metric_keys = list(records[0].metrics.keys()) if records else []
    ratio_keys = list(records[0].ratios.keys()) if records else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["block", "run_id", "label", "npz_path", *metric_keys, *[f"ratio_{key}" for key in ratio_keys]],
        )
        writer.writeheader()
        for record in records:
            row = {
                "block": record.block,
                "run_id": record.run_id,
                "label": record.label.replace("\n", " "),
                "npz_path": str(record.npz_path),
            }
            row.update(record.metrics)
            row.update({f"ratio_{key}": value for key, value in record.ratios.items()})
            writer.writerow(row)
    print(f"wrote: {path}")


def write_single_case_overviews(
    records: list[MatrixRecord],
    image_format: str,
    *,
    overwrite: bool,
    output_root: Path | None = None,
) -> None:
    previous_root = _CASE_VISUALS_ROOT
    if output_root is not None:
        set_case_visuals_root(output_root)
    try:
        for record in records:
            out_path = case_visuals_dir(record) / f"single_case_overview.{image_format}"
            if out_path.exists() and not overwrite:
                print(f"Skipping existing: {out_path}")
                continue
            data = single_case.load_npz_dict(record.npz_path)
            manifest = single_case.load_manifest(record.npz_path)
            fig = single_case.build_figure(record.npz_path, data, manifest)
            if ensure_output(out_path, overwrite=overwrite):
                fig.savefig(out_path, dpi=220, bbox_inches="tight")
                print(f"wrote: {out_path}")
            plt.close(fig)
    finally:
        set_case_visuals_root(previous_root)


def main() -> None:
    args = parse_args()
    input_root = resolve_repo_path(args.input_root)
    output_root = resolve_repo_path(args.output_root)
    safety_yaml = resolve_repo_path(args.safety_yaml)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    records = discover_records(input_root, safety_yaml)
    if not records:
        raise RuntimeError(f"No completed phase 1 metrics.npz runs found under: {input_root}")

    print(f"Discovered {len(records)} matrix runs under: {input_root}")
    set_plot_safety_limits(load_safety_limits(safety_yaml))
    comparative_root = output_root / "comparative_visuals"
    write_summary_csv(records, output_root)
    plot_comparative_suites(records, comparative_root, args.format, overwrite=bool(args.overwrite))
    write_single_case_overviews(records, args.format, overwrite=bool(args.overwrite), output_root=output_root)
    write_per_protocol_visuals(records, args.format, overwrite=bool(args.overwrite), output_root=output_root)


if __name__ == "__main__":
    main()
