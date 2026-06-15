from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from dynaphos.experiment.artifacts import discover_completed_runs
from dynaphos.paths import package_file
from dynaphos.safety.limits import load_safety_limits
from dynaphos.studies.common import resolve_path as resolve_repo_path
from dynaphos.studies.common import sanitize_path_part
from dynaphos.studies.phase1 import (
    add_limit_line,
    contrasting_cell_text_color,
    display_preprocessing_label,
    normalize_preprocessing_label,
    peak_value_and_time,
    video_end_time_minutes,
    write_peak_temperature_heatmaps,
)


POWER_LEVELS_MW = (0.0, 10.0, 20.0, 50.0)
FIT_POWER_LEVELS_MW = (0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 65.0)
FIT_ROLE = "ic_only_fit"
VALIDATION_ROLE = "additivity_validation"
PREPROCESSING_ORDER = ("dog", "canny", "gt")
MAX_SERIES_POINTS = 1600
DEFAULT_SAFETY = package_file("safety")


@dataclass(frozen=True)
class IcPowerRecord:
    npz_path: Path
    manifest_path: Path
    run_id: str
    preprocessing: str
    preprocessing_label: str
    ic_power_mw: float
    amplitude_uA: float
    coords_yaml: str
    grid: str
    analysis_role: str
    electrode_heat_enabled: bool
    reported_total_power_mw: float
    max_focal_dT_C: float
    max_focal_time_s: float
    max_mean_dT_C: float
    max_mean_time_s: float


@dataclass(frozen=True)
class Phase1Record:
    npz_path: Path
    run_id: str
    preprocessing: str
    amplitude_uA: float
    coords_yaml: str
    grid: str
    peak_mean_dT_C: float
    peak_focal_dT_C: float


@dataclass(frozen=True)
class LinearityResult:
    grid: str
    coords_yaml: str
    powers_mw: np.ndarray
    measured_dT_C: np.ndarray
    fitted_dT_C: np.ndarray
    residuals_C: np.ndarray
    slope_C_per_mW: float
    free_intercept_C: float
    free_slope_C_per_mW: float
    r_squared: float
    rmse_C: float
    max_abs_residual_C: float
    full_scale_dT_C: float
    interval_slopes_C_per_mW: np.ndarray
    coverage_complete: bool
    linearity_pass: bool
    additivity_max_abs_residual_C: float
    additivity_pass: bool
    analysis_valid: bool
    status: str


def grid_label(coords_yaml: str | Path) -> str:
    stem = Path(str(coords_yaml)).stem
    return stem.removeprefix("coords_")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def finite_max(values: np.ndarray, default: float = math.nan) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.max(finite))


def series_from_key(data: np.lib.npyio.NpzFile, key: str, *, max_points: int = MAX_SERIES_POINTS) -> tuple[np.ndarray, np.ndarray] | None:
    if key not in data.files:
        return None
    y = np.asarray(data[key], dtype=np.float64).reshape(-1)
    if y.size == 0:
        return None
    x = np.arange(y.size, dtype=np.float64)
    for time_key in ("thermal_time_s", "time_s"):
        if time_key not in data.files:
            continue
        candidate = np.asarray(data[time_key], dtype=np.float64).reshape(-1)
        if candidate.size == y.size:
            x = candidate / 60.0
            break
    if y.size > max_points:
        indices = np.linspace(0, y.size - 1, max_points).astype(int)
        x = x[indices]
        y = y[indices]
    return x, y


def infer_ic_power_from_name(name: str) -> float:
    for token in name.replace("-", "_").split("_"):
        token_l = token.strip().lower()
        if token_l.endswith("mw"):
            try:
                return float(token_l[:-2])
            except ValueError:
                continue
    return math.nan


def discover_ic_power_records(
    input_root: str | Path,
    safety_yaml: str | Path = DEFAULT_SAFETY,
    *,
    analysis_roles: Iterable[str] = (FIT_ROLE, VALIDATION_ROLE),
) -> list[IcPowerRecord]:
    root = resolve_repo_path(input_root)
    accepted_roles = {str(role) for role in analysis_roles}
    records: list[IcPowerRecord] = []
    for run in discover_completed_runs(root):
        npz_path = run.metrics_path
        manifest_path = run.manifest_path
        manifest = run.manifest
        metadata = manifest.get("metadata", {}) or {}
        analysis_role = str(metadata.get("analysis_role", manifest.get("analysis_role", "")))
        if analysis_role not in accepted_roles:
            continue
        preprocessing = normalize_preprocessing_label(
            manifest.get("source_input_label", manifest.get("preprocessing_method", "unknown"))
        )
        power = float(manifest.get("internal_circuit_power_mw", math.nan))
        if not np.isfinite(power):
            power = infer_ic_power_from_name(npz_path.parent.name)
        if not np.isfinite(power):
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            max_focal, max_focal_time_s, _ = peak_value_and_time(data, "max_dT")
            max_mean, max_mean_time_s, _ = peak_value_and_time(data, "mean_dT")
            total_power = (
                float(np.asarray(data["internal_circuit_power_total_mW"]).reshape(-1)[0])
                if "internal_circuit_power_total_mW" in data.files
                else math.nan
            )
        coords_yaml = str(manifest.get("coords_yaml", ""))
        records.append(
            IcPowerRecord(
                npz_path=npz_path,
                manifest_path=manifest_path,
                run_id=str(manifest.get("run_id", npz_path.parent.name)),
                preprocessing=preprocessing,
                preprocessing_label=display_preprocessing_label(preprocessing),
                ic_power_mw=power,
                amplitude_uA=float(manifest.get("amplitude_uA", math.nan)),
                coords_yaml=coords_yaml,
                grid=grid_label(coords_yaml),
                analysis_role=analysis_role,
                electrode_heat_enabled=bool(manifest.get("electrode_heat_enabled", True)),
                reported_total_power_mw=total_power,
                max_focal_dT_C=max_focal,
                max_focal_time_s=max_focal_time_s,
                max_mean_dT_C=max_mean,
                max_mean_time_s=max_mean_time_s,
            )
        )
    return sorted(records, key=record_sort_key)


def record_sort_key(record: IcPowerRecord) -> tuple[float, float, float, str]:
    try:
        preprocessing_index = PREPROCESSING_ORDER.index(record.preprocessing)
    except ValueError:
        preprocessing_index = len(PREPROCESSING_ORDER)
    return (
        0.0 if record.electrode_heat_enabled else 1.0,
        float(preprocessing_index),
        float(record.ic_power_mw),
        record.run_id,
    )


def records_by_preprocessing(records: Iterable[IcPowerRecord]) -> dict[str, list[IcPowerRecord]]:
    grouped: dict[str, list[IcPowerRecord]] = {}
    for record in records:
        grouped.setdefault(record.preprocessing, []).append(record)
    return {key: sorted(value, key=lambda record: record.ic_power_mw) for key, value in grouped.items()}


def driven_records(records: Iterable[IcPowerRecord]) -> list[IcPowerRecord]:
    return [record for record in records if record.electrode_heat_enabled]


def baseline_records_by_power(records: Iterable[IcPowerRecord]) -> dict[tuple[str, float], IcPowerRecord]:
    baselines: dict[tuple[str, float], IcPowerRecord] = {}
    for record in records:
        if record.analysis_role != FIT_ROLE:
            continue
        key = record.grid, float(record.ic_power_mw)
        if key in baselines:
            raise ValueError(f"Multiple IC-only fit records found for {record.grid} at {record.ic_power_mw:g} mW.")
        baselines[key] = record
    return baselines


def condition_key(record: IcPowerRecord) -> tuple[str, str, float]:
    return record.grid, record.preprocessing, float(record.amplitude_uA)


def condition_label(record: IcPowerRecord) -> str:
    return f"{record.grid}, {display_preprocessing_label(record.preprocessing)}, {record.amplitude_uA:g} uA"


def records_by_condition(records: Iterable[IcPowerRecord]) -> dict[tuple[str, str, float], list[IcPowerRecord]]:
    grouped: dict[tuple[str, str, float], list[IcPowerRecord]] = {}
    for record in driven_records(records):
        grouped.setdefault(condition_key(record), []).append(record)
    return {
        key: sorted(value, key=lambda record: record.ic_power_mw)
        for key, value in grouped.items()
    }


def write_temperature_summary_csv(records: list[IcPowerRecord], output_root: str | Path) -> Path:
    out_dir = resolve_repo_path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "temperature_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "preprocessing",
                "ic_power_mw",
                "amplitude_uA",
                "coords_yaml",
                "grid",
                "analysis_role",
                "electrode_heat_enabled",
                "configured_power_per_ic_mw",
                "reported_bilateral_total_power_mw",
                "max_focal_dT_C",
                "max_focal_time_s",
                "max_mean_dT_C",
                "max_mean_time_s",
                "npz_path",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "preprocessing": record.preprocessing,
                    "ic_power_mw": record.ic_power_mw,
                    "amplitude_uA": record.amplitude_uA,
                    "coords_yaml": record.coords_yaml,
                    "grid": record.grid,
                    "analysis_role": record.analysis_role,
                    "electrode_heat_enabled": record.electrode_heat_enabled,
                    "configured_power_per_ic_mw": record.ic_power_mw,
                    "reported_bilateral_total_power_mw": record.reported_total_power_mw,
                    "max_focal_dT_C": record.max_focal_dT_C,
                    "max_focal_time_s": record.max_focal_time_s,
                    "max_mean_dT_C": record.max_mean_dT_C,
                    "max_mean_time_s": record.max_mean_time_s,
                    "npz_path": str(record.npz_path),
                }
            )
    return path


def save_figure(fig: plt.Figure, path: Path, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        print(f"Skipping existing: {path}")
        plt.close(fig)
        return
    fig.savefig(path, dpi=190, bbox_inches="tight")
    print(f"wrote: {path}")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#D7DBE0", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def ic_power_color_scale() -> tuple[plt.Normalize, object]:
    return (
        plt.Normalize(vmin=min(FIT_POWER_LEVELS_MW), vmax=max(FIT_POWER_LEVELS_MW)),
        plt.get_cmap("viridis"),
    )


def plot_temperature_evolution_by_preprocessing(
    records: list[IcPowerRecord],
    output_root: str | Path,
    *,
    image_format: str = "png",
    overwrite: bool = True,
    safety_yaml: str | Path = DEFAULT_SAFETY,
) -> list[Path]:
    limits = load_safety_limits(safety_yaml)
    paths: list[Path] = []
    out_dir = resolve_repo_path(output_root)
    grouped = records_by_condition(records)
    normalization, colormap = ic_power_color_scale()
    for grid, preprocessing, amplitude_uA in sorted(
        grouped,
        key=lambda key: (
            key[0],
            PREPROCESSING_ORDER.index(key[1]) if key[1] in PREPROCESSING_ORDER else 99,
            key[2],
        ),
    ):
        prep_records = grouped[(grid, preprocessing, amplitude_uA)]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
        plotted = False
        video_end_times: set[float] = set()
        for record in prep_records:
            color = colormap(normalization(record.ic_power_mw))
            with np.load(record.npz_path, allow_pickle=True) as data:
                focal = series_from_key(data, "max_dT")
                mean = series_from_key(data, "mean_dT")
                video_end_min = video_end_time_minutes(data)
            label = f"{record.ic_power_mw:g} mW"
            if focal is not None:
                axes[0].plot(focal[0], focal[1], color=color, linewidth=1.9, label=label)
                plotted = True
            if mean is not None:
                axes[1].plot(mean[0], mean[1], color=color, linewidth=1.9, label=label)
                plotted = True
            if np.isfinite(video_end_min):
                video_end_times.add(float(video_end_min))
        for video_end_min in sorted(video_end_times):
            for ax in axes:
                ax.axvline(video_end_min, color="#4B5563", linestyle="--", linewidth=1.1, alpha=0.85)
        axes[0].set_title("Max focal temperature rise")
        axes[0].set_ylabel("Max focal dT (C)")
        axes[1].set_title("Mean temperature rise")
        axes[1].set_ylabel("Mean dT (C)")
        for ax in axes:
            ax.set_xlabel("Time (min)")
            add_limit_line(ax, limits.get("temperature_increase_C", math.nan))
            style_axes(ax)
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="IC power",
                frameon=False,
                fontsize=8,
                title_fontsize=8,
                ncol=min(7, len(handles)),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
            )
            fig.subplots_adjust(bottom=0.22)
        fig.suptitle(
            f"Temperature evolution - {grid}, {display_preprocessing_label(preprocessing)}, {amplitude_uA:g} uA"
        )
        if not plotted:
            for ax in axes:
                ax.text(0.5, 0.5, "No temperature series", ha="center", va="center", transform=ax.transAxes)
        path = out_dir / (
            f"temperature_evolution_{sanitize_path_part(grid)}_{sanitize_path_part(preprocessing)}"
            f"_amp_{amplitude_uA:g}uA.{image_format}"
        )
        save_figure(fig, path, overwrite=overwrite)
        paths.append(path)
    return paths


def plot_ic_only_temperature_evolution(
    records: list[IcPowerRecord],
    output_root: str | Path,
    *,
    image_format: str = "png",
    overwrite: bool = True,
    safety_yaml: str | Path = DEFAULT_SAFETY,
) -> list[Path]:
    limits = load_safety_limits(safety_yaml)
    fit_records = [record for record in records if record.analysis_role == FIT_ROLE]
    grids = sorted({record.grid for record in fit_records})
    powers = sorted({float(record.ic_power_mw) for record in fit_records})
    if not powers:
        return []
    normalization, colormap = ic_power_color_scale()
    out_dir = resolve_repo_path(output_root)
    paths: list[Path] = []

    for grid in grids:
        grid_records = sorted(
            (record for record in fit_records if record.grid == grid),
            key=lambda record: record.ic_power_mw,
        )
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
        plotted = False
        video_end_times: set[float] = set()
        for record in grid_records:
            with np.load(record.npz_path, allow_pickle=True) as data:
                focal = series_from_key(data, "max_dT")
                mean = series_from_key(data, "mean_dT")
                video_end_min = video_end_time_minutes(data)
            color = colormap(normalization(record.ic_power_mw))
            label = f"{record.ic_power_mw:g} mW"
            if focal is not None:
                axes[0].plot(focal[0], focal[1], color=color, linewidth=1.9, label=label)
                plotted = True
            if mean is not None:
                axes[1].plot(mean[0], mean[1], color=color, linewidth=1.9, label=label)
                plotted = True
            if np.isfinite(video_end_min):
                video_end_times.add(float(video_end_min))
        for video_end_min in sorted(video_end_times):
            for ax in axes:
                ax.axvline(video_end_min, color="#4B5563", linestyle="--", linewidth=1.1, alpha=0.85)

        axes[0].set_title("Maximum focal temperature rise")
        axes[0].set_ylabel("Max focal dT (C)")
        axes[1].set_title("Spatial mean temperature rise")
        axes[1].set_ylabel("Mean dT (C)")
        for ax in axes:
            ax.set_xlabel("Time (min)")
            add_limit_line(ax, limits.get("temperature_increase_C", math.nan))
            style_axes(ax)
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="Power per IC / hemisphere",
                frameon=False,
                ncol=min(7, len(handles)),
                loc="lower center",
                bbox_to_anchor=(0.5, -0.03),
                fontsize=8,
                title_fontsize=8,
            )
            fig.subplots_adjust(bottom=0.22)
        fig.suptitle(f"IC-only temperature evolution - {grid}")
        if not plotted:
            for ax in axes:
                ax.text(
                    0.5,
                    0.5,
                    "No IC-only temperature series",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        path = out_dir / f"ic_only_temperature_evolution_{sanitize_path_part(grid)}.{image_format}"
        save_figure(fig, path, overwrite=overwrite)
        paths.append(path)
    return paths


def value_grid(records: list[IcPowerRecord], getter: str) -> tuple[list[str], list[float], np.ndarray]:
    selected = driven_records(records)
    conditions = sorted(
        {condition_key(record) for record in selected},
        key=lambda key: (
            key[0],
            PREPROCESSING_ORDER.index(key[1]) if key[1] in PREPROCESSING_ORDER else 99,
            key[2],
        ),
    )
    labels = [condition_label(next(record for record in selected if condition_key(record) == key)) for key in conditions]
    powers = sorted({float(record.ic_power_mw) for record in selected})
    values = np.full((len(conditions), len(powers)), np.nan, dtype=np.float64)
    for record in selected:
        row = conditions.index(condition_key(record))
        col = powers.index(float(record.ic_power_mw))
        values[row, col] = float(getattr(record, getter))
    return labels, powers, values


def plot_temperature_value_grid(
    records: list[IcPowerRecord],
    output_root: str | Path,
    *,
    getter: str,
    title: str,
    filename: str,
    image_format: str = "png",
    overwrite: bool = True,
) -> Path:
    conditions, powers, values = value_grid(records, getter)
    out_dir = resolve_repo_path(output_root)
    fig, ax = plt.subplots(figsize=(max(6.4, 1.15 * len(powers) + 2.5), max(3.8, 0.55 * len(conditions) + 2.0)))
    finite = values[np.isfinite(values)]
    vmax = float(np.max(finite)) if finite.size else 1.0
    image = ax.imshow(values, cmap="hot", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(powers)))
    ax.set_xticklabels([f"{power:g} mW" for power in powers], fontsize=11)
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_yticklabels(conditions, fontsize=11)
    ax.set_xlabel("IC Power", fontsize=12)
    ax.set_ylabel("Video-Driven Condition", fontsize=12)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "n/a" if not np.isfinite(value) else f"{value:.3g}"
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                color=contrasting_cell_text_color(value, vmin=0.0, vmax=vmax, cmap="hot"),
                fontsize=10,
            )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("dT (C)")
    path = out_dir / f"{filename}.{image_format}"
    save_figure(fig, path, overwrite=overwrite)
    return path


def load_temperature_series(record: IcPowerRecord, key: str) -> tuple[np.ndarray, np.ndarray] | None:
    with np.load(record.npz_path, allow_pickle=True) as data:
        return series_from_key(data, key)


def subtract_temperature_series(
    driven: tuple[np.ndarray, np.ndarray] | None,
    baseline: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if driven is None or baseline is None:
        return None
    x, y = driven
    baseline_x, baseline_y = baseline
    if x.size == 0 or baseline_x.size == 0:
        return None
    baseline_interp = np.interp(x, baseline_x, baseline_y)
    return x, y - baseline_interp


def write_joule_heating_offset_summary_csv(
    records: list[IcPowerRecord],
    output_root: str | Path,
) -> Path:
    out_dir = resolve_repo_path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "joule_heating_offset_summary.csv"
    baselines = baseline_records_by_power(records)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "preprocessing",
                "amplitude_uA",
                "ic_power_mw",
                "max_focal_offset_C",
                "max_mean_offset_C",
                "driven_npz_path",
                "baseline_npz_path",
            ],
        )
        writer.writeheader()
        for record in driven_records(records):
            baseline = baselines.get((record.grid, float(record.ic_power_mw)))
            if baseline is None:
                continue
            focal = subtract_temperature_series(
                load_temperature_series(record, "max_dT"),
                load_temperature_series(baseline, "max_dT"),
            )
            mean = subtract_temperature_series(
                load_temperature_series(record, "mean_dT"),
                load_temperature_series(baseline, "mean_dT"),
            )
            writer.writerow(
                {
                    "preprocessing": record.preprocessing,
                    "amplitude_uA": record.amplitude_uA,
                    "ic_power_mw": record.ic_power_mw,
                    "max_focal_offset_C": finite_max(focal[1]) if focal is not None else math.nan,
                    "max_mean_offset_C": finite_max(mean[1]) if mean is not None else math.nan,
                    "driven_npz_path": str(record.npz_path),
                    "baseline_npz_path": str(baseline.npz_path),
                }
            )
    return path


def plot_joule_heating_offsets(
    records: list[IcPowerRecord],
    output_root: str | Path,
    *,
    image_format: str = "png",
    overwrite: bool = True,
) -> list[Path]:
    out_dir = resolve_repo_path(output_root)
    baselines = baseline_records_by_power(records)
    normalization, colormap = ic_power_color_scale()
    paths: list[Path] = []
    for (grid, preprocessing, amplitude_uA), condition_records in records_by_condition(records).items():
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
        plotted = False
        video_end_times: set[float] = set()
        for record in condition_records:
            baseline = baselines.get((record.grid, float(record.ic_power_mw)))
            if baseline is None:
                continue
            color = colormap(normalization(record.ic_power_mw))
            focal = subtract_temperature_series(
                load_temperature_series(record, "max_dT"),
                load_temperature_series(baseline, "max_dT"),
            )
            mean = subtract_temperature_series(
                load_temperature_series(record, "mean_dT"),
                load_temperature_series(baseline, "mean_dT"),
            )
            with np.load(record.npz_path, allow_pickle=True) as data:
                video_end_min = video_end_time_minutes(data)
            label = f"{record.ic_power_mw:g} mW"
            if focal is not None:
                axes[0].plot(focal[0], focal[1], color=color, linewidth=1.9, label=label)
                plotted = True
            if mean is not None:
                axes[1].plot(mean[0], mean[1], color=color, linewidth=1.9, label=label)
                plotted = True
            if np.isfinite(video_end_min):
                video_end_times.add(float(video_end_min))
        for video_end_min in sorted(video_end_times):
            for ax in axes:
                ax.axvline(video_end_min, color="#4B5563", linestyle="--", linewidth=1.1, alpha=0.85)
        axes[0].set_title("Max focal Joule-heating offset")
        axes[0].set_ylabel("Driven - IC-only dT (C)")
        axes[1].set_title("Mean Joule-heating offset")
        axes[1].set_ylabel("Driven - IC-only dT (C)")
        for ax in axes:
            ax.set_xlabel("Time (min)")
            style_axes(ax)
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                title="IC power",
                frameon=False,
                fontsize=8,
                title_fontsize=8,
                ncol=min(7, len(handles)),
                loc="lower center",
                bbox_to_anchor=(0.5, 0.01),
            )
            fig.subplots_adjust(bottom=0.22)
        fig.suptitle(
            f"Electrode Joule-heating offset - {grid}, "
            f"{display_preprocessing_label(preprocessing)}, {amplitude_uA:g} uA"
        )
        if not plotted:
            for ax in axes:
                ax.text(0.5, 0.5, "No matched baseline", ha="center", va="center", transform=ax.transAxes)
        path = out_dir / (
            f"joule_heating_offset_{sanitize_path_part(grid)}_{sanitize_path_part(preprocessing)}"
            f"_amp_{amplitude_uA:g}uA.{image_format}"
        )
        save_figure(fig, path, overwrite=overwrite)
        paths.append(path)
    return paths


def discover_phase1_records(input_root: str | Path) -> list[Phase1Record]:
    root = resolve_repo_path(input_root)
    records: list[Phase1Record] = []
    for run in discover_completed_runs(root):
        npz_path = run.metrics_path
        manifest = run.manifest
        if str(manifest.get("block", "")) != "amplitude_grid_preprocessing":
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            if "mean_dT" not in data.files or "max_dT" not in data.files:
                continue
            peak_mean = finite_max(np.asarray(data["mean_dT"]))
            peak_focal = finite_max(np.asarray(data["max_dT"]))
        coords_yaml = str(manifest.get("coords_yaml", ""))
        records.append(
            Phase1Record(
                npz_path=npz_path,
                run_id=str(manifest.get("run_id", npz_path.parent.name)),
                preprocessing=normalize_preprocessing_label(
                    manifest.get("source_input_label", manifest.get("preprocessing_method", "unknown"))
                ),
                amplitude_uA=float(manifest.get("amplitude_uA", math.nan)),
                coords_yaml=coords_yaml,
                grid=grid_label(coords_yaml),
                peak_mean_dT_C=peak_mean,
                peak_focal_dT_C=peak_focal,
            )
        )
    return records


def phase1_key(record: Phase1Record) -> tuple[str, str, float]:
    return record.grid, record.preprocessing, float(record.amplitude_uA)


def fit_origin_linearity(
    records: list[IcPowerRecord],
    *,
    expected_powers_mw: Iterable[float] = FIT_POWER_LEVELS_MW,
    r_squared_min: float = 0.999,
    residual_fraction_max: float = 0.01,
) -> LinearityResult:
    if not records:
        raise ValueError("At least one IC-only fit record is required.")
    grid = records[0].grid
    if any(record.grid != grid for record in records):
        raise ValueError("A linearity fit may contain only one electrode grid.")
    by_power: dict[float, IcPowerRecord] = {}
    for record in records:
        power = float(record.ic_power_mw)
        if power in by_power:
            raise ValueError(f"Duplicate IC-only fit record for {grid} at {power:g} mW.")
        by_power[power] = record

    powers = np.asarray(sorted(by_power), dtype=np.float64)
    measured = np.asarray([by_power[power].max_mean_dT_C for power in powers], dtype=np.float64)
    finite = np.isfinite(powers) & np.isfinite(measured)
    powers = powers[finite]
    measured = measured[finite]
    expected = np.asarray(tuple(expected_powers_mw), dtype=np.float64)
    coverage_complete = (
        powers.size == expected.size
        and np.allclose(powers, expected, rtol=0.0, atol=1e-9)
    )

    denominator = float(np.dot(powers, powers))
    slope = float(np.dot(powers, measured) / denominator) if denominator > 0.0 else math.nan
    fitted = slope * powers
    residuals = measured - fitted
    sse = float(np.dot(residuals, residuals))
    centered = measured - float(np.mean(measured)) if measured.size else measured
    sst = float(np.dot(centered, centered))
    r_squared = 1.0 - sse / sst if sst > 0.0 else (1.0 if sse == 0.0 else math.nan)
    rmse = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else math.nan
    max_residual = finite_max(np.abs(residuals))
    full_scale = finite_max(measured, default=0.0) - (
        float(np.min(measured)) if measured.size else 0.0
    )
    if powers.size >= 2:
        free_slope, free_intercept = np.polyfit(powers, measured, 1)
        interval_slopes = np.diff(measured) / np.diff(powers)
    else:
        free_slope = free_intercept = math.nan
        interval_slopes = np.asarray([], dtype=np.float64)
    residual_limit = residual_fraction_max * full_scale
    linearity_pass = bool(
        coverage_complete
        and np.isfinite(slope)
        and slope > 0.0
        and np.isfinite(r_squared)
        and r_squared >= r_squared_min
        and np.isfinite(max_residual)
        and max_residual <= residual_limit
    )
    status = "pass" if linearity_pass else "linearity_failed"
    if not coverage_complete:
        status = "incomplete_power_coverage"
    return LinearityResult(
        grid=grid,
        coords_yaml=records[0].coords_yaml,
        powers_mw=powers,
        measured_dT_C=measured,
        fitted_dT_C=fitted,
        residuals_C=residuals,
        slope_C_per_mW=slope,
        free_intercept_C=float(free_intercept),
        free_slope_C_per_mW=float(free_slope),
        r_squared=r_squared,
        rmse_C=rmse,
        max_abs_residual_C=max_residual,
        full_scale_dT_C=full_scale,
        interval_slopes_C_per_mW=interval_slopes,
        coverage_complete=coverage_complete,
        linearity_pass=linearity_pass,
        additivity_max_abs_residual_C=math.nan,
        additivity_pass=False,
        analysis_valid=False,
        status=status,
    )


def analyze_ic_power(
    records: list[IcPowerRecord],
    phase1_records: list[Phase1Record],
) -> list[LinearityResult]:
    fit_by_grid: dict[str, list[IcPowerRecord]] = {}
    for record in records:
        if record.analysis_role == FIT_ROLE:
            fit_by_grid.setdefault(record.grid, []).append(record)
    phase1_by_key = {phase1_key(record): record for record in phase1_records}
    results: list[LinearityResult] = []
    for grid in sorted(fit_by_grid):
        fit = fit_origin_linearity(fit_by_grid[grid])
        additivity_residuals: list[float] = []
        measured_values: list[float] = []
        validation_records = [
            record
            for record in records
            if record.grid == grid and record.analysis_role == VALIDATION_ROLE
        ]
        for record in validation_records:
            phase1 = phase1_by_key.get((grid, record.preprocessing, float(record.amplitude_uA)))
            if phase1 is None or not np.isfinite(fit.slope_C_per_mW):
                continue
            predicted = phase1.peak_mean_dT_C + fit.slope_C_per_mW * record.ic_power_mw
            additivity_residuals.append(record.max_mean_dT_C - predicted)
            measured_values.append(record.max_mean_dT_C)
        max_additivity_residual = (
            finite_max(np.abs(np.asarray(additivity_residuals, dtype=np.float64)))
            if additivity_residuals
            else math.nan
        )
        additivity_scale = finite_max(np.asarray(measured_values), default=0.0)
        additivity_pass = bool(
            len(additivity_residuals) == 2
            and max_additivity_residual <= 0.01 * additivity_scale
        )
        analysis_valid = fit.linearity_pass and additivity_pass
        status = fit.status
        if fit.linearity_pass and not additivity_pass:
            status = "additivity_failed"
        if analysis_valid:
            status = "pass"
        results.append(
            LinearityResult(
                **{
                    **fit.__dict__,
                    "additivity_max_abs_residual_C": max_additivity_residual,
                    "additivity_pass": additivity_pass,
                    "analysis_valid": analysis_valid,
                    "status": status,
                }
            )
        )
    return results


def write_linearity_summary_csv(results: list[LinearityResult], output_root: str | Path) -> Path:
    out_dir = resolve_repo_path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ic_power_linearity_summary.csv"
    fields = [
        "grid",
        "coords_yaml",
        "power_scope",
        "slope_C_per_mW",
        "free_slope_C_per_mW",
        "free_intercept_C",
        "r_squared",
        "rmse_C",
        "max_abs_residual_C",
        "full_scale_dT_C",
        "interval_slopes_C_per_mW",
        "coverage_complete",
        "linearity_pass",
        "additivity_max_abs_residual_C",
        "additivity_pass",
        "analysis_valid",
        "status",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    **{field: getattr(result, field) for field in fields if hasattr(result, field)},
                    "power_scope": "per_ic_per_hemisphere",
                    "interval_slopes_C_per_mW": ";".join(
                        f"{value:.12g}" for value in result.interval_slopes_C_per_mW
                    ),
                }
            )
    return path


def build_phase1_budget_rows(
    phase1_records: list[Phase1Record],
    results: list[LinearityResult],
    *,
    temperature_limit_C: float,
    power_derating_factor: float,
) -> list[dict[str, object]]:
    results_by_grid = {result.grid: result for result in results}
    rows: list[dict[str, object]] = []
    for record in phase1_records:
        result = results_by_grid.get(record.grid)
        remaining = temperature_limit_C - record.peak_mean_dT_C
        valid = result is not None and result.analysis_valid and result.slope_C_per_mW > 0.0
        raw_power = max(0.0, remaining / result.slope_C_per_mW) if valid else math.nan
        recommended = power_derating_factor * raw_power if valid else math.nan
        status = "valid"
        if not valid:
            status = "invalid_ic_analysis"
        elif remaining <= 0.0:
            status = "mean_limit_already_reached"
        rows.append(
            {
                "run_id": record.run_id,
                "grid": record.grid,
                "preprocessing": record.preprocessing,
                "amplitude_uA": record.amplitude_uA,
                "peak_mean_dT_C": record.peak_mean_dT_C,
                "peak_focal_dT_C": record.peak_focal_dT_C,
                "focal_limit_exceeded": record.peak_focal_dT_C >= temperature_limit_C,
                "temperature_limit_C": temperature_limit_C,
                "remaining_mean_budget_C": remaining,
                "slope_C_per_mW": result.slope_C_per_mW if result is not None else math.nan,
                "raw_max_ic_power_mW": raw_power,
                "power_derating_factor": power_derating_factor,
                "recommended_max_ic_power_mW": recommended,
                "status": status,
                "npz_path": str(record.npz_path),
            }
        )
    return rows


def write_phase1_budget_csv(rows: list[dict[str, object]], output_root: str | Path) -> Path:
    out_dir = resolve_repo_path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "phase1_ic_power_budget.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_linearity_results(
    results: list[LinearityResult],
    output_root: str | Path,
    *,
    image_format: str,
    overwrite: bool,
) -> list[Path]:
    out_dir = resolve_repo_path(output_root)
    paths: list[Path] = []
    normalization, colormap = ic_power_color_scale()
    for result in results:
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        ax.scatter(
            result.powers_mw,
            result.measured_dT_C,
            c=result.powers_mw,
            cmap=colormap,
            norm=normalization,
            edgecolors="#333333",
            linewidths=0.45,
            zorder=3,
            label="simulated data",
        )
        ax.set_xlabel("IC Power per IC / Hemisphere (mW)")
        ax.set_ylabel("Peak mean dT (C)")
        ax.set_title(f"IC thermal linearity - {result.grid} - {result.status}")
        style_axes(ax)
        path = out_dir / f"ic_power_linearity_{sanitize_path_part(result.grid)}.{image_format}"
        save_figure(fig, path, overwrite=overwrite)
        paths.append(path)
    return paths


def plot_phase1_budget(
    rows: list[dict[str, object]],
    output_root: str | Path,
    *,
    image_format: str,
    overwrite: bool,
) -> Path:
    out_dir = resolve_repo_path(output_root)

    def grid_sort_value(value: object) -> tuple[float, str]:
        label = str(value)
        digits = "".join(character for character in label if character.isdigit() or character == ".")
        try:
            return float(digits), label
        except ValueError:
            return math.inf, label

    def preprocessing_sort_value(value: object) -> tuple[int, str]:
        normalized = normalize_preprocessing_label(value)
        try:
            return PREPROCESSING_ORDER.index(normalized), normalized
        except ValueError:
            return len(PREPROCESSING_ORDER), normalized

    ordered = sorted(
        rows,
        key=lambda row: (
            grid_sort_value(row["grid"]),
            float(row["amplitude_uA"]),
            preprocessing_sort_value(row["preprocessing"]),
        ),
    )
    values = np.asarray([float(row["recommended_max_ic_power_mW"]) for row in ordered], dtype=np.float64)
    x = np.arange(len(ordered), dtype=np.float64)
    preprocessing_colors = {
        preprocessing: plt.get_cmap("tab10")(index)
        for index, preprocessing in enumerate(PREPROCESSING_ORDER)
    }
    colors = [
        preprocessing_colors.get(
            normalize_preprocessing_label(row["preprocessing"]),
            "#6B7280",
        )
        for row in ordered
    ]

    fig, ax = plt.subplots(figsize=(max(11.0, 0.48 * len(rows) + 3.0), 6.4))
    valid = np.isfinite(values)
    ax.bar(x[valid], values[valid], color=np.asarray(colors, dtype=object)[valid].tolist(), width=0.78)
    if np.any(~valid):
        ax.scatter(x[~valid], np.zeros(np.count_nonzero(~valid)), marker="x", color="#6B7280", zorder=4)
    ax.set_xticks(x)
    short_preprocessing_labels = {
        "dog": "DoG",
        "canny": "Canny",
        "gt": "Hand\nseg.",
    }
    ax.set_xticklabels(
        [
            short_preprocessing_labels.get(
                normalize_preprocessing_label(row["preprocessing"]),
                display_preprocessing_label(row["preprocessing"]),
            )
            for row in ordered
        ],
        fontsize=7,
    )
    ax.set_ylabel("Recommended Maximum IC Power per IC / Hemisphere (mW)")
    ax.set_title("Phase 1 mean-temperature IC power budget (10% derating)")

    amplitude_groups: list[tuple[int, int, str]] = []
    grid_groups: list[tuple[int, int, str]] = []
    start = 0
    while start < len(ordered):
        grid = str(ordered[start]["grid"])
        amplitude = float(ordered[start]["amplitude_uA"])
        end = start + 1
        while (
            end < len(ordered)
            and str(ordered[end]["grid"]) == grid
            and np.isclose(float(ordered[end]["amplitude_uA"]), amplitude)
        ):
            end += 1
        amplitude_groups.append((start, end, f"{amplitude:g} µA"))
        start = end
    start = 0
    while start < len(ordered):
        grid = str(ordered[start]["grid"])
        end = start + 1
        while end < len(ordered) and str(ordered[end]["grid"]) == grid:
            end += 1
        grid_groups.append((start, end, grid))
        start = end

    for start, end, label in amplitude_groups:
        center = 0.5 * (start + end - 1)
        ax.text(
            center,
            -0.12,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
        )
        if end < len(ordered):
            ax.axvline(end - 0.5, color="#D7DBE0", linewidth=0.8)
    for start, end, label in grid_groups:
        center = 0.5 * (start + end - 1)
        ax.text(
            center,
            -0.22,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        if end < len(ordered):
            ax.axvline(end - 0.5, color="#6B7280", linewidth=1.4)

    style_axes(ax)
    fig.subplots_adjust(bottom=0.28, left=0.08, right=0.98, top=0.90)
    path = out_dir / f"phase1_ic_power_budget.{image_format}"
    save_figure(fig, path, overwrite=overwrite)
    return path


def write_analysis_summary(
    results: list[LinearityResult],
    budget_rows: list[dict[str, object]],
    output_root: str | Path,
) -> list[Path]:
    out_dir = resolve_repo_path(output_root)
    finite_rows = [
        row for row in budget_rows
        if np.isfinite(float(row["recommended_max_ic_power_mW"]))
    ]
    worst = min(finite_rows, key=lambda row: float(row["recommended_max_ic_power_mW"])) if finite_rows else None
    payload = {
        "metric": "peak spatial mean temperature rise",
        "power_scope": "per IC / hemisphere",
        "all_grids_valid": (
            bool(results)
            and all(result.analysis_valid for result in results)
            and all(row["status"] != "invalid_ic_analysis" for row in budget_rows)
        ),
        "grid_results": [
            {
                "grid": result.grid,
                "slope_C_per_mW": result.slope_C_per_mW,
                "r_squared": result.r_squared,
                "analysis_valid": result.analysis_valid,
                "status": result.status,
            }
            for result in results
        ],
        "conservative_limit": (
            {
                "phase1_run_id": worst["run_id"],
                "raw_max_ic_power_mW": worst["raw_max_ic_power_mW"],
                "recommended_max_ic_power_mW": worst["recommended_max_ic_power_mW"],
            }
            if worst is not None
            else None
        ),
        "focal_limit_exceedance_count": sum(bool(row["focal_limit_exceeded"]) for row in budget_rows),
    }
    yaml_path = out_dir / "ic_power_safety_summary.yaml"
    text_path = out_dir / "ic_power_safety_summary.txt"
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    lines = [
        "IC power safety analysis",
        "Metric: peak spatial mean temperature rise",
        "Power scope: per IC / hemisphere",
        f"All grid analyses valid: {payload['all_grids_valid']}",
    ]
    if worst is not None:
        lines.extend(
            [
                f"Conservative phase 1 case: {worst['run_id']}",
                f"Raw maximum IC power: {float(worst['raw_max_ic_power_mW']):.6f} mW",
                f"Recommended maximum IC power: {float(worst['recommended_max_ic_power_mW']):.6f} mW",
            ]
        )
    else:
        lines.append("No valid maximum IC power recommendation is available.")
    lines.append(
        f"Phase 1 cases whose focal rise reaches/exceeds the limit: "
        f"{payload['focal_limit_exceedance_count']}"
    )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [yaml_path, text_path]


def write_ic_power_temperature_visuals(
    input_root: str | Path,
    output_root: str | Path | None = None,
    *,
    phase1_input_root: str | Path = "results/safety/simulation_pipeline/amplitude_grid_preprocessing",
    temperature_limit_C: float = 2.0,
    power_derating_factor: float = 0.9,
    image_format: str = "png",
    overwrite: bool = True,
    safety_yaml: str | Path = DEFAULT_SAFETY,
) -> list[Path]:
    root = resolve_repo_path(input_root)
    out_dir = resolve_repo_path(output_root) if output_root is not None else root / "visuals"
    records = discover_ic_power_records(root, safety_yaml)
    if not records:
        raise RuntimeError(f"No completed phase 2 metrics.npz runs found under: {root}")
    phase1_records = discover_phase1_records(phase1_input_root)
    if not phase1_records:
        raise RuntimeError(f"No completed phase 1 metrics.npz runs found under: {resolve_repo_path(phase1_input_root)}")
    results = analyze_ic_power(records, phase1_records)
    budget_rows = build_phase1_budget_rows(
        phase1_records,
        results,
        temperature_limit_C=temperature_limit_C,
        power_derating_factor=power_derating_factor,
    )

    written = [
        write_temperature_summary_csv(records, out_dir),
        write_linearity_summary_csv(results, out_dir),
        write_phase1_budget_csv(budget_rows, out_dir),
    ]
    written.extend(write_analysis_summary(results, budget_rows, out_dir))
    written.extend(
        plot_linearity_results(
            results,
            out_dir,
            image_format=image_format,
            overwrite=overwrite,
        )
    )
    written.append(
        plot_phase1_budget(
            budget_rows,
            out_dir,
            image_format=image_format,
            overwrite=overwrite,
        )
    )
    written.extend(
        plot_temperature_evolution_by_preprocessing(
            records,
            out_dir,
            image_format=image_format,
            overwrite=overwrite,
            safety_yaml=safety_yaml,
        )
    )
    written.extend(
        plot_ic_only_temperature_evolution(
            records,
            out_dir,
            image_format=image_format,
            overwrite=overwrite,
            safety_yaml=safety_yaml,
        )
    )
    written.extend(
        write_peak_temperature_heatmaps(
            records,
            out_dir,
            image_format=image_format,
            overwrite=overwrite,
        )
    )
    written.append(
        plot_temperature_value_grid(
            records,
            out_dir,
            getter="max_focal_dT_C",
            title="Max focal temperature rise by IC power",
            filename="max_focal_temperature_grid",
            image_format=image_format,
            overwrite=overwrite,
        )
    )
    written.append(
        plot_temperature_value_grid(
            records,
            out_dir,
            getter="max_mean_dT_C",
            title="Max mean temperature rise by IC power",
            filename="max_mean_temperature_grid",
            image_format=image_format,
            overwrite=overwrite,
        )
    )
    if baseline_records_by_power(records):
        written.append(write_joule_heating_offset_summary_csv(records, out_dir))
        written.extend(
            plot_joule_heating_offsets(
                records,
                out_dir,
                image_format=image_format,
                overwrite=overwrite,
            )
        )
    return written


def run_phase2_analysis(
    results_root: str | Path,
    phase1_root: str | Path,
    output_root: str | Path | None = None,
    *,
    image_format: str = "png",
    overwrite: bool = True,
    safety_yaml: str | Path | None = None,
) -> list[Path]:
    root = resolve_repo_path(results_root)
    output = (
        resolve_repo_path(output_root)
        if output_root is not None
        else root / "comparative_visuals"
    )
    return write_ic_power_temperature_visuals(
        root,
        output,
        phase1_input_root=phase1_root,
        image_format=image_format,
        overwrite=overwrite,
        safety_yaml=package_file("safety") if safety_yaml is None else safety_yaml,
    )
