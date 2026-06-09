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
from matplotlib.lines import Line2D

from dynaphos.safety.io import resolve_repo_path
from dynaphos.safety.visualize import (
    contrasting_cell_text_color,
    normalize_preprocessing_label,
)


RASTER_ANALYSIS_ROLES = ("worst_case_screen", "transfer_panel")
DEFAULT_TEMPERATURE_LIMIT_C = 2.0
DEFAULT_POWER_DERATING_FACTOR = 0.9
RASTER_MODE_COLORS = {
    "none": "#D62728",
    "checkerboard": "#1F77B4",
    "random": "#2CA02C",
}
RASTER_MODE_LABELS = {
    "none": "No raster",
    "checkerboard": "Checkerboard",
    "random": "Pseudo-random",
}
RASTER_GROUP_MARKERS = {3: "^", 4: "s", 5: "p"}
PHASE3_COMPARATIVE_PLOT_STEMS = {
    "charge_temperature_tradeoff",
    "duty_fraction_comparison",
    "focal_temperature_reduction_fraction",
    "mean_temperature_reduction_fraction",
    "raster_effect_budget_table",
    "total_protocol_charge_reduction_fraction",
    "worst_case_group_trends",
    "worst_case_temperature_evolution",
}
PLOT_SUFFIXES = {".png", ".pdf", ".svg"}


@dataclass(frozen=True)
class SafetyRun:
    npz_path: Path
    manifest_path: Path
    run_id: str
    video: str
    coords_yaml: str
    grid: str
    preprocessing: str
    amplitude_uA: float
    appearance_threshold_uA: float
    frequency_hz: float
    pulse_width_us: float
    internal_circuit_power_mw: float
    raster_mode: str
    raster_groups: int
    duration_s: float
    analysis_role: str
    matched_phase1_run_id: str
    expected_duty_fraction: float
    expected_cycle_rate_hz: float
    total_protocol_charge_mC: float
    max_window_charge_total_nC: float
    max_window_charge_per_electrode_nC: float
    peak_mean_dT_C: float
    peak_focal_dT_C: float
    peak_active_fraction: float
    observed_cycle_rate_hz: float


def finite_max(values: object, default: float = math.nan) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else float(default)


def finite_last(values: object, default: float = math.nan) -> float:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    return float(finite[-1]) if finite.size else float(default)


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def path_identity(value: str | Path) -> str:
    return str(Path(value).resolve()).casefold()


def load_safety_run(npz_path: Path, manifest: dict) -> SafetyRun:
    metadata = manifest.get("metadata", {}) or {}
    with np.load(npz_path, allow_pickle=True) as data:
        protocol_charge = np.asarray(
            data["protocol_charge_per_electrode_nC"]
            if "protocol_charge_per_electrode_nC" in data.files
            else [],
            dtype=np.float64,
        )
        electrode_count = max(
            1,
            int(np.asarray(data["electrode_grid_ids"]).size)
            if "electrode_grid_ids" in data.files
            else int(protocol_charge.size),
        )
        if "active_electrode_count" in data.files:
            peak_active = finite_max(data["active_electrode_count"], 0.0)
        elif "amplitude_per_electrode_uA" in data.files:
            amplitude = np.asarray(data["amplitude_per_electrode_uA"])
            peak_active = float(np.max(np.sum(amplitude > 0.0, axis=1))) if amplitude.ndim == 2 else 0.0
        else:
            peak_active = 0.0
        duration_s = finite_last(data["time_s"]) if "time_s" in data.files else math.nan
        observed_cycle_rate = (
            float(np.asarray(data["raster_rate_hz"]).reshape(-1)[0])
            if "raster_rate_hz" in data.files
            else 0.0
        )
        return SafetyRun(
            npz_path=npz_path,
            manifest_path=npz_path.parent / "run_manifest.yaml",
            run_id=str(manifest.get("run_id", npz_path.parent.name)),
            video=str(manifest.get("video", "")),
            coords_yaml=str(manifest.get("coords_yaml", "")),
            grid=Path(str(manifest.get("coords_yaml", ""))).stem.removeprefix("coords_"),
            preprocessing=normalize_preprocessing_label(
                manifest.get("source_input_label", manifest.get("preprocessing_method", "unknown"))
            ),
            amplitude_uA=float(manifest.get("amplitude_uA", math.nan)),
            appearance_threshold_uA=float(manifest.get("appearance_threshold_uA", math.nan)),
            frequency_hz=float(manifest.get("frequency_hz", math.nan)),
            pulse_width_us=float(manifest.get("pulse_width_us", math.nan)),
            internal_circuit_power_mw=float(manifest.get("internal_circuit_power_mw", math.nan)),
            raster_mode=str(manifest.get("raster_mode_normalized", manifest.get("raster_mode", "none"))),
            raster_groups=int(manifest.get("raster_groups", 1) or 1),
            duration_s=duration_s,
            analysis_role=str(metadata.get("analysis_role", "")),
            matched_phase1_run_id=str(metadata.get("matched_phase1_run_id", "")),
            expected_duty_fraction=float(metadata.get("expected_duty_fraction", math.nan)),
            expected_cycle_rate_hz=float(metadata.get("expected_raster_cycle_rate_hz", 0.0)),
            total_protocol_charge_mC=float(np.nansum(protocol_charge)) / 1e6,
            max_window_charge_total_nC=finite_max(
                data["window_charge_total_nC"] if "window_charge_total_nC" in data.files else []
            ),
            max_window_charge_per_electrode_nC=finite_max(
                data["window_charge_per_electrode_nC"]
                if "window_charge_per_electrode_nC" in data.files
                else []
            ),
            peak_mean_dT_C=finite_max(data["mean_dT"] if "mean_dT" in data.files else []),
            peak_focal_dT_C=finite_max(data["max_dT"] if "max_dT" in data.files else []),
            peak_active_fraction=peak_active / electrode_count,
            observed_cycle_rate_hz=observed_cycle_rate,
        )


def discover_raster_runs(input_root: str | Path) -> list[SafetyRun]:
    root = resolve_repo_path(input_root)
    records: list[SafetyRun] = []
    for npz_path in sorted(root.rglob("safety_metrics.npz")):
        manifest_path = npz_path.parent / "run_manifest.yaml"
        if not manifest_path.exists():
            continue
        manifest = load_manifest(manifest_path)
        metadata = manifest.get("metadata", {}) or {}
        if str(metadata.get("analysis_role", "")) not in RASTER_ANALYSIS_ROLES:
            continue
        records.append(load_safety_run(npz_path, manifest))
    return records


def discover_phase1_runs(input_root: str | Path) -> dict[str, SafetyRun]:
    root = resolve_repo_path(input_root)
    records: dict[str, SafetyRun] = {}
    for npz_path in sorted(root.rglob("safety_metrics.npz")):
        manifest_path = npz_path.parent / "run_manifest.yaml"
        if not manifest_path.exists():
            continue
        manifest = load_manifest(manifest_path)
        if str(manifest.get("block", "")) != "amplitude_grid_preprocessing":
            continue
        record = load_safety_run(npz_path, manifest)
        if record.run_id in records:
            raise ValueError(f"Duplicate phase 1 baseline run_id: {record.run_id}")
        records[record.run_id] = record
    return records


def baseline_mismatch_reasons(raster: SafetyRun, baseline: SafetyRun) -> list[str]:
    reasons: list[str] = []
    comparisons = (
        ("video", path_identity(raster.video), path_identity(baseline.video)),
        ("coords_yaml", path_identity(raster.coords_yaml), path_identity(baseline.coords_yaml)),
        ("preprocessing", raster.preprocessing, baseline.preprocessing),
        ("amplitude_uA", raster.amplitude_uA, baseline.amplitude_uA),
        ("appearance_threshold_uA", raster.appearance_threshold_uA, baseline.appearance_threshold_uA),
        ("frequency_hz", raster.frequency_hz, baseline.frequency_hz),
        ("pulse_width_us", raster.pulse_width_us, baseline.pulse_width_us),
        ("duration_s", raster.duration_s, baseline.duration_s),
        ("internal_circuit_power_mw", raster.internal_circuit_power_mw, baseline.internal_circuit_power_mw),
    )
    for name, actual, expected in comparisons:
        if isinstance(actual, float):
            if not np.isclose(actual, expected, rtol=0.0, atol=1e-6, equal_nan=False):
                reasons.append(name)
        elif actual != expected:
            reasons.append(name)
    if baseline.raster_mode != "none":
        reasons.append("baseline_raster_mode")
    if not np.isclose(raster.internal_circuit_power_mw, 0.0, atol=1e-12):
        reasons.append("raster_ic_power")
    return reasons


def safe_ratio(value: float, baseline: float) -> float:
    if not np.isfinite(value) or not np.isfinite(baseline) or baseline <= 0.0:
        return math.nan
    return value / baseline


def build_raster_effect_rows(
    raster_runs: Iterable[SafetyRun],
    phase1_runs: dict[str, SafetyRun],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raster in raster_runs:
        baseline = phase1_runs.get(raster.matched_phase1_run_id)
        reasons = ["missing_baseline"] if baseline is None else baseline_mismatch_reasons(raster, baseline)
        valid = baseline is not None and not reasons
        metrics = (
            ("total_protocol_charge", "total_protocol_charge_mC"),
            ("window_charge_total", "max_window_charge_total_nC"),
            ("window_charge_per_electrode", "max_window_charge_per_electrode_nC"),
            ("mean_temperature", "peak_mean_dT_C"),
            ("focal_temperature", "peak_focal_dT_C"),
        )
        row: dict[str, object] = {
            "run_id": raster.run_id,
            "protocol_id": f"{raster.raster_mode}__groups_{raster.raster_groups}",
            "analysis_role": raster.analysis_role,
            "matched_phase1_run_id": raster.matched_phase1_run_id,
            "grid": raster.grid,
            "preprocessing": raster.preprocessing,
            "amplitude_uA": raster.amplitude_uA,
            "raster_mode": raster.raster_mode,
            "raster_groups": raster.raster_groups,
            "expected_duty_fraction": raster.expected_duty_fraction,
            "expected_cycle_rate_hz": raster.expected_cycle_rate_hz,
            "observed_cycle_rate_hz": raster.observed_cycle_rate_hz,
            "peak_active_fraction": raster.peak_active_fraction,
            "baseline_match_valid": valid,
            "status": "valid" if valid else "baseline_mismatch",
            "mismatch_fields": ";".join(reasons),
            "npz_path": str(raster.npz_path),
        }
        for prefix, attribute in metrics:
            raster_value = float(getattr(raster, attribute))
            baseline_value = float(getattr(baseline, attribute)) if baseline is not None else math.nan
            ratio = safe_ratio(raster_value, baseline_value) if valid else math.nan
            row[f"raster_{prefix}"] = raster_value
            row[f"baseline_{prefix}"] = baseline_value
            row[f"{prefix}_ratio"] = ratio
            row[f"{prefix}_reduction_fraction"] = 1.0 - ratio if np.isfinite(ratio) else math.nan
        row["observed_charge_duty_fraction"] = row["total_protocol_charge_ratio"]
        row["charge_duty_error"] = (
            float(row["observed_charge_duty_fraction"]) - raster.expected_duty_fraction
            if np.isfinite(float(row["observed_charge_duty_fraction"]))
            else math.nan
        )
        rows.append(row)

    paired = {
        (str(row["matched_phase1_run_id"]), int(row["raster_groups"]), str(row["raster_mode"])): row
        for row in rows
    }
    for row in rows:
        other_mode = "random" if row["raster_mode"] == "checkerboard" else "checkerboard"
        other = paired.get((str(row["matched_phase1_run_id"]), int(row["raster_groups"]), other_mode))
        for metric in ("total_protocol_charge_ratio", "mean_temperature_ratio", "focal_temperature_ratio"):
            row[f"{metric}_vs_other_mode"] = (
                float(row[metric]) - float(other[metric])
                if other is not None and np.isfinite(float(row[metric])) and np.isfinite(float(other[metric]))
                else math.nan
            )
    return rows


def write_rows_csv(rows: list[dict[str, object]], output_root: str | Path) -> Path:
    out_dir = resolve_repo_path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "raster_effect_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def remove_obsolete_phase3_plots(output_root: str | Path) -> list[Path]:
    root = resolve_repo_path(output_root)
    if not root.exists():
        return []
    removed: list[Path] = []
    for path in root.rglob("*"):
        if (
            path.is_file()
            and path.suffix.casefold() in PLOT_SUFFIXES
            and path.stem.casefold() not in PHASE3_COMPARATIVE_PLOT_STEMS
        ):
            path.unlink()
            removed.append(path)
    return removed


def raster_legend_handles(*, include_baseline: bool = False) -> list[Line2D]:
    handles: list[Line2D] = []
    if include_baseline:
        handles.append(
            Line2D(
                [0],
                [0],
                color=RASTER_MODE_COLORS["none"],
                linewidth=2.2,
                label=RASTER_MODE_LABELS["none"],
            )
        )
    handles.extend(
        Line2D(
            [0],
            [0],
            color=RASTER_MODE_COLORS[mode],
            linewidth=2.0,
            label=RASTER_MODE_LABELS[mode],
        )
        for mode in ("checkerboard", "random")
    )
    handles.extend(
        Line2D(
            [0],
            [0],
            color="#444444",
            marker=RASTER_GROUP_MARKERS[groups],
            linestyle="none",
            markersize=7,
            label=f"{groups} groups",
        )
        for groups in (3, 4, 5)
    )
    return handles


def load_ic_power_slopes(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    summary_path = resolve_repo_path(path)
    if not summary_path.exists():
        return {}
    slopes: dict[str, float] = {}
    with open(summary_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("analysis_valid", "")).strip().casefold() not in {"true", "1", "yes"}:
                continue
            try:
                slope = float(row["slope_C_per_mW"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(slope) and slope > 0.0:
                slopes[str(row.get("grid", ""))] = slope
    return slopes


def build_raster_comparison_table_rows(
    rows: list[dict[str, object]],
    slopes_C_per_mW: dict[str, float],
    *,
    temperature_limit_C: float = DEFAULT_TEMPERATURE_LIMIT_C,
    power_derating_factor: float = DEFAULT_POWER_DERATING_FACTOR,
) -> list[dict[str, object]]:
    table_rows: list[dict[str, object]] = []
    for row in rows:
        grid = str(row["grid"])
        slope = float(slopes_C_per_mW.get(grid, math.nan))
        valid = bool(row["baseline_match_valid"]) and np.isfinite(slope) and slope > 0.0
        baseline_temperature = float(row["baseline_mean_temperature"])
        raster_temperature = float(row["raster_mean_temperature"])
        old_raw = (
            max(0.0, (temperature_limit_C - baseline_temperature) / slope)
            if valid and np.isfinite(baseline_temperature)
            else math.nan
        )
        new_raw = (
            max(0.0, (temperature_limit_C - raster_temperature) / slope)
            if valid and np.isfinite(raster_temperature)
            else math.nan
        )
        table_rows.append(
            {
                "grid": grid,
                "preprocessing": row["preprocessing"],
                "amplitude_uA": row["amplitude_uA"],
                "raster_mode": row["raster_mode"],
                "groups": row["raster_groups"],
                "total_charge_no_raster_mC": row["baseline_total_protocol_charge"],
                "total_charge_raster_mC": row["raster_total_protocol_charge"],
                "max_mean_temp_no_raster_C": baseline_temperature,
                "max_mean_temp_raster_C": raster_temperature,
                "charge_reduction_percent": 100.0 * float(row["total_protocol_charge_reduction_fraction"]),
                "temperature_reduction_percent": 100.0 * float(row["mean_temperature_reduction_fraction"]),
                "ic_slope_C_per_mW": slope,
                "old_raw_ic_power_budget_mW": old_raw,
                "new_raw_ic_power_budget_mW": new_raw,
                "old_recommended_ic_power_budget_mW": power_derating_factor * old_raw,
                "new_recommended_ic_power_budget_mW": power_derating_factor * new_raw,
                "temperature_limit_C": temperature_limit_C,
                "power_derating_factor": power_derating_factor,
                "budget_status": "valid" if valid else "invalid_or_missing_ic_slope",
            }
        )
    return table_rows


def write_raster_comparison_table_csv(
    rows: list[dict[str, object]],
    output_root: str | Path,
) -> Path:
    out_dir = resolve_repo_path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "raster_effect_budget_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["grid"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_raster_comparison_table(
    rows: list[dict[str, object]],
    path: Path,
    *,
    overwrite: bool,
) -> Path:
    columns = (
        ("grid", "Grid"),
        ("preprocessing", "Input"),
        ("amplitude_uA", "Amp\n(uA)"),
        ("raster_mode", "Mode"),
        ("groups", "Groups"),
        ("total_charge_no_raster_mC", "Charge off\n(mC)"),
        ("total_charge_raster_mC", "Charge raster\n(mC)"),
        ("max_mean_temp_no_raster_C", "Mean dT off\n(C)"),
        ("max_mean_temp_raster_C", "Mean dT raster\n(C)"),
        ("charge_reduction_percent", "Charge red.\n(%)"),
        ("temperature_reduction_percent", "Temp red.\n(%)"),
        ("old_raw_ic_power_budget_mW", "Old IC budget\n(mW)"),
        ("new_raw_ic_power_budget_mW", "New IC budget\n(mW)"),
    )

    def display(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.3f}" if np.isfinite(float(value)) else "N/A"
        return str(value)

    fig_height = max(4.0, 0.36 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(20.0, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=[[display(row[key]) for key, _ in columns] for row in rows],
        colLabels=[label for _, label in columns],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    for column in range(len(columns)):
        table.auto_set_column_width(column)
    ax.set_title(
        "Raster effects and IC power budget (raw limit at 2 C mean temperature rise)",
        pad=14,
    )
    return save_figure(fig, path, overwrite=overwrite)


def save_figure(fig: plt.Figure, path: Path, *, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        plt.close(fig)
        return path
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def load_mean_temperature_series(run: SafetyRun, *, max_points: int = 1800) -> tuple[np.ndarray, np.ndarray]:
    with np.load(run.npz_path, allow_pickle=True) as data:
        if "mean_dT" not in data.files:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        temperature = np.asarray(data["mean_dT"], dtype=np.float64).reshape(-1)
        time_s = np.arange(temperature.size, dtype=np.float64)
        for key in ("thermal_time_s", "time_s"):
            if key not in data.files:
                continue
            candidate = np.asarray(data[key], dtype=np.float64).reshape(-1)
            if candidate.size == temperature.size:
                time_s = candidate
                break
    finite = np.isfinite(time_s) & np.isfinite(temperature)
    time_s = time_s[finite]
    temperature = temperature[finite]
    if time_s.size > max_points:
        indices = np.linspace(0, time_s.size - 1, max_points).astype(int)
        time_s = time_s[indices]
        temperature = temperature[indices]
    return time_s / 60.0, temperature


def plot_worst_case_temperature_evolution(
    raster_runs: list[SafetyRun],
    phase1_runs: dict[str, SafetyRun],
    path: Path,
    *,
    overwrite: bool,
) -> Path:
    screen = sorted(
        (
            run
            for run in raster_runs
            if run.analysis_role == "worst_case_screen"
            and np.isclose(run.internal_circuit_power_mw, 0.0, atol=1e-12)
        ),
        key=lambda run: (run.raster_mode, run.raster_groups),
    )
    if not screen:
        raise RuntimeError("No 0 mW worst-case screen runs found for the temperature evolution plot.")
    baseline_ids = {run.matched_phase1_run_id for run in screen}
    if len(baseline_ids) != 1:
        raise ValueError("Worst-case screen runs do not share one matched phase 1 baseline.")
    baseline_id = next(iter(baseline_ids))
    baseline = phase1_runs.get(baseline_id)
    if baseline is None:
        raise RuntimeError(f"Missing worst-case phase 1 baseline: {baseline_id}")

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    baseline_time, baseline_temperature = load_mean_temperature_series(baseline)
    ax.plot(
        baseline_time,
        baseline_temperature,
        color=RASTER_MODE_COLORS["none"],
        linewidth=2.2,
        zorder=4,
    )
    for run in screen:
        time_min, temperature = load_mean_temperature_series(run)
        ax.plot(
            time_min,
            temperature,
            color=RASTER_MODE_COLORS[run.raster_mode],
            marker=RASTER_GROUP_MARKERS[run.raster_groups],
            markevery=max(1, time_min.size // 18),
            markersize=5.0,
            linewidth=1.45,
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Spatial mean temperature rise (C)")
    ax.set_title("Worst-case mean temperature evolution at 0 mW IC power")
    ax.grid(alpha=0.25)
    ax.legend(
        handles=raster_legend_handles(include_baseline=True),
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
    )
    fig.subplots_adjust(bottom=0.27)
    return save_figure(fig, path, overwrite=overwrite)


def protocol_label(row: dict[str, object]) -> str:
    mode = "PR" if row["raster_mode"] == "random" else "CB"
    return f"{mode}-{int(row['raster_groups'])}"


def condition_order(rows: list[dict[str, object]]) -> list[str]:
    return list(dict.fromkeys(str(row["matched_phase1_run_id"]) for row in rows))


def plot_reduction_heatmaps(
    rows: list[dict[str, object]],
    output_root: Path,
    image_format: str,
    *,
    overwrite: bool,
) -> list[Path]:
    conditions = condition_order(rows)
    protocols = sorted({protocol_label(row) for row in rows}, key=lambda value: (int(value.split("-")[1]), value))
    paths: list[Path] = []
    for metric, title in (
        ("total_protocol_charge_reduction_fraction", "Total protocol charge reduction"),
        ("mean_temperature_reduction_fraction", "Peak mean temperature reduction"),
        ("focal_temperature_reduction_fraction", "Peak focal temperature reduction"),
    ):
        values = np.full((len(conditions), len(protocols)), np.nan)
        for row in rows:
            values[conditions.index(str(row["matched_phase1_run_id"])), protocols.index(protocol_label(row))] = float(row[metric])
        fig, ax = plt.subplots(figsize=(max(7.0, len(protocols) * 1.0), max(4.0, len(conditions) * 0.55)))
        image = ax.imshow(values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                if np.isfinite(values[y, x]):
                    ax.text(
                        x,
                        y,
                        f"{100 * values[y, x]:.1f}%",
                        ha="center",
                        va="center",
                        color=contrasting_cell_text_color(
                            values[y, x],
                            vmin=0.0,
                            vmax=1.0,
                            cmap="RdYlGn",
                        ),
                        fontsize=8,
                    )
        ax.set_xticks(range(len(protocols)), protocols)
        ax.set_yticks(range(len(conditions)), conditions)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="reduction fraction")
        paths.append(save_figure(fig, output_root / f"{metric}.{image_format}", overwrite=overwrite))
    return paths


def plot_duty_comparison(rows: list[dict[str, object]], path: Path, *, overwrite: bool) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for row in rows:
        observed = float(row["observed_charge_duty_fraction"])
        if not np.isfinite(observed):
            continue
        groups = int(row["raster_groups"])
        ax.scatter(
            float(row["expected_duty_fraction"]),
            observed,
            marker=RASTER_GROUP_MARKERS[groups],
            color=RASTER_MODE_COLORS[str(row["raster_mode"])],
            alpha=0.8,
        )
    ax.plot([0.18, 0.35], [0.18, 0.35], color="#555555", linestyle="--")
    ax.set_xlabel("ideal duty fraction (1 / groups)")
    ax.set_ylabel("observed total-charge ratio")
    handles = raster_legend_handles()
    handles.append(Line2D([0], [0], color="#555555", linestyle="--", label="Ideal"))
    ax.legend(handles=handles, frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    return save_figure(fig, path, overwrite=overwrite)


def plot_worst_case_trends(rows: list[dict[str, object]], path: Path, *, overwrite: bool) -> Path:
    selected = [row for row in rows if row["analysis_role"] == "worst_case_screen"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for mode in ("checkerboard", "random"):
        mode_rows = sorted((row for row in selected if row["raster_mode"] == mode), key=lambda row: int(row["raster_groups"]))
        groups = [int(row["raster_groups"]) for row in mode_rows]
        charge_ratios = [float(row["total_protocol_charge_ratio"]) for row in mode_rows]
        temperature_ratios = [float(row["mean_temperature_ratio"]) for row in mode_rows]
        axes[0].plot(groups, charge_ratios, color=RASTER_MODE_COLORS[mode])
        axes[1].plot(groups, temperature_ratios, color=RASTER_MODE_COLORS[mode])
        for group, charge_ratio, temperature_ratio in zip(groups, charge_ratios, temperature_ratios):
            axes[0].scatter(group, charge_ratio, color=RASTER_MODE_COLORS[mode], marker=RASTER_GROUP_MARKERS[group])
            axes[1].scatter(group, temperature_ratio, color=RASTER_MODE_COLORS[mode], marker=RASTER_GROUP_MARKERS[group])
    axes[0].set_ylabel("total-charge ratio vs phase 1")
    axes[1].set_ylabel("peak mean-temperature ratio vs phase 1")
    for ax in axes:
        ax.set_xlabel("raster groups")
        ax.set_xticks((3, 4, 5))
        ax.grid(alpha=0.25)
    fig.legend(
        handles=raster_legend_handles(),
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.subplots_adjust(bottom=0.22)
    return save_figure(fig, path, overwrite=overwrite)


def plot_mode_comparison(rows: list[dict[str, object]], path: Path, *, overwrite: bool) -> Path:
    valid = [row for row in rows if bool(row["baseline_match_valid"])]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(valid))
    colors = ["#1F4E79" if row["raster_mode"] == "checkerboard" else "#C05621" for row in valid]
    ax.scatter(x, [float(row["mean_temperature_ratio"]) for row in valid], c=colors, label="mean temperature")
    ax.scatter(x, [float(row["total_protocol_charge_ratio"]) for row in valid], c=colors, marker="x", label="charge")
    ax.set_ylabel("ratio vs matched phase 1")
    ax.set_xlabel("valid raster runs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, path, overwrite=overwrite)


def plot_charge_temperature_tradeoff(rows: list[dict[str, object]], path: Path, *, overwrite: bool) -> Path:
    valid = [
        row for row in rows
        if np.isfinite(float(row["total_protocol_charge_ratio"]))
        and np.isfinite(float(row["mean_temperature_ratio"]))
    ]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for row in valid:
        mode = str(row["raster_mode"])
        groups = int(row["raster_groups"])
        ax.scatter(
            float(row["total_protocol_charge_ratio"]),
            float(row["mean_temperature_ratio"]),
            marker=RASTER_GROUP_MARKERS[groups],
            color=RASTER_MODE_COLORS[mode],
            alpha=0.8,
        )
    ax.set_xlabel("total-charge ratio vs phase 1")
    ax.set_ylabel("peak mean-temperature ratio vs phase 1")
    ax.grid(alpha=0.25)
    ax.legend(handles=raster_legend_handles(), frameon=False, ncol=2)
    return save_figure(fig, path, overwrite=overwrite)


def write_summary(rows: list[dict[str, object]], output_root: Path) -> list[Path]:
    transfer = [row for row in rows if row["analysis_role"] == "transfer_panel"]
    protocols = {}
    for groups in (4, 5):
        selected = [row for row in transfer if int(row["raster_groups"]) == groups]
        valid = [row for row in selected if bool(row["baseline_match_valid"])]
        protocols[f"groups_{groups}"] = {
            "valid_runs": len(valid),
            "expected_runs": 10,
            "all_cases_reduce_charge_and_mean_temperature": (
                len(valid) == 10
                and all(float(row["total_protocol_charge_reduction_fraction"]) > 0.0 for row in valid)
                and all(float(row["mean_temperature_reduction_fraction"]) > 0.0 for row in valid)
            ),
            "median_charge_ratio": float(np.median([row["total_protocol_charge_ratio"] for row in valid])) if valid else math.nan,
            "median_mean_temperature_ratio": float(np.median([row["mean_temperature_ratio"] for row in valid])) if valid else math.nan,
        }
    payload = {
        "baseline": "matched phase 1 raster-off run at 0 mW IC power",
        "valid_run_count": sum(bool(row["baseline_match_valid"]) for row in rows),
        "expected_run_count": 26,
        "transfer_consistency": protocols,
    }
    yaml_path = output_root / "raster_effect_summary.yaml"
    text_path = output_root / "raster_effect_summary.txt"
    output_root.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    lines = [
        "Raster safety effects",
        f"Valid matched runs: {payload['valid_run_count']}/{payload['expected_run_count']}",
    ]
    for name, values in protocols.items():
        lines.append(
            f"{name}: transferable={values['all_cases_reduce_charge_and_mean_temperature']} "
            f"median_charge_ratio={values['median_charge_ratio']:.6f} "
            f"median_mean_temperature_ratio={values['median_mean_temperature_ratio']:.6f}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [yaml_path, text_path]


def write_raster_effect_outputs(
    input_root: str | Path,
    phase1_input_root: str | Path,
    output_root: str | Path,
    *,
    ic_power_linearity_summary: str | Path | None = None,
    temperature_limit_C: float = DEFAULT_TEMPERATURE_LIMIT_C,
    power_derating_factor: float = DEFAULT_POWER_DERATING_FACTOR,
    image_format: str = "png",
    overwrite: bool = True,
) -> list[Path]:
    raster_runs = discover_raster_runs(input_root)
    if not raster_runs:
        raise RuntimeError(f"No role-tagged phase 3 raster runs found under: {resolve_repo_path(input_root)}")
    phase1_runs = discover_phase1_runs(phase1_input_root)
    if not phase1_runs:
        raise RuntimeError(f"No phase 1 baselines found under: {resolve_repo_path(phase1_input_root)}")
    rows = build_raster_effect_rows(raster_runs, phase1_runs)
    out_dir = resolve_repo_path(output_root)
    remove_obsolete_phase3_plots(out_dir)
    written = [write_rows_csv(rows, out_dir)]
    table_rows = build_raster_comparison_table_rows(
        rows,
        load_ic_power_slopes(ic_power_linearity_summary),
        temperature_limit_C=temperature_limit_C,
        power_derating_factor=power_derating_factor,
    )
    written.append(write_raster_comparison_table_csv(table_rows, out_dir))
    written.append(
        plot_raster_comparison_table(
            table_rows,
            out_dir / f"raster_effect_budget_table.{image_format}",
            overwrite=overwrite,
        )
    )
    written.extend(plot_reduction_heatmaps(rows, out_dir, image_format, overwrite=overwrite))
    written.append(plot_duty_comparison(rows, out_dir / f"duty_fraction_comparison.{image_format}", overwrite=overwrite))
    written.append(plot_worst_case_trends(rows, out_dir / f"worst_case_group_trends.{image_format}", overwrite=overwrite))
    written.append(plot_charge_temperature_tradeoff(rows, out_dir / f"charge_temperature_tradeoff.{image_format}", overwrite=overwrite))
    written.append(
        plot_worst_case_temperature_evolution(
            raster_runs,
            phase1_runs,
            out_dir / f"worst_case_temperature_evolution.{image_format}",
            overwrite=overwrite,
        )
    )
    written.extend(write_summary(rows, out_dir))
    return written
