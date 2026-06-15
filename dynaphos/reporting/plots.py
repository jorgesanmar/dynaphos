from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dynaphos.safety.evaluation import ReportVisualization


def _safe_key_part(value: object) -> str:
    safe = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in str(value)
    )
    return safe.strip("._") or "item"


def _text_scalar(value: Any) -> str:
    array = np.asarray(value)
    if array.size == 0:
        return ""
    raw = array.reshape(-1)[0]
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)


def _float_scalar(value: Any, default: float = math.nan) -> float:
    try:
        array = np.asarray(value)
        return float(array.reshape(-1)[0]) if array.size else default
    except (TypeError, ValueError):
        return default


def _finite_max(value: Any, default: float = math.nan) -> float:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return default
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if finite.size else default


def _peak_time(data: dict[str, np.ndarray], series_key: str) -> float:
    series = np.asarray(data.get(series_key, []), dtype=np.float64).reshape(-1)
    if series.size == 0 or not np.any(np.isfinite(series)):
        return math.nan
    finite_indices = np.flatnonzero(np.isfinite(series))
    index = int(finite_indices[int(np.argmax(series[finite_indices]))])
    for time_key in ("thermal_time_s", "time_s"):
        times = np.asarray(data.get(time_key, []), dtype=np.float64).reshape(-1)
        if times.size == series.size and np.isfinite(times[index]):
            return float(times[index])
    return math.nan


def _extent_for_grid(
    data: dict[str, np.ndarray],
    grid_name: str,
) -> list[float] | None:
    suffix = _safe_key_part(grid_name)
    for key in (f"extent_mm_{suffix}", "extent_mm"):
        values = np.asarray(data.get(key, []), dtype=np.float64).reshape(-1)
        if values.size >= 4 and np.all(np.isfinite(values[:4])):
            return [float(value) for value in values[:4]]
    return None


def _peak_grid_names(data: dict[str, np.ndarray], peak_kind: str) -> list[str]:
    names: list[str] = []
    for name_key in ("peak_heatmap_grid_names", "heatmap_grid_names"):
        for raw_name in np.asarray(data.get(name_key, [])).reshape(-1):
            name = _text_scalar(raw_name)
            if name and name not in names:
                names.append(name)
    prefixes = (f"dT_peak_{peak_kind}_", "dT_heatmaps_")
    for key in data:
        for prefix in prefixes:
            if key.startswith(prefix):
                name = key.removeprefix(prefix)
                if name and name not in names:
                    names.append(name)
    return names


def _hemisphere_sort_key(grid_name: str) -> tuple[int, str]:
    normalized = str(grid_name).strip().casefold()
    if "left" in normalized:
        return 0, normalized
    if "right" in normalized:
        return 1, normalized
    return 2, normalized


def _peak_maps(
    data: dict[str, np.ndarray],
    peak_kind: str,
) -> list[tuple[str, np.ndarray, list[float] | None, float, bool]]:
    series_key = "max_dT" if peak_kind == "focal" else "mean_dT"
    series_peak_time = _peak_time(data, series_key)
    stored_peak_time = _float_scalar(
        data.get(f"peak_{peak_kind}_time_s"),
        series_peak_time,
    )
    maps: list[tuple[str, np.ndarray, list[float] | None, float, bool]] = []
    snapshot_times = np.asarray(
        data.get("heatmap_times_s", []),
        dtype=np.float64,
    ).reshape(-1)

    for grid_name in _peak_grid_names(data, peak_kind):
        suffix = _safe_key_part(grid_name)
        exact = np.asarray(data.get(f"dT_peak_{peak_kind}_{suffix}", []))
        if exact.ndim == 2 and exact.size:
            maps.append(
                (
                    grid_name,
                    exact.astype(np.float64, copy=False),
                    _extent_for_grid(data, grid_name),
                    stored_peak_time,
                    True,
                )
            )
            continue

        stack = np.asarray(data.get(f"dT_heatmaps_{suffix}", []))
        if stack.ndim != 3 or stack.shape[0] == 0:
            continue
        if (
            snapshot_times.size == stack.shape[0]
            and np.isfinite(series_peak_time)
            and np.any(np.isfinite(snapshot_times))
        ):
            finite_indices = np.flatnonzero(np.isfinite(snapshot_times))
            nearest = int(
                np.argmin(np.abs(snapshot_times[finite_indices] - series_peak_time))
            )
            index = int(finite_indices[nearest])
            map_time = float(snapshot_times[index])
        else:
            summaries = (
                np.nanmax(stack, axis=(1, 2))
                if peak_kind == "focal"
                else np.nanmean(stack, axis=(1, 2))
            )
            finite_indices = np.flatnonzero(np.isfinite(summaries))
            index = (
                int(finite_indices[int(np.argmax(summaries[finite_indices]))])
                if finite_indices.size
                else 0
            )
            map_time = (
                float(snapshot_times[index])
                if snapshot_times.size > index and np.isfinite(snapshot_times[index])
                else math.nan
            )
        maps.append(
            (
                grid_name,
                np.asarray(stack[index], dtype=np.float64),
                _extent_for_grid(data, grid_name),
                map_time,
                False,
            )
        )
    return sorted(maps, key=lambda item: _hemisphere_sort_key(item[0]))


def _relative_asset_path(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir.parent).as_posix()


def _save_figure(
    fig: plt.Figure,
    path: Path,
    output_dir: Path,
    *,
    name: str,
    label: str,
    metric_name: str | None,
    description: str,
) -> ReportVisualization:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return ReportVisualization(
        name=name,
        label=label,
        path=_relative_asset_path(path, output_dir),
        media_type="image/png",
        metric_name=metric_name,
        description=description,
    )


def _write_total_charge_heatmap(
    data: dict[str, np.ndarray],
    output_dir: Path,
) -> ReportVisualization | None:
    charge_nC = np.asarray(
        data.get("protocol_charge_per_electrode_nC", []),
        dtype=np.float64,
    ).reshape(-1)
    xy_mm = np.asarray(data.get("electrode_xy_mm", []), dtype=np.float64)
    if (
        charge_nC.size == 0
        or xy_mm.ndim != 2
        or xy_mm.shape[1] < 2
        or xy_mm.shape[0] != charge_nC.size
    ):
        return None

    charge_uC = charge_nC / 1e3
    total_mC = float(np.nansum(charge_nC)) / 1e6
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    marker_size = float(np.clip(12_000.0 / charge_nC.size, 2.5, 24.0))
    image = ax.scatter(
        xy_mm[:, 0],
        xy_mm[:, 1],
        c=charge_uC,
        cmap="viridis",
        marker="s",
        s=marker_size,
        edgecolors="#24313d",
        linewidths=0.12,
    )
    x_center = 0.5 * (float(np.min(xy_mm[:, 0])) + float(np.max(xy_mm[:, 0])))
    y_center = 0.5 * (float(np.min(xy_mm[:, 1])) + float(np.max(xy_mm[:, 1])))
    spatial_span = max(
        float(np.ptp(xy_mm[:, 0])),
        float(np.ptp(xy_mm[:, 1])),
        1.0,
    )
    half_span = 0.55 * spatial_span
    ax.set_xlim(x_center - half_span, x_center + half_span)
    ax.set_ylim(y_center - half_span, y_center + half_span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Total Protocol Charge\nArray total: {total_mC:.2f} mC")
    ax.grid(alpha=0.2)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Charge per electrode (µC)")
    path = output_dir / "total_charge_heatmap.png"
    return _save_figure(
        fig,
        path,
        output_dir,
        name="total_charge_heatmap",
        label="Total protocol charge heatmap",
        metric_name="session_charge",
        description="Final accumulated protocol charge for each electrode.",
    )


def _write_peak_temperature_heatmap(
    data: dict[str, np.ndarray],
    output_dir: Path,
    peak_kind: str,
) -> ReportVisualization | None:
    maps = _peak_maps(data, peak_kind)
    if not maps:
        return None
    series_key = "max_dT" if peak_kind == "focal" else "mean_dT"
    peak_value = _finite_max(data.get(series_key))
    peak_time = _peak_time(data, series_key)
    vmax = _finite_max(
        np.concatenate([heatmap.reshape(-1) for _, heatmap, _, _, _ in maps]),
        1.0,
    )
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0

    fig, axes = plt.subplots(
        1,
        len(maps),
        figsize=(6.2 * len(maps), 5.6),
        squeeze=False,
    )
    image = None
    for ax, (grid_name, heatmap, extent, map_time, exact) in zip(
        axes.reshape(-1),
        maps,
    ):
        image = ax.imshow(
            heatmap,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="inferno",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        detail = f"{grid_name}\nMap time: {map_time:.2f} s" if np.isfinite(map_time) else grid_name
        if not exact:
            detail += "\nNearest saved snapshot"
        ax.set_title(detail, fontsize=10)
        ax.set_xlabel("x (mm)" if extent is not None else "x pixel")
        ax.set_ylabel("y (mm)" if extent is not None else "y pixel")
        if peak_kind == "focal" and heatmap.size and np.any(np.isfinite(heatmap)):
            row, column = np.unravel_index(int(np.nanargmax(heatmap)), heatmap.shape)
            if extent is None:
                marker_x, marker_y = float(column), float(row)
            else:
                xmin, xmax, ymin, ymax = extent
                marker_x = xmin + (column + 0.5) * (xmax - xmin) / heatmap.shape[1]
                marker_y = ymin + (row + 0.5) * (ymax - ymin) / heatmap.shape[0]
            ax.plot(marker_x, marker_y, "x", color="cyan", markersize=7, mew=1.5)

    if image is not None:
        cbar = fig.colorbar(
            image,
            ax=axes.reshape(-1).tolist(),
            fraction=0.025,
            pad=0.02,
        )
        cbar.set_label("Temperature rise (°C)")
    metric_label = "focal" if peak_kind == "focal" else "spatial mean"
    time_label = f" at {peak_time:.2f} s" if np.isfinite(peak_time) else ""
    fig.suptitle(
        f"Peak {metric_label} temperature rise: {peak_value:.2f} °C{time_label}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(top=0.76, bottom=0.12, wspace=0.22)
    filename = f"max_{peak_kind}_temperature_heatmap.png"
    metric_name = "temperature" if peak_kind == "focal" else "mean_temperature"
    return _save_figure(
        fig,
        output_dir / filename,
        output_dir,
        name=f"max_{peak_kind}_temperature_heatmap",
        label=f"Maximum {metric_label} temperature heatmap",
        metric_name=metric_name,
        description=f"Spatial temperature-rise map at the peak {metric_label} temperature.",
    )


def write_standard_plots(
    metrics_path: str | Path,
    output_dir: str | Path,
) -> list[ReportVisualization]:
    metrics_path = Path(metrics_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(metrics_path, allow_pickle=True) as bundle:
        data = {key: np.asarray(bundle[key]) for key in bundle.files}

    written: list[ReportVisualization] = []
    time_s = np.asarray(data.get("time_s", []), dtype=np.float64)
    amplitude = data.get("amplitude_per_electrode_uA")
    if amplitude is not None and amplitude.ndim == 2 and len(time_s) == amplitude.shape[0]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(time_s, np.max(amplitude, axis=1), label="maximum")
        active = np.isfinite(amplitude) & (amplitude > 0.0)
        active_count = np.sum(active, axis=1)
        active_mean = np.divide(
            np.sum(np.where(active, amplitude, 0.0), axis=1),
            active_count,
            out=np.full(amplitude.shape[0], np.nan, dtype=np.float64),
            where=active_count > 0,
        )
        ax.plot(time_s, active_mean, label="mean active")
        ax.set(xlabel="Time (s)", ylabel="Current (µA)", title="Delivered stimulation")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=2,
            frameon=False,
        )
        ax.grid(alpha=0.25)
        fig.subplots_adjust(bottom=0.25)
        written.append(
            _save_figure(
                fig,
                output_dir / "delivered_stimulation.png",
                output_dir,
                name="delivered_stimulation",
                label="Delivered stimulation",
                metric_name="amplitude",
                description="Maximum and active-electrode mean delivered current over time.",
            )
        )

    thermal_time = np.asarray(data.get("thermal_time_s", []), dtype=np.float64)
    max_dt = np.asarray(data.get("max_dT", []), dtype=np.float64)
    mean_dt = np.asarray(data.get("mean_dT", []), dtype=np.float64)
    if len(thermal_time) and len(thermal_time) == len(max_dt):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(thermal_time, max_dt, label="focal maximum")
        if len(mean_dt) == len(thermal_time):
            ax.plot(thermal_time, mean_dt, label="spatial mean")
        ax.set(
            xlabel="Time (s)",
            ylabel="Temperature rise (°C)",
            title="Thermal response",
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.20),
            ncol=2,
            frameon=False,
        )
        ax.grid(alpha=0.25)
        fig.subplots_adjust(bottom=0.25)
        written.append(
            _save_figure(
                fig,
                output_dir / "thermal_response.png",
                output_dir,
                name="thermal_response",
                label="Thermal response",
                metric_name="temperature",
                description="Focal maximum and spatial mean temperature rise over time.",
            )
        )

    for visualization in (
        _write_total_charge_heatmap(data, output_dir),
        _write_peak_temperature_heatmap(data, output_dir, "focal"),
        _write_peak_temperature_heatmap(data, output_dir, "mean"),
    ):
        if visualization is not None:
            written.append(visualization)
    return written
