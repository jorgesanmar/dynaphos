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

from dynaphos.safety.io import DEFAULT_SAFETY, load_safety_limits, resolve_repo_path, sanitize_path_part
from dynaphos.safety.visualize import add_limit_line, normalize_preprocessing_label


POWER_LEVELS_MW = (0.0, 10.0, 20.0, 50.0)
PREPROCESSING_ORDER = ("dog", "canny", "gt")
LINE_COLORS = ("#1F4E79", "#2E8B57", "#8B1E3F", "#C05621")
MAX_SERIES_POINTS = 1600


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
    max_focal_dT_C: float
    max_mean_dT_C: float


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
    if "time_s" in data.files:
        x = np.asarray(data["time_s"], dtype=np.float64).reshape(-1) / 60.0
        if x.size != y.size:
            x = np.arange(y.size, dtype=np.float64)
    else:
        x = np.arange(y.size, dtype=np.float64)
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


def discover_ic_power_records(input_root: str | Path, safety_yaml: str | Path = DEFAULT_SAFETY) -> list[IcPowerRecord]:
    root = resolve_repo_path(input_root)
    records: list[IcPowerRecord] = []
    for npz_path in sorted(root.rglob("safety_metrics.npz")):
        manifest_path = npz_path.parent / "run_manifest.yaml"
        if not manifest_path.exists():
            continue
        manifest = load_yaml(manifest_path)
        preprocessing = normalize_preprocessing_label(
            manifest.get("source_input_label", manifest.get("preprocessing_method", "unknown"))
        )
        power = float(manifest.get("internal_circuit_power_mw", math.nan))
        if not np.isfinite(power):
            power = infer_ic_power_from_name(npz_path.parent.name)
        if not np.isfinite(power):
            continue
        with np.load(npz_path, allow_pickle=True) as data:
            max_focal = finite_max(np.asarray(data["max_dT"])) if "max_dT" in data.files else math.nan
            max_mean = finite_max(np.asarray(data["mean_dT"])) if "mean_dT" in data.files else math.nan
        records.append(
            IcPowerRecord(
                npz_path=npz_path,
                manifest_path=manifest_path,
                run_id=str(manifest.get("run_id", npz_path.parent.name)),
                preprocessing=preprocessing,
                preprocessing_label=preprocessing,
                ic_power_mw=power,
                amplitude_uA=float(manifest.get("amplitude_uA", math.nan)),
                coords_yaml=str(manifest.get("coords_yaml", "")),
                max_focal_dT_C=max_focal,
                max_mean_dT_C=max_mean,
            )
        )
    return sorted(records, key=record_sort_key)


def record_sort_key(record: IcPowerRecord) -> tuple[float, float, str]:
    try:
        preprocessing_index = PREPROCESSING_ORDER.index(record.preprocessing)
    except ValueError:
        preprocessing_index = len(PREPROCESSING_ORDER)
    return (float(preprocessing_index), float(record.ic_power_mw), record.run_id)


def records_by_preprocessing(records: Iterable[IcPowerRecord]) -> dict[str, list[IcPowerRecord]]:
    grouped: dict[str, list[IcPowerRecord]] = {}
    for record in records:
        grouped.setdefault(record.preprocessing, []).append(record)
    return {key: sorted(value, key=lambda record: record.ic_power_mw) for key, value in grouped.items()}


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
                "max_focal_dT_C",
                "max_mean_dT_C",
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
                    "max_focal_dT_C": record.max_focal_dT_C,
                    "max_mean_dT_C": record.max_mean_dT_C,
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
    grouped = records_by_preprocessing(records)
    for preprocessing in sorted(grouped, key=lambda key: PREPROCESSING_ORDER.index(key) if key in PREPROCESSING_ORDER else 99):
        prep_records = grouped[preprocessing]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
        plotted = False
        for index, record in enumerate(prep_records):
            color = LINE_COLORS[index % len(LINE_COLORS)]
            with np.load(record.npz_path, allow_pickle=True) as data:
                focal = series_from_key(data, "max_dT")
                mean = series_from_key(data, "mean_dT")
            label = f"{record.ic_power_mw:g} mW"
            if focal is not None:
                axes[0].plot(focal[0], focal[1], color=color, linewidth=1.9, label=label)
                plotted = True
            if mean is not None:
                axes[1].plot(mean[0], mean[1], color=color, linewidth=1.9, label=label)
                plotted = True
        axes[0].set_title("Max focal temperature rise")
        axes[0].set_ylabel("max focal dT (C)")
        axes[1].set_title("Mean temperature rise")
        axes[1].set_ylabel("mean dT (C)")
        for ax in axes:
            ax.set_xlabel("time (min)")
            add_limit_line(ax, limits.get("temperature_increase_C", math.nan))
            style_axes(ax)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(title="IC power", frameon=False, fontsize=8, title_fontsize=8)
        fig.suptitle(f"Temperature evolution - {preprocessing}")
        if not plotted:
            for ax in axes:
                ax.text(0.5, 0.5, "No temperature series", ha="center", va="center", transform=ax.transAxes)
        path = out_dir / f"temperature_evolution_{sanitize_path_part(preprocessing)}.{image_format}"
        save_figure(fig, path, overwrite=overwrite)
        paths.append(path)
    return paths


def value_grid(records: list[IcPowerRecord], getter: str) -> tuple[list[str], list[float], np.ndarray]:
    preprocessors = [key for key in PREPROCESSING_ORDER if any(record.preprocessing == key for record in records)]
    preprocessors.extend(
        sorted({record.preprocessing for record in records if record.preprocessing not in preprocessors})
    )
    powers = sorted({float(record.ic_power_mw) for record in records})
    values = np.full((len(preprocessors), len(powers)), np.nan, dtype=np.float64)
    for record in records:
        row = preprocessors.index(record.preprocessing)
        col = powers.index(float(record.ic_power_mw))
        values[row, col] = float(getattr(record, getter))
    return preprocessors, powers, values


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
    preprocessors, powers, values = value_grid(records, getter)
    out_dir = resolve_repo_path(output_root)
    fig, ax = plt.subplots(figsize=(max(6.4, 1.15 * len(powers) + 2.5), max(3.8, 0.8 * len(preprocessors) + 2.0)))
    finite = values[np.isfinite(values)]
    vmax = float(np.max(finite)) if finite.size else 1.0
    image = ax.imshow(values, cmap="YlOrRd", vmin=0.0, vmax=vmax)
    ax.set_xticks(np.arange(len(powers)))
    ax.set_xticklabels([f"{power:g} mW" for power in powers])
    ax.set_yticks(np.arange(len(preprocessors)))
    ax.set_yticklabels(preprocessors)
    ax.set_xlabel("IC power")
    ax.set_ylabel("preprocessing")
    ax.set_title(title)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "n/a" if not np.isfinite(value) else f"{value:.3g}"
            ax.text(col, row, label, ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("dT (C)")
    path = out_dir / f"{filename}.{image_format}"
    save_figure(fig, path, overwrite=overwrite)
    return path


def write_ic_power_temperature_visuals(
    input_root: str | Path,
    output_root: str | Path | None = None,
    *,
    image_format: str = "png",
    overwrite: bool = True,
    safety_yaml: str | Path = DEFAULT_SAFETY,
) -> list[Path]:
    root = resolve_repo_path(input_root)
    out_dir = resolve_repo_path(output_root) if output_root is not None else root / "visuals"
    records = discover_ic_power_records(root, safety_yaml)
    if not records:
        raise RuntimeError(f"No IC-power safety_metrics.npz files found under: {root}")

    written = [write_temperature_summary_csv(records, out_dir)]
    written.extend(
        plot_temperature_evolution_by_preprocessing(
            records,
            out_dir,
            image_format=image_format,
            overwrite=overwrite,
            safety_yaml=safety_yaml,
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
    return written
