"""
Generate comparison plots for the declarative safety experiment matrix.

The input is the directory written by tools/safety/run_experiment_matrix.py:
one subdirectory per experiment block, with one run_manifest.yaml and
safety_metrics.npz per case.
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
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.safety.common import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SAFETY,
    DEFAULT_VISUALS_ROOT,
    load_safety_limits,
    resolve_repo_path,
    sanitize_path_part,
)


DEFAULT_INPUT_ROOT = Path(DEFAULT_OUTPUT_ROOT)
DEFAULT_OUTPUT_ROOT = Path(DEFAULT_VISUALS_ROOT) / "experiment_matrix_summary"

BLOCK_ORDER = ("amplitude", "electrode_density", "preprocessing", "internal_circuit", "rastering")
SUMMARY_METRICS = (
    ("safety_margin", "Worst safety-limit ratio", "ratio"),
    ("peak_current_uA", "Peak current", "uA"),
    ("peak_charge_per_phase_nC", "Peak charge per phase", "nC"),
    ("peak_shannon_k", "Peak Shannon k", ""),
    ("max_active_pct", "Max active electrodes", "%"),
    ("max_total_charge_rate_uC_s", "Max total charge rate", "uC/s"),
    ("max_dT_C", "Max temperature increase", "C"),
)

COLORS = {
    "amplitude": "#1F4E79",
    "electrode_density": "#2E8B57",
    "preprocessing": "#8B1E3F",
    "internal_circuit": "#C05621",
    "rastering": "#5B5EA6",
}
FALLBACK_COLOR = "#6B7280"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparison plots for run_experiment_matrix.py outputs."
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


def metric_or_peak(data: np.lib.npyio.NpzFile, scalar_key: str, series_key: str, default: float = 0.0) -> float:
    value = scalar(data, scalar_key, math.nan)
    if np.isfinite(value):
        return value
    return max_value(data, series_key, default)


def safe_ratio(value: float, limit: float) -> float:
    if not np.isfinite(value) or not np.isfinite(limit) or limit <= 0.0:
        return math.nan
    return value / limit


def block_label_and_sort(block: str, manifest: dict) -> tuple[str, float]:
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
        return str(manifest.get("source_input_label", manifest.get("preprocessing_method", ""))), 0.0
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
        peak_charge = metric_or_peak(data, "peak_charge_per_phase_nC_exact", "charge_per_phase_mean_nC")
        peak_shannon = metric_or_peak(data, "peak_shannon_k_exact", "shannon_k_mean")
        peak_current = metric_or_peak(data, "peak_current_amplitude_uA_exact", "current_amplitude_per_electrode_uA")
        max_dT = metric_or_peak(data, "peak_max_dT_C", "max_dT")
        window_charge_electrode = max_value(data, "window_charge_per_electrode_nC")
        window_charge_total = max_value(data, "window_charge_total_nC")
        active_count = max_value(data, "active_count")
        active_pct = 100.0 * active_count / float(electrode_count)
        total_charge_rate = max_value(data, "charge_per_second_total_nC_s") / 1e3
        final_charge_total = max_value(data, "protocol_charge_total_nC") / 1e3
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
        "max_active_pct": active_pct,
        "max_total_charge_rate_uC_s": total_charge_rate,
        "final_total_charge_uC": final_charge_total,
        "max_window_charge_per_electrode_nC": window_charge_electrode,
        "max_window_charge_total_nC": window_charge_total,
        "max_dT_C": max_dT,
        "max_cem43_min": cem43,
    }
    return metrics, ratios


def discover_records(input_root: Path, safety_yaml: Path) -> list[MatrixRecord]:
    limits = load_safety_limits(safety_yaml)
    records: list[MatrixRecord] = []
    for npz_path in sorted(input_root.rglob("safety_metrics.npz")):
        manifest_path = npz_path.parent / "run_manifest.yaml"
        if not manifest_path.exists():
            print(f"Skipping {npz_path}: missing run_manifest.yaml")
            continue
        manifest = load_yaml(manifest_path)
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


def records_by_block(records: Iterable[MatrixRecord], block: str) -> list[MatrixRecord]:
    return sorted((record for record in records if record.block == block), key=lambda r: (r.sort_value, r.run_id))


def plot_block_metric(records: list[MatrixRecord], block: str, metric_key: str, title: str, unit: str, output_root: Path, image_format: str, *, overwrite: bool) -> None:
    selected = records_by_block(records, block)
    if not selected:
        return
    values = [record.metrics.get(metric_key, math.nan) for record in selected]
    labels = [record.label for record in selected]
    color = COLORS.get(block, FALLBACK_COLOR)

    width = max(6.5, 0.75 * len(selected) + 2.5)
    fig, ax = plt.subplots(figsize=(width, 4.2))
    ax.bar(range(len(selected)), values, color=color, alpha=0.9)
    ax.set_xticks(range(len(selected)))
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ylabel = title if not unit else f"{title} ({unit})"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} - {block.replace('_', ' ')}")
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if metric_key == "safety_margin":
        ax.axhline(1.0, color="#8B1E3F", linestyle="--", linewidth=1.2, label="limit")
        ax.legend(frameon=False)

    path = output_root / block / f"{sanitize_path_part(metric_key)}.{image_format}"
    save_figure(fig, path, overwrite=overwrite)


def plot_all_block_summaries(records: list[MatrixRecord], output_root: Path, image_format: str, *, overwrite: bool) -> None:
    for block in sorted({record.block for record in records}, key=block_index):
        for metric_key, title, unit in SUMMARY_METRICS:
            plot_block_metric(records, block, metric_key, title, unit, output_root, image_format, overwrite=overwrite)


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


def plot_ratio_breakdown(records: list[MatrixRecord], output_root: Path, image_format: str, *, overwrite: bool) -> None:
    ratio_keys = ("charge_per_phase", "window_charge_per_electrode", "window_charge_total", "active_percentage", "temperature")
    for block in sorted({record.block for record in records}, key=block_index):
        selected = records_by_block(records, block)
        if not selected:
            continue
        x = np.arange(len(selected), dtype=np.float64)
        width = 0.8 / len(ratio_keys)
        fig, ax = plt.subplots(figsize=(max(7.5, 0.85 * len(selected) + 2.5), 4.8))
        for idx, key in enumerate(ratio_keys):
            offset = (idx - (len(ratio_keys) - 1) / 2.0) * width
            values = [record.ratios.get(key, math.nan) for record in selected]
            ax.bar(x + offset, values, width=width, label=key.replace("_", " "))
        ax.axhline(1.0, color="#8B1E3F", linestyle="--", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([record.label for record in selected])
        ax.set_ylabel("Ratio to configured limit")
        ax.set_title(f"Safety-limit ratio breakdown - {block.replace('_', ' ')}")
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        save_figure(fig, output_root / block / f"limit_ratio_breakdown.{image_format}", overwrite=overwrite)


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


def main() -> None:
    args = parse_args()
    input_root = resolve_repo_path(args.input_root)
    output_root = resolve_repo_path(args.output_root)
    safety_yaml = resolve_repo_path(args.safety_yaml)
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    records = discover_records(input_root, safety_yaml)
    if not records:
        raise RuntimeError(f"No matrix safety_metrics.npz files found under: {input_root}")

    print(f"Discovered {len(records)} matrix runs under: {input_root}")
    write_summary_csv(records, output_root)
    plot_overall_safety(records, output_root, args.format, overwrite=bool(args.overwrite))
    plot_ratio_breakdown(records, output_root, args.format, overwrite=bool(args.overwrite))
    plot_all_block_summaries(records, output_root, args.format, overwrite=bool(args.overwrite))


if __name__ == "__main__":
    main()
