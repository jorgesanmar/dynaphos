from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos.safety.impedance import Impedance, compute_device_power
from dynaphos.safety.protocols import (
    STANDARD_AMPLITUDE_UA,
    STANDARD_FREQUENCY_HZ,
    STANDARD_PULSE_WIDTH_US,
)
from tools.experiments.bioheat3d.common import (
    configure_bioheat,
    resolve_repo_path,
    simulate_constant_power,
    write_csv,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Bioheat3D sweeps over fixed device power and over the number "
            "of simultaneously active electrodes."
        )
    )
    parser.add_argument("--params", default="config/params.yaml")
    parser.add_argument("--coords-yaml", default="config/coords_800um.yaml")
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument(
        "--power-values-mw",
        default=None,
        help=(
            "Comma or space separated fixed powers in mW for the power sweep. "
            "If omitted, --power-min-mw/--power-max-mw/--power-steps are used."
        ),
    )
    parser.add_argument("--power-min-mw", type=float, default=0.0)
    parser.add_argument("--power-max-mw", type=float, default=100.0)
    parser.add_argument("--power-steps", type=int, default=5)
    parser.add_argument(
        "--constant-power-mw",
        type=float,
        default=None,
        help=(
            "Fixed background IC power for the active-electrode sweep. "
            "Defaults to bioheat.device_constant_power_mw from params."
        ),
    )
    parser.add_argument(
        "--active-counts",
        default=None,
        help=(
            "Comma or space separated active-electrode counts for the second "
            "sweep. Use 'all' for the full electrode grid. Defaults to powers "
            "of two plus all electrodes."
        ),
    )
    parser.add_argument("--amplitude-uA", type=float, default=STANDARD_AMPLITUDE_UA)
    parser.add_argument("--frequency-hz", type=float, default=STANDARD_FREQUENCY_HZ)
    parser.add_argument("--pulse-width-us", type=float, default=STANDARD_PULSE_WIDTH_US)
    parser.add_argument(
        "--driver-efficiency",
        type=float,
        default=None,
        help="Override bioheat.driver_efficiency for the active-electrode sweep.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--output-dir",
        default="results/safety/bioheat3d_tests/power_and_active_sweeps",
    )
    parser.add_argument("--skip-power-sweep", action="store_true")
    parser.add_argument("--skip-active-electrode-sweep", action="store_true")
    return parser.parse_args()


def parse_float_values(value: str) -> list[float]:
    tokens = str(value).replace(",", " ").split()
    if not tokens:
        raise ValueError("At least one numeric value is required.")
    return [float(token) for token in tokens]


def unique_floats(values: list[float]) -> list[float]:
    unique: list[float] = []
    for value in values:
        if not np.isfinite(value):
            raise ValueError(f"Non-finite power value: {value}")
        if all(not np.isclose(value, seen, rtol=0.0, atol=1e-12) for seen in unique):
            unique.append(float(value))
    return unique


def power_values_from_args(args: argparse.Namespace) -> list[float]:
    if args.power_values_mw is not None:
        values = parse_float_values(args.power_values_mw)
    else:
        if int(args.power_steps) <= 0:
            raise ValueError("--power-steps must be > 0.")
        values = np.linspace(
            float(args.power_min_mw),
            float(args.power_max_mw),
            int(args.power_steps),
            dtype=np.float64,
        ).tolist()
    values = unique_floats(values)
    if any(value < 0.0 for value in values):
        raise ValueError("Power values must be >= 0 mW.")
    return values


def default_active_counts(total: int) -> list[int]:
    counts = [0]
    count = 1
    while count < total:
        counts.append(count)
        count *= 2
    counts.append(total)
    return sorted(set(counts))


def active_counts_from_args(value: str | None, total: int) -> list[int]:
    if value is None:
        return default_active_counts(total)

    counts: list[int] = []
    for token in str(value).replace(",", " ").split():
        if token.strip().lower() == "all":
            count = total
        else:
            count = int(token)
        if count < 0 or count > total:
            raise ValueError(f"Active-electrode counts must be between 0 and {total}, got {count}.")
        counts.append(count)
    if not counts:
        raise ValueError("At least one active-electrode count is required.")
    return sorted(set(counts))


def value_token(value: float) -> str:
    text = f"{float(value):g}".replace("-", "neg").replace(".", "p")
    return text or "0"


def write_final_temperature_sections(*, path: Path, stem: str, bio) -> Path:
    temperature = bio.temperature_C.detach().cpu().numpy().astype(np.float32)
    rise = bio.dT.detach().cpu().numpy().astype(np.float32)
    source_z = int(bio.source_z_index)
    max_z, max_y, max_x = np.unravel_index(int(np.argmax(rise)), rise.shape)
    horizontal = temperature[source_z]
    vertical = temperature[:, max_y, :]
    source_mask = bio.ic_footprint_mask.detach().cpu().numpy().astype(bool)

    vmin = float(getattr(bio, "baseline_T", np.nanmin(temperature)))
    vmax = float(np.nanmax(temperature)) if temperature.size else vmin
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(temperature))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    xmin, xmax, ymin, ymax, zmin, zmax = bio.volume_extent_mm
    source_center_z_mm = (source_z + 0.5) * bio.voxel_size_mm
    y_section_mm = ymin + (max_y + 0.5) * bio.voxel_size_mm
    x_hot_mm = xmin + (max_x + 0.5) * bio.voxel_size_mm
    z_hot_mm = (max_z + 0.5) * bio.voxel_size_mm

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    fig.suptitle(f"{stem}: final temperature sections", fontsize=13)

    im0 = axes[0].imshow(
        horizontal,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    if np.any(source_mask) and not np.all(source_mask):
        axes[0].contour(
            source_mask.astype(float),
            levels=[0.5],
            origin="lower",
            extent=(xmin, xmax, ymin, ymax),
            colors="white",
            linewidths=1.0,
        )
    axes[0].axhline(y_section_mm, color="cyan", linewidth=1.0, linestyle="--")
    axes[0].plot(x_hot_mm, y_section_mm, marker="x", color="cyan", markersize=6, mew=1.4)
    axes[0].set_title(f"Horizontal x-y source layer z={source_center_z_mm:.1f} mm")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")

    im1 = axes[1].imshow(
        vertical,
        origin="lower",
        extent=(xmin, xmax, zmin, zmax),
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].axhline(bio.brain_depth_mm, color="white", linewidth=1.0, linestyle="--", alpha=0.9)
    if bio.skull_thickness_mm > 0.0:
        axes[1].axhline(
            bio.brain_depth_mm + bio.skull_thickness_mm,
            color="white",
            linewidth=1.0,
            linestyle=":",
            alpha=0.9,
        )
    axes[1].plot(x_hot_mm, z_hot_mm, marker="x", color="cyan", markersize=6, mew=1.4)
    axes[1].set_title(f"Vertical x-z section through y={y_section_mm:.1f} mm")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z from brain bottom (mm)")

    cbar = fig.colorbar(im1, ax=axes, shrink=0.92)
    cbar.set_label("Temperature (C)")
    for ax in axes:
        ax.tick_params(labelsize=9)

    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def summarize_final_case(
    *,
    bio,
    records: list[dict[str, float]],
    case_label: str,
    sections_png: Path,
    final_temperature_sections_png: Path,
    extra: dict[str, float | int | str],
) -> dict[str, float | int | str]:
    final = records[-1]
    max_dT = float(final["max_dT_C"])
    source_mean_dT = float(final["source_plane_mean_dT_C"])
    projection_mean_dT = float(final["mean_projection_dT_C"])
    volume_mean_dT = float(bio.dT.mean().detach().cpu().item())
    baseline = float(bio.baseline_T)
    row: dict[str, float | int | str] = {
        "case": case_label,
        "duration_s": float(final["time_s"]),
        "dt_s": float(extra["dt_s"]),
        "max_dT_C": max_dT,
        "source_plane_mean_dT_C": source_mean_dT,
        "projection_mean_dT_C": projection_mean_dT,
        "volume_mean_dT_C": volume_mean_dT,
        "max_temperature_C": baseline + max_dT,
        "source_plane_mean_temperature_C": baseline + source_mean_dT,
        "projection_mean_temperature_C": baseline + projection_mean_dT,
        "volume_mean_temperature_C": baseline + volume_mean_dT,
        "source_area_mm2": float(bio.source_area_m2 * 1e6),
        "source_volume_mm3": float(bio.source_volume_m3 * 1e9),
        "sections_png": str(sections_png),
        "final_temperature_sections_png": str(final_temperature_sections_png),
    }
    row.update(extra)
    return row


def append_summary_path(summary_path: Path, key: str, value: Path) -> None:
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def plot_temperature_vs_power(path: Path, rows: list[dict[str, float | int | str]]) -> Path | None:
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: float(row["power_mw"]))
    power_mw = np.asarray([row["power_mw"] for row in rows], dtype=np.float64)
    max_dT = np.asarray([row["max_dT_C"] for row in rows], dtype=np.float64)
    source_mean = np.asarray([row["source_plane_mean_dT_C"] for row in rows], dtype=np.float64)
    volume_mean = np.asarray([row["volume_mean_dT_C"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    ax.plot(power_mw, max_dT, marker="o", linewidth=2.0, color="#dc2626", label="Maximum dT")
    ax.plot(
        power_mw,
        source_mean,
        marker="s",
        linewidth=2.0,
        color="#2563eb",
        label="Mean dT in electrode footprint",
    )
    ax.plot(
        power_mw,
        volume_mean,
        marker="^",
        linewidth=1.6,
        linestyle=":",
        color="#4b5563",
        label="Whole-volume mean dT",
    )
    ax.set_title("Final temperature rise vs fixed device power")
    ax.set_xlabel("Fixed device power (mW)")
    ax.set_ylabel("Final temperature rise dT (C)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_temperature_vs_active_count(path: Path, rows: list[dict[str, float | int | str]]) -> Path | None:
    if not rows:
        return None
    rows = sorted(rows, key=lambda row: int(row["active_electrode_count"]))
    active_count = np.asarray([row["active_electrode_count"] for row in rows], dtype=np.float64)
    max_dT = np.asarray([row["max_dT_C"] for row in rows], dtype=np.float64)
    source_mean = np.asarray([row["source_plane_mean_dT_C"] for row in rows], dtype=np.float64)
    volume_mean = np.asarray([row["volume_mean_dT_C"] for row in rows], dtype=np.float64)
    total_power_mw = np.asarray([row["total_power_mw"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 4.9), constrained_layout=True)
    line1 = ax.plot(active_count, max_dT, marker="o", linewidth=2.0, color="#dc2626", label="Maximum dT")
    line2 = ax.plot(
        active_count,
        source_mean,
        marker="s",
        linewidth=2.0,
        color="#2563eb",
        label="Mean dT in electrode footprint",
    )
    line3 = ax.plot(
        active_count,
        volume_mean,
        marker="^",
        linewidth=1.6,
        linestyle=":",
        color="#4b5563",
        label="Whole-volume mean dT",
    )
    ax.set_title("Final temperature rise vs active-electrode count")
    ax.set_xlabel("Simultaneously active electrodes")
    ax.set_ylabel("Final temperature rise dT (C)")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    line4 = ax2.plot(
        active_count,
        total_power_mw,
        marker="d",
        linewidth=1.8,
        linestyle="--",
        color="#059669",
        label="Total IC power",
    )
    ax2.set_ylabel("Total IC power (mW)")

    lines = line1 + line2 + line3 + line4
    ax.legend(lines, [line.get_label() for line in lines], loc="best", frameon=False)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def run_power_sweep(args: argparse.Namespace, out_dir: Path) -> list[dict[str, float | int | str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    for power_mw in power_values_from_args(args):
        stem = f"power_{value_token(power_mw)}mW"
        print(f"[power] {power_mw:g} mW")
        bio, _params, elec_xy_mm = configure_bioheat(
            params_path=args.params,
            coords_yaml=args.coords_yaml,
            device=args.device,
            constant_power_mw=power_mw,
        )
        records = simulate_constant_power(
            bio,
            power_W=float(power_mw) * 1e-3,
            duration_s=float(args.duration_s),
            dt_s=float(args.dt_s),
        )
        write_outputs(
            out_dir=out_dir,
            stem=stem,
            bio=bio,
            records=records,
            metadata={
                "mode": "fixed_power_sweep_no_stimulation",
                "electrode_count": int(elec_xy_mm.shape[0]),
                "fixed_power_mw": float(power_mw),
                "duration_s": float(args.duration_s),
                "dt_s": float(args.dt_s),
            },
        )
        temperature_sections = write_final_temperature_sections(
            path=out_dir / f"{stem}_final_temperature_sections.png",
            stem=stem,
            bio=bio,
        )
        append_summary_path(out_dir / f"{stem}_summary.txt", "final_temperature_sections_png", temperature_sections)
        rows.append(
            summarize_final_case(
                bio=bio,
                records=records,
                case_label=stem,
                sections_png=out_dir / f"{stem}_sections.png",
                final_temperature_sections_png=temperature_sections,
                extra={
                    "sweep": "power",
                    "power_mw": float(power_mw),
                    "electrode_count": int(elec_xy_mm.shape[0]),
                    "dt_s": float(args.dt_s),
                },
            )
        )
    write_csv(out_dir / "power_sweep_summary.csv", rows)
    plot_path = plot_temperature_vs_power(out_dir / "power_sweep_final_temperature_rise_vs_power.png", rows)
    if plot_path is not None:
        print(f"[power] summary plot: {plot_path}")
    return rows


def compute_standard_total_power(
    *,
    bio,
    impedance_ohm: torch.Tensor,
    active_count: int,
    n_electrodes: int,
    amplitude_uA: float,
    pulse_width_us: float,
    frequency_hz: float,
    relative_stim_duration: float,
) -> tuple[float, float]:
    amplitude = torch.zeros(n_electrodes, dtype=torch.float32, device=bio.device)
    if active_count > 0:
        amplitude[:active_count] = float(amplitude_uA) * 1e-6

    pulse_width = torch.full(
        (n_electrodes,),
        float(pulse_width_us) * 1e-6,
        dtype=torch.float32,
        device=bio.device,
    )
    frequency = torch.full(
        (n_electrodes,),
        float(frequency_hz),
        dtype=torch.float32,
        device=bio.device,
    )
    stim_ic_power_W, total_power_W = compute_device_power(
        amplitude=amplitude,
        impedance_ohm=impedance_ohm.to(device=bio.device, dtype=torch.float32),
        pulse_width_s=pulse_width,
        frequency_hz=frequency,
        relative_stim_duration=float(relative_stim_duration),
        constant_power_W=float(bio.device_constant_power_W),
        driver_efficiency=float(bio.driver_efficiency),
    )
    return (
        float(stim_ic_power_W.detach().cpu().item()),
        float(total_power_W.detach().cpu().item()),
    )


def run_active_electrode_sweep(args: argparse.Namespace, out_dir: Path) -> list[dict[str, float | int | str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_bio, base_params, elec_xy_mm = configure_bioheat(
        params_path=args.params,
        coords_yaml=args.coords_yaml,
        device=args.device,
        constant_power_mw=args.constant_power_mw,
    )
    if args.driver_efficiency is not None:
        base_bio.driver_efficiency = float(args.driver_efficiency)

    n_electrodes = int(elec_xy_mm.shape[0])
    counts = active_counts_from_args(args.active_counts, n_electrodes)
    base_params.setdefault("default_stim", {})["freq_default"] = float(args.frequency_hz)
    base_params.setdefault("run", {})["gpu"] = None if args.device == "cpu" else base_params.get("run", {}).get("gpu", None)
    impedance = Impedance(base_params, shape=(n_electrodes,), verbose=False)
    impedance_ohm = impedance.state.detach().clone()
    relative_stim_duration = float(base_params.get("default_stim", {}).get("relative_stim_duration", 1.0))
    constant_power_mw = float(base_bio.device_constant_power_W * 1e3)
    driver_efficiency = float(base_bio.driver_efficiency)
    del base_bio

    rows: list[dict[str, float | int | str]] = []
    for active_count in counts:
        stem = f"active_{active_count:04d}_of_{n_electrodes}"
        print(f"[active] {active_count}/{n_electrodes} electrodes")
        bio, _params, _elec_xy_mm = configure_bioheat(
            params_path=args.params,
            coords_yaml=args.coords_yaml,
            device=args.device,
            constant_power_mw=constant_power_mw,
        )
        bio.driver_efficiency = driver_efficiency
        stim_ic_power_W, total_power_W = compute_standard_total_power(
            bio=bio,
            impedance_ohm=impedance_ohm,
            active_count=int(active_count),
            n_electrodes=n_electrodes,
            amplitude_uA=float(args.amplitude_uA),
            pulse_width_us=float(args.pulse_width_us),
            frequency_hz=float(args.frequency_hz),
            relative_stim_duration=relative_stim_duration,
        )
        records = simulate_constant_power(
            bio,
            power_W=total_power_W,
            duration_s=float(args.duration_s),
            dt_s=float(args.dt_s),
        )
        write_outputs(
            out_dir=out_dir,
            stem=stem,
            bio=bio,
            records=records,
            metadata={
                "mode": "active_electrode_count_sweep",
                "electrode_count": n_electrodes,
                "active_electrode_count": int(active_count),
                "amplitude_uA": float(args.amplitude_uA),
                "pulse_width_us": float(args.pulse_width_us),
                "frequency_hz": float(args.frequency_hz),
                "relative_stim_duration": relative_stim_duration,
                "constant_power_mw": constant_power_mw,
                "driver_efficiency": driver_efficiency,
                "stim_ic_power_mw": stim_ic_power_W * 1e3,
                "total_power_mw": total_power_W * 1e3,
                "duration_s": float(args.duration_s),
                "dt_s": float(args.dt_s),
            },
        )
        temperature_sections = write_final_temperature_sections(
            path=out_dir / f"{stem}_final_temperature_sections.png",
            stem=stem,
            bio=bio,
        )
        append_summary_path(out_dir / f"{stem}_summary.txt", "final_temperature_sections_png", temperature_sections)
        rows.append(
            summarize_final_case(
                bio=bio,
                records=records,
                case_label=stem,
                sections_png=out_dir / f"{stem}_sections.png",
                final_temperature_sections_png=temperature_sections,
                extra={
                    "sweep": "active_electrode_count",
                    "electrode_count": n_electrodes,
                    "active_electrode_count": int(active_count),
                    "amplitude_uA": float(args.amplitude_uA),
                    "pulse_width_us": float(args.pulse_width_us),
                    "frequency_hz": float(args.frequency_hz),
                    "relative_stim_duration": relative_stim_duration,
                    "constant_power_mw": constant_power_mw,
                    "driver_efficiency": driver_efficiency,
                    "stim_ic_power_mw": stim_ic_power_W * 1e3,
                    "total_power_mw": total_power_W * 1e3,
                    "dt_s": float(args.dt_s),
                },
            )
        )
    write_csv(out_dir / "active_electrode_sweep_summary.csv", rows)
    plot_path = plot_temperature_vs_active_count(
        out_dir / "active_electrode_sweep_final_temperature_rise_vs_active_count.png",
        rows,
    )
    if plot_path is not None:
        print(f"[active] summary plot: {plot_path}")
    return rows


def main() -> None:
    args = parse_args()
    if args.skip_power_sweep and args.skip_active_electrode_sweep:
        raise ValueError("Both sweeps are disabled.")

    out_root = resolve_repo_path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_power_sweep:
        run_power_sweep(args, out_root / "power_sweep")
    if not args.skip_active_electrode_sweep:
        run_active_electrode_sweep(args, out_root / "active_electrode_sweep")
    print(f"Saved Bioheat3D sweep outputs to: {out_root}")


if __name__ == "__main__":
    main()
