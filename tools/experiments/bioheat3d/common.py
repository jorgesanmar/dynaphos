from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos.safety.bioheat import Bioheat3D


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_yaml(path: str | Path) -> dict:
    with open(resolve_repo_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_electrode_xy_mm(path: str | Path) -> np.ndarray:
    coords = load_yaml(path)
    return np.column_stack(
        [
            np.asarray(coords["x"], dtype=np.float64),
            np.asarray(coords["y"], dtype=np.float64),
        ]
    )


def configure_bioheat(
    *,
    params_path: str | Path,
    coords_yaml: str | Path,
    device: str,
    constant_power_mw: float | None = None,
) -> tuple[Bioheat3D, dict, np.ndarray]:
    params = load_yaml(params_path)
    params.setdefault("run", {})["gpu"] = None if device == "cpu" else params.get("run", {}).get("gpu", None)
    params.setdefault("bioheat", {})
    if constant_power_mw is not None:
        params["bioheat"]["device_constant_power_mw"] = float(constant_power_mw)
        params["bioheat"]["internal_circuit_power_mw"] = float(constant_power_mw)

    elec_xy_mm = load_electrode_xy_mm(coords_yaml)
    bio = Bioheat3D(params=params, elec_xy_mm=elec_xy_mm, device=device)
    return bio, params, elec_xy_mm


def source_plane_mean(bio: Bioheat3D) -> float:
    source_plane = bio.source_plane_dT()
    mask = bio.ic_footprint_mask.to(device=source_plane.device, dtype=torch.bool)
    if torch.any(mask):
        return float(source_plane[mask].mean().detach().cpu().item())
    return float(source_plane.mean().detach().cpu().item())


def simulate_constant_power(
    bio: Bioheat3D,
    *,
    power_W: float,
    duration_s: float,
    dt_s: float,
) -> list[dict[str, float]]:
    if duration_s <= 0.0:
        raise ValueError("duration_s must be > 0.")
    if dt_s <= 0.0:
        raise ValueError("dt_s must be > 0.")

    n_steps = int(np.ceil(duration_s / dt_s))
    records: list[dict[str, float]] = []
    elapsed_s = 0.0
    for step in range(n_steps):
        step_dt = min(dt_s, duration_s - elapsed_s)
        if step_dt <= 0.0:
            break
        bio.update(float(power_W), float(step_dt))
        elapsed_s += float(step_dt)
        projection = bio.dT_projection()
        records.append(
            {
                "step": float(step + 1),
                "time_s": float(elapsed_s),
                "power_W": float(power_W),
                "max_dT_C": float(bio.dT.max().detach().cpu().item()),
                "mean_projection_dT_C": float(projection.mean().detach().cpu().item()),
                "source_plane_mean_dT_C": source_plane_mean(bio),
            }
        )
    return records


def write_csv(path: Path, rows: Iterable[dict[str, float]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_section_figure(*, path: Path, stem: str, bio: Bioheat3D) -> Path:
    volume = bio.dT.detach().cpu().numpy().astype(np.float32)
    source_z = int(bio.source_z_index)
    max_z, max_y, max_x = np.unravel_index(int(np.argmax(volume)), volume.shape)
    horizontal = volume[source_z]
    vertical = volume[:, max_y, :]
    source_mask = bio.ic_footprint_mask.detach().cpu().numpy().astype(bool)

    vmax = float(np.nanmax(volume)) if volume.size else 0.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1e-6

    xmin, xmax, ymin, ymax, zmin, zmax = bio.volume_extent_mm
    total_z_mm = float(zmax - zmin)
    source_center_z_mm = (source_z + 0.5) * bio.voxel_size_mm
    source_bottom_z_mm = source_z * bio.voxel_size_mm
    source_top_z_mm = (source_z + 1) * bio.voxel_size_mm
    y_section_mm = ymin + (max_y + 0.5) * bio.voxel_size_mm

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
    fig.suptitle(f"{stem}: final temperature rise sections", fontsize=13)

    im0 = axes[0].imshow(
        horizontal,
        origin="lower",
        extent=(xmin, xmax, ymin, ymax),
        cmap="inferno",
        vmin=0.0,
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
    x_hot_mm = xmin + (max_x + 0.5) * bio.voxel_size_mm
    label_box = {
        "boxstyle": "round,pad=0.25",
        "facecolor": "black",
        "edgecolor": "none",
        "alpha": 0.55,
    }
    axes[0].axhline(y_section_mm, color="cyan", linewidth=1.0, linestyle="--")
    axes[0].plot(x_hot_mm, y_section_mm, marker="x", color="cyan", markersize=6, mew=1.4)
    axes[0].text(
        0.02,
        0.96,
        "Brain source layer",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        color="white",
        fontsize=9,
        bbox=label_box,
    )
    axes[0].set_title(f"Horizontal x-y source layer z={source_center_z_mm:.1f} mm")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")

    im1 = axes[1].imshow(
        vertical,
        origin="lower",
        extent=(xmin, xmax, zmin, zmax),
        cmap="inferno",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    axes[1].axhspan(source_bottom_z_mm, source_top_z_mm, color="white", alpha=0.10, linewidth=0)
    axes[1].axhline(bio.brain_depth_mm, color="white", linewidth=1.0, linestyle="--", alpha=0.9)
    if bio.skull_thickness_mm > 0.0:
        axes[1].axhline(
            bio.brain_depth_mm + bio.skull_thickness_mm,
            color="white",
            linewidth=1.0,
            linestyle=":",
            alpha=0.9,
        )
    tissue_layers = [
        ("Brain", 0.0, float(bio.brain_depth_mm)),
        ("Skull", float(bio.brain_depth_mm), float(bio.brain_depth_mm + bio.skull_thickness_mm)),
        (
            "Scalp",
            float(bio.brain_depth_mm + bio.skull_thickness_mm),
            float(bio.brain_depth_mm + bio.skull_thickness_mm + bio.scalp_thickness_mm),
        ),
    ]
    for label, z0, z1 in tissue_layers:
        if z1 <= z0:
            continue
        axes[1].text(
            0.985,
            0.5 * (z0 + z1),
            label,
            transform=axes[1].get_yaxis_transform(),
            ha="right",
            va="center",
            color="white",
            fontsize=8,
            bbox=label_box,
        )
    axes[1].plot(x_hot_mm, (max_z + 0.5) * bio.voxel_size_mm, marker="x", color="cyan", markersize=6, mew=1.4)
    axes[1].set_title(f"Vertical x-z section through y={y_section_mm:.1f} mm")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z from brain bottom (mm)")

    cbar = fig.colorbar(im1, ax=axes, shrink=0.92)
    cbar.set_label("Temperature rise dT (C)")
    for ax in axes:
        ax.tick_params(labelsize=9)

    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_source_mean_figure(*, path: Path, stem: str, records: list[dict[str, float]]) -> Path | None:
    if not records:
        return None

    time_s = np.asarray([row["time_s"] for row in records], dtype=np.float64)
    source_mean = np.asarray([row["source_plane_mean_dT_C"] for row in records], dtype=np.float64)
    max_dT = np.asarray([row["max_dT_C"] for row in records], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    ax.plot(
        time_s,
        source_mean,
        color="#2563eb",
        linewidth=2.0,
        label="Mean dT in electrode footprint",
    )
    ax.plot(
        time_s,
        max_dT,
        color="#dc2626",
        linewidth=1.5,
        linestyle="--",
        alpha=0.85,
        label="Maximum dT",
    )
    ax.set_title(f"{stem}: electrode-grid mean temperature rise")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature rise dT (C)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)

    if source_mean.size:
        final_text = f"Final footprint mean: {source_mean[-1]:.4f} C"
        ax.annotate(
            final_text,
            xy=(time_s[-1], source_mean[-1]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="#1f2937",
        )

    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def write_outputs(
    *,
    out_dir: str | Path,
    stem: str,
    bio: Bioheat3D,
    records: list[dict[str, float]],
    metadata: dict[str, float | int | str],
) -> Path:
    out_path = resolve_repo_path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    time_s = np.asarray([row["time_s"] for row in records], dtype=np.float32)
    max_dT = np.asarray([row["max_dT_C"] for row in records], dtype=np.float32)
    source_mean = np.asarray([row["source_plane_mean_dT_C"] for row in records], dtype=np.float32)
    power_W = np.asarray([row["power_W"] for row in records], dtype=np.float32)

    np.savez(
        out_path / f"{stem}.npz",
        time_s=time_s,
        power_W=power_W,
        max_dT_C=max_dT,
        source_plane_mean_dT_C=source_mean,
        dT_final=bio.dT_projection().detach().cpu().numpy().astype(np.float32),
        dT_source_plane=bio.source_plane_dT().detach().cpu().numpy().astype(np.float32),
        dT_volume_final=bio.dT.detach().cpu().numpy().astype(np.float32),
        extent_mm=np.asarray(bio.extent_mm, dtype=np.float32),
        voxel_size_mm=np.asarray(bio.voxel_size_mm, dtype=np.float32),
        voxel_spacing_mm=np.asarray(bio.voxel_spacing_mm, dtype=np.float32),
        z_extent_mm=np.asarray(bio.z_extent_mm, dtype=np.float32),
        volume_extent_mm=np.asarray(bio.volume_extent_mm, dtype=np.float32),
        dT_volume_axis_order=np.asarray("z_y_x"),
        source_voxel_count=np.asarray(bio.source_voxel_count, dtype=np.float32),
        source_area_m2=np.asarray(bio.source_area_m2, dtype=np.float32),
        source_volume_m3=np.asarray(bio.source_volume_m3, dtype=np.float32),
        ic_power_density_W_m3=np.asarray(bio.ic_power_density_W_m3, dtype=np.float32),
        **{key: np.asarray(value) for key, value in metadata.items()},
    )
    write_csv(out_path / f"{stem}.csv", records)
    section_path = write_section_figure(path=out_path / f"{stem}_sections.png", stem=stem, bio=bio)
    source_mean_plot_path = write_source_mean_figure(
        path=out_path / f"{stem}_source_mean_dT_over_time.png",
        stem=stem,
        records=records,
    )

    final = records[-1] if records else {
        "time_s": 0.0,
        "power_W": 0.0,
        "max_dT_C": 0.0,
        "source_plane_mean_dT_C": 0.0,
    }
    summary_lines = [
        f"stem={stem}",
        f"duration_s={final['time_s']:.6f}",
        f"power_W={final['power_W']:.9f}",
        f"max_dT_C={final['max_dT_C']:.9f}",
        f"source_plane_mean_dT_C={final['source_plane_mean_dT_C']:.9f}",
        f"source_voxel_count={bio.source_voxel_count:.0f}",
        f"voxel_spacing_mm={bio.voxel_spacing_mm}",
        f"volume_extent_mm={bio.volume_extent_mm}",
        f"source_area_mm2={bio.source_area_m2 * 1e6:.6f}",
        f"source_volume_mm3={bio.source_volume_m3 * 1e9:.6f}",
        f"ic_power_density_W_m3={bio.ic_power_density_W_m3:.9f}",
        f"sections_png={section_path}",
    ]
    if source_mean_plot_path is not None:
        summary_lines.append(f"source_mean_dT_plot_png={source_mean_plot_path}")
    summary_lines.extend(f"{key}={value}" for key, value in metadata.items())
    (out_path / f"{stem}_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return out_path
