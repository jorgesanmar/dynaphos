"""
Analyze charge and thermal safety for already preprocessed videos.

This script sweeps across preprocessing variants, raster modes, and internal
circuit heat modes, then runs one shared phosphene simulation per raster mode
while branching the bioheat model for each IC inclusion setting.

High-level flow:
1. Resolve one or more preprocessed video inputs.
2. For each requested run, read the video FPS and derive raster timing.
3. Simulate stimulation, track safety metrics, and update bioheat.
4. Save raw metrics for later plotting, plus phosphene preview videos.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Allow the script to be executed directly from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos import cortex_models, utils
from dynaphos.simulator import Bioheat2D, GaussianSimulator
from dynaphos.utils import Map
from tools.phosphenes.visualize_phosphene_representation import (
    build_comparison_frame,
    open_video_writer,
    render_phosphene_frame_from_state,
)


ACADEMIC_COLORS = {
    "band": "#9FB3C8",
    "mean": "#1F4E79",
    "min": "#6B7280",
    "max": "#8B1E3F",
}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg"}
VALID_PREPROCESS_METHODS = ("groundtruth", "canny", "dog")
VALID_RASTER_MODES = ("none", "checkerboard", "random", "pseudo_random", "pseudorandom", "pseudo-random")
RASTER_MODE_ALIASES = {
    "pseudorandom": "random",
    "pseudo-random": "random",
    "pseudo_random": "random",
}
VALID_IC_HEAT_MODES = ("with", "without")
VALID_PHOSPHENE_MODES = ("safety_centers", "visual")
HEATMAP_SNAPSHOT_COUNT = 20
HEATMAP_UPDATE_INTERVAL_FRAMES = 10
HEATMAP_SNAPSHOT_FRACTIONS = np.linspace(
    0.05, 1.0, num=HEATMAP_SNAPSHOT_COUNT, dtype=np.float64
)


# Plot styling helpers used by all saved figures.
def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=10, width=0.8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


def smooth_series(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if window <= 1 or y.size == 0:
        return y
    window = min(window, y.size)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def disable_temporal_dynamics(params: dict) -> dict:
    """Apply run-local simulator overrides for safety analysis only.

    We disable habituation by preventing trace buildup and freeze the
    activation-threshold distribution so thresholds remain fixed. Decay terms
    are intentionally left unchanged: setting them to 1.0 would make the
    simulator compute zero integration steps and trigger division-by-zero
    errors.
    """
    params.setdefault("temporal_dynamics", {})
    params["temporal_dynamics"]["trace_increase_rate"] = 0.0

    params.setdefault("thresholding", {})
    params["thresholding"]["activation_threshold_sd"] = 0.0

    params.setdefault("safety", {})
    params["safety"]["temporal_dynamics_disabled"] = True
    return params


def apply_stim_scale_override(params: dict, stim_scale: float | None) -> tuple[float, float]:
    """Apply the same stimulus-scale override semantics used by the visualizer CLI."""
    sampling = params.setdefault("sampling", {})
    base_stimulus_scale = float(sampling.get("stimulus_scale", 1.0))

    if stim_scale is None:
        return base_stimulus_scale, base_stimulus_scale

    stim_scale = float(stim_scale)
    if not 0.0 <= stim_scale <= 1.0:
        raise ValueError("--stim-scale must be between 0.0 and 1.0 inclusive.")

    effective_stimulus_scale = base_stimulus_scale * stim_scale
    sampling["stimulus_scale"] = effective_stimulus_scale
    return base_stimulus_scale, effective_stimulus_scale


def normalize_raster_label(mode: str) -> str:
    normalized = str(mode).strip().lower().replace(" ", "_").replace("-", "_")
    if normalized == "pseudorandom":
        return "pseudo_random"
    return normalized


def normalize_raster_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace(" ", "_")
    return RASTER_MODE_ALIASES.get(normalized, normalized)


def resolve_appearance_threshold_a(params: dict, appearance_threshold_uA: float | None) -> float:
    if appearance_threshold_uA is None:
        return float(params.get("thresholding", {}).get("rheobase", 0.0))
    threshold_uA = float(appearance_threshold_uA)
    if threshold_uA < 0.0:
        raise ValueError(f"appearance_threshold_uA must be >= 0, got {threshold_uA}.")
    return threshold_uA * 1e-6


def apply_appearance_threshold(stim_raw: torch.Tensor, threshold_a: float) -> torch.Tensor:
    threshold = torch.as_tensor(float(threshold_a), dtype=stim_raw.dtype, device=stim_raw.device)
    return torch.where(stim_raw >= threshold, stim_raw, torch.zeros_like(stim_raw))


def sanitize_path_part(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)
    return safe.strip("._") or "item"


def cem43_weight_from_temperature(temp_c: torch.Tensor) -> torch.Tensor:
    """Return the CEM43 weighting term for an absolute temperature map in Celsius."""
    temp_c = temp_c.to(dtype=torch.float32)
    active = temp_c > 39.0
    base = torch.where(
        temp_c < 43.0,
        torch.full_like(temp_c, 0.25),
        torch.full_like(temp_c, 0.5),
    )
    return torch.where(active, torch.pow(base, 43.0 - temp_c), torch.zeros_like(temp_c))


def percentile_from_tensor_list(values: list[torch.Tensor], percentile: float) -> float:
    flattened = [value.reshape(-1) for value in values if value is not None and value.numel() > 0]
    if not flattened:
        return float("nan")

    merged = torch.cat(flattened)
    k = int(np.ceil((float(percentile) / 100.0) * merged.numel()))
    k = max(1, min(k, int(merged.numel())))
    return float(torch.kthvalue(merged, k).values.item())


def percentile_tensor_from_tensor_list(values: list[torch.Tensor], percentile: float, device: torch.device) -> torch.Tensor:
    flattened = [value.reshape(-1) for value in values if value is not None and value.numel() > 0]
    if not flattened:
        return torch.tensor(float("nan"), dtype=torch.float32, device=device)

    merged = torch.cat(flattened)
    k = int(np.ceil((float(percentile) / 100.0) * merged.numel()))
    k = max(1, min(k, int(merged.numel())))
    return torch.kthvalue(merged, k).values


# Input discovery helpers.
def resolve_video_inputs(video_arg: str) -> list[Path]:
    video_path = Path(video_arg)
    if not video_path.is_absolute():
        video_path = (PROJECT_ROOT / video_path).resolve()

    if video_path.is_dir():
        video_paths = sorted(
            path for path in video_path.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not video_paths:
            raise RuntimeError(f"No video files found in input directory: {video_path}")
        return video_paths

    return [video_path]


def resolve_preprocessed_video_inputs(video_dir: Path) -> list[Path]:
    # Prefer the nested export layout produced by the preprocessing pipeline.
    nested_paths = sorted(
        path.resolve()
        for path in video_dir.rglob("preprocessed.mp4")
        if path.is_file() and path.parent.name.lower() in VALID_PREPROCESS_METHODS
    )
    if nested_paths:
        return nested_paths

    video_paths = sorted(
        path.resolve()
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not video_paths:
        raise RuntimeError(f"No preprocessed video files found in input directory: {video_dir}")
    return video_paths


def normalize_preprocessing_methods(
    methods: list[str] | tuple[str, ...] | None,
) -> list[str] | None:
    if methods is None:
        return None

    normalized = []
    for method in methods:
        method_l = str(method).strip().lower()
        if method_l not in VALID_PREPROCESS_METHODS:
            raise ValueError(
                f"Unsupported preprocessing method '{method}'. "
                f"Expected one of {list(VALID_PREPROCESS_METHODS)}."
            )
        if method_l not in normalized:
            normalized.append(method_l)

    if not normalized:
        raise ValueError("At least one preprocessing method must be provided.")
    return normalized


def normalize_ic_heat_modes(modes: list[str] | tuple[str, ...]) -> list[str]:
    normalized = []
    for mode in modes:
        mode_l = str(mode).strip().lower()
        if mode_l not in VALID_IC_HEAT_MODES:
            raise ValueError(
                f"Unsupported IC heat mode '{mode}'. Expected one of {list(VALID_IC_HEAT_MODES)}."
            )
        if mode_l not in normalized:
            normalized.append(mode_l)

    if not normalized:
        raise ValueError("At least one IC heat mode must be provided.")
    return normalized


def normalize_phosphene_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in VALID_PHOSPHENE_MODES:
        raise ValueError(
            f"Unsupported phosphene mode '{mode}'. Expected one of {list(VALID_PHOSPHENE_MODES)}."
        )
    return normalized


def split_preprocessed_video_stem(stem: str) -> tuple[str, str | None]:
    stem_l = stem.lower()
    for method in VALID_PREPROCESS_METHODS:
        suffix = f"_{method}"
        if stem_l.endswith(suffix):
            return stem[:-len(suffix)], method
    return stem, None


def describe_preprocessed_video_path(video_path: Path) -> tuple[str, str | None]:
    if video_path.stem.lower() == "preprocessed":
        method = video_path.parent.name.lower()
        if method in VALID_PREPROCESS_METHODS:
            return video_path.parent.parent.name, method
    return split_preprocessed_video_stem(video_path.stem)


def resolve_preprocessed_video_runs(
    video_arg: str,
    preprocessing_methods: list[str] | None,
) -> list[dict]:
    video_path = Path(video_arg)
    if not video_path.is_absolute():
        video_path = (PROJECT_ROOT / video_path).resolve()

    requested_methods = preprocessing_methods

    if video_path.is_dir():
        video_paths = resolve_preprocessed_video_inputs(video_path)
        runs = []
        for path in video_paths:
            base_stem, detected_method = describe_preprocessed_video_path(path)
            if requested_methods is not None and detected_method not in requested_methods:
                continue
            runs.append(
                {
                    "video_path": path.resolve(),
                    "preprocessing_method": detected_method or "unspecified",
                    "base_stem": base_stem,
                }
            )

        if not runs:
            requested = ", ".join(requested_methods or [])
            raise RuntimeError(
                f"No preprocessed videos matching methods [{requested}] were found in: {video_path}"
            )
        return runs

    base_stem, detected_method = describe_preprocessed_video_path(video_path)
    if requested_methods is None:
        return [
            {
                "video_path": video_path.resolve(),
                "preprocessing_method": detected_method or "unspecified",
                "base_stem": base_stem,
            }
        ]

    candidate_roots = []
    nested_root = None
    if (
        video_path.stem.lower() == "preprocessed"
        and video_path.parent.name.lower() in VALID_PREPROCESS_METHODS
        and video_path.parent.parent != video_path.parent
    ):
        nested_root = video_path.parent.parent.parent
    # Search close-by variants first, then fall back to the standard repo folder.
    for candidate_root in [nested_root, video_path.parent, (PROJECT_ROOT / "videos" / "preprocessed").resolve()]:
        if candidate_root is not None and candidate_root not in candidate_roots:
            candidate_roots.append(candidate_root)

    runs = []
    missing = []
    preferred_suffix = video_path.suffix.lower()

    for method in requested_methods:
        found_path = None
        for candidate_root in candidate_roots:
            nested_candidate = candidate_root / base_stem / method / "preprocessed.mp4"
            if nested_candidate.exists():
                found_path = nested_candidate.resolve()
                break

            suffixes = [preferred_suffix] if preferred_suffix else []
            suffixes.extend(ext for ext in sorted(VIDEO_EXTENSIONS) if ext not in suffixes)
            for suffix in suffixes:
                candidate_path = candidate_root / f"{base_stem}_{method}{suffix}"
                if candidate_path.exists():
                    found_path = candidate_path.resolve()
                    break
            if found_path is not None:
                break

        if found_path is None:
            missing.append(method)
            continue

        runs.append(
            {
                "video_path": found_path,
                "preprocessing_method": method,
                "base_stem": base_stem,
            }
        )

    if missing:
        searched_dirs = ", ".join(str(path) for path in candidate_roots)
        missing_str = ", ".join(missing)
        raise RuntimeError(
            f"Could not find preprocessed video variants for method(s) [{missing_str}] "
            f"using base name '{base_stem}'. Searched in: {searched_dirs}"
        )

    return runs


def normalize_raster_modes(modes: list[str] | tuple[str, ...]) -> list[str]:
    normalized = []
    for mode in modes:
        mode_l = normalize_raster_label(mode)
        if mode_l not in {normalize_raster_label(value) for value in VALID_RASTER_MODES}:
            raise ValueError(
                f"Unsupported raster mode '{mode}'. Expected one of {list(VALID_RASTER_MODES)}."
            )
        if mode_l not in normalized:
            normalized.append(mode_l)
    if not normalized:
        raise ValueError("At least one raster mode must be provided.")
    return normalized


def build_video_output_root(output_root: Path, video_path: Path, stem_override: str | None = None) -> Path:
    # Keep the output tree compact and organized by logical video name instead
    # of mirroring the full source path on disk.
    stem = sanitize_path_part(stem_override or video_path.stem)
    return output_root / stem


def write_run_manifest(out_dir: Path, *, video_path: Path, params_path: Path,
                       safety_path: Path, coords_yaml: Path,
                       preprocessing_method: str, ic_heat_mode: str,
                       raster_name: str,
                       appearance_threshold_uA: float | None,
                       internal_circuit_power_mw: float,
                       stim_scale: float | None,
                       stimulus_scale_base: float,
                       stimulus_scale_effective: float,
                       phosphene_mode: str,
                       enable_cem43: bool):
    manifest = {
        "video": str(video_path),
        "preprocessing_method": str(preprocessing_method),
        "internal_circuit_heat_mode": str(ic_heat_mode),
        "raster_mode": str(raster_name),
        "raster_mode_normalized": normalize_raster_mode(raster_name),
        "appearance_threshold_uA": (
            None if appearance_threshold_uA is None else float(appearance_threshold_uA)
        ),
        "internal_circuit_power_mw": float(internal_circuit_power_mw),
        "stim_scale": None if stim_scale is None else float(stim_scale),
        "stimulus_scale_base": float(stimulus_scale_base),
        "stimulus_scale_effective": float(stimulus_scale_effective),
        "phosphene_mode": str(phosphene_mode),
        "enable_cem43": bool(enable_cem43),
        "params": str(params_path),
        "safety_yaml": str(safety_path),
        "coords_yaml": str(coords_yaml),
    }
    with open(out_dir / "run_manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


# Raster timing helpers.
def resolve_video_fps(cap: cv2.VideoCapture, video_path: Path, fallback_fps: float) -> tuple[float, str]:
    video_fps = float(cap.get(cv2.CAP_PROP_FPS))
    if video_fps > 0:
        return video_fps, "video"
    if float(fallback_fps) > 0:
        print(
            f"Warning: could not read FPS from {video_path.name}; "
            f"falling back to params FPS = {float(fallback_fps):.6f}."
        )
        return float(fallback_fps), "params_fallback"
    raise RuntimeError(f"Unable to determine FPS for input video: {video_path}")


def compute_raster_timing(video_fps: float, groups: int, raster_enabled: bool) -> dict[str, float]:
    if float(video_fps) <= 0:
        raise ValueError(f"Video FPS must be > 0, got {video_fps}.")
    if int(groups) <= 0:
        raise ValueError(f"Raster groups must be > 0, got {groups}.")

    if not raster_enabled:
        return {
            "video_fps": float(video_fps),
            "cycle_rate_hz": 0.0,
            "group_step_rate_hz": 0.0,
            "group_interval_s": 0.0,
        }

    # GaussianSimulator expects raster_rate_hz to be the full cycle rate.
    # If there are N groups and the video advances one group per frame,
    # then the cycle rate is video_fps / N.
    return {
        "video_fps": float(video_fps),
        "cycle_rate_hz": float(video_fps) / float(groups),
        "group_step_rate_hz": float(video_fps),
        "group_interval_s": 1.0 / float(video_fps),
    }


def validate_and_report_raster_timing(
    sim: GaussianSimulator,
    *,
    video_path: Path,
    raster_name: str,
    groups: int,
    fps_source: str,
    timing: dict[str, float],
):
    if not sim.raster_enabled:
        return

    # Confirm that the simulator timing matches the intended setup:
    # one group step per frame and one full cycle every `groups` frames.
    actual_cycle_rate_hz = float(sim.raster_rate_hz)
    actual_group_interval_s = float(sim.raster_group_interval_s)
    actual_group_step_rate_hz = 0.0
    if np.isfinite(actual_group_interval_s) and actual_group_interval_s > 0:
        actual_group_step_rate_hz = 1.0 / actual_group_interval_s

    if not np.isclose(actual_cycle_rate_hz, timing["cycle_rate_hz"], rtol=1e-6, atol=1e-9):
        raise RuntimeError(
            f"Raster cycle rate mismatch for {video_path.name} ({raster_name}): "
            f"expected {timing['cycle_rate_hz']:.6f} Hz from video_fps/groups, "
            f"got {actual_cycle_rate_hz:.6f} Hz."
        )
    if not np.isclose(actual_group_interval_s, timing["group_interval_s"], rtol=1e-6, atol=1e-9):
        raise RuntimeError(
            f"Raster group interval mismatch for {video_path.name} ({raster_name}): "
            f"expected {timing['group_interval_s']:.9f} s, got {actual_group_interval_s:.9f} s."
        )

    print(
        f"Raster timing | video={video_path.name} | mode={raster_name} | "
        f"fps={timing['video_fps']:.6f} ({fps_source}) | groups={int(groups)} | "
        f"cycle_rate={actual_cycle_rate_hz:.6f} Hz | "
        f"group_step_rate={actual_group_step_rate_hz:.6f} Hz | "
        f"group_interval={actual_group_interval_s:.9f} s | validated=1_frame_per_group_step"
    )


def resolve_simulation_frame_count(cap: cv2.VideoCapture, max_frames: int) -> int:
    reported_frames = int(round(float(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    if reported_frames > 0:
        if max_frames > 0:
            return min(reported_frames, int(max_frames))
        return reported_frames
    if max_frames > 0:
        return int(max_frames)
    return 0


def build_heatmap_snapshot_schedule(total_frames: int, dt: float) -> list[dict[str, float | int]]:
    if total_frames <= 0:
        return []

    schedule = []
    seen_frames = set()
    total_frames_f = float(total_frames)
    for target_fraction in HEATMAP_SNAPSHOT_FRACTIONS:
        frame_idx = int(np.clip(np.ceil(total_frames_f * float(target_fraction)), 1, total_frames))
        if frame_idx in seen_frames:
            continue
        seen_frames.add(frame_idx)
        schedule.append(
            {
                "frame_idx": frame_idx,
                "time_s": float(frame_idx) * float(dt),
                "target_fraction": float(target_fraction),
                "actual_fraction": float(frame_idx) / total_frames_f,
            }
        )
    return schedule


def safe_nanmax(values, default: float = np.nan) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return float(default)
    try:
        return float(np.nanmax(array))
    except ValueError:
        return float(default)


def safe_last(values, default: float = np.nan) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[-1])


def scalar_from_value(value, default: float = np.nan) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def derive_charge_density_uc_cm2(
    charge_per_phase_nC: np.ndarray,
    electrode_surface_area_cm2: float,
) -> np.ndarray:
    charge_per_phase_nC = np.asarray(charge_per_phase_nC, dtype=np.float32)
    if not np.isfinite(electrode_surface_area_cm2) or electrode_surface_area_cm2 <= 0:
        return np.zeros_like(charge_per_phase_nC, dtype=np.float32)
    return charge_per_phase_nC / (1e3 * float(electrode_surface_area_cm2))


def derive_current_amplitude_uA(
    charge_per_phase_nC: np.ndarray,
    pulse_width_s: np.ndarray,
) -> np.ndarray:
    charge_per_phase_nC = np.asarray(charge_per_phase_nC, dtype=np.float32)
    pulse_width_s = np.asarray(pulse_width_s, dtype=np.float32)
    safe_pulse_width = np.where(pulse_width_s > 0, pulse_width_s, np.nan).astype(np.float32)
    current = np.zeros_like(charge_per_phase_nC, dtype=np.float32)
    np.divide(
        charge_per_phase_nC,
        safe_pulse_width * 1e9,
        out=current,
        where=np.isfinite(safe_pulse_width),
    )
    return current


def derive_shannon_k(
    charge_per_phase_nC: np.ndarray,
    electrode_surface_area_cm2: float,
) -> np.ndarray:
    charge_per_phase_nC = np.asarray(charge_per_phase_nC, dtype=np.float32)
    charge_density_uc_cm2 = derive_charge_density_uc_cm2(
        charge_per_phase_nC,
        electrode_surface_area_cm2,
    )
    charge_per_phase_uC = charge_per_phase_nC / 1e3
    shannon_k = np.full_like(charge_per_phase_uC, -np.inf, dtype=np.float32)
    valid = (charge_per_phase_uC > 0) & (charge_density_uc_cm2 > 0)
    shannon_k[valid] = (
        np.log10(charge_per_phase_uC[valid]) +
        np.log10(charge_density_uc_cm2[valid])
    ).astype(np.float32)
    return shannon_k


def build_run_summary_text(
    *,
    video_path: Path,
    preprocessing_method: str,
    ic_heat_mode: str,
    raster_name: str,
    coords_yaml: Path,
    stim_scale: float | None,
    stimulus_scale_base: float,
    stimulus_scale_effective: float,
    total_frames: int,
    video_fps: float,
    raster_timing: dict[str, float],
    groups: int,
    internal_circuit_power_mw: float,
    pulse_frequency_hz: np.ndarray,
    common_metrics: dict[str, np.ndarray],
    thermal_metrics: dict[str, np.ndarray],
    thermal_grids: dict[str, dict],
    thermal_snapshots: dict[str, object] | None,
) -> str:
    time_s = np.asarray(common_metrics["time_s"])
    final_time_s = safe_last(time_s, default=0.0)
    total_charge_rate = np.asarray(common_metrics["total_charge_per_second_nC"])
    total_window_charge = np.asarray(common_metrics["total_window_nC"])
    total_protocol_charge = np.asarray(common_metrics["total_protocol_nC"])
    protocol_charge_per_grid = common_metrics.get("total_protocol_nC_per_grid", {})
    stimulated_count = np.asarray(common_metrics["stimulated_electrode_count"])
    active_count = np.asarray(common_metrics["active_count"])
    max_dT = np.asarray(thermal_metrics["max_dT"])
    mean_dT = np.asarray(thermal_metrics["mean_dT"])
    area_gt1 = np.asarray(thermal_metrics["area_gt1_mm2"])
    area_gt2 = np.asarray(thermal_metrics["area_gt2_mm2"])
    area_gt3 = np.asarray(thermal_metrics["area_gt3_mm2"])
    max_cem43 = np.asarray(thermal_metrics.get("max_cem43", []))
    p99_cem43 = np.asarray(thermal_metrics.get("p99_cem43", []))
    electrode_time_s = np.asarray(common_metrics.get("electrode_time_s", time_s))
    sampled_charge_per_phase = np.asarray(common_metrics.get("charge_per_phase_nC", []))
    pulse_width_s = np.asarray(common_metrics.get("pulse_width_s", []))
    electrode_surface_area_cm2 = scalar_from_value(
        common_metrics.get("electrode_surface_area_cm2", np.asarray(np.nan, dtype=np.float32))
    )
    fixed_firing_threshold_uA = scalar_from_value(
        common_metrics.get("fixed_firing_threshold_uA", np.asarray(np.nan, dtype=np.float32))
    )
    input_binarized_for_safety = bool(
        scalar_from_value(
            common_metrics.get("input_binarized_for_safety", np.asarray(False, dtype=np.bool_)),
            default=0.0,
        )
    )
    temporal_dynamics_disabled = bool(
        scalar_from_value(
            common_metrics.get("temporal_dynamics_disabled", np.asarray(False, dtype=np.bool_)),
            default=0.0,
        )
    )
    trace_increase_rate = scalar_from_value(
        common_metrics.get("trace_increase_rate", np.asarray(np.nan, dtype=np.float32))
    )
    activation_threshold_sd = scalar_from_value(
        common_metrics.get("activation_threshold_sd", np.asarray(np.nan, dtype=np.float32))
    )
    metrics_save_every_n_frames = int(
        scalar_from_value(
            common_metrics.get("electrode_metrics_save_every_n_frames", np.asarray(1, dtype=np.int32)),
            default=1.0,
        )
    )
    peak_charge_per_phase_nC = scalar_from_value(
        common_metrics.get(
            "peak_charge_per_phase_nC_exact",
            np.asarray(safe_nanmax(sampled_charge_per_phase), dtype=np.float32),
        )
    )
    peak_current_amplitude_uA = scalar_from_value(
        common_metrics.get(
            "peak_current_amplitude_uA_exact",
            np.asarray(
                safe_nanmax(derive_current_amplitude_uA(sampled_charge_per_phase, pulse_width_s))
                if sampled_charge_per_phase.size > 0 and pulse_width_s.size > 0 else np.nan,
                dtype=np.float32,
            ),
        )
    )
    peak_charge_density_uc_cm2 = scalar_from_value(
        common_metrics.get(
            "peak_charge_density_uc_cm2_exact",
            np.asarray(
                safe_nanmax(
                    derive_charge_density_uc_cm2(
                        sampled_charge_per_phase,
                        electrode_surface_area_cm2,
                    )
                ) if sampled_charge_per_phase.size > 0 else np.nan,
                dtype=np.float32,
            ),
        )
    )
    peak_shannon_k = scalar_from_value(
        common_metrics.get(
            "peak_shannon_k_exact",
            np.asarray(
                safe_nanmax(
                    derive_shannon_k(
                        sampled_charge_per_phase,
                        electrode_surface_area_cm2,
                    )
                ) if sampled_charge_per_phase.size > 0 else np.nan,
                dtype=np.float32,
            ),
        )
    )

    snapshot_times = np.asarray(
        [] if thermal_snapshots is None else thermal_snapshots.get("times_s", []),
    )
    snapshot_target_fractions = np.asarray(
        [] if thermal_snapshots is None else thermal_snapshots.get("target_fractions", []),
    )

    lines = [
        "Simulation configuration",
        f"video={video_path}",
        f"preprocessing_method={preprocessing_method}",
        f"ic_heat_mode={ic_heat_mode}",
        f"raster_mode={raster_name}",
        f"coords_yaml={coords_yaml}",
        f"stim_scale={'default' if stim_scale is None else f'{float(stim_scale):.6f}'}",
        f"stimulus_scale_base={float(stimulus_scale_base):.6f}",
        f"stimulus_scale_effective={float(stimulus_scale_effective):.6f}",
        f"input_binarized_for_safety={input_binarized_for_safety}",
        f"temporal_dynamics_disabled={temporal_dynamics_disabled}",
        f"trace_increase_rate={trace_increase_rate:.6g}",
        f"activation_threshold_sd={activation_threshold_sd:.6g}",
        f"frames={int(total_frames)}",
        f"duration_s={final_time_s:.6f}",
        f"fps={float(video_fps):.6f}",
        f"raster_groups={int(groups)}",
        f"raster_cycle_rate_hz={float(raster_timing['cycle_rate_hz']):.6f}",
        f"raster_group_step_rate_hz={float(raster_timing['group_step_rate_hz']):.6f}",
        f"raster_group_interval_s={float(raster_timing['group_interval_s']):.6f}",
        f"internal_circuit_power_total_mW={float(internal_circuit_power_mw):.6f}",
        f"electrode_metrics_saved_every_n_frames={int(metrics_save_every_n_frames)}",
        f"electrode_metrics_saved_count={int(electrode_time_s.size)}",
        f"thermal_grid_names={', '.join(sorted(thermal_grids.keys()))}",
        "",
        "Saved heatmaps",
        f"heatmap_target_count={int(HEATMAP_SNAPSHOT_COUNT)}",
        f"heatmap_saved_count={int(snapshot_times.size)}",
    ]

    if snapshot_times.size > 0:
        lines.append(
            "heatmap_target_fractions="
            + ", ".join(f"{100.0 * value:.1f}%" for value in snapshot_target_fractions)
        )
        lines.append(
            "heatmap_times_s="
            + ", ".join(f"{value:.3f}" for value in snapshot_times)
        )

    lines.extend(
        [
            "",
            "Electrical results",
            f"pulse_frequency_hz_min={float(np.min(pulse_frequency_hz)):.6f}",
            f"pulse_frequency_hz_mean={float(np.mean(pulse_frequency_hz)):.6f}",
            f"pulse_frequency_hz_max={float(np.max(pulse_frequency_hz)):.6f}",
            f"fixed_firing_threshold_uA={fixed_firing_threshold_uA:.6f}",
            f"peak_current_amplitude_uA={peak_current_amplitude_uA:.6f}",
            f"peak_charge_per_phase_nC={peak_charge_per_phase_nC:.6f}",
            f"peak_charge_density_uC_cm2={peak_charge_density_uc_cm2:.6f}",
            f"peak_shannon_k={peak_shannon_k:.6f}",
            f"peak_total_charge_per_second_nC_s={safe_nanmax(total_charge_rate):.6f}",
            f"peak_total_window_charge_nC={safe_nanmax(total_window_charge):.6f}",
            f"final_total_protocol_charge_nC={safe_last(total_protocol_charge):.6f}",
            f"peak_active_electrode_count={int(safe_nanmax(active_count, default=0.0))}",
            f"peak_stimulated_electrode_count={int(safe_nanmax(stimulated_count, default=0.0))}",
            "",
            "Thermal results",
            f"peak_max_dT_C={safe_nanmax(max_dT):.6f}",
            f"final_max_dT_C={safe_last(max_dT):.6f}",
            f"peak_mean_dT_C={safe_nanmax(mean_dT):.6f}",
            f"final_mean_dT_C={safe_last(mean_dT):.6f}",
            f"peak_area_gt1_mm2={safe_nanmax(area_gt1):.6f}",
            f"peak_area_gt2_mm2={safe_nanmax(area_gt2):.6f}",
            f"peak_area_gt3_mm2={safe_nanmax(area_gt3):.6f}",
        ]
    )
    if max_cem43.size > 0 and p99_cem43.size > 0:
        lines.extend(
            [
                f"peak_max_cem43_min={safe_nanmax(max_cem43):.6f}",
                f"final_max_cem43_min={safe_last(max_cem43):.6f}",
                f"peak_p99_cem43_min={safe_nanmax(p99_cem43):.6f}",
                f"final_p99_cem43_min={safe_last(p99_cem43):.6f}",
            ]
        )

    for grid_name in sorted(thermal_grids.keys()):
        grid_data = thermal_grids[grid_name]
        grid_max = safe_nanmax(grid_data["dT_final"])
        grid_cem43_max = safe_nanmax(grid_data.get("cem43_final", []))
        if grid_name in protocol_charge_per_grid:
            grid_protocol_charge = np.asarray(protocol_charge_per_grid[grid_name])
            lines.append(f"final_total_protocol_charge_nC[{grid_name}]={safe_last(grid_protocol_charge):.6f}")
        lines.append(f"final_grid_peak_dT_C[{grid_name}]={grid_max:.6f}")
        if "cem43_final" in grid_data:
            lines.append(f"final_grid_peak_cem43_min[{grid_name}]={grid_cem43_max:.6f}")

    return "\n".join(lines) + "\n"


def write_run_summary(out_dir: Path, summary_text: str) -> None:
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write(summary_text)


def load_tagged_mapping(params: dict, coords_yaml: Path):
    x_raw_mm, y_raw_mm = utils.load_coordinates_from_yaml(str(coords_yaml))
    x_raw_mm = np.asarray(x_raw_mm, dtype=float)
    y_raw_mm = np.asarray(y_raw_mm, dtype=float)

    coords_cortex_input = Map(x=x_raw_mm, y=y_raw_mm)
    rng = np.random.default_rng(int(params.get("run", {}).get("seed", 42)))
    if hasattr(cortex_models, "get_full_field_mapping_from_cortex"):
        mapping = cortex_models.get_full_field_mapping_from_cortex(
            params["cortex_model"],
            coordinates_cortex=coords_cortex_input,
            rng=rng,
        )
        return {
            "phosphene_map": mapping.phosphene_map,
            "indices": np.asarray(mapping.indices, dtype=np.int64),
            "cortical_coordinates": mapping.cortical_coordinates,
            "grid_ids": np.asarray(mapping.grid_ids, dtype=np.int64),
            "base_indices": np.asarray(mapping.base_indices, dtype=np.int64),
        }

    if hasattr(cortex_models, "get_visual_field_coordinates_from_cortex_full"):
        phosphene_map, remaining_indices = cortex_models.get_visual_field_coordinates_from_cortex_full(
            params["cortex_model"],
            coordinates_cortex=coords_cortex_input,
            rng=rng,
        )
    else:
        phosphene_map, remaining_indices = cortex_models.get_visual_field_coordinates_from_cortex(
            params["cortex_model"],
            coordinates_cortex=coords_cortex_input,
            rng=rng,
        )

    remaining_indices = np.asarray(remaining_indices, dtype=np.int64)
    x_expanded, y_expanded = cortex_models.expand_electrode_coordinate_arrays(
        x_raw_mm,
        y_raw_mm,
        remaining_indices,
    )
    return {
        "phosphene_map": phosphene_map,
        "indices": remaining_indices,
        "cortical_coordinates": Map(x=x_expanded[remaining_indices], y=y_expanded[remaining_indices]),
        "grid_ids": np.zeros(len(remaining_indices), dtype=np.int64),
        "base_indices": remaining_indices.copy(),
    }


# Coordinate and simulator setup helpers.
def build_surviving_electrode_set(
    cortical_coordinates: Map,
    remaining_indices: np.ndarray,
    device: torch.device,
):
    x_coords, y_coords = cortical_coordinates.cartesian
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)
    phys_idx_unique, first_idx, inv_map = np.unique(
        np.asarray(remaining_indices, dtype=np.int64),
        return_index=True,
        return_inverse=True,
    )

    elec_xy_mm_surv = np.stack(
        [x_coords[first_idx], y_coords[first_idx]], axis=1
    ).astype(np.float64)
    inv_map_t = torch.tensor(inv_map.astype(np.int64), dtype=torch.long, device=device)
    return elec_xy_mm_surv, inv_map_t


def build_surviving_electrode_metadata(
    cortical_coordinates: Map,
    remaining_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x_coords, y_coords = cortical_coordinates.cartesian
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)
    electrode_ids, first_idx = np.unique(
        np.asarray(remaining_indices, dtype=np.int64),
        return_index=True,
    )
    elec_xy_mm = np.stack(
        [x_coords[first_idx], y_coords[first_idx]],
        axis=1,
    ).astype(np.float64)
    return electrode_ids.astype(np.int64), elec_xy_mm


def build_electrode_grid_ids(
    remaining_indices: np.ndarray,
    grid_ids: np.ndarray,
) -> np.ndarray:
    remaining_indices = np.asarray(remaining_indices, dtype=np.int64)
    grid_ids = np.asarray(grid_ids, dtype=np.int64)
    _, first_idx = np.unique(remaining_indices, return_index=True)
    return grid_ids[first_idx].astype(np.int64)


def align_mapping_to_simulator(mapping: dict, sim: GaussianSimulator) -> dict:
    order = np.asarray(
        getattr(sim, "electrode_tags", np.arange(len(mapping["indices"]), dtype=np.int64)),
        dtype=np.int64,
    )
    if len(order) != len(mapping["indices"]):
        raise RuntimeError(
            "Simulator electrode ordering does not match phosphene mapping length."
        )

    x_coords, y_coords = mapping["cortical_coordinates"].cartesian
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)

    return {
        "phosphene_map": mapping["phosphene_map"],
        "indices": np.asarray(mapping["indices"], dtype=np.int64)[order],
        "grid_ids": np.asarray(mapping["grid_ids"], dtype=np.int64)[order],
        "base_indices": np.asarray(mapping["base_indices"], dtype=np.int64)[order],
        "cortical_coordinates": Map(x=x_coords[order], y=y_coords[order]),
    }


def build_bioheat_grid_sets(
    cortical_coordinates: Map,
    grid_ids: np.ndarray,
    base_indices: np.ndarray,
    device: torch.device,
) -> dict[int, dict]:
    x_coords, y_coords = cortical_coordinates.cartesian
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)
    grid_ids = np.asarray(grid_ids, dtype=np.int64)
    base_indices = np.asarray(base_indices, dtype=np.int64)

    grid_sets = {}
    for grid_id in np.unique(grid_ids):
        mask = grid_ids == int(grid_id)
        local_indices = base_indices[mask]
        x_grid = x_coords[mask]
        y_grid = y_coords[mask]
        unique_local, first_idx, inverse = np.unique(
            local_indices,
            return_index=True,
            return_inverse=True,
        )
        grid_sets[int(grid_id)] = {
            "name": "right" if int(grid_id) == 0 else "left",
            "mask_t": torch.tensor(mask, dtype=torch.bool, device=device),
            "inverse_t": torch.tensor(inverse.astype(np.int64), dtype=torch.long, device=device),
            "elec_xy_mm": np.stack([x_grid[first_idx], y_grid[first_idx]], axis=1).astype(np.float64),
            "n_elec": int(len(unique_local)),
        }
    return grid_sets


def move_simulator_tensors_to_device(sim: GaussianSimulator, device: torch.device):
    # Some simulator tensors are created eagerly, so we migrate them explicitly
    # after construction to keep the script device-agnostic.
    if hasattr(sim, "data_kwargs") and isinstance(sim.data_kwargs, dict):
        sim.data_kwargs["device"] = str(device)

    for attr in ["phosphene_maps"]:
        if hasattr(sim, attr):
            v = getattr(sim, attr)
            if torch.is_tensor(v):
                setattr(sim, attr, v.to(device))

    for attr in ["activation", "trace", "sigma", "brightness", "threshold", "impedance"]:
        if hasattr(sim, attr):
            obj = getattr(sim, attr)
            if hasattr(obj, "state") and torch.is_tensor(obj.state):
                obj.state = obj.state.to(device)

    for attr in ["_pulse_width", "_frequency", "_zero", "_inf", "cumulative_charge_uC"]:
        if hasattr(sim, attr):
            v = getattr(sim, attr)
            if torch.is_tensor(v):
                setattr(sim, attr, v.to(device))

    if hasattr(sim, "raster_schedule") and sim.raster_schedule is not None:
        sim.raster_schedule = [m.to(device) for m in sim.raster_schedule]


def aggregate_metric_tensor(metric: torch.Tensor, inv_map_t: torch.Tensor, n_elec: int) -> torch.Tensor:
    metric = metric.reshape(-1).to(inv_map_t.device)
    out = torch.zeros(n_elec, device=inv_map_t.device, dtype=metric.dtype)
    out.scatter_add_(0, inv_map_t, metric)
    return out


def aggregate_metric(metric: torch.Tensor, inv_map_t: torch.Tensor, n_elec: int) -> np.ndarray:
    return aggregate_metric_tensor(metric, inv_map_t, n_elec).detach().cpu().numpy().astype(np.float32)


def tensor_list_to_numpy(values: list[torch.Tensor], dtype=np.float32) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=dtype)
    stacked = torch.stack([value.detach().reshape(()) for value in values])
    return stacked.cpu().numpy().astype(dtype, copy=False)


def tensor_rows_to_numpy(values: list[torch.Tensor], dtype=np.float32) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=dtype)
    stacked = torch.stack([value.detach().reshape(-1) for value in values], dim=0)
    return stacked.cpu().numpy().astype(dtype, copy=False)


def scalar_tensor_to_numpy(value: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def prepare_frame(frame: np.ndarray, target_res: tuple[int, int]) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.shape != target_res:
        frame = cv2.resize(frame, (target_res[1], target_res[0]), interpolation=cv2.INTER_AREA)
    return frame


def should_binarize_preprocessed_input(preprocessing_method: str) -> bool:
    return str(preprocessing_method).strip().lower() in {"groundtruth", "canny"}


def restore_binary_preprocessed_frame(frame: np.ndarray) -> np.ndarray:
    if frame.size == 0 or int(np.max(frame)) <= 0:
        return np.zeros_like(frame, dtype=np.uint8)
    frame_u8 = np.asarray(frame, dtype=np.uint8)
    _, binary = cv2.threshold(frame_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


# Plotting helpers.
def time_axis_minutes(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) / 60.0


def plot_series(x: np.ndarray, y: np.ndarray, out_path: Path, title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(time_axis_minutes(x), y, linewidth=2.0, color=ACADEMIC_COLORS["mean"])
    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dual_series(x: np.ndarray, y1: np.ndarray, y2: np.ndarray, out_path: Path,
                     title: str, ylabel: str, label1: str, label2: str):
    plt.figure(figsize=(8, 4))
    x_min = time_axis_minutes(x)
    plt.plot(x_min, y1, linewidth=1.8, label=label1)
    plt.plot(x_min, y2, linewidth=1.8, label=label2)
    plt.xlabel("time (min)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cloud_series(x: np.ndarray, values: np.ndarray, out_path: Path,
                      title: str, ylabel: str, smooth_window: int = 9):
    values = np.asarray(values, dtype=np.float64)
    plot_values = values.copy()
    plot_values[plot_values <= 0] = np.nan

    # Show the envelope across electrodes instead of drawing every trace.
    with np.errstate(invalid="ignore"):
        min_vals = np.nanmin(plot_values, axis=1)
        mean_vals = np.nanmean(plot_values, axis=1)
        max_vals = np.nanmax(plot_values, axis=1)

    min_vals = np.nan_to_num(min_vals, nan=0.0)
    mean_vals = np.nan_to_num(mean_vals, nan=0.0)
    max_vals = np.nan_to_num(max_vals, nan=0.0)

    min_smooth = smooth_series(min_vals, smooth_window)
    mean_smooth = smooth_series(mean_vals, smooth_window)
    max_smooth = smooth_series(max_vals, smooth_window)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.fill_between(
        time_axis_minutes(x), min_smooth, max_smooth, alpha=0.35,
        color=ACADEMIC_COLORS["band"], label="Range"
    )
    ax.plot(time_axis_minutes(x), mean_smooth, color=ACADEMIC_COLORS["mean"], linewidth=2.2, label="Mean")
    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multi_series(x: np.ndarray, series: list[np.ndarray], labels: list[str],
                      out_path: Path, title: str, ylabel: str,
                      ymin: float | None = None):
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#1F4E79", "#2E8B57", "#8B1E3F"]
    x_min = time_axis_minutes(x)
    for y, label, color in zip(series, labels, colors):
        ax.plot(x_min, y, linewidth=2.0, label=label, color=color)
    ax.set_xlabel("Time (min)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)
    _style_axes(ax)
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_temperature_snapshot(dT_map: np.ndarray, extent_mm: tuple[float, float, float, float],
                              out_path: Path, time_s: float):
    xmin, xmax, ymin, ymax = extent_mm
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(
        dT_map,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        aspect="equal",
        cmap="inferno",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("ΔT (°C)", fontsize=11)
    ax.set_xlabel("x (mm)", fontsize=11)
    ax.set_ylabel("y (mm)", fontsize=11)
    ax.set_title(f"Temperature Rise at t = {time_s / 60.0:.2f} min", fontsize=12, pad=10)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mode_outputs(
    *,
    out_dir: Path,
    video_fps: float,
    raster_rate_hz: float,
    raster_timing: dict[str, float],
    pulse_frequency_hz: np.ndarray,
    common_metrics: dict[str, np.ndarray],
    thermal_metrics: dict[str, np.ndarray],
    thermal_grids: dict[str, dict],
    thermal_snapshots: dict[str, object] | None,
    internal_circuit_power_mw: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    if thermal_grids:
        hottest_grid_name = max(
            thermal_grids,
            key=lambda name: float(np.max(thermal_grids[name]["dT_final"])),
        )
        cem43_grid_names = [
            name for name, grid_data in thermal_grids.items()
            if "cem43_final" in grid_data
        ]
        extent_mm = np.asarray(thermal_grids[hottest_grid_name]["extent_mm"], dtype=np.float32)
        if cem43_grid_names:
            hottest_cem43_grid_name = max(
                cem43_grid_names,
                key=lambda name: float(np.max(thermal_grids[name]["cem43_final"])),
            )
            cem43_extent_mm = np.asarray(thermal_grids[hottest_cem43_grid_name]["extent_mm"], dtype=np.float32)
        else:
            hottest_cem43_grid_name = None
            cem43_extent_mm = None
        footprint_pixels = float(thermal_grids[hottest_grid_name]["ic_footprint_pixel_count"])
        power_density_w_m3 = float(thermal_grids[hottest_grid_name]["ic_power_density_W_m3"])
    else:
        hottest_grid_name = "aggregate"
        hottest_cem43_grid_name = None
        extent_mm = np.zeros(4, dtype=np.float32)
        cem43_extent_mm = None
        footprint_pixels = 0.0
        power_density_w_m3 = 0.0

    save_payload = {
        "time_s": common_metrics["time_s"],
        "electrode_time_s": common_metrics["electrode_time_s"],
        "electrode_frame_indices": common_metrics["electrode_frame_indices"],
        "electrode_metrics_save_every_n_frames": common_metrics["electrode_metrics_save_every_n_frames"],
        "frame_charge_total_nC": common_metrics["frame_charge_total_nC"],
        "charge_per_second_total_nC_s": common_metrics["total_charge_per_second_nC"],
        "charge_per_phase_mean_nC": common_metrics["mean_charge_per_phase_nC"],
        "charge_density_mean_uc_cm2": common_metrics["mean_charge_density_uc_cm2"],
        "shannon_k_mean": common_metrics["mean_shannon_k"],
        "charge_per_second_mean_per_electrode_nC_s": common_metrics["mean_charge_per_second_per_electrode_nC_s"],
        "window_charge_total_nC": common_metrics["total_window_nC"],
        "protocol_charge_total_nC": common_metrics["total_protocol_nC"],
        "final_protocol_charge_per_electrode_nC": common_metrics["final_protocol_charge_per_electrode_nC"],
        "relative_stim_duration": common_metrics["relative_stim_duration"],
        "charge_window_s": common_metrics["charge_window_s"],
        "pulse_width_s": common_metrics["pulse_width_s"],
        "electrode_surface_area_cm2": common_metrics["electrode_surface_area_cm2"],
        "max_dT": thermal_metrics["max_dT"],
        "mean_dT": thermal_metrics["mean_dT"],
        "area_gt1_mm2": thermal_metrics["area_gt1_mm2"],
        "area_gt2_mm2": thermal_metrics["area_gt2_mm2"],
        "area_gt3_mm2": thermal_metrics["area_gt3_mm2"],
        "active_count": common_metrics["active_count"],
        "stimulated_electrode_count": common_metrics["stimulated_electrode_count"],
        "dT_final": thermal_metrics["dT_final"],
        "fps": np.asarray(video_fps, dtype=np.float32),
        "raster_rate_hz": np.asarray(raster_rate_hz, dtype=np.float32),
        "raster_group_step_rate_hz": np.asarray(raster_timing["group_step_rate_hz"], dtype=np.float32),
        "raster_group_interval_s": np.asarray(raster_timing["group_interval_s"], dtype=np.float32),
        "extent_mm": extent_mm,
        "pulse_frequency_hz": pulse_frequency_hz.astype(np.float32),
        "electrode_ids": common_metrics["electrode_ids"],
        "electrode_xy_mm": common_metrics["electrode_xy_mm"],
        "electrode_impedance_ohm": common_metrics["electrode_impedance_ohm"],
        "electrode_grid_ids": common_metrics["electrode_grid_ids"],
        "electrode_grid_names": common_metrics["electrode_grid_names"],
        "fixed_firing_threshold_uA": common_metrics["fixed_firing_threshold_uA"],
        "appearance_threshold_uA": common_metrics["appearance_threshold_uA"],
        "appearance_threshold_explicit": common_metrics["appearance_threshold_explicit"],
        "input_binarized_for_safety": common_metrics["input_binarized_for_safety"],
        "temporal_dynamics_disabled": common_metrics["temporal_dynamics_disabled"],
        "trace_increase_rate": common_metrics["trace_increase_rate"],
        "activation_threshold_sd": common_metrics["activation_threshold_sd"],
        "current_amplitude_per_electrode_uA": common_metrics["current_amplitude_per_electrode_uA"],
        "internal_circuit_power_total_mW": np.asarray(internal_circuit_power_mw, dtype=np.float32),
        "internal_circuit_footprint_pixels": np.asarray(footprint_pixels, dtype=np.float32),
        "internal_circuit_power_density_W_m3": np.asarray(power_density_w_m3, dtype=np.float32),
        "peak_charge_per_phase_nC_exact": common_metrics["peak_charge_per_phase_nC_exact"],
        "peak_current_amplitude_uA_exact": common_metrics["peak_current_amplitude_uA_exact"],
        "peak_charge_density_uc_cm2_exact": common_metrics["peak_charge_density_uc_cm2_exact"],
        "peak_shannon_k_exact": common_metrics["peak_shannon_k_exact"],
        "peak_charge_per_second_per_electrode_nC_s_exact": common_metrics["peak_charge_per_second_per_electrode_nC_s_exact"],
    }
    if "max_cem43" in thermal_metrics:
        save_payload["max_cem43"] = thermal_metrics["max_cem43"]
    if "p99_cem43" in thermal_metrics:
        save_payload["p99_cem43"] = thermal_metrics["p99_cem43"]
    if "cem43_final" in thermal_metrics:
        save_payload["cem43_final"] = thermal_metrics["cem43_final"]
    if cem43_extent_mm is not None:
        save_payload["cem43_extent_mm"] = cem43_extent_mm

    if "total_protocol_nC_per_grid" in common_metrics:
        save_payload["protocol_charge_grid_names"] = np.asarray(
            list(common_metrics["total_protocol_nC_per_grid"].keys())
        )
        for grid_name, series in common_metrics["total_protocol_nC_per_grid"].items():
            suffix = sanitize_path_part(grid_name)
            save_payload[f"protocol_charge_total_nC_{suffix}"] = np.asarray(series, dtype=np.float32)
    if "total_charge_per_second_nC_per_grid" in common_metrics:
        save_payload["charge_per_second_grid_names"] = np.asarray(
            list(common_metrics["total_charge_per_second_nC_per_grid"].keys())
        )
        for grid_name, series in common_metrics["total_charge_per_second_nC_per_grid"].items():
            suffix = sanitize_path_part(grid_name)
            save_payload[f"charge_per_second_total_nC_s_{suffix}"] = np.asarray(series, dtype=np.float32)

    if thermal_grids:
        save_payload["thermal_grid_names"] = np.asarray(list(thermal_grids.keys()))
        save_payload["dT_final_reference_grid"] = np.asarray(hottest_grid_name)
        save_payload["stationary_temperature_C"] = np.asarray(
            thermal_grids[hottest_grid_name]["stationary_temperature_C"],
            dtype=np.float32,
        )
        save_payload["stationary_dT_C"] = np.asarray(
            thermal_grids[hottest_grid_name]["stationary_dT_C"],
            dtype=np.float32,
        )
        if hottest_cem43_grid_name is not None:
            save_payload["cem43_final_reference_grid"] = np.asarray(hottest_cem43_grid_name)
        for grid_name, grid_data in thermal_grids.items():
            suffix = sanitize_path_part(grid_name)
            save_payload[f"dT_final_{suffix}"] = grid_data["dT_final"]
            save_payload[f"stationary_temperature_C_{suffix}"] = np.asarray(
                grid_data["stationary_temperature_C"],
                dtype=np.float32,
            )
            save_payload[f"stationary_dT_C_{suffix}"] = np.asarray(
                grid_data["stationary_dT_C"],
                dtype=np.float32,
            )
            if "cem43_final" in grid_data:
                save_payload[f"cem43_final_{suffix}"] = grid_data["cem43_final"]
            save_payload[f"extent_mm_{suffix}"] = np.asarray(grid_data["extent_mm"], dtype=np.float32)
            save_payload[f"internal_circuit_footprint_pixels_{suffix}"] = np.asarray(
                grid_data["ic_footprint_pixel_count"],
                dtype=np.float32,
            )
            save_payload[f"internal_circuit_power_density_W_m3_{suffix}"] = np.asarray(
                grid_data["ic_power_density_W_m3"],
                dtype=np.float32,
            )
            if "max_dT_per_grid" in thermal_metrics and grid_name in thermal_metrics["max_dT_per_grid"]:
                save_payload[f"max_dT_{suffix}"] = np.asarray(
                    thermal_metrics["max_dT_per_grid"][grid_name],
                    dtype=np.float32,
                )
            if "mean_dT_per_grid" in thermal_metrics and grid_name in thermal_metrics["mean_dT_per_grid"]:
                save_payload[f"mean_dT_{suffix}"] = np.asarray(
                    thermal_metrics["mean_dT_per_grid"][grid_name],
                    dtype=np.float32,
                )

    if thermal_snapshots:
        save_payload["heatmap_frame_indices"] = np.asarray(
            thermal_snapshots.get("frame_indices", []),
            dtype=np.int32,
        )
        save_payload["heatmap_times_s"] = np.asarray(
            thermal_snapshots.get("times_s", []),
            dtype=np.float32,
        )
        save_payload["heatmap_target_fractions"] = np.asarray(
            thermal_snapshots.get("target_fractions", []),
            dtype=np.float32,
        )
        save_payload["heatmap_actual_fractions"] = np.asarray(
            thermal_snapshots.get("actual_fractions", []),
            dtype=np.float32,
        )
        save_payload["heatmap_grid_names"] = np.asarray(
            list(thermal_snapshots.get("grids", {}).keys())
        )
        for grid_name, snapshots in thermal_snapshots.get("grids", {}).items():
            suffix = sanitize_path_part(grid_name)
            save_payload[f"dT_heatmaps_{suffix}"] = np.asarray(snapshots, dtype=np.float32)
        for grid_name, snapshots in thermal_snapshots.get("cem43_grids", {}).items():
            suffix = sanitize_path_part(grid_name)
            save_payload[f"cem43_heatmaps_{suffix}"] = np.asarray(snapshots, dtype=np.float32)

    np.savez(out_dir / "safety_metrics.npz", **save_payload)


# Main simulation pass for one video / preprocessing / raster combination.
# The phosphene simulation is shared, while bioheat branches per IC mode.
@torch.inference_mode()
def run_one_mode(*, params: dict, coords_yaml: Path, video_path: Path,
                 mode_out_dirs: dict[str, Path], preview_out_dirs: dict[str, Path],
                 preprocessing_method: str,
                 stim_scale: float | None,
                 stimulus_scale_base: float,
                 raster_name: str, groups: int, max_frames: int,
                 internal_circuit_power_mw: float, preview_seconds: float,
                 snapshot_interval_s: float, save_every_n_frames: int | None,
                 enable_cem43: bool,
                 device: torch.device,
                 phosphene_mode: str = "safety_centers",
                 thermal_update_interval_frames: int | None = None,
                 appearance_threshold_uA: float | None = None):
    _ = snapshot_interval_s  # Legacy CLI option kept for compatibility.
    if not mode_out_dirs:
        raise ValueError("run_one_mode requires at least one IC mode output directory.")
    if set(mode_out_dirs) != set(preview_out_dirs):
        raise ValueError("mode_out_dirs and preview_out_dirs must have the same IC heat mode keys.")
    phosphene_mode = normalize_phosphene_mode(phosphene_mode)

    for out_dir in mode_out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)
    for out_dir in preview_out_dirs.values():
        out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open preprocessed video: {video_path}")

    # Read FPS from the actual media so raster timing follows the video, not
    # a possibly stale value from the params file.
    video_fps, fps_source = resolve_video_fps(
        cap,
        video_path,
        fallback_fps=float(params["run"]["fps"]),
    )
    raster_label = normalize_raster_label(raster_name)
    raster_pattern_name = normalize_raster_mode(raster_label)
    raster_enabled = raster_pattern_name != "none"
    raster_timing = compute_raster_timing(video_fps, int(groups), raster_enabled)
    applied_raster_rate_hz = float(raster_timing["cycle_rate_hz"])

    # Work on a run-local params copy so one input video does not leak timing
    # or other run-specific state into the next one.
    simulation_params = yaml.safe_load(yaml.safe_dump(params))
    simulation_params = disable_temporal_dynamics(simulation_params)
    simulation_params.setdefault("run", {})
    simulation_params["run"]["fps"] = float(video_fps)
    explicit_appearance_threshold = appearance_threshold_uA is not None
    fixed_firing_threshold_a = resolve_appearance_threshold_a(simulation_params, appearance_threshold_uA)
    if explicit_appearance_threshold:
        simulation_params.setdefault("thresholding", {})["rheobase"] = 0.0
    stimulus_scale_effective = float(
        simulation_params.get("sampling", {}).get("stimulus_scale", stimulus_scale_base)
    )

    mapping = load_tagged_mapping(simulation_params, coords_yaml)
    phosphene_map = mapping["phosphene_map"]
    raster_coordinates = mapping["cortical_coordinates"]

    raster_pattern = raster_pattern_name if raster_enabled else "random"

    sim = GaussianSimulator(
        simulation_params,
        phosphene_map,
        raster_coordinates=raster_coordinates,
        raster_enabled=raster_enabled,
        raster_pattern=raster_pattern,
        raster_num_groups=int(groups),
        raster_rate_hz=applied_raster_rate_hz,
        phosphene_mode=phosphene_mode,
    )
    move_simulator_tensors_to_device(sim, device)
    sim.reset()
    aligned_mapping = align_mapping_to_simulator(mapping, sim)
    elec_xy_mm_surv, inv_map_t = build_surviving_electrode_set(
        aligned_mapping["cortical_coordinates"],
        aligned_mapping["indices"],
        device=device,
    )
    n_elec = elec_xy_mm_surv.shape[0]
    electrode_ids, electrode_xy_mm = build_surviving_electrode_metadata(
        aligned_mapping["cortical_coordinates"],
        aligned_mapping["indices"],
    )
    impedance_phos_t = sim.impedance.state.reshape(-1).to(device=device, dtype=torch.float32)
    impedance_sum_t = torch.zeros(n_elec, dtype=torch.float32, device=device)
    impedance_count_t = torch.zeros(n_elec, dtype=torch.float32, device=device)
    impedance_sum_t.scatter_add_(0, inv_map_t, impedance_phos_t)
    impedance_count_t.scatter_add_(0, inv_map_t, torch.ones_like(impedance_phos_t))
    electrode_impedance_ohm = (
        impedance_sum_t / impedance_count_t.clamp_min(1.0)
    ).detach().cpu().numpy().astype(np.float32)
    bioheat_grid_sets = build_bioheat_grid_sets(
        aligned_mapping["cortical_coordinates"],
        aligned_mapping["grid_ids"],
        aligned_mapping["base_indices"],
        device=device,
    )
    electrode_grid_ids = build_electrode_grid_ids(
        aligned_mapping["indices"],
        aligned_mapping["grid_ids"],
    )
    grid_name_by_id = {
        int(grid_id): grid_info["name"] for grid_id, grid_info in bioheat_grid_sets.items()
    }
    validate_and_report_raster_timing(
        sim,
        video_path=video_path,
        raster_name=raster_label,
        groups=int(groups),
        fps_source=fps_source,
        timing=raster_timing,
    )

    fps = float(video_fps)
    dt = 1.0 / fps
    total_simulation_frames = resolve_simulation_frame_count(cap, max_frames)
    heatmap_snapshot_schedule = build_heatmap_snapshot_schedule(total_simulation_frames, dt)
    heatmap_snapshot_lookup = {
        int(item["frame_idx"]): item for item in heatmap_snapshot_schedule
    }
    target_res = tuple(int(v) for v in simulation_params["run"]["resolution"])
    binarize_input = should_binarize_preprocessed_input(preprocessing_method)
    preview_max_frames = max(0, int(round(float(preview_seconds) * fps)))
    if phosphene_mode != "visual":
        if preview_max_frames > 0:
            print(
                "phosphene preview skipped because phosphene_mode=safety_centers; "
                "use --phosphene-mode visual to render previews."
            )
        preview_max_frames = 0
    progress_started_at = time.perf_counter()

    sampled_frame_indices = []
    sampled_time_s = []
    frame_charge_total_nC = []
    total_charge_per_second_nC = []
    mean_charge_per_phase_nC = []
    mean_charge_density_uc_cm2 = []
    mean_shannon_k = []
    mean_charge_per_second_per_electrode_nC_s = []
    total_window_nC = []
    total_protocol_nC = []
    total_protocol_nC_per_grid = {
        grid_info["name"]: [] for grid_info in bioheat_grid_sets.values()
    }
    total_charge_per_second_nC_per_grid = {
        grid_info["name"]: [] for grid_info in bioheat_grid_sets.values()
    }
    current_amplitude_per_electrode_uA = []
    final_protocol_charge_per_electrode_nC = torch.zeros(n_elec, dtype=torch.float32, device=device)
    charge_per_second_window_entries = []
    charge_per_second_window_time_s = 0.0
    charge_per_second_window_elec_nC = torch.zeros(n_elec, dtype=torch.float32, device=device)
    active_count = []
    stimulated_electrode_count = []
    time_s = []

    pulse_frequency_hz_t = aggregate_metric_tensor(sim._frequency.reshape(-1), inv_map_t, n_elec).to(
        device=device,
        dtype=torch.float32,
    )
    pulse_width_s_t = aggregate_metric_tensor(sim._pulse_width.reshape(-1), inv_map_t, n_elec).to(
        device=device,
        dtype=torch.float32,
    )
    pulse_frequency_hz = pulse_frequency_hz_t.detach().cpu().numpy().astype(np.float32)
    pulse_width_s = pulse_width_s_t.detach().cpu().numpy().astype(np.float32)
    electrode_area_cm2 = float(simulation_params["safety"]["electrode_surface_area_cm2"])
    relative_stim_duration = float(simulation_params.get("default_stim", {}).get("relative_stim_duration", 1.0))
    configured_save_every = (
        int(save_every_n_frames)
        if save_every_n_frames is not None
        else int(simulation_params.get("safety", {}).get("metrics_save_every_n_frames", 5))
    )
    if configured_save_every <= 0:
        raise ValueError("save_every_n_frames must be >= 1.")
    exact_electrical_peaks = {
        "peak_charge_per_phase_nC_exact": torch.tensor(0.0, dtype=torch.float32, device=device),
        "peak_current_amplitude_uA_exact": torch.tensor(0.0, dtype=torch.float32, device=device),
        "peak_charge_density_uc_cm2_exact": torch.tensor(0.0, dtype=torch.float32, device=device),
        "peak_shannon_k_exact": torch.tensor(-torch.inf, dtype=torch.float32, device=device),
        "peak_charge_per_second_per_electrode_nC_s_exact": torch.tensor(0.0, dtype=torch.float32, device=device),
    }
    heat_window_power_sum_phos = None
    heat_window_frame_count = 0
    configured_thermal_update_interval = (
        int(thermal_update_interval_frames)
        if thermal_update_interval_frames is not None
        else HEATMAP_UPDATE_INTERVAL_FRAMES
    )
    if configured_thermal_update_interval <= 0:
        raise ValueError("thermal_update_interval_frames must be >= 1.")
    electrode_grid_ids_t = torch.tensor(electrode_grid_ids, dtype=torch.long, device=device)

    mode_states = {}
    for ic_heat_mode, out_dir in mode_out_dirs.items():
        mode_params = yaml.safe_load(yaml.safe_dump(simulation_params))
        mode_params.setdefault("bioheat", {})
        ic_power_mw = float(internal_circuit_power_mw) if ic_heat_mode == "with" else 0.0
        grid_heat_states = {}
        for grid_id, grid_info in bioheat_grid_sets.items():
            grid_params = yaml.safe_load(yaml.safe_dump(mode_params))
            grid_params.setdefault("bioheat", {})
            grid_params["bioheat"]["internal_circuit_power_mw"] = ic_power_mw
            bio = Bioheat2D(params=grid_params, elec_xy_mm=grid_info["elec_xy_mm"], device=device)
            grid_heat_states[grid_info["name"]] = {
                "bio": bio,
                "mask_t": grid_info["mask_t"],
                "inverse_t": grid_info["inverse_t"],
                "n_elec": grid_info["n_elec"],
                "baseline_temp": float(bio.baseline_T),
                "pixel_area_mm2": float(bio.dx * bio.dy) * 1e6,
                "prev_dT_map": torch.zeros_like(bio.T),
            }
            if enable_cem43:
                grid_heat_states[grid_info["name"]]["cem43_map"] = torch.zeros_like(bio.T)

        mode_states[ic_heat_mode] = {
            "out_dir": out_dir,
            "preview_out_dir": preview_out_dirs[ic_heat_mode],
            "internal_circuit_power_mw": ic_power_mw * max(len(grid_heat_states), 1),
            "grid_heat_states": grid_heat_states,
            "preview_phosphene_writer": None,
            "preview_comparison_writer": None,
            "snapshot_frame_indices": [],
            "snapshot_times_s": [],
            "snapshot_target_fractions": [],
            "snapshot_actual_fractions": [],
            "snapshot_grids": {grid_name: [] for grid_name in grid_heat_states},
            "grid_max_dT": {grid_name: [] for grid_name in grid_heat_states},
            "grid_mean_dT": {grid_name: [] for grid_name in grid_heat_states},
            "max_dT": [],
            "mean_dT": [],
            "area_gt1_mm2": [],
            "area_gt2_mm2": [],
            "area_gt3_mm2": [],
        }
        if enable_cem43:
            mode_states[ic_heat_mode]["snapshot_cem43_grids"] = {
                grid_name: [] for grid_name in grid_heat_states
            }
            mode_states[ic_heat_mode]["max_cem43"] = []
            mode_states[ic_heat_mode]["p99_cem43"] = []

    def update_bioheat_from_window(window_power_sum_phos: torch.Tensor, window_frame_count: int) -> None:
        """Advance bioheat once using the mean delivered power over a frame window."""
        if window_frame_count <= 0:
            return

        window_power_phos = window_power_sum_phos / float(window_frame_count)
        window_dt = dt * float(window_frame_count)
        for state in mode_states.values():
            for grid_name, grid_state in state["grid_heat_states"].items():
                bio = grid_state["bio"]
                fp_grid = window_power_phos[grid_state["mask_t"]]
                p_grid = torch.zeros(
                    grid_state["n_elec"],
                    device=device,
                    dtype=fp_grid.dtype,
                )
                p_grid.scatter_add_(0, grid_state["inverse_t"], fp_grid)
                if enable_cem43:
                    cem43_weight_prev = cem43_weight_from_temperature(bio.T)
                bio.update(p_grid, window_dt)
                dT_map = (bio.T - grid_state["baseline_temp"]).detach()
                if enable_cem43:
                    cem43_weight_cur = cem43_weight_from_temperature(bio.T)
                    grid_state["cem43_map"].add_(
                        0.5 * (cem43_weight_prev + cem43_weight_cur) * (window_dt / 60.0)
                    )
                grid_state["prev_dT_map"] = dT_map

    def record_thermal_metrics_and_snapshots(snapshot_entry: dict[str, object] | None) -> None:
        for state in mode_states.values():
            grid_max_dT = []
            grid_cem43_maps = [] if enable_cem43 else None
            grid_max_cem43 = [] if enable_cem43 else None
            weighted_mean_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
            weighted_mean_count = 0
            area_gt1_total = torch.tensor(0.0, dtype=torch.float32, device=device)
            area_gt2_total = torch.tensor(0.0, dtype=torch.float32, device=device)
            area_gt3_total = torch.tensor(0.0, dtype=torch.float32, device=device)

            for grid_name, grid_state in state["grid_heat_states"].items():
                dT_map = grid_state["prev_dT_map"]
                grid_max_value = dT_map.max()
                grid_mean_value = dT_map.mean()
                state["grid_max_dT"][grid_name].append(grid_max_value)
                state["grid_mean_dT"][grid_name].append(grid_mean_value)

                grid_max_dT.append(grid_max_value)
                if enable_cem43:
                    grid_cem43_max_value = grid_state["cem43_map"].max()
                    grid_cem43_maps.append(grid_state["cem43_map"])
                    grid_max_cem43.append(grid_cem43_max_value)
                weighted_mean_sum += grid_mean_value * int(dT_map.numel())
                weighted_mean_count += int(dT_map.numel())
                area_gt1_total += (dT_map > 1.0).sum(dtype=torch.float32) * grid_state["pixel_area_mm2"]
                area_gt2_total += (dT_map > 2.0).sum(dtype=torch.float32) * grid_state["pixel_area_mm2"]
                area_gt3_total += (dT_map > 3.0).sum(dtype=torch.float32) * grid_state["pixel_area_mm2"]

            if not grid_max_dT:
                raise RuntimeError("No bioheat grids were configured for the current mode.")

            state["max_dT"].append(torch.stack(grid_max_dT).max())
            state["mean_dT"].append(weighted_mean_sum / max(weighted_mean_count, 1))
            state["area_gt1_mm2"].append(area_gt1_total)
            state["area_gt2_mm2"].append(area_gt2_total)
            state["area_gt3_mm2"].append(area_gt3_total)
            if enable_cem43:
                state["max_cem43"].append(torch.stack(grid_max_cem43).max())
                state["p99_cem43"].append(percentile_tensor_from_tensor_list(grid_cem43_maps, 99.0, device))

            if snapshot_entry is not None:
                state["snapshot_frame_indices"].append(int(snapshot_entry["frame_idx"]))
                state["snapshot_times_s"].append(float(snapshot_entry["time_s"]))
                state["snapshot_target_fractions"].append(float(snapshot_entry["target_fraction"]))
                state["snapshot_actual_fractions"].append(float(snapshot_entry["actual_fraction"]))
                for grid_name, grid_state in state["grid_heat_states"].items():
                    state["snapshot_grids"][grid_name].append(
                        grid_state["prev_dT_map"].detach().cpu().numpy().astype(np.float32)
                    )
                    if enable_cem43:
                        state["snapshot_cem43_grids"][grid_name].append(
                            grid_state["cem43_map"].detach().cpu().numpy().astype(np.float32)
                        )

    def remove_latest_thermal_metric_samples() -> None:
        for state in mode_states.values():
            for series in state["grid_max_dT"].values():
                if series:
                    series.pop()
            for series in state["grid_mean_dT"].values():
                if series:
                    series.pop()
            for key in ("max_dT", "mean_dT", "area_gt1_mm2", "area_gt2_mm2", "area_gt3_mm2"):
                if state[key]:
                    state[key].pop()
            if enable_cem43:
                for key in ("max_cem43", "p99_cem43"):
                    if state[key]:
                        state[key].pop()

    frame_idx = 0

    def push_charge_per_second_window(dt_s: float, delta_charge_elec_nC: torch.Tensor) -> torch.Tensor:
        nonlocal charge_per_second_window_time_s, charge_per_second_window_elec_nC
        delta_charge_elec_nC = delta_charge_elec_nC.to(device=device, dtype=torch.float32)
        charge_per_second_window_entries.append([float(dt_s), delta_charge_elec_nC])
        charge_per_second_window_time_s += float(dt_s)
        charge_per_second_window_elec_nC = charge_per_second_window_elec_nC + delta_charge_elec_nC

        while charge_per_second_window_time_s > 1.0 + 1e-12 and charge_per_second_window_entries:
            excess_s = charge_per_second_window_time_s - 1.0
            head_dt_s, head_charge_nC = charge_per_second_window_entries[0]
            if head_dt_s <= excess_s + 1e-12:
                charge_per_second_window_entries.pop(0)
                charge_per_second_window_time_s -= head_dt_s
                charge_per_second_window_elec_nC = charge_per_second_window_elec_nC - head_charge_nC
            else:
                fraction = excess_s / head_dt_s
                trim_charge_nC = head_charge_nC * fraction
                charge_per_second_window_entries[0] = [head_dt_s - excess_s, head_charge_nC - trim_charge_nC]
                charge_per_second_window_time_s -= excess_s
                charge_per_second_window_elec_nC = charge_per_second_window_elec_nC - trim_charge_nC

        return charge_per_second_window_elec_nC.clone()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            if max_frames > 0 and frame_idx > int(max_frames):
                break

            gray = prepare_frame(frame, target_res)
            if binarize_input:
                gray = restore_binary_preprocessed_frame(gray)
            stim_raw = sim.sample_stimulus(gray, rescale=True).reshape(-1).to(device)

            stim = apply_appearance_threshold(stim_raw, fixed_firing_threshold_a)

            # The phosphene simulation is shared across IC modes, so the
            # electrical update runs only once per frame.
            sim.update(stim, dt=dt, temperature_increase=None)

            if preview_max_frames > 0 and frame_idx <= preview_max_frames:
                percept_u8 = render_phosphene_frame_from_state(sim)
                comparison_u8 = build_comparison_frame(
                    gray,
                    gray,
                    percept_u8,
                    input_stage="preprocessed",
                )

                for state in mode_states.values():
                    if state["preview_phosphene_writer"] is None:
                        height, width = percept_u8.shape
                        state["preview_phosphene_writer"] = open_video_writer(
                            state["preview_out_dir"] / "phosphene_preview.mp4",
                            fps,
                            (width, height),
                            is_color=False,
                        )
                    if state["preview_comparison_writer"] is None:
                        height, width = comparison_u8.shape
                        state["preview_comparison_writer"] = open_video_writer(
                            state["preview_out_dir"] / "phosphene_preview_comparison.mp4",
                            fps,
                            (width, height),
                            is_color=False,
                        )

                    state["preview_phosphene_writer"].write(percept_u8)
                    state["preview_comparison_writer"].write(comparison_u8)

            # SafetyTracker stores per-phosphene values; aggregate them back to
            # physical electrodes so electrical and thermal metrics align.
            current_amplitude_elec = aggregate_metric_tensor((stim * 1e6).reshape(-1), inv_map_t, n_elec).to(
                dtype=torch.float32
            )
            current_amplitude_per_electrode_uA.append(current_amplitude_elec.clone())
            q_phase_elec = aggregate_metric_tensor(
                sim.safety_tracker.last_charge_per_phase_nC,
                inv_map_t,
                n_elec,
            ).to(dtype=torch.float32)
            q_window_elec = aggregate_metric_tensor(
                sim.safety_tracker.window_charge_per_electrode_nC,
                inv_map_t,
                n_elec,
            ).to(dtype=torch.float32)
            q_protocol_elec = aggregate_metric_tensor(
                sim.safety_tracker.protocol_charge_per_electrode_nC,
                inv_map_t,
                n_elec,
            ).to(dtype=torch.float32)
            final_protocol_charge_per_electrode_nC = q_protocol_elec

            charge_density_elec = q_phase_elec / 1e3 / electrode_area_cm2
            shannon_elec = torch.full_like(charge_density_elec, -torch.inf)
            valid = (q_phase_elec > 0) & (charge_density_elec > 0)
            shannon_elec[valid] = torch.log10(q_phase_elec[valid] / 1e3) + torch.log10(charge_density_elec[valid])

            # One-second accumulated charge from the delivered per-frame amplitudes.
            # Biphasic stimulation has two phases per pulse; pulse width is per phase.
            frame_charge_elec_nC = (
                2.0
                * current_amplitude_elec
                * pulse_width_s_t
                * pulse_frequency_hz_t
                * float(dt)
                * relative_stim_duration
                * 1e3
            )
            charge_per_second_elec = push_charge_per_second_window(float(dt), frame_charge_elec_nC)
            frame_charge_total_nC.append(frame_charge_elec_nC.sum())

            # Accumulate delivered heat input, then step Bioheat2D once per
            # 10-frame window with the corresponding 10/fps duration.
            fp_phos = sim.frame_power.reshape(-1).to(device)
            if heat_window_power_sum_phos is None:
                heat_window_power_sum_phos = torch.zeros_like(fp_phos)
            heat_window_power_sum_phos.add_(fp_phos)
            heat_window_frame_count += 1
            if (
                heat_window_frame_count >= configured_thermal_update_interval or
                (total_simulation_frames > 0 and frame_idx >= total_simulation_frames)
            ):
                update_bioheat_from_window(
                    heat_window_power_sum_phos,
                    heat_window_frame_count,
                )
                heat_window_power_sum_phos.zero_()
                heat_window_frame_count = 0

            current_time_s = frame_idx * dt
            snapshot_entry = heatmap_snapshot_lookup.get(frame_idx)
            record_thermal_metrics_and_snapshots(snapshot_entry)

            exact_electrical_peaks["peak_charge_per_phase_nC_exact"] = torch.maximum(
                exact_electrical_peaks["peak_charge_per_phase_nC_exact"],
                q_phase_elec.max() if q_phase_elec.numel() else torch.tensor(0.0, dtype=torch.float32, device=device),
            )
            exact_electrical_peaks["peak_current_amplitude_uA_exact"] = torch.maximum(
                exact_electrical_peaks["peak_current_amplitude_uA_exact"],
                current_amplitude_elec.max()
                if current_amplitude_elec.numel()
                else torch.tensor(0.0, dtype=torch.float32, device=device),
            )
            exact_electrical_peaks["peak_charge_density_uc_cm2_exact"] = torch.maximum(
                exact_electrical_peaks["peak_charge_density_uc_cm2_exact"],
                charge_density_elec.max()
                if charge_density_elec.numel()
                else torch.tensor(0.0, dtype=torch.float32, device=device),
            )
            exact_electrical_peaks["peak_shannon_k_exact"] = torch.maximum(
                exact_electrical_peaks["peak_shannon_k_exact"],
                shannon_elec.max()
                if shannon_elec.numel()
                else torch.tensor(-torch.inf, dtype=torch.float32, device=device),
            )
            exact_electrical_peaks["peak_charge_per_second_per_electrode_nC_s_exact"] = torch.maximum(
                exact_electrical_peaks["peak_charge_per_second_per_electrode_nC_s_exact"],
                charge_per_second_elec.max()
                if charge_per_second_elec.numel()
                else torch.tensor(0.0, dtype=torch.float32, device=device),
            )

            if (frame_idx - 1) % configured_save_every == 0:
                sampled_frame_indices.append(int(frame_idx))
                sampled_time_s.append(float(current_time_s))

            total_charge_per_second_nC.append(charge_per_second_elec.sum())
            active_q = q_phase_elec > 0.0
            active_count_t = active_q.sum()
            active_denominator = active_count_t.clamp_min(1).to(dtype=torch.float32)
            finite_shannon = torch.isfinite(shannon_elec) & active_q
            finite_shannon_count = finite_shannon.sum()
            finite_shannon_denominator = finite_shannon_count.clamp_min(1).to(dtype=torch.float32)
            mean_charge_per_phase_nC.append((q_phase_elec * active_q).sum() / active_denominator)
            mean_charge_density_uc_cm2.append((charge_density_elec * active_q).sum() / active_denominator)
            mean_shannon_k.append(
                torch.where(
                    finite_shannon_count > 0,
                    torch.where(finite_shannon, shannon_elec, torch.zeros_like(shannon_elec)).sum()
                    / finite_shannon_denominator,
                    torch.tensor(-torch.inf, dtype=torch.float32, device=device),
                )
            )
            mean_charge_per_second_per_electrode_nC_s.append(
                (charge_per_second_elec * active_q).sum() / active_denominator
            )
            total_window_nC.append(q_window_elec.sum())
            total_protocol_nC.append(q_protocol_elec.sum())
            for grid_id, grid_name in grid_name_by_id.items():
                grid_mask = electrode_grid_ids_t == int(grid_id)
                total_protocol_nC_per_grid[grid_name].append(q_protocol_elec[grid_mask].sum())
                total_charge_per_second_nC_per_grid[grid_name].append(charge_per_second_elec[grid_mask].sum())
            # Count physical electrodes carrying non-zero charge in the current
            # frame after the raster mask has already been applied upstream.
            active_count.append(active_count_t.to(dtype=torch.int32))

            if explicit_appearance_threshold:
                stimulated_electrode_count.append(active_count_t.to(dtype=torch.int32))
            else:
                # Legacy path: count physical electrodes whose simulated
                # activation exceeds the model's activation threshold.
                supra_phos = torch.greater(sim.activation.get(), sim.threshold.get()).reshape(-1)
                if sim.raster_enabled:
                    supra_phos = supra_phos & sim.get_current_raster_mask().reshape(-1).bool()
                supra_elec = torch.zeros(n_elec, device=device, dtype=torch.int32)
                supra_elec.scatter_add_(0, inv_map_t, supra_phos.to(torch.int32))
                stimulated_electrode_count.append((supra_elec > 0).sum().to(dtype=torch.int32))
            time_s.append(current_time_s)

            if frame_idx == 1 or frame_idx % 10 == 0:
                elapsed_s = max(time.perf_counter() - progress_started_at, 1e-9)
                frame_rate = float(frame_idx) / elapsed_s
                if total_simulation_frames > 0:
                    progress_pct = 100.0 * float(frame_idx) / float(total_simulation_frames)
                    print(
                        f"{raster_label}: frame {frame_idx}/{total_simulation_frames} "
                        f"({progress_pct:5.1f}%) | elapsed {elapsed_s:7.1f}s | "
                        f"{frame_rate:5.2f} frames/s",
                        end="\r",
                        flush=True,
                    )
                else:
                    print(
                        f"{raster_label}: frame {frame_idx} | elapsed {elapsed_s:7.1f}s | "
                        f"{frame_rate:5.2f} frames/s",
                        end="\r",
                        flush=True,
                    )
    finally:
        cap.release()
        for state in mode_states.values():
            if state["preview_phosphene_writer"] is not None:
                state["preview_phosphene_writer"].release()
            if state["preview_comparison_writer"] is not None:
                state["preview_comparison_writer"].release()

    if frame_idx > 0 and heat_window_frame_count > 0 and heat_window_power_sum_phos is not None:
        update_bioheat_from_window(heat_window_power_sum_phos, heat_window_frame_count)
        remove_latest_thermal_metric_samples()
        record_thermal_metrics_and_snapshots(None)

    if frame_idx > 0:
        print()

    if frame_idx == 0:
        raise RuntimeError("No frames were read from the input video.")

    common_metrics = {
        "time_s": np.asarray(time_s, dtype=np.float32),
        "electrode_time_s": np.asarray(sampled_time_s, dtype=np.float32),
        "electrode_frame_indices": np.asarray(sampled_frame_indices, dtype=np.int32),
        "electrode_metrics_save_every_n_frames": np.asarray(configured_save_every, dtype=np.int32),
        "frame_charge_total_nC": tensor_list_to_numpy(frame_charge_total_nC),
        "total_charge_per_second_nC": tensor_list_to_numpy(total_charge_per_second_nC),
        "mean_charge_per_phase_nC": tensor_list_to_numpy(mean_charge_per_phase_nC),
        "mean_charge_density_uc_cm2": tensor_list_to_numpy(mean_charge_density_uc_cm2),
        "mean_shannon_k": tensor_list_to_numpy(mean_shannon_k),
        "mean_charge_per_second_per_electrode_nC_s": tensor_list_to_numpy(
            mean_charge_per_second_per_electrode_nC_s,
        ),
        "total_window_nC": tensor_list_to_numpy(total_window_nC),
        "total_protocol_nC": tensor_list_to_numpy(total_protocol_nC),
        "final_protocol_charge_per_electrode_nC": scalar_tensor_to_numpy(
            final_protocol_charge_per_electrode_nC,
        ),
        "total_protocol_nC_per_grid": {
            grid_name: tensor_list_to_numpy(series)
            for grid_name, series in total_protocol_nC_per_grid.items()
        },
        "total_charge_per_second_nC_per_grid": {
            grid_name: tensor_list_to_numpy(series)
            for grid_name, series in total_charge_per_second_nC_per_grid.items()
        },
        "electrode_ids": electrode_ids.astype(np.int64),
        "electrode_xy_mm": electrode_xy_mm.astype(np.float32),
        "electrode_impedance_ohm": electrode_impedance_ohm,
        "electrode_grid_ids": np.asarray(electrode_grid_ids, dtype=np.int32),
        "electrode_grid_names": np.asarray([grid_name_by_id[int(grid_id)] for grid_id in electrode_grid_ids]),
        "active_count": tensor_list_to_numpy(active_count, dtype=np.int32),
        "stimulated_electrode_count": tensor_list_to_numpy(stimulated_electrode_count, dtype=np.int32),
        "pulse_width_s": np.asarray(pulse_width_s, dtype=np.float32),
        "relative_stim_duration": np.asarray(relative_stim_duration, dtype=np.float32),
        "charge_window_s": np.asarray(float(sim.safety_tracker.charge_window_s), dtype=np.float32),
        "electrode_surface_area_cm2": np.asarray(electrode_area_cm2, dtype=np.float32),
        "fixed_firing_threshold_uA": np.asarray(fixed_firing_threshold_a * 1e6, dtype=np.float32),
        "appearance_threshold_uA": np.asarray(fixed_firing_threshold_a * 1e6, dtype=np.float32),
        "appearance_threshold_explicit": np.asarray(explicit_appearance_threshold, dtype=np.bool_),
        "input_binarized_for_safety": np.asarray(binarize_input, dtype=np.bool_),
        "temporal_dynamics_disabled": np.asarray(
            bool(simulation_params.get("safety", {}).get("temporal_dynamics_disabled", False)),
            dtype=np.bool_,
        ),
        "trace_increase_rate": np.asarray(
            float(simulation_params.get("temporal_dynamics", {}).get("trace_increase_rate", np.nan)),
            dtype=np.float32,
        ),
        "activation_threshold_sd": np.asarray(
            float(simulation_params.get("thresholding", {}).get("activation_threshold_sd", np.nan)),
            dtype=np.float32,
        ),
        "current_amplitude_per_electrode_uA": np.asarray(
            tensor_rows_to_numpy(current_amplitude_per_electrode_uA),
            dtype=np.float32,
        ),
        "peak_charge_per_phase_nC_exact": np.asarray(
            scalar_tensor_to_numpy(exact_electrical_peaks["peak_charge_per_phase_nC_exact"]),
            dtype=np.float32,
        ),
        "peak_current_amplitude_uA_exact": np.asarray(
            scalar_tensor_to_numpy(exact_electrical_peaks["peak_current_amplitude_uA_exact"]),
            dtype=np.float32,
        ),
        "peak_charge_density_uc_cm2_exact": np.asarray(
            scalar_tensor_to_numpy(exact_electrical_peaks["peak_charge_density_uc_cm2_exact"]),
            dtype=np.float32,
        ),
        "peak_shannon_k_exact": np.asarray(
            scalar_tensor_to_numpy(exact_electrical_peaks["peak_shannon_k_exact"]),
            dtype=np.float32,
        ),
        "peak_charge_per_second_per_electrode_nC_s_exact": np.asarray(
            scalar_tensor_to_numpy(exact_electrical_peaks["peak_charge_per_second_per_electrode_nC_s_exact"]),
            dtype=np.float32,
        ),
    }

    for ic_heat_mode, state in mode_states.items():
        if any(grid_state["prev_dT_map"] is None for grid_state in state["grid_heat_states"].values()):
            raise RuntimeError(f"No temperature results were produced for IC mode '{ic_heat_mode}'.")

        thermal_grids = {}
        for grid_name, grid_state in state["grid_heat_states"].items():
            dT_final = grid_state["prev_dT_map"].detach().cpu().numpy().astype(np.float32)
            stationary_dT_C = float(np.max(dT_final)) if dT_final.size else float("nan")
            thermal_grids[grid_name] = {
                "dT_final": dT_final,
                "extent_mm": grid_state["bio"].extent_mm,
                "ic_footprint_pixel_count": grid_state["bio"].ic_footprint_pixel_count,
                "ic_power_density_W_m3": grid_state["bio"].ic_power_density_W_m3,
                "stationary_dT_C": stationary_dT_C,
                "stationary_temperature_C": float(grid_state["baseline_temp"]) + stationary_dT_C,
            }
            if enable_cem43:
                thermal_grids[grid_name]["cem43_final"] = (
                    grid_state["cem43_map"].detach().cpu().numpy().astype(np.float32)
                )

        reference_grid_name = max(
            thermal_grids,
            key=lambda name: float(np.max(thermal_grids[name]["dT_final"])),
        )
        thermal_snapshots = {
            "frame_indices": np.asarray(state["snapshot_frame_indices"], dtype=np.int32),
            "times_s": np.asarray(state["snapshot_times_s"], dtype=np.float32),
            "target_fractions": np.asarray(state["snapshot_target_fractions"], dtype=np.float32),
            "actual_fractions": np.asarray(state["snapshot_actual_fractions"], dtype=np.float32),
            "grids": {
                grid_name: np.asarray(snapshot_stack, dtype=np.float32)
                for grid_name, snapshot_stack in state["snapshot_grids"].items()
                if snapshot_stack
            },
        }
        if enable_cem43:
            cem43_reference_grid_name = max(
                thermal_grids,
                key=lambda name: float(np.max(thermal_grids[name]["cem43_final"])),
            )
            thermal_snapshots["cem43_grids"] = {
                grid_name: np.asarray(snapshot_stack, dtype=np.float32)
                for grid_name, snapshot_stack in state["snapshot_cem43_grids"].items()
                if snapshot_stack
            }
        thermal_metrics = {
            "max_dT": tensor_list_to_numpy(state["max_dT"]),
            "mean_dT": tensor_list_to_numpy(state["mean_dT"]),
            "area_gt1_mm2": tensor_list_to_numpy(state["area_gt1_mm2"]),
            "area_gt2_mm2": tensor_list_to_numpy(state["area_gt2_mm2"]),
            "area_gt3_mm2": tensor_list_to_numpy(state["area_gt3_mm2"]),
            "dT_final": thermal_grids[reference_grid_name]["dT_final"],
            "max_dT_per_grid": {
                grid_name: tensor_list_to_numpy(series)
                for grid_name, series in state["grid_max_dT"].items()
            },
            "mean_dT_per_grid": {
                grid_name: tensor_list_to_numpy(series)
                for grid_name, series in state["grid_mean_dT"].items()
            },
        }
        if enable_cem43:
            thermal_metrics["max_cem43"] = tensor_list_to_numpy(state["max_cem43"])
            thermal_metrics["p99_cem43"] = tensor_list_to_numpy(state["p99_cem43"])
            thermal_metrics["cem43_final"] = thermal_grids[cem43_reference_grid_name]["cem43_final"]

        save_mode_outputs(
            out_dir=state["out_dir"],
            video_fps=video_fps,
            raster_rate_hz=applied_raster_rate_hz,
            raster_timing=raster_timing,
            pulse_frequency_hz=pulse_frequency_hz,
            common_metrics=common_metrics,
            thermal_metrics=thermal_metrics,
            thermal_grids=thermal_grids,
            thermal_snapshots=thermal_snapshots,
            internal_circuit_power_mw=state["internal_circuit_power_mw"],
        )
        summary_text = build_run_summary_text(
            video_path=video_path,
            preprocessing_method=preprocessing_method,
            ic_heat_mode=ic_heat_mode,
            raster_name=raster_label,
            coords_yaml=coords_yaml,
            stim_scale=stim_scale,
            stimulus_scale_base=stimulus_scale_base,
            stimulus_scale_effective=stimulus_scale_effective,
            total_frames=frame_idx,
            video_fps=video_fps,
            raster_timing=raster_timing,
            groups=groups,
            internal_circuit_power_mw=state["internal_circuit_power_mw"],
            pulse_frequency_hz=pulse_frequency_hz,
            common_metrics=common_metrics,
            thermal_metrics=thermal_metrics,
            thermal_grids=thermal_grids,
            thermal_snapshots=thermal_snapshots,
        )
        write_run_summary(state["out_dir"], summary_text)

    return


@torch.inference_mode()
def main():
    # CLI entry point: resolve inputs, configure the runtime, then iterate over
    # every requested preprocessing / raster combination, branching IC heat
    # only at the bioheat and output stages.
    ap = argparse.ArgumentParser(description="Analyze stimulation safety from preprocessed videos using the full array across preprocessing, rastering, and internal-circuit heat settings.")
    ap.add_argument("--params", type=str, default="config/params.yaml")
    ap.add_argument("--safety_yaml", type=str, default="config/safety.yaml")
    ap.add_argument("--video", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="results/safety/safety_analysis")
    ap.add_argument("--visuals_dir", type=str, default="results/safety/visuals")
    ap.add_argument("--max_frames", type=int, default=0, help="0 = full video")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument(
        "--stim-scale",
        type=float,
        default=None,
        help=(
            "Scale factor applied to sampling.stimulus_scale from params.yaml. "
            "Example: --stim-scale 0.8 keeps 80%% of the original value."
        ),
    )
    ap.add_argument(
        "--raster-modes",
        nargs="+",
        default=list(VALID_RASTER_MODES),
        choices=list(VALID_RASTER_MODES),
        help="Raster modes to run. Example: --raster-modes none",
    )
    ap.add_argument(
        "--preprocessing-methods",
        nargs="+",
        default=None,
        choices=list(VALID_PREPROCESS_METHODS),
        help=(
            "Preprocessing variants to run for the requested video. "
            "The script looks for either sibling files named like <video_stem>_canny.mp4 "
            "or nested exports like videos/preprocessed/<video>/<method>/preprocessed.mp4."
        ),
    )
    ap.add_argument(
        "--ic-heat-modes",
        nargs="+",
        default=list(VALID_IC_HEAT_MODES),
        choices=list(VALID_IC_HEAT_MODES),
        help="Whether to include the internal-circuit heat source. Example: --ic-heat-modes with without",
    )
    ap.add_argument(
        "--internal_circuit_power_mw",
        type=float,
        default=13.0,
        help="Constant internal-circuit heat source distributed across the full electrode grid.",
    )
    ap.add_argument(
        "--appearance-threshold-uA",
        "--appearance_threshold_uA",
        dest="appearance_threshold_uA",
        type=float,
        default=None,
        help=(
            "Optional current threshold in microamps used to gate delivered stimulation. "
            "When omitted, the legacy thresholding.rheobase value is used."
        ),
    )
    ap.add_argument(
        "--preview_seconds",
        type=float,
        default=15.0,
        help="Duration of the phosphene preview clip saved for each raster mode.",
    )
    ap.add_argument(
        "--snapshot_interval_s",
        type=float,
        default=5.0,
        help=(
            "Legacy option kept for CLI compatibility. The pipeline now saves 20 heatmap "
            "snapshots at 5%% intervals of the simulated duration."
        ),
    )
    ap.add_argument(
        "--save_every_n_frames",
        type=int,
        default=None,
        help=(
            "Persist heavy electrode-wise metrics every N frames while still simulating every frame. "
            "Defaults to params['safety']['metrics_save_every_n_frames'] or 5."
        ),
    )
    ap.add_argument(
        "--phosphene-mode",
        choices=list(VALID_PHOSPHENE_MODES),
        default="safety_centers",
        help=(
            "safety_centers skips full phosphene maps and samples only phosphene center pixels; "
            "visual builds full phosphene maps for previews and percept rendering."
        ),
    )
    ap.add_argument(
        "--sim-resolution",
        type=int,
        default=None,
        help="Override simulator width/height in pixels for memory-heavy grids, e.g. 128.",
    )
    ap.add_argument(
        "--thermal-update-interval-frames",
        type=int,
        default=None,
        help=(
            "Update the bioheat model every N video frames. Larger values reduce "
            "thermal compute and temporary memory pressure while still simulating "
            "electrical safety every frame."
        ),
    )
    ap.add_argument(
        "--enable-cem43",
        action="store_true",
        help="Enable CEM43 thermal dose maps and metrics. Disabled by default to keep analysis runs lighter.",
    )
    args = ap.parse_args()

    if int(args.groups) <= 0:
        raise ValueError("--groups must be > 0.")
    
    #Define paths
    params_path = (PROJECT_ROOT / args.params).resolve() if not Path(args.params).is_absolute() else Path(args.params)
    safety_path = (PROJECT_ROOT / args.safety_yaml).resolve() if not Path(args.safety_yaml).is_absolute() else Path(args.safety_yaml)
    coords_yaml = (PROJECT_ROOT / "config" / "coords_800um.yaml").resolve()
    out_root = (PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    visuals_root = (PROJECT_ROOT / args.visuals_dir).resolve() if not Path(args.visuals_dir).is_absolute() else Path(args.visuals_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    visuals_root.mkdir(parents=True, exist_ok=True)
    preprocessing_methods = normalize_preprocessing_methods(args.preprocessing_methods)
    video_runs = resolve_preprocessed_video_runs(args.video, preprocessing_methods)

    #Load parameters 
    params = load_yaml(params_path)
    if args.sim_resolution is not None:
        sim_resolution = int(args.sim_resolution)
        if sim_resolution <= 0:
            raise ValueError("--sim-resolution must be > 0.")
        params.setdefault("run", {})["resolution"] = [sim_resolution, sim_resolution]
    base_stimulus_scale, effective_stimulus_scale = apply_stim_scale_override(
        params,
        args.stim_scale,
    )
    params.setdefault("safety", {})
    params["safety"]["guidelines_path"] = str(safety_path)
    params["safety"]["charge_window_s"] = 5.0
    params["safety"]["metrics_save_every_n_frames"] = int(
        args.save_every_n_frames
        if args.save_every_n_frames is not None
        else params["safety"].get("metrics_save_every_n_frames", 5)
    )
    if int(params["safety"]["metrics_save_every_n_frames"]) <= 0:
        raise ValueError("--save_every_n_frames must be >= 1.")

    if args.stim_scale is not None:
        print(
            "stimulus scale override applied: "
            f"base={base_stimulus_scale:.6f} | "
            f"factor={float(args.stim_scale):.6f} | "
            f"effective={effective_stimulus_scale:.6f}"
        )

    # Respect the project GPU setting when available, but fall back to CPU cleanly.
    gpu_id = params["run"].get("gpu", None)
    if gpu_id is None or not torch.cuda.is_available():
        device = torch.device("cpu")
        params["run"]["gpu"] = None
    else:
        torch.cuda.set_device(int(gpu_id))
        device = torch.device(f"cuda:{int(gpu_id)}")

    raster_modes = normalize_raster_modes(args.raster_modes)
    ic_heat_modes = normalize_ic_heat_modes(args.ic_heat_modes)
    phosphene_mode = normalize_phosphene_mode(args.phosphene_mode)
    total_runs = len(video_runs) * len(raster_modes)
    run_index = 0

    
    for video_run in video_runs:
        video_path = video_run["video_path"]
        preprocessing_method = str(video_run["preprocessing_method"])
        base_stem = str(video_run["base_stem"])
        video_out_root = build_video_output_root(out_root, video_path, stem_override=base_stem)
        visuals_video_out_root = build_video_output_root(visuals_root, video_path, stem_override=base_stem)
        method_out_root = video_out_root / sanitize_path_part(preprocessing_method)
        visuals_method_out_root = visuals_video_out_root / sanitize_path_part(preprocessing_method)
        method_out_root.mkdir(parents=True, exist_ok=True)
        visuals_method_out_root.mkdir(parents=True, exist_ok=True)

        for raster_name in raster_modes:
            run_index += 1
            mode_out_dirs = {}
            preview_out_dirs = {}
            for ic_heat_mode in ic_heat_modes:
                out_dir = method_out_root / sanitize_path_part(ic_heat_mode) / sanitize_path_part(raster_name)
                out_dir.mkdir(parents=True, exist_ok=True)
                mode_out_dirs[ic_heat_mode] = out_dir
                write_run_manifest(
                    out_dir,
                    video_path=video_path,
                    params_path=params_path,
                    safety_path=safety_path,
                    coords_yaml=coords_yaml,
                    preprocessing_method=preprocessing_method,
                    ic_heat_mode=ic_heat_mode,
                    raster_name=raster_name,
                    appearance_threshold_uA=args.appearance_threshold_uA,
                    internal_circuit_power_mw=float(args.internal_circuit_power_mw) if ic_heat_mode == "with" else 0.0,
                    stim_scale=args.stim_scale,
                    stimulus_scale_base=base_stimulus_scale,
                    stimulus_scale_effective=effective_stimulus_scale,
                    phosphene_mode=phosphene_mode,
                    enable_cem43=bool(args.enable_cem43),
                )
                preview_out_dir = visuals_method_out_root / sanitize_path_part(ic_heat_mode) / sanitize_path_part(raster_name)
                preview_out_dir.mkdir(parents=True, exist_ok=True)
                preview_out_dirs[ic_heat_mode] = preview_out_dir

            # Print a compact run header so terminal logs stay searchable.
            print(
                f"[{run_index}/{total_runs}] Processing {video_path.name} | "
                f"preprocess={preprocessing_method} | raster={raster_name} | "
                f"ic_heat={', '.join(ic_heat_modes)} | "
                f"cem43={'on' if args.enable_cem43 else 'off'}"
            )
            run_one_mode(
                params=params,
                coords_yaml=coords_yaml,
                video_path=video_path,
                mode_out_dirs=mode_out_dirs,
                preview_out_dirs=preview_out_dirs,
                preprocessing_method=preprocessing_method,
                stim_scale=args.stim_scale,
                stimulus_scale_base=base_stimulus_scale,
                raster_name=raster_name,
                groups=int(args.groups),
                max_frames=int(args.max_frames),
                internal_circuit_power_mw=float(args.internal_circuit_power_mw),
                preview_seconds=float(args.preview_seconds),
                snapshot_interval_s=float(args.snapshot_interval_s),
                save_every_n_frames=int(params["safety"]["metrics_save_every_n_frames"]),
                enable_cem43=bool(args.enable_cem43),
                device=device,
                phosphene_mode=phosphene_mode,
                thermal_update_interval_frames=args.thermal_update_interval_frames,
                appearance_threshold_uA=args.appearance_threshold_uA,
            )
            for out_dir in mode_out_dirs.values():
                print(f"saved safety analysis to: {out_dir}")


if __name__ == "__main__":
    main()
