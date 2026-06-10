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
import shutil
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

from dynaphos.simulation import cortex as cortex_models
from dynaphos.simulation import utils
from dynaphos.safety.bioheat import Bioheat2D
from dynaphos.safety.impedance import Impedance, compute_frame_power
from dynaphos.safety.tracking import SafetyTracker
from dynaphos.simulation.simulator import GaussianSimulator, apply_appearance_threshold, compute_raster_timing
from dynaphos.simulation.utils import Map
from dynaphos.simulation.mapping import load_mapping as load_tagged_mapping
from dynaphos.media.rendering import (
    build_comparison_frame,
    open_video_writer,
    prepare_stimulus_frame,
    render_phosphene_frame_from_state,
)
from dynaphos.strategies.base import StrategyContext, StrategyInput
from dynaphos.experiment.metrics import write_metrics


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
HEATMAP_SNAPSHOT_PERCENT_STEP = 5
HEATMAP_SNAPSHOT_COUNT = 100 // HEATMAP_SNAPSHOT_PERCENT_STEP
HEATMAP_UPDATE_INTERVAL_FRAMES = 10
HEATMAP_SNAPSHOT_FRACTIONS = (
    np.arange(
        HEATMAP_SNAPSHOT_PERCENT_STEP,
        100 + HEATMAP_SNAPSHOT_PERCENT_STEP,
        HEATMAP_SNAPSHOT_PERCENT_STEP,
        dtype=np.float64,
    )
    / 100.0
)
_PROGRESS_LINE_LEN = 0
_PROGRESS_OUTPUT_ENABLED = True


def write_progress_line(message: str) -> None:
    """Rewrite one terminal line for high-frequency progress updates."""
    global _PROGRESS_LINE_LEN, _PROGRESS_OUTPUT_ENABLED

    if not _PROGRESS_OUTPUT_ENABLED:
        return
    isatty = getattr(sys.stderr, "isatty", None)
    if callable(isatty) and not isatty():
        return

    columns = max(shutil.get_terminal_size(fallback=(120, 20)).columns, 20)
    line = str(message)[: columns - 1]
    padding = " " * max(_PROGRESS_LINE_LEN - len(line), 0)
    try:
        sys.stderr.write(f"\r{line}{padding}")
        sys.stderr.flush()
    except OSError:
        _PROGRESS_OUTPUT_ENABLED = False
        _PROGRESS_LINE_LEN = 0
        return
    _PROGRESS_LINE_LEN = len(line)


def finish_progress_line() -> None:
    global _PROGRESS_LINE_LEN, _PROGRESS_OUTPUT_ENABLED

    if _PROGRESS_LINE_LEN <= 0 or not _PROGRESS_OUTPUT_ENABLED:
        return
    try:
        sys.stderr.write("\n")
        sys.stderr.flush()
    except OSError:
        _PROGRESS_OUTPUT_ENABLED = False
    finally:
        _PROGRESS_LINE_LEN = 0


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


def activation_threshold_from_current_threshold(params: dict, threshold_a: float) -> float:
    """Convert a current threshold into the simulator activation-threshold units."""
    threshold_a = float(threshold_a)
    if threshold_a < 0.0:
        raise ValueError(f"threshold_a must be >= 0, got {threshold_a}.")

    default_stim = params.get("default_stim", {}) or {}
    pulse_width_s = float(default_stim.get("pw_default", 0.0))
    frequency_hz = float(default_stim.get("freq_default", 0.0))
    relative_stim_duration = float(default_stim.get("relative_stim_duration", 1.0))
    if pulse_width_s < 0.0:
        raise ValueError(f"default_stim.pw_default must be >= 0, got {pulse_width_s}.")
    if frequency_hz < 0.0:
        raise ValueError(f"default_stim.freq_default must be >= 0, got {frequency_hz}.")
    if relative_stim_duration < 0.0:
        raise ValueError(
            "default_stim.relative_stim_duration must be >= 0, "
            f"got {relative_stim_duration}."
        )

    activation_input = threshold_a * pulse_width_s * frequency_hz * relative_stim_duration
    temporal = params.get("temporal_dynamics", {}) or {}
    decay_per_second = float(temporal.get("activation_decay_per_second", 1.0))
    if decay_per_second <= 0.0:
        raise ValueError(
            "temporal_dynamics.activation_decay_per_second must be > 0, "
            f"got {decay_per_second}."
        )
    decay_rate = -float(np.log(decay_per_second))
    if decay_rate <= 0.0:
        return activation_input
    return activation_input / decay_rate


def apply_adaptive_activation_threshold(params: dict, current_threshold_a: float) -> float:
    activation_threshold = activation_threshold_from_current_threshold(params, current_threshold_a)
    thresholding = params.setdefault("thresholding", {})
    thresholding["activation_threshold"] = float(activation_threshold)
    thresholding["activation_threshold_sd"] = 0.0
    return float(activation_threshold)




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


def thermal_projection(value: torch.Tensor) -> torch.Tensor:
    """Return the 2D map used for summary heatmaps from a 2D or 3D solver state."""
    if value.ndim == 3:
        return value.max(dim=0).values
    return value


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
    video_paths = []
    for path in sorted(video_dir.rglob("*")):
        if (
            not path.is_file()
            or path.suffix.lower() not in VIDEO_EXTENSIONS
        ):
            continue
        _base_stem, detected_method = describe_preprocessed_video_path(path)
        if detected_method in VALID_PREPROCESS_METHODS:
            video_paths.append(path.resolve())
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
    # Search close-by side-by-side variants first, then fall back to the standard repo folder.
    for candidate_root in [video_path.parent, (PROJECT_ROOT / "videos").resolve()]:
        if candidate_root is not None and candidate_root not in candidate_roots:
            candidate_roots.append(candidate_root)

    runs = []
    missing = []
    preferred_suffix = video_path.suffix.lower()

    for method in requested_methods:
        found_path = None
        for candidate_root in candidate_roots:
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


def write_execution_manifest(out_dir: Path, *, video_path: Path, params_path: Path,
                       safety_path: Path, coords_yaml: Path,
                       preprocessing_method: str, ic_heat_mode: str,
                       raster_name: str,
                       appearance_threshold_uA: float | None,
                       internal_circuit_power_mw: float,
                       stim_scale: float | None,
                       stimulus_scale_base: float,
                       stimulus_scale_effective: float,
                       phosphene_mode: str,
                       cooldown_seconds: float,
                       cooldown_baseline_tolerance_C: float,
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
        "cooldown_seconds": float(cooldown_seconds),
        "cooldown_baseline_tolerance_C": float(cooldown_baseline_tolerance_C),
        "enable_cem43": bool(enable_cem43),
        "params": str(params_path),
        "safety_yaml": str(safety_path),
        "coords_yaml": str(coords_yaml),
    }
    with open(out_dir / "manifest.yaml", "w", encoding="utf-8") as f:
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


def resolve_thermal_update_interval_frames(
    params: dict,
    *,
    fps: float,
    override_frames: int | None = None,
) -> int:
    """Resolve bioheat cadence to video frames.

    Thermal integration defaults to one video frame so each frame's electrode
    load power is injected at the electrode locations.
    """
    if override_frames is not None:
        frames = int(override_frames)
        if frames <= 0:
            raise ValueError("thermal_update_interval_frames must be >= 1.")
        return frames

    bioheat_params = params.get("bioheat", {}) or {}
    configured_frames = bioheat_params.get("thermal_update_interval_frames", None)
    if configured_frames is not None:
        frames = int(configured_frames)
        if frames <= 0:
            raise ValueError("bioheat.thermal_update_interval_frames must be >= 1.")
        return frames

    if float(fps) <= 0.0:
        raise ValueError(f"fps must be > 0 to resolve thermal cadence, got {fps}.")
    interval_s = float(bioheat_params.get("thermal_update_interval_s", 1.0 / float(fps)))
    if interval_s <= 0.0:
        raise ValueError("bioheat.thermal_update_interval_s must be > 0.")
    return max(1, int(round(interval_s * float(fps))))


def resolve_cooldown_frame_count(
    *,
    fps: float,
    cooldown_seconds: float,
) -> int:
    """Return the maximum number of thermal-only cooldown frames to append."""
    cooldown_seconds = float(cooldown_seconds)
    if cooldown_seconds <= 0.0:
        return 0
    if float(fps) <= 0.0:
        raise ValueError(f"fps must be > 0 to resolve cooldown frames, got {fps}.")
    return max(0, int(round(cooldown_seconds * float(fps))))


def resolve_implant_off_tail_start_frame(
    total_frames: int,
    *,
    fps: float,
    tail_seconds: float,
) -> int | None:
    """Return the first implant-off frame, excluding short debug clips."""
    tail_frames = int(round(float(fps) * float(tail_seconds)))
    if tail_frames <= 0 or int(total_frames) <= tail_frames:
        return None
    return int(total_frames) - tail_frames


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


def resolve_device_constant_power_mw(params: dict, override_mw: float | None) -> float:
    if override_mw is not None:
        return float(override_mw)
    bioheat = params.get("bioheat", {}) or {}
    return float(
        bioheat.get(
            "device_constant_power_mw",
            bioheat.get("internal_circuit_power_mw", 13.0),
        )
    )


def configure_runtime_device(params: dict, *, force_cpu: bool = False) -> torch.device:
    gpu_id = params.get("run", {}).get("gpu", None)
    if force_cpu or gpu_id is None or gpu_id is False or not torch.cuda.is_available():
        params.setdefault("run", {})["gpu"] = None
        return torch.device("cpu")
    try:
        torch.cuda.set_device(int(gpu_id))
        device = torch.device(f"cuda:{int(gpu_id)}")
        _ = (torch.ones(1, device=device) + 1.0).item()
        return device
    except Exception as exc:
        print(
            "Warning: CUDA is visible but failed a simple tensor test; "
            f"falling back to CPU. CUDA error: {exc}"
        )
        params.setdefault("run", {})["gpu"] = None
        return torch.device("cpu")


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
    thermal_time_s = np.asarray(thermal_metrics.get("thermal_time_s", time_s))
    final_time_s = safe_last(thermal_time_s, default=safe_last(time_s, default=0.0))
    amplitude = np.asarray(common_metrics.get("amplitude_per_electrode_uA", []), dtype=np.float32)
    charge_per_phase = np.asarray(common_metrics.get("charge_per_phase_per_electrode_nC", []), dtype=np.float32)
    charge_density = np.asarray(common_metrics.get("charge_density_per_electrode_uc_cm2", []), dtype=np.float32)
    shannon = np.asarray(common_metrics.get("shannon_k_per_electrode", []), dtype=np.float32)
    charge_rate = np.asarray(common_metrics.get("charge_per_second_per_electrode_nC_s", []), dtype=np.float32)
    total_charge_rate = np.sum(charge_rate, axis=1) if charge_rate.ndim == 2 else np.asarray([], dtype=np.float32)
    total_window_charge = np.asarray(common_metrics.get("window_charge_total_nC", []), dtype=np.float32)
    protocol_charge = np.asarray(common_metrics.get("protocol_charge_per_electrode_nC", []), dtype=np.float32)
    active_count = np.sum(amplitude > 0.0, axis=1) if amplitude.ndim == 2 else np.asarray([], dtype=np.int32)
    max_dT = np.asarray(thermal_metrics["max_dT"])
    mean_dT = np.asarray(thermal_metrics["mean_dT"])
    area_gt1 = np.asarray(thermal_metrics["area_gt1_mm2"])
    area_gt2 = np.asarray(thermal_metrics["area_gt2_mm2"])
    area_gt3 = np.asarray(thermal_metrics["area_gt3_mm2"])
    internal_circuit_power_W = np.asarray(thermal_metrics.get("internal_circuit_power_W", []), dtype=np.float32)
    electrode_load_power_W = np.asarray(thermal_metrics.get("electrode_load_power_W", []), dtype=np.float32)
    max_cem43 = np.asarray(thermal_metrics.get("max_cem43", []))
    p99_cem43 = np.asarray(thermal_metrics.get("p99_cem43", []))
    threshold_uA = scalar_from_value(
        common_metrics.get("threshold_uA", np.asarray(np.nan, dtype=np.float32))
    )
    metrics_save_every_n_frames = int(
        scalar_from_value(
            common_metrics.get("save_every_n_frames", np.asarray(1, dtype=np.int32)),
            default=1.0,
        )
    )
    peak_charge_per_phase_nC = safe_nanmax(charge_per_phase)
    peak_current_amplitude_uA = safe_nanmax(amplitude)
    peak_charge_density_uc_cm2 = safe_nanmax(charge_density)
    peak_shannon_k = safe_nanmax(shannon)

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
        f"frames={int(total_frames)}",
        f"duration_s={final_time_s:.6f}",
        f"fps={float(video_fps):.6f}",
        f"raster_groups={int(groups)}",
        f"raster_cycle_rate_hz={float(raster_timing['cycle_rate_hz']):.6f}",
        f"raster_group_step_rate_hz={float(raster_timing['group_step_rate_hz']):.6f}",
        f"raster_group_interval_s={float(raster_timing['group_interval_s']):.6f}",
        f"internal_circuit_power_total_mW={float(internal_circuit_power_mw):.6f}",
        f"save_every_n_frames={int(metrics_save_every_n_frames)}",
        f"electrical_frame_count={int(time_s.size)}",
        f"thermal_sample_count={int(thermal_time_s.size)}",
        f"cooldown_max_seconds={scalar_from_value(common_metrics.get('cooldown_max_seconds', 0.0), default=0.0):.6f}",
        f"cooldown_duration_s={scalar_from_value(common_metrics.get('cooldown_duration_s', 0.0), default=0.0):.6f}",
        f"cooldown_start_s={scalar_from_value(common_metrics.get('cooldown_start_s', np.nan), default=np.nan):.6f}",
        f"cooldown_baseline_tolerance_C={scalar_from_value(common_metrics.get('cooldown_baseline_tolerance_C', 0.0), default=0.0):.6f}",
        f"cooldown_stop_reason={str(np.asarray(common_metrics.get('cooldown_stop_reason', np.asarray('disabled'))).item())}",
        f"cooldown_thermal_frames={int(scalar_from_value(common_metrics.get('cooldown_thermal_frames', 0), default=0.0))}",
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

    if "amplitude_per_electrode_uA" in common_metrics:
        lines.extend(
            [
                "",
                "Electrical results",
                f"pulse_frequency_hz_min={float(np.min(pulse_frequency_hz)):.6f}",
                f"pulse_frequency_hz_mean={float(np.mean(pulse_frequency_hz)):.6f}",
                f"pulse_frequency_hz_max={float(np.max(pulse_frequency_hz)):.6f}",
                f"threshold_uA={threshold_uA:.6f}",
                f"peak_current_amplitude_uA={peak_current_amplitude_uA:.6f}",
                f"peak_charge_per_phase_nC={peak_charge_per_phase_nC:.6f}",
                f"peak_charge_density_uC_cm2={peak_charge_density_uc_cm2:.6f}",
                f"peak_shannon_k={peak_shannon_k:.6f}",
                f"peak_total_charge_per_second_nC_s={safe_nanmax(total_charge_rate):.6f}",
                f"peak_total_window_charge_nC={safe_nanmax(total_window_charge):.6f}",
                f"final_total_protocol_charge_nC={float(np.nansum(protocol_charge)):.6f}",
                f"peak_active_electrode_count={int(safe_nanmax(active_count, default=0.0))}",
            ]
        )
    else:
        lines.extend(["", "Electrical results", "electrical_tracking=disabled"])

    lines.extend(
        [
            "",
            "Thermal results",
            f"peak_max_dT_C={safe_nanmax(max_dT):.6f}",
            f"final_max_dT_C={safe_last(max_dT):.6f}",
            f"peak_mean_dT_C={safe_nanmax(mean_dT):.6f}",
            f"final_mean_dT_C={safe_last(mean_dT):.6f}",
            f"peak_area_gt1_mm2={safe_nanmax(area_gt1):.6f}",
            f"peak_area_gt2_mm2={safe_nanmax(area_gt2):.6f}",
            f"peak_area_gt3_mm2={safe_nanmax(area_gt3):.6f}",
            f"peak_internal_circuit_power_mW={safe_nanmax(internal_circuit_power_W * 1e3):.6f}",
            f"mean_internal_circuit_power_mW={float(np.nanmean(internal_circuit_power_W * 1e3)) if internal_circuit_power_W.size else float('nan'):.6f}",
            f"peak_electrode_load_power_mW={safe_nanmax(electrode_load_power_W * 1e3):.6f}",
            f"mean_electrode_load_power_mW={float(np.nanmean(electrode_load_power_W * 1e3)) if electrode_load_power_W.size else float('nan'):.6f}",
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
        lines.append(f"final_grid_peak_dT_C[{grid_name}]={grid_max:.6f}")
        if "cem43_final" in grid_data:
            lines.append(f"final_grid_peak_cem43_min[{grid_name}]={grid_cem43_max:.6f}")

    return "\n".join(lines) + "\n"


def write_diagnostics(out_dir: Path, summary_text: str) -> None:
    with open(out_dir / "diagnostics.txt", "w", encoding="utf-8") as handle:
        handle.write(summary_text)


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


def build_electrode_base_indices(
    remaining_indices: np.ndarray,
    base_indices: np.ndarray,
) -> np.ndarray:
    remaining_indices = np.asarray(remaining_indices, dtype=np.int64)
    base_indices = np.asarray(base_indices, dtype=np.int64)
    _, first_idx = np.unique(remaining_indices, return_index=True)
    return base_indices[first_idx].astype(np.int64)


def build_full_electrode_reference(
    coords_yaml: Path,
    mapping: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the full electrode ID/location table used by saved metrics.

    The simulator may only contain phosphenes that survive cortical mapping and
    FOV filtering. Safety outputs and bioheat placement should still use the
    full implanted coordinate table, with zero-valued metrics for electrodes
    that were not represented by a phosphene in this run.
    """
    x_raw_mm, y_raw_mm = utils.load_coordinates_from_yaml(str(coords_yaml))
    x_raw_mm = np.asarray(x_raw_mm, dtype=np.float64).reshape(-1)
    y_raw_mm = np.asarray(y_raw_mm, dtype=np.float64).reshape(-1)
    if x_raw_mm.size != y_raw_mm.size or x_raw_mm.size == 0:
        raise ValueError(f"Invalid electrode coordinate YAML: {coords_yaml}")

    n_base = int(x_raw_mm.size)
    mapped_indices = np.asarray(mapping["indices"], dtype=np.int64).reshape(-1)
    mapped_grid_ids = np.asarray(mapping.get("grid_ids", []), dtype=np.int64).reshape(-1)
    is_full_field = (
        mapped_indices.size > 0
        and (
            int(mapped_indices.max()) >= n_base
            or np.unique(mapped_grid_ids).size > 1
        )
    )

    if is_full_field:
        electrode_ids = np.arange(2 * n_base, dtype=np.int64)
        electrode_xy_mm = np.concatenate(
            [
                np.column_stack([np.abs(x_raw_mm), y_raw_mm]),
                np.column_stack([-np.abs(x_raw_mm), y_raw_mm]),
            ],
            axis=0,
        ).astype(np.float64)
        electrode_grid_ids = np.concatenate(
            [
                np.zeros(n_base, dtype=np.int64),
                np.ones(n_base, dtype=np.int64),
            ]
        )
        electrode_base_indices = np.concatenate(
            [
                np.arange(n_base, dtype=np.int64),
                np.arange(n_base, dtype=np.int64),
            ]
        )
    else:
        electrode_ids = np.arange(n_base, dtype=np.int64)
        electrode_xy_mm = np.column_stack([x_raw_mm, y_raw_mm]).astype(np.float64)
        electrode_grid_ids = np.zeros(n_base, dtype=np.int64)
        electrode_base_indices = np.arange(n_base, dtype=np.int64)

    return electrode_ids, electrode_xy_mm, electrode_grid_ids, electrode_base_indices


def build_reference_index(
    electrode_ids: np.ndarray,
    reference_electrode_ids: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    electrode_ids = np.asarray(electrode_ids, dtype=np.int64).reshape(-1)
    reference_electrode_ids = np.asarray(reference_electrode_ids, dtype=np.int64).reshape(-1)
    positions = {int(electrode_id): idx for idx, electrode_id in enumerate(reference_electrode_ids)}
    missing = [int(electrode_id) for electrode_id in electrode_ids if int(electrode_id) not in positions]
    if missing:
        raise ValueError(
            "Surviving electrode IDs are missing from the full electrode reference: "
            f"{missing[:10]}"
        )
    return torch.tensor(
        [positions[int(electrode_id)] for electrode_id in electrode_ids],
        dtype=torch.long,
        device=device,
    )


def expand_metric_to_reference_tensor(
    metric: torch.Tensor,
    compact_to_reference_t: torch.Tensor,
    reference_count: int,
    *,
    fill_value: float = 0.0,
) -> torch.Tensor:
    metric = metric.reshape(-1).to(compact_to_reference_t.device)
    if metric.numel() != compact_to_reference_t.numel():
        raise ValueError(
            "Metric length does not match compact electrode mapping: "
            f"{metric.numel()} vs {compact_to_reference_t.numel()}."
        )
    out = torch.full(
        (int(reference_count),),
        float(fill_value),
        dtype=metric.dtype,
        device=metric.device,
    )
    out.scatter_(0, compact_to_reference_t, metric)
    return out


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
            "grid_id": int(grid_id),
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


def raster_groups_to_electrodes(
    sim: GaussianSimulator,
    inv_map_t: torch.Tensor,
    n_elec: int,
) -> torch.Tensor:
    """Return one raster group label per physical electrode."""
    if not sim.raster_enabled or sim.raster_groups_flat is None:
        return torch.zeros(n_elec, device=inv_map_t.device, dtype=torch.int32)

    phos_groups = sim.raster_groups_flat.reshape(-1).to(inv_map_t.device, dtype=torch.long)
    num_groups = int(sim.raster_num_groups)
    counts = torch.zeros(
        (n_elec, num_groups),
        device=inv_map_t.device,
        dtype=torch.int32,
    )
    counts.scatter_add_(
        0,
        inv_map_t.reshape(-1, 1).expand(-1, num_groups),
        torch.nn.functional.one_hot(phos_groups, num_classes=num_groups).to(dtype=torch.int32),
    )
    electrode_groups = torch.argmax(counts, dim=1).to(dtype=torch.int32)
    return torch.where(
        counts.sum(dim=1) > 0,
        electrode_groups,
        torch.full_like(electrode_groups, -1),
    )


def aggregate_metric(metric: torch.Tensor, inv_map_t: torch.Tensor, n_elec: int) -> np.ndarray:
    return aggregate_metric_tensor(metric, inv_map_t, n_elec).detach().cpu().numpy().astype(np.float32)


def tensor_list_to_numpy(values: list[torch.Tensor | np.ndarray | float], dtype=np.float32) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=dtype)
    if not any(torch.is_tensor(value) for value in values):
        return np.asarray(values, dtype=dtype)
    stacked = torch.stack([value.detach().reshape(()) for value in values])
    return stacked.cpu().numpy().astype(dtype, copy=False)


def tensor_rows_to_numpy(values: list[torch.Tensor | np.ndarray], dtype=np.float32) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=dtype)
    if not any(torch.is_tensor(value) for value in values):
        return np.asarray(values, dtype=dtype)
    stacked = torch.stack([value.detach().reshape(-1) for value in values], dim=0)
    return stacked.cpu().numpy().astype(dtype, copy=False)


def scalar_tensor_to_numpy(value: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def tensor_row_to_numpy(value: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return value.detach().reshape(-1).cpu().numpy().astype(dtype, copy=True)


def tensor_scalar_to_float(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().reshape(()).cpu().item())
    return float(value)


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


def plot_device_power_diagnostics(
    time_s: np.ndarray,
    active_electrodes: np.ndarray,
    internal_circuit_power_W: np.ndarray,
    electrode_load_power_W: np.ndarray,
    out_path: Path,
    active_time_s: np.ndarray | None = None,
) -> None:
    n_power = min(
        len(time_s),
        len(internal_circuit_power_W),
        len(electrode_load_power_W),
    )
    if n_power <= 0:
        return

    x_min = time_axis_minutes(time_s[:n_power])
    internal_circuit_mW = np.asarray(internal_circuit_power_W[:n_power], dtype=np.float64) * 1e3
    electrode_load_mW = np.asarray(electrode_load_power_W[:n_power], dtype=np.float64) * 1e3

    fig, ax_active = plt.subplots(figsize=(9.0, 4.8))
    active_line = []
    active_time = np.asarray(active_time_s if active_time_s is not None else time_s, dtype=np.float64).reshape(-1)
    active_values = np.asarray(active_electrodes, dtype=np.float64).reshape(-1)
    n_active = min(active_time.size, active_values.size)
    if n_active > 0:
        active_line = ax_active.plot(
            time_axis_minutes(active_time[:n_active]),
            active_values[:n_active],
            color="#475569",
            linewidth=1.5,
            alpha=0.85,
            label="Active electrodes",
        )
    ax_active.set_xlabel("Time (min)", fontsize=11)
    ax_active.set_ylabel("Active electrodes", fontsize=11, color="#475569")
    ax_active.tick_params(axis="y", labelcolor="#475569")
    _style_axes(ax_active)

    ax_power = ax_active.twinx()
    power_lines = []
    power_lines.extend(
        ax_power.plot(x_min, internal_circuit_mW, color="#2563eb", linewidth=1.8, label="Internal-circuit heat")
    )
    power_lines.extend(
        ax_power.plot(x_min, electrode_load_mW, color="#7C3AED", linewidth=1.3, label="Electrode load heat")
    )
    ax_power.set_ylabel("Device power (mW)", fontsize=11)
    ax_power.spines["top"].set_visible(False)
    ax_power.tick_params(labelsize=10, width=0.8)

    lines = active_line + power_lines
    ax_active.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=9, loc="upper left")
    ax_active.set_title("Video-driven device power", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_thermal_response_diagnostics(
    time_s: np.ndarray,
    mean_dT: np.ndarray,
    max_dT: np.ndarray,
    out_path: Path,
) -> None:
    n = min(len(time_s), len(mean_dT), len(max_dT))
    if n <= 0:
        return

    plot_multi_series(
        np.asarray(time_s[:n], dtype=np.float64),
        [
            np.asarray(mean_dT[:n], dtype=np.float64),
            np.asarray(max_dT[:n], dtype=np.float64),
        ],
        ["Mean dT in electrode footprint", "Maximum dT"],
        out_path,
        "Video-driven thermal response",
        "Temperature rise dT (C)",
        ymin=0.0,
    )


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
    cbar.set_label("Î”T (Â°C)", fontsize=11)
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
    electrode_heat_enabled: bool,
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
        voxel_size_mm = np.asarray(thermal_grids[hottest_grid_name]["voxel_size_mm"], dtype=np.float32)
        bioheat_model = str(thermal_grids[hottest_grid_name].get("bioheat_model", "bioheat"))
    else:
        hottest_grid_name = "aggregate"
        hottest_cem43_grid_name = None
        extent_mm = np.zeros(4, dtype=np.float32)
        cem43_extent_mm = None
        footprint_pixels = 0.0
        power_density_w_m3 = 0.0
        voxel_size_mm = np.asarray(0.0, dtype=np.float32)
        bioheat_model = "bioheat"

    save_payload = {
        "time_s": common_metrics["time_s"],
        "thermal_time_s": thermal_metrics.get("thermal_time_s", common_metrics["time_s"]),
        "device_power_time_s": thermal_metrics.get(
            "device_power_time_s",
            thermal_metrics.get("thermal_time_s", common_metrics["time_s"]),
        ),
        "cooldown_max_seconds": common_metrics.get(
            "cooldown_max_seconds",
            np.asarray(0.0, dtype=np.float32),
        ),
        "cooldown_duration_s": common_metrics.get(
            "cooldown_duration_s",
            np.asarray(0.0, dtype=np.float32),
        ),
        "cooldown_start_s": common_metrics.get(
            "cooldown_start_s",
            np.asarray(np.nan, dtype=np.float32),
        ),
        "cooldown_baseline_tolerance_C": common_metrics.get(
            "cooldown_baseline_tolerance_C",
            np.asarray(0.0, dtype=np.float32),
        ),
        "cooldown_stop_reason": common_metrics.get(
            "cooldown_stop_reason",
            np.asarray("disabled"),
        ),
        "cooldown_thermal_frames": common_metrics.get(
            "cooldown_thermal_frames",
            np.asarray(0, dtype=np.int32),
        ),
        "save_every_n_frames": common_metrics["save_every_n_frames"],
        "threshold_uA": common_metrics["threshold_uA"],
        "relative_stim_duration": common_metrics["relative_stim_duration"],
        "pulse_width_s": common_metrics["pulse_width_s"],
        "electrode_surface_area_cm2": common_metrics["electrode_surface_area_cm2"],
        "max_dT": thermal_metrics["max_dT"],
        "mean_dT": thermal_metrics["mean_dT"],
        "area_gt1_mm2": thermal_metrics["area_gt1_mm2"],
        "area_gt2_mm2": thermal_metrics["area_gt2_mm2"],
        "area_gt3_mm2": thermal_metrics["area_gt3_mm2"],
        "dT_final": thermal_metrics["dT_final"],
        "internal_circuit_power_W": thermal_metrics["internal_circuit_power_W"],
        "electrode_load_power_W": thermal_metrics["electrode_load_power_W"],
        "fps": np.asarray(video_fps, dtype=np.float32),
        "raster_mode": common_metrics["raster_mode"],
        "raster_mode_normalized": common_metrics["raster_mode_normalized"],
        "raster_num_groups": common_metrics["raster_num_groups"],
        "raster_rate_hz": np.asarray(raster_rate_hz, dtype=np.float32),
        "raster_group_step_rate_hz": np.asarray(raster_timing["group_step_rate_hz"], dtype=np.float32),
        "raster_group_interval_s": np.asarray(raster_timing["group_interval_s"], dtype=np.float32),
        "raster_reshuffle_interval_s": common_metrics["raster_reshuffle_interval_s"],
        "extent_mm": extent_mm,
        "voxel_size_mm": voxel_size_mm,
        "bioheat_model": np.asarray(bioheat_model),
        "pulse_frequency_hz": pulse_frequency_hz.astype(np.float32),
        "electrode_ids": common_metrics["electrode_ids"],
        "electrode_base_indices": common_metrics["electrode_base_indices"],
        "electrode_xy_mm": common_metrics["electrode_xy_mm"],
        "electrode_impedance_ohm": common_metrics["electrode_impedance_ohm"],
        "electrode_grid_ids": common_metrics["electrode_grid_ids"],
        "electrode_grid_names": common_metrics["electrode_grid_names"],
        "internal_circuit_power_total_mW": np.asarray(internal_circuit_power_mw, dtype=np.float32),
        "electrode_heat_enabled": np.asarray(bool(electrode_heat_enabled)),
        "internal_circuit_footprint_pixels": np.asarray(footprint_pixels, dtype=np.float32),
        "internal_circuit_power_density_W_m3": np.asarray(power_density_w_m3, dtype=np.float32),
    }
    electrical_payload_keys = (
        "amplitude_per_electrode_uA",
        "charge_per_phase_per_electrode_nC",
        "charge_density_per_electrode_uc_cm2",
        "shannon_k_per_electrode",
        "charge_per_second_per_electrode_nC_s",
        "window_charge_per_electrode_nC",
        "window_charge_total_nC",
        "protocol_charge_per_electrode_nC",
        "power_per_electrode_W",
        "pulse_width_per_electrode_s",
        "pulse_frequency_per_electrode_hz",
        "active_electrode_count",
        "window_exceedance_time_start_s",
        "window_exceedance_time_end_s",
        "window_exceedance_scope",
        "window_exceedance_electrode_id",
        "window_exceedance_charge_nC",
        "window_exceedance_limit_nC",
        "charge_window_s",
        "raster_active_group",
        "raster_group_assignment_frame_indices",
        "raster_group_assignment_times_s",
        "raster_group_assignments",
    )
    for key in electrical_payload_keys:
        if key in common_metrics:
            save_payload[key] = common_metrics[key]
    if "max_cem43" in thermal_metrics:
        save_payload["max_cem43"] = thermal_metrics["max_cem43"]
    if "p99_cem43" in thermal_metrics:
        save_payload["p99_cem43"] = thermal_metrics["p99_cem43"]
    if "cem43_final" in thermal_metrics:
        save_payload["cem43_final"] = thermal_metrics["cem43_final"]
    if cem43_extent_mm is not None:
        save_payload["cem43_extent_mm"] = cem43_extent_mm

    if thermal_grids:
        save_payload["thermal_grid_names"] = np.asarray(list(thermal_grids.keys()))
        save_payload["dT_final_reference_grid"] = np.asarray(hottest_grid_name)
        save_payload["final_peak_temperature_C"] = np.asarray(
            thermal_grids[hottest_grid_name]["final_peak_temperature_C"],
            dtype=np.float32,
        )
        save_payload["final_peak_dT_C"] = np.asarray(
            thermal_grids[hottest_grid_name]["final_peak_dT_C"],
            dtype=np.float32,
        )
        if hottest_cem43_grid_name is not None:
            save_payload["cem43_final_reference_grid"] = np.asarray(hottest_cem43_grid_name)
        for grid_name, grid_data in thermal_grids.items():
            suffix = sanitize_path_part(grid_name)
            save_payload[f"dT_final_{suffix}"] = grid_data["dT_final"]
            save_payload[f"final_peak_temperature_C_{suffix}"] = np.asarray(
                grid_data["final_peak_temperature_C"],
                dtype=np.float32,
            )
            save_payload[f"final_peak_dT_C_{suffix}"] = np.asarray(
                grid_data["final_peak_dT_C"],
                dtype=np.float32,
            )
            if "cem43_final" in grid_data:
                save_payload[f"cem43_final_{suffix}"] = grid_data["cem43_final"]
            save_payload[f"extent_mm_{suffix}"] = np.asarray(grid_data["extent_mm"], dtype=np.float32)
            save_payload[f"voxel_size_mm_{suffix}"] = np.asarray(grid_data["voxel_size_mm"], dtype=np.float32)
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

    write_metrics(out_dir / "metrics.npz", save_payload)


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
                 save_every_n_frames: int | None,
                 enable_cem43: bool,
                 device: torch.device,
                 phosphene_mode: str = "safety_centers",
                 thermal_update_interval_frames: int | None = None,
                 cooldown_seconds: float = 300.0,
                 cooldown_baseline_tolerance_C: float = 1e-3,
                 appearance_threshold_uA: float | None = None,
                 track_electrical: bool = True,
                 electrode_heat_enabled: bool = True,
                 strategy=None,
                 input_stage: str = "preprocessed",
                 preprocessing_options: dict | None = None):
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
        apply_adaptive_activation_threshold(simulation_params, fixed_firing_threshold_a)
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
    n_elec_surv = elec_xy_mm_surv.shape[0]
    surviving_electrode_ids, _surviving_electrode_xy_mm = build_surviving_electrode_metadata(
        aligned_mapping["cortical_coordinates"],
        aligned_mapping["indices"],
    )
    electrode_ids, electrode_xy_mm, electrode_grid_ids, electrode_base_indices = build_full_electrode_reference(
        coords_yaml,
        aligned_mapping,
    )
    n_elec = int(electrode_ids.shape[0])
    compact_to_reference_t = build_reference_index(
        surviving_electrode_ids,
        electrode_ids,
        device=device,
    )
    impedance = Impedance(
        simulation_params,
        sim.shape,
        rng=np.random.default_rng(int(simulation_params.get("run", {}).get("seed", 42))),
        verbose=bool(simulation_params.get("run", {}).get("print_stats", False)),
    )
    impedance.state = impedance.state.to(device=device, dtype=torch.float32)
    safety_tracker = (
        SafetyTracker(
            params=simulation_params,
            num_electrodes=sim.num_phosphenes,
            data_kwargs={**sim.data_kwargs, "device": str(device)},
        )
        if track_electrical
        else None
    )
    impedance_phos_t = impedance.state.reshape(-1).to(device=device, dtype=torch.float32)
    impedance_sum_t = torch.zeros(n_elec_surv, dtype=torch.float32, device=device)
    impedance_count_t = torch.zeros(n_elec_surv, dtype=torch.float32, device=device)
    impedance_sum_t.scatter_add_(0, inv_map_t, impedance_phos_t)
    impedance_count_t.scatter_add_(0, inv_map_t, torch.ones_like(impedance_phos_t))
    electrode_impedance_ohm_surv = (
        impedance_sum_t / impedance_count_t.clamp_min(1.0)
    )
    electrode_impedance_ohm = expand_metric_to_reference_tensor(
        electrode_impedance_ohm_surv,
        compact_to_reference_t,
        n_elec,
        fill_value=float("nan"),
    ).detach().cpu().numpy().astype(np.float32)
    bioheat_grid_sets = build_bioheat_grid_sets(
        Map(x=electrode_xy_mm[:, 0], y=electrode_xy_mm[:, 1]),
        electrode_grid_ids,
        electrode_base_indices,
        device=device,
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
    video_frame_limit = resolve_simulation_frame_count(cap, max_frames)
    cooldown_seconds = float(cooldown_seconds)
    cooldown_baseline_tolerance_C = float(cooldown_baseline_tolerance_C)
    if cooldown_baseline_tolerance_C < 0.0:
        raise ValueError("cooldown_baseline_tolerance_C must be >= 0.")
    cooldown_max_frames = resolve_cooldown_frame_count(
        fps=fps,
        cooldown_seconds=cooldown_seconds,
    )
    scheduled_thermal_frames = (
        video_frame_limit + cooldown_max_frames
        if video_frame_limit > 0
        else video_frame_limit
    )
    heatmap_snapshot_schedule = build_heatmap_snapshot_schedule(scheduled_thermal_frames, dt)
    heatmap_snapshot_lookup = {
        int(item["frame_idx"]): item for item in heatmap_snapshot_schedule
    }
    cooldown_thermal_frames = 0
    cooldown_stop_reason = "disabled" if cooldown_max_frames <= 0 else "not_started"
    if cooldown_max_frames > 0:
        print(
            "Post-video thermal cooldown | "
            f"max_duration={cooldown_seconds:.2f}s | "
            f"baseline_tolerance={cooldown_baseline_tolerance_C:.6f} C | "
            "device power off"
        )
    if not track_electrical:
        print("Electrical tracking disabled; saving thermal metrics only.")
    target_res = tuple(int(v) for v in simulation_params["run"]["resolution"])
    input_stage = str(input_stage).strip().lower()
    if input_stage not in {"original", "preprocessed"}:
        raise ValueError("input_stage must be either 'original' or 'preprocessed'.")
    preprocessing_options = dict(preprocessing_options or {})
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

    amplitude_per_electrode_uA = []
    charge_per_phase_per_electrode_nC = []
    charge_density_per_electrode_uc_cm2 = []
    shannon_k_per_electrode = []
    charge_per_second_per_electrode_nC_s = []
    window_charge_per_electrode_nC = []
    window_charge_total_nC = []
    power_per_electrode_W = []
    pulse_width_per_electrode_s = []
    pulse_frequency_per_electrode_hz = []
    active_electrode_count = []
    raster_active_group = []
    raster_group_assignment_frame_indices = []
    raster_group_assignment_times_s = []
    raster_group_assignments = []
    window_exceedance_time_start_s = []
    window_exceedance_time_end_s = []
    window_exceedance_scope = []
    window_exceedance_electrode_id = []
    window_exceedance_charge_nC = []
    window_exceedance_limit_nC = []
    final_protocol_charge_per_electrode_nC = torch.zeros(n_elec, dtype=torch.float32, device=device)
    time_s = []

    pulse_frequency_hz_compact_t = aggregate_metric_tensor(sim._frequency.reshape(-1), inv_map_t, n_elec_surv).to(
        device=device,
        dtype=torch.float32,
    )
    pulse_width_s_compact_t = aggregate_metric_tensor(sim._pulse_width.reshape(-1), inv_map_t, n_elec_surv).to(
        device=device,
        dtype=torch.float32,
    )
    pulse_frequency_hz = expand_metric_to_reference_tensor(
        pulse_frequency_hz_compact_t,
        compact_to_reference_t,
        n_elec,
    ).detach().cpu().numpy().astype(np.float32)
    pulse_width_s = expand_metric_to_reference_tensor(
        pulse_width_s_compact_t,
        compact_to_reference_t,
        n_elec,
    ).detach().cpu().numpy().astype(np.float32)
    strategy_rng = np.random.default_rng(int(simulation_params.get("run", {}).get("seed", 42)))
    strategy_context = None
    if strategy is not None:
        strategy_context = StrategyContext(
            electrode_ids=electrode_ids.copy(),
            electrode_xy_mm=electrode_xy_mm.copy(),
            default_pulse_width_s=pulse_width_s.copy(),
            default_frequency_hz=pulse_frequency_hz.copy(),
            fps=fps,
            seed=int(simulation_params.get("run", {}).get("seed", 42)),
        )
        strategy.initialize(strategy_context)
    electrode_area_cm2 = float(simulation_params["safety"]["electrode_surface_area_cm2"])
    relative_stim_duration = float(simulation_params.get("default_stim", {}).get("relative_stim_duration", 1.0))
    configured_save_every = (
        int(save_every_n_frames)
        if save_every_n_frames is not None
        else int(simulation_params.get("safety", {}).get("metrics_save_every_n_frames", 5))
    )
    if configured_save_every <= 0:
        raise ValueError("save_every_n_frames must be >= 1.")
    last_raster_assignment_version: int | None = None
    configured_thermal_update_interval = resolve_thermal_update_interval_frames(
        simulation_params,
        fps=fps,
        override_frames=thermal_update_interval_frames,
    )
    print(
        "Thermal cadence | "
        f"stimulation update every video frame ({dt:.3f}s); "
        f"cooldown batch up to {configured_thermal_update_interval} frame(s)"
    )
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
            grid_params["bioheat"]["device_constant_power_mw"] = ic_power_mw
            bio = Bioheat2D(params=grid_params, elec_xy_mm=grid_info["elec_xy_mm"], device=device)
            electrode_mask_t = torch.tensor(
                electrode_grid_ids == int(grid_id),
                dtype=torch.bool,
                device=device,
            )
            grid_heat_states[grid_info["name"]] = {
                "bio": bio,
                "mask_t": grid_info["mask_t"],
                "inverse_t": grid_info["inverse_t"],
                "n_elec": grid_info["n_elec"],
                "electrode_mask_t": electrode_mask_t,
                "baseline_temp": float(bio.baseline_T),
                "pixel_area_mm2": float(bio.dx * bio.dy) * 1e6,
                "device_constant_power_W": float(ic_power_mw) * 1e-3,
                "prev_dT_map": torch.zeros_like(bio.dT_projection()),
            }
            if enable_cem43:
                grid_heat_states[grid_info["name"]]["cem43_map"] = torch.zeros_like(bio.dT)

        mode_states[ic_heat_mode] = {
            "out_dir": out_dir,
            "preview_out_dir": preview_out_dirs[ic_heat_mode],
            "internal_circuit_power_mw": ic_power_mw * max(len(grid_heat_states), 1),
            "ic_enabled": ic_heat_mode == "with",
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
            "internal_circuit_power_W": [],
            "electrode_load_power_W": [],
            "device_power_time_s": [],
            "thermal_time_s": [],
        }
        if enable_cem43:
            mode_states[ic_heat_mode]["snapshot_cem43_grids"] = {
                grid_name: [] for grid_name in grid_heat_states
            }
            mode_states[ic_heat_mode]["max_cem43"] = []
            mode_states[ic_heat_mode]["p99_cem43"] = []

    def update_bioheat_from_electrode_power(
        electrode_power_elec: torch.Tensor,
        update_dt: float,
        *,
        include_internal_power: bool = True,
    ) -> None:
        """Advance Bioheat2D using per-electrode dissipated load power."""
        if update_dt <= 0.0:
            return

        for state in mode_states.values():
            for grid_name, grid_state in state["grid_heat_states"].items():
                bio = grid_state["bio"]
                p_grid = electrode_power_elec[grid_state["electrode_mask_t"]]
                if enable_cem43:
                    cem43_weight_prev = cem43_weight_from_temperature(bio.temperature_C)
                original_internal_power_W = float(getattr(bio, "internal_circuit_power_W", 0.0))
                if not include_internal_power:
                    bio.internal_circuit_power_W = 0.0
                try:
                    bio.update(p_grid, update_dt)
                finally:
                    bio.internal_circuit_power_W = original_internal_power_W
                dT_map = bio.dT_projection().detach()
                if enable_cem43:
                    cem43_weight_cur = cem43_weight_from_temperature(bio.temperature_C)
                    grid_state["cem43_map"].add_(
                        0.5 * (cem43_weight_prev + cem43_weight_cur) * (update_dt / 60.0)
                    )
                grid_state["prev_dT_map"] = dT_map

    def record_thermal_metrics_and_snapshots(
        snapshot_entry: dict[str, object] | None,
        sample_time_s: float,
    ) -> None:
        for state in mode_states.values():
            state["thermal_time_s"].append(float(sample_time_s))
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
                ic_mask = grid_state["bio"].ic_footprint_mask.to(device=dT_map.device, dtype=torch.bool)
                if torch.any(ic_mask):
                    grid_mean_value = dT_map[ic_mask].mean()
                    grid_mean_weight = int(torch.count_nonzero(ic_mask).item())
                else:
                    grid_mean_value = dT_map.mean()
                    grid_mean_weight = int(dT_map.numel())
                state["grid_max_dT"][grid_name].append(tensor_scalar_to_float(grid_max_value))
                state["grid_mean_dT"][grid_name].append(tensor_scalar_to_float(grid_mean_value))

                grid_max_dT.append(grid_max_value)
                if enable_cem43:
                    grid_cem43_max_value = grid_state["cem43_map"].max()
                    grid_cem43_maps.append(grid_state["cem43_map"])
                    grid_max_cem43.append(grid_cem43_max_value)
                weighted_mean_sum += grid_mean_value * grid_mean_weight
                weighted_mean_count += grid_mean_weight
                area_gt1_total += (dT_map > 1.0).sum(dtype=torch.float32) * grid_state["pixel_area_mm2"]
                area_gt2_total += (dT_map > 2.0).sum(dtype=torch.float32) * grid_state["pixel_area_mm2"]
                area_gt3_total += (dT_map > 3.0).sum(dtype=torch.float32) * grid_state["pixel_area_mm2"]

            if not grid_max_dT:
                raise RuntimeError("No bioheat grids were configured for the current mode.")

            state["max_dT"].append(tensor_scalar_to_float(torch.stack(grid_max_dT).max()))
            state["mean_dT"].append(tensor_scalar_to_float(weighted_mean_sum / max(weighted_mean_count, 1)))
            state["area_gt1_mm2"].append(tensor_scalar_to_float(area_gt1_total))
            state["area_gt2_mm2"].append(tensor_scalar_to_float(area_gt2_total))
            state["area_gt3_mm2"].append(tensor_scalar_to_float(area_gt3_total))
            if enable_cem43:
                state["max_cem43"].append(tensor_scalar_to_float(torch.stack(grid_max_cem43).max()))
                state["p99_cem43"].append(
                    tensor_scalar_to_float(percentile_tensor_from_tensor_list(grid_cem43_maps, 99.0, device))
                )

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
                            thermal_projection(grid_state["cem43_map"]).detach().cpu().numpy().astype(np.float32)
                        )

    def remove_latest_thermal_metric_samples() -> None:
        for state in mode_states.values():
            if state["thermal_time_s"]:
                state["thermal_time_s"].pop()
            for series in state["grid_max_dT"].values():
                if series:
                    series.pop()
            for series in state["grid_mean_dT"].values():
                if series:
                    series.pop()
            for key in (
                "max_dT",
                "mean_dT",
                "area_gt1_mm2",
                "area_gt2_mm2",
                "area_gt3_mm2",
            ):
                if state[key]:
                    state[key].pop()
            if enable_cem43:
                for key in ("max_cem43", "p99_cem43"):
                    if state[key]:
                        state[key].pop()

    def current_peak_dT_C() -> float:
        peaks = []
        for state in mode_states.values():
            for grid_state in state["grid_heat_states"].values():
                dT_map = grid_state["prev_dT_map"]
                if dT_map is not None and dT_map.numel() > 0:
                    peaks.append(float(dT_map.max().detach().cpu().item()))
        if not peaks:
            return 0.0
        return max(peaks)

    frame_idx = 0
    thermal_frame_idx = 0

    try:
        while True:
            if max_frames > 0 and frame_idx >= int(max_frames):
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            thermal_frame_idx = frame_idx

            if input_stage == "original":
                _input_frame, gray = prepare_stimulus_frame(
                    frame,
                    input_stage="original",
                    render_resolution_xy=target_res,
                    preprocessing_method=preprocessing_method,
                    dog_sigma_low=float(preprocessing_options.get("dog_sigma_low", 2.0)),
                    dog_sigma_high=float(preprocessing_options.get("dog_sigma_high", 6.0)),
                    canny_low=float(preprocessing_options.get("canny_low", 75.0)),
                    canny_high=float(preprocessing_options.get("canny_high", 170.0)),
                    use_cuda=bool(preprocessing_options.get("use_cuda", False)),
                )
            else:
                gray = prepare_frame(frame, target_res)
                if binarize_input:
                    gray = restore_binary_preprocessed_frame(gray)
            stim_raw = sim.sample_stimulus(gray, rescale=True).reshape(-1).to(device)

            stim = apply_appearance_threshold(stim_raw, fixed_firing_threshold_a)
            pulse_width_command = None
            frequency_command = None
            if strategy is not None:
                baseline_compact = aggregate_metric_tensor(
                    stim,
                    inv_map_t,
                    n_elec_surv,
                ).to(dtype=torch.float32)
                baseline_electrode = expand_metric_to_reference_tensor(
                    baseline_compact,
                    compact_to_reference_t,
                    n_elec,
                )
                command = strategy.step(
                    StrategyInput(
                        time_s=float((frame_idx - 1) * dt),
                        frame_index=int(frame_idx),
                        stimulus_image=np.asarray(gray, dtype=np.float32) / 255.0,
                        baseline_amplitude_A=baseline_electrode.detach().cpu().numpy(),
                        electrode_ids=electrode_ids.copy(),
                        electrode_xy_mm=electrode_xy_mm.copy(),
                        rng=strategy_rng,
                    )
                ).validated(strategy_context)
                amplitude_reference = torch.as_tensor(
                    command.amplitude_A,
                    dtype=torch.float32,
                    device=device,
                )
                pulse_width_reference = torch.as_tensor(
                    command.pulse_width_s,
                    dtype=torch.float32,
                    device=device,
                )
                frequency_reference = torch.as_tensor(
                    command.frequency_hz,
                    dtype=torch.float32,
                    device=device,
                )
                stim = amplitude_reference[compact_to_reference_t][inv_map_t]
                pulse_width_command = pulse_width_reference[compact_to_reference_t][inv_map_t]
                frequency_command = frequency_reference[compact_to_reference_t][inv_map_t]

            # The phosphene simulation is shared across IC modes, so delivered
            # current and raster state are updated only once per frame.
            sim.update(
                stim,
                pulse_width=pulse_width_command,
                frequency=frequency_command,
                dt=dt,
                temperature_increase=None,
            )
            if track_electrical:
                safety_tracker.update(
                    charge_per_s=sim.delivered_charge_per_second,
                    frequency=sim.current_frequency,
                    dt_s=dt,
                    temperature_increase=None,
                )

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

            current_time_s = frame_idx * dt

            current_amplitude_compact = aggregate_metric_tensor(
                (sim.delivered_amplitude * 1e6).reshape(-1),
                inv_map_t,
                n_elec_surv,
            ).to(
                dtype=torch.float32
            )
            current_amplitude_elec = expand_metric_to_reference_tensor(
                current_amplitude_compact,
                compact_to_reference_t,
                n_elec,
            )
            current_pulse_width_compact = aggregate_metric_tensor(
                sim.current_pulse_width.reshape(-1),
                inv_map_t,
                n_elec_surv,
            ).to(dtype=torch.float32)
            current_frequency_compact = aggregate_metric_tensor(
                sim.current_frequency.reshape(-1),
                inv_map_t,
                n_elec_surv,
            ).to(dtype=torch.float32)
            current_pulse_width_elec = expand_metric_to_reference_tensor(
                current_pulse_width_compact,
                compact_to_reference_t,
                n_elec,
            )
            current_frequency_elec = expand_metric_to_reference_tensor(
                current_frequency_compact,
                compact_to_reference_t,
                n_elec,
            )

            # Per-electrode load power is the Bioheat2D electrode heat input.
            _instant_power, frame_power = compute_frame_power(
                sim.delivered_amplitude,
                impedance.state,
                sim.current_pulse_width,
                sim.current_frequency,
                relative_stim_duration,
            )
            fp_phos = frame_power.reshape(-1).to(device)
            frame_power_compact = aggregate_metric_tensor(fp_phos, inv_map_t, n_elec_surv).to(dtype=torch.float32)
            frame_power_elec = expand_metric_to_reference_tensor(
                frame_power_compact,
                compact_to_reference_t,
                n_elec,
            )
            if not electrode_heat_enabled:
                frame_power_elec = torch.zeros_like(frame_power_elec)
            if track_electrical:
                # SafetyTracker stores per-phosphene values; aggregate them
                # back to physical electrodes for the general safety pipeline.
                q_phase_compact = aggregate_metric_tensor(
                    safety_tracker.last_charge_per_phase_nC,
                    inv_map_t,
                    n_elec_surv,
                ).to(dtype=torch.float32)
                q_window_compact = aggregate_metric_tensor(
                    safety_tracker.window_charge_per_electrode_nC,
                    inv_map_t,
                    n_elec_surv,
                ).to(dtype=torch.float32)
                q_protocol_compact = aggregate_metric_tensor(
                    safety_tracker.protocol_charge_per_electrode_nC,
                    inv_map_t,
                    n_elec_surv,
                ).to(dtype=torch.float32)
                q_phase_elec = expand_metric_to_reference_tensor(
                    q_phase_compact,
                    compact_to_reference_t,
                    n_elec,
                )
                q_window_elec = expand_metric_to_reference_tensor(
                    q_window_compact,
                    compact_to_reference_t,
                    n_elec,
                )
                q_protocol_elec = expand_metric_to_reference_tensor(
                    q_protocol_compact,
                    compact_to_reference_t,
                    n_elec,
                )
                final_protocol_charge_per_electrode_nC = q_protocol_elec

                charge_density_elec = q_phase_elec / 1e3 / electrode_area_cm2
                shannon_elec = torch.full_like(charge_density_elec, -torch.inf)
                valid = (q_phase_elec > 0) & (charge_density_elec > 0)
                shannon_elec[valid] = torch.log10(q_phase_elec[valid] / 1e3) + torch.log10(charge_density_elec[valid])

                charge_rate_compact_nC_s = (
                    2.0
                    * current_amplitude_compact
                    * current_pulse_width_compact
                    * current_frequency_compact
                    * relative_stim_duration
                    * 1e3
                )
                charge_rate_elec_nC_s = expand_metric_to_reference_tensor(
                    charge_rate_compact_nC_s,
                    compact_to_reference_t,
                    n_elec,
                )
                active_count_frame = torch.count_nonzero(current_amplitude_elec > 0.0).to(dtype=torch.float32)
                active_electrode_count.append(tensor_scalar_to_float(active_count_frame))

            for state in mode_states.values():
                state_internal_circuit_power_W = 0.0
                state_electrode_load_power_W = 0.0
                for grid_state in state["grid_heat_states"].values():
                    electrode_mask = grid_state["electrode_mask_t"]
                    grid_load_power_W = frame_power_elec[electrode_mask].reshape(-1).sum()
                    constant_power_W = float(grid_state["device_constant_power_W"])
                    electrode_load_power_value = float(grid_load_power_W.detach().cpu().item())
                    internal_circuit_power_value = constant_power_W if state["ic_enabled"] else 0.0

                    state_internal_circuit_power_W += internal_circuit_power_value
                    state_electrode_load_power_W += electrode_load_power_value

                state["internal_circuit_power_W"].append(state_internal_circuit_power_W)
                state["electrode_load_power_W"].append(state_electrode_load_power_W)
                state["device_power_time_s"].append(float(current_time_s))

            update_bioheat_from_electrode_power(frame_power_elec, dt)

            if track_electrical:
                total_window_charge_nC = q_window_elec.sum()
                window_start_s = max(0.0, current_time_s - float(safety_tracker.charge_window_s))
                over_window = torch.nonzero(
                    q_window_elec > float(safety_tracker.acc_limit_per_electrode_nC),
                    as_tuple=False,
                ).reshape(-1)
                for idx_t in over_window.detach().cpu().tolist():
                    idx = int(idx_t)
                    window_exceedance_time_start_s.append(window_start_s)
                    window_exceedance_time_end_s.append(float(current_time_s))
                    window_exceedance_scope.append("electrode")
                    window_exceedance_electrode_id.append(int(electrode_ids[idx]) if idx < len(electrode_ids) else idx)
                    window_exceedance_charge_nC.append(float(q_window_elec[idx].detach().cpu().item()))
                    window_exceedance_limit_nC.append(float(safety_tracker.acc_limit_per_electrode_nC))
                if float(total_window_charge_nC.detach().cpu().item()) > float(safety_tracker.acc_limit_total_nC):
                    window_exceedance_time_start_s.append(window_start_s)
                    window_exceedance_time_end_s.append(float(current_time_s))
                    window_exceedance_scope.append("total")
                    window_exceedance_electrode_id.append(-1)
                    window_exceedance_charge_nC.append(float(total_window_charge_nC.detach().cpu().item()))
                    window_exceedance_limit_nC.append(float(safety_tracker.acc_limit_total_nC))

                raster_active_group.append(int(sim.current_raster_group) if sim.raster_enabled else -1)
                raster_assignment_version = (
                    int(getattr(sim, "raster_assignment_version", 0))
                    if sim.raster_enabled
                    else -1
                )
                if raster_assignment_version != last_raster_assignment_version:
                    raster_assignment_compact = raster_groups_to_electrodes(sim, inv_map_t, n_elec_surv)
                    raster_assignment = expand_metric_to_reference_tensor(
                        raster_assignment_compact.to(dtype=torch.float32),
                        compact_to_reference_t,
                        n_elec,
                        fill_value=-1.0,
                    ).to(dtype=torch.int32)
                    raster_group_assignment_frame_indices.append(int(frame_idx))
                    raster_group_assignment_times_s.append(float(current_time_s))
                    raster_group_assignments.append(tensor_row_to_numpy(raster_assignment, dtype=np.int32))
                    last_raster_assignment_version = raster_assignment_version

            snapshot_entry = heatmap_snapshot_lookup.get(frame_idx)
            record_thermal_metrics_and_snapshots(snapshot_entry, current_time_s)

            if track_electrical:
                amplitude_per_electrode_uA.append(tensor_row_to_numpy(current_amplitude_elec))
                charge_per_phase_per_electrode_nC.append(tensor_row_to_numpy(q_phase_elec))
                charge_density_per_electrode_uc_cm2.append(tensor_row_to_numpy(charge_density_elec))
                shannon_k_per_electrode.append(tensor_row_to_numpy(shannon_elec))
                charge_per_second_per_electrode_nC_s.append(tensor_row_to_numpy(charge_rate_elec_nC_s))
                window_charge_per_electrode_nC.append(tensor_row_to_numpy(q_window_elec))
                window_charge_total_nC.append(tensor_scalar_to_float(total_window_charge_nC))
                power_per_electrode_W.append(tensor_row_to_numpy(frame_power_elec))
                pulse_width_per_electrode_s.append(tensor_row_to_numpy(current_pulse_width_elec))
                pulse_frequency_per_electrode_hz.append(tensor_row_to_numpy(current_frequency_elec))
            time_s.append(current_time_s)

            if frame_idx == 1 or frame_idx % 10 == 0:
                elapsed_s = max(time.perf_counter() - progress_started_at, 1e-9)
                frame_rate = float(frame_idx) / elapsed_s
                if video_frame_limit > 0:
                    progress_pct = 100.0 * float(frame_idx) / float(video_frame_limit)
                    write_progress_line(
                        f"{raster_label}: video frame {frame_idx}/{video_frame_limit} "
                        f"({progress_pct:5.1f}%) | elapsed {elapsed_s:7.1f}s | "
                        f"{frame_rate:5.2f} frames/s"
                    )
                else:
                    write_progress_line(
                        f"{raster_label}: frame {frame_idx} | elapsed {elapsed_s:7.1f}s | "
                        f"{frame_rate:5.2f} frames/s"
                    )

        if cooldown_max_frames > 0:
            cooldown_start_frame = thermal_frame_idx
            cooldown_started_at = time.perf_counter()
            snapshot_frames = sorted(
                frame for frame in heatmap_snapshot_lookup
                if frame > thermal_frame_idx
            )
            snapshot_cursor = 0
            zero_power_elec = torch.zeros(n_elec, dtype=torch.float32, device=device)
            if current_peak_dT_C() <= cooldown_baseline_tolerance_C:
                cooldown_stop_reason = "baseline"
            while cooldown_thermal_frames < cooldown_max_frames and cooldown_stop_reason != "baseline":
                frames_remaining = cooldown_max_frames - cooldown_thermal_frames
                segment_frames = min(configured_thermal_update_interval, frames_remaining)
                if snapshot_cursor < len(snapshot_frames):
                    next_snapshot_frame = snapshot_frames[snapshot_cursor]
                    if next_snapshot_frame > thermal_frame_idx:
                        segment_frames = min(segment_frames, next_snapshot_frame - thermal_frame_idx)

                update_bioheat_from_electrode_power(
                    zero_power_elec,
                    dt * float(segment_frames),
                    include_internal_power=False,
                )
                thermal_frame_idx += segment_frames
                cooldown_thermal_frames += segment_frames
                current_time_s = thermal_frame_idx * dt
                for state in mode_states.values():
                    state["internal_circuit_power_W"].append(0.0)
                    state["electrode_load_power_W"].append(0.0)
                    state["device_power_time_s"].append(float(current_time_s))

                snapshot_entry = heatmap_snapshot_lookup.get(thermal_frame_idx)
                record_thermal_metrics_and_snapshots(snapshot_entry, current_time_s)
                if snapshot_entry is not None:
                    snapshot_cursor += 1

                if current_peak_dT_C() <= cooldown_baseline_tolerance_C:
                    cooldown_stop_reason = "baseline"

                if cooldown_thermal_frames == cooldown_max_frames or (thermal_frame_idx - cooldown_start_frame) % max(configured_thermal_update_interval * 10, 1) == 0:
                    elapsed_s = max(time.perf_counter() - cooldown_started_at, 1e-9)
                    progress_pct = 100.0 * float(cooldown_thermal_frames) / float(cooldown_max_frames)
                    write_progress_line(
                        f"{raster_label}: cooldown frame {cooldown_thermal_frames}/{cooldown_max_frames} "
                        f"({progress_pct:5.1f}%) | elapsed {elapsed_s:7.1f}s"
                    )

            if cooldown_stop_reason == "not_started":
                cooldown_stop_reason = "time_limit"
    finally:
        cap.release()
        for state in mode_states.values():
            if state["preview_phosphene_writer"] is not None:
                state["preview_phosphene_writer"].release()
            if state["preview_comparison_writer"] is not None:
                state["preview_comparison_writer"].release()

    if frame_idx > 0:
        finish_progress_line()

    if frame_idx == 0:
        raise RuntimeError("No frames were read from the input video.")

    common_metrics = {
        "time_s": np.asarray(time_s, dtype=np.float32),
        "save_every_n_frames": np.asarray(configured_save_every, dtype=np.int32),
        "threshold_uA": np.asarray(fixed_firing_threshold_a * 1e6, dtype=np.float32),
        "cooldown_max_seconds": np.asarray(cooldown_seconds, dtype=np.float32),
        "cooldown_duration_s": np.asarray(
            float(cooldown_thermal_frames) * dt,
            dtype=np.float32,
        ),
        "cooldown_start_s": np.asarray(
            np.nan if cooldown_max_frames <= 0 else float(frame_idx) * dt,
            dtype=np.float32,
        ),
        "cooldown_baseline_tolerance_C": np.asarray(cooldown_baseline_tolerance_C, dtype=np.float32),
        "cooldown_stop_reason": np.asarray(cooldown_stop_reason),
        "cooldown_thermal_frames": np.asarray(
            cooldown_thermal_frames,
            dtype=np.int32,
        ),
        "raster_mode": np.asarray(raster_label),
        "raster_mode_normalized": np.asarray(raster_pattern_name),
        "raster_num_groups": np.asarray(int(groups), dtype=np.int32),
        "raster_reshuffle_interval_s": np.asarray(
            float(getattr(sim, "raster_reshuffle_interval_s", 0.0)) if sim.raster_enabled else 0.0,
            dtype=np.float32,
        ),
        "electrode_ids": electrode_ids.astype(np.int64),
        "electrode_base_indices": electrode_base_indices.astype(np.int64),
        "electrode_xy_mm": electrode_xy_mm.astype(np.float32),
        "electrode_impedance_ohm": electrode_impedance_ohm,
        "electrode_grid_ids": np.asarray(electrode_grid_ids, dtype=np.int32),
        "electrode_grid_names": np.asarray([grid_name_by_id[int(grid_id)] for grid_id in electrode_grid_ids]),
        "pulse_width_s": np.asarray(pulse_width_s, dtype=np.float32),
        "relative_stim_duration": np.asarray(relative_stim_duration, dtype=np.float32),
        "electrode_surface_area_cm2": np.asarray(electrode_area_cm2, dtype=np.float32),
    }
    if track_electrical:
        common_metrics.update(
            {
                "amplitude_per_electrode_uA": np.asarray(
                    tensor_rows_to_numpy(amplitude_per_electrode_uA),
                    dtype=np.float32,
                ),
                "charge_per_phase_per_electrode_nC": np.asarray(
                    tensor_rows_to_numpy(charge_per_phase_per_electrode_nC),
                    dtype=np.float32,
                ),
                "charge_density_per_electrode_uc_cm2": np.asarray(
                    tensor_rows_to_numpy(charge_density_per_electrode_uc_cm2),
                    dtype=np.float32,
                ),
                "shannon_k_per_electrode": np.asarray(
                    tensor_rows_to_numpy(shannon_k_per_electrode),
                    dtype=np.float32,
                ),
                "charge_per_second_per_electrode_nC_s": np.asarray(
                    tensor_rows_to_numpy(charge_per_second_per_electrode_nC_s),
                    dtype=np.float32,
                ),
                "window_charge_per_electrode_nC": np.asarray(
                    tensor_rows_to_numpy(window_charge_per_electrode_nC),
                    dtype=np.float32,
                ),
                "window_charge_total_nC": tensor_list_to_numpy(window_charge_total_nC),
                "protocol_charge_per_electrode_nC": scalar_tensor_to_numpy(
                    final_protocol_charge_per_electrode_nC,
                ),
                "power_per_electrode_W": np.asarray(
                    tensor_rows_to_numpy(power_per_electrode_W),
                    dtype=np.float32,
                ),
                "pulse_width_per_electrode_s": np.asarray(
                    tensor_rows_to_numpy(pulse_width_per_electrode_s),
                    dtype=np.float32,
                ),
                "pulse_frequency_per_electrode_hz": np.asarray(
                    tensor_rows_to_numpy(pulse_frequency_per_electrode_hz),
                    dtype=np.float32,
                ),
                "active_electrode_count": tensor_list_to_numpy(active_electrode_count),
                "window_exceedance_time_start_s": np.asarray(window_exceedance_time_start_s, dtype=np.float32),
                "window_exceedance_time_end_s": np.asarray(window_exceedance_time_end_s, dtype=np.float32),
                "window_exceedance_scope": np.asarray(window_exceedance_scope, dtype="<U16"),
                "window_exceedance_electrode_id": np.asarray(window_exceedance_electrode_id, dtype=np.int64),
                "window_exceedance_charge_nC": np.asarray(window_exceedance_charge_nC, dtype=np.float32),
                "window_exceedance_limit_nC": np.asarray(window_exceedance_limit_nC, dtype=np.float32),
                "raster_active_group": np.asarray(raster_active_group, dtype=np.int32),
                "raster_group_assignment_frame_indices": np.asarray(
                    raster_group_assignment_frame_indices,
                    dtype=np.int32,
                ),
                "raster_group_assignment_times_s": np.asarray(
                    raster_group_assignment_times_s,
                    dtype=np.float32,
                ),
                "raster_group_assignments": np.asarray(
                    tensor_rows_to_numpy(raster_group_assignments, dtype=np.int32),
                    dtype=np.int32,
                ),
                "charge_window_s": np.asarray(float(safety_tracker.charge_window_s), dtype=np.float32),
            }
        )

    for ic_heat_mode, state in mode_states.items():
        if any(grid_state["prev_dT_map"] is None for grid_state in state["grid_heat_states"].values()):
            raise RuntimeError(f"No temperature results were produced for IC mode '{ic_heat_mode}'.")

        thermal_grids = {}
        for grid_name, grid_state in state["grid_heat_states"].items():
            bio = grid_state["bio"]
            dT_final = grid_state["prev_dT_map"].detach().cpu().numpy().astype(np.float32)
            final_peak_dT_C = float(np.max(dT_final)) if dT_final.size else float("nan")
            grid_payload = {
                "dT_final": dT_final,
                "extent_mm": bio.extent_mm,
                "voxel_size_mm": bio.voxel_size_mm,
                "bioheat_model": str(getattr(bio, "model", "bioheat")),
                "ic_footprint_pixel_count": bio.ic_footprint_pixel_count,
                "ic_power_density_W_m3": bio.ic_power_density_W_m3,
                "final_peak_dT_C": final_peak_dT_C,
                "final_peak_temperature_C": float(grid_state["baseline_temp"]) + final_peak_dT_C,
            }
            if enable_cem43:
                grid_payload["cem43_final"] = (
                    thermal_projection(grid_state["cem43_map"]).detach().cpu().numpy().astype(np.float32)
                )
            thermal_grids[grid_name] = grid_payload

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
            "thermal_time_s": np.asarray(state["thermal_time_s"], dtype=np.float32),
            "max_dT": tensor_list_to_numpy(state["max_dT"]),
            "mean_dT": tensor_list_to_numpy(state["mean_dT"]),
            "area_gt1_mm2": tensor_list_to_numpy(state["area_gt1_mm2"]),
            "area_gt2_mm2": tensor_list_to_numpy(state["area_gt2_mm2"]),
            "area_gt3_mm2": tensor_list_to_numpy(state["area_gt3_mm2"]),
            "dT_final": thermal_grids[reference_grid_name]["dT_final"],
            "device_power_time_s": np.asarray(state["device_power_time_s"], dtype=np.float32),
            "internal_circuit_power_W": np.asarray(state["internal_circuit_power_W"], dtype=np.float32),
            "electrode_load_power_W": np.asarray(state["electrode_load_power_W"], dtype=np.float32),
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
            electrode_heat_enabled=electrode_heat_enabled,
        )
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
            "The script looks for side-by-side files named like <video_stem>_canny.mp4."
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
        "--device-constant-power-mw",
        "--device_constant_power_mw",
        dest="device_constant_power_mw",
        type=float,
        default=None,
        help=(
            "Constant per-grid device power in mW. Defaults to "
            "bioheat.device_constant_power_mw from params.yaml."
        ),
    )
    ap.add_argument(
        "--appearance-threshold-uA",
        "--appearance_threshold_uA",
        dest="appearance_threshold_uA",
        type=float,
        default=None,
        help=(
            "Optional current threshold in microamps used to gate delivered stimulation. "
            "When omitted, thresholding.rheobase is used."
        ),
    )
    ap.add_argument(
        "--preview_seconds",
        type=float,
        default=15.0,
        help="Duration of the phosphene preview clip saved for each raster mode.",
    )
    ap.add_argument(
        "--save_every_n_frames",
        type=int,
        default=None,
        help=(
            "Output metadata value recorded as save_every_n_frames. Electrical safety arrays "
            "are saved per frame."
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
            "Maximum thermal-only cooldown batch size in video frames. Stimulation bioheat "
            "updates every video frame using that frame's per-electrode "
            "dissipated load power."
        ),
    )
    ap.add_argument(
        "--cooldown-seconds",
        type=float,
        default=300.0,
        help=(
            "After the input video ends, continue thermal-only cooling with "
            "all device power off for up to this many seconds."
        ),
    )
    ap.add_argument(
        "--cooldown-baseline-tolerance-C",
        "--cooldown-baseline-tolerance-c",
        dest="cooldown_baseline_tolerance_C",
        type=float,
        default=1e-3,
        help=(
            "Stop post-video cooling early when the peak temperature rise "
            "across all thermal grids is at or below this value."
        ),
    )
    ap.add_argument(
        "--enable-cem43",
        action="store_true",
        help="Enable CEM43 thermal dose maps and metrics. Disabled by default to keep analysis runs lighter.",
    )
    ap.add_argument("--force-cpu", action="store_true")
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
    device_constant_power_mw = resolve_device_constant_power_mw(
        params,
        args.device_constant_power_mw,
    )
    params.setdefault("bioheat", {})
    params["bioheat"]["device_constant_power_mw"] = float(device_constant_power_mw)
    params["bioheat"]["internal_circuit_power_mw"] = float(device_constant_power_mw)
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
    print(f"device constant power for IC heat mode: {device_constant_power_mw:.6f} mW per grid")

    # Respect the project GPU setting when available, but fall back to CPU cleanly.
    device = configure_runtime_device(params, force_cpu=bool(args.force_cpu))

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
                write_execution_manifest(
                    out_dir,
                    video_path=video_path,
                    params_path=params_path,
                    safety_path=safety_path,
                    coords_yaml=coords_yaml,
                    preprocessing_method=preprocessing_method,
                    ic_heat_mode=ic_heat_mode,
                    raster_name=raster_name,
                    appearance_threshold_uA=args.appearance_threshold_uA,
                    internal_circuit_power_mw=device_constant_power_mw if ic_heat_mode == "with" else 0.0,
                    stim_scale=args.stim_scale,
                    stimulus_scale_base=base_stimulus_scale,
                    stimulus_scale_effective=effective_stimulus_scale,
                    phosphene_mode=phosphene_mode,
                    cooldown_seconds=float(args.cooldown_seconds),
                    cooldown_baseline_tolerance_C=float(args.cooldown_baseline_tolerance_C),
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
                internal_circuit_power_mw=float(device_constant_power_mw),
                preview_seconds=float(args.preview_seconds),
                save_every_n_frames=int(params["safety"]["metrics_save_every_n_frames"]),
                enable_cem43=bool(args.enable_cem43),
                device=device,
                phosphene_mode=phosphene_mode,
                thermal_update_interval_frames=args.thermal_update_interval_frames,
                cooldown_seconds=float(args.cooldown_seconds),
                cooldown_baseline_tolerance_C=float(args.cooldown_baseline_tolerance_C),
                appearance_threshold_uA=args.appearance_threshold_uA,
            )
            for out_dir in mode_out_dirs.values():
                print(f"saved safety analysis to: {out_dir}")


if __name__ == "__main__":
    main()
