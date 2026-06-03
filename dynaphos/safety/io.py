from __future__ import annotations

import argparse
import copy
import gc
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos.safety import runner as simulation_runner


DEFAULT_PARAMS = "config/params.yaml"
DEFAULT_SAFETY = "config/safety.yaml"
DEFAULT_OUTPUT_ROOT = "results/safety/simulation_pipeline"
DEFAULT_VISUALS_ROOT = "results/safety/simulation_pipeline/visuals"
GROUNDTRUTH_VIDEO = "videos/SANPO/SANPOgt30min.mp4"
DOG_VIDEO = "videos/preprocessed/SANPO/dog/preprocessed.mp4"
CANNY_VIDEO = "videos/preprocessed/SANPO/canny/preprocessed.mp4"
STANDARD_GRID = "config/coords_800um.yaml"
FULL_FOV_GRIDS = (
    "config/coords_400um.yaml",
    "config/coords_800um.yaml",
    "config/coords_1200um.yaml",
)

STANDARD_AMPLITUDE_UA = 60.0
STANDARD_FREQUENCY_HZ = 300.0
STANDARD_PULSE_WIDTH_US = 170.0
STANDARD_APPEARANCE_THRESHOLD_UA = 30.0

RASTER_MODE_ALIASES = {
    "pseudorandom": "random",
    "pseudo-random": "random",
    "pseudo_random": "random",
}


def normalize_raster_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace(" ", "_")
    return RASTER_MODE_ALIASES.get(normalized, normalized)


@dataclass(frozen=True)
class SimulationCase:
    block: str
    run_id: str
    video: str
    coords_yaml: str
    preprocessing_method: str
    amplitude_uA: float = STANDARD_AMPLITUDE_UA
    frequency_hz: float = STANDARD_FREQUENCY_HZ
    pulse_width_us: float = STANDARD_PULSE_WIDTH_US
    raster_mode: str = "none"
    appearance_threshold_uA: float | None = None
    source_input_label: str = "SANPO_groundtruth"
    internal_circuit_power_mw: float = 0.0
    ic_heat_mode: str = "without"
    metadata: dict[str, object] = field(default_factory=dict)


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def load_yaml(path: str | Path) -> dict:
    with open(resolve_repo_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sanitize_path_part(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return safe.strip("._") or "item"


def configure_device(params: dict, *, force_cpu: bool = False) -> torch.device:
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


def build_params_for_case(base_params: dict, case: SimulationCase, safety_yaml: Path) -> tuple[dict, float]:
    params = copy.deepcopy(base_params)
    params.setdefault("sampling", {})["stimulus_scale"] = float(case.amplitude_uA) * 1e-6
    params.setdefault("default_stim", {})["freq_default"] = float(case.frequency_hz)
    params["default_stim"]["pw_default"] = float(case.pulse_width_us) * 1e-6
    params.setdefault("safety", {})["guidelines_path"] = str(safety_yaml)
    return params, float(params["sampling"]["stimulus_scale"])


def write_manifest(
    out_dir: Path,
    *,
    case: SimulationCase,
    params_path: Path,
    safety_yaml: Path,
    output_root: Path,
    max_frames: int,
    groups: int,
    phosphene_mode: str,
    implant_off_tail_seconds: float,
    enable_cem43: bool,
) -> None:
    manifest = {
        "block": case.block,
        "run_id": case.run_id,
        "video": str(resolve_repo_path(case.video)),
        "coords_yaml": str(resolve_repo_path(case.coords_yaml)),
        "preprocessing_method": case.preprocessing_method,
        "amplitude_uA": float(case.amplitude_uA),
        "frequency_hz": float(case.frequency_hz),
        "pulse_width_us": float(case.pulse_width_us),
        "raster_mode": case.raster_mode,
        "raster_mode_normalized": normalize_raster_mode(case.raster_mode),
        "raster_groups": int(groups),
        "appearance_threshold_uA": (
            None if case.appearance_threshold_uA is None else float(case.appearance_threshold_uA)
        ),
        "source_input_label": case.source_input_label,
        "internal_circuit_power_mw": float(case.internal_circuit_power_mw),
        "ic_heat_mode": case.ic_heat_mode,
        "params": str(params_path),
        "safety_yaml": str(safety_yaml),
        "output_root": str(output_root),
        "max_frames": int(max_frames),
        "metadata": dict(case.metadata),
        "phosphene_mode": str(phosphene_mode),
        "implant_off_tail_seconds": float(implant_off_tail_seconds),
        "enable_cem43": bool(enable_cem43),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_manifest.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(manifest, handle, sort_keys=False)


def add_common_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--safety-yaml", default=DEFAULT_SAFETY)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--visuals-root", default=DEFAULT_VISUALS_ROOT)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = full video")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument(
        "--phosphene-mode",
        choices=("safety_centers", "visual"),
        default="safety_centers",
        help=(
            "safety_centers skips full phosphene maps and samples center pixels; "
            "visual builds full maps for phosphene previews."
        ),
    )
    parser.add_argument("--save-every-n-frames", type=int, default=None)
    parser.add_argument(
        "--sim-resolution",
        type=int,
        default=None,
        help="Override simulator width/height in pixels for memory-heavy runs, e.g. 128.",
    )
    parser.add_argument(
        "--thermal-update-interval-frames",
        type=int,
        default=None,
        help=(
            "Maximum cooldown batch size in video frames. Stimulation bioheat "
            "updates every video frame."
        ),
    )
    parser.add_argument(
        "--implant-off-tail-seconds",
        type=float,
        default=0.0,
        help=(
            "Treat the final N seconds as implant-off cooldown: skip video "
            "decoding and electrical logging while advancing bioheat with zero "
            "device power."
        ),
    )
    parser.add_argument("--preview-seconds", type=float, default=0.0)
    parser.add_argument(
        "--preview-policy",
        choices=("all", "none", "worst-case", "standard", "matrix-representative"),
        default="all",
        help=(
            "Choose which cases write phosphene preview MP4s. This does not "
            "change simulation or saved safety metrics. matrix-representative "
            "renders the standard-amplitude grid/preprocessing plane plus a "
            "single-grid amplitude check."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Run independent simulation cases in parallel. Keep at 1 for a "
            "single GPU unless benchmarking shows spare memory/CPU capacity."
        ),
    )
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument(
        "--enable-cem43",
        action="store_true",
        help="Compute CEM43 thermal-dose maps and metrics. Disabled by default for faster safety sweeps.",
    )
    parser.add_argument("--dry-run", action="store_true", help="List planned runs without simulating.")


def iter_case_lines(cases: Iterable[SimulationCase]) -> Iterable[str]:
    for index, case in enumerate(cases, start=1):
        yield (
            f"{index:03d} | {case.block} | {case.run_id} | "
            f"grid={Path(case.coords_yaml).stem} | video={Path(case.video).name} | "
            f"amp={case.amplitude_uA:g}uA freq={case.frequency_hz:g}Hz "
            f"pw={case.pulse_width_us:g}us threshold="
            f"{'default' if case.appearance_threshold_uA is None else f'{case.appearance_threshold_uA:g}uA'} "
            f"raster={case.raster_mode} "
            f"ic={case.internal_circuit_power_mw:g}mW"
        )


def should_write_preview(case: SimulationCase, preview_policy: str, preview_seconds: float) -> bool:
    if float(preview_seconds) <= 0.0:
        return False
    policy = str(preview_policy).strip().lower()
    if policy == "all":
        return True
    if policy == "none":
        return False
    sweep = str((case.metadata or {}).get("sweep", "")).strip().lower()
    if policy == "worst-case":
        return sweep == "worst_case"
    if policy == "standard":
        return (
            float(case.amplitude_uA) == STANDARD_AMPLITUDE_UA
            and float(case.frequency_hz) == STANDARD_FREQUENCY_HZ
            and float(case.pulse_width_us) == STANDARD_PULSE_WIDTH_US
        )
    if policy == "matrix-representative":
        is_standard_protocol = (
            float(case.frequency_hz) == STANDARD_FREQUENCY_HZ
            and float(case.pulse_width_us) == STANDARD_PULSE_WIDTH_US
        )
        if not is_standard_protocol:
            return False

        is_standard_amplitude = float(case.amplitude_uA) == STANDARD_AMPLITUDE_UA
        if case.block != "amplitude_grid_preprocessing":
            return is_standard_amplitude

        grid_label = str((case.metadata or {}).get("grid_label", Path(case.coords_yaml).stem)).strip().lower()
        source_label = str(
            (case.metadata or {}).get("preprocessing_label", case.source_input_label)
        ).strip().lower()
        preprocessing_method = str(case.preprocessing_method).strip().lower()
        is_groundtruth = source_label in {"gt", "groundtruth"} or preprocessing_method == "groundtruth"
        is_amplitude_check_grid = grid_label in {"coords_1200um", "1200um"}
        return is_standard_amplitude or (is_amplitude_check_grid and is_groundtruth)
    raise ValueError(f"Unsupported preview policy: {preview_policy}")


def run_one_case(
    index: int,
    total: int,
    case: SimulationCase,
    args_dict: dict[str, object],
) -> Path:
    params_path = resolve_repo_path(str(args_dict["params"]))
    safety_yaml = resolve_repo_path(str(args_dict["safety_yaml"]))
    output_root = resolve_repo_path(str(args_dict["output_root"]))
    visuals_root = resolve_repo_path(str(args_dict["visuals_root"]))

    base_params = load_yaml(params_path)
    device = configure_device(base_params, force_cpu=bool(args_dict["force_cpu"]))

    out_dir = output_root / case.block / sanitize_path_part(case.run_id)
    preview_dir = visuals_root / case.block / sanitize_path_part(case.run_id)
    params, stimulus_scale = build_params_for_case(base_params, case, safety_yaml)
    sim_resolution = args_dict.get("sim_resolution")
    if sim_resolution is not None:
        sim_resolution = int(sim_resolution)
        if sim_resolution <= 0:
            raise ValueError("--sim-resolution must be > 0.")
        params.setdefault("run", {})["resolution"] = [sim_resolution, sim_resolution]

    write_manifest(
        out_dir,
        case=case,
        params_path=params_path,
        safety_yaml=safety_yaml,
        output_root=output_root,
        max_frames=int(args_dict["max_frames"]),
        groups=int(args_dict["groups"]),
        phosphene_mode=str(args_dict["phosphene_mode"]),
        implant_off_tail_seconds=float(args_dict["implant_off_tail_seconds"]),
        enable_cem43=bool(args_dict["enable_cem43"]),
    )
    preview_seconds = (
        float(args_dict["preview_seconds"])
        if should_write_preview(case, str(args_dict["preview_policy"]), float(args_dict["preview_seconds"]))
        else 0.0
    )
    print(
        f"[{index}/{total}] {case.run_id}\n"
        f"  video={Path(case.video).name} | grid={Path(case.coords_yaml).stem} | "
        f"amp={case.amplitude_uA:g} uA | freq={case.frequency_hz:g} Hz | "
        f"pw={case.pulse_width_us:g} us | threshold="
        f"{'default' if case.appearance_threshold_uA is None else f'{case.appearance_threshold_uA:g} uA'} | "
        f"raster={case.raster_mode} | "
        f"ic={case.internal_circuit_power_mw:g} mW | preview={preview_seconds:g}s | "
        f"off_tail={float(args_dict['implant_off_tail_seconds']):g}s"
    )
    try:
        simulation_runner.run_one_mode(
            params=params,
            coords_yaml=resolve_repo_path(case.coords_yaml),
            video_path=resolve_repo_path(case.video),
            mode_out_dirs={case.ic_heat_mode: out_dir},
            preview_out_dirs={case.ic_heat_mode: preview_dir},
            preprocessing_method=case.preprocessing_method,
            stim_scale=None,
            stimulus_scale_base=stimulus_scale,
            raster_name=case.raster_mode,
            appearance_threshold_uA=case.appearance_threshold_uA,
            groups=int(args_dict["groups"]),
            max_frames=int(args_dict["max_frames"]),
            internal_circuit_power_mw=float(case.internal_circuit_power_mw),
            preview_seconds=preview_seconds,
            snapshot_interval_s=5.0,
            save_every_n_frames=args_dict["save_every_n_frames"],
            thermal_update_interval_frames=args_dict["thermal_update_interval_frames"],
            implant_off_tail_seconds=float(args_dict["implant_off_tail_seconds"]),
            phosphene_mode=str(args_dict["phosphene_mode"]),
            enable_cem43=bool(args_dict["enable_cem43"]),
            track_electrical=bool(args_dict.get("track_electrical", True)),
            device=device,
        )
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print(f"saved: {out_dir}")
    return out_dir


def run_cases(cases: list[SimulationCase], args: argparse.Namespace) -> None:
    params_path = resolve_repo_path(args.params)
    safety_yaml = resolve_repo_path(args.safety_yaml)
    output_root = resolve_repo_path(args.output_root)
    visuals_root = resolve_repo_path(args.visuals_root)

    if safety_yaml.name != "safety.yaml":
        raise ValueError(f"This pipeline must use config/safety.yaml, got: {safety_yaml}")

    if args.dry_run:
        preview_count = sum(
            1
            for case in cases
            if should_write_preview(case, str(args.preview_policy), float(args.preview_seconds))
        )
        print(f"Planned simulations: {len(cases)}")
        print(
            f"Planned phosphene previews: {preview_count} "
            f"(policy={args.preview_policy}, seconds={float(args.preview_seconds):g})"
        )
        print(f"Implant-off cooldown tail: {float(args.implant_off_tail_seconds):g}s")
        for line in iter_case_lines(cases):
            print(line)
        return

    args_dict = {
        "params": str(params_path),
        "safety_yaml": str(safety_yaml),
        "output_root": str(output_root),
        "visuals_root": str(visuals_root),
        "max_frames": int(args.max_frames),
        "groups": int(args.groups),
        "phosphene_mode": str(args.phosphene_mode),
        "save_every_n_frames": args.save_every_n_frames,
        "sim_resolution": args.sim_resolution,
        "thermal_update_interval_frames": args.thermal_update_interval_frames,
        "implant_off_tail_seconds": float(args.implant_off_tail_seconds),
        "preview_seconds": float(args.preview_seconds),
        "preview_policy": str(args.preview_policy),
        "force_cpu": bool(args.force_cpu),
        "enable_cem43": bool(args.enable_cem43),
        "track_electrical": bool(getattr(args, "track_electrical", True)),
    }
    workers = int(args.workers)
    if workers <= 0:
        raise ValueError("--workers must be >= 1.")

    if workers == 1:
        base_params = load_yaml(params_path)
        device = configure_device(base_params, force_cpu=bool(args.force_cpu))
        if device.type == "cuda":
            print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
        else:
            print("Using device: cpu")
        for index, case in enumerate(cases, start=1):
            run_one_case(index, len(cases), case, args_dict)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return

    print(
        f"Running {len(cases)} cases with {workers} worker processes. "
        "Use --workers 1 for a single GPU unless this has been benchmarked."
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(run_one_case, index, len(cases), case, args_dict)
            for index, case in enumerate(cases, start=1)
        ]
        for future in as_completed(futures):
            future.result()


def safety_value_to_nC(section: dict, *, default_factor: float) -> float:
    value = float((section or {}).get("value", np.inf))
    unit = str((section or {}).get("unit", "")).strip().lower().replace("µ", "u")
    if unit.startswith("uc"):
        return value * 1e3
    if unit.startswith("nc"):
        return value
    if unit.startswith("c"):
        return value * 1e9
    return value * float(default_factor)


def load_safety_limits(safety_yaml: str | Path = DEFAULT_SAFETY) -> dict[str, float]:
    cfg = load_yaml(safety_yaml)
    guidelines = cfg.get("stimulation_safety_guidelines", {}) or {}
    acc = guidelines.get("accumulated_charge", {}) or {}
    acc_limits = acc.get("limits", {}) or {}
    temp = guidelines.get("temperature", {}) or {}
    temp_inc = temp.get("temperature_increase", {}) or {}
    chronic = cfg.get("chronic", {}) or {}
    return {
        "charge_per_phase_nC": float((guidelines.get("charge_per_phase", {}) or {}).get("value", np.inf)),
        "window_charge_per_electrode_nC": safety_value_to_nC(
            acc_limits.get("per_electrode", {}) or {},
            default_factor=1.0,
        ),
        "window_charge_total_nC": safety_value_to_nC(
            acc_limits.get("total_all_electrodes", {}) or {},
            default_factor=1e3,
        ),
        "simultaneous_activation_pct": float(
            ((guidelines.get("simultaneous_activation", {}) or {}).get("max_percentage_of_electrodes", {}) or {}).get(
                "value",
                np.inf,
            )
        ),
        "temperature_increase_C": float(
            (temp_inc.get("absolute_max_temperature_increase", {}) or {}).get("value", np.inf)
        ),
        "cem43_min": float((temp.get("cem43", {}) or {}).get("value", np.inf)),
        "session_charge_limit_mC": float(chronic.get("session_charge_limit_c", np.inf)) * 1e3,
    }


def scalar_max(data: dict[str, np.ndarray], key: str, default: float = 0.0) -> float:
    if key not in data:
        return float(default)
    values = np.asarray(data[key], dtype=np.float64)
    if values.size == 0:
        return float(default)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(default)
    return float(np.max(finite))


def scalar_value(data: dict[str, np.ndarray], key: str, default: float = np.nan) -> float:
    if key not in data:
        return float(default)
    values = np.asarray(data[key])
    if values.size == 0:
        return float(default)
    try:
        return float(values.reshape(-1)[0])
    except (TypeError, ValueError):
        return float(default)


def derive_frame_charge_per_electrode_nC(data: dict[str, np.ndarray]) -> np.ndarray | None:
    if "frame_charge_per_electrode_nC" in data:
        return np.asarray(data["frame_charge_per_electrode_nC"], dtype=np.float64)
    amplitude_key = (
        "amplitude_per_electrode_uA"
        if "amplitude_per_electrode_uA" in data
        else "current_amplitude_per_electrode_uA"
    )
    required = (amplitude_key, "pulse_width_s", "pulse_frequency_hz", "fps")
    if any(key not in data for key in required):
        return None
    amplitude = np.asarray(data[amplitude_key], dtype=np.float64)
    pulse_width = np.asarray(data["pulse_width_s"], dtype=np.float64).reshape(-1)
    frequency = np.asarray(data["pulse_frequency_hz"], dtype=np.float64).reshape(-1)
    fps = scalar_value(data, "fps")
    if amplitude.ndim != 2 or amplitude.shape[1] != pulse_width.size or pulse_width.size != frequency.size:
        return None
    if not np.isfinite(fps) or fps <= 0.0:
        return None
    relative_stim_duration = scalar_value(data, "relative_stim_duration", 1.0)
    if not np.isfinite(relative_stim_duration):
        relative_stim_duration = 1.0
    return (
        2.0
        * amplitude
        * pulse_width.reshape(1, -1)
        * frequency.reshape(1, -1)
        * (1.0 / fps)
        * relative_stim_duration
        * 1e3
    )


def max_rolling_charge_per_electrode_nC(data: dict[str, np.ndarray], window_s: float) -> float:
    frame_charge = derive_frame_charge_per_electrode_nC(data)
    if frame_charge is None or frame_charge.ndim != 2 or frame_charge.size == 0:
        return 0.0
    fps = scalar_value(data, "fps")
    if not np.isfinite(fps) or fps <= 0.0 or not np.isfinite(window_s) or window_s <= 0.0:
        return 0.0
    window_frames = max(1, int(round(window_s * fps)))
    cumulative = np.cumsum(frame_charge, axis=0, dtype=np.float64)
    rolling = cumulative.copy()
    if window_frames < frame_charge.shape[0]:
        rolling[window_frames:] = cumulative[window_frames:] - cumulative[:-window_frames]
    return float(np.nanmax(rolling))


def score_safety_metrics(npz_path: Path, limits: dict[str, float]) -> tuple[float, dict[str, float]]:
    with np.load(npz_path, allow_pickle=True) as bundle:
        data = {key: bundle[key] for key in bundle.files}
    electrode_count = max(1, int(np.asarray(data.get("electrode_grid_ids", [])).size))
    window_charge_per_electrode_nC = (
        scalar_max(data, "window_charge_per_electrode_nC")
        if "window_charge_per_electrode_nC" in data
        else max_rolling_charge_per_electrode_nC(data, scalar_value(data, "charge_window_s", 5.0))
    )
    if "amplitude_per_electrode_uA" in data:
        amplitude = np.asarray(data["amplitude_per_electrode_uA"], dtype=np.float64)
    else:
        amplitude = np.asarray(data.get("current_amplitude_per_electrode_uA", []), dtype=np.float64)
    if amplitude.ndim == 2:
        active_count = float(np.nanmax(np.sum(amplitude > 0.0, axis=1))) if amplitude.size else 0.0
    elif amplitude.ndim == 1 and amplitude.size:
        active_count = float(np.count_nonzero(amplitude > 0.0))
    else:
        active_count = scalar_max(data, "active_count")
    ratios = {
        "charge_per_phase": (
            scalar_max(data, "charge_per_phase_per_electrode_nC")
            if "charge_per_phase_per_electrode_nC" in data
            else scalar_max(data, "peak_charge_per_phase_nC_exact")
        ) / limits["charge_per_phase_nC"],
        "window_charge_per_electrode": window_charge_per_electrode_nC / limits["window_charge_per_electrode_nC"],
        "window_charge_total": scalar_max(data, "window_charge_total_nC") / limits["window_charge_total_nC"],
        "active_percentage": (
            100.0 * active_count / float(electrode_count)
        ) / limits["simultaneous_activation_pct"],
        "temperature": scalar_max(data, "max_dT") / limits["temperature_increase_C"],
        "cem43": scalar_max(data, "max_cem43") / limits["cem43_min"],
    }
    finite_ratios = [value for value in ratios.values() if np.isfinite(value)]
    return (max(finite_ratios) if finite_ratios else 0.0), ratios


def select_worst_prior_run(output_root: str | Path, safety_yaml: str | Path) -> tuple[Path, float, dict[str, float]]:
    root = resolve_repo_path(output_root)
    limits = load_safety_limits(safety_yaml)
    candidates = sorted(
        path for path in root.rglob("safety_metrics.npz")
        if "rastering_rerun" not in path.parts
    )
    if not candidates:
        raise RuntimeError(f"No prior safety_metrics.npz files found under: {root}")

    scored = [(path, *score_safety_metrics(path, limits)) for path in candidates]
    return max(scored, key=lambda item: item[1])


def load_case_from_manifest(manifest_path: Path, *, block: str, run_id: str) -> SimulationCase:
    manifest = load_yaml(manifest_path)
    return SimulationCase(
        block=block,
        run_id=run_id,
        video=str(manifest["video"]),
        coords_yaml=str(manifest["coords_yaml"]),
        preprocessing_method=str(manifest.get("preprocessing_method", "groundtruth")),
        amplitude_uA=float(manifest["amplitude_uA"]),
        frequency_hz=float(manifest["frequency_hz"]),
        pulse_width_us=float(manifest["pulse_width_us"]),
        raster_mode="random",
        appearance_threshold_uA=(
            None
            if manifest.get("appearance_threshold_uA", None) is None
            else float(manifest["appearance_threshold_uA"])
        ),
        source_input_label=str(manifest.get("source_input_label", manifest.get("preprocessing_method", "groundtruth"))),
        internal_circuit_power_mw=float(manifest.get("internal_circuit_power_mw", 0.0)),
        ic_heat_mode=str(manifest.get("ic_heat_mode", "without")),
        metadata={"source_manifest": str(manifest_path)},
    )
