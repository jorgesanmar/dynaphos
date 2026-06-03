from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos import utils
from dynaphos.simulator import Bioheat2D, GaussianSimulator, Impedance
from tools.safety import simulation_runner


DEFAULT_PARAMS = "config/params.yaml"
DEFAULT_THERMAL_UPDATE_INTERVAL_FRAMES = 15
THERMAL_PREFIXES = (
    "max_dT",
    "mean_dT",
    "area_gt1_mm2",
    "area_gt2_mm2",
    "area_gt3_mm2",
    "dT_final",
    "extent_mm",
    "stationary_temperature_C",
    "stationary_dT_C",
    "max_cem43",
    "p99_cem43",
    "cem43_final",
    "cem43_extent_mm",
    "cem43_heatmaps",
    "dT_heatmaps",
    "heatmap_",
    "internal_circuit_footprint_pixels",
    "internal_circuit_power_density_W_m3",
)
THERMAL_KEYS = {
    "extent_mm",
    "thermal_grid_names",
    "dT_final_reference_grid",
    "cem43_final_reference_grid",
    "internal_circuit_power_total_mW",
}


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def load_yaml(path: str | Path) -> dict:
    with open(resolve_repo_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def configure_device(params: dict, *, force_cpu: bool) -> torch.device:
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
        print(f"Warning: CUDA failed a tensor smoke test; using CPU. CUDA error: {exc}")
        params.setdefault("run", {})["gpu"] = None
        return torch.device("cpu")


def discover_npz_inputs(input_path: Path) -> list[Path]:
    input_path = input_path.resolve()
    if input_path.is_file():
        if input_path.name != "safety_metrics.npz":
            raise ValueError(f"Expected a safety_metrics.npz file, got: {input_path}")
        return [input_path]
    if input_path.is_dir():
        paths = sorted(input_path.rglob("safety_metrics.npz"))
        if not paths:
            raise RuntimeError(f"No safety_metrics.npz files found under: {input_path}")
        return paths
    raise FileNotFoundError(input_path)


def manifest_for_npz(npz_path: Path) -> Path | None:
    manifest = npz_path.with_name("run_manifest.yaml")
    return manifest if manifest.exists() else None


def resolve_coords_yaml(npz_path: Path, fallback: str | None) -> Path:
    manifest = manifest_for_npz(npz_path)
    if manifest is not None:
        data = load_yaml(manifest)
        coords = data.get("coords_yaml")
        if coords:
            return resolve_repo_path(coords)
    if fallback:
        return resolve_repo_path(fallback)
    raise RuntimeError(
        f"Could not resolve coords_yaml for {npz_path}. "
        "Add a run_manifest.yaml next to it or pass --coords-yaml."
    )


def load_run_manifest(npz_path: Path) -> dict:
    manifest = manifest_for_npz(npz_path)
    return load_yaml(manifest) if manifest is not None else {}


def infer_ic_power_mw(npz_data: dict[str, np.ndarray]) -> float:
    if "internal_circuit_power_total_mW" not in npz_data:
        return 0.0
    value = np.asarray(npz_data["internal_circuit_power_total_mW"], dtype=np.float64)
    if value.size == 0:
        return 0.0
    return float(value.reshape(-1)[0])


def grid_name_for_id(grid_id: int, existing_names: np.ndarray | None) -> str:
    _ = existing_names
    return "right" if int(grid_id) == 0 else "left"


def sanitize_path_part(value: object) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in text)
    return safe.strip("._") or "item"


def short_progress_label(value: object, max_len: int = 48) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    keep = max(8, max_len - 3)
    return "..." + text[-keep:]


def build_electrode_context(
    params: dict,
    coords_yaml: Path,
    n_electrodes: int,
    device: torch.device,
    *,
    npz_data: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if npz_data is not None and "electrode_xy_mm" in npz_data:
        elec_xy_mm = np.asarray(npz_data["electrode_xy_mm"], dtype=np.float64)
        if elec_xy_mm.shape == (int(n_electrodes), 2):
            electrode_ids = np.asarray(
                npz_data.get("electrode_ids", np.arange(int(n_electrodes), dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1)
            if electrode_ids.size != int(n_electrodes):
                electrode_ids = np.arange(int(n_electrodes), dtype=np.int64)
            if "electrode_impedance_ohm" in npz_data:
                impedance = np.asarray(npz_data["electrode_impedance_ohm"], dtype=np.float32).reshape(-1)
                if impedance.size == int(n_electrodes):
                    return electrode_ids, elec_xy_mm, impedance
            impedance = Impedance(
                params,
                (int(n_electrodes), 1, 1),
                rng=np.random.default_rng(int(params.get("run", {}).get("seed", 42))),
                verbose=False,
            )
            return electrode_ids, elec_xy_mm, impedance.state.reshape(-1).detach().cpu().numpy().astype(np.float32)

    sim_mismatch: RuntimeError | None = None
    mapping = simulation_runner.load_tagged_mapping(params, coords_yaml)
    sim = GaussianSimulator(
        params,
        mapping["phosphene_map"],
        raster_coordinates=mapping["cortical_coordinates"],
        raster_enabled=False,
        raster_pattern="random",
        raster_num_groups=1,
        raster_rate_hz=0.0,
        phosphene_mode="safety_centers",
    )
    simulation_runner.move_simulator_tensors_to_device(sim, device)
    aligned_mapping = simulation_runner.align_mapping_to_simulator(mapping, sim)
    elec_xy_mm, inv_map_t = simulation_runner.build_surviving_electrode_set(
        aligned_mapping["cortical_coordinates"],
        aligned_mapping["indices"],
        device=device,
    )
    if elec_xy_mm.shape[0] == int(n_electrodes):
        electrode_ids, _ = simulation_runner.build_surviving_electrode_metadata(
            aligned_mapping["cortical_coordinates"],
            aligned_mapping["indices"],
        )
        impedance_phos = sim.impedance.state.reshape(-1).to(device=device, dtype=torch.float32)
        impedance_sum = torch.zeros(int(n_electrodes), dtype=torch.float32, device=device)
        counts = torch.zeros(int(n_electrodes), dtype=torch.float32, device=device)
        impedance_sum.scatter_add_(0, inv_map_t, impedance_phos)
        counts.scatter_add_(0, inv_map_t, torch.ones_like(impedance_phos))
        impedance_elec = impedance_sum / counts.clamp_min(1.0)
        return electrode_ids, elec_xy_mm, impedance_elec.detach().cpu().numpy().astype(np.float32)

    sim_mismatch = RuntimeError(
        f"Amplitude column count ({n_electrodes}) does not match reconstructed "
        f"surviving-electrode count ({elec_xy_mm.shape[0]}) for {coords_yaml}."
    )

    x_raw_mm, y_raw_mm = utils.load_coordinates_from_yaml(str(coords_yaml))
    x_raw_mm = np.asarray(x_raw_mm, dtype=np.float64)
    y_raw_mm = np.asarray(y_raw_mm, dtype=np.float64)
    if x_raw_mm.size > 0:
        electrode_ids, raw_xy_mm = build_legacy_ids_xy_from_coords(
            x_raw_mm,
            y_raw_mm,
            int(n_electrodes),
            None if npz_data is None else npz_data.get("electrode_grid_ids"),
        )
        if raw_xy_mm is None or electrode_ids is None:
            raise RuntimeError(
                f"{sim_mismatch} Raw coordinate count is {x_raw_mm.size}, so the script "
                "cannot safely align saved amplitudes to electrode coordinates."
            )
        print(
            "  warning: electrode_xy_mm was not stored in this npz; using coords-yaml "
            "legacy ordering fallback for electrode locations.",
            flush=True,
        )
        impedance = Impedance(
            params,
            (int(n_electrodes), 1, 1),
            rng=np.random.default_rng(int(params.get("run", {}).get("seed", 42))),
            verbose=False,
        )
        impedance_elec = impedance.state.reshape(-1).detach().cpu().numpy().astype(np.float32)
        return electrode_ids, raw_xy_mm, impedance_elec

    raise RuntimeError(
        f"{sim_mismatch} Raw coordinate count is {x_raw_mm.size}, so the script "
        "cannot safely align saved amplitudes to electrode coordinates."
    )


def sort_raw_coords_like_pipeline(x_raw_mm: np.ndarray, y_raw_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sort_idx = np.lexsort((np.asarray(x_raw_mm, dtype=np.float64), -np.asarray(y_raw_mm, dtype=np.float64)))
    return np.asarray(x_raw_mm, dtype=np.float64)[sort_idx], np.asarray(y_raw_mm, dtype=np.float64)[sort_idx]


def build_legacy_ids_xy_from_coords(
    x_raw_mm: np.ndarray,
    y_raw_mm: np.ndarray,
    n_electrodes: int,
    electrode_grid_ids: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    x_sorted, y_sorted = sort_raw_coords_like_pipeline(x_raw_mm, y_raw_mm)
    n_base = int(x_sorted.size)
    if n_base <= 0:
        return None, None

    if n_electrodes == n_base:
        ids = np.arange(n_base, dtype=np.int64)
        xy = np.stack([x_sorted, y_sorted], axis=1).astype(np.float64)
        return ids, xy

    if electrode_grid_ids is not None:
        grid_ids = np.asarray(electrode_grid_ids, dtype=np.int32).reshape(-1)
        if grid_ids.size == n_electrodes:
            parts = []
            for grid_id in np.unique(grid_ids):
                count = int(np.sum(grid_ids == int(grid_id)))
                if count > n_base:
                    return None, None
                x_grid = x_sorted[:count]
                y_grid = y_sorted[:count]
                ids = np.arange(count, dtype=np.int64)
                if int(grid_id) != 0:
                    x_grid = -x_grid
                    ids = ids + n_base
                parts.append((grid_id, ids, np.stack([x_grid, y_grid], axis=1)))
            ids_by_grid = {int(grid_id): ids for grid_id, ids, _ in parts}
            xy_by_grid = {int(grid_id): xy for grid_id, _, xy in parts}
            cursors = {int(grid_id): 0 for grid_id in xy_by_grid}
            out_ids = np.zeros(n_electrodes, dtype=np.int64)
            out = np.zeros((n_electrodes, 2), dtype=np.float64)
            for idx, grid_id in enumerate(grid_ids):
                gid = int(grid_id)
                pos = cursors[gid]
                out_ids[idx] = ids_by_grid[gid][pos]
                out[idx] = xy_by_grid[gid][pos]
                cursors[gid] += 1
            return out_ids, out

    if n_electrodes <= 2 * n_base:
        right_count = min(n_base, n_electrodes)
        left_count = max(0, n_electrodes - right_count)
        x_parts = [x_sorted[:right_count]]
        y_parts = [y_sorted[:right_count]]
        id_parts = [np.arange(right_count, dtype=np.int64)]
        if left_count:
            x_parts.append(-x_sorted[:left_count])
            y_parts.append(y_sorted[:left_count])
            id_parts.append(np.arange(left_count, dtype=np.int64) + n_base)
        ids = np.concatenate(id_parts)
        xy = np.stack([np.concatenate(x_parts), np.concatenate(y_parts)], axis=1).astype(np.float64)
        return ids, xy

    return None, None


def read_scalar(data: dict[str, np.ndarray], key: str, default: float) -> float:
    if key not in data:
        return float(default)
    values = np.asarray(data[key])
    if values.size == 0:
        return float(default)
    try:
        return float(values.reshape(-1)[0])
    except (TypeError, ValueError):
        return float(default)


def read_vector(
    data: dict[str, np.ndarray],
    key: str,
    n_electrodes: int,
    default: float,
) -> np.ndarray:
    if key not in data:
        return np.full(n_electrodes, float(default), dtype=np.float32)
    values = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if values.size == 1:
        return np.full(n_electrodes, float(values[0]), dtype=np.float32)
    if values.size != n_electrodes:
        raise RuntimeError(
            f"Expected {key} to have 1 or {n_electrodes} values, got {values.size}."
        )
    return values.astype(np.float32, copy=False)


def cem43_weight_from_temperature(temp_c: torch.Tensor) -> torch.Tensor:
    return simulation_runner.cem43_weight_from_temperature(temp_c)


def percentile_tensor(values: list[torch.Tensor], percentile: float, device: torch.device) -> torch.Tensor:
    return simulation_runner.percentile_tensor_from_tensor_list(values, percentile, device)


def thermal_key_is_stale(key: str) -> bool:
    return key in THERMAL_KEYS or any(key == prefix or key.startswith(prefix + "_") for prefix in THERMAL_PREFIXES)


def recompute_temperature(
    npz_data: dict[str, np.ndarray],
    params: dict,
    coords_yaml: Path,
    device: torch.device,
    *,
    manifest: dict,
    thermal_update_interval_frames: int,
    enable_cem43: bool,
    progress_label: str | None = None,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    if "current_amplitude_per_electrode_uA" not in npz_data:
        raise RuntimeError("Missing current_amplitude_per_electrode_uA.")
    amplitude_uA = np.asarray(npz_data["current_amplitude_per_electrode_uA"], dtype=np.float32)
    if amplitude_uA.ndim != 2 or amplitude_uA.shape[0] == 0 or amplitude_uA.shape[1] == 0:
        raise RuntimeError(
            "current_amplitude_per_electrode_uA must be a non-empty 2D array "
            f"(got shape {amplitude_uA.shape})."
        )
    if thermal_update_interval_frames <= 0:
        raise ValueError("--thermal-update-interval-frames must be >= 1.")

    frame_count, n_electrodes = amplitude_uA.shape
    run_params = yaml.safe_load(yaml.safe_dump(params))
    fps = read_scalar(npz_data, "fps", run_params.get("run", {}).get("fps", 20.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise RuntimeError(f"Invalid fps value: {fps}")
    run_params.setdefault("run", {})["fps"] = float(fps)

    pulse_width_s = read_vector(
        npz_data,
        "pulse_width_s",
        n_electrodes,
        float(manifest.get("pulse_width_us", float(run_params.get("default_stim", {}).get("pw_default", 170e-6)) * 1e6))
        * 1e-6,
    )
    frequency_hz = read_vector(
        npz_data,
        "pulse_frequency_hz",
        n_electrodes,
        float(manifest.get("frequency_hz", run_params.get("default_stim", {}).get("freq_default", 300.0))),
    )
    relative_stim_duration = read_scalar(
        npz_data,
        "relative_stim_duration",
        float(run_params.get("default_stim", {}).get("relative_stim_duration", 1.0)),
    )

    electrode_ids, elec_xy_mm, impedance_ohm = build_electrode_context(
        run_params,
        coords_yaml,
        n_electrodes,
        device,
        npz_data=npz_data,
    )
    electrode_grid_ids = np.asarray(
        npz_data.get("electrode_grid_ids", np.zeros(n_electrodes, dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)
    if electrode_grid_ids.size != n_electrodes:
        electrode_grid_ids = np.zeros(n_electrodes, dtype=np.int32)
    existing_grid_names = (
        np.asarray(npz_data["electrode_grid_names"])
        if "electrode_grid_names" in npz_data
        else None
    )

    ic_power_total_mw = infer_ic_power_mw(npz_data)
    unique_grid_ids = np.unique(electrode_grid_ids)
    ic_power_per_grid_mw = ic_power_total_mw / max(len(unique_grid_ids), 1)
    dt = 1.0 / float(fps)
    duty = 2.0 * pulse_width_s * frequency_hz * float(relative_stim_duration)

    grid_states: dict[str, dict[str, object]] = {}
    for grid_id in unique_grid_ids:
        mask_np = electrode_grid_ids == int(grid_id)
        grid_name = grid_name_for_id(int(grid_id), existing_grid_names)
        grid_params = yaml.safe_load(yaml.safe_dump(run_params))
        grid_params.setdefault("bioheat", {})["internal_circuit_power_mw"] = float(ic_power_per_grid_mw)
        bio = Bioheat2D(params=grid_params, elec_xy_mm=elec_xy_mm[mask_np], device=device)
        grid_states[grid_name] = {
            "mask_np": mask_np,
            "bio": bio,
            "baseline_temp": float(bio.baseline_T),
            "pixel_area_mm2": float(bio.dx * bio.dy) * 1e6,
            "prev_dT": torch.zeros_like(bio.T),
            "max_dT": [],
            "mean_dT": [],
        }
        if enable_cem43:
            grid_states[grid_name]["cem43_map"] = torch.zeros_like(bio.T)
            grid_states[grid_name]["max_cem43"] = []

    max_dT = []
    mean_dT = []
    area_gt1 = []
    area_gt2 = []
    area_gt3 = []
    max_cem43 = []
    p99_cem43 = []
    power_sum = np.zeros(n_electrodes, dtype=np.float64)
    window_frames = 0

    def update_window(window_power_sum: np.ndarray, count: int) -> None:
        if count <= 0:
            return
        mean_power = (window_power_sum / float(count)).astype(np.float32, copy=False)
        window_dt = dt * float(count)
        for state in grid_states.values():
            bio: Bioheat2D = state["bio"]  # type: ignore[assignment]
            mask_np = state["mask_np"]  # type: ignore[assignment]
            p_grid = torch.as_tensor(mean_power[mask_np], dtype=torch.float32, device=device)
            if enable_cem43:
                prev_weight = cem43_weight_from_temperature(bio.T)
            bio.update(p_grid, window_dt)
            if enable_cem43:
                cur_weight = cem43_weight_from_temperature(bio.T)
                state["cem43_map"].add_(0.5 * (prev_weight + cur_weight) * (window_dt / 60.0))  # type: ignore[index]
            state["prev_dT"] = (bio.T - float(state["baseline_temp"])).detach()

    def record_metrics() -> None:
        grid_max_values = []
        grid_cem43_maps = []
        grid_cem43_max_values = []
        weighted_mean_sum = torch.tensor(0.0, dtype=torch.float32, device=device)
        weighted_mean_count = 0
        a1 = torch.tensor(0.0, dtype=torch.float32, device=device)
        a2 = torch.tensor(0.0, dtype=torch.float32, device=device)
        a3 = torch.tensor(0.0, dtype=torch.float32, device=device)
        for state in grid_states.values():
            dT = state["prev_dT"]  # type: ignore[assignment]
            grid_max = dT.max()
            grid_mean = dT.mean()
            state["max_dT"].append(grid_max)  # type: ignore[index]
            state["mean_dT"].append(grid_mean)  # type: ignore[index]
            grid_max_values.append(grid_max)
            weighted_mean_sum += grid_mean * int(dT.numel())
            weighted_mean_count += int(dT.numel())
            pixel_area = float(state["pixel_area_mm2"])
            a1 += (dT > 1.0).sum(dtype=torch.float32) * pixel_area
            a2 += (dT > 2.0).sum(dtype=torch.float32) * pixel_area
            a3 += (dT > 3.0).sum(dtype=torch.float32) * pixel_area
            if enable_cem43:
                cem43_map = state["cem43_map"]  # type: ignore[assignment]
                grid_cem43_maps.append(cem43_map)
                grid_cem43_max = cem43_map.max()
                state["max_cem43"].append(grid_cem43_max)  # type: ignore[index]
                grid_cem43_max_values.append(grid_cem43_max)

        max_dT.append(torch.stack(grid_max_values).max())
        mean_dT.append(weighted_mean_sum / max(weighted_mean_count, 1))
        area_gt1.append(a1)
        area_gt2.append(a2)
        area_gt3.append(a3)
        if enable_cem43:
            max_cem43.append(torch.stack(grid_cem43_max_values).max())
            p99_cem43.append(percentile_tensor(grid_cem43_maps, 99.0, device))

    frame_iter = range(frame_count)
    if show_progress and tqdm is not None:
        frame_iter = tqdm(
            frame_iter,
            total=frame_count,
            desc=progress_label or "thermal",
            unit="frame",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
        )

    for frame_idx in frame_iter:
        amp_a = amplitude_uA[frame_idx].astype(np.float64, copy=False) * 1e-6
        power_sum += (amp_a * amp_a) * impedance_ohm.astype(np.float64) * duty.astype(np.float64)
        window_frames += 1
        if window_frames >= thermal_update_interval_frames or frame_idx == frame_count - 1:
            update_window(power_sum, window_frames)
            power_sum.fill(0.0)
            window_frames = 0
        record_metrics()
        if show_progress and tqdm is None:
            completed = frame_idx + 1
            step = max(1, frame_count // 20)
            if completed == frame_count or completed % step == 0:
                pct = 100.0 * completed / max(frame_count, 1)
                label = progress_label or "thermal"
                print(f"  {label}: {completed}/{frame_count} frames ({pct:.1f}%)", flush=True)

    thermal_grids: dict[str, dict[str, np.ndarray | float | tuple[float, float, float, float]]] = {}
    for grid_name, state in grid_states.items():
        bio: Bioheat2D = state["bio"]  # type: ignore[assignment]
        dT_final = state["prev_dT"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]
        thermal_grids[grid_name] = {
            "dT_final": dT_final,
            "extent_mm": np.asarray(bio.extent_mm, dtype=np.float32),
            "stationary_dT_C": float(np.max(dT_final)) if dT_final.size else float("nan"),
            "stationary_temperature_C": float(state["baseline_temp"]) + float(np.max(dT_final)),
            "ic_footprint_pixel_count": float(bio.ic_footprint_pixel_count),
            "ic_power_density_W_m3": float(bio.ic_power_density_W_m3),
            "max_dT": simulation_runner.tensor_list_to_numpy(state["max_dT"]),  # type: ignore[arg-type]
            "mean_dT": simulation_runner.tensor_list_to_numpy(state["mean_dT"]),  # type: ignore[arg-type]
        }
        if enable_cem43:
            thermal_grids[grid_name]["cem43_final"] = (
                state["cem43_map"].detach().cpu().numpy().astype(np.float32)  # type: ignore[index]
            )
            thermal_grids[grid_name]["max_cem43"] = simulation_runner.tensor_list_to_numpy(  # type: ignore[arg-type]
                state["max_cem43"],  # type: ignore[index]
            )

    hottest_grid_name = max(
        thermal_grids,
        key=lambda name: float(np.nanmax(thermal_grids[name]["dT_final"])),
    )
    payload = {
        "max_dT": simulation_runner.tensor_list_to_numpy(max_dT),
        "mean_dT": simulation_runner.tensor_list_to_numpy(mean_dT),
        "area_gt1_mm2": simulation_runner.tensor_list_to_numpy(area_gt1),
        "area_gt2_mm2": simulation_runner.tensor_list_to_numpy(area_gt2),
        "area_gt3_mm2": simulation_runner.tensor_list_to_numpy(area_gt3),
        "dT_final": thermal_grids[hottest_grid_name]["dT_final"],
        "extent_mm": thermal_grids[hottest_grid_name]["extent_mm"],
        "thermal_grid_names": np.asarray(list(thermal_grids.keys())),
        "dT_final_reference_grid": np.asarray(hottest_grid_name),
        "electrode_ids": electrode_ids.astype(np.int64),
        "electrode_xy_mm": elec_xy_mm.astype(np.float32),
        "electrode_impedance_ohm": impedance_ohm.astype(np.float32),
        "stationary_temperature_C": np.asarray(
            thermal_grids[hottest_grid_name]["stationary_temperature_C"],
            dtype=np.float32,
        ),
        "stationary_dT_C": np.asarray(
            thermal_grids[hottest_grid_name]["stationary_dT_C"],
            dtype=np.float32,
        ),
        "internal_circuit_power_total_mW": np.asarray(ic_power_total_mw, dtype=np.float32),
        "internal_circuit_footprint_pixels": np.asarray(
            thermal_grids[hottest_grid_name]["ic_footprint_pixel_count"],
            dtype=np.float32,
        ),
        "internal_circuit_power_density_W_m3": np.asarray(
            thermal_grids[hottest_grid_name]["ic_power_density_W_m3"],
            dtype=np.float32,
        ),
    }
    if enable_cem43:
        hottest_cem43_grid_name = max(
            thermal_grids,
            key=lambda name: float(np.nanmax(thermal_grids[name]["cem43_final"])),
        )
        payload["max_cem43"] = simulation_runner.tensor_list_to_numpy(max_cem43)
        payload["p99_cem43"] = simulation_runner.tensor_list_to_numpy(p99_cem43)
        payload["cem43_final"] = thermal_grids[hottest_cem43_grid_name]["cem43_final"]
        payload["cem43_final_reference_grid"] = np.asarray(hottest_cem43_grid_name)
        payload["cem43_extent_mm"] = thermal_grids[hottest_cem43_grid_name]["extent_mm"]

    for grid_name, grid_data in thermal_grids.items():
        suffix = sanitize_path_part(grid_name)
        payload[f"dT_final_{suffix}"] = grid_data["dT_final"]
        payload[f"extent_mm_{suffix}"] = grid_data["extent_mm"]
        payload[f"stationary_temperature_C_{suffix}"] = np.asarray(
            grid_data["stationary_temperature_C"],
            dtype=np.float32,
        )
        payload[f"stationary_dT_C_{suffix}"] = np.asarray(
            grid_data["stationary_dT_C"],
            dtype=np.float32,
        )
        payload[f"max_dT_{suffix}"] = grid_data["max_dT"]
        payload[f"mean_dT_{suffix}"] = grid_data["mean_dT"]
        payload[f"internal_circuit_footprint_pixels_{suffix}"] = np.asarray(
            grid_data["ic_footprint_pixel_count"],
            dtype=np.float32,
        )
        payload[f"internal_circuit_power_density_W_m3_{suffix}"] = np.asarray(
            grid_data["ic_power_density_W_m3"],
            dtype=np.float32,
        )
        if enable_cem43:
            payload[f"cem43_final_{suffix}"] = grid_data["cem43_final"]
            payload[f"max_cem43_{suffix}"] = grid_data["max_cem43"]

    return payload


def load_npz_payload(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as bundle:
        return {key: bundle[key] for key in bundle.files}


def write_atomic_npz(npz_path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_path = npz_path.with_name(f"{npz_path.stem}.tmp{npz_path.suffix}")
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, npz_path)


def backup_npz(npz_path: Path) -> Path:
    backup_path = npz_path.with_name("safety_metrics.original_temperature_backup.npz")
    if not backup_path.exists():
        shutil.copy2(npz_path, backup_path)
    return backup_path


def overwrite_one(
    npz_path: Path,
    params: dict,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    data = load_npz_payload(npz_path)
    manifest = load_run_manifest(npz_path)
    coords_yaml = resolve_coords_yaml(npz_path, args.coords_yaml)
    thermal_payload = recompute_temperature(
        data,
        params,
        coords_yaml,
        device,
        manifest=manifest,
        thermal_update_interval_frames=int(args.thermal_update_interval_frames),
        enable_cem43=bool(args.enable_cem43),
        progress_label=short_progress_label(npz_path.parent.name),
        show_progress=not bool(args.no_progress),
    )
    merged = {key: value for key, value in data.items() if not thermal_key_is_stale(key)}
    merged.update(thermal_payload)
    if bool(args.backup):
        backup_path = backup_npz(npz_path)
        print(f"  backup: {backup_path}")
    write_atomic_npz(npz_path, merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute Bioheat2D temperature metrics from saved electrode amplitudes "
            "and overwrite thermal fields in safety_metrics.npz files."
        )
    )
    parser.add_argument("--input", required=True, help="A safety_metrics.npz file or directory to search.")
    parser.add_argument("--params", default=DEFAULT_PARAMS)
    parser.add_argument("--coords-yaml", default=None, help="Fallback coords YAML when run_manifest.yaml is absent.")
    parser.add_argument(
        "--thermal-update-interval-frames",
        type=int,
        default=DEFAULT_THERMAL_UPDATE_INTERVAL_FRAMES,
    )
    parser.add_argument("--backup", dest="backup", action="store_true", default=True)
    parser.add_argument("--no-backup", dest="backup", action="store_false")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--enable-cem43", dest="enable_cem43", action="store_true", default=True)
    parser.add_argument("--disable-cem43", dest="enable_cem43", action="store_false")
    parser.add_argument("--no-progress", action="store_true", help="Disable per-session frame progress bars.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_yaml(args.params)
    device = configure_device(params, force_cpu=bool(args.force_cpu))
    paths = discover_npz_inputs(resolve_repo_path(args.input))
    print(f"Recomputing thermal metrics for {len(paths)} file(s) on {device}.")
    for index, npz_path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {npz_path}")
        overwrite_one(npz_path, params, args, device)
        print("  overwritten")


if __name__ == "__main__":
    main()
