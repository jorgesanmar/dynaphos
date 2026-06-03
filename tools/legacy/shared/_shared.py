from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import yaml

from dynaphos import cortex_models, utils
from dynaphos.image_processing import image_preprocessing
from dynaphos.simulator import GaussianSimulator
from dynaphos.utils import Map


PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg"}
PREPROCESS_METHODS = ("none", "dog", "canny", "sobel")
RASTER_PATTERNS = ("none", "horizontal", "vertical", "checkerboard", "random")


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clone_params(params: Dict) -> Dict:
    return copy.deepcopy(params)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def resolve_media_inputs(input_arg: str) -> List[Path]:
    input_path = resolve_repo_path(input_arg)
    if input_path.is_dir():
        media_paths = [
            path.resolve()
            for path in sorted(input_path.iterdir())
            if path.is_file() and (is_image_file(path) or is_video_file(path))
        ]
        if media_paths:
            return media_paths

        nested_media_paths = [
            path.resolve()
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and (is_image_file(path) or is_video_file(path))
        ]
        if nested_media_paths:
            return nested_media_paths

        raise RuntimeError(f"No supported media files found in: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return [input_path.resolve()]


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    side = min(height, width)
    start_y = (height - side) // 2
    start_x = (width - side) // 2
    return frame[start_y:start_y + side, start_x:start_x + side]


def ensure_gray_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame_f32 = frame.astype(np.float32, copy=False)
    min_val = float(frame_f32.min())
    max_val = float(frame_f32.max())
    if max_val <= min_val:
        return np.zeros_like(frame_f32, dtype=np.uint8)
    normalized = (frame_f32 - min_val) / (max_val - min_val)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def as_uint8_preserving_range(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    frame_arr = np.asarray(frame)
    if np.issubdtype(frame_arr.dtype, np.floating):
        max_val = float(np.nanmax(frame_arr)) if frame_arr.size else 0.0
        if max_val <= 1.0:
            return np.clip(frame_arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.clip(frame_arr, 0.0, 255.0).astype(np.uint8)


def prepare_square_gray_frame(frame: np.ndarray, render_size: int) -> np.ndarray:
    gray = ensure_gray_frame(frame)
    square = center_crop_square(gray)
    return cv2.resize(square, (int(render_size), int(render_size)), interpolation=cv2.INTER_AREA)


def preprocess_gray_frame(
    frame_gray: np.ndarray,
    method: str,
    *,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float,
    canny_high: float,
    use_cuda: bool,
) -> np.ndarray:
    method = str(method).strip().lower()
    if method not in PREPROCESS_METHODS:
        raise ValueError(
            f"Unsupported preprocessing method '{method}'. Expected one of {list(PREPROCESS_METHODS)}."
        )

    if method == "dog":
        sigma_low, sigma_high = sorted((float(dog_sigma_low), float(dog_sigma_high)))
        processed = image_preprocessing(
            frame_gray,
            method="dog",
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            use_cuda=bool(use_cuda),
        )
        return normalize_to_uint8(processed)

    blurred = cv2.GaussianBlur(frame_gray, (9, 9), 5)

    if method == "none":
        return normalize_to_uint8(blurred)

    if method == "canny":
        low, high = sorted((float(canny_low), float(canny_high)))
        processed = image_preprocessing(
            blurred,
            method="canny",
            threshold_low=low,
            threshold_high=high,
        )
    else:
        processed = image_preprocessing(blurred, method="sobel")

    return normalize_to_uint8(processed)


def prepare_simulation_frame(
    frame: np.ndarray,
    *,
    render_size: int,
    input_stage: str,
    preprocessing_method: str,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float,
    canny_high: float,
    use_cuda: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    input_stage = str(input_stage).strip().lower()

    if input_stage == "original":
        original_gray = prepare_square_gray_frame(frame, render_size)
        processed = preprocess_gray_frame(
            original_gray,
            preprocessing_method,
            dog_sigma_low=dog_sigma_low,
            dog_sigma_high=dog_sigma_high,
            canny_low=canny_low,
            canny_high=canny_high,
            use_cuda=use_cuda,
        )
        return original_gray, processed
    if input_stage == "preprocessed":
        processed = ensure_gray_frame(frame)
        if processed.shape != (int(render_size), int(render_size)):
            is_downsampling = processed.shape[0] > int(render_size) or processed.shape[1] > int(render_size)
            processed = cv2.resize(
                processed,
                (int(render_size), int(render_size)),
                interpolation=cv2.INTER_AREA if is_downsampling else cv2.INTER_LINEAR,
            )
        processed = as_uint8_preserving_range(processed)
        return processed, processed
    else:
        raise ValueError("input_stage must be either 'original' or 'preprocessed'.")


def to_bgr_u8(frame: np.ndarray) -> np.ndarray:
    frame_u8 = normalize_to_uint8(frame)
    if frame_u8.ndim == 2:
        return cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
    return frame_u8


def add_panel_label(frame: np.ndarray, label: str, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    labeled = to_bgr_u8(frame).copy()
    cv2.putText(
        labeled,
        str(label),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return labeled


def make_panel(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    columns: Optional[int] = None,
    background: int = 0,
) -> np.ndarray:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length.")
    if not images:
        raise ValueError("At least one image is required to build a panel.")

    labeled_images = [add_panel_label(image, label) for image, label in zip(images, labels)]
    height = max(image.shape[0] for image in labeled_images)
    width = max(image.shape[1] for image in labeled_images)
    if columns is None or columns < 1:
        columns = len(labeled_images)
    rows = int(np.ceil(float(len(labeled_images)) / float(columns)))

    blank = np.full((height, width, 3), int(background), dtype=np.uint8)
    padded: List[np.ndarray] = []
    for image in labeled_images:
        canvas = blank.copy()
        canvas[: image.shape[0], : image.shape[1]] = image
        padded.append(canvas)

    while len(padded) < rows * columns:
        padded.append(blank.copy())

    row_images = []
    for row_idx in range(rows):
        row_start = row_idx * columns
        row_images.append(np.hstack(padded[row_start:row_start + columns]))
    return np.vstack(row_images)


def resolve_coords_yaml(coords_yaml: Optional[str]) -> Path:
    if coords_yaml:
        return resolve_repo_path(coords_yaml)

    preferred = PROJECT_ROOT / "config" / "grid_coords_dipole_valid.yaml"
    if preferred.exists():
        return preferred.resolve()
    fallback = PROJECT_ROOT / "config" / "grid_coords_dipole.yaml"
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError("Could not find a default dipole grid YAML file in config/.")


def configure_runtime_params(
    params: Dict,
    *,
    render_resolution: Optional[int] = None,
    fps_override: Optional[float] = None,
    force_cpu: bool = False,
    disable_temporal_dynamics: bool = False,
) -> Dict:
    params = clone_params(params)
    if render_resolution is not None:
        render_resolution = int(render_resolution)
        params.setdefault("run", {})
        params["run"]["resolution"] = [render_resolution, render_resolution]
    if fps_override is not None and float(fps_override) > 0:
        params.setdefault("run", {})
        params["run"]["fps"] = float(fps_override)

    params["run"]["batch_size"] = 0
    if force_cpu or not torch.cuda.is_available():
        params["run"]["gpu"] = None

    if disable_temporal_dynamics:
        temporal = params.setdefault("temporal_dynamics", {})
        temporal["trace_increase_rate"] = 0.0
        temporal["activation_decay_per_second"] = 0.999999
        temporal["trace_decay_per_second"] = 0.999999

    return params


def infer_device_from_params(params: Dict) -> torch.device:
    gpu_id = params.get("run", {}).get("gpu", None)
    if gpu_id in (None, False) or not torch.cuda.is_available():
        params["run"]["gpu"] = None
        return torch.device("cpu")
    torch.cuda.set_device(int(gpu_id))
    return torch.device(f"cuda:{int(gpu_id)}")


def load_coordinate_maps(params: Dict, coords_yaml: Path) -> Tuple[Map, Map, np.ndarray, Map]:
    x_raw, y_raw = utils.load_coordinates_from_yaml(str(coords_yaml))
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    coordinates_cortex = Map(x=x_raw, y=y_raw)

    if hasattr(cortex_models, "get_full_field_mapping_from_cortex"):
        mapping = cortex_models.get_full_field_mapping_from_cortex(
            params["cortex_model"],
            coordinates_cortex=coordinates_cortex,
        )
        phosphene_map = mapping.phosphene_map
        indices = np.asarray(mapping.indices, dtype=np.int64)
        raster_coordinates = mapping.cortical_coordinates
    else:
        if hasattr(cortex_models, "get_visual_field_coordinates_from_cortex_full"):
            mapping_out = cortex_models.get_visual_field_coordinates_from_cortex_full(
                params["cortex_model"],
                coordinates_cortex,
            )
        else:
            mapping_out = cortex_models.get_visual_field_coordinates_from_cortex(
                params["cortex_model"],
                coordinates_cortex,
            )

        if isinstance(mapping_out, tuple):
            phosphene_map, indices = mapping_out
        else:
            phosphene_map = mapping_out
            indices = np.arange(len(phosphene_map), dtype=np.int64)

        indices = np.asarray(indices, dtype=np.int64)
        x_expanded, y_expanded = cortex_models.expand_electrode_coordinate_arrays(x_raw, y_raw, indices)
        raster_coordinates = Map(
            x=x_expanded[indices],
            y=y_expanded[indices],
        )
    return coordinates_cortex, phosphene_map, indices, raster_coordinates


def create_simulator(
    *,
    params_path: str,
    coords_yaml: Optional[str],
    render_resolution: Optional[int] = None,
    fps_override: Optional[float] = None,
    raster_mode: str = "none",
    raster_num_groups: int = 4,
    raster_rate_hz: float = 1.0,
    force_cpu: bool = False,
    disable_temporal_dynamics: bool = False,
) -> Tuple[GaussianSimulator, Dict, Map, Map, np.ndarray, Map]:
    params_path_resolved = resolve_repo_path(params_path)
    coords_yaml_resolved = resolve_coords_yaml(coords_yaml)
    params = load_yaml(params_path_resolved)
    params = configure_runtime_params(
        params,
        render_resolution=render_resolution,
        fps_override=fps_override,
        force_cpu=force_cpu,
        disable_temporal_dynamics=disable_temporal_dynamics,
    )
    coordinates_cortex, phosphene_map, indices, raster_coordinates = load_coordinate_maps(params, coords_yaml_resolved)

    raster_mode = str(raster_mode).strip().lower()
    if raster_mode not in RASTER_PATTERNS:
        raise ValueError(f"Unsupported raster mode '{raster_mode}'. Expected one of {list(RASTER_PATTERNS)}.")

    raster_enabled = raster_mode != "none"
    raster_pattern = raster_mode if raster_enabled else "checkerboard"
    simulator = GaussianSimulator(
        params,
        phosphene_map,
        raster_coordinates=raster_coordinates,
        raster_enabled=raster_enabled,
        raster_pattern=raster_pattern,
        raster_num_groups=int(raster_num_groups),
        raster_rate_hz=float(raster_rate_hz),
    )
    return simulator, params, coordinates_cortex, phosphene_map, indices, raster_coordinates


def move_simulator_tensors_to_device(simulator: GaussianSimulator, device: torch.device) -> None:
    if hasattr(simulator, "data_kwargs") and isinstance(simulator.data_kwargs, dict):
        simulator.data_kwargs["device"] = str(device)
    if hasattr(simulator, "device"):
        simulator.device = torch.device(device)

    for attr in [
        "phosphene_maps",
        "magnification",
        "_pulse_width",
        "_frequency",
        "_zero",
        "_inf",
        "cumulative_charge_uC",
        "_phosphene_centers",
        "_sampling_mask",
        "power_indices",
        "power_weights",
    ]:
        if hasattr(simulator, attr):
            value = getattr(simulator, attr)
            if torch.is_tensor(value):
                setattr(simulator, attr, value.to(device))

    for attr in ["activation", "trace", "sigma", "brightness", "threshold", "impedance"]:
        if hasattr(simulator, attr):
            obj = getattr(simulator, attr)
            if hasattr(obj, "data_kwargs") and isinstance(obj.data_kwargs, dict):
                obj.data_kwargs["device"] = str(device)
            if hasattr(obj, "state") and torch.is_tensor(obj.state):
                obj.state = obj.state.to(device)
            for tensor_attr in ["scale", "fps", "rel_stim_duration", "decay_rate", "slope", "cps_half", "_a", "_b"]:
                if hasattr(obj, tensor_attr):
                    tensor_value = getattr(obj, tensor_attr)
                    if torch.is_tensor(tensor_value):
                        setattr(obj, tensor_attr, tensor_value.to(device))

    if hasattr(simulator, "safety_tracker"):
        tracker = simulator.safety_tracker
        if hasattr(tracker, "data_kwargs") and isinstance(tracker.data_kwargs, dict):
            tracker.data_kwargs["device"] = str(device)
        for attr in [
            "window_charge_per_electrode_nC",
            "protocol_charge_per_electrode_nC",
            "last_charge_per_phase_nC",
            "last_charge_density_uc_cm2",
            "last_shannon_k",
        ]:
            if hasattr(tracker, attr):
                value = getattr(tracker, attr)
                if torch.is_tensor(value):
                    setattr(tracker, attr, value.to(device))

    if hasattr(simulator, "raster_groups_flat") and torch.is_tensor(simulator.raster_groups_flat):
        simulator.raster_groups_flat = simulator.raster_groups_flat.to(device)
    if hasattr(simulator, "raster_schedule") and simulator.raster_schedule is not None:
        simulator.raster_schedule = [mask.to(device) for mask in simulator.raster_schedule]
    if hasattr(simulator, "instant_power") and torch.is_tensor(simulator.instant_power):
        simulator.instant_power = simulator.instant_power.to(device)
    if hasattr(simulator, "frame_power") and torch.is_tensor(simulator.frame_power):
        simulator.frame_power = simulator.frame_power.to(device)
    if hasattr(simulator, "frame_energy") and torch.is_tensor(simulator.frame_energy):
        simulator.frame_energy = simulator.frame_energy.to(device)
    if hasattr(simulator, "effective_charge_per_second") and torch.is_tensor(simulator.effective_charge_per_second):
        simulator.effective_charge_per_second = simulator.effective_charge_per_second.to(device)


def get_render_components(
    simulator: GaussianSimulator,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    activation = simulator.gaussian_activation()
    raster_mask = simulator.get_current_raster_mask() if simulator.raster_enabled else None
    supra_threshold = torch.greater(simulator.activation.get(), simulator.threshold.get())
    intensity = torch.where(supra_threshold, simulator.brightness.get(), simulator._zero)
    if raster_mask is not None:
        intensity = intensity * raster_mask
    return activation, intensity, raster_mask


def render_percept_from_state(
    simulator: GaussianSimulator,
    *,
    activation: Optional[torch.Tensor] = None,
    intensity: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if activation is None or intensity is None:
        activation, intensity, _ = get_render_components(simulator)
    return torch.sum(intensity * activation, dim=simulator._electrode_dimension).clamp(0, 1)


def render_raster_activity_map(
    simulator: GaussianSimulator,
    *,
    activation: Optional[torch.Tensor] = None,
    raster_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if activation is None:
        activation = simulator.gaussian_activation()
    if raster_mask is None:
        raster_mask = simulator.get_current_raster_mask() if simulator.raster_enabled else None
    if raster_mask is not None:
        weights = raster_mask
    else:
        weights = torch.ones(simulator.shape, **simulator.data_kwargs)
    raster_field = torch.sum(weights * activation, dim=simulator._electrode_dimension)
    max_val = torch.amax(raster_field)
    if torch.isfinite(max_val) and float(max_val.item()) > 0.0:
        raster_field = raster_field / max_val
    return raster_field.clamp(0, 1)


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    image_np = image.detach().cpu().numpy()
    return normalize_to_uint8(image_np * 255.0 if image_np.max() <= 1.0 else image_np)


def build_surviving_electrode_set(
    x_raw_mm: np.ndarray,
    y_raw_mm: np.ndarray,
    remaining_indices: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor]:
    x_expanded, y_expanded = cortex_models.expand_electrode_coordinate_arrays(
        x_raw_mm,
        y_raw_mm,
        remaining_indices,
    )
    phys_idx_unique, inverse = np.unique(np.asarray(remaining_indices, dtype=np.int64), return_inverse=True)
    elec_xy_mm_surv = np.stack(
        [x_expanded[phys_idx_unique], y_expanded[phys_idx_unique]],
        axis=1,
    ).astype(np.float64)
    inverse_t = torch.tensor(inverse.astype(np.int64), dtype=torch.long, device=device)
    return elec_xy_mm_surv, inverse_t


def aggregate_metric(metric: torch.Tensor, inverse_map_t: torch.Tensor, n_electrodes: int) -> np.ndarray:
    metric = metric.reshape(-1).to(inverse_map_t.device)
    out = torch.zeros(n_electrodes, device=inverse_map_t.device, dtype=metric.dtype)
    out.scatter_add_(0, inverse_map_t, metric)
    return out.detach().cpu().numpy().astype(np.float32)


def open_video_writer(path: Path, fps: float, size: Tuple[int, int], is_color: bool = True) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), tuple(int(v) for v in size), bool(is_color))
