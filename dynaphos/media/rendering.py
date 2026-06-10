"""Render phosphene outputs from image and video inputs."""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from dynaphos.paths import package_file
from dynaphos.simulation import cortex as cortex_models
from dynaphos.simulation import utils
from dynaphos.media.preprocessing import image_preprocessing
from dynaphos.simulation.simulator import GaussianSimulator


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg"}
PREPROCESS_METHODS = ("none", "dog", "canny", "sobel")
RASTER_PATTERNS = ("none", "horizontal", "vertical", "checkerboard", "random")


def sanitize_path_part(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value).strip())
    return safe.strip("._") or "unspecified"


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def resolve_coords_yaml(coords_yaml: str | None) -> Path:
    if coords_yaml:
        return resolve_repo_path(coords_yaml)
    return package_file("coords_400um")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def split_preprocessed_media_stem(stem: str) -> tuple[str, str | None]:
    stem_l = stem.lower()
    for method in PREPROCESS_METHODS:
        suffix = f"_{method}"
        if stem_l.endswith(suffix):
            return stem[:-len(suffix)], method
    return stem, None


def resolve_media_inputs(input_arg: str) -> list[Path]:
    # The CLI accepts either a single file or a directory. For directories we
    # first look at the top level, then recurse so batch runs can point at a
    # whole media tree.
    input_path = resolve_repo_path(input_arg)
    if input_path.is_dir():
        media_paths = [
            path.resolve()
            for path in sorted(input_path.iterdir())
            if path.is_file() and (is_image_file(path) or is_video_file(path))
        ]
        if media_paths:
            return media_paths

        recursive_media_paths = [
            path.resolve()
            for path in sorted(input_path.rglob("*"))
            if path.is_file() and (is_image_file(path) or is_video_file(path))
        ]
        if recursive_media_paths:
            return recursive_media_paths

        raise RuntimeError(f"No supported media files found in: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return [input_path.resolve()]


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, secs = divmod(seconds, 60.0)
    hours, minutes = divmod(minutes, 60.0)
    if hours >= 1:
        return f"{int(hours):d}:{int(minutes):02d}:{secs:05.2f}"
    return f"{int(minutes):02d}:{secs:05.2f}"


def resolve_output_stem(media_path: Path) -> str:
    base_stem, detected_method = split_preprocessed_media_stem(media_path.stem)
    return base_stem if detected_method is not None else media_path.stem


def resolve_input_type(media_path: Path) -> str:
    # The output tree separates image and video runs
    if is_image_file(media_path):
        return "image"
    return "video"


def resolve_output_method(media_path: Path, input_stage: str, preprocessing_method: str) -> str:
    # When the input is an original recording, the preprocessing method comes from the CLI.
    # For side-by-side preprocessed files, infer the method from the filename suffix.
    stage = str(input_stage).strip().lower()
    if stage == "original":
        return str(preprocessing_method).strip().lower()

    _base_stem, detected_method = split_preprocessed_media_stem(media_path.stem)
    if detected_method is not None:
        return detected_method
    return "preprocessed"


def resolve_electrode_grid_name(coords_yaml: Path) -> str:
    return sanitize_path_part(coords_yaml.stem)


def format_float_for_path(value: float) -> str:
    return f"{float(value):g}".replace("-", "minus_").replace(".", "p")


def resolve_view_angle_name(params: dict) -> str:
    view_angle = float(params.get("run", {}).get("view_angle", 16.0))
    return f"view_angle_{format_float_for_path(view_angle)}deg"


def get_render_resolution_xy(params: dict) -> tuple[int, int]:
    # The simulator renders into the configured percept image size, which can
    # be overridden from the CLI.
    resolution = params.get("run", {}).get("resolution", None)
    if resolution is None or len(resolution) != 2:
        raise ValueError(f"Expected params['run']['resolution'] to contain [width, height], got: {resolution!r}")
    return int(resolution[0]), int(resolution[1])


def get_frame_shape(render_resolution_xy: tuple[int, int]) -> tuple[int, int]:
    render_width, render_height = render_resolution_xy
    return int(render_height), int(render_width)


def to_display_u8(frame: np.ndarray) -> np.ndarray:
    # OpenCV video and image writing expects uint8 display images. This helper
    # normalizes the different intermediate array types used in the pipeline.
    frame_arr = np.asarray(frame)
    if frame_arr.dtype == np.uint8:
        return frame_arr
    if np.issubdtype(frame_arr.dtype, np.floating):
        max_val = float(np.nanmax(frame_arr)) if frame_arr.size else 0.0
        if max_val <= 1.0:
            frame_arr = frame_arr * 255.0
    return np.clip(frame_arr, 0.0, 255.0).astype(np.uint8)


def normalize_to_u8(frame: np.ndarray) -> np.ndarray:
    frame_arr = np.asarray(frame)
    if frame_arr.size == 0:
        return frame_arr.astype(np.uint8)

    frame_f32 = frame_arr.astype(np.float32, copy=False)
    min_val = float(np.nanmin(frame_f32))
    max_val = float(np.nanmax(frame_f32))
    if not np.isfinite(min_val) or not np.isfinite(max_val) or max_val <= min_val:
        return np.zeros_like(frame_arr, dtype=np.uint8)

    normalized = (frame_f32 - min_val) / (max_val - min_val)
    return np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)


def normalize_to_uint8(frame: np.ndarray) -> np.ndarray:
    return normalize_to_u8(frame)


def ensure_gray_original_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def collapse_preprocessed_frame(frame: np.ndarray) -> np.ndarray:
    # Preprocessed inputs must already be single-channel stimulation masks.
    # Accept true grayscale arrays, single-channel arrays, or decoded video
    # frames. OpenCV commonly returns grayscale MP4 files as 3-channel BGR, and
    # lossy codecs can introduce tiny per-channel differences, so fall back to
    # an explicit grayscale conversion for multi-channel video frames.
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0]
    if frame.ndim == 3 and frame.shape[2] >= 3:
        first_channel = frame[:, :, 0]
        if np.all(frame[:, :, 1] == first_channel) and np.all(frame[:, :, 2] == first_channel):
            return first_channel
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError(
        "Preprocessed inputs must already be single-channel stimulation frames. "
        f"Expected a 2D grayscale frame or decoded BGR video frame, got shape {frame.shape!r}."
    )


def center_crop_square(frame: np.ndarray) -> np.ndarray:
    # The simulator assumes square input images, so videos are center-cropped
    # before resizing.
    height, width = frame.shape[:2]
    if height == width:
        return frame
    side = min(height, width)
    start_y = (height - side) // 2
    start_x = (width - side) // 2
    return frame[start_y:start_y + side, start_x:start_x + side]


def prepare_square_gray_frame(frame: np.ndarray, render_size: int) -> np.ndarray:
    gray = ensure_gray_original_frame(frame)
    square = center_crop_square(gray)
    return cv2.resize(square, (int(render_size), int(render_size)), interpolation=cv2.INTER_AREA)


def preprocess_gray_frame(
    frame_gray: np.ndarray,
    method: str,
    *,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float = 75.0,
    canny_high: float = 170.0,
    use_cuda: bool = False,
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
        return normalize_to_u8(processed)

    blurred = cv2.GaussianBlur(frame_gray, (9, 9), 5)
    if method == "none":
        return normalize_to_u8(blurred)
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
    return normalize_to_u8(processed)


def _to_bgr_u8(frame: np.ndarray) -> np.ndarray:
    frame_u8 = normalize_to_u8(frame)
    if frame_u8.ndim == 2:
        return cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
    return frame_u8


def _add_panel_label(frame: np.ndarray, label: str, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    labeled = _to_bgr_u8(frame).copy()
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
    images: list[np.ndarray] | tuple[np.ndarray, ...],
    labels: list[str] | tuple[str, ...],
    *,
    columns: int | None = None,
    background: int = 0,
) -> np.ndarray:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length.")
    if not images:
        raise ValueError("At least one image is required to build a panel.")

    labeled_images = [_add_panel_label(image, label) for image, label in zip(images, labels)]
    height = max(image.shape[0] for image in labeled_images)
    width = max(image.shape[1] for image in labeled_images)
    if columns is None or columns < 1:
        columns = len(labeled_images)
    rows = int(np.ceil(float(len(labeled_images)) / float(columns)))

    blank = np.full((height, width, 3), int(background), dtype=np.uint8)
    padded: list[np.ndarray] = []
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


def prepare_original_frame(
    frame: np.ndarray,
    *,
    render_resolution_xy: tuple[int, int],
    preprocessing_method: str,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float = 75.0,
    canny_high: float = 170.0,
    use_cuda: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    # This is the "raw media" path:
    # original frame -> grayscale -> square crop -> resize -> preprocessing.
    # Canny and "none" use a smoothing pass; DoG performs its own blurs.
    gray = ensure_gray_original_frame(frame)
    square = center_crop_square(gray)
    render_width, render_height = render_resolution_xy
    resized = cv2.resize(square, (render_width, render_height), interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (9, 9), 5)

    method = str(preprocessing_method).strip().lower()
    if method not in PREPROCESS_METHODS:
        raise ValueError(
            f"Unsupported preprocessing method '{preprocessing_method}'. "
            f"Expected one of {list(PREPROCESS_METHODS)}."
        )

    if method == "none":
        processed = blurred
    elif method == "dog":
        sigma_low, sigma_high = sorted((float(dog_sigma_low), float(dog_sigma_high)))
        processed = image_preprocessing(
            resized,
            method="dog",
            sigma_low=sigma_low,
            sigma_high=sigma_high,
            use_cuda=bool(use_cuda),
        )
    elif method == "canny":
        threshold_low, threshold_high = sorted((float(canny_low), float(canny_high)))
        processed = image_preprocessing(
            blurred,
            method="canny",
            threshold_low=threshold_low,
            threshold_high=threshold_high,
        )
    else:
        processed = image_preprocessing(blurred, method="sobel")

    return to_display_u8(resized), to_display_u8(processed)


def prepare_preprocessed_frame(frame: np.ndarray, *, render_resolution_xy: tuple[int, int]) -> np.ndarray:
    # This is the "already preprocessed" path:
    # treat the frame as the stimulation image, stretch decoded values to the
    # full 0..255 range, and only resize if needed so it matches the simulator
    # resolution.
    processed = normalize_to_u8(collapse_preprocessed_frame(frame))
    expected_shape = get_frame_shape(render_resolution_xy)
    if processed.shape != expected_shape:
        is_downsampling = (
            processed.shape[0] > expected_shape[0] or processed.shape[1] > expected_shape[1]
        )
        interpolation = cv2.INTER_AREA if is_downsampling else cv2.INTER_LINEAR
        processed = cv2.resize(
            processed,
            (int(render_resolution_xy[0]), int(render_resolution_xy[1])),
            interpolation=interpolation,
        )
    return processed


def prepare_stimulus_frame(
    frame: np.ndarray,
    *,
    input_stage: str,
    render_resolution_xy: tuple[int, int],
    preprocessing_method: str,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float = 75.0,
    canny_high: float = 170.0,
    use_cuda: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    stage = str(input_stage).strip().lower()
    if stage == "original":
        # take the raw video frame, resize it
        # to the simulator resolution, and generate the stimulation image with
        # the requested preprocessing method.
        return prepare_original_frame(
            frame,
            render_resolution_xy=render_resolution_xy,
            preprocessing_method=preprocessing_method,
            dog_sigma_low=float(dog_sigma_low),
            dog_sigma_high=float(dog_sigma_high),
            canny_low=float(canny_low),
            canny_high=float(canny_high),
            use_cuda=bool(use_cuda),
        )
    if stage == "preprocessed":
        # This path is for videos already exported by the preprocessing tools.
        # In that case the input frame itself is the stimulation image.
        processed = prepare_preprocessed_frame(frame, render_resolution_xy=render_resolution_xy)
        return processed, processed
    raise ValueError("input_stage must be either 'original' or 'preprocessed'.")


def configure_params(
    *,
    params_path: str,
    render_resolution: int | None,
    view_angle: float | None,
    fps_override: float | None,
    stim_scale: float | None,
    force_cpu: bool,
    disable_temporal_dynamics: bool,
) -> dict:
    # Load the simulator config once, then apply run-local overrides from the
    # CLI without modifying the checked-in YAML file.
    params = utils.load_params(str(resolve_repo_path(params_path)))
    params.setdefault("run", {})

    if render_resolution is not None:
        params["run"]["resolution"] = [int(render_resolution), int(render_resolution)]

    if view_angle is not None:
        view_angle = float(view_angle)
        if view_angle <= 0.0:
            raise ValueError("--view-angle must be greater than 0.")
        params["run"]["view_angle"] = view_angle

    if fps_override is not None and float(fps_override) > 0:
        params["run"]["fps"] = float(fps_override)

    if stim_scale is not None:
        stim_scale = float(stim_scale)
        if not 0.0 <= stim_scale <= 1.0:
            raise ValueError("--stim-scale must be between 0.0 and 1.0 inclusive.")
        sampling = params.setdefault("sampling", {})
        base_stimulus_scale = float(sampling.get("stimulus_scale", 1.0))
        sampling["stimulus_scale"] = base_stimulus_scale * stim_scale

    params["run"]["batch_size"] = 0

    if force_cpu or not torch.cuda.is_available():
        params["run"]["gpu"] = None

    if disable_temporal_dynamics:
        # For visualization it is often easier to reason about the percept when
        # temporal carry-over is flattened.
        temporal = params.setdefault("temporal_dynamics", {})
        temporal["trace_increase_rate"] = 0.0
        temporal["activation_decay_per_second"] = 0.999999
        temporal["trace_decay_per_second"] = 0.999999

    return params


def build_phosphene_coordinates(params: dict, coords_yaml: Path) -> tuple[utils.Map, utils.Map]:
    # Load physical cortical electrode positions, then map them into visual
    # field coordinates where phosphenes are rendered.
    x_raw, y_raw = utils.load_coordinates_from_yaml(str(coords_yaml))
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    coordinates_cortex = utils.Map(x=x_raw, y=y_raw)

    mapping = cortex_models.get_full_field_mapping_from_cortex(
        params["cortex_model"],
        coordinates_cortex=coordinates_cortex,
    )
    phosphene_coordinates = mapping.phosphene_map
    raster_coordinates = mapping.cortical_coordinates
    return phosphene_coordinates, raster_coordinates


def get_phosphene_bounds_deg(phosphene_coordinates: utils.Map) -> dict[str, float]:
    x_deg, y_deg = phosphene_coordinates.cartesian
    x_deg = np.asarray(x_deg, dtype=float)
    y_deg = np.asarray(y_deg, dtype=float)
    finite = np.isfinite(x_deg) & np.isfinite(y_deg)
    if not np.any(finite):
        raise ValueError("Mapped phosphene coordinates do not contain any finite visual-field positions.")
    return {
        "x_min": float(np.min(x_deg[finite])),
        "x_max": float(np.max(x_deg[finite])),
        "y_min": float(np.min(y_deg[finite])),
        "y_max": float(np.max(y_deg[finite])),
    }


def create_simulator(
    *,
    params_path: str,
    coords_yaml: str | None,
    render_resolution: int | None,
    view_angle: float | None,
    fps_override: float | None,
    stim_scale: float | None,
    raster_mode: str,
    groups: int,
    raster_rate_hz: float,
    force_cpu: bool,
    disable_temporal_dynamics: bool,
) -> tuple[GaussianSimulator, dict, tuple[int, int], dict[str, float]]:
    # Build a ready-to-run simulator from:
    # params + chosen grid + chosen raster settings + chosen runtime overrides.
    params = configure_params(
        params_path=params_path,
        render_resolution=render_resolution,
        view_angle=view_angle,
        fps_override=fps_override,
        stim_scale=stim_scale,
        force_cpu=force_cpu,
        disable_temporal_dynamics=disable_temporal_dynamics,
    )
    coords_path = resolve_coords_yaml(coords_yaml)
    phosphene_coordinates, raster_coordinates = build_phosphene_coordinates(params, coords_path)

    raster_mode = str(raster_mode).strip().lower()
    if raster_mode not in RASTER_PATTERNS:
        raise ValueError(f"Unsupported raster mode '{raster_mode}'. Expected one of {list(RASTER_PATTERNS)}.")

    # The simulator needs two coordinate systems:
    # - phosphene_coordinates: where each phosphene is rendered in visual space
    # - raster_coordinates: where raster groups are assigned on the cortical grid
    raster_enabled = raster_mode != "none"
    raster_pattern = raster_mode if raster_enabled else "checkerboard"
    simulator = GaussianSimulator(
        params,
        phosphene_coordinates,
        raster_coordinates=raster_coordinates,
        raster_enabled=raster_enabled,
        raster_pattern=raster_pattern,
        raster_num_groups=int(groups),
        raster_rate_hz=float(raster_rate_hz),
    )
    render_resolution_xy = get_render_resolution_xy(params)
    phosphene_bounds_deg = get_phosphene_bounds_deg(phosphene_coordinates)
    return simulator, params, render_resolution_xy, phosphene_bounds_deg


def render_phosphene_frame(simulator: GaussianSimulator, stim_pattern: torch.Tensor) -> np.ndarray:
    # One simulator call advances the internal state and renders the current
    # phosphene percept as a grayscale image. Keep the tensor on GPU until the
    # final uint8 conversion so each frame does only one host transfer.
    phosphene = simulator(stim_pattern).clamp(0, 1)
    phosphene_u8 = (phosphene * 255.0).to(torch.uint8)
    return phosphene_u8.detach().cpu().numpy()


def render_phosphene_frame_from_state(simulator: GaussianSimulator) -> np.ndarray:
    # Safety simulations call update() directly so electrical and thermal
    # metrics can be recorded before rendering. This mirrors the renderer above
    # without advancing simulator state a second time.
    activation = simulator.gaussian_activation()
    supra = torch.greater(simulator.activation.get(), simulator.threshold.get())
    value = torch.where(supra, simulator.brightness.get(), simulator._zero)
    if simulator.raster_enabled:
        value = value * simulator.get_current_raster_mask()
    phosphene = torch.sum(value * activation, dim=simulator._electrode_dimension).clamp(0, 1)
    phosphene_u8 = (phosphene * 255.0).to(torch.uint8)
    return phosphene_u8.detach().cpu().numpy()


def visual_field_bounds_to_pixels(
    fov_bounds_deg: dict[str, float],
    params: dict,
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    origin_x, origin_y = params.get("run", {}).get("origin", [0, 0])
    view_angle = float(params.get("run", {}).get("view_angle", 16.0))
    x_view_min = float(origin_x) - view_angle / 2.0
    x_view_max = float(origin_x) + view_angle / 2.0
    y_view_min = float(origin_y) - view_angle / 2.0
    y_view_max = float(origin_y) + view_angle / 2.0

    def to_x_pixel(x_deg: float) -> int:
        value = (float(x_deg) - x_view_min) / (x_view_max - x_view_min) * (width - 1)
        return int(np.clip(round(value), 0, width - 1))

    def to_y_pixel(y_deg: float) -> int:
        value = (float(y_deg) - y_view_min) / (y_view_max - y_view_min) * (height - 1)
        return int(np.clip(round(value), 0, height - 1))

    x0 = to_x_pixel(fov_bounds_deg["x_min"])
    x1 = to_x_pixel(fov_bounds_deg["x_max"])
    y0 = to_y_pixel(fov_bounds_deg["y_min"])
    y1 = to_y_pixel(fov_bounds_deg["y_max"])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def draw_phosphene_fov_bounds(
    frame: np.ndarray,
    *,
    fov_bounds_deg: dict[str, float] | None,
    params: dict | None,
) -> np.ndarray:
    if fov_bounds_deg is None or params is None:
        return to_display_u8(frame)

    display = to_display_u8(frame).copy()
    x0, y0, x1, y1 = visual_field_bounds_to_pixels(fov_bounds_deg, params, display.shape)

    # Draw the actual mapped phosphene-coordinate extent. This is based on the
    # selected electrode grid after cortical-to-visual-field mapping, not on the
    # configured full render view angle.
    cv2.rectangle(display, (x0, y0), (x1, y1), color=255, thickness=1, lineType=cv2.LINE_AA)
    cv2.rectangle(
        display,
        (max(0, x0 - 1), max(0, y0 - 1)),
        (min(display.shape[1] - 1, x1 + 1), min(display.shape[0] - 1, y1 + 1)),
        color=0,
        thickness=1,
        lineType=cv2.LINE_AA,
    )

    return display


def build_comparison_frame(
    input_frame: np.ndarray,
    stimulus_frame: np.ndarray,
    phosphene_frame: np.ndarray,
    *,
    fov_bounds_deg: dict[str, float] | None = None,
    params: dict | None = None,
    input_stage: str = "original",
) -> np.ndarray:
    # Useful debugging view: source input, stimulation mask, and resulting
    # percept. All panels are already prepared at the simulator resolution.
    if str(input_stage).strip().lower() == "preprocessed":
        return np.concatenate(
            [
                draw_phosphene_fov_bounds(stimulus_frame, fov_bounds_deg=fov_bounds_deg, params=params),
                to_display_u8(phosphene_frame),
            ],
            axis=1,
        )

    return np.concatenate(
        [
            draw_phosphene_fov_bounds(input_frame, fov_bounds_deg=fov_bounds_deg, params=params),
            draw_phosphene_fov_bounds(stimulus_frame, fov_bounds_deg=fov_bounds_deg, params=params),
            to_display_u8(phosphene_frame),
        ],
        axis=1,
    )


def open_video_writer(path: Path, fps: float, size: tuple[int, int], is_color: bool) -> cv2.VideoWriter:
    # Centralize video-writer creation so all outputs use the same codec and
    # parent-directory handling.
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), tuple(int(v) for v in size), bool(is_color))


def sample_stimulus_pattern(simulator: GaussianSimulator, stimulus_frame: np.ndarray) -> torch.Tensor:
    # Convert the 2D stimulation image into one amplitude value per phosphene /
    # electrode sampling site. This is the signal that gets fed into the
    # simulator on each frame.
    return simulator.sample_stimulus(stimulus_frame, rescale=True)


@torch.inference_mode()
def process_image(
    media_path: Path,
    output_dir: Path,
    *,
    input_stage: str,
    preprocessing_method: str,
    render_resolution: int | None,
    view_angle: float | None,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float,
    canny_high: float,
    use_cuda: bool,
    params_path: str,
    coords_yaml: str | None,
    stim_scale: float | None,
    raster_mode: str,
    groups: int,
    raster_rate_hz: float,
    force_cpu: bool,
    disable_temporal_dynamics: bool,
) -> None:
    frame = cv2.imread(str(media_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Unable to load image: {media_path}")

    coords_path = resolve_coords_yaml(coords_yaml)
    electrode_grid_name = resolve_electrode_grid_name(coords_path)

    # Image mode is the simplest path through the pipeline: build the
    # simulator, prepare one stimulation image, render one phosphene image.
    simulator, _, render_resolution_xy, _ = create_simulator(
        params_path=params_path,
        coords_yaml=str(coords_path),
        render_resolution=render_resolution,
        view_angle=view_angle,
        fps_override=None,
        stim_scale=stim_scale,
        raster_mode=raster_mode,
        groups=int(groups),
        raster_rate_hz=float(raster_rate_hz),
        force_cpu=force_cpu,
        disable_temporal_dynamics=disable_temporal_dynamics,
    )
    simulator.reset()

    output_stem = resolve_output_stem(media_path)
    output_method = resolve_output_method(media_path, input_stage, preprocessing_method)
    output_file_stem = output_stem
    view_angle_name = resolve_view_angle_name(simulator.params)
    input_frame, stimulus_frame = prepare_stimulus_frame(
        frame,
        input_stage=input_stage,
        render_resolution_xy=render_resolution_xy,
        preprocessing_method=preprocessing_method,
        dog_sigma_low=float(dog_sigma_low),
        dog_sigma_high=float(dog_sigma_high),
        canny_low=float(canny_low),
        canny_high=float(canny_high),
        use_cuda=bool(use_cuda),
    )
    stim_pattern = sample_stimulus_pattern(simulator, stimulus_frame)
    phosphene_frame = render_phosphene_frame(simulator, stim_pattern)

    media_out_dir = output_dir / resolve_input_type(media_path) / output_stem / electrode_grid_name / view_angle_name / output_method
    media_out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(media_out_dir / f"{output_file_stem}_phosphene.png"), phosphene_frame)
    print(f"saved phosphene image outputs to: {media_out_dir}")
    print(f"electrode grid: {electrode_grid_name} ({coords_path})")
    print(f"saved phosphene image to: {media_out_dir / f'{output_file_stem}_phosphene.png'}")


@torch.inference_mode()
def process_video(
    media_path: Path,
    output_dir: Path,
    *,
    input_stage: str,
    preprocessing_method: str,
    render_resolution: int | None,
    view_angle: float | None,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float,
    canny_high: float,
    use_cuda: bool,
    params_path: str,
    coords_yaml: str | None,
    stim_scale: float | None,
    raster_mode: str,
    groups: int,
    raster_rate_hz: float,
    force_cpu: bool,
    disable_temporal_dynamics: bool,
    max_frames: int,
    max_seconds: float,
    save_comparison_video: bool,
) -> None:
    process_start_time = time.perf_counter()
    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {media_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0 and total_frames > 0:
        total_frames = min(total_frames, int(max_frames))
    if max_seconds > 0.0 and fps > 0.0:
        seconds_frames = int(math.ceil(max_seconds * fps))
        if total_frames > 0:
            total_frames = min(total_frames, seconds_frames)
        else:
            total_frames = seconds_frames

    coords_path = resolve_coords_yaml(coords_yaml)
    electrode_grid_name = resolve_electrode_grid_name(coords_path)

    # For videos we override the simulator FPS with the file FPS so the temporal
    # dynamics and optional raster timing advance at the same pace as the video.
    # Video mode reuses the same simulator across all frames so temporal
    # dynamics and raster scheduling evolve continuously over the clip.
    simulator, params, render_resolution_xy, phosphene_bounds_deg = create_simulator(
        params_path=params_path,
        coords_yaml=str(coords_path),
        render_resolution=render_resolution,
        view_angle=view_angle,
        fps_override=float(fps),
        stim_scale=stim_scale,
        raster_mode=raster_mode,
        groups=int(groups),
        raster_rate_hz=float(raster_rate_hz),
        force_cpu=force_cpu,
        disable_temporal_dynamics=disable_temporal_dynamics,
    )
    simulator.reset()

    output_stem = resolve_output_stem(media_path)
    output_method = resolve_output_method(media_path, input_stage, preprocessing_method)
    output_file_stem = output_stem
    view_angle_name = resolve_view_angle_name(params)
    media_out_dir = output_dir / resolve_input_type(media_path) / output_stem / electrode_grid_name / view_angle_name / output_method
    media_out_dir.mkdir(parents=True, exist_ok=True)

    phosphene_writer = None
    comparison_writer = None
    frame_idx = 0
    simulation_elapsed_s = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and frame_idx >= int(max_frames):
                break
            if max_seconds > 0.0 and fps > 0.0 and frame_idx >= int(math.ceil(max_seconds * fps)):
                break

            # Convert the current source frame into the stimulation image that
            # will be sampled at phosphene receptive fields.
            input_frame, stimulus_frame = prepare_stimulus_frame(
                frame,
                input_stage=input_stage,
                render_resolution_xy=render_resolution_xy,
                preprocessing_method=preprocessing_method,
                dog_sigma_low=float(dog_sigma_low),
                dog_sigma_high=float(dog_sigma_high),
                canny_low=float(canny_low),
                canny_high=float(canny_high),
                use_cuda=bool(use_cuda),
            )
            # Core rendering loop:
            # source frame -> stimulation mask -> sampled electrode amplitudes
            # -> simulator update -> phosphene frame.
            stim_pattern = sample_stimulus_pattern(simulator, stimulus_frame)
            simulation_start_time = time.perf_counter()
            phosphene_frame = render_phosphene_frame(simulator, stim_pattern)
            simulation_elapsed_s += time.perf_counter() - simulation_start_time

            if phosphene_writer is None:
                height, width = phosphene_frame.shape[:2]
                phosphene_writer = open_video_writer(
                    media_out_dir / f"{output_file_stem}_phosphenes.mp4",
                    fps,
                    (width, height),
                    is_color=False,
                )

            # The phosphene-only MP4 is the main deliverable of this script.
            phosphene_writer.write(phosphene_frame)

            comparison_frame = None
            if bool(save_comparison_video):
                comparison_frame = build_comparison_frame(
                    input_frame,
                    stimulus_frame,
                    phosphene_frame,
                    fov_bounds_deg=phosphene_bounds_deg,
                    params=params,
                    input_stage=input_stage,
                )

            if bool(save_comparison_video):
                # Debugging export that shows the input, stimulation mask, and
                # rendered percept side by side.
                if comparison_writer is None:
                    if comparison_frame is None:
                        comparison_frame = build_comparison_frame(
                            input_frame,
                            stimulus_frame,
                            phosphene_frame,
                            fov_bounds_deg=phosphene_bounds_deg,
                            params=params,
                            input_stage=input_stage,
                        )
                    height, width = comparison_frame.shape[:2]
                    comparison_writer = open_video_writer(
                        media_out_dir / f"{output_file_stem}_comparison.mp4",
                        fps,
                        (width, height),
                        is_color=False,
                    )
                comparison_writer.write(comparison_frame)

            frame_idx += 1
            if total_frames > 0:
                progress = int((frame_idx / total_frames) * 100)
                print(
                    f"  {media_path.name}: {frame_idx}/{total_frames} frames ({progress}%)",
                    end="\r",
                    flush=True,
                )
            else:
                print(
                    f"  {media_path.name}: {frame_idx} frames",
                    end="\r",
                    flush=True,
                )
    finally:
        cap.release()
        if phosphene_writer is not None:
            phosphene_writer.release()
        if comparison_writer is not None:
            comparison_writer.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames were processed from: {media_path}")

    total_elapsed_s = time.perf_counter() - process_start_time
    print()
    print(f"saved phosphene video outputs to: {media_out_dir}")
    print(f"electrode grid: {electrode_grid_name} ({coords_path})")
    print(
        f"simulation summary: resolution={render_resolution_xy[0]}x{render_resolution_xy[1]} | "
        f"frames={frame_idx} | simulate={format_duration(simulation_elapsed_s)} | "
        f"total={format_duration(total_elapsed_s)}"
    )


@torch.inference_mode()
def main(argv: list[str] | None = None) -> None:
    # CLI entry point. In practice, the most important user choices are:
    # - --input
    # - --input-stage
    # - --preprocessing-method (for original inputs)
    # - --raster-mode
    # - --output-dir
    # Comparison video export is enabled by default because it is the easiest
    # way to verify what source signal produced the phosphene output.
    parser = argparse.ArgumentParser(
        prog="dynaphos render",
        description="Visualize phosphene rendering for images or videos using the dynaphos simulator pipeline."
    )
    parser.add_argument(
        "--input",
        "--video",
        dest="input_path",
        type=str,
        default="videos",
        help="Image, video, or directory containing media files.",
    )
    parser.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=str,
        default="results/phosphenes",
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--media-type",
        choices=("auto", "image", "video"),
        default="auto",
        help="Restrict processing to images, videos, or infer from each input path.",
    )
    parser.add_argument(
        "--input-stage",
        type=str,
        default="original",
        choices=["original", "preprocessed"],
        help="Whether the input is a raw recording or an already preprocessed single-channel stimulation video.",
    )
    parser.add_argument(
        "--preprocessing-method",
        type=str,
        default="dog",
        choices=list(PREPROCESS_METHODS),
        help="Preprocessing method used when --input-stage=original.",
    )
    parser.add_argument("--params", type=str, default=str(package_file("params")))
    parser.add_argument(
        "--coords-yaml",
        "--coords_yaml",
        dest="coords_yaml",
        type=str,
        default=None,
        help="Electrode coordinates YAML. Defaults to the packaged 400 um array.",
    )
    parser.add_argument(
        "--render-resolution",
        "--render_resolution",
        dest="render_resolution",
        type=int,
        default=None,
        help="Override simulator resolution. Defaults to the value already stored in params.yaml.",
    )
    parser.add_argument(
        "--view-angle",
        "--view_angle",
        dest="view_angle",
        type=float,
        default=None,
        help="Override run.view_angle from params.yaml. Outputs are grouped by this value.",
    )
    parser.add_argument(
        "--raster-mode",
        "--raster",
        dest="raster_mode",
        type=str,
        default="none",
        choices=list(RASTER_PATTERNS),
        help="Raster strategy applied during rendering. Use 'none' for the simplest first run.",
    )
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--raster-rate-hz", "--raster_rate_hz", dest="raster_rate_hz", type=float, default=1.0)
    parser.add_argument("--dog-sigma-low", type=float, default=2.0)
    parser.add_argument("--dog-sigma-high", type=float, default=6.0)
    parser.add_argument(
        "--stim-scale",
        type=float,
        default=None,
        help=(
            "Scale factor applied to sampling.stimulus_scale from params.yaml. "
            "Example: --stim-scale 0.8 keeps 80%% of the original value."
        ),
    )
    parser.add_argument("--canny-low", type=float, default=75.0)
    parser.add_argument("--canny-high", type=float, default=170.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the full video.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Process only the first N seconds of video. 0 processes the full video.",
    )
    parser.add_argument(
        "--save-comparison-video",
        dest="save_comparison_video",
        action="store_true",
        default=True,
        help="Save a side-by-side input/stimulus/phosphene comparison video.",
    )
    parser.add_argument(
        "--skip-comparison-video",
        dest="save_comparison_video",
        action="store_false",
        help="Only save the phosphene video.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use OpenCV CUDA for DoG preprocessing when available.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force the simulator to run on CPU even when CUDA is available.",
    )
    parser.add_argument(
        "--keep-temporal-dynamics",
        action="store_true",
        help="Preserve the simulator temporal dynamics instead of flattening them for visualization.",
    )
    args = parser.parse_args(argv)

    use_cuda = bool(
        args.use_cuda
        and hasattr(cv2, "cuda")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
    )
    if args.use_cuda and not use_cuda:
        print("CUDA was requested for preprocessing but is not available. Falling back to CPU.")

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    disable_temporal_dynamics = not bool(args.keep_temporal_dynamics)

    # The script accepts either one file or a directory. Each video becomes a
    # phosphene MP4 in the chosen output directory.
    for media_path in resolve_media_inputs(args.input_path):
        is_image = is_image_file(media_path)
        if args.media_type == "image" and not is_image:
            raise ValueError(f"render_image received a non-image input: {media_path}")
        if args.media_type == "video" and is_image:
            raise ValueError(f"render_video received an image input: {media_path}")

        if is_image:
            process_image(
                media_path,
                output_dir,
                input_stage=str(args.input_stage),
                preprocessing_method=str(args.preprocessing_method),
                render_resolution=args.render_resolution,
                view_angle=args.view_angle,
                dog_sigma_low=float(args.dog_sigma_low),
                dog_sigma_high=float(args.dog_sigma_high),
                canny_low=float(args.canny_low),
                canny_high=float(args.canny_high),
                use_cuda=use_cuda,
                params_path=str(args.params),
                coords_yaml=args.coords_yaml,
                stim_scale=args.stim_scale,
                raster_mode=str(args.raster_mode),
                groups=int(args.groups),
                raster_rate_hz=float(args.raster_rate_hz),
                force_cpu=bool(args.force_cpu),
                disable_temporal_dynamics=disable_temporal_dynamics,
            )
        else:
            process_video(
                media_path,
                output_dir,
                input_stage=str(args.input_stage),
                preprocessing_method=str(args.preprocessing_method),
                render_resolution=args.render_resolution,
                view_angle=args.view_angle,
                dog_sigma_low=float(args.dog_sigma_low),
                dog_sigma_high=float(args.dog_sigma_high),
                canny_low=float(args.canny_low),
                canny_high=float(args.canny_high),
                use_cuda=use_cuda,
                params_path=str(args.params),
                coords_yaml=args.coords_yaml,
                stim_scale=args.stim_scale,
                raster_mode=str(args.raster_mode),
                groups=int(args.groups),
                raster_rate_hz=float(args.raster_rate_hz),
                force_cpu=bool(args.force_cpu),
                disable_temporal_dynamics=disable_temporal_dynamics,
                max_frames=int(args.max_frames),
                max_seconds=float(args.seconds),
                save_comparison_video=bool(args.save_comparison_video),
            )


if __name__ == "__main__":
    main()
