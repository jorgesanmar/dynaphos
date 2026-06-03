import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos import cortex_models, utils
from dynaphos.utils import Map


def load_grid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r") as f:
        data = yaml.safe_load(f)

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    if len(x) != len(y):
        raise ValueError(f"x and y lengths differ in {path}: {len(x)} != {len(y)}")
    return x, y


def save_grid(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    data = {
        "x": [float(value) for value in x],
        "y": [float(value) for value in y],
    }
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def select_circular_visual_fov(
    x: np.ndarray,
    y: np.ndarray,
    *,
    params: dict,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if target_count > len(x):
        raise ValueError(f"target_count ({target_count}) exceeds candidate electrodes ({len(x)})")

    visual = cortex_models._map_cortical_coordinates_to_displayed_visual_field(
        params["cortex_model"],
        "right",
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    )
    x_vis, y_vis = visual.cartesian
    radius = np.hypot(np.asarray(x_vis, dtype=float), np.asarray(y_vis, dtype=float))

    order = np.lexsort((np.asarray(y, dtype=float), np.asarray(x, dtype=float), radius))
    keep = np.sort(order[:target_count])
    return np.asarray(x, dtype=float)[keep], np.asarray(y, dtype=float)[keep]


def visual_field_radius(
    x: np.ndarray,
    y: np.ndarray,
    *,
    params: dict,
) -> np.ndarray:
    visual = cortex_models._map_cortical_coordinates_to_displayed_visual_field(
        params["cortex_model"],
        "right",
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
    )
    x_vis, y_vis = visual.cartesian
    return np.hypot(np.asarray(x_vis, dtype=float), np.asarray(y_vis, dtype=float))


def visual_radius_for_target_count(
    x: np.ndarray,
    y: np.ndarray,
    *,
    params: dict,
    target_count: int,
) -> float:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if target_count > len(x):
        raise ValueError(f"target_count ({target_count}) exceeds candidate electrodes ({len(x)})")

    radius = visual_field_radius(x, y, params=params)
    order = np.lexsort((np.asarray(y, dtype=float), np.asarray(x, dtype=float), radius))
    return float(radius[order[target_count - 1]])


def select_visual_radius(
    x: np.ndarray,
    y: np.ndarray,
    *,
    params: dict,
    max_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = visual_field_radius(x, y, params=params)
    keep = radius <= float(max_radius) + 1e-12
    return np.asarray(x, dtype=float)[keep], np.asarray(y, dtype=float)[keep]


def build_valid_mask(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_x = np.asarray(sorted(set(x)), dtype=float)
    unique_y = np.asarray(sorted(set(y)), dtype=float)
    x_to_ix = {value: index for index, value in enumerate(unique_x)}
    y_to_iy = {value: index for index, value in enumerate(unique_y)}

    mask = np.zeros((len(unique_y), len(unique_x)), dtype=bool)
    for x_value, y_value in zip(x, y):
        mask[y_to_iy[y_value], x_to_ix[x_value]] = True

    return unique_x, unique_y, mask


def resample_valid_footprint(
    x: np.ndarray,
    y: np.ndarray,
    *,
    original_spacing_um: float,
    target_spacing_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    unique_x, unique_y, mask = build_valid_mask(x, y)

    scale = float(target_spacing_um) / float(original_spacing_um)
    if scale <= 0:
        raise ValueError("target_spacing_um must be positive")

    old_x_intervals = len(unique_x) - 1
    old_y_intervals = len(unique_y) - 1
    new_x_intervals = max(1, int(round(old_x_intervals / scale)))
    new_y_intervals = max(1, int(round(old_y_intervals / scale)))

    new_x_values = np.linspace(unique_x[0], unique_x[-1], new_x_intervals + 1)
    new_y_values = np.linspace(unique_y[0], unique_y[-1], new_y_intervals + 1)

    old_x_index = np.linspace(0.0, float(old_x_intervals), new_x_intervals + 1)
    old_y_index = np.linspace(0.0, float(old_y_intervals), new_y_intervals + 1)

    valid_rows = np.flatnonzero(mask.any(axis=1))
    left_edge = np.full(mask.shape[0], np.nan, dtype=float)
    right_edge = np.full(mask.shape[0], np.nan, dtype=float)
    for row in valid_rows:
        valid_cols = np.flatnonzero(mask[row])
        left_edge[row] = valid_cols[0]
        right_edge[row] = valid_cols[-1]

    left_interp = np.interp(old_y_index, valid_rows, left_edge[valid_rows])
    right_interp = np.interp(old_y_index, valid_rows, right_edge[valid_rows])

    new_x = []
    new_y = []
    tolerance = 1e-9
    for row_index, y_value in enumerate(new_y_values):
        row_is_valid = (
            old_x_index >= left_interp[row_index] - tolerance
        ) & (
            old_x_index <= right_interp[row_index] + tolerance
        )
        for x_value in new_x_values[row_is_valid]:
            new_x.append(x_value)
            new_y.append(y_value)

    return np.asarray(new_x, dtype=float), np.asarray(new_y, dtype=float)


def resample_valid_footprint_square_pitch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    target_pitch_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    unique_x, unique_y, mask = build_valid_mask(x, y)

    target_pitch_mm = float(target_pitch_um) / 1000.0
    if target_pitch_mm <= 0:
        raise ValueError("target_pitch_um must be positive")

    old_dx = float(np.diff(unique_x).mean())
    old_dy = float(np.diff(unique_y).mean())

    x_min = float(unique_x[0])
    x_max = float(unique_x[-1])
    y_min = float(unique_y[0])
    y_max = float(unique_y[-1])

    new_x_values = np.arange(x_min, x_max + 1e-9, target_pitch_mm)
    new_y_values = np.arange(y_min, y_max + 1e-9, target_pitch_mm)

    old_x_index = (new_x_values - x_min) / old_dx
    old_y_index = (new_y_values - y_min) / old_dy

    valid_rows = np.flatnonzero(mask.any(axis=1))
    left_edge = np.full(mask.shape[0], np.nan, dtype=float)
    right_edge = np.full(mask.shape[0], np.nan, dtype=float)
    for row in valid_rows:
        valid_cols = np.flatnonzero(mask[row])
        left_edge[row] = valid_cols[0]
        right_edge[row] = valid_cols[-1]

    left_interp = np.interp(old_y_index, valid_rows, left_edge[valid_rows])
    right_interp = np.interp(old_y_index, valid_rows, right_edge[valid_rows])

    new_x = []
    new_y = []
    tolerance = 1e-9
    for row_index, y_value in enumerate(new_y_values):
        row_is_valid = (
            old_x_index >= left_interp[row_index] - tolerance
        ) & (
            old_x_index <= right_interp[row_index] + tolerance
        )
        for x_value in new_x_values[row_is_valid]:
            new_x.append(x_value)
            new_y.append(y_value)

    return np.asarray(new_x, dtype=float), np.asarray(new_y, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create electrode-grid YAMLs with new spacings over the same valid footprint."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "config" / "grid_coords_dipole_valid.yaml",
        help="Source coordinate YAML defining the current valid footprint.",
    )
    parser.add_argument(
        "--original-spacing-um",
        type=float,
        default=800.0,
        help="Nominal spacing of the source grid in micrometers.",
    )
    parser.add_argument(
        "--target-spacing-um",
        type=float,
        nargs="+",
        default=[400.0, 1200.0],
        help="Target nominal spacings in micrometers.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "config",
        help="Directory where generated YAMLs are written.",
    )
    parser.add_argument(
        "--square-pitch-um",
        type=float,
        default=None,
        help="Generate one grid with the exact same pitch in x and y (in micrometers).",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=PROJECT_ROOT / "config" / "params.yaml",
        help="Params YAML used for visual-field mapping when circular visual selection is enabled.",
    )
    parser.add_argument(
        "--circular-visual-fov",
        action="store_true",
        help="Select electrodes by smallest visual-field radius instead of cortical x limits.",
    )
    parser.add_argument(
        "--target-counts",
        type=int,
        nargs="+",
        default=None,
        help="Electrode counts to keep for each generated target spacing.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Optional filename prefix for generated YAMLs, for example full_fov.",
    )
    parser.add_argument(
        "--same-visual-fov",
        action="store_true",
        help=(
            "Derive a noiseless circular visual-field radius from one reference grid/count, "
            "then keep that same visual radius for every target spacing."
        ),
    )
    parser.add_argument(
        "--reference-spacing-um",
        type=float,
        default=1200.0,
        help="Target spacing used to derive the shared visual FOV when --same-visual-fov is enabled.",
    )
    parser.add_argument(
        "--reference-count",
        type=int,
        default=100,
        help="Number of reference-grid electrodes used to derive the shared visual FOV.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = args.params
    if not params_path.is_absolute():
        params_path = (PROJECT_ROOT / params_path).resolve()

    source_x, source_y = load_grid(input_path)
    print(f"source: {input_path}")
    print(f"source electrodes: {len(source_x)}")

    params = None
    if args.circular_visual_fov or args.same_visual_fov:
        params = utils.load_params(str(params_path))
        params.setdefault("cortex_model", {})
        params["cortex_model"] = dict(params["cortex_model"])
        params["cortex_model"]["dropout_rate"] = 0.0
        params["cortex_model"]["noise_scale"] = 0.0

    if args.same_visual_fov and args.circular_visual_fov:
        raise ValueError("--same-visual-fov and --circular-visual-fov are mutually exclusive")
    if args.same_visual_fov and args.square_pitch_um is not None:
        raise ValueError("--same-visual-fov is only supported with --target-spacing-um")

    shared_visual_radius = None
    if args.same_visual_fov:
        reference_x, reference_y = resample_valid_footprint(
            source_x,
            source_y,
            original_spacing_um=args.original_spacing_um,
            target_spacing_um=args.reference_spacing_um,
        )
        shared_visual_radius = visual_radius_for_target_count(
            reference_x,
            reference_y,
            params=params,
            target_count=args.reference_count,
        )
        print(
            f"same visual FOV: reference {args.reference_spacing_um:g} um / "
            f"{args.reference_count} electrodes -> radius {shared_visual_radius:.6f} deg"
        )

    if args.square_pitch_um is not None:
        new_x, new_y = resample_valid_footprint_square_pitch(
            source_x,
            source_y,
            target_pitch_um=args.square_pitch_um,
        )
        if args.circular_visual_fov:
            if not args.target_counts or len(args.target_counts) != 1:
                raise ValueError("--square-pitch-um with --circular-visual-fov requires exactly one --target-counts value")
            new_x, new_y = select_circular_visual_fov(
                new_x,
                new_y,
                params=params,
                target_count=args.target_counts[0],
            )
        spacing_label = int(args.square_pitch_um) if args.square_pitch_um.is_integer() else args.square_pitch_um
        prefix = args.output_prefix or f"{input_path.stem}_square"
        output_path = output_dir / f"{prefix}_{spacing_label}um.yaml"
        save_grid(output_path, new_x, new_y)

        unique_x = np.unique(new_x)
        unique_y = np.unique(new_y)
        x_pitch = np.diff(unique_x).mean() if len(unique_x) > 1 else float("nan")
        y_pitch = np.diff(unique_y).mean() if len(unique_y) > 1 else float("nan")
        print(
            f"square {args.square_pitch_um:g} um -> {len(new_x)} electrodes | "
            f"x pitch {x_pitch:.6f} mm | y pitch {y_pitch:.6f} mm | saved {output_path}"
        )
        return

    if args.circular_visual_fov and args.target_counts is not None and len(args.target_counts) != len(args.target_spacing_um):
        raise ValueError("--target-counts must have one value per --target-spacing-um")

    for index, target_spacing_um in enumerate(args.target_spacing_um):
        new_x, new_y = resample_valid_footprint(
            source_x,
            source_y,
            original_spacing_um=args.original_spacing_um,
            target_spacing_um=target_spacing_um,
        )
        if args.same_visual_fov:
            new_x, new_y = select_visual_radius(
                new_x,
                new_y,
                params=params,
                max_radius=shared_visual_radius,
            )
        if args.circular_visual_fov:
            if args.target_counts is None:
                raise ValueError("--circular-visual-fov requires --target-counts")
            new_x, new_y = select_circular_visual_fov(
                new_x,
                new_y,
                params=params,
                target_count=args.target_counts[index],
            )
        spacing_label = int(target_spacing_um) if target_spacing_um.is_integer() else target_spacing_um
        prefix = args.output_prefix or input_path.stem
        output_path = output_dir / f"{prefix}_{spacing_label}um.yaml"
        save_grid(output_path, new_x, new_y)

        unique_x = np.unique(new_x)
        unique_y = np.unique(new_y)
        x_pitch = np.diff(unique_x).mean() if len(unique_x) > 1 else float("nan")
        y_pitch = np.diff(unique_y).mean() if len(unique_y) > 1 else float("nan")
        print(
            f"{target_spacing_um:g} um -> {len(new_x)} electrodes | "
            f"x pitch {x_pitch:.6f} mm | y pitch {y_pitch:.6f} mm | saved {output_path}"
        )


if __name__ == "__main__":
    main()
