from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.safety.common import (  # noqa: E402
    DEFAULT_SAFETY,
    DEFAULT_VISUALS_ROOT,
    STANDARD_AMPLITUDE_UA,
    STANDARD_FREQUENCY_HZ,
    STANDARD_PULSE_WIDTH_US,
    SimulationCase,
    add_common_cli,
    configure_device,
    iter_case_lines,
    load_yaml,
    resolve_repo_path,
    run_one_case,
    sanitize_path_part,
    write_manifest,
)
from tools.safety.run_basic_stimulation_sweep import (  # noqa: E402
    BASIC_STIMULATION_BLOCK,
    build_cases,
)
from tools.safety.recompute_temperature_from_npz import (  # noqa: E402
    recompute_temperature,
    thermal_key_is_stale,
)


DEFAULT_DERIVED_OUTPUT_ROOT = "results/safety/simulation_pipeline_derived"
DEFAULT_DERIVED_VISUALS_ROOT = "results/safety/simulation_pipeline_derived_visuals"
CEM43_KEY_PARTS = ("cem43",)


def case_output_dir(output_root: Path, case: SimulationCase) -> Path:
    return output_root / case.block / sanitize_path_part(case.run_id)


def case_npz_path(output_root: Path, case: SimulationCase) -> Path:
    return case_output_dir(output_root, case) / "safety_metrics.npz"


def is_standard_case(case: SimulationCase) -> bool:
    return (
        np.isclose(float(case.amplitude_uA), STANDARD_AMPLITUDE_UA)
        and np.isclose(float(case.frequency_hz), STANDARD_FREQUENCY_HZ)
        and np.isclose(float(case.pulse_width_us), STANDARD_PULSE_WIDTH_US)
    )


def canonical_base_case_for_grid(cases: list[SimulationCase]) -> dict[str, SimulationCase]:
    by_grid: dict[str, list[SimulationCase]] = {}
    for case in cases:
        if is_standard_case(case):
            by_grid.setdefault(str(case.coords_yaml), []).append(case)

    result = {}
    for coords_yaml, grid_cases in by_grid.items():
        preferred = [
            case for case in grid_cases
            if str((case.metadata or {}).get("sweep", "")) == "amplitude"
        ]
        result[coords_yaml] = (preferred or grid_cases)[0]
    return result


def load_npz_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as bundle:
        return {key: bundle[key] for key in bundle.files}


def write_npz(path: Path, payload: dict[str, np.ndarray], *, compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if compress:
        np.savez_compressed(tmp_path, **payload)
    else:
        np.savez(tmp_path, **payload)
    os.replace(tmp_path, path)


def randles_real_impedance(params: dict, frequency_hz: float) -> float:
    imp = params.get("impedance", {}) or {}
    rtis = float(imp.get("Rtis", 9900.0))
    cdl = float(imp.get("Cdl", 113.4e-9))
    rct = float(imp.get("Rct", 2.2e6))
    sigma_w = float(imp.get("sigma_w", 2.5e6))
    omega = 2.0 * np.pi * float(frequency_hz)
    if omega <= 0.0:
        raise ValueError(f"Frequency must be > 0 Hz, got {frequency_hz}.")
    z_faradaic = rct / (1.0 + 1j * omega * rct * cdl)
    z_warburg = sigma_w / np.sqrt(1j * omega)
    return float(np.real(rtis + z_faradaic + z_warburg))


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


def bool_value(data: dict[str, np.ndarray], key: str, default: bool = False) -> bool:
    if key not in data:
        return bool(default)
    values = np.asarray(data[key])
    if values.size == 0:
        return bool(default)
    return bool(values.reshape(-1)[0])


def same_shape_filled_like(values: np.ndarray, fill_value: float) -> np.ndarray:
    return np.full(np.asarray(values).shape, float(fill_value), dtype=np.float32)


def finite_add(values: np.ndarray, offset: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(arr)
    arr[finite] = arr[finite] + np.float32(offset)
    return arr


def scaled(values: np.ndarray, scale: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) * np.float32(scale)).astype(np.float32, copy=False)


def is_cem43_key(key: str) -> bool:
    key_l = key.lower()
    return any(part in key_l for part in CEM43_KEY_PARTS)


def is_charge_rate_key(key: str) -> bool:
    return (
        key in {
            "frame_charge_total_nC",
            "charge_per_second_total_nC_s",
            "charge_per_second_mean_per_electrode_nC_s",
            "mean_charge_per_second_per_electrode_nC_s",
            "total_charge_per_second_nC",
            "total_window_nC",
            "total_protocol_nC",
            "window_charge_total_nC",
            "protocol_charge_total_nC",
            "final_protocol_charge_per_electrode_nC",
            "protocol_charge_per_electrode_nC",
            "peak_charge_per_second_per_electrode_nC_s_exact",
        }
        or key.startswith("charge_per_second_total_nC_s_")
        or key.startswith("protocol_charge_total_nC_")
    )


def is_charge_per_phase_key(key: str) -> bool:
    return key in {
        "charge_per_phase_mean_nC",
        "charge_density_mean_uc_cm2",
        "peak_charge_per_phase_nC_exact",
        "peak_charge_density_uc_cm2_exact",
    }


def validate_scaling_assumptions(
    base_payload: dict[str, np.ndarray],
    base_case: SimulationCase,
    target_case: SimulationCase,
    *,
    allow_nonbinary_scaling: bool,
) -> None:
    ic_power_mw = scalar_value(base_payload, "internal_circuit_power_total_mW", 0.0)
    if abs(ic_power_mw) > 1e-9:
        raise RuntimeError(
            "The derived basic stimulation sweep expects no internal-circuit heat. "
            f"The base run includes internal-circuit heat ({ic_power_mw:g} mW)."
        )

    if not allow_nonbinary_scaling and not bool_value(base_payload, "input_binarized_for_safety", False):
        raise RuntimeError(
            "The base run was not marked as binarized for safety. "
            "Amplitude scaling can change active electrodes for grayscale inputs; "
            "rerun fully or pass --allow-nonbinary-scaling if this is intentional."
        )

    threshold_uA = scalar_value(base_payload, "fixed_firing_threshold_uA", np.nan)
    if np.isfinite(threshold_uA):
        if float(base_case.amplitude_uA) <= threshold_uA or float(target_case.amplitude_uA) <= threshold_uA:
            raise RuntimeError(
                "Derived amplitude scaling assumes all nonzero binary samples remain above "
                f"threshold. base={base_case.amplitude_uA:g} uA, "
                f"target={target_case.amplitude_uA:g} uA, threshold={threshold_uA:g} uA."
            )


def derive_payload(
    base_payload: dict[str, np.ndarray],
    params: dict,
    *,
    base_case: SimulationCase,
    target_case: SimulationCase,
    base_npz: Path,
    allow_nonbinary_scaling: bool,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    validate_scaling_assumptions(
        base_payload,
        base_case,
        target_case,
        allow_nonbinary_scaling=allow_nonbinary_scaling,
    )

    amp_scale = float(target_case.amplitude_uA) / float(base_case.amplitude_uA)
    pw_scale = float(target_case.pulse_width_us) / float(base_case.pulse_width_us)
    freq_scale = float(target_case.frequency_hz) / float(base_case.frequency_hz)
    base_real_z = randles_real_impedance(params, float(base_case.frequency_hz))
    target_real_z = randles_real_impedance(params, float(target_case.frequency_hz))
    impedance_scale = target_real_z / base_real_z
    charge_phase_scale = amp_scale * pw_scale
    charge_scale = charge_phase_scale * freq_scale
    power_scale = amp_scale * amp_scale * pw_scale * freq_scale * impedance_scale
    shannon_offset = 2.0 * np.log10(charge_phase_scale) if charge_phase_scale > 0 else -np.inf

    payload: dict[str, np.ndarray] = {}
    for key, value in base_payload.items():
        if is_cem43_key(key):
            continue
        if thermal_key_is_stale(key) and key != "internal_circuit_power_total_mW":
            continue
        arr = np.asarray(value)
        if key == "current_amplitude_per_electrode_uA":
            payload[key] = scaled(arr, amp_scale)
        elif key == "pulse_width_s":
            payload[key] = same_shape_filled_like(arr, float(target_case.pulse_width_us) * 1e-6)
        elif key == "pulse_frequency_hz":
            payload[key] = same_shape_filled_like(arr, float(target_case.frequency_hz))
        elif key == "electrode_impedance_ohm":
            payload[key] = scaled(arr, impedance_scale)
        elif key == "peak_current_amplitude_uA_exact":
            payload[key] = scaled(arr, amp_scale)
        elif is_charge_rate_key(key):
            payload[key] = scaled(arr, charge_scale)
        elif is_charge_per_phase_key(key):
            payload[key] = scaled(arr, charge_phase_scale)
        elif key in {"shannon_k_mean", "peak_shannon_k_exact"}:
            payload[key] = finite_add(arr, shannon_offset)
        else:
            payload[key] = arr.copy()

    payload["derived_from_npz"] = np.asarray(str(base_npz))
    payload["derived_from_run_id"] = np.asarray(base_case.run_id)
    payload["derived_amplitude_scale"] = np.asarray(amp_scale, dtype=np.float32)
    payload["derived_charge_per_phase_scale"] = np.asarray(charge_phase_scale, dtype=np.float32)
    payload["derived_charge_rate_scale"] = np.asarray(charge_scale, dtype=np.float32)
    payload["derived_impedance_scale"] = np.asarray(impedance_scale, dtype=np.float32)
    payload["derived_power_input_scale"] = np.asarray(power_scale, dtype=np.float32)
    payload["derived_temperature_recomputed"] = np.asarray(True, dtype=np.bool_)
    payload["derived_cem43_disabled"] = np.asarray(True, dtype=np.bool_)

    scales = {
        "amplitude": amp_scale,
        "charge_per_phase": charge_phase_scale,
        "charge_rate": charge_scale,
        "impedance": impedance_scale,
        "power_input": power_scale,
        "shannon_offset": shannon_offset,
        "base_real_impedance_ohm": base_real_z,
        "target_real_impedance_ohm": target_real_z,
    }
    return payload, scales


def safe_nanmax(values: object) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else float("nan")


def safe_last(values: object) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    return float(finite[-1]) if finite.size else float("nan")


def write_derived_summary(
    out_dir: Path,
    *,
    target_case: SimulationCase,
    base_case: SimulationCase,
    base_npz: Path,
    payload: dict[str, np.ndarray],
    scales: dict[str, float],
    thermal_update_interval_frames: int,
) -> None:
    time_s = np.asarray(payload.get("time_s", []), dtype=np.float64)
    lines = [
        "Derived basic stimulation simulation",
        f"run_id={target_case.run_id}",
        f"base_run_id={base_case.run_id}",
        f"base_npz={base_npz}",
        f"video={resolve_repo_path(target_case.video)}",
        f"coords_yaml={resolve_repo_path(target_case.coords_yaml)}",
        f"amplitude_uA={float(target_case.amplitude_uA):.6f}",
        f"frequency_hz={float(target_case.frequency_hz):.6f}",
        f"pulse_width_us={float(target_case.pulse_width_us):.6f}",
        f"frames={int(time_s.size)}",
        f"duration_s={safe_last(time_s):.6f}",
        "CEM43 disabled: temperatures remain below the active range used for thermal dose.",
        "Electrical traces are derived from the canonical base run.",
        "Thermal metrics are recomputed by running Bioheat2D/Pennes independently from the derived power input.",
        f"thermal_update_interval_frames={int(thermal_update_interval_frames)}",
        "",
        "Scale factors",
        f"amplitude_scale={scales['amplitude']:.9g}",
        f"charge_per_phase_scale={scales['charge_per_phase']:.9g}",
        f"charge_rate_scale={scales['charge_rate']:.9g}",
        f"impedance_scale={scales['impedance']:.9g}",
        f"power_input_scale={scales['power_input']:.9g}",
        f"shannon_k_offset={scales['shannon_offset']:.9g}",
        f"base_real_impedance_ohm={scales['base_real_impedance_ohm']:.9g}",
        f"target_real_impedance_ohm={scales['target_real_impedance_ohm']:.9g}",
        "",
        "Derived results",
        f"peak_current_amplitude_uA={safe_nanmax(payload.get('current_amplitude_per_electrode_uA', [])):.6f}",
        f"peak_charge_per_phase_nC={safe_nanmax(payload.get('peak_charge_per_phase_nC_exact', [])):.6f}",
        f"peak_charge_density_uC_cm2={safe_nanmax(payload.get('peak_charge_density_uc_cm2_exact', [])):.6f}",
        f"peak_total_charge_per_second_nC_s={safe_nanmax(payload.get('charge_per_second_total_nC_s', [])):.6f}",
        f"final_total_protocol_charge_nC={safe_last(payload.get('protocol_charge_total_nC', [])):.6f}",
        f"peak_max_dT_C={safe_nanmax(payload.get('max_dT', [])):.6f}",
        f"final_max_dT_C={safe_last(payload.get('max_dT', [])):.6f}",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_derived_manifest(
    out_dir: Path,
    *,
    case: SimulationCase,
    base_case: SimulationCase,
    params_path: Path,
    safety_yaml: Path,
    output_root: Path,
    max_frames: int,
    groups: int,
    phosphene_mode: str,
    scales: dict[str, float],
) -> None:
    metadata = dict(case.metadata)
    metadata.update(
        {
            "derived": True,
            "derived_from_run_id": base_case.run_id,
            "derived_cem43_disabled": True,
            "derived_temperature_recomputed": True,
            "derived_power_input_scale": float(scales["power_input"]),
        }
    )
    derived_case = replace(case, metadata=metadata)
    write_manifest(
        out_dir,
        case=derived_case,
        params_path=params_path,
        safety_yaml=safety_yaml,
        output_root=output_root,
        max_frames=max_frames,
        groups=groups,
        phosphene_mode=phosphene_mode,
        enable_cem43=False,
    )


def derive_one_case(
    *,
    base_npz: Path,
    base_case: SimulationCase,
    target_case: SimulationCase,
    params: dict,
    params_path: Path,
    safety_yaml: Path,
    output_root: Path,
    groups: int,
    max_frames: int,
    phosphene_mode: str,
    allow_nonbinary_scaling: bool,
    thermal_update_interval_frames: int,
    device,
    show_progress: bool,
    overwrite: bool,
    compress: bool,
) -> Path:
    out_dir = case_output_dir(output_root, target_case)
    npz_path = out_dir / "safety_metrics.npz"
    if npz_path.exists() and not overwrite:
        return npz_path

    base_payload = load_npz_payload(base_npz)
    payload, scales = derive_payload(
        base_payload,
        params,
        base_case=base_case,
        target_case=target_case,
        base_npz=base_npz,
        allow_nonbinary_scaling=allow_nonbinary_scaling,
    )
    thermal_payload = recompute_temperature(
        payload,
        params,
        resolve_repo_path(target_case.coords_yaml),
        device,
        manifest={
            "pulse_width_us": float(target_case.pulse_width_us),
            "frequency_hz": float(target_case.frequency_hz),
            "internal_circuit_power_mw": float(target_case.internal_circuit_power_mw),
        },
        thermal_update_interval_frames=int(thermal_update_interval_frames),
        enable_cem43=False,
        progress_label=sanitize_path_part(target_case.run_id),
        show_progress=show_progress,
    )
    payload.update(thermal_payload)
    write_derived_manifest(
        out_dir,
        case=target_case,
        base_case=base_case,
        params_path=params_path,
        safety_yaml=safety_yaml,
        output_root=output_root,
        max_frames=max_frames,
        groups=groups,
        phosphene_mode=phosphene_mode,
        scales=scales,
    )
    write_derived_summary(
        out_dir,
        target_case=target_case,
        base_case=base_case,
        base_npz=base_npz,
        payload=payload,
        scales=scales,
        thermal_update_interval_frames=int(thermal_update_interval_frames),
    )
    write_npz(npz_path, payload, compress=compress)
    return npz_path


def build_runner_args_dict(args: argparse.Namespace, params_path: Path, safety_yaml: Path,
                           output_root: Path, visuals_root: Path) -> dict[str, object]:
    return {
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
        "preview_seconds": float(args.preview_seconds),
        "preview_policy": str(args.preview_policy),
        "force_cpu": bool(args.force_cpu),
        "enable_cem43": False,
    }


def ensure_base_runs(
    *,
    base_cases: dict[str, SimulationCase],
    args: argparse.Namespace,
    params_path: Path,
    safety_yaml: Path,
    output_root: Path,
    visuals_root: Path,
    base_root: Path,
) -> dict[str, Path]:
    base_npz_by_grid: dict[str, Path] = {}
    runner_args = build_runner_args_dict(args, params_path, safety_yaml, output_root, visuals_root)
    total = len(base_cases)
    for index, (coords_yaml, base_case) in enumerate(base_cases.items(), start=1):
        existing_base = case_npz_path(base_root, base_case)
        output_base = case_npz_path(output_root, base_case)
        if existing_base.exists() and not bool(args.force_base):
            base_npz_by_grid[coords_yaml] = existing_base
            print(f"[base {index}/{total}] reuse {existing_base}")
            continue
        if base_root != output_root and not existing_base.exists():
            raise FileNotFoundError(
                f"Missing base run under --base-root: {existing_base}. "
                "Either omit --base-root so the script can run it, or point to a root that contains it."
            )
        if output_base.exists() and not bool(args.force_base):
            base_npz_by_grid[coords_yaml] = output_base
            print(f"[base {index}/{total}] reuse {output_base}")
            continue

        print(f"[base {index}/{total}] running {base_case.run_id}")
        run_one_case(index, total, base_case, runner_args)
        base_npz_by_grid[coords_yaml] = output_base
    return base_npz_by_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run/reuse one standard basic-stimulation simulation per grid and derive "
            "the remaining amplitude/frequency/pulse-width electrical traces, then "
            "recompute Bioheat2D/Pennes independently for each derived power input."
        )
    )
    add_common_cli(parser)
    parser.set_defaults(
        output_root=DEFAULT_DERIVED_OUTPUT_ROOT,
        visuals_root=DEFAULT_DERIVED_VISUALS_ROOT,
        preview_seconds=0.0,
        preview_policy="none",
        phosphene_mode="visual",
    )
    parser.add_argument(
        "--base-root",
        default=None,
        help=(
            "Optional output root containing the canonical standard runs to reuse. "
            "Defaults to --output-root."
        ),
    )
    parser.add_argument(
        "--overwrite-derived",
        action="store_true",
        help="Overwrite derived safety_metrics.npz files if they already exist.",
    )
    parser.add_argument(
        "--force-base",
        action="store_true",
        help="Rerun canonical standard base simulations even if their outputs exist.",
    )
    parser.add_argument(
        "--allow-nonbinary-scaling",
        action="store_true",
        help="Allow proportional amplitude scaling even when the base input was not binarized.",
    )
    parser.add_argument(
        "--compress-derived",
        action="store_true",
        help="Write derived NPZ files with numpy compression to save disk space.",
    )
    parser.add_argument(
        "--no-thermal-progress",
        action="store_true",
        help="Disable per-case progress bars while recomputing Bioheat2D for derived cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.enable_cem43):
        raise ValueError("The derived sweep intentionally disables CEM43. Omit --enable-cem43.")

    cases = build_cases()
    base_cases = canonical_base_case_for_grid(cases)
    params_path = resolve_repo_path(args.params)
    safety_yaml = resolve_repo_path(args.safety_yaml)
    output_root = resolve_repo_path(args.output_root)
    visuals_root = resolve_repo_path(args.visuals_root)
    base_root = resolve_repo_path(args.base_root) if args.base_root else output_root

    if safety_yaml.name != Path(DEFAULT_SAFETY).name:
        raise ValueError(f"This pipeline must use config/safety.yaml, got: {safety_yaml}")

    if bool(args.dry_run):
        print(f"Canonical base simulations: {len(base_cases)}")
        for line in iter_case_lines(base_cases.values()):
            print(line)
        print(f"Derived/copy outputs: {len(cases)}")
        for line in iter_case_lines(cases):
            print(line)
        return

    params = load_yaml(params_path)
    device = configure_device(params, force_cpu=bool(args.force_cpu))
    if args.thermal_update_interval_frames is None:
        args.thermal_update_interval_frames = 10

    output_root.mkdir(parents=True, exist_ok=True)
    visuals_root.mkdir(parents=True, exist_ok=True)
    base_npz_by_grid = ensure_base_runs(
        base_cases=base_cases,
        args=args,
        params_path=params_path,
        safety_yaml=safety_yaml,
        output_root=output_root,
        visuals_root=visuals_root,
        base_root=base_root,
    )

    written = 0
    skipped_same_base = 0
    skipped_existing = 0
    for index, case in enumerate(cases, start=1):
        base_case = base_cases[str(case.coords_yaml)]
        base_npz = base_npz_by_grid[str(case.coords_yaml)]
        target_npz = case_npz_path(output_root, case)
        if target_npz.resolve() == base_npz.resolve() and not bool(args.overwrite_derived):
            skipped_same_base += 1
            print(f"[{index}/{len(cases)}] base already present: {case.run_id}")
            continue
        if target_npz.exists() and not bool(args.overwrite_derived):
            skipped_existing += 1
            print(f"[{index}/{len(cases)}] exists, skipped: {case.run_id}")
            continue
        print(f"[{index}/{len(cases)}] deriving {case.run_id}")
        derive_one_case(
            base_npz=base_npz,
            base_case=base_case,
            target_case=case,
            params=params,
            params_path=params_path,
            safety_yaml=safety_yaml,
            output_root=output_root,
            groups=int(args.groups),
            max_frames=int(args.max_frames),
            phosphene_mode=str(args.phosphene_mode),
            allow_nonbinary_scaling=bool(args.allow_nonbinary_scaling),
            thermal_update_interval_frames=int(args.thermal_update_interval_frames),
            device=device,
            show_progress=not bool(args.no_thermal_progress),
            overwrite=bool(args.overwrite_derived),
            compress=bool(args.compress_derived),
        )
        written += 1

    print(
        f"Derived sweep ready: wrote {written}, "
        f"kept {skipped_same_base} canonical base run(s), skipped {skipped_existing} existing run(s)."
    )
    print(f"Output root: {output_root / BASIC_STIMULATION_BLOCK}")


if __name__ == "__main__":
    main()
