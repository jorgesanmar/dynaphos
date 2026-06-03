from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos.safety.impedance import Impedance, compute_device_power
from dynaphos.safety.io import (
    STANDARD_AMPLITUDE_UA,
    STANDARD_FREQUENCY_HZ,
    STANDARD_PULSE_WIDTH_US,
)
from tools.safety.bioheat3d.bioheat3d_test_utils import (
    configure_bioheat,
    simulate_constant_power,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 60-second Bioheat3D simulation using standard stimulation "
            "settings and activity-dependent device power."
        )
    )
    parser.add_argument("--params", default="config/params.yaml")
    parser.add_argument("--coords-yaml", default="config/coords_800um.yaml")
    parser.add_argument("--duration-s", type=float, default=300.0)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--amplitude-uA", type=float, default=STANDARD_AMPLITUDE_UA)
    parser.add_argument("--frequency-hz", type=float, default=STANDARD_FREQUENCY_HZ)
    parser.add_argument("--pulse-width-us", type=float, default=STANDARD_PULSE_WIDTH_US)
    parser.add_argument(
        "--active-electrodes",
        default="all",
        help="Number of simultaneously active electrodes, or 'all'. Defaults to all.",
    )
    parser.add_argument(
        "--constant-power-mw",
        type=float,
        default=None,
        help="Override bioheat.device_constant_power_mw. Defaults to params.yaml.",
    )
    parser.add_argument(
        "--driver-efficiency",
        type=float,
        default=None,
        help="Override bioheat.driver_efficiency. Defaults to params.yaml.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", default="results/safety/bioheat3d_tests/standard_60s")
    return parser.parse_args()


def active_count_from_arg(value: str, total: int) -> int:
    if str(value).strip().lower() == "all":
        return int(total)
    count = int(value)
    if count < 0 or count > total:
        raise ValueError(f"--active-electrodes must be between 0 and {total}, or 'all'.")
    return count


def main() -> None:
    args = parse_args()
    bio, params, elec_xy_mm = configure_bioheat(
        params_path=args.params,
        coords_yaml=args.coords_yaml,
        device=args.device,
        constant_power_mw=args.constant_power_mw,
    )
    if args.driver_efficiency is not None:
        bio.driver_efficiency = float(args.driver_efficiency)

    n_elec = int(elec_xy_mm.shape[0])
    active_count = active_count_from_arg(args.active_electrodes, n_elec)
    amplitude = torch.zeros(n_elec, dtype=torch.float32, device=bio.device)
    if active_count > 0:
        amplitude[:active_count] = float(args.amplitude_uA) * 1e-6

    params.setdefault("default_stim", {})["freq_default"] = float(args.frequency_hz)
    params.setdefault("run", {})["gpu"] = None
    impedance = Impedance(params, shape=(n_elec,), verbose=False)
    impedance_ohm = impedance.state.to(device=bio.device, dtype=torch.float32)

    pulse_width = torch.full(
        (n_elec,),
        float(args.pulse_width_us) * 1e-6,
        dtype=torch.float32,
        device=bio.device,
    )
    frequency = torch.full(
        (n_elec,),
        float(args.frequency_hz),
        dtype=torch.float32,
        device=bio.device,
    )
    relative_stim_duration = float(params.get("default_stim", {}).get("relative_stim_duration", 1.0))
    stim_ic_power_W, total_power_W = compute_device_power(
        amplitude=amplitude,
        impedance_ohm=impedance_ohm,
        pulse_width_s=pulse_width,
        frequency_hz=frequency,
        relative_stim_duration=relative_stim_duration,
        constant_power_W=float(bio.device_constant_power_W),
        driver_efficiency=float(bio.driver_efficiency),
    )

    total_power_value_W = float(total_power_W.detach().cpu().item())
    stim_ic_power_value_W = float(stim_ic_power_W.detach().cpu().item())
    records = simulate_constant_power(
        bio,
        power_W=total_power_value_W,
        duration_s=float(args.duration_s),
        dt_s=float(args.dt_s),
    )
    out_dir = write_outputs(
        out_dir=args.output_dir,
        stem="bioheat3d_standard_60s",
        bio=bio,
        records=records,
        metadata={
            "mode": "standard_constant_activity",
            "electrode_count": n_elec,
            "active_electrode_count": active_count,
            "amplitude_uA": float(args.amplitude_uA),
            "pulse_width_us": float(args.pulse_width_us),
            "frequency_hz": float(args.frequency_hz),
            "relative_stim_duration": relative_stim_duration,
            "constant_power_mw": float(bio.device_constant_power_W * 1e3),
            "driver_efficiency": float(bio.driver_efficiency),
            "stim_ic_power_mw": stim_ic_power_value_W * 1e3,
            "total_power_mw": total_power_value_W * 1e3,
            "duration_s": float(args.duration_s),
            "dt_s": float(args.dt_s),
        },
    )
    final = records[-1]
    print(f"Saved standard Bioheat3D test to: {out_dir}")
    print(
        "power: "
        f"PstimIC={stim_ic_power_value_W * 1e3:.6f} mW | "
        f"Pconstant={bio.device_constant_power_W * 1e3:.6f} mW | "
        f"Ptotal={total_power_value_W * 1e3:.6f} mW"
    )
    print(
        "final: "
        f"t={final['time_s']:.3f}s | "
        f"active={active_count}/{n_elec} | "
        f"max_dT={final['max_dT_C']:.6f} C | "
        f"source_mean_dT={final['source_plane_mean_dT_C']:.6f} C"
    )


if __name__ == "__main__":
    main()
