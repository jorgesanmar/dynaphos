from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.safety.bioheat3d.bioheat3d_test_utils import (
    configure_bioheat,
    simulate_constant_power,
    write_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Bioheat3D with no stimulation and only fixed device power. "
            "This checks the temperature rise from Pconstant alone."
        )
    )
    parser.add_argument("--params", default="config/params.yaml")
    parser.add_argument("--coords-yaml", default="config/coords_800um.yaml")
    parser.add_argument("--duration-s", type=float, default=600.0)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument(
        "--power-mw",
        type=float,
        default=None,
        help="Fixed power in mW. Defaults to bioheat.device_constant_power_mw from params.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output-dir", default="results/safety/bioheat3d_tests/fixed_power")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bio, params, elec_xy_mm = configure_bioheat(
        params_path=args.params,
        coords_yaml=args.coords_yaml,
        device=args.device,
        constant_power_mw=args.power_mw,
    )
    power_mw = (
        float(args.power_mw)
        if args.power_mw is not None
        else float(params.get("bioheat", {}).get("device_constant_power_mw", 13.0))
    )
    records = simulate_constant_power(
        bio,
        power_W=power_mw * 1e-3,
        duration_s=float(args.duration_s),
        dt_s=float(args.dt_s),
    )
    out_dir = write_outputs(
        out_dir=args.output_dir,
        stem="bioheat3d_fixed_power",
        bio=bio,
        records=records,
        metadata={
            "mode": "fixed_power_no_stimulation",
            "electrode_count": int(elec_xy_mm.shape[0]),
            "fixed_power_mw": float(power_mw),
            "duration_s": float(args.duration_s),
            "dt_s": float(args.dt_s),
        },
    )
    final = records[-1]
    print(f"Saved fixed-power Bioheat3D test to: {out_dir}")
    print(
        "final: "
        f"t={final['time_s']:.3f}s | "
        f"power={power_mw:.6f} mW | "
        f"max_dT={final['max_dT_C']:.6f} C | "
        f"source_mean_dT={final['source_plane_mean_dT_C']:.6f} C"
    )


if __name__ == "__main__":
    main()
