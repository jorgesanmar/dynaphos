from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos.safety.experiments import build_cases
from dynaphos.safety.ic_power_visualize import write_ic_power_temperature_visuals
from dynaphos.safety.io import add_common_cli, resolve_repo_path, run_cases, sanitize_path_part


DEFAULT_CONFIG = "config/safety_experiments_phase2_ic_power.yaml"
DEFAULT_OUTPUT_ROOT = "results/safety/simulation_pipeline"
DEFAULT_IC_POWER_ROOT = Path(DEFAULT_OUTPUT_ROOT) / "ic_power"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the IC-power safety phase and generate temperature-only comparison visuals."
    )
    add_common_cli(parser)
    parser.set_defaults(
        output_root=DEFAULT_OUTPUT_ROOT,
        visuals_root=str(DEFAULT_IC_POWER_ROOT / "previews"),
        preview_seconds=0.0,
        preview_policy="none",
        phosphene_mode="safety_centers",
        cooldown_seconds=300.0,
        cooldown_baseline_tolerance_C=1e-3,
    )
    parser.add_argument("--matrix-config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Re-run cases even when safety_metrics.npz already exists.",
    )
    parser.add_argument(
        "--visuals-output-root",
        default=str(DEFAULT_IC_POWER_ROOT / "visuals"),
        help="Directory for temperature-only IC-power comparison figures.",
    )
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--no-visuals", action="store_true")
    return parser.parse_args()


def case_output_dir(output_parent: Path, block: str, run_id: str) -> Path:
    return output_parent / sanitize_path_part(block) / sanitize_path_part(run_id)


def main() -> None:
    args = parse_args()
    args.track_electrical = False
    cases = build_cases(blocks=("ic_power",), matrix_path=args.matrix_config)
    output_parent = resolve_repo_path(args.output_root)
    pending = []
    skipped = []
    for case in cases:
        out_dir = case_output_dir(output_parent, case.block, case.run_id)
        if not args.include_existing and (out_dir / "safety_metrics.npz").exists():
            skipped.append(case)
        else:
            pending.append(case)

    if skipped:
        print(f"Skipping {len(skipped)} completed IC-power case(s):")
        for case in skipped:
            print(f"  {case.run_id}")
    if pending:
        run_cases(pending, args)
    else:
        print("No IC-power simulations to run.")

    if args.dry_run or args.no_visuals:
        return

    ic_power_root = output_parent / "ic_power"
    written = write_ic_power_temperature_visuals(
        ic_power_root,
        args.visuals_output_root,
        image_format=args.format,
        overwrite=True,
        safety_yaml=args.safety_yaml,
    )
    print(f"Wrote {len(written)} IC-power temperature output(s) under: {resolve_repo_path(args.visuals_output_root)}")


if __name__ == "__main__":
    main()
