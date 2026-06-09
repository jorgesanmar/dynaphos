from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos.safety.experiments import (
    available_blocks,
    build_cases,
    normalize_block_selection,
)
from dynaphos.safety.io import add_common_cli, resolve_repo_path, run_cases, sanitize_path_part
from dynaphos.safety.raster_effects import (
    remove_obsolete_phase3_plots,
    write_raster_effect_outputs,
)


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "safety_experiments_phase3_rastering.yaml"
DEFAULT_BLOCKS = ("raster_protocols",)
DEFAULT_OUTPUT_ROOT = "results/safety/simulation_pipeline"
DEFAULT_VISUALS_ROOT = "results/safety/simulation_pipeline/raster_protocols"
DEFAULT_IC_POWER_LINEARITY_SUMMARY = (
    Path(DEFAULT_OUTPUT_ROOT) / "ic_power" / "comparative_visuals" / "ic_power_linearity_summary.csv"
)


def run_visualizer(args: argparse.Namespace) -> None:
    input_root = resolve_repo_path(args.output_root) / "raster_protocols"
    visuals_root = resolve_repo_path(args.visuals_root)
    output_root = visuals_root / "comparative_visuals"
    remove_obsolete_phase3_plots(output_root)
    written = write_raster_effect_outputs(
        input_root,
        args.phase1_input_root,
        output_root,
        ic_power_linearity_summary=args.ic_power_linearity_summary,
        temperature_limit_C=args.temperature_limit_C,
        power_derating_factor=args.power_derating_factor,
        image_format=args.format,
        overwrite=True,
    )
    print(f"Wrote {len(written)} raster-effect output(s) under: {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run phase 3 raster protocols and compare them with matched phase 1 baselines."
        )
    )
    add_common_cli(parser)
    parser.set_defaults(
        output_root=DEFAULT_OUTPUT_ROOT,
        visuals_root=DEFAULT_VISUALS_ROOT,
        preview_seconds=0.0,
        preview_policy="none",
        phosphene_mode="safety_centers",
        cooldown_seconds=300.0,
        cooldown_baseline_tolerance_C=1e-3,
    )
    parser.add_argument(
        "--matrix-config",
        default=str(DEFAULT_CONFIG),
        help="YAML file defining the phase-3 raster protocol cases.",
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=list(DEFAULT_BLOCKS),
        help=(
            "Experiment blocks to run. Values may be space or comma separated. "
            f"Available in the default config: {', '.join(available_blocks(DEFAULT_CONFIG))}."
        ),
    )
    parser.add_argument(
        "--skip-visualizer",
        action="store_true",
        help="Only write safety_metrics.npz outputs; skip summary and plot generation.",
    )
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Re-run cases even when safety_metrics.npz already exists.",
    )
    parser.add_argument(
        "--phase1-input-root",
        default=str(Path(DEFAULT_OUTPUT_ROOT) / "amplitude_grid_preprocessing"),
        help="Completed phase 1 raster-off results used as matched baselines.",
    )
    parser.add_argument(
        "--ic-power-linearity-summary",
        default=str(DEFAULT_IC_POWER_LINEARITY_SUMMARY),
        help="Phase 2 grid-specific IC thermal slopes used for old/new IC power budgets.",
    )
    parser.add_argument(
        "--temperature-limit-C",
        type=float,
        default=2.0,
        help="Maximum allowed spatial mean temperature rise in degrees C.",
    )
    parser.add_argument(
        "--power-derating-factor",
        type=float,
        default=0.9,
        help="Multiplier used for the recommended IC power budget columns.",
    )
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    return parser.parse_args()


def case_output_dir(output_parent: Path, block: str, run_id: str) -> Path:
    return output_parent / sanitize_path_part(block) / sanitize_path_part(run_id)


def main() -> None:
    warnings.warn(
        "run_phase3_rastering.py is deprecated; use the sweep files under "
        "examples/studies/phase3/.",
        DeprecationWarning,
        stacklevel=2,
    )
    args = parse_args()
    if args.temperature_limit_C <= 0.0:
        raise ValueError("--temperature-limit-C must be > 0.")
    if not 0.0 < args.power_derating_factor <= 1.0:
        raise ValueError("--power-derating-factor must be in the interval (0, 1].")
    cases = build_cases(
        blocks=normalize_block_selection(args.blocks),
        matrix_path=args.matrix_config,
    )
    output_parent = resolve_repo_path(args.output_root)
    pending = [
        case
        for case in cases
        if args.include_existing
        or not (case_output_dir(output_parent, case.block, case.run_id) / "safety_metrics.npz").exists()
    ]
    skipped = [case for case in cases if case not in pending]
    if skipped:
        print(f"Skipping {len(skipped)} completed raster protocol case(s):")
        for case in skipped:
            print(f"  {case.run_id}")
    if pending:
        run_cases(pending, args)
    else:
        print("No raster protocol simulations to run.")
    if not args.dry_run and not args.skip_visualizer:
        run_visualizer(args)


if __name__ == "__main__":
    main()
