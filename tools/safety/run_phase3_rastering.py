from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos.safety import visualize
from dynaphos.safety.experiments import (
    available_blocks,
    build_cases,
    normalize_block_selection,
)
from dynaphos.safety.io import add_common_cli, resolve_repo_path, run_cases


DEFAULT_CONFIG = PROJECT_ROOT / "config" / "safety_experiments_phase3_rastering.yaml"
DEFAULT_BLOCKS = ("raster_protocols",)
DEFAULT_OUTPUT_ROOT = "results/safety/simulation_pipeline"
DEFAULT_VISUALS_ROOT = "results/safety/simulation_pipeline/raster_protocols/visuals"


def run_visualizer(args: argparse.Namespace) -> None:
    input_root = resolve_repo_path(args.output_root) / "raster_protocols"
    output_root = resolve_repo_path(args.visuals_root)
    safety_yaml = resolve_repo_path(args.safety_yaml)
    records = visualize.discover_records(input_root, safety_yaml)
    if not records:
        print(f"No safety_metrics.npz files found under: {input_root}; skipping visualizer.")
        return

    print(f"Running raster protocol visualizer for {len(records)} completed simulation(s).")
    visualize.set_plot_safety_limits(visualize.load_safety_limits(safety_yaml))
    comparative_root = output_root / "comparative_visuals"
    visualize.write_summary_csv(records, output_root)
    visualize.plot_ratio_breakdown(records, comparative_root, "png", overwrite=True)
    visualize.plot_all_block_summaries(records, comparative_root, "png", overwrite=True)
    visualize.plot_comparative_suites(records, comparative_root, "png", overwrite=True)
    visualize.write_single_case_overviews(records, "png", overwrite=True, output_root=output_root)
    visualize.write_per_protocol_visuals(records, "png", overwrite=True, output_root=output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the phase-3 raster protocol safety matrix for DOG preprocessing "
            "at 60 uA and 0 mW internal-circuit power."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = build_cases(
        blocks=normalize_block_selection(args.blocks),
        matrix_path=args.matrix_config,
    )
    run_cases(cases, args)
    if not args.dry_run and not args.skip_visualizer:
        run_visualizer(args)


if __name__ == "__main__":
    main()
