from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dynaphos.config import load_experiment, load_sweep
from dynaphos.experiment import run_experiment, run_sweep
from dynaphos.paths import package_file
from dynaphos.reporting import write_report_bundle
from dynaphos.strategies import available_strategies, load_strategy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynaphos",
        description="Run DynaPhos stimulation experiments and safety reports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an experiment configuration without running it.",
    )
    validate_parser.add_argument("config", type=Path)

    run_parser = subparsers.add_parser("run", help="Run one experiment.")
    run_parser.add_argument("config", type=Path)

    sweep_parser = subparsers.add_parser("sweep", help="Run a configuration sweep.")
    sweep_parser.add_argument("config", type=Path)
    sweep_parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs with a completed manifest and canonical metrics.",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate or refresh reports for an existing run directory.",
    )
    report_parser.add_argument("run_dir", type=Path)
    report_parser.add_argument("--safety-limits", type=Path, default=None)
    report_parser.add_argument("--no-figures", action="store_true")

    strategies_parser = subparsers.add_parser("strategies", help="Inspect strategies.")
    strategy_subparsers = strategies_parser.add_subparsers(
        dest="strategies_command",
        required=True,
    )
    strategy_subparsers.add_parser("list", help="List built-in strategies.")

    render_parser = subparsers.add_parser("render", help="Render phosphene media.")
    render_subparsers = render_parser.add_subparsers(
        dest="render_command",
        required=True,
    )
    render_subparsers.add_parser(
        "image",
        help="Render one or more images.",
        add_help=False,
    )
    render_subparsers.add_parser(
        "video",
        help="Render one or more videos.",
        add_help=False,
    )

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Inspect media preprocessing.",
    )
    preprocess_subparsers = preprocess_parser.add_subparsers(
        dest="preprocess_command",
        required=True,
    )
    preprocess_subparsers.add_parser(
        "compare",
        help="Compare preprocessing methods.",
        add_help=False,
    )

    electrodes_parser = subparsers.add_parser(
        "electrodes",
        help="Inspect electrode arrays.",
    )
    electrodes_subparsers = electrodes_parser.add_subparsers(
        dest="electrodes_command",
        required=True,
    )
    electrodes_subparsers.add_parser(
        "plot",
        help="Plot electrode arrays and mappings.",
        add_help=False,
    )

    study_parser = subparsers.add_parser(
        "study",
        help="Regenerate phase 1-3 comparative analyses.",
    )
    study_subparsers = study_parser.add_subparsers(dest="study_command", required=True)
    phase1 = study_subparsers.add_parser("phase1")
    phase1.add_argument("results_root", type=Path)
    phase1.add_argument("--output-root", type=Path, default=None)
    phase2 = study_subparsers.add_parser("phase2")
    phase2.add_argument("results_root", type=Path)
    phase2.add_argument("--phase1-root", type=Path, required=True)
    phase2.add_argument("--output-root", type=Path, default=None)
    phase3 = study_subparsers.add_parser("phase3")
    phase3.add_argument("results_root", type=Path)
    phase3.add_argument("--phase1-root", type=Path, required=True)
    phase3.add_argument("--phase2-root", type=Path, required=True)
    phase3.add_argument("--output-root", type=Path, default=None)
    for study_command in (phase1, phase2, phase3):
        study_command.add_argument(
            "--format",
            default="png",
            choices=("png", "pdf", "svg"),
        )
    return parser


def _existing_metrics(run_dir: Path) -> Path:
    metrics = run_dir / "metrics.npz"
    if not metrics.exists():
        raise FileNotFoundError(f"No canonical metrics.npz found under: {run_dir}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, forwarded = parser.parse_known_args(argv)
    forwards_arguments = args.command in {"render", "preprocess", "electrodes"}
    if forwarded and not forwards_arguments:
        parser.error(f"unrecognized arguments: {' '.join(forwarded)}")
    try:
        if args.command == "validate":
            config = load_experiment(args.config)
            load_strategy(config.strategy)
            input_path = (
                config.input.path
                if config.input.path is not None
                else package_file(f"fixture_{config.input.builtin}")
            )
            if not input_path.exists():
                raise FileNotFoundError(f"Input does not exist: {input_path}")
            print(f"Valid experiment: {args.config.resolve()}")
            return 0

        if args.command == "run":
            result = run_experiment(args.config)
            print(f"{result.report.status_label}: {result.run_dir}")
            print(f"Report: {result.report_html_path}")
            return 0

        if args.command == "sweep":
            sweep = load_sweep(args.config)
            results = run_sweep(sweep, resume=bool(args.resume))
            print(f"Completed {len(results)} experiment(s).")
            for result in results:
                print(f"{result.report.status_label}: {result.run_dir}")
            return 0

        if args.command == "report":
            run_dir = args.run_dir.resolve()
            report = write_report_bundle(
                run_dir,
                metrics_path=_existing_metrics(run_dir),
                safety_limits_path=args.safety_limits,
                write_figures=not args.no_figures,
            )
            print(f"{report.status_label}: {run_dir / 'report.html'}")
            return 0

        if args.command == "strategies" and args.strategies_command == "list":
            for name in available_strategies():
                print(name)
            return 0

        if args.command == "render":
            from dynaphos.media.rendering import main as render_main

            render_main(["--media-type", args.render_command, *forwarded])
            return 0

        if args.command == "preprocess" and args.preprocess_command == "compare":
            from dynaphos.media.comparison import main as compare_main

            compare_main(forwarded)
            return 0

        if args.command == "electrodes" and args.electrodes_command == "plot":
            from dynaphos.electrodes.visualization import main as electrodes_main

            electrodes_main(forwarded)
            return 0

        if args.command == "study":
            from dynaphos.studies import (
                run_phase1_analysis,
                run_phase2_analysis,
                run_phase3_analysis,
            )

            if args.study_command == "phase1":
                written = run_phase1_analysis(
                    args.results_root,
                    args.output_root,
                    image_format=args.format,
                )
            elif args.study_command == "phase2":
                written = run_phase2_analysis(
                    args.results_root,
                    args.phase1_root,
                    args.output_root,
                    image_format=args.format,
                )
            else:
                written = run_phase3_analysis(
                    args.results_root,
                    args.phase1_root,
                    args.phase2_root,
                    args.output_root,
                    image_format=args.format,
                )
            print(f"Wrote {len(written)} study output(s).")
            return 0
    except (
        FileNotFoundError,
        FileExistsError,
        ImportError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    sys.exit(main())
