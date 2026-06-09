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
    return parser


def _existing_metrics(run_dir: Path) -> Path:
    preferred = run_dir / "metrics.npz"
    if preferred.exists():
        return preferred
    legacy = run_dir / "safety_metrics.npz"
    if legacy.exists():
        return legacy
    raise FileNotFoundError(f"No metrics.npz or safety_metrics.npz found under: {run_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
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
            results = run_sweep(sweep)
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
    except (FileNotFoundError, FileExistsError, ImportError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 1


if __name__ == "__main__":
    sys.exit(main())
