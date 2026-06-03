from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.safety.common import SimulationCase, add_common_cli, run_cases
from tools.safety.experiment_matrix import build_cases as build_matrix_cases


def build_cases() -> list[SimulationCase]:
    return build_matrix_cases(blocks=("preprocessing",))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the SANPO preprocessing safety sweep.")
    add_common_cli(parser)
    run_cases(build_cases(), parser.parse_args())


if __name__ == "__main__":
    main()
