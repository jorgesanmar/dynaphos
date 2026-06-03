from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.safety.common import (
    DEFAULT_OUTPUT_ROOT,
    add_common_cli,
    load_case_from_manifest,
    resolve_repo_path,
    run_cases,
    select_worst_prior_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-select and rerun the worst SANPO safety case with random rastering.")
    add_common_cli(parser)
    args = parser.parse_args()

    output_root = resolve_repo_path(args.output_root or DEFAULT_OUTPUT_ROOT)
    selected_npz, score, ratios = select_worst_prior_run(output_root, args.safety_yaml)
    manifest_path = selected_npz.parent / "run_manifest.yaml"
    if not manifest_path.exists():
        raise RuntimeError(f"Selected run has no manifest: {manifest_path}")

    source_id = selected_npz.parent.name
    case = load_case_from_manifest(
        manifest_path,
        block="rastering_rerun",
        run_id=f"random_raster__{source_id}",
    )
    case = type(case)(
        **{
            **case.__dict__,
            "metadata": {
                **case.metadata,
                "source_safety_metrics": str(selected_npz),
                "selection_score": float(score),
                "selection_ratios": ratios,
            },
        }
    )

    if args.dry_run:
        print(f"Selected prior run: {selected_npz}")
        print(f"Selection score: {score:.6f}")
        print(yaml.safe_dump({"ratios": ratios}, sort_keys=True).strip())
    run_cases([case], args)


if __name__ == "__main__":
    main()
