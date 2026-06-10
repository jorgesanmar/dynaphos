from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynaphos.experiment.manifest import MANIFEST_NAME, load_manifest
from dynaphos.experiment.metrics import METRICS_NAME


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    manifest_path: Path
    metrics_path: Path
    manifest: dict[str, Any]


def open_run(path: str | Path, *, require_completed: bool = True) -> RunArtifacts:
    run_dir = Path(path).resolve()
    manifest_path = run_dir / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    if require_completed and manifest.get("status") != "completed":
        raise ValueError(
            f"Run is not completed ({manifest.get('status', 'unknown')}): {run_dir}"
        )
    metrics_path = run_dir / METRICS_NAME
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Run manifest is completed but metrics.npz is missing: {run_dir}"
        )
    return RunArtifacts(run_dir, manifest_path, metrics_path, manifest)


def discover_completed_runs(root: str | Path) -> list[RunArtifacts]:
    source = Path(root).resolve()
    runs: list[RunArtifacts] = []
    for manifest_path in sorted(source.rglob(MANIFEST_NAME)):
        manifest = load_manifest(manifest_path)
        status = str(manifest.get("status", "unknown"))
        if status != "completed":
            warnings.warn(
                f"Skipping {manifest_path.parent}: run status is {status!r}.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        metrics_path = manifest_path.parent / METRICS_NAME
        if not metrics_path.exists():
            raise FileNotFoundError(
                "Run manifest is completed but metrics.npz is missing: "
                f"{manifest_path.parent}"
            )
        runs.append(
            RunArtifacts(
                run_dir=manifest_path.parent,
                manifest_path=manifest_path,
                metrics_path=metrics_path,
                manifest=manifest,
            )
        )
    return runs
