from __future__ import annotations

from pathlib import Path

from dynaphos.experiment.artifacts import RunArtifacts, discover_completed_runs


DEFAULT_RESULTS_ROOT = Path("results/safety/simulation_pipeline")


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def sanitize_path_part(value: object) -> str:
    safe = "".join(
        character if character.isalnum() or character in ("-", "_", ".") else "_"
        for character in str(value).strip()
    )
    return safe.strip("._") or "unspecified"


def study_runs(root: str | Path, block: str) -> list[RunArtifacts]:
    return [
        run
        for run in discover_completed_runs(root)
        if str(run.manifest.get("block", "")) == block
    ]
