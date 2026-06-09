from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path


BUILTIN_FILES = {
    "params": "data/defaults/params.yaml",
    "safety": "data/defaults/safety.yaml",
    "coords_400um": "data/electrodes/coords_400um.yaml",
    "coords_800um": "data/electrodes/coords_800um.yaml",
    "coords_1200um": "data/electrodes/coords_1200um.yaml",
    "fixture_cat": "data/fixtures/cat.jpg",
}


def package_file(name: str) -> Path:
    try:
        relative = BUILTIN_FILES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(BUILTIN_FILES))
        raise ValueError(f"Unknown built-in resource '{name}'. Available: {choices}.") from exc

    resource = files("dynaphos").joinpath(relative)
    if resource.is_file():
        with as_file(resource) as path:
            return Path(path).resolve()

    # Source-tree compatibility while the historical config directory remains.
    project_root = Path(__file__).resolve().parents[1]
    if name == "fixture_cat":
        source_fallback = project_root / "examples" / "cat.jpg"
    else:
        source_fallback = project_root / "config" / Path(relative).name
    if source_fallback.exists():
        return source_fallback.resolve()
    raise FileNotFoundError(f"Packaged DynaPhos resource is missing: {relative}")


def resolve_user_path(path: str | Path, *, relative_to: Path | None = None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and relative_to is not None:
        resolved = relative_to / resolved
    return resolved.resolve()
