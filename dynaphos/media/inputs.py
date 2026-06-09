from __future__ import annotations

from pathlib import Path


def resolve_input(path: str | Path, *, relative_to: Path | None = None) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute() and relative_to is not None:
        resolved = relative_to / resolved
    resolved = resolved.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    return resolved
