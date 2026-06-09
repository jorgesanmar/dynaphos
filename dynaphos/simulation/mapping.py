from __future__ import annotations

from pathlib import Path

from dynaphos.safety.runner import load_tagged_mapping


def load_mapping(params: dict, coordinates: str | Path):
    return load_tagged_mapping(params, Path(coordinates))
