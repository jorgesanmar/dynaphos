from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


METRICS_NAME = "metrics.npz"


def write_metrics(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    compressed: bool = True,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    writer(output, **payload)
    return output


def load_metrics(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if source.is_dir():
        source = source / METRICS_NAME
    elif source.name != METRICS_NAME:
        raise ValueError(
            f"Expected canonical {METRICS_NAME}, got: {source.name}"
        )
    if not source.exists():
        raise FileNotFoundError(f"Missing canonical metrics: {source}")
    with np.load(source, allow_pickle=True) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}
