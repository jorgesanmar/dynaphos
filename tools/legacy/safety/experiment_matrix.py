from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import yaml

from tools.safety.common import (
    PROJECT_ROOT,
    SimulationCase,
    normalize_raster_mode,
    resolve_repo_path,
)


DEFAULT_MATRIX_CONFIG = PROJECT_ROOT / "config" / "safety_experiments.yaml"


def load_experiment_matrix(path: str | Path = DEFAULT_MATRIX_CONFIG) -> dict:
    matrix_path = resolve_repo_path(path)
    with open(matrix_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def available_blocks(path: str | Path = DEFAULT_MATRIX_CONFIG) -> tuple[str, ...]:
    matrix = load_experiment_matrix(path)
    return tuple((matrix.get("blocks", {}) or {}).keys())


def normalize_block_selection(blocks: Sequence[str] | None) -> tuple[str, ...] | None:
    if blocks is None:
        return None

    normalized: list[str] = []
    for block in blocks:
        for value in str(block).split(","):
            value_l = value.strip().lower()
            if value_l and value_l not in normalized:
                normalized.append(value_l)
    return tuple(normalized) if normalized else None


def _float_value(values: dict, key: str) -> float:
    return float(values[key])


def _case_from_values(block_name: str, block_cfg: dict, values: dict) -> SimulationCase:
    metadata = dict(values.get("metadata", {}) or {})
    metadata.setdefault("experiment_block", block_name)
    metadata.setdefault("experiment_description", str(block_cfg.get("description", "")))
    metadata.setdefault("sweep", block_name)
    metadata.setdefault("source_input_label", str(values.get("source_input_label", values["preprocessing_method"])))

    raster_mode = str(values.get("raster_mode", "none"))
    metadata.setdefault("raster_mode_normalized", normalize_raster_mode(raster_mode))

    return SimulationCase(
        block=block_name,
        run_id=str(values["run_id"]),
        video=str(values["video"]),
        coords_yaml=str(values["coords_yaml"]),
        preprocessing_method=str(values["preprocessing_method"]),
        amplitude_uA=_float_value(values, "amplitude_uA"),
        frequency_hz=_float_value(values, "frequency_hz"),
        pulse_width_us=_float_value(values, "pulse_width_us"),
        raster_mode=raster_mode,
        appearance_threshold_uA=_float_value(values, "appearance_threshold_uA"),
        source_input_label=str(values.get("source_input_label", values["preprocessing_method"])),
        internal_circuit_power_mw=_float_value(values, "internal_circuit_power_mw"),
        ic_heat_mode=str(values["ic_heat_mode"]),
        metadata=metadata,
    )


def build_cases(
    *,
    blocks: Sequence[str] | None = None,
    matrix_path: str | Path = DEFAULT_MATRIX_CONFIG,
) -> list[SimulationCase]:
    matrix = load_experiment_matrix(matrix_path)
    defaults = dict(matrix.get("defaults", {}) or {})
    block_cfgs = matrix.get("blocks", {}) or {}
    selected_blocks = normalize_block_selection(blocks)

    if selected_blocks is None:
        block_names: Iterable[str] = block_cfgs.keys()
    else:
        unknown = [block for block in selected_blocks if block not in block_cfgs]
        if unknown:
            available = ", ".join(block_cfgs.keys())
            requested = ", ".join(unknown)
            raise ValueError(f"Unknown safety experiment block(s): {requested}. Available blocks: {available}")
        block_names = selected_blocks

    cases: list[SimulationCase] = []
    seen_run_ids: set[str] = set()
    for block_name in block_names:
        block_cfg = block_cfgs[block_name] or {}
        for raw_case in block_cfg.get("cases", []) or []:
            values = {**defaults, **(raw_case or {})}
            if "run_id" not in values:
                raise ValueError(f"Missing run_id in safety experiment block '{block_name}'.")
            run_id = str(values["run_id"])
            if run_id in seen_run_ids:
                raise ValueError(f"Duplicate safety experiment run_id: {run_id}")
            seen_run_ids.add(run_id)
            cases.append(_case_from_values(block_name, block_cfg, values))

    return cases
