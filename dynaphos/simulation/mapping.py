from __future__ import annotations

from pathlib import Path

import numpy as np

from dynaphos.simulation import cortex
from dynaphos.simulation import utils
from dynaphos.simulation.utils import Map


def load_mapping(params: dict, coordinates: str | Path):
    coordinates = Path(coordinates)
    x_raw_mm, y_raw_mm = utils.load_coordinates_from_yaml(str(coordinates))
    x_raw_mm = np.asarray(x_raw_mm, dtype=float)
    y_raw_mm = np.asarray(y_raw_mm, dtype=float)
    cortical_input = Map(x=x_raw_mm, y=y_raw_mm)
    rng = np.random.default_rng(int(params.get("run", {}).get("seed", 42)))
    mapping = cortex.get_full_field_mapping_from_cortex(
        params["cortex_model"],
        coordinates_cortex=cortical_input,
        rng=rng,
    )
    return {
        "phosphene_map": mapping.phosphene_map,
        "indices": np.asarray(mapping.indices, dtype=np.int64),
        "cortical_coordinates": mapping.cortical_coordinates,
        "grid_ids": np.asarray(mapping.grid_ids, dtype=np.int64),
        "base_indices": np.asarray(mapping.base_indices, dtype=np.int64),
    }


load_tagged_mapping = load_mapping
