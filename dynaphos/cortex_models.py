import logging
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Union

import numpy as np

from dynaphos.utils import (Map, cartesian_to_complex, polar_to_complex,
                            complex_to_polar)


@dataclass(frozen=True)
class FullFieldMapping:
    phosphene_map: Map
    indices: np.ndarray
    cortical_coordinates: Map
    grid_ids: np.ndarray
    base_indices: np.ndarray


def mirror_map_along_y_axis(coordinates: Map) -> Map:
    x_coords, y_coords = coordinates.cartesian
    return Map(
        x=-np.asarray(x_coords, dtype=np.float64),
        y=np.asarray(y_coords, dtype=np.float64),
    )


def _sort_cortical_coordinates(
        coordinates_cortex: Map,
        base_indices: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y = coordinates_cortex.cartesian
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if base_indices is None:
        base_indices = np.arange(len(x), dtype=np.int64)
    else:
        base_indices = np.asarray(base_indices, dtype=np.int64)

    sort_idx = np.lexsort((x, -y))
    return x[sort_idx], y[sort_idx], base_indices[sort_idx]


def _map_cortical_coordinates_to_displayed_visual_field(
        params: dict,
        hemisphere: str,
        x_coords: np.ndarray,
        y_coords: np.ndarray) -> Map:
    cortex_to_visual_field = get_mapping_from_cortex_to_visual_field(params)
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)
    if hemisphere not in {"right", "left"}:
        raise ValueError("hemisphere must be 'right' or 'left'.")

    # The analytical mapping is defined for one hemisphere. In displayed visual
    # coordinates, that branch corresponds to the contralateral hemifield
    # produced by the right cortical hemisphere.
    z_model = cortex_to_visual_field(cartesian_to_complex(np.abs(x_coords), y_coords))
    coordinates_display = Map(z=z_model).flip(hor=True, vert=True)
    if hemisphere == "left":
        coordinates_display = mirror_map_along_y_axis(coordinates_display)
    return coordinates_display


def _map_displayed_visual_field_to_cortex(params: dict,
                                          coordinates_visual: Map,
                                          hemisphere: str) -> Map:
    if hemisphere not in {"right", "left"}:
        raise ValueError("hemisphere must be 'right' or 'left'.")

    visual_field_to_cortex = get_mapping_from_visual_field_to_cortex(params)
    coordinates_model = coordinates_visual
    if hemisphere == "left":
        coordinates_model = mirror_map_along_y_axis(coordinates_model)
    coordinates_model = coordinates_model.flip(hor=True, vert=True)
    z_cortex = visual_field_to_cortex(coordinates_model.complex)
    x_coords, y_coords = Map(z=z_cortex).cartesian
    if hemisphere == "left":
        x_coords = -np.abs(np.asarray(x_coords, dtype=np.float64))
    else:
        x_coords = np.abs(np.asarray(x_coords, dtype=np.float64))
    return Map(x=x_coords, y=np.asarray(y_coords, dtype=np.float64))


def _prepare_cortical_grid_for_full_field(
        params: dict,
        coordinates_cortex: Map,
        *,
        base_indices: np.ndarray,
        hemisphere: str,
        grid_id: int,
        global_index_offset: int,
        rng: Optional[np.random.Generator] = None) -> FullFieldMapping:
    x, y, base_indices = _sort_cortical_coordinates(coordinates_cortex, base_indices)

    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    num_active = int(n * (1 - params['dropout_rate']))
    active_mask = rng.choice(np.arange(n), num_active, replace=False)

    x = x[active_mask]
    y = y[active_mask]
    base_indices = base_indices[active_mask]

    noise = rng.normal(scale=params['noise_scale'], size=(2, len(x)))
    x = x + noise[0]
    y = y + noise[1]

    cortex_to_visual_field = get_mapping_from_cortex_to_visual_field(params)
    z_model = cortex_to_visual_field(cartesian_to_complex(np.abs(x), y))
    r, phi = complex_to_polar(z_model)
    fov_mask = (r >= 0) & (r <= 90) & (phi > -np.pi / 2) & (phi < np.pi / 2)

    x = x[fov_mask]
    y = y[fov_mask]
    base_indices = base_indices[fov_mask]

    phosphene_map = _map_cortical_coordinates_to_displayed_visual_field(
        params,
        hemisphere,
        x,
        y,
    )
    cortical_coordinates = Map(x=x, y=y)
    global_indices = base_indices + int(global_index_offset)
    grid_ids = np.full(len(global_indices), int(grid_id), dtype=np.int64)

    return FullFieldMapping(
        phosphene_map=phosphene_map,
        indices=global_indices.astype(np.int64),
        cortical_coordinates=cortical_coordinates,
        grid_ids=grid_ids,
        base_indices=base_indices.astype(np.int64),
    )


def get_full_field_mapping_from_cortex(
        params: dict,
        coordinates_cortex: Optional[Map] = None,
        rng: Optional[np.random.Generator] = None) -> FullFieldMapping:
    if coordinates_cortex is None:
        coordinates_cortex = get_cortex_coordinates_grid(params, 32, 32, x_max=40)

    x_original, y_original, base_indices = _sort_cortical_coordinates(coordinates_cortex)
    original_cortex = Map(x=x_original, y=y_original)

    original_cortex = Map(x=np.abs(x_original), y=y_original)

    original_phosphenes = _map_cortical_coordinates_to_displayed_visual_field(
        params,
        "right",
        x_original,
        y_original,
    )
    mirrored_phosphenes = mirror_map_along_y_axis(original_phosphenes)
    mirrored_cortex = _map_displayed_visual_field_to_cortex(
        params,
        mirrored_phosphenes,
        "left",
    )

    left_mapping = _prepare_cortical_grid_for_full_field(
        params,
        original_cortex,
        base_indices=base_indices,
        hemisphere="right",
        grid_id=0,
        global_index_offset=0,
        rng=rng,
    )
    right_mapping = _prepare_cortical_grid_for_full_field(
        params,
        mirrored_cortex,
        base_indices=base_indices,
        hemisphere="left",
        grid_id=1,
        global_index_offset=len(base_indices),
        rng=rng,
    )

    z_parts = []
    x_parts = []
    y_parts = []
    idx_parts = []
    grid_parts = []
    base_parts = []
    for mapping in [left_mapping, right_mapping]:
        if len(mapping.phosphene_map) == 0:
            continue
        z_parts.append(np.asarray(mapping.phosphene_map.complex))
        x_coords, y_coords = mapping.cortical_coordinates.cartesian
        x_parts.append(np.asarray(x_coords))
        y_parts.append(np.asarray(y_coords))
        idx_parts.append(np.asarray(mapping.indices, dtype=np.int64))
        grid_parts.append(np.asarray(mapping.grid_ids, dtype=np.int64))
        base_parts.append(np.asarray(mapping.base_indices, dtype=np.int64))

    if z_parts:
        phosphene_map = Map(z=np.concatenate(z_parts))
        cortical_coordinates = Map(x=np.concatenate(x_parts), y=np.concatenate(y_parts))
        indices = np.concatenate(idx_parts).astype(np.int64)
        grid_ids = np.concatenate(grid_parts).astype(np.int64)
        surviving_base_indices = np.concatenate(base_parts).astype(np.int64)
    else:
        phosphene_map = Map(z=np.asarray([], dtype=np.complex128))
        cortical_coordinates = Map(
            x=np.asarray([], dtype=np.float64),
            y=np.asarray([], dtype=np.float64),
        )
        indices = np.asarray([], dtype=np.int64)
        grid_ids = np.asarray([], dtype=np.int64)
        surviving_base_indices = np.asarray([], dtype=np.int64)

    return FullFieldMapping(
        phosphene_map=phosphene_map,
        indices=indices,
        cortical_coordinates=cortical_coordinates,
        grid_ids=grid_ids,
        base_indices=surviving_base_indices,
    )


def get_visual_field_coordinates_from_cortex(
        params: dict, coordinates_cortex: Optional[Map] = None,
        rng: Optional[np.random.Generator] = None) -> Tuple[Map, np.ndarray]:
    """
    Returns:
        Map: Location of phosphenes.
        np.ndarray: The original electrode index for each phosphene.
    """
    if coordinates_cortex is None:
        coordinates_cortex = get_cortex_coordinates_grid(params, 32, 32, x_max=40)

    x, y = coordinates_cortex.cartesian

    # 1. Initialize Indices and Sort (Up-to-Down, Left-to-Right)
    # Primary sort: y (descending for 'up'), Secondary sort: x (ascending for 'right')
    # Since visual coordinates often have index 0 at top, we sort by -y
    indices = np.arange(len(x))
    sort_idx = np.lexsort((x, -y))

    x = x[sort_idx]
    y = y[sort_idx]
    indices = indices[sort_idx]

    # 2. Add Dropout (Maintain index alignment)
    if rng is None: rng = np.random.default_rng()
    n = len(x)
    active_mask = rng.choice(np.arange(n), int(n * (1 - params['dropout_rate'])), replace=False)

    x = x[active_mask]
    y = y[active_mask]
    indices = indices[active_mask]

    # 3. Add Noise
    noise = rng.normal(scale=params['noise_scale'], size=(2, len(x)))
    x = x + noise[0]
    y = y + noise[1]

    # 4. Map to Visual Field
    cortex_to_visual_field = get_mapping_from_cortex_to_visual_field(params)
    z = cortex_to_visual_field(cartesian_to_complex(x, y))

    # 5. Remove Out of View (Maintain index alignment)
    r, phi = complex_to_polar(z)
    fov_mask = (r >= 0) & (r <= 90) & (phi > -np.pi / 2) & (phi < np.pi / 2)

    z = z[fov_mask]
    indices = indices[fov_mask]

    phosphene_map = Map(z=z).flip(hor=True, vert=True)

    return phosphene_map, indices

# def get_visual_field_coordinates_from_cortex_full(
#         params: dict, coordinates_cortex: Optional[Map] = None,
#         rng: Optional[np.random.Generator] = None) -> Map:
#     """Initialize phosphene locations in the full field of view.

#     :param params: dictionary with the several parameters in subdictionaries.
#     :param coordinates_cortex: Visuotopic map of with electrode locations on
#         cortex.
#     If None, default coordinates will be used.
#     :param rng: Numpy random number generator.
#     :return: Phosphene locations.
#     """
#     args = (params, coordinates_cortex, rng)
#     r_left, phi_left = get_visual_field_coordinates_from_cortex(*args).polar
#     r_right, phi_right = get_visual_field_coordinates_from_cortex(*args).polar
#     r = np.concatenate([r_left, r_right])
#     phi = np.concatenate([phi_left, np.pi - phi_right])
#     return Map(r=r, phi=phi)

def get_visual_field_coordinates_from_cortex_full(
        params: dict, coordinates_cortex: Optional[Map] = None,
        rng: Optional[np.random.Generator] = None) -> Tuple[Map, np.ndarray]:
    """
    Returns phosphene locations for both hemifields and the corresponding
    original electrode indices.
    """
    mapping = get_full_field_mapping_from_cortex(
        params,
        coordinates_cortex=coordinates_cortex,
        rng=rng,
    )
    return mapping.phosphene_map, mapping.indices


def expand_electrode_coordinate_arrays(
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Repeat base electrode coordinates enough times to index mirrored copies."""
    x_coords = np.asarray(x_coords, dtype=np.float64)
    y_coords = np.asarray(y_coords, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)

    if x_coords.ndim != 1 or y_coords.ndim != 1:
        raise ValueError("Electrode coordinates must be 1D arrays.")
    if len(x_coords) != len(y_coords):
        raise ValueError("x_coords and y_coords must have the same length.")
    if indices.size == 0:
        return x_coords.copy(), y_coords.copy()
    if np.any(indices < 0):
        raise ValueError("Electrode indices must be non-negative.")

    n_base = len(x_coords)
    if n_base == 0:
        raise ValueError("Electrode coordinates must not be empty.")

    n_copies = int(indices.max()) // n_base + 1
    return np.tile(x_coords, n_copies), np.tile(y_coords, n_copies)

def get_visual_field_coordinates_probabilistically(
        params: dict, n_phosphenes: int,
        rng: Optional[np.random.Generator] = None) -> Map:
    """Generate a number of phosphene locations probabilistically.

    :param params: Model parameters.
    :param n_phosphenes: Number of phosphenes.
    :param rng: Numpy random number generator.
    :return: Polar coordinates of n_phosphenes phosphenes.
    """
    if rng is None:
        rng = np.random.default_rng()

    max_r = params['run']['view_angle'] / 2
    min_r = params['run']['min_angle']

    valid_ecc = np.linspace(min_r, max_r, 1000)
    weights = get_cortical_magnification(valid_ecc, params['cortex_model'])

    probs = weights / np.sum(weights)
    r = rng.choice(valid_ecc, size=n_phosphenes, replace=True, p=probs)
    phi = 2 * np.pi * rng.random(n_phosphenes)

    return Map(r=r, phi=phi)


def get_visual_field_coordinates_grid() -> Map:
    ecc_range = np.arange(0, 90, 1)
    ang_range = np.linspace(-np.pi / 2, np.pi / 2, 10)
    r, phi = np.meshgrid(ecc_range, ang_range)
    return Map(r=r.ravel(), phi=phi.ravel())


def get_cortex_coordinates_grid(params: dict, n_electrodes_x: int,
                                n_electrodes_y: int,
                                x_max: Optional[int] = None) -> Map:
    coordinates = get_cortex_coordinates_default(params)
    x, y = coordinates.cartesian
    x_min, x_max = np.min(x), x_max or np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    xrange = np.linspace(x_min, x_max, n_electrodes_x)
    yrange = np.linspace(y_min, y_max, n_electrodes_y)
    x, y = np.meshgrid(xrange, yrange)

    return Map(x.ravel(), y.ravel())


def get_cortex_coordinates_default(params: dict) -> Map:
    """Generate cortical map.

    :return: Cortical coordinates.
    """
    coordinates_visual_field = get_visual_field_coordinates_grid()
    visual_field_to_cortex = get_mapping_from_visual_field_to_cortex(params)
    z = visual_field_to_cortex(coordinates_visual_field.complex)
    return Map(z=z)


def get_mapping_from_visual_field_to_cortex(params: dict) -> Callable:
    mapping_model = params['model']
    a = params['a']
    b = params['b']
    k = params['k']
    alpha = params['alpha']
    if mapping_model == 'monopole':
        def f(z): return k * np.log(1 + z / a)
    elif mapping_model == 'dipole':
        def f(z): return k * np.log(b * (z + a) / (a * (z + b)))
    elif mapping_model == 'wedge-dipole':
        def wedge(r, phi): return polar_to_complex(r, alpha * phi)
        def dipole(z): return k * np.log(b * (z + a) / (a * (z + b)))
        def f(z): return dipole(wedge(*complex_to_polar(z)))
    else:
        raise NotImplementedError
    return f


def get_mapping_from_cortex_to_visual_field(params: dict) -> Callable:
    mapping_model = params['model']
    a = params['a']
    b = params['b']
    k = params['k']
    alpha = params['alpha']
    if mapping_model == 'monopole':
        def f(w): return a * np.exp(w / k) - a
    elif mapping_model == 'dipole':
        def f(w):
            e = np.exp(w / k)
            return a * b * (e - 1) / (b - a * e)
    elif mapping_model == 'wedge-dipole':
        def wedge_inverse(z):
            r, phi = complex_to_polar(z)
            return polar_to_complex(r, phi / alpha)

        def dipole_inverse(w):
            e = np.exp(w / k)
            return a * b * (e - 1) / (b - a * e)

        def f(w): return wedge_inverse(dipole_inverse(w))
    else:
        raise NotImplementedError
    return f


def get_cortical_magnification(
        r: Union[np.ndarray, torch.Tensor],
        params: dict) -> Union[np.ndarray, torch.Tensor]:
    mapping_model = params['model']
    a = params['a']
    b = params['b']
    k = params['k']
    if mapping_model == 'monopole':
        return k / (r + a)
    if mapping_model in ['dipole', 'wedge-dipole']:
        return k * (1 / (r + a) - 1 / (r + b))
    raise NotImplementedError


def remove_out_of_view(z: np.ndarray) -> np.ndarray:
    r, phi = complex_to_polar(z)

    z = z[(r >= 0) & (r <= 90) & (phi > -np.pi / 2) & (phi < np.pi / 2)]

    logging.info(f"Removed {len(r) - len(z)} of {len(r)} phosphene locations.")

    return z


def add_noise(x: np.ndarray, y: np.ndarray, noise_scale: Optional[float] = 0.,
              rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray,
                                                                  np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    noise = rng.normal(scale=noise_scale, size=(2, len(x)))

    return x + noise[0], y + noise[1]


def add_dropout(x: np.ndarray, y: np.ndarray,
                dropout_rate: Optional[float] = 0.,
                rng: Optional[np.random.Generator] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    active = rng.choice(np.arange(n), int(n * (1 - dropout_rate)),
                        replace=False)

    return x[active], y[active]
