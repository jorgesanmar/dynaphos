"""Pennes bioheat models used by dynaphos safety analyses."""

from __future__ import annotations

import numpy as np
import torch


def _convex_hull(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= 2:
        return points.copy()
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygon_mask(grid_xx: np.ndarray, grid_yy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    if len(polygon_xy) < 3:
        return np.zeros_like(grid_xx, dtype=bool)

    from matplotlib.path import Path

    points = np.column_stack([grid_xx.ravel(), grid_yy.ravel()])
    mask = Path(polygon_xy, closed=True).contains_points(points, radius=1e-12)
    return mask.reshape(grid_xx.shape)


def _get_layer_params(bh: dict, layer: str, defaults: dict[str, float]) -> dict[str, float]:
    values = {key: float(value) for key, value in defaults.items()}
    values.update({key: float(value) for key, value in (bh.get(layer, {}) or {}).items()})
    return values


class Bioheat2D:
    """
    Coarse 2D Pennes-style dT solver for electrode and implant heat.

    The model tracks temperature rise above the baseline temperature in a
    single tissue sheet:

        d(dT)/dt = alpha * Lap(dT) - beta * dT + Q / (rho * c)

    ``update`` accepts one frame-averaged dissipated load-power value per
    electrode. Those values are injected at the nearest electrode grid cells,
    while any configured constant internal-circuit power is spread over the
    electrode-grid footprint. Stimulation-dependent IC driver loss is not part
    of this heat source. No vertical conduction or convection term is included
    in this 2D model.
    """

    DEFAULT_ELECTRODE_DIAMETER_MM = 0.08
    PAPER_BRAIN = {
        "rho": 1041.0,
        "c": 3640.0,
        "k": 0.528,
        "perfusion": 0.0097,
        "q_met": 10383.0,
    }
    PAPER_BLOOD = {"rho": 1060.0, "c": 3840.0}

    def __init__(self, params: dict, elec_xy_mm: np.ndarray, device: str | torch.device = "cpu"):
        self.params = params
        self.device = torch.device(device)
        bh = params.get("bioheat", {}) or {}

        self.model = str(bh.get("model", "2d_pennes"))
        self.baseline_T = float(bh.get("T_b", bh.get("baseline_temp_C", 37.0)))
        self.boundary_T = self.baseline_T
        self.T_b = self.baseline_T

        self.requested_domain_xy_mm = float(bh.get("domain_mm", bh.get("domain_xy_mm", 100.0)))
        self.electrode_diameter_mm = float(
            bh.get("electrode_diameter_mm", self.DEFAULT_ELECTRODE_DIAMETER_MM)
        )
        self.voxel_size_mm = float(
            bh.get("voxel_size_mm", bh.get("dx_mm", self.electrode_diameter_mm))
        )
        legacy_source_thickness_mm = float(
            bh.get(
                "source_thickness_mm",
                bh.get("thermal_layer_thickness_mm", self.voxel_size_mm),
            )
        )
        self.electrode_source_thickness_mm = float(
            bh.get("electrode_source_thickness_mm", self.voxel_size_mm)
        )
        self.ic_source_thickness_mm = float(
            bh.get(
                "ic_source_thickness_mm",
                bh.get("internal_circuit_source_thickness_mm", legacy_source_thickness_mm),
            )
        )
        self.source_thickness_mm = max(
            self.electrode_source_thickness_mm,
            self.ic_source_thickness_mm,
        )
        if self.requested_domain_xy_mm <= 0.0 or self.voxel_size_mm <= 0.0:
            raise ValueError("bioheat domain and grid spacing must be positive.")
        if self.electrode_diameter_mm <= 0.0:
            raise ValueError("bioheat.electrode_diameter_mm must be positive.")
        if self.electrode_source_thickness_mm <= 0.0:
            raise ValueError("bioheat.electrode_source_thickness_mm must be positive.")
        if self.ic_source_thickness_mm <= 0.0:
            raise ValueError("bioheat.ic_source_thickness_mm must be positive.")

        self.dx = self.voxel_size_mm * 1e-3
        self.dy = self.voxel_size_mm * 1e-3
        self.dz = self.source_thickness_mm * 1e-3
        self.electrode_dz = self.electrode_source_thickness_mm * 1e-3
        self.ic_dz = self.ic_source_thickness_mm * 1e-3
        self.voxel_size_m = self.dx
        self.voxel_volume_m3 = self.dx * self.dy * self.dz
        self.electrode_voxel_volume_m3 = self.dx * self.dy * self.electrode_dz
        self.ic_voxel_volume_m3 = self.dx * self.dy * self.ic_dz
        self.voxel_volume_mm3 = self.voxel_size_mm * self.voxel_size_mm * self.source_thickness_mm
        self.electrode_voxel_volume_mm3 = (
            self.voxel_size_mm * self.voxel_size_mm * self.electrode_source_thickness_mm
        )
        self.ic_voxel_volume_mm3 = (
            self.voxel_size_mm * self.voxel_size_mm * self.ic_source_thickness_mm
        )
        self.pixel_area_mm2 = self.voxel_size_mm ** 2

        self.W = max(1, int(round(self.requested_domain_xy_mm / self.voxel_size_mm)))
        self.H = self.W
        self.D = 1
        self.domain_xy_mm = self.W * self.voxel_size_mm
        self.depth_mm = self.source_thickness_mm
        self.source_z_index = 0

        elec_xy_mm = np.asarray(elec_xy_mm, dtype=np.float64)
        if elec_xy_mm.ndim != 2 or elec_xy_mm.shape[1] != 2 or elec_xy_mm.shape[0] == 0:
            raise ValueError("elec_xy_mm must be a non-empty [n, 2] array.")
        self.elec_xy_mm = elec_xy_mm
        self.n_electrodes = int(elec_xy_mm.shape[0])

        cx = float(elec_xy_mm[:, 0].mean())
        cy = float(elec_xy_mm[:, 1].mean())
        half = 0.5 * self.domain_xy_mm
        self.xmin_mm = cx - half
        self.xmax_mm = cx + half
        self.ymin_mm = cy - half
        self.ymax_mm = cy + half

        self.brain = _get_layer_params(bh, "brain", self.PAPER_BRAIN)
        self.blood = _get_layer_params(bh, "blood", self.PAPER_BLOOD)
        rho = float(self.brain["rho"])
        c = float(self.brain["c"])
        k = float(self.brain["k"])
        perfusion = float(self.brain["perfusion"])
        self.alpha = torch.tensor(k / (rho * c), dtype=torch.float32, device=self.device)
        self.beta = torch.tensor(
            perfusion * float(self.blood["rho"]) * float(self.blood["c"]) / (rho * c),
            dtype=torch.float32,
            device=self.device,
        )
        self.inv_rhoc = torch.tensor(1.0 / (rho * c), dtype=torch.float32, device=self.device)
        # Cache the float32-rounded scalars so update() never syncs the device
        # to recompute them. Equal to float(self.alpha.item()) by construction.
        self.alpha_f = float(self.alpha.item())
        self.beta_f = float(self.beta.item())

        self.device_constant_power_W = float(
            bh.get("device_constant_power_mw", bh.get("internal_circuit_power_mw", 13.0))
        ) * 1e-3
        self.internal_circuit_power_W = float(
            bh.get("internal_circuit_power_mw", bh.get("device_constant_power_mw", 13.0))
        ) * 1e-3
        self.driver_efficiency = float(bh.get("driver_efficiency", 0.8))
        if self.driver_efficiency <= 0.0 or self.driver_efficiency > 1.0:
            raise ValueError("bioheat.driver_efficiency must be in the range (0, 1].")

        self._configure_grid_sources(elec_xy_mm)

        self.dT = torch.zeros((self.H, self.W), dtype=torch.float32, device=self.device)
        self.last_electrode_power_W = 0.0
        self.last_total_power_W = 0.0
        self.ic_power_density_W_m3 = (
            self.internal_circuit_power_W / self.source_volume_m3
            if self.internal_circuit_power_W > 0.0 and self.source_volume_m3 > 0.0
            else 0.0
        )

    def _configure_grid_sources(self, elec_xy_mm: np.ndarray) -> None:
        x_centers = self.xmin_mm + (np.arange(self.W, dtype=np.float64) + 0.5) * self.voxel_size_mm
        y_centers = self.ymin_mm + (np.arange(self.H, dtype=np.float64) + 0.5) * self.voxel_size_mm
        grid_xx, grid_yy = np.meshgrid(x_centers, y_centers, indexing="xy")

        ix = np.clip(np.searchsorted(x_centers, elec_xy_mm[:, 0]), 0, self.W - 1)
        iy = np.clip(np.searchsorted(y_centers, elec_xy_mm[:, 1]), 0, self.H - 1)
        left_ix = np.clip(ix - 1, 0, self.W - 1)
        left_iy = np.clip(iy - 1, 0, self.H - 1)
        ix = np.where(
            np.abs(x_centers[left_ix] - elec_xy_mm[:, 0]) < np.abs(x_centers[ix] - elec_xy_mm[:, 0]),
            left_ix,
            ix,
        )
        iy = np.where(
            np.abs(y_centers[left_iy] - elec_xy_mm[:, 1]) < np.abs(y_centers[iy] - elec_xy_mm[:, 1]),
            left_iy,
            iy,
        )
        self.electrode_x_idx = torch.as_tensor(ix.astype(np.int64), dtype=torch.long, device=self.device)
        self.electrode_y_idx = torch.as_tensor(iy.astype(np.int64), dtype=torch.long, device=self.device)
        self.electrode_flat_idx = self.electrode_y_idx * self.W + self.electrode_x_idx

        electrode_footprint = np.zeros((self.H, self.W), dtype=bool)
        electrode_footprint[iy, ix] = True

        hull = _convex_hull(elec_xy_mm)
        if len(hull) >= 3:
            footprint = _polygon_mask(grid_xx, grid_yy, hull)
        else:
            footprint = electrode_footprint

        if not np.any(footprint):
            footprint = electrode_footprint

        self.ic_footprint_mask = torch.as_tensor(footprint.astype(np.float32), dtype=torch.float32, device=self.device)
        self.source_mask = self.ic_footprint_mask.to(dtype=torch.bool)
        self.ic_footprint_pixel_count = float(np.count_nonzero(footprint))
        self.source_voxel_count = self.ic_footprint_pixel_count
        self.source_area_m2 = self.ic_footprint_pixel_count * self.dx * self.dy
        self.source_volume_m3 = self.source_voxel_count * self.ic_voxel_volume_m3

    def _padded_dT(self) -> torch.Tensor:
        padded = torch.zeros((self.H + 2, self.W + 2), dtype=self.dT.dtype, device=self.dT.device)
        padded[1:-1, 1:-1] = self.dT
        return padded

    @property
    def T(self) -> torch.Tensor:
        """Absolute temperature map in Celsius, retained for compatibility."""
        return self.baseline_T + self.dT

    @property
    def temperature_C(self) -> torch.Tensor:
        return self.T

    @property
    def extent_mm(self) -> tuple[float, float, float, float]:
        return (self.xmin_mm, self.xmax_mm, self.ymin_mm, self.ymax_mm)

    @property
    def z_extent_mm(self) -> tuple[float, float]:
        return (0.0, self.depth_mm)

    @property
    def volume_extent_mm(self) -> tuple[float, float, float, float, float, float]:
        return (*self.extent_mm, *self.z_extent_mm)

    @property
    def voxel_spacing_mm(self) -> tuple[float, float, float]:
        return (self.voxel_size_mm, self.voxel_size_mm, self.source_thickness_mm)

    @property
    def voxel_spacing_m(self) -> tuple[float, float, float]:
        return (self.dx, self.dy, self.dz)

    def dT_projection(self) -> torch.Tensor:
        return self.dT

    def source_plane_dT(self) -> torch.Tensor:
        return self.dT

    def update(self, electrode_power_W: float | torch.Tensor, dt: float) -> None:
        if dt <= 0.0:
            return

        power = torch.as_tensor(electrode_power_W, dtype=torch.float32, device=self.device).reshape(-1)
        if power.numel() == 1 and self.n_electrodes != 1:
            power = torch.full(
                (self.n_electrodes,),
                float(power.item()) / float(self.n_electrodes),
                dtype=torch.float32,
                device=self.device,
            )
        if power.numel() != self.n_electrodes:
            raise ValueError(
                "Bioheat2D.update expects one power value per electrode; "
                f"got {power.numel()} for {self.n_electrodes} electrodes."
            )
        power = torch.clamp(power, min=0.0)
        # Write-only diagnostics: keep on-device to avoid a per-call host sync.
        self.last_electrode_power_W = power.sum()
        self.last_total_power_W = self.last_electrode_power_W + max(self.internal_circuit_power_W, 0.0)

        # Always scatter: with all-zero power this is a no-op that yields the
        # same zero power_density, but avoids the per-frame torch.any host sync.
        source_power = torch.zeros(self.H * self.W, dtype=torch.float32, device=self.device)
        source_power.scatter_add_(0, self.electrode_flat_idx, power)
        power_density = source_power.reshape(self.H, self.W) / self.electrode_voxel_volume_m3

        if self.internal_circuit_power_W > 0.0 and self.source_volume_m3 > 0.0:
            self.ic_power_density_W_m3 = self.internal_circuit_power_W / self.source_volume_m3
            power_density = power_density + self.source_mask.to(dtype=torch.float32) * self.ic_power_density_W_m3
        else:
            self.ic_power_density_W_m3 = 0.0

        source_term = power_density * self.inv_rhoc
        alpha = self.alpha_f
        beta = self.beta_f
        dt_diffusion = (self.voxel_size_m ** 2) / (4.0 * alpha + 1e-30)
        dt_sub_max = 0.45 * dt_diffusion
        if beta > 0.0:
            dt_sub_max = min(dt_sub_max, 0.5 / beta)
        n_sub = int(np.ceil(float(dt) / dt_sub_max)) if dt > dt_sub_max else 1
        dt_sub = float(dt) / max(n_sub, 1)

        for _ in range(n_sub):
            padded = self._padded_dT()
            lap = (
                (padded[1:-1, 2:] - 2.0 * self.dT + padded[1:-1, :-2]) / (self.dx ** 2)
                + (padded[2:, 1:-1] - 2.0 * self.dT + padded[:-2, 1:-1]) / (self.dy ** 2)
            )
            self.dT = self.dT + dt_sub * (self.alpha * lap - self.beta * self.dT + source_term)


class Bioheat3D:
    """
    Coarse 3D Pennes-style dT solver for implant-level device heat.

    The model tracks temperature rise above 37 C:

        d(dT)/dt = alpha * Lap(dT) - beta * dT + Q_device / (rho * c)

    The only heat source is total device power spread uniformly through a
    1-voxel source layer at the top of the brain under skull/scalp.
    """

    PAPER_BRAIN = {"rho": 1041.0, "c": 3640.0, "k": 0.528, "perfusion": 0.0097, "q_met": 10383.0}
    PAPER_SKULL = {"rho": 1990.0, "c": 1300.0, "k": 0.650, "perfusion": 0.00099, "q_met": 26.0}
    PAPER_SCALP = {"rho": 1100.0, "c": 3150.0, "k": 0.342, "perfusion": 0.0022, "q_met": 1100.0}
    PAPER_BLOOD = {"rho": 1060.0, "c": 3840.0}

    def __init__(self, params: dict, elec_xy_mm: np.ndarray, device: str = "cpu"):
        self.params = params
        self.device = torch.device(device)
        bh = params.get("bioheat", {}) or {}

        self.model = str(bh.get("model", "3d_ic_only"))
        self.baseline_T = float(bh.get("T_b", bh.get("baseline_temp_C", 37.0)))
        self.boundary_T = self.baseline_T
        self.T_b = self.baseline_T

        self.requested_domain_xy_mm = float(bh.get("domain_xy_mm", bh.get("domain_mm", 100.0)))
        self.requested_brain_depth_mm = float(bh.get("brain_depth_mm", 50.0))
        self.requested_skull_thickness_mm = float(bh.get("skull_thickness_mm", 5.0))
        self.requested_scalp_thickness_mm = float(bh.get("scalp_thickness_mm", 3.0))
        self.voxel_size_mm = float(bh.get("voxel_size_mm", 1.0))
        if self.requested_domain_xy_mm <= 0 or self.requested_brain_depth_mm <= 0 or self.voxel_size_mm <= 0:
            raise ValueError("bioheat domain and voxel sizes must be positive.")
        if self.requested_skull_thickness_mm < 0 or self.requested_scalp_thickness_mm < 0:
            raise ValueError("bioheat skull/scalp thicknesses must be >= 0.")

        self.voxel_size_m = self.voxel_size_mm * 1e-3
        self.dx = self.voxel_size_m
        self.dy = self.voxel_size_m
        self.dz = self.voxel_size_m
        self.voxel_volume_m3 = self.dx * self.dy * self.dz
        self.voxel_volume_mm3 = self.voxel_size_mm ** 3
        self.pixel_area_mm2 = self.voxel_size_mm ** 2

        self.W = max(1, int(round(self.requested_domain_xy_mm / self.voxel_size_mm)))
        self.H = self.W
        self.brain_cells = max(1, int(round(self.requested_brain_depth_mm / self.voxel_size_mm)))
        self.skull_cells = max(0, int(round(self.requested_skull_thickness_mm / self.voxel_size_mm)))
        self.scalp_cells = max(0, int(round(self.requested_scalp_thickness_mm / self.voxel_size_mm)))
        self.D = self.brain_cells + self.skull_cells + self.scalp_cells
        self.source_z_index = self.brain_cells - 1
        self.domain_xy_mm = self.W * self.voxel_size_mm
        self.brain_depth_mm = self.brain_cells * self.voxel_size_mm
        self.skull_thickness_mm = self.skull_cells * self.voxel_size_mm
        self.scalp_thickness_mm = self.scalp_cells * self.voxel_size_mm
        self.depth_mm = self.D * self.voxel_size_mm

        elec_xy_mm = np.asarray(elec_xy_mm, dtype=np.float64)
        if elec_xy_mm.ndim != 2 or elec_xy_mm.shape[1] != 2 or elec_xy_mm.shape[0] == 0:
            raise ValueError("elec_xy_mm must be a non-empty [n, 2] array.")
        cx = float(elec_xy_mm[:, 0].mean())
        cy = float(elec_xy_mm[:, 1].mean())
        half = 0.5 * self.domain_xy_mm
        self.xmin_mm = cx - half
        self.xmax_mm = cx + half
        self.ymin_mm = cy - half
        self.ymax_mm = cy + half

        self.brain = _get_layer_params(bh, "brain", self.PAPER_BRAIN)
        self.skull = _get_layer_params(bh, "skull", self.PAPER_SKULL)
        self.scalp = _get_layer_params(bh, "scalp", self.PAPER_SCALP)
        self.blood = _get_layer_params(bh, "blood", self.PAPER_BLOOD)

        self.device_constant_power_W = float(
            bh.get("device_constant_power_mw", bh.get("internal_circuit_power_mw", 13.0))
        ) * 1e-3
        self.driver_efficiency = float(bh.get("driver_efficiency", 0.8))
        if self.driver_efficiency <= 0.0 or self.driver_efficiency > 1.0:
            raise ValueError("bioheat.driver_efficiency must be in the range (0, 1].")

        self.top_heat_transfer_W_m2_K = float(bh.get("top_heat_transfer_W_m2_K", 5.0))

        self._configure_material_tensors()
        self._configure_source_mask(elec_xy_mm)

        self.dT = torch.zeros((self.D, self.H, self.W), dtype=torch.float32, device=self.device)
        self.last_total_power_W = 0.0
        self.ic_power_density_W_m3 = 0.0

    def _configure_material_tensors(self) -> None:
        layer_values = []
        for params, count in (
            (self.brain, self.brain_cells),
            (self.skull, self.skull_cells),
            (self.scalp, self.scalp_cells),
        ):
            layer_values.extend([params] * count)

        rho = np.asarray([layer["rho"] for layer in layer_values], dtype=np.float32)
        c = np.asarray([layer["c"] for layer in layer_values], dtype=np.float32)
        k = np.asarray([layer["k"] for layer in layer_values], dtype=np.float32)
        perfusion = np.asarray([layer["perfusion"] for layer in layer_values], dtype=np.float32)

        alpha = k / (rho * c)
        beta = perfusion * float(self.blood["rho"]) * float(self.blood["c"]) / (rho * c)
        inv_rhoc = 1.0 / (rho * c)

        shape = (self.D, 1, 1)
        self.alpha = torch.as_tensor(alpha.reshape(shape), dtype=torch.float32, device=self.device)
        self.beta = torch.as_tensor(beta.reshape(shape), dtype=torch.float32, device=self.device)
        self.inv_rhoc = torch.as_tensor(inv_rhoc.reshape(shape), dtype=torch.float32, device=self.device)
        self.top_robin_coeff = 0.0
        if self.top_heat_transfer_W_m2_K > 0.0 and self.scalp_cells > 0:
            k_top = max(float(self.scalp["k"]), 1e-30)
            bi = self.top_heat_transfer_W_m2_K * self.dz / k_top
            self.top_robin_coeff = float(bi / (1.0 + 0.5 * bi))

    def _configure_source_mask(self, elec_xy_mm: np.ndarray) -> None:
        x_centers = self.xmin_mm + (np.arange(self.W, dtype=np.float64) + 0.5) * self.voxel_size_mm
        y_centers = self.ymin_mm + (np.arange(self.H, dtype=np.float64) + 0.5) * self.voxel_size_mm
        grid_xx, grid_yy = np.meshgrid(x_centers, y_centers, indexing="xy")

        hull = _convex_hull(elec_xy_mm)
        if len(hull) >= 3:
            footprint = _polygon_mask(grid_xx, grid_yy, hull)
        else:
            cx = float(elec_xy_mm[:, 0].mean())
            cy = float(elec_xy_mm[:, 1].mean())
            footprint = (
                (np.abs(grid_xx - cx) <= 0.5 * self.voxel_size_mm)
                & (np.abs(grid_yy - cy) <= 0.5 * self.voxel_size_mm)
            )

        if not np.any(footprint):
            ix = int(np.clip(np.argmin(np.abs(x_centers - float(elec_xy_mm[:, 0].mean()))), 0, self.W - 1))
            iy = int(np.clip(np.argmin(np.abs(y_centers - float(elec_xy_mm[:, 1].mean()))), 0, self.H - 1))
            footprint[iy, ix] = True

        source = np.zeros((self.D, self.H, self.W), dtype=bool)
        source[self.source_z_index, :, :] = footprint

        self.ic_footprint_mask = torch.as_tensor(footprint.astype(np.float32), dtype=torch.float32, device=self.device)
        self.source_mask = torch.as_tensor(source, dtype=torch.bool, device=self.device)
        self.ic_footprint_pixel_count = float(np.count_nonzero(footprint))
        self.source_voxel_count = float(np.count_nonzero(source))
        self.source_area_m2 = self.ic_footprint_pixel_count * self.dx * self.dy
        self.source_volume_m3 = self.source_voxel_count * self.voxel_volume_m3

    def _padded_dT(self) -> torch.Tensor:
        padded = torch.zeros((self.D + 2, self.H + 2, self.W + 2), dtype=self.dT.dtype, device=self.dT.device)
        padded[1:-1, 1:-1, 1:-1] = self.dT
        if self.top_robin_coeff > 0.0:
            top = self.dT[-1]
            padded[-1, 1:-1, 1:-1] = top - self.top_robin_coeff * top
        return padded

    @property
    def T(self) -> torch.Tensor:
        """Absolute temperature map in Celsius, retained for compatibility."""
        return self.baseline_T + self.dT

    @property
    def temperature_C(self) -> torch.Tensor:
        return self.T

    @property
    def extent_mm(self) -> tuple[float, float, float, float]:
        return (self.xmin_mm, self.xmax_mm, self.ymin_mm, self.ymax_mm)

    @property
    def z_extent_mm(self) -> tuple[float, float]:
        return (0.0, self.depth_mm)

    @property
    def volume_extent_mm(self) -> tuple[float, float, float, float, float, float]:
        return (*self.extent_mm, *self.z_extent_mm)

    @property
    def voxel_spacing_mm(self) -> tuple[float, float, float]:
        return (self.voxel_size_mm, self.voxel_size_mm, self.voxel_size_mm)

    @property
    def voxel_spacing_m(self) -> tuple[float, float, float]:
        return (self.dx, self.dy, self.dz)

    def dT_projection(self) -> torch.Tensor:
        return self.dT.max(dim=0).values

    def source_plane_dT(self) -> torch.Tensor:
        return self.dT[self.source_z_index]

    def update(self, total_power_W: float | torch.Tensor, dt: float) -> None:
        if torch.is_tensor(total_power_W):
            power_W = float(total_power_W.detach().sum().cpu().item())
        else:
            power_W = float(total_power_W)
        power_W = max(power_W, 0.0)
        self.last_total_power_W = power_W

        S = torch.zeros_like(self.dT)
        if power_W > 0.0 and self.source_volume_m3 > 0.0:
            self.ic_power_density_W_m3 = power_W / self.source_volume_m3
            S[self.source_mask] = self.ic_power_density_W_m3
            S = S * self.inv_rhoc
        else:
            self.ic_power_density_W_m3 = 0.0

        max_alpha = float(self.alpha.max().item())
        dt_diffusion = (self.voxel_size_m ** 2) / (6.0 * max_alpha + 1e-30)
        max_reaction = float(self.beta.max().item())
        if self.top_heat_transfer_W_m2_K > 0.0 and self.scalp_cells > 0:
            rho_c_top = float(self.scalp["rho"]) * float(self.scalp["c"])
            max_reaction = max(max_reaction, self.top_heat_transfer_W_m2_K / (rho_c_top * self.dz))
        dt_sub_max = 0.45 * dt_diffusion
        if max_reaction > 0.0:
            dt_sub_max = min(dt_sub_max, 0.5 / max_reaction)
        n_sub = int(np.ceil(float(dt) / dt_sub_max)) if dt > dt_sub_max else 1
        dt_sub = float(dt) / max(n_sub, 1)

        for _ in range(n_sub):
            padded = self._padded_dT()
            lap = (
                (padded[1:-1, 1:-1, 2:] - 2.0 * self.dT + padded[1:-1, 1:-1, :-2]) / (self.dx ** 2)
                + (padded[1:-1, 2:, 1:-1] - 2.0 * self.dT + padded[1:-1, :-2, 1:-1]) / (self.dy ** 2)
                + (padded[2:, 1:-1, 1:-1] - 2.0 * self.dT + padded[:-2, 1:-1, 1:-1]) / (self.dz ** 2)
            )
            self.dT = self.dT + dt_sub * (self.alpha * lap - self.beta * self.dT + S)
