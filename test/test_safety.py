from __future__ import annotations

import numpy as np
import torch

from dynaphos.experiment.execution import (
    resolve_implant_off_tail_start_frame,
    resolve_thermal_update_interval_frames,
)
from dynaphos.safety.bioheat import Bioheat2D, Bioheat3D
from dynaphos.safety.impedance import compute_device_power, compute_frame_power
from dynaphos.safety.tracking import SafetyTracker
from dynaphos.simulation.simulator import apply_appearance_threshold


ELECTRODES = np.asarray(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=np.float64,
)


def test_appearance_threshold_gate_is_inclusive() -> None:
    stimulus = torch.tensor([29e-6, 30e-6, 31e-6])
    filtered = apply_appearance_threshold(stimulus, 30e-6)
    assert torch.equal(filtered, torch.tensor([0.0, 30e-6, 31e-6]))


def test_safety_tracker_charge_metrics_from_amplitudes() -> None:
    tracker = SafetyTracker(
        params={
            "run": {"fps": 20, "dtype": "float32", "gpu": None},
            "default_stim": {"relative_stim_duration": 0.5},
            "safety": {
                "guidelines_path": "config/safety.yaml",
                "enable_charge_guard": False,
            },
        },
        num_electrodes=2,
        data_kwargs={"device": "cpu", "dtype": torch.float32},
    )
    amplitude = torch.tensor([60e-6, 0.0])
    pulse_width = torch.tensor([170e-6, 170e-6])
    frequency = torch.tensor([300.0, 300.0])
    tracker.update(
        charge_per_s=amplitude * pulse_width * frequency,
        frequency=frequency,
        dt_s=0.05,
    )
    assert torch.isclose(
        tracker.last_charge_per_phase_nC[0],
        torch.tensor(10.2),
        atol=1e-4,
    )
    assert tracker.last_charge_per_phase_nC[1] == 0.0
    assert tracker.window_charge_per_electrode_nC[0] > 0.0
    assert torch.isfinite(tracker.last_shannon_k[0])


def test_compute_frame_power_uses_impedance_and_duty_cycle() -> None:
    instant, frame = compute_frame_power(
        amplitude=torch.tensor([2.0]),
        impedance_ohm=torch.tensor([3.0]),
        pulse_width_s=torch.tensor([0.25]),
        frequency_hz=torch.tensor([2.0]),
        relative_stim_duration=0.5,
    )
    assert torch.equal(instant, torch.tensor([12.0]))
    assert torch.equal(frame, torch.tensor([6.0]))


def test_compute_device_power_excludes_electrode_heat() -> None:
    stimulation, total = compute_device_power(
        amplitude=torch.tensor([2.0, 3.0]),
        impedance_ohm=torch.tensor([4.0, 5.0]),
        pulse_width_s=torch.tensor([0.1, 0.2]),
        frequency_hz=torch.tensor([10.0, 20.0]),
        relative_stim_duration=0.5,
        constant_power_W=0.25,
        driver_efficiency=0.8,
    )
    assert torch.equal(stimulation, torch.tensor(0.0))
    assert torch.equal(total, torch.tensor(0.25))


def test_bioheat2d_no_power_stays_at_zero() -> None:
    bio = Bioheat2D(
        params={
            "bioheat": {
                "domain_xy_mm": 6.0,
                "voxel_size_mm": 1.0,
                "internal_circuit_power_mw": 0.0,
                "device_constant_power_mw": 0.0,
                "T_b": 37.0,
            }
        },
        elec_xy_mm=ELECTRODES,
        device="cpu",
    )
    for _ in range(5):
        bio.update(torch.zeros(4), dt=0.1)
    assert torch.allclose(bio.dT, torch.zeros_like(bio.dT), atol=1e-6)
    assert torch.allclose(bio.T, torch.full_like(bio.T, 37.0), atol=1e-6)


def test_bioheat2d_accepts_electrode_and_internal_power() -> None:
    bio = Bioheat2D(
        params={
            "bioheat": {
                "domain_xy_mm": 6.0,
                "voxel_size_mm": 1.0,
                "internal_circuit_power_mw": 1.0,
                "device_constant_power_mw": 1.0,
                "T_b": 37.0,
            }
        },
        elec_xy_mm=ELECTRODES,
        device="cpu",
    )
    bio.update(torch.tensor([0.002, 0.0, 0.0, 0.0]), dt=0.1)
    assert bio.dT.max() > 0.0
    assert np.isclose(bio.last_electrode_power_W, 0.002)
    assert np.isclose(bio.last_total_power_W, 0.003)


def test_bioheat3d_no_power_stays_at_zero() -> None:
    bio = Bioheat3D(
        params={
            "bioheat": {
                "domain_xy_mm": 6.0,
                "brain_depth_mm": 3.0,
                "skull_thickness_mm": 1.0,
                "scalp_thickness_mm": 1.0,
                "voxel_size_mm": 1.0,
                "T_b": 37.0,
            }
        },
        elec_xy_mm=ELECTRODES,
        device="cpu",
    )
    for _ in range(5):
        bio.update(0.0, dt=0.1)
    assert torch.allclose(bio.dT, torch.zeros_like(bio.dT), atol=1e-6)
    assert torch.allclose(bio.T, torch.full_like(bio.T, 37.0), atol=1e-6)


def test_bioheat3d_source_is_at_brain_skull_interface() -> None:
    bio = Bioheat3D(
        params={
            "bioheat": {
                "domain_xy_mm": 6.0,
                "brain_depth_mm": 3.0,
                "skull_thickness_mm": 1.0,
                "scalp_thickness_mm": 1.0,
                "voxel_size_mm": 1.0,
            }
        },
        elec_xy_mm=ELECTRODES,
        device="cpu",
    )
    source_layers = torch.nonzero(
        bio.source_mask.any(dim=(1, 2)),
        as_tuple=False,
    ).reshape(-1)
    assert torch.equal(source_layers, torch.tensor([bio.brain_cells - 1]))


def test_fixed_numerical_fingerprint() -> None:
    stimulation = apply_appearance_threshold(
        torch.tensor([20e-6, 30e-6, 60e-6, 90e-6]),
        30e-6,
    )
    np.testing.assert_allclose(
        stimulation.numpy(),
        [0.0, 30e-6, 60e-6, 90e-6],
        rtol=0.0,
        atol=2e-12,
    )

    tracker = SafetyTracker(
        params={
            "run": {"fps": 20, "dtype": "float64", "gpu": None},
            "default_stim": {"relative_stim_duration": 0.5},
            "safety": {
                "guidelines_path": "config/safety.yaml",
                "enable_charge_guard": False,
            },
        },
        num_electrodes=4,
        data_kwargs={"device": "cpu", "dtype": torch.float64},
    )
    pulse_width = torch.full((4,), 170e-6, dtype=torch.float64)
    frequency = torch.full((4,), 300.0, dtype=torch.float64)
    tracker.update(
        charge_per_s=stimulation.double() * pulse_width * frequency,
        frequency=frequency,
        dt_s=0.05,
    )
    np.testing.assert_allclose(
        tracker.last_charge_per_phase_nC.numpy(),
        [0.0, 5.1, 10.2, 15.3],
        rtol=1e-7,
        atol=1e-7,
    )

    _, frame_power = compute_frame_power(
        stimulation.double(),
        torch.tensor([1000.0, 1200.0, 1400.0, 1600.0], dtype=torch.float64),
        pulse_width,
        frequency,
        relative_stim_duration=0.5,
    )
    np.testing.assert_allclose(
        frame_power.numpy(),
        [0.0, 5.508e-8, 2.5704e-7, 6.6096e-7],
        rtol=1e-7,
        atol=1e-14,
    )

    bio = Bioheat2D(
        params={
            "bioheat": {
                "domain_xy_mm": 6.0,
                "voxel_size_mm": 1.0,
                "internal_circuit_power_mw": 1.0,
                "device_constant_power_mw": 1.0,
                "T_b": 37.0,
            }
        },
        elec_xy_mm=ELECTRODES,
        device="cpu",
    )
    means = []
    maxima = []
    for power in (
        torch.tensor([0.002, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.001, 0.0, 0.0]),
        torch.zeros(4),
    ):
        bio.update(power, dt=0.1)
        means.append(float(bio.dT.mean()))
        maxima.append(float(bio.dT.max()))
    np.testing.assert_allclose(
        means,
        [0.0021992098, 0.0036630582, 0.0043923110],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        maxima,
        [0.061577875, 0.067000978, 0.072283022],
        rtol=1e-6,
    )


def test_thermal_update_defaults_to_one_video_frame() -> None:
    params = {"bioheat": {}}
    assert resolve_thermal_update_interval_frames(params, fps=20.0) == 1
    assert resolve_thermal_update_interval_frames(params, fps=29.97) == 1
    assert resolve_thermal_update_interval_frames(
        params,
        fps=20.0,
        override_frames=5,
    ) == 5


def test_implant_off_tail_ignores_short_debug_clips() -> None:
    assert resolve_implant_off_tail_start_frame(
        30 * 60 * 20,
        fps=20.0,
        tail_seconds=300.0,
    ) == 25 * 60 * 20
    assert (
        resolve_implant_off_tail_start_frame(
            60 * 20,
            fps=20.0,
            tail_seconds=300.0,
        )
        is None
    )
