from __future__ import annotations

from itertools import product
from pathlib import Path
from argparse import Namespace
import sys

import numpy as np
import torch
import yaml

from dynaphos.safety.impedance import compute_device_power, compute_frame_power
from dynaphos.safety.bioheat import Bioheat2D, Bioheat3D
from dynaphos.safety.io import (
    SimulationCase,
    normalize_raster_mode,
    run_cases,
    should_write_preview,
    write_manifest,
)
from dynaphos.safety.experiments import build_cases
from dynaphos.safety.ic_power_visualize import (
    discover_ic_power_records,
    write_ic_power_temperature_visuals,
)
from dynaphos.safety.runner import (
    apply_adaptive_activation_threshold,
    activation_threshold_from_current_threshold,
    aggregate_metric_tensor,
    build_bioheat_grid_sets,
    build_electrode_base_indices,
    build_electrode_grid_ids,
    build_full_electrode_reference,
    build_reference_index,
    build_surviving_electrode_metadata,
    build_surviving_electrode_set,
    expand_metric_to_reference_tensor,
    resolve_cooldown_frame_count,
    resolve_thermal_update_interval_frames,
)
from dynaphos.safety.tracking import SafetyTracker
from dynaphos.safety import visualize
from dynaphos.simulator import GaussianSimulator, apply_appearance_threshold
from dynaphos.pipeline import render_phosphene_frame_from_state
from dynaphos.utils import Map
from tools.safety import run_phase1
from tools.safety import run_phase2_ic_power
from tools.safety import run_phase3_rastering


def write_fake_matrix_record(root: Path, *, amplitude: float, grid_um: int, preprocessing: str) -> None:
    run_id = f"coords_{grid_um}um__{preprocessing}__amp_{amplitude:g}uA"
    run_dir = root / "amplitude_grid_preprocessing" / run_id
    run_dir.mkdir(parents=True)
    n_electrodes = {400: 4, 800: 3, 1200: 2}[grid_um]
    active = np.full((3, n_electrodes), amplitude, dtype=np.float32)
    if preprocessing == "dog":
        active[1] *= np.linspace(0.5, 1.0, n_electrodes, dtype=np.float32)
    shannon = np.where(active > 0, 0.1 + active / 100.0, -np.inf).astype(np.float32)
    charge_rate = active * 10.0
    protocol_charge = np.sum(charge_rate, axis=0).astype(np.float32)
    xy = np.column_stack([np.arange(n_electrodes), np.zeros(n_electrodes)]).astype(np.float32)
    heatmaps = np.asarray(
        [
            np.full((2, 2), 0.01 * amplitude, dtype=np.float32),
            np.full((2, 2), 0.02 * amplitude, dtype=np.float32),
            np.full((2, 2), 0.03 * amplitude, dtype=np.float32),
        ]
    )
    np.savez(
        run_dir / "safety_metrics.npz",
        time_s=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        electrode_grid_ids=np.arange(n_electrodes, dtype=np.int32),
        electrode_xy_mm=xy,
        amplitude_per_electrode_uA=active,
        shannon_k_per_electrode=shannon,
        charge_per_second_per_electrode_nC_s=charge_rate,
        window_charge_per_electrode_nC=charge_rate,
        window_charge_total_nC=np.sum(charge_rate, axis=1),
        protocol_charge_per_electrode_nC=protocol_charge,
        pulse_width_s=np.full(n_electrodes, 170e-6, dtype=np.float32),
        pulse_frequency_hz=np.full(n_electrodes, 300.0, dtype=np.float32),
        relative_stim_duration=np.asarray(1.0, dtype=np.float32),
        electrode_surface_area_cm2=np.asarray(3.5e-5, dtype=np.float32),
        mean_dT=np.asarray([0.01, 0.02, 0.03], dtype=np.float32) * amplitude,
        max_dT=np.asarray([0.02, 0.04, 0.06], dtype=np.float32) * amplitude,
        heatmap_times_s=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        heatmap_frame_indices=np.asarray([0, 1, 2], dtype=np.int32),
        heatmap_grid_names=np.asarray(["left"]),
        dT_heatmaps_left=heatmaps,
        dT_final=heatmaps[-1],
        extent_mm_left=np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
        final_peak_temperature_C_left=np.asarray(37.0 + 0.03 * amplitude, dtype=np.float32),
        final_peak_dT_C_left=np.asarray(0.03 * amplitude, dtype=np.float32),
    )
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "block": "amplitude_grid_preprocessing",
                "run_id": run_id,
                "amplitude_uA": amplitude,
                "appearance_threshold_uA": amplitude / 2.0,
                "coords_yaml": f"config/coords_{grid_um}um.yaml",
                "preprocessing_method": "groundtruth" if preprocessing == "gt" else preprocessing,
                "source_input_label": preprocessing,
                "internal_circuit_power_mw": 15.0,
            }
        ),
        encoding="utf-8",
    )


def write_fake_matrix_records(root: Path) -> None:
    for amplitude, grid_um, preprocessing in product(
        (10.0, 60.0, 120.0),
        (400, 800, 1200),
        ("dog", "canny", "gt"),
    ):
        write_fake_matrix_record(root, amplitude=amplitude, grid_um=grid_um, preprocessing=preprocessing)


def write_fake_ic_power_record(root: Path, *, preprocessing: str, power_mw: float) -> None:
    run_id = f"coords_800um__{preprocessing}__amp_60uA_{power_mw:g}mW"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    np.savez(
        run_dir / "safety_metrics.npz",
        time_s=np.asarray([0.0, 60.0, 120.0], dtype=np.float32),
        electrode_grid_ids=np.arange(3, dtype=np.int32),
        max_dT=np.asarray([0.01, 0.02, 0.03], dtype=np.float32) * (power_mw + 1.0),
        mean_dT=np.asarray([0.005, 0.01, 0.015], dtype=np.float32) * (power_mw + 1.0),
    )
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "block": "ic_power",
                "run_id": run_id,
                "amplitude_uA": 60.0,
                "appearance_threshold_uA": 30.0,
                "coords_yaml": "config/coords_800um.yaml",
                "preprocessing_method": "groundtruth" if preprocessing == "gt" else preprocessing,
                "source_input_label": preprocessing,
                "internal_circuit_power_mw": power_mw,
                "frequency_hz": 300.0,
                "pulse_width_us": 170.0,
                "raster_mode": "none",
                "ic_heat_mode": "with",
            }
        ),
        encoding="utf-8",
    )


def test_visualizer_temperature_grid_values_use_max_temperature_series(tmp_path: Path) -> None:
    write_fake_matrix_record(tmp_path, amplitude=60.0, grid_um=800, preprocessing="gt")
    npz_path = next(tmp_path.rglob("safety_metrics.npz"))

    with np.load(npz_path, allow_pickle=True) as data:
        assert np.isclose(visualize.scalar_max_dT(data), 0.06 * 60.0)


def test_safety_experiment_matrix_expands_expected_cases() -> None:
    cases = build_cases()

    assert len(cases) == 27
    assert {case.block for case in cases} == {"amplitude_grid_preprocessing"}

    amplitude_pairs = sorted({
        (case.amplitude_uA, case.appearance_threshold_uA)
        for case in cases
    })
    assert amplitude_pairs == [(10.0, 5.0), (60.0, 30.0), (120.0, 60.0)]

    density_grids = {
        Path(case.coords_yaml).name
        for case in cases
    }
    assert density_grids == {"coords_400um.yaml", "coords_800um.yaml", "coords_1200um.yaml"}

    preprocessing_inputs = {
        case.preprocessing_method: case.video
        for case in cases
    }
    assert set(preprocessing_inputs) == {"groundtruth", "canny", "dog"}
    assert preprocessing_inputs["groundtruth"] == "videos/SANPO/SANPO25min.mp4"
    assert preprocessing_inputs["canny"] == "videos/SANPO/SANPOcanny25min.mp4"
    assert preprocessing_inputs["dog"] == "videos/SANPO/SANPOdog25min.mp4"

    expected_combinations = set(
        product(
            (10.0, 60.0, 120.0),
            ("coords_400um.yaml", "coords_800um.yaml", "coords_1200um.yaml"),
            ("dog", "canny", "groundtruth"),
        )
    )
    actual_combinations = {
        (case.amplitude_uA, Path(case.coords_yaml).name, case.preprocessing_method)
        for case in cases
    }
    assert actual_combinations == expected_combinations

    assert {case.internal_circuit_power_mw for case in cases} == {0.0}
    assert {case.ic_heat_mode for case in cases} == {"with"}
    assert {case.raster_mode for case in cases} == {"none"}


def test_ic_power_phase_config_expands_expected_cases() -> None:
    cases = build_cases(matrix_path="config/safety_experiments_phase2_ic_power.yaml")

    assert len(cases) == 12
    assert {case.block for case in cases} == {"ic_power"}
    assert {case.amplitude_uA for case in cases} == {60.0}
    assert {case.appearance_threshold_uA for case in cases} == {30.0}
    assert {Path(case.coords_yaml).name for case in cases} == {"coords_800um.yaml"}
    assert {case.preprocessing_method for case in cases} == {"dog", "canny", "groundtruth"}
    assert {case.internal_circuit_power_mw for case in cases} == {0.0, 10.0, 20.0, 50.0}
    assert {case.ic_heat_mode for case in cases} == {"with"}
    assert {case.raster_mode for case in cases} == {"none"}


def test_rastering_phase_config_expands_expected_cases() -> None:
    cases = build_cases(matrix_path="config/safety_experiments_phase3_rastering.yaml")

    assert len(cases) == 6
    assert {case.block for case in cases} == {"raster_protocols"}
    assert {case.video for case in cases} == {"videos/SANPO/SANPOdog25min.mp4"}
    assert {Path(case.coords_yaml).name for case in cases} == {"coords_800um.yaml"}
    assert {case.preprocessing_method for case in cases} == {"dog"}
    assert {case.amplitude_uA for case in cases} == {60.0}
    assert {case.appearance_threshold_uA for case in cases} == {30.0}
    assert {case.internal_circuit_power_mw for case in cases} == {0.0}
    assert {case.ic_heat_mode for case in cases} == {"with"}
    assert {case.raster_mode for case in cases} == {"checkerboard", "pseudo_random"}
    assert {case.raster_groups for case in cases} == {3, 4, 5}

    expected = {
        ("checkerboard", 3, 5.0),
        ("checkerboard", 4, 3.75),
        ("checkerboard", 5, 3.0),
        ("pseudo_random", 3, 5.0),
        ("pseudo_random", 4, 3.75),
        ("pseudo_random", 5, 3.0),
    }
    actual = {
        (
            case.raster_mode,
            case.raster_groups,
            case.metadata["expected_raster_cycle_rate_hz"],
        )
        for case in cases
    }
    assert actual == expected
    assert len({case.run_id for case in cases}) == 6


def test_matrix_representative_preview_policy_selects_eleven_cases() -> None:
    cases = build_cases()
    selected = [
        case
        for case in cases
        if should_write_preview(case, "matrix-representative", preview_seconds=60.0)
    ]

    assert len(selected) == 11
    assert {
        (case.amplitude_uA, Path(case.coords_yaml).name, case.preprocessing_method)
        for case in selected
    } == {
        *{
            (60.0, grid_name, preprocessing)
            for grid_name in ("coords_400um.yaml", "coords_800um.yaml", "coords_1200um.yaml")
            for preprocessing in ("dog", "canny", "groundtruth")
        },
        (10.0, "coords_1200um.yaml", "groundtruth"),
        (120.0, "coords_1200um.yaml", "groundtruth"),
    }


def test_matrix_factor_extraction_and_pairwise_grouping(tmp_path: Path) -> None:
    write_fake_matrix_records(tmp_path)

    records = visualize.discover_records(tmp_path, Path("config/safety.yaml"))
    selected = [record for record in records if record.block == "amplitude_grid_preprocessing"]
    factors_by_run_id, records_by_key = visualize.matrix_factor_maps(selected)

    assert len(selected) == 27
    assert [value.key for value in visualize.sorted_factor_values(factors_by_run_id, "amplitude")] == [
        "amp_10uA",
        "amp_60uA",
        "amp_120uA",
    ]
    assert [value.key for value in visualize.sorted_factor_values(factors_by_run_id, "grid")] == [
        "400um",
        "800um",
        "1200um",
    ]
    assert [value.key for value in visualize.sorted_factor_values(factors_by_run_id, "preprocessing")] == [
        "dog",
        "canny",
        "gt",
    ]

    for x_factor, y_factor, fixed_factor, _pair_name in visualize.PAIRWISE_SPECS:
        x_values, y_values, fixed_values = visualize.pairwise_sheet_values(
            factors_by_run_id,
            x_factor,
            y_factor,
            fixed_factor,
        )
        assert len(fixed_values) == 3
        for fixed_value in fixed_values:
            present = 0
            for x_value in x_values:
                for y_value in y_values:
                    values = {x_factor: x_value, y_factor: y_value, fixed_factor: fixed_value}
                    if visualize.matrix_record_for_values(records_by_key, values) is not None:
                        present += 1
            assert present == 9


def test_matrix_comparative_visuals_are_summary_only(tmp_path: Path) -> None:
    write_fake_matrix_records(tmp_path)

    records = visualize.discover_records(tmp_path, Path("config/safety.yaml"))
    out_root = tmp_path / "visuals"
    visualize.plot_comparative_suites(records, out_root, "png", overwrite=True)

    matrix_root = out_root / "amplitude_grid_preprocessing"
    assert (matrix_root / "active_electrodes" / "active_electrodes_by_grid_preprocessing.png").exists()
    assert not (matrix_root / "active_electrodes" / "mean_active_electrodes_by_grid.png").exists()
    assert not (matrix_root / "active_electrodes" / "max_active_electrodes_by_grid.png").exists()
    assert (matrix_root / "shannon_k" / "mean_shannon_k_by_amplitude.png").exists()
    assert (matrix_root / "shannon_k" / "max_shannon_k_by_amplitude.png").exists()
    assert (matrix_root / "summary_grids" / "charge_per_second" / "max_amplitude_vs_electrode_grid.png").exists()
    assert not (matrix_root / "summary_grids" / "charge_per_second" / "max_electrode_grid_vs_preprocessing.png").exists()
    assert not (matrix_root / "summary_grids" / "summed_charge_per_second" / "mean_amplitude_vs_preprocessing.png").exists()
    assert (matrix_root / "summary_grids" / "total_charge" / "summed_amplitude_vs_electrode_grid.png").exists()
    assert (matrix_root / "summary_grids" / "max_mean_temperature" / "max_amplitude_vs_electrode_grid.png").exists()
    assert (matrix_root / "summary_grids" / "max_focal_temperature" / "max_amplitude_vs_electrode_grid.png").exists()
    assert not (matrix_root / "summary_grids" / "stationary_temperature").exists()
    assert (matrix_root / "total_charge" / "electrode_distribution_boxplots.png").exists()
    assert not (matrix_root / "pairwise" / "mean_shannon_k_over_time").exists()


def test_appearance_threshold_gate_is_inclusive() -> None:
    stim_raw = torch.tensor([4.999e-6, 5.0e-6, 6.0e-6], dtype=torch.float32)
    gated = apply_appearance_threshold(stim_raw, 5.0e-6)

    assert torch.equal(gated, torch.tensor([0.0, 5.0e-6, 6.0e-6], dtype=torch.float32))


def test_amplitude_mean_series_uses_zero_for_no_active_electrodes() -> None:
    data = {
        "time_s": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        "amplitude_per_electrode_uA": np.asarray(
            [
                [0.0, 0.0],
                [60.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    series = visualize.mean_metric_series(data, "amplitude", max_points=10)

    assert series is not None
    np.testing.assert_allclose(series[0], np.asarray([1.0, 2.0, 3.0], dtype=np.float32) / 60.0)
    np.testing.assert_allclose(series[1], np.asarray([0.0, 60.0, 0.0], dtype=np.float32))


def test_amplitude_downsampling_preserves_instantaneous_currents() -> None:
    data = {
        "time_s": np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        "amplitude_per_electrode_uA": np.asarray(
            [
                [0.0, 0.0],
                [60.0, 0.0],
                [0.0, 0.0],
                [60.0, 0.0],
            ],
            dtype=np.float32,
        ),
    }

    series = visualize.mean_metric_series(data, "amplitude", max_points=2)

    assert series is not None
    np.testing.assert_allclose(series[1], np.asarray([0.0, 60.0], dtype=np.float32))


def test_explicit_appearance_threshold_adapts_visual_activation_threshold() -> None:
    params = yaml.safe_load(Path("config/params.yaml").read_text(encoding="utf-8"))
    params["run"]["gpu"] = None
    params["run"]["resolution"] = [256, 256]
    params["run"]["fps"] = 20
    params["run"]["batch_size"] = 0
    params["sampling"]["sampling_method"] = "center"
    params["sampling"]["stimulus_scale"] = 10.0e-6
    params["temporal_dynamics"]["trace_increase_rate"] = 0.0
    params["thresholding"]["rheobase"] = 0.0
    params["thresholding"]["activation_threshold_sd"] = 0.0
    expected_threshold = activation_threshold_from_current_threshold(params, 5.0e-6)
    actual_threshold = apply_adaptive_activation_threshold(params, 5.0e-6)

    assert actual_threshold == expected_threshold

    sim = GaussianSimulator(
        params,
        Map(x=np.asarray([0.0]), y=np.asarray([0.0])),
        phosphene_mode="visual",
    )
    frame = np.full((256, 256), 255, dtype=np.uint8)
    for _ in range(20):
        stim_raw = sim.sample_stimulus(frame, rescale=True).reshape(-1)
        sim.update(apply_appearance_threshold(stim_raw, 5.0e-6), dt=1.0 / 20.0)

    assert torch.greater(sim.activation.get(), sim.threshold.get()).any()
    assert render_phosphene_frame_from_state(sim).max() > 0


def test_raster_aliases_normalize_to_random() -> None:
    assert normalize_raster_mode("pseudo_random") == "random"
    assert normalize_raster_mode("pseudorandom") == "random"
    assert normalize_raster_mode("random") == "random"


def test_bioheat_power_vectors_follow_aligned_electrode_coordinates() -> None:
    device = torch.device("cpu")
    # Deliberately shuffled phosphene/simulator order. The expected electrode
    # order is recovered from the global mapping indices, not from array order.
    remaining_indices = np.asarray([5, 2, 102, 105, 100, 0], dtype=np.int64)
    base_indices = np.asarray([5, 2, 2, 5, 0, 0], dtype=np.int64)
    grid_ids = np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int64)
    cortical = Map(
        x=np.asarray([50.0, 20.0, -20.0, -50.0, 0.0, 0.0]),
        y=np.asarray([5.0, 2.0, 2.0, 5.0, 0.0, 0.0]),
    )

    electrode_ids, electrode_xy = build_surviving_electrode_metadata(cortical, remaining_indices)
    electrode_base_indices = build_electrode_base_indices(remaining_indices, base_indices)
    electrode_grid_ids = build_electrode_grid_ids(remaining_indices, grid_ids)
    _elec_xy_mm_surv, inv_map_t = build_surviving_electrode_set(cortical, remaining_indices, device)

    assert electrode_ids.tolist() == [0, 2, 5, 100, 102, 105]
    assert electrode_base_indices.tolist() == [0, 2, 5, 0, 2, 5]
    assert electrode_grid_ids.tolist() == [0, 0, 0, 1, 1, 1]
    np.testing.assert_allclose(
        electrode_xy,
        np.asarray([[0.0, 0.0], [20.0, 2.0], [50.0, 5.0], [0.0, 0.0], [-20.0, 2.0], [-50.0, 5.0]]),
    )

    phos_power = torch.tensor([50.0, 20.0, 1020.0, 1050.0, 1000.0, 0.0], dtype=torch.float32)
    electrode_power = aggregate_metric_tensor(phos_power, inv_map_t, len(electrode_ids))
    np.testing.assert_allclose(
        electrode_power.numpy(),
        np.asarray([0.0, 20.0, 50.0, 1000.0, 1020.0, 1050.0], dtype=np.float32),
    )

    grid_sets = build_bioheat_grid_sets(cortical, grid_ids, base_indices, device)
    for grid_id, expected_xy, expected_power in [
        (0, [[0.0, 0.0], [20.0, 2.0], [50.0, 5.0]], [0.0, 20.0, 50.0]),
        (1, [[0.0, 0.0], [-20.0, 2.0], [-50.0, 5.0]], [1000.0, 1020.0, 1050.0]),
    ]:
        mask = torch.tensor(electrode_grid_ids == grid_id, dtype=torch.bool)
        np.testing.assert_allclose(grid_sets[grid_id]["elec_xy_mm"], np.asarray(expected_xy))
        np.testing.assert_allclose(electrode_power[mask].numpy(), np.asarray(expected_power, dtype=np.float32))


def test_full_electrode_reference_keeps_zero_slots_for_missing_electrodes(tmp_path: Path) -> None:
    coords_yaml = tmp_path / "coords.yaml"
    coords_yaml.write_text(
        yaml.safe_dump({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]}),
        encoding="utf-8",
    )
    mapping = {
        "indices": np.asarray([2, 4], dtype=np.int64),
        "grid_ids": np.asarray([0, 1], dtype=np.int64),
    }

    electrode_ids, electrode_xy, grid_ids, base_indices = build_full_electrode_reference(
        coords_yaml,
        mapping,
    )
    compact_to_full = build_reference_index(
        np.asarray([2, 4], dtype=np.int64),
        electrode_ids,
        torch.device("cpu"),
    )
    expanded = expand_metric_to_reference_tensor(
        torch.tensor([20.0, 40.0], dtype=torch.float32),
        compact_to_full,
        len(electrode_ids),
    )

    assert electrode_ids.tolist() == [0, 1, 2, 3, 4, 5]
    assert grid_ids.tolist() == [0, 0, 0, 1, 1, 1]
    assert base_indices.tolist() == [0, 1, 2, 0, 1, 2]
    np.testing.assert_allclose(
        electrode_xy,
        np.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [-1.0, 10.0],
                [-2.0, 20.0],
                [-3.0, 30.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        expanded.numpy(),
        np.asarray([0.0, 0.0, 20.0, 0.0, 40.0, 0.0], dtype=np.float32),
    )


def test_manifest_records_threshold_raster_and_source_metadata(tmp_path: Path) -> None:
    case = SimulationCase(
        block="rastering",
        run_id="coords_800um__raster_pseudo_random__standard",
        video="videos/SANPO/SANPO25min.mp4",
        coords_yaml="config/coords_800um.yaml",
        preprocessing_method="groundtruth",
        raster_mode="pseudo_random",
        appearance_threshold_uA=30.0,
        source_input_label="SANPO_groundtruth",
        metadata={"experiment_block": "rastering"},
    )

    write_manifest(
        tmp_path,
        case=case,
        params_path=Path("config/params.yaml"),
        safety_yaml=Path("config/safety.yaml"),
        output_root=tmp_path,
        max_frames=2,
        groups=4,
        phosphene_mode="safety_centers",
        cooldown_seconds=300.0,
        cooldown_baseline_tolerance_C=1e-3,
        enable_cem43=False,
    )

    manifest = yaml.safe_load((tmp_path / "run_manifest.yaml").read_text(encoding="utf-8"))
    assert manifest["appearance_threshold_uA"] == 30.0
    assert manifest["raster_mode"] == "pseudo_random"
    assert manifest["raster_mode_normalized"] == "random"
    assert manifest["source_input_label"] == "SANPO_groundtruth"
    assert manifest["metadata"]["experiment_block"] == "rastering"
    assert manifest["cooldown_seconds"] == 300.0
    assert manifest["cooldown_baseline_tolerance_C"] == 1e-3


def test_safety_tracker_charge_metrics_from_amplitudes() -> None:
    params = {
        "run": {"fps": 20, "dtype": "float32", "gpu": None},
        "default_stim": {"relative_stim_duration": 0.5},
        "safety": {"guidelines_path": "config/safety.yaml", "enable_charge_guard": False},
    }
    tracker = SafetyTracker(
        params=params,
        num_electrodes=2,
        data_kwargs={"device": "cpu", "dtype": torch.float32},
    )

    amplitude = torch.tensor([60e-6, 0.0], dtype=torch.float32)
    pulse_width = torch.tensor([170e-6, 170e-6], dtype=torch.float32)
    frequency = torch.tensor([300.0, 300.0], dtype=torch.float32)
    tracker.update(charge_per_s=amplitude * pulse_width * frequency, frequency=frequency, dt_s=0.05)

    assert torch.isclose(tracker.last_charge_per_phase_nC[0], torch.tensor(10.2), atol=1e-4)
    assert tracker.last_charge_per_phase_nC[1] == 0.0
    assert tracker.window_charge_per_electrode_nC[0] > 0.0
    assert tracker.protocol_charge_per_electrode_nC[0] == tracker.window_charge_per_electrode_nC[0]
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


def test_compute_device_power_excludes_electrode_joule_heat_from_ic_heat() -> None:
    stim_ic_power, total_power = compute_device_power(
        amplitude=torch.tensor([2.0, 3.0]),
        impedance_ohm=torch.tensor([4.0, 5.0]),
        pulse_width_s=torch.tensor([0.1, 0.2]),
        frequency_hz=torch.tensor([10.0, 20.0]),
        relative_stim_duration=0.5,
        constant_power_W=0.25,
        driver_efficiency=0.8,
    )

    assert torch.equal(stim_ic_power, torch.tensor(0.0))
    assert torch.equal(total_power, torch.tensor(0.25))

    ideal_stim_ic_power, ideal_total_power = compute_device_power(
        amplitude=torch.tensor([2.0]),
        impedance_ohm=torch.tensor([4.0]),
        pulse_width_s=torch.tensor([0.1]),
        frequency_hz=torch.tensor([10.0]),
        relative_stim_duration=0.5,
        constant_power_W=0.25,
        driver_efficiency=1.0,
    )
    assert torch.equal(ideal_stim_ic_power, torch.tensor(0.0))
    assert torch.equal(ideal_total_power, torch.tensor(0.25))


def test_bioheat2d_no_power_stays_at_zero_temperature_rise() -> None:
    elec_xy_mm = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
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
        elec_xy_mm=elec_xy_mm,
        device="cpu",
    )

    for _ in range(5):
        bio.update(torch.zeros(4, dtype=torch.float32), dt=0.1)

    assert bio.dT.shape == (bio.H, bio.W)
    assert torch.allclose(bio.dT, torch.zeros_like(bio.dT), atol=1e-6)
    assert torch.allclose(bio.T, torch.full_like(bio.T, 37.0), atol=1e-6)


def test_bioheat2d_accepts_per_electrode_power_and_ic_power() -> None:
    elec_xy_mm = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
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
        elec_xy_mm=elec_xy_mm,
        device="cpu",
    )

    bio.update(torch.tensor([0.002, 0.0, 0.0, 0.0], dtype=torch.float32), dt=0.1)

    assert bio.dT.max() > 0.0
    assert bio.source_mask.ndim == 2
    assert bio.ic_power_density_W_m3 > 0.0
    assert np.isclose(bio.last_electrode_power_W, 0.002, rtol=1e-6)
    assert np.isclose(bio.last_total_power_W, 0.003, rtol=1e-6)


def test_bioheat2d_internal_circuit_uses_convex_hull_footprint() -> None:
    elec_xy_mm = np.asarray(
        [
            [-2.0, -2.0],
            [2.0, -2.0],
            [-2.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    bio = Bioheat2D(
        params={
            "bioheat": {
                "domain_xy_mm": 8.0,
                "voxel_size_mm": 0.08,
                "internal_circuit_power_mw": 0.0,
                "device_constant_power_mw": 0.0,
            }
        },
        elec_xy_mm=elec_xy_mm,
        device="cpu",
    )

    assert bio.voxel_size_mm == 0.08
    electrode_cell_count = int(np.unique(
        np.column_stack(
            [
                bio.electrode_y_idx.detach().cpu().numpy(),
                bio.electrode_x_idx.detach().cpu().numpy(),
            ]
        ),
        axis=0,
    ).shape[0])
    assert int(bio.ic_footprint_mask.sum().item()) > electrode_cell_count


def test_bioheat3d_no_power_stays_at_zero_temperature_rise() -> None:
    elec_xy_mm = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
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
        elec_xy_mm=elec_xy_mm,
        device="cpu",
    )

    for _ in range(5):
        bio.update(0.0, dt=0.1)

    assert torch.allclose(bio.dT, torch.zeros_like(bio.dT), atol=1e-6)
    assert torch.allclose(bio.T, torch.full_like(bio.T, 37.0), atol=1e-6)


def test_bioheat3d_source_is_only_at_brain_skull_interface() -> None:
    elec_xy_mm = np.asarray(
        [
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
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
        elec_xy_mm=elec_xy_mm,
        device="cpu",
    )

    source_layers = torch.nonzero(bio.source_mask.any(dim=(1, 2)), as_tuple=False).reshape(-1)

    assert torch.equal(source_layers, torch.tensor([bio.brain_cells - 1]))
    assert int(bio.source_mask.sum().item()) == int(bio.ic_footprint_mask.sum().item())


def test_bioheat3d_volume_metadata_reports_cubic_voxels_from_params() -> None:
    elec_xy_mm = np.asarray(
        [
            [-0.5, -0.5],
            [0.5, -0.5],
            [-0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    bio = Bioheat3D(
        params={
            "bioheat": {
                "domain_xy_mm": 4.0,
                "brain_depth_mm": 2.0,
                "skull_thickness_mm": 0.5,
                "scalp_thickness_mm": 0.5,
                "voxel_size_mm": 0.75,
            }
        },
        elec_xy_mm=elec_xy_mm,
        device="cpu",
    )

    xmin, xmax, ymin, ymax, zmin, zmax = bio.volume_extent_mm
    np.testing.assert_allclose(bio.voxel_spacing_mm, (0.75, 0.75, 0.75))
    np.testing.assert_allclose(bio.voxel_spacing_m, (0.00075, 0.00075, 0.00075))
    np.testing.assert_allclose(
        (
            (xmax - xmin) / bio.W,
            (ymax - ymin) / bio.H,
            (zmax - zmin) / bio.D,
        ),
        bio.voxel_spacing_mm,
    )
    np.testing.assert_allclose(bio.voxel_volume_mm3, 0.75 ** 3)
    assert bio.dT.shape == (bio.D, bio.H, bio.W)


def test_safety_runner_dry_run_lists_cases(capsys) -> None:
    args = Namespace(
        params="config/params.yaml",
        safety_yaml="config/safety.yaml",
        output_root="results/safety/test_dry_run",
        visuals_root="results/safety/test_dry_run_visuals",
        max_frames=0,
        groups=4,
        phosphene_mode="safety_centers",
        save_every_n_frames=None,
        sim_resolution=None,
        thermal_update_interval_frames=None,
        cooldown_seconds=0.0,
        cooldown_baseline_tolerance_C=1e-3,
        preview_seconds=0.0,
        preview_policy="none",
        workers=1,
        force_cpu=True,
        enable_cem43=False,
        dry_run=True,
    )

    run_cases(build_cases(blocks=("amplitude_grid_preprocessing",)), args)

    captured = capsys.readouterr()
    assert "Planned simulations: 27" in captured.out
    assert "Planned phosphene previews: 0" in captured.out
    assert "coords_1200um__gt__amp_120uA" in captured.out


def test_experiment_matrix_defaults_metrics_only(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_phase1.py", "--dry-run"])

    args = run_phase1.parse_args()

    assert args.preview_seconds == 0.0
    assert args.preview_policy == "none"
    assert args.phosphene_mode == "safety_centers"
    assert args.cooldown_seconds == 300.0
    assert args.cooldown_baseline_tolerance_C == 1e-3


def test_phase3_rastering_defaults_metrics_only(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_phase3_rastering.py", "--dry-run"])

    args = run_phase3_rastering.parse_args()

    assert args.matrix_config.endswith("safety_experiments_phase3_rastering.yaml")
    assert args.blocks == ["raster_protocols"]
    assert args.preview_seconds == 0.0
    assert args.preview_policy == "none"
    assert args.phosphene_mode == "safety_centers"
    assert args.cooldown_seconds == 300.0
    assert args.cooldown_baseline_tolerance_C == 1e-3


def test_experiment_matrix_runs_visualizer_after_simulations(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    args = Namespace(
        blocks=None,
        matrix_config="config/safety_experiments_phase1.yaml",
        output_root=str(tmp_path / "runs"),
        visuals_root=str(tmp_path / "visuals"),
        safety_yaml="config/safety.yaml",
        dry_run=False,
    )

    monkeypatch.setattr(run_phase1, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_phase1,
        "build_cases",
        lambda *, blocks, matrix_path: calls.append("build_cases") or ["case"],
    )
    monkeypatch.setattr(
        run_phase1,
        "run_cases",
        lambda cases, parsed_args: calls.append("simulations"),
    )
    monkeypatch.setattr(
        run_phase1.visualize,
        "discover_records",
        lambda input_root, safety_yaml: calls.append("discover") or [object()],
    )

    for name, function_name in (
        ("summary_csv", "write_summary_csv"),
        ("ratio_breakdown", "plot_ratio_breakdown"),
        ("block_summaries", "plot_all_block_summaries"),
        ("comparative_suites", "plot_comparative_suites"),
        ("single_case_overviews", "write_single_case_overviews"),
        ("per_protocol_visuals", "write_per_protocol_visuals"),
    ):
        monkeypatch.setattr(
            run_phase1.visualize,
            function_name,
            lambda *args, _name=name, **kwargs: calls.append(_name),
        )

    run_phase1.main()

    assert calls == [
        "build_cases",
        "simulations",
        "discover",
        "summary_csv",
        "ratio_breakdown",
        "block_summaries",
        "comparative_suites",
        "single_case_overviews",
        "per_protocol_visuals",
    ]


def test_ic_power_phase_skips_completed_zero_mw(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    zero_dir = tmp_path / "runs" / "ic_power" / "coords_800um__gt__amp_60uA_0mW"
    zero_dir.mkdir(parents=True)
    np.savez(zero_dir / "safety_metrics.npz", max_dT=np.asarray([0.0]), mean_dT=np.asarray([0.0]))

    args = Namespace(
        matrix_config="config/safety_experiments_phase2_ic_power.yaml",
        output_root=str(tmp_path / "runs"),
        visuals_root=str(tmp_path / "previews"),
        visuals_output_root=str(tmp_path / "visuals"),
        safety_yaml="config/safety.yaml",
        include_existing=False,
        dry_run=False,
        no_visuals=True,
        params="config/params.yaml",
        max_frames=0,
        groups=4,
        phosphene_mode="safety_centers",
        save_every_n_frames=None,
        sim_resolution=None,
        thermal_update_interval_frames=None,
        cooldown_seconds=300.0,
        cooldown_baseline_tolerance_C=1e-3,
        preview_seconds=0.0,
        preview_policy="none",
        workers=1,
        force_cpu=True,
        enable_cem43=False,
        format="png",
    )

    monkeypatch.setattr(run_phase2_ic_power, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_phase2_ic_power,
        "run_cases",
        lambda cases, parsed_args: calls.append([case.run_id for case in cases]),
    )

    run_phase2_ic_power.main()

    assert len(calls) == 1
    assert "coords_800um__gt__amp_60uA_0mW" not in calls[0]
    assert len(calls[0]) == 11


def test_ic_power_temperature_visuals_are_temperature_only(tmp_path: Path) -> None:
    for preprocessing in ("dog", "canny", "gt"):
        for power_mw in (0.0, 10.0):
            write_fake_ic_power_record(tmp_path / "ic_power", preprocessing=preprocessing, power_mw=power_mw)

    records = discover_ic_power_records(tmp_path / "ic_power", "config/safety.yaml")
    out_root = tmp_path / "visuals"
    written = write_ic_power_temperature_visuals(
        tmp_path / "ic_power",
        out_root,
        image_format="png",
        overwrite=True,
        safety_yaml="config/safety.yaml",
    )

    assert len(records) == 6
    assert {record.ic_power_mw for record in records} == {0.0, 10.0}
    assert {record.preprocessing for record in records} == {"dog", "canny", "gt"}
    assert (out_root / "temperature_evolution_dog.png").exists()
    assert (out_root / "temperature_evolution_canny.png").exists()
    assert (out_root / "temperature_evolution_gt.png").exists()
    assert (out_root / "max_focal_temperature_grid.png").exists()
    assert (out_root / "max_mean_temperature_grid.png").exists()
    assert (out_root / "temperature_summary.csv").exists()
    assert not (out_root / "charge_over_whole_protocol.png").exists()
    assert len(written) == 6


def test_thermal_update_defaults_to_one_video_frame() -> None:
    params = {"bioheat": {}}

    assert resolve_thermal_update_interval_frames(params, fps=20.0) == 1
    assert resolve_thermal_update_interval_frames(params, fps=29.97) == 1
    assert resolve_thermal_update_interval_frames(
        params,
        fps=20.0,
        override_frames=5,
    ) == 5


def test_cooldown_frame_count_is_independent_of_video_length() -> None:
    assert resolve_cooldown_frame_count(
        fps=20.0,
        cooldown_seconds=300.0,
    ) == 5 * 60 * 20
    assert resolve_cooldown_frame_count(
        fps=20.0,
        cooldown_seconds=0.0,
    ) == 0


def test_safety_visualization_writes_summary_plot(tmp_path: Path) -> None:
    run_dir = tmp_path / "amplitude" / "case"
    run_dir.mkdir(parents=True)
    np.savez(
        run_dir / "safety_metrics.npz",
        time_s=np.asarray([0.1, 0.2], dtype=np.float32),
        electrode_grid_ids=np.asarray([0, 1], dtype=np.int32),
        electrode_xy_mm=np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        amplitude_per_electrode_uA=np.asarray([[60.0, 0.0], [60.0, 30.0]], dtype=np.float32),
        charge_per_phase_per_electrode_nC=np.asarray([[10.0, 0.0], [10.0, 5.0]], dtype=np.float32),
        charge_density_per_electrode_uc_cm2=np.asarray([[333.0, 0.0], [333.0, 166.0]], dtype=np.float32),
        shannon_k_per_electrode=np.asarray([[0.5, -np.inf], [0.5, 0.2]], dtype=np.float32),
        charge_per_second_per_electrode_nC_s=np.asarray([[1000.0, 0.0], [1000.0, 500.0]], dtype=np.float32),
        protocol_charge_per_electrode_nC=np.asarray([1000.0, 500.0], dtype=np.float32),
        window_charge_per_electrode_nC=np.asarray([[1000.0, 0.0], [1000.0, 500.0]], dtype=np.float32),
        window_charge_total_nC=np.asarray([1000.0, 1500.0], dtype=np.float32),
        pulse_width_s=np.asarray([170e-6, 170e-6], dtype=np.float32),
        pulse_frequency_hz=np.asarray([300.0, 300.0], dtype=np.float32),
        relative_stim_duration=np.asarray(0.5, dtype=np.float32),
        electrode_surface_area_cm2=np.asarray(3.0e-5, dtype=np.float32),
        active_count=np.asarray([1, 2], dtype=np.int32),
        mean_dT=np.asarray([0.05, 0.08], dtype=np.float32),
        max_dT=np.asarray([0.1, 0.12], dtype=np.float32),
        area_gt1_mm2=np.asarray([0.0, 0.0], dtype=np.float32),
        area_gt2_mm2=np.asarray([0.0, 0.0], dtype=np.float32),
        area_gt3_mm2=np.asarray([0.0, 0.0], dtype=np.float32),
        heatmap_times_s=np.asarray([0.1, 0.2], dtype=np.float32),
        heatmap_frame_indices=np.asarray([1, 2], dtype=np.int32),
        heatmap_target_fractions=np.asarray([0.5, 1.0], dtype=np.float32),
        heatmap_actual_fractions=np.asarray([0.5, 1.0], dtype=np.float32),
        heatmap_grid_names=np.asarray(["left"]),
        dT_heatmaps_left=np.asarray(
            [
                [[0.01, 0.02], [0.03, 0.04]],
                [[0.02, 0.03], [0.04, 0.05]],
            ],
            dtype=np.float32,
        ),
        extent_mm_left=np.asarray([-1.0, 1.0, -1.0, 1.0], dtype=np.float32),
    )
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "block": "amplitude",
                "run_id": "case",
                "amplitude_uA": 60.0,
                "appearance_threshold_uA": 30.0,
                "coords_yaml": "config/coords_800um.yaml",
                "preprocessing_method": "groundtruth",
            }
        ),
        encoding="utf-8",
    )

    records = visualize.discover_records(tmp_path, Path("config/safety.yaml"))
    with np.load(run_dir / "safety_metrics.npz", allow_pickle=True) as data:
        assert np.allclose(visualize.time_axis_for_length(data, 2), np.asarray([0.1, 0.2]) / 60.0)
    out_root = tmp_path / "visuals"
    comparative_root = out_root / "comparative_visuals"
    visualize.plot_comparative_suites(records, comparative_root, "png", overwrite=True)
    visualize.write_per_protocol_visuals(records, "png", overwrite=True, output_root=out_root)

    assert not (out_root / "overall_safety_margin.png").exists()
    assert (comparative_root / "amplitude" / "shannon_k_over_time.png").exists()
    assert (out_root / "amplitude" / "case" / "charge_over_whole_protocol.png").exists()
    assert (out_root / "amplitude" / "case" / "temperature_heatmaps_left.png").exists()
    assert len(list((out_root / "amplitude" / "case" / "temperature_heatmaps_left").glob("*.png"))) == 2
