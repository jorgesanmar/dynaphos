from __future__ import annotations

from itertools import product
from pathlib import Path
from argparse import Namespace
from types import SimpleNamespace
import sys

import numpy as np
import pytest
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
    FIT_POWER_LEVELS_MW,
    FIT_ROLE,
    VALIDATION_ROLE,
    analyze_ic_power,
    build_phase1_budget_rows,
    discover_ic_power_records,
    discover_phase1_records,
    fit_origin_linearity,
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
    raster_groups_to_electrodes,
)
from dynaphos.safety.raster_effects import (
    baseline_mismatch_reasons,
    build_raster_comparison_table_rows,
    build_raster_effect_rows,
    discover_phase1_runs,
    discover_raster_runs,
    remove_obsolete_phase3_plots,
    write_raster_effect_outputs,
)
from dynaphos.safety.tracking import SafetyTracker
from dynaphos.safety import visualize
from dynaphos.simulator import GaussianSimulator, apply_appearance_threshold
from dynaphos.pipeline import render_phosphene_frame_from_state
from dynaphos.utils import Map
from tools.safety import run_phase1_amp_grid_prep as run_phase1
from tools.safety import run_phase2_ic_power
from tools.safety import run_phase3_rastering


def write_fake_matrix_record(
    root: Path,
    *,
    amplitude: float,
    grid_um: int,
    preprocessing: str,
    internal_circuit_power_mw: float = 15.0,
) -> None:
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
                "video": str(
                    Path(
                        f"videos/SANPO/SANPO{'gt' if preprocessing == 'gt' else preprocessing}25min.mp4"
                    ).resolve()
                ),
                "amplitude_uA": amplitude,
                "appearance_threshold_uA": amplitude / 2.0,
                "coords_yaml": f"config/coords_{grid_um}um.yaml",
                "preprocessing_method": "groundtruth" if preprocessing == "gt" else preprocessing,
                "source_input_label": preprocessing,
                "frequency_hz": 300.0,
                "pulse_width_us": 170.0,
                "raster_mode": "none",
                "raster_mode_normalized": "none",
                "raster_groups": 1,
                "internal_circuit_power_mw": internal_circuit_power_mw,
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


def write_fake_ic_power_record(
    root: Path,
    *,
    preprocessing: str,
    power_mw: float,
    grid_um: int = 800,
    amplitude_uA: float = 60.0,
    electrode_heat_enabled: bool = True,
    analysis_role: str | None = None,
    mean_slope: float = 0.015,
    mean_offset: float | None = None,
) -> None:
    suffix = "" if electrode_heat_enabled else "__ic_only"
    run_id = f"coords_{grid_um}um__{preprocessing}__amp_{amplitude_uA:g}uA_{power_mw:g}mW{suffix}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    role = analysis_role or (VALIDATION_ROLE if electrode_heat_enabled else FIT_ROLE)
    offset = (0.05 if electrode_heat_enabled else 0.0) if mean_offset is None else mean_offset
    np.savez(
        run_dir / "safety_metrics.npz",
        time_s=np.asarray([0.0, 60.0, 120.0], dtype=np.float32),
        electrode_grid_ids=np.arange(3, dtype=np.int32),
        thermal_time_s=np.asarray([0.0, 60.0, 120.0], dtype=np.float32),
        max_dT=(
            np.asarray([0.01, 0.02, 0.03], dtype=np.float32) * power_mw
            + (0.1 if electrode_heat_enabled else 0.0)
        ),
        mean_dT=(
            np.asarray([mean_slope / 3.0, mean_slope * 2.0 / 3.0, mean_slope], dtype=np.float32)
            * power_mw
            + offset
        ),
        internal_circuit_power_total_mW=np.asarray(power_mw * 2.0, dtype=np.float32),
    )
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "block": "ic_power",
                "run_id": run_id,
                "amplitude_uA": amplitude_uA,
                "appearance_threshold_uA": amplitude_uA / 2.0,
                "coords_yaml": f"config/coords_{grid_um}um.yaml",
                "preprocessing_method": "groundtruth" if preprocessing == "gt" else preprocessing,
                "source_input_label": preprocessing,
                "internal_circuit_power_mw": power_mw,
                "frequency_hz": 300.0,
                "pulse_width_us": 170.0,
                "raster_mode": "none",
                "ic_heat_mode": "with",
                "electrode_heat_enabled": electrode_heat_enabled,
                "metadata": {"analysis_role": role},
            }
        ),
        encoding="utf-8",
    )


def write_fake_raster_record(
    root: Path,
    *,
    baseline_run_id: str,
    grid_um: int,
    preprocessing: str,
    amplitude: float,
    mode: str,
    groups: int,
    ratio: float,
    role: str = "transfer_panel",
    internal_circuit_power_mw: float = 0.0,
) -> None:
    normalized_mode = "random" if mode == "pseudo_random" else mode
    run_id = f"{role}__{baseline_run_id}__{mode}__groups_{groups}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    np.savez(
        run_dir / "safety_metrics.npz",
        time_s=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        electrode_grid_ids=np.arange(4, dtype=np.int32),
        protocol_charge_per_electrode_nC=np.full(4, amplitude * 30.0 * ratio, dtype=np.float32),
        window_charge_total_nC=np.full(3, amplitude * 40.0 * ratio, dtype=np.float32),
        window_charge_per_electrode_nC=np.full((3, 4), amplitude * 10.0 * ratio, dtype=np.float32),
        mean_dT=np.asarray([0.01, 0.02, 0.03], dtype=np.float32) * amplitude * ratio,
        max_dT=np.asarray([0.02, 0.04, 0.06], dtype=np.float32) * amplitude * ratio,
        active_electrode_count=np.asarray([1, 1, 1], dtype=np.int32),
        raster_rate_hz=np.asarray(15.0 / groups, dtype=np.float32),
    )
    video_label = "gt" if preprocessing == "gt" else preprocessing
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "block": "raster_protocols",
                "run_id": run_id,
                "video": str(Path(f"videos/SANPO/SANPO{video_label}25min.mp4").resolve()),
                "coords_yaml": f"config/coords_{grid_um}um.yaml",
                "preprocessing_method": "groundtruth" if preprocessing == "gt" else preprocessing,
                "source_input_label": preprocessing,
                "amplitude_uA": amplitude,
                "appearance_threshold_uA": amplitude / 2.0,
                "frequency_hz": 300.0,
                "pulse_width_us": 170.0,
                "raster_mode": mode,
                "raster_mode_normalized": normalized_mode,
                "raster_groups": groups,
                "internal_circuit_power_mw": internal_circuit_power_mw,
                "metadata": {
                    "analysis_role": role,
                    "matched_phase1_run_id": baseline_run_id,
                    "expected_duty_fraction": 1.0 / groups,
                    "expected_raster_cycle_rate_hz": 15.0 / groups,
                },
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
    assert preprocessing_inputs["groundtruth"] == "videos/SANPO/SANPOgt25min.mp4"
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
    validation = [case for case in cases if case.metadata["analysis_role"] == VALIDATION_ROLE]
    fit_cases = [case for case in cases if case.metadata["analysis_role"] == FIT_ROLE]

    assert len(cases) == 27
    assert {case.block for case in cases} == {"ic_power"}
    assert len(fit_cases) == 21
    assert len(validation) == 6
    assert all(not case.electrode_heat_enabled for case in fit_cases)
    assert all(case.electrode_heat_enabled for case in validation)
    assert {Path(case.coords_yaml).name for case in cases} == {
        "coords_400um.yaml",
        "coords_800um.yaml",
        "coords_1200um.yaml",
    }
    assert {case.internal_circuit_power_mw for case in fit_cases} == set(FIT_POWER_LEVELS_MW)
    assert {case.internal_circuit_power_mw for case in validation} == {5.0, 50.0}
    assert {(case.preprocessing_method, case.amplitude_uA) for case in validation} == {("canny", 120.0)}
    assert {case.ic_heat_mode for case in cases} == {"with"}
    assert {case.raster_mode for case in cases} == {"none"}


def test_rastering_phase_config_expands_expected_cases() -> None:
    cases = build_cases(matrix_path="config/safety_experiments_phase3_rastering.yaml")
    screen = [case for case in cases if case.metadata["analysis_role"] == "worst_case_screen"]
    transfer = [case for case in cases if case.metadata["analysis_role"] == "transfer_panel"]

    assert len(cases) == 26
    assert {case.block for case in cases} == {"raster_protocols"}
    assert len(screen) == 6
    assert len(transfer) == 20
    assert {Path(case.coords_yaml).name for case in cases} == {
        "coords_400um.yaml",
        "coords_800um.yaml",
        "coords_1200um.yaml",
    }
    assert {case.preprocessing_method for case in cases} == {"canny", "dog", "groundtruth"}
    assert {case.amplitude_uA for case in cases} == {60.0, 120.0}
    assert {case.internal_circuit_power_mw for case in cases} == {0.0}
    assert {case.ic_heat_mode for case in cases} == {"with"}
    assert {case.raster_mode for case in cases} == {"checkerboard", "pseudo_random"}
    assert {case.raster_groups for case in screen} == {3, 4, 5}
    assert {case.raster_groups for case in transfer} == {4, 5}

    expected = {
        (mode, groups, rate)
        for mode in ("checkerboard", "pseudo_random")
        for groups, rate in ((3, 5.0), (4, 3.75), (5, 3.0))
    }
    actual = {
        (
            case.raster_mode,
            case.raster_groups,
            case.metadata["expected_raster_cycle_rate_hz"],
        )
        for case in screen
    }
    assert actual == expected
    assert len({case.run_id for case in cases}) == 26
    assert all(np.isclose(case.metadata["expected_duty_fraction"], 1.0 / case.raster_groups) for case in cases)
    assert all(case.metadata["matched_phase1_run_id"].startswith("coords_") for case in cases)


def test_matrix_axis_metadata_is_merged() -> None:
    cases = build_cases(matrix_path="config/safety_experiments_phase3_rastering.yaml")
    case = next(case for case in cases if case.run_id.endswith("pseudo_random__groups_5"))

    assert case.metadata["analysis_role"] in {"worst_case_screen", "transfer_panel"}
    assert case.metadata["raster_pattern"] == "pseudo_random"
    assert case.metadata["expected_duty_fraction"] == 0.2
    assert case.metadata["expected_raster_cycle_rate_hz"] == 3.0
    assert case.metadata["protocol_id"] == "random__groups_5"


def test_raster_effects_match_phase1_and_calculate_reductions(tmp_path: Path) -> None:
    baseline_id = "coords_400um__canny__amp_120uA"
    write_fake_matrix_record(
        tmp_path / "phase1",
        amplitude=120.0,
        grid_um=400,
        preprocessing="canny",
        internal_circuit_power_mw=0.0,
    )
    for mode in ("checkerboard", "pseudo_random"):
        write_fake_raster_record(
            tmp_path / "phase3",
            baseline_run_id=baseline_id,
            grid_um=400,
            preprocessing="canny",
            amplitude=120.0,
            mode=mode,
            groups=4,
            ratio=0.25,
            role="worst_case_screen",
        )

    phase1 = discover_phase1_runs(tmp_path / "phase1")
    raster = discover_raster_runs(tmp_path / "phase3")
    rows = build_raster_effect_rows(raster, phase1)

    assert len(rows) == 2
    assert all(row["baseline_match_valid"] for row in rows)
    assert all(np.isclose(row["total_protocol_charge_ratio"], 0.25) for row in rows)
    assert all(np.isclose(row["mean_temperature_ratio"], 0.25) for row in rows)
    assert all(np.isclose(row["focal_temperature_reduction_fraction"], 0.75) for row in rows)
    assert all(np.isclose(row["total_protocol_charge_ratio_vs_other_mode"], 0.0) for row in rows)
    table = build_raster_comparison_table_rows(rows, {"400um": 0.02})
    assert np.isclose(table[0]["old_raw_ic_power_budget_mW"], 0.0)
    assert np.isclose(table[0]["new_raw_ic_power_budget_mW"], 55.0)
    assert np.isclose(table[0]["charge_reduction_percent"], 75.0)
    assert np.isclose(table[0]["temperature_reduction_percent"], 75.0)


def test_raster_baseline_matching_rejects_protocol_and_power_mismatches(tmp_path: Path) -> None:
    baseline_id = "coords_400um__canny__amp_120uA"
    write_fake_matrix_record(
        tmp_path / "phase1",
        amplitude=120.0,
        grid_um=400,
        preprocessing="canny",
        internal_circuit_power_mw=0.0,
    )
    write_fake_raster_record(
        tmp_path / "phase3",
        baseline_run_id=baseline_id,
        grid_um=400,
        preprocessing="canny",
        amplitude=120.0,
        mode="checkerboard",
        groups=4,
        ratio=0.25,
        internal_circuit_power_mw=10.0,
    )

    baseline = discover_phase1_runs(tmp_path / "phase1")[baseline_id]
    raster = discover_raster_runs(tmp_path / "phase3")[0]
    reasons = baseline_mismatch_reasons(raster, baseline)

    assert "internal_circuit_power_mw" in reasons
    assert "raster_ic_power" in reasons
    assert build_raster_effect_rows([raster], {baseline_id: baseline})[0]["status"] == "baseline_mismatch"


def test_raster_discovery_ignores_stale_records_without_analysis_role(tmp_path: Path) -> None:
    write_fake_raster_record(
        tmp_path,
        baseline_run_id="coords_400um__canny__amp_120uA",
        grid_um=400,
        preprocessing="canny",
        amplitude=120.0,
        mode="checkerboard",
        groups=4,
        ratio=0.25,
    )
    manifest_path = next(tmp_path.rglob("run_manifest.yaml"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"].pop("analysis_role")
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    assert discover_raster_runs(tmp_path) == []


def test_raster_effects_handle_zero_baseline_metrics(tmp_path: Path) -> None:
    baseline_id = "coords_400um__canny__amp_0uA"
    write_fake_matrix_record(
        tmp_path / "phase1",
        amplitude=0.0,
        grid_um=400,
        preprocessing="canny",
        internal_circuit_power_mw=0.0,
    )
    write_fake_raster_record(
        tmp_path / "phase3",
        baseline_run_id=baseline_id,
        grid_um=400,
        preprocessing="canny",
        amplitude=0.0,
        mode="checkerboard",
        groups=4,
        ratio=0.25,
    )

    rows = build_raster_effect_rows(
        discover_raster_runs(tmp_path / "phase3"),
        discover_phase1_runs(tmp_path / "phase1"),
    )

    assert rows[0]["baseline_match_valid"]
    assert np.isnan(rows[0]["total_protocol_charge_ratio"])
    assert np.isnan(rows[0]["mean_temperature_ratio"])


def test_raster_effect_visualizer_writes_comparison_outputs(tmp_path: Path) -> None:
    baseline_id = "coords_400um__canny__amp_120uA"
    write_fake_matrix_record(
        tmp_path / "phase1",
        amplitude=120.0,
        grid_um=400,
        preprocessing="canny",
        internal_circuit_power_mw=0.0,
    )
    for mode in ("checkerboard", "pseudo_random"):
        write_fake_raster_record(
            tmp_path / "phase3",
            baseline_run_id=baseline_id,
            grid_um=400,
            preprocessing="canny",
            amplitude=120.0,
            mode=mode,
            groups=4,
            ratio=0.25,
            role="worst_case_screen",
        )

    written = write_raster_effect_outputs(
        tmp_path / "phase3",
        tmp_path / "phase1",
        tmp_path / "visuals",
    )

    assert (tmp_path / "visuals" / "raster_effect_summary.csv").exists()
    assert (tmp_path / "visuals" / "raster_effect_budget_table.csv").exists()
    assert (tmp_path / "visuals" / "raster_effect_budget_table.png").exists()
    assert (tmp_path / "visuals" / "worst_case_temperature_evolution.png").exists()
    assert (tmp_path / "visuals" / "duty_fraction_comparison.png").exists()
    assert (tmp_path / "visuals" / "charge_temperature_tradeoff.png").exists()
    assert (tmp_path / "visuals" / "raster_effect_summary.yaml").exists()
    assert len(written) == 12


def test_phase3_visualizer_removes_obsolete_plots(tmp_path: Path) -> None:
    obsolete = tmp_path / "nested" / "mean_temperature_evolution.png"
    retained = tmp_path / "nested" / "charge_temperature_tradeoff.png"
    obsolete.parent.mkdir(parents=True)
    obsolete.write_bytes(b"old")
    retained.write_bytes(b"keep")

    removed = remove_obsolete_phase3_plots(tmp_path)

    assert removed == [obsolete]
    assert not obsolete.exists()
    assert retained.exists()


def test_charge_grid_titles_use_current_safety_limits() -> None:
    visualize.set_plot_safety_limits(visualize.load_safety_limits("config/safety.yaml"))

    assert visualize.title_with_charge_limit(
        "Mean Charge Per Second Per Electrode",
        "charge_per_second",
        "uC/s/electrode",
    ).endswith("(per-electrode limit: 2 uC/s/electrode)")
    assert visualize.title_with_charge_limit(
        "Mean Summed Charge Per Second",
        "summed_charge_per_second",
        "mC/s",
    ).endswith("(total-array limit: 150.0 mC/s)")
    assert visualize.title_with_charge_limit(
        "Summed Total Charge",
        "total_charge",
        "mC",
    ).endswith("(session limit: 938.0 mC)")


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

    matrix_root = out_root
    for grid_um in (400, 800, 1200):
        assert (matrix_root / "active_electrodes" / f"active_electrodes_{grid_um}um.png").exists()
    assert not (matrix_root / "active_electrodes" / "active_electrodes_by_grid_preprocessing.png").exists()
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


def test_hot_grid_text_color_tracks_cell_luminance() -> None:
    assert visualize.contrasting_cell_text_color(0.0, vmin=0.0, vmax=1.0, cmap="hot") == "white"
    assert visualize.contrasting_cell_text_color(1.0, vmin=0.0, vmax=1.0, cmap="hot") == "black"
    assert visualize.contrasting_cell_text_color(0.0, vmin=0.0, vmax=1.0, cmap="viridis") == "white"
    assert visualize.contrasting_cell_text_color(1.0, vmin=0.0, vmax=1.0, cmap="viridis") == "black"
    assert visualize.contrasting_cell_text_color(0.5, vmin=0.0, vmax=1.0, cmap="RdYlGn") == "black"
    assert visualize.contrasting_cell_text_color(np.nan, vmin=0.0, vmax=1.0, cmap="hot") == visualize.FALLBACK_COLOR


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


def test_raster_groups_to_electrodes_vectorized_mode_and_missing_values() -> None:
    sim = SimpleNamespace(
        raster_enabled=True,
        raster_num_groups=3,
        raster_groups_flat=torch.tensor([2, 1, 2, 2, 1], dtype=torch.long),
    )
    inv_map = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

    groups = raster_groups_to_electrodes(sim, inv_map, n_elec=3)

    assert torch.equal(groups, torch.tensor([1, 2, -1], dtype=torch.int32))


def test_pseudo_random_reshuffle_increments_assignment_version() -> None:
    sim = SimpleNamespace(
        raster_groups_flat=torch.tensor([0, 1, 2], dtype=torch.long),
        raster_num_groups=3,
        raster_assignment_version=0,
        _raster_group_pool=[
            torch.tensor([0, 1, 2], dtype=torch.long),
            torch.tensor([2, 0, 1], dtype=torch.long),
        ],
        _raster_group_pool_index=0,
        _sorted_raster_coords=None,
        raster_schedule=[
            torch.zeros((3, 1, 1)),
            torch.zeros((3, 1, 1)),
            torch.zeros((3, 1, 1)),
        ],
        shape=(3, 1, 1),
    )

    GaussianSimulator._reshuffle_random_pattern(sim)

    assert sim.raster_assignment_version == 1
    assert torch.equal(sim.raster_groups_flat, torch.tensor([2, 0, 1]))


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
    assert manifest["electrode_heat_enabled"] is True
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
    assert args.phase1_input_root.endswith("amplitude_grid_preprocessing")
    assert args.format == "png"


def test_ic_power_phase_defaults_to_all_cases(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["run_phase2_ic_power.py", "--dry-run"])

    args = run_phase2_ic_power.parse_args()

    assert args.baseline_only is False
    assert args.phase1_input_root.endswith("amplitude_grid_preprocessing")
    assert args.temperature_limit_C == 2.0
    assert args.power_derating_factor == 0.9


def test_ic_power_baseline_only_selects_twenty_one_cases() -> None:
    cases = build_cases(matrix_path="config/safety_experiments_phase2_ic_power.yaml")

    selected = run_phase2_ic_power.select_cases(cases, baseline_only=True)

    assert len(selected) == 21
    assert all(not case.electrode_heat_enabled for case in selected)
    assert {case.internal_circuit_power_mw for case in selected} == set(FIT_POWER_LEVELS_MW)


def test_experiment_matrix_runs_visualizer_after_simulations(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    args = Namespace(
        blocks=None,
        matrix_config="config/safety_experiments_phase1_amp_grid_prep.yaml",
        output_root=str(tmp_path / "runs"),
        visuals_root=str(tmp_path / "visuals"),
        safety_yaml="config/safety.yaml",
        include_existing=False,
        dry_run=False,
    )

    monkeypatch.setattr(run_phase1, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_phase1,
        "build_cases",
        lambda *, blocks, matrix_path: calls.append("build_cases") or [
            SimulationCase(
                block="amplitude_grid_preprocessing",
                run_id="case",
                video="video.mp4",
                coords_yaml="coords.yaml",
                preprocessing_method="canny",
            )
        ],
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
        "comparative_suites",
        "single_case_overviews",
        "per_protocol_visuals",
    ]


def test_ic_power_phase_skips_completed_zero_mw(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    zero_dir = tmp_path / "runs" / "ic_power" / "coords_800um__ic_only__0mW"
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
        phase1_input_root=str(tmp_path / "phase1"),
        temperature_limit_C=2.0,
        power_derating_factor=0.9,
    )

    monkeypatch.setattr(run_phase2_ic_power, "parse_args", lambda: args)
    monkeypatch.setattr(
        run_phase2_ic_power,
        "run_cases",
        lambda cases, parsed_args: calls.append([case.run_id for case in cases]),
    )

    run_phase2_ic_power.main()

    assert len(calls) == 1
    assert "coords_800um__ic_only__0mW" not in calls[0]
    assert len(calls[0]) == 26


def test_ic_power_temperature_visuals_are_temperature_only(tmp_path: Path) -> None:
    for grid_um in (400, 800, 1200):
        write_fake_matrix_record(tmp_path / "phase1", amplitude=120.0, grid_um=grid_um, preprocessing="canny")
        for power_mw in FIT_POWER_LEVELS_MW:
            write_fake_ic_power_record(
                tmp_path / "ic_power",
                preprocessing="baseline",
                grid_um=grid_um,
                power_mw=power_mw,
                electrode_heat_enabled=False,
            )
        for power_mw in (5.0, 50.0):
            write_fake_ic_power_record(
                tmp_path / "ic_power",
                preprocessing="canny",
                grid_um=grid_um,
                amplitude_uA=120.0,
                power_mw=power_mw,
                mean_offset=3.6,
            )

    records = discover_ic_power_records(tmp_path / "ic_power", "config/safety.yaml")
    out_root = tmp_path / "visuals"
    written = write_ic_power_temperature_visuals(
        tmp_path / "ic_power",
        out_root,
        image_format="png",
        overwrite=True,
        safety_yaml="config/safety.yaml",
        phase1_input_root=tmp_path / "phase1",
    )

    assert len(records) == 27
    assert {record.grid for record in records} == {"400um", "800um", "1200um"}
    assert (out_root / "temperature_evolution_400um_canny_amp_120uA.png").exists()
    assert (out_root / "ic_only_temperature_evolution_400um.png").exists()
    assert (out_root / "ic_only_temperature_evolution_800um.png").exists()
    assert (out_root / "ic_only_temperature_evolution_1200um.png").exists()
    assert (out_root / "ic_power_linearity_800um.png").exists()
    assert (out_root / "ic_power_linearity_summary.csv").exists()
    assert (out_root / "phase1_ic_power_budget.csv").exists()
    assert (out_root / "phase1_ic_power_budget.png").exists()
    assert (out_root / "ic_power_safety_summary.yaml").exists()
    assert (out_root / "max_focal_temperature_grid.png").exists()
    assert (out_root / "max_mean_temperature_grid.png").exists()
    assert (out_root / "temperature_summary.csv").exists()
    assert (out_root / "joule_heating_offset_summary.csv").exists()
    assert (out_root / "joule_heating_offset_800um_canny_amp_120uA.png").exists()
    assert not (out_root / "charge_over_whole_protocol.png").exists()
    assert len(written) == 21


def test_ic_power_origin_fit_uses_configured_per_ic_power(tmp_path: Path) -> None:
    for power_mw in FIT_POWER_LEVELS_MW:
        write_fake_ic_power_record(
            tmp_path,
            preprocessing="baseline",
            power_mw=power_mw,
            electrode_heat_enabled=False,
            mean_slope=0.02,
        )

    records = discover_ic_power_records(tmp_path)
    result = fit_origin_linearity(records)

    assert np.isclose(result.slope_C_per_mW, 0.02, rtol=1e-6)
    assert result.linearity_pass
    assert all(np.isclose(record.reported_total_power_mw, 2.0 * record.ic_power_mw) for record in records)


def test_ic_power_fit_reports_nonzero_intercept_and_rejects_nonlinearity(tmp_path: Path) -> None:
    for power_mw in FIT_POWER_LEVELS_MW:
        write_fake_ic_power_record(
            tmp_path,
            preprocessing="baseline",
            power_mw=power_mw,
            electrode_heat_enabled=False,
            mean_slope=0.02 if power_mw != 35.0 else 0.03,
            mean_offset=0.1,
        )

    result = fit_origin_linearity(discover_ic_power_records(tmp_path))

    assert result.free_intercept_C > 0.0
    assert not result.linearity_pass
    assert result.status == "linearity_failed"


def test_ic_power_fit_marks_missing_power_coverage_invalid(tmp_path: Path) -> None:
    for power_mw in FIT_POWER_LEVELS_MW[:-1]:
        write_fake_ic_power_record(
            tmp_path,
            preprocessing="baseline",
            power_mw=power_mw,
            electrode_heat_enabled=False,
        )

    result = fit_origin_linearity(discover_ic_power_records(tmp_path))

    assert not result.coverage_complete
    assert not result.linearity_pass
    assert result.status == "incomplete_power_coverage"


def test_ic_power_fit_rejects_duplicate_grid_power_records(tmp_path: Path) -> None:
    write_fake_ic_power_record(
        tmp_path / "a",
        preprocessing="baseline",
        power_mw=10.0,
        electrode_heat_enabled=False,
    )
    write_fake_ic_power_record(
        tmp_path / "b",
        preprocessing="baseline",
        power_mw=10.0,
        electrode_heat_enabled=False,
    )

    with pytest.raises(ValueError, match="Duplicate IC-only fit record"):
        fit_origin_linearity(discover_ic_power_records(tmp_path))


def test_ic_power_discovery_ignores_records_without_analysis_role(tmp_path: Path) -> None:
    write_fake_ic_power_record(
        tmp_path,
        preprocessing="baseline",
        power_mw=10.0,
        electrode_heat_enabled=False,
    )
    manifest_path = next(tmp_path.rglob("run_manifest.yaml"))
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("metadata")
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    assert discover_ic_power_records(tmp_path) == []


def test_phase1_budget_uses_grid_fit_and_clamps_over_limit_to_zero(tmp_path: Path) -> None:
    for power_mw in FIT_POWER_LEVELS_MW:
        write_fake_ic_power_record(
            tmp_path / "ic_power",
            preprocessing="baseline",
            power_mw=power_mw,
            electrode_heat_enabled=False,
            mean_slope=0.02,
        )
    write_fake_matrix_record(tmp_path / "phase1", amplitude=120.0, grid_um=800, preprocessing="canny")
    phase1_records = discover_phase1_records(tmp_path / "phase1")
    fit = fit_origin_linearity(discover_ic_power_records(tmp_path / "ic_power"))
    valid_fit = fit.__class__(
        **{
            **fit.__dict__,
            "additivity_pass": True,
            "analysis_valid": True,
            "status": "pass",
        }
    )

    rows = build_phase1_budget_rows(
        phase1_records,
        [valid_fit],
        temperature_limit_C=2.0,
        power_derating_factor=0.9,
    )

    assert rows[0]["status"] == "mean_limit_already_reached"
    assert rows[0]["raw_max_ic_power_mW"] == 0.0
    assert rows[0]["recommended_max_ic_power_mW"] == 0.0
    assert rows[0]["focal_limit_exceeded"] is True


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
    assert (comparative_root / "shannon_k_over_time.png").exists()
    assert (run_dir / "charge_over_whole_protocol.png").exists()
    assert (run_dir / "temperature_heatmaps_left.png").exists()
    assert len(list((run_dir / "temperature_heatmaps_left").glob("*.png"))) == 2
