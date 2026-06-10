from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dynaphos.experiment.artifacts import discover_completed_runs, open_run
from dynaphos.experiment.manifest import load_manifest
from dynaphos.experiment.metrics import load_metrics
from dynaphos.studies.phase1 import discover_records, run_phase1_analysis
from dynaphos.studies.phase2 import discover_ic_power_records, run_phase2_analysis
from dynaphos.studies.phase3 import (
    discover_phase1_runs,
    discover_raster_runs,
    run_phase3_analysis,
)


def write_run(
    root: Path,
    run_id: str,
    *,
    block: str,
    status: str = "completed",
    metadata: dict | None = None,
    write_metrics: bool = True,
    manifest_overrides: dict | None = None,
    metrics_overrides: dict | None = None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "block": block,
        "video": "inputs/video.mp4",
        "coords_yaml": "arrays/coords_800um.yaml",
        "source_input_label": "canny",
        "preprocessing_method": "canny",
        "amplitude_uA": 60.0,
        "appearance_threshold_uA": 30.0,
        "pulse_width_us": 170.0,
        "frequency_hz": 300.0,
        "internal_circuit_power_mw": 0.0,
        "raster_mode_normalized": "none",
        "raster_groups": 1,
        "electrode_heat_enabled": True,
        "metadata": metadata or {},
    }
    manifest.update(manifest_overrides or {})
    (run_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    if write_metrics:
        amplitude = np.asarray([[60.0, 0.0], [60.0, 30.0]], dtype=np.float32)
        metrics = {
            "time_s": np.asarray([0.0, 1.0], dtype=np.float32),
            "thermal_time_s": np.asarray([0.0, 1.0], dtype=np.float32),
            "electrode_grid_ids": np.asarray([0, 1], dtype=np.int32),
            "electrode_ids": np.asarray([0, 1], dtype=np.int32),
            "electrode_xy_mm": np.asarray(
                [[0.0, 0.0], [1.0, 0.0]],
                dtype=np.float32,
            ),
            "amplitude_per_electrode_uA": amplitude,
            "charge_per_phase_per_electrode_nC": amplitude * 0.17,
            "charge_density_per_electrode_uc_cm2": amplitude,
            "shannon_k_per_electrode": np.where(amplitude > 0, 0.5, -np.inf),
            "charge_per_second_per_electrode_nC_s": amplitude * 10.0,
            "protocol_charge_per_electrode_nC": np.asarray([1000.0, 500.0]),
            "window_charge_per_electrode_nC": amplitude,
            "window_charge_total_nC": np.asarray([60.0, 90.0]),
            "pulse_width_s": np.asarray([170e-6, 170e-6]),
            "pulse_frequency_hz": np.asarray([300.0, 300.0]),
            "mean_dT": np.asarray([0.1, 0.2]),
            "max_dT": np.asarray([0.2, 0.3]),
            "cooldown_start_s": np.asarray(1.0),
        }
        metrics.update(metrics_overrides or {})
        np.savez(run_dir / "metrics.npz", **metrics)
    return run_dir


def test_discovery_skips_failed_runs(tmp_path: Path) -> None:
    completed = write_run(tmp_path, "completed", block="phase")
    write_run(tmp_path, "failed", block="phase", status="failed", write_metrics=False)
    with pytest.warns(RuntimeWarning, match="status"):
        runs = discover_completed_runs(tmp_path)
    assert [run.run_dir for run in runs] == [completed]


def test_completed_manifest_without_metrics_is_an_error(tmp_path: Path) -> None:
    run_dir = write_run(
        tmp_path,
        "broken",
        block="phase",
        write_metrics=False,
    )
    with pytest.raises(FileNotFoundError, match="metrics.npz"):
        open_run(run_dir)
    with pytest.raises(FileNotFoundError, match="metrics.npz"):
        discover_completed_runs(tmp_path)


def test_canonical_readers_reject_alias_filenames(tmp_path: Path) -> None:
    manifest_alias = tmp_path / "run_manifest.yaml"
    manifest_alias.write_text(
        yaml.safe_dump({"schema_version": 1, "status": "completed"}),
        encoding="utf-8",
    )
    metrics_alias = tmp_path / "safety_metrics.npz"
    np.savez(metrics_alias, value=np.asarray([1.0]))

    with pytest.raises(ValueError, match="manifest.yaml"):
        load_manifest(manifest_alias)
    with pytest.raises(ValueError, match="metrics.npz"):
        load_metrics(metrics_alias)


def test_phase_discovery_uses_only_canonical_artifacts(tmp_path: Path) -> None:
    phase1_root = tmp_path / "phase1"
    phase2_root = tmp_path / "phase2"
    phase3_root = tmp_path / "phase3"
    baseline = write_run(
        phase1_root,
        "baseline",
        block="amplitude_grid_preprocessing",
    )
    phase2 = write_run(
        phase2_root,
        "ic",
        block="ic_power",
        metadata={"analysis_role": "ic_only_fit"},
    )
    manifest_path = phase2 / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["electrode_heat_enabled"] = False
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    raster = write_run(
        phase3_root,
        "raster",
        block="raster_protocols",
        metadata={
            "analysis_role": "transfer_panel",
            "matched_phase1_run_id": "baseline",
            "expected_duty_fraction": 0.25,
            "expected_raster_cycle_rate_hz": 3.75,
        },
    )
    raster_manifest = yaml.safe_load(
        (raster / "manifest.yaml").read_text(encoding="utf-8")
    )
    raster_manifest.update(
        {"raster_mode_normalized": "checkerboard", "raster_groups": 4}
    )
    (raster / "manifest.yaml").write_text(
        yaml.safe_dump(raster_manifest, sort_keys=False),
        encoding="utf-8",
    )

    phase1_records = discover_records(phase1_root, Path("config/safety.yaml"))
    phase2_records = discover_ic_power_records(phase2_root)
    phase1_runs = discover_phase1_runs(phase1_root)
    phase3_runs = discover_raster_runs(phase3_root)

    assert len(phase1_records) == 1
    assert len(phase2_records) == 1
    assert set(phase1_runs) == {"baseline"}
    assert len(phase3_runs) == 1
    assert phase3_runs[0].manifest_path.name == "manifest.yaml"
    assert baseline.exists()


def test_phase1_analysis_writes_synthetic_outputs(tmp_path: Path) -> None:
    root = tmp_path / "phase1"
    output = tmp_path / "phase1-analysis"
    write_run(root, "baseline", block="amplitude_grid_preprocessing")

    written = run_phase1_analysis(root, output)

    assert output / "matrix_summary.csv" in written
    assert any(path.suffix == ".png" for path in written)


def test_phase2_analysis_writes_synthetic_outputs(tmp_path: Path) -> None:
    phase1_root = tmp_path / "phase1"
    phase2_root = tmp_path / "phase2"
    output = tmp_path / "phase2-analysis"
    for preprocessing in ("canny", "dog"):
        write_run(
            phase1_root,
            f"baseline-{preprocessing}",
            block="amplitude_grid_preprocessing",
            manifest_overrides={
                "preprocessing_method": preprocessing,
                "source_input_label": preprocessing,
            },
        )

    slope = 0.01
    for power in (0.0, 5.0, 10.0, 20.0, 35.0, 50.0, 65.0):
        write_run(
            phase2_root,
            f"fit-{power:g}",
            block="ic_power",
            metadata={"analysis_role": "ic_only_fit"},
            manifest_overrides={
                "internal_circuit_power_mw": power,
                "electrode_heat_enabled": False,
            },
            metrics_overrides={
                "mean_dT": np.asarray([0.0, slope * power]),
                "max_dT": np.asarray([0.0, 1.5 * slope * power]),
                "internal_circuit_power_total_mW": np.asarray(power * 2.0),
            },
        )
    for preprocessing in ("canny", "dog"):
        write_run(
            phase2_root,
            f"validation-{preprocessing}",
            block="ic_power",
            metadata={"analysis_role": "additivity_validation"},
            manifest_overrides={
                "preprocessing_method": preprocessing,
                "source_input_label": preprocessing,
                "internal_circuit_power_mw": 20.0,
                "electrode_heat_enabled": True,
            },
            metrics_overrides={
                "mean_dT": np.asarray([0.2, 0.4]),
                "max_dT": np.asarray([0.3, 0.6]),
                "internal_circuit_power_total_mW": np.asarray(40.0),
            },
        )

    written = run_phase2_analysis(phase2_root, phase1_root, output)

    assert output / "ic_power_linearity_summary.csv" in written
    assert output / "phase1_ic_power_budget.csv" in written
    assert any(path.suffix == ".png" for path in written)


def test_phase3_analysis_writes_synthetic_outputs(tmp_path: Path) -> None:
    phase1_root = tmp_path / "phase1"
    phase2_root = tmp_path / "phase2"
    phase3_root = tmp_path / "phase3"
    output = tmp_path / "phase3-analysis"
    write_run(phase1_root, "baseline", block="amplitude_grid_preprocessing")

    raster_metrics = {
        "protocol_charge_per_electrode_nC": np.asarray([500.0, 250.0]),
        "window_charge_per_electrode_nC": np.asarray(
            [[30.0, 0.0], [30.0, 15.0]]
        ),
        "window_charge_total_nC": np.asarray([30.0, 45.0]),
        "mean_dT": np.asarray([0.05, 0.1]),
        "max_dT": np.asarray([0.1, 0.15]),
        "raster_rate_hz": np.asarray(3.75),
    }
    for role, mode in (
        ("worst_case_screen", "checkerboard"),
        ("transfer_panel", "random"),
    ):
        write_run(
            phase3_root,
            f"{role}-{mode}",
            block="raster_protocols",
            metadata={
                "analysis_role": role,
                "matched_phase1_run_id": "baseline",
                "expected_duty_fraction": 0.5,
                "expected_raster_cycle_rate_hz": 3.75,
            },
            manifest_overrides={
                "raster_mode_normalized": mode,
                "raster_groups": 4,
            },
            metrics_overrides=raster_metrics,
        )

    phase2_output = phase2_root / "comparative_visuals"
    phase2_output.mkdir(parents=True)
    (phase2_output / "ic_power_linearity_summary.csv").write_text(
        "grid,slope_C_per_mW,analysis_valid\n800um,0.01,True\n",
        encoding="utf-8",
    )

    written = run_phase3_analysis(phase3_root, phase1_root, phase2_root, output)

    assert output / "raster_effect_summary.csv" in written
    assert output / "raster_effect_summary.yaml" in written
    assert any(path.suffix == ".png" for path in written)
