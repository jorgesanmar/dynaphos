from pathlib import Path

import numpy as np

from dynaphos import ExperimentConfig, run_experiment
from dynaphos.strategies import StimulusCommand, StrategyContext, StrategyInput


class FixedPulseStrategy:
    def __init__(self, options: dict | None = None):
        self.options = options or {}
        self.context = None

    def initialize(self, context: StrategyContext) -> None:
        self.context = context

    def step(self, frame: StrategyInput) -> StimulusCommand:
        return StimulusCommand(
            amplitude_A=frame.baseline_amplitude_A,
            pulse_width_s=np.full(frame.electrode_ids.size, 50e-6),
            frequency_hz=np.full(frame.electrode_ids.size, 20.0),
            activation_mask=np.ones(frame.electrode_ids.size, dtype=bool),
        )


def test_tiny_cpu_experiment_writes_public_output_contract(tmp_path: Path) -> None:
    config = ExperimentConfig.from_dict(
        {
            "input": {
                "builtin": "cat",
                "stage": "original",
                "preprocessing_method": "dog",
            },
            "electrode_array": {"builtin": "coords_1200um"},
            "protocol": {
                "amplitude_uA": 10,
                "appearance_threshold_uA": 5,
                "pulse_width_us": 170,
                "frequency_hz": 100,
                "relative_stim_duration": 1,
                "internal_circuit_power_mW": 0,
            },
            "strategy": {
                "import_path": "test.test_public_api:FixedPulseStrategy",
            },
            "simulation": {
                "force_cpu": True,
                "resolution": 32,
                "max_frames": 1,
                "preview_seconds": 0,
                "phosphene_mode": "safety_centers",
                "cooldown_seconds": 0,
            },
            "output": {
                "root": str(tmp_path),
                "run_id": "tiny",
                "write_figures": False,
            },
        }
    )

    result = run_experiment(config)

    assert result.run_dir == tmp_path / "tiny"
    for name in (
        "manifest.yaml",
        "report.html",
        "report.json",
        "summary.csv",
        "metrics.npz",
    ):
        assert (result.run_dir / name).exists()
    assert not (result.run_dir / "run_manifest.yaml").exists()
    assert not (result.run_dir / "safety_metrics.npz").exists()
    assert not (result.run_dir / "summary.txt").exists()
    with np.load(result.metrics_path) as data:
        assert int(data["save_every_n_frames"]) == 1
        assert np.isclose(np.max(data["pulse_width_per_electrode_s"]), 50e-6)
        assert np.isclose(np.max(data["pulse_frequency_per_electrode_hz"]), 20.0)
        assert np.isfinite(float(data["peak_focal_time_s"]))
        assert np.isfinite(float(data["peak_mean_time_s"]))
        assert any(key.startswith("dT_peak_focal_") for key in data.files)
        assert any(key.startswith("dT_peak_mean_") for key in data.files)


def phase_config(
    tmp_path: Path,
    *,
    run_id: str,
    amplitude_uA: float,
    appearance_threshold_uA: float,
    internal_circuit_power_mW: float = 0.0,
    track_electrical: bool = True,
    electrode_heat_enabled: bool = True,
    experiment_block: str = "amplitude_grid_preprocessing",
) -> ExperimentConfig:
    return ExperimentConfig.from_dict(
        {
            "input": {
                "builtin": "cat",
                "stage": "original",
                "preprocessing_method": "dog",
            },
            "electrode_array": {"builtin": "coords_1200um"},
            "protocol": {
                "amplitude_uA": amplitude_uA,
                "appearance_threshold_uA": appearance_threshold_uA,
                "pulse_width_us": 170,
                "frequency_hz": 100,
                "relative_stim_duration": 1,
                "internal_circuit_power_mW": internal_circuit_power_mW,
            },
            "strategy": {"name": "direct"},
            "simulation": {
                "force_cpu": True,
                "resolution": 32,
                "max_frames": 1,
                "preview_seconds": 0,
                "phosphene_mode": "safety_centers",
                "cooldown_seconds": 0,
            },
            "safety": {
                "track_electrical": track_electrical,
                "electrode_heat_enabled": electrode_heat_enabled,
            },
            "metadata": {"experiment_block": experiment_block},
            "output": {
                "root": str(tmp_path / "runs"),
                "run_id": run_id,
                "write_figures": False,
            },
        }
    )


def test_phase_amplitude_runs_reuse_stimulation_cache(tmp_path: Path) -> None:
    first = run_experiment(
        phase_config(
            tmp_path,
            run_id="phase1-low",
            amplitude_uA=10.0,
            appearance_threshold_uA=5.0,
        )
    )
    second = run_experiment(
        phase_config(
            tmp_path,
            run_id="phase1-high",
            amplitude_uA=60.0,
            appearance_threshold_uA=30.0,
        )
    )
    uncached = run_experiment(
        phase_config(
            tmp_path,
            run_id="phase1-high-uncached",
            amplitude_uA=60.0,
            appearance_threshold_uA=30.0,
            experiment_block="uncached_test",
        )
    )

    with np.load(first.metrics_path) as first_data:
        assert not bool(first_data["stimulation_cache_hit"])
        cache_path = Path(str(first_data["stimulation_cache_path"]))
        assert cache_path.exists()
    with np.load(second.metrics_path) as second_data:
        assert bool(second_data["stimulation_cache_hit"])
        assert Path(str(second_data["stimulation_cache_path"])) == cache_path
        with np.load(uncached.metrics_path) as uncached_data:
            for key in (
                "amplitude_per_electrode_uA",
                "charge_per_phase_per_electrode_nC",
                "power_per_electrode_W",
                "max_dT",
                "mean_dT",
            ):
                assert np.allclose(
                    second_data[key],
                    uncached_data[key],
                    rtol=1e-6,
                    atol=1e-9,
                    equal_nan=True,
                )


def test_ic_only_run_skips_video_stimulation_pipeline(tmp_path: Path) -> None:
    result = run_experiment(
        phase_config(
            tmp_path,
            run_id="phase2-ic-only",
            amplitude_uA=0.0,
            appearance_threshold_uA=0.0,
            internal_circuit_power_mW=5.0,
            track_electrical=False,
            electrode_heat_enabled=False,
            experiment_block="ic_power",
        )
    )
    reference = run_experiment(
        phase_config(
            tmp_path,
            run_id="phase2-ic-reference",
            amplitude_uA=0.0,
            appearance_threshold_uA=0.0,
            internal_circuit_power_mW=5.0,
            track_electrical=True,
            electrode_heat_enabled=False,
            experiment_block="ic_reference_test",
        )
    )

    with np.load(result.metrics_path) as data:
        assert bool(data["ic_only_fast_path"])
        assert not bool(data["stimulation_cache_hit"])
        assert "amplitude_per_electrode_uA" not in data.files
        assert float(np.max(data["max_dT"])) > 0.0
        with np.load(reference.metrics_path) as reference_data:
            assert np.allclose(data["max_dT"], reference_data["max_dT"])
            assert np.allclose(data["mean_dT"], reference_data["mean_dT"])
