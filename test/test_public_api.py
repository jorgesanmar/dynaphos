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
        "run_manifest.yaml",
        "report.html",
        "report.json",
        "summary.csv",
        "metrics.npz",
    ):
        assert (result.run_dir / name).exists()
    assert (result.run_dir / "safety_metrics.npz").exists()
    with np.load(result.metrics_path) as data:
        assert np.isclose(np.max(data["pulse_width_per_electrode_s"]), 50e-6)
        assert np.isclose(np.max(data["pulse_frequency_per_electrode_hz"]), 20.0)
