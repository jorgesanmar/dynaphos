from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from dynaphos.config import SweepConfig
from dynaphos.experiment import runner


def write_base_experiment(path: Path, output_root: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "input": {"builtin": "cat"},
                "protocol": {"amplitude_uA": 60.0},
                "output": {
                    "root": str(output_root),
                    "run_id": "resume",
                    "overwrite": False,
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("state", "expected_calls"),
    [
        ("completed", 0),
        ("failed", 1),
        ("partial", 1),
        ("corrupt", 1),
    ],
)
def test_sweep_resume_states(
    monkeypatch,
    tmp_path: Path,
    state: str,
    expected_calls: int,
) -> None:
    output_root = tmp_path / "runs"
    experiment_path = tmp_path / "experiment.yaml"
    write_base_experiment(experiment_path, output_root)
    sweep = SweepConfig(
        experiment=experiment_path,
        matrix={"protocol.amplitude_uA": [50.0]},
        output_root=output_root,
    )
    run_dir = output_root / "resume__001__amplitude_uA-50.0"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.yaml"
    if state == "corrupt":
        manifest_path.write_text("schema_version: [", encoding="utf-8")
        np.savez(run_dir / "metrics.npz", value=np.asarray([1.0]))
    else:
        status = {"completed": "completed", "failed": "failed", "partial": "running"}[
            state
        ]
        manifest_path.write_text(
            yaml.safe_dump({"schema_version": 1, "status": status}),
            encoding="utf-8",
        )
        if state == "completed":
            np.savez(run_dir / "metrics.npz", value=np.asarray([1.0]))

    calls = []

    def fake_run_experiment(config):
        calls.append(config)
        return object()

    monkeypatch.setattr(runner, "run_experiment", fake_run_experiment)
    results = runner.run_sweep(sweep, resume=True)

    assert len(calls) == expected_calls
    assert len(results) == expected_calls
    if calls:
        assert calls[0].output.overwrite is True
