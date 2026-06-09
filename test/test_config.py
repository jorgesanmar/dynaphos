from pathlib import Path

import pytest
import yaml

from dynaphos.config import ExperimentConfig, InputConfig, load_experiment, load_sweep


def test_experiment_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="unknown field"):
        ExperimentConfig.from_dict(
            {
                "input": {"path": "input.mp4"},
                "protocol": {"amplitude": 60},
            },
            path="experiment",
        )


def test_experiment_paths_are_relative_to_config(tmp_path: Path) -> None:
    media = tmp_path / "media"
    media.mkdir()
    video = media / "input.mp4"
    video.touch()
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": {"path": "media/input.mp4"},
                "electrode_array": {"coordinates": "array.yaml"},
                "simulation": {"params": "params.yaml"},
                "safety": {"limits": "limits.yaml"},
                "output": {"root": "outputs"},
            }
        ),
        encoding="utf-8",
    )

    config = load_experiment(config_path)

    assert config.input.path == video.resolve()
    assert config.electrode_array.coordinates == (tmp_path / "array.yaml").resolve()
    assert config.output.root == (tmp_path / "outputs").resolve()


def test_custom_strategy_import_path_does_not_require_name() -> None:
    config = ExperimentConfig.from_dict(
        {
            "input": {"path": "input.mp4"},
            "strategy": {"import_path": "package.module:Strategy"},
        }
    )

    assert config.strategy.name is None
    assert config.strategy.import_path == "package.module:Strategy"


def test_direct_dataclass_construction_coerces_paths() -> None:
    config = ExperimentConfig(input=InputConfig(path="input.mp4"))

    assert config.input.path == Path("input.mp4")
    assert config.output.root == Path("results")


def test_sweep_expands_cartesian_product(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text("input:\n  builtin: cat\n", encoding="utf-8")
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        yaml.safe_dump(
            {
                "experiment": "experiment.yaml",
                "matrix": {
                    "protocol.amplitude_uA": [10, 20],
                    "strategy.name": ["direct", "checkerboard"],
                },
            }
        ),
        encoding="utf-8",
    )

    sweep = load_sweep(sweep_path)

    assert sweep.experiment == experiment.resolve()
    assert len(sweep.variants()) == 4
