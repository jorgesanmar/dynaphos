from pathlib import Path

import pytest
import yaml

from dynaphos.cli import main


def test_cli_lists_strategies(capsys) -> None:
    assert main(["strategies", "list"]) == 0
    assert "direct" in capsys.readouterr().out


def test_cli_validates_bundled_fixture(tmp_path: Path, capsys) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "input": {
                    "builtin": "cat",
                    "stage": "original",
                    "preprocessing_method": "dog",
                }
            }
        ),
        encoding="utf-8",
    )

    assert main(["validate", str(config)]) == 0
    assert "Valid experiment" in capsys.readouterr().out


def test_cli_forwards_render_arguments(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(
        "dynaphos.media.rendering.main",
        lambda argv=None: captured.extend(argv or []),
    )
    assert main(["render", "image", "--input", "example.png"]) == 0
    assert captured == ["--media-type", "image", "--input", "example.png"]


def test_cli_forwards_video_render_arguments(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(
        "dynaphos.media.rendering.main",
        lambda argv=None: captured.extend(argv or []),
    )
    assert main(["render", "video", "--input", "example.mp4"]) == 0
    assert captured == ["--media-type", "video", "--input", "example.mp4"]


def test_cli_forwards_preprocessing_arguments(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(
        "dynaphos.media.comparison.main",
        lambda argv=None: captured.extend(argv or []),
    )
    assert main(["preprocess", "compare", "--input", "example.mp4"]) == 0
    assert captured == ["--input", "example.mp4"]


def test_cli_forwards_electrode_arguments(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(
        "dynaphos.electrodes.visualization.main",
        lambda argv=None: captured.extend(argv or []),
    )
    assert main(["electrodes", "plot", "--grid", "array.yaml"]) == 0
    assert captured == ["--grid", "array.yaml"]


def test_cli_forwards_sweep_resume(monkeypatch, tmp_path: Path) -> None:
    sweep = object()
    captured = {}
    monkeypatch.setattr("dynaphos.cli.load_sweep", lambda path: sweep)

    def fake_run_sweep(config, *, resume=False):
        captured.update(config=config, resume=resume)
        return []

    monkeypatch.setattr("dynaphos.cli.run_sweep", fake_run_sweep)
    assert main(["sweep", str(tmp_path / "sweep.yaml"), "--resume"]) == 0
    assert captured == {"config": sweep, "resume": True}


@pytest.mark.parametrize(
    ("phase", "arguments", "target"),
    [
        ("phase1", ["phase1"], "run_phase1_analysis"),
        (
            "phase2",
            ["phase2", "--phase1-root", "phase1"],
            "run_phase2_analysis",
        ),
        (
            "phase3",
            [
                "phase3",
                "--phase1-root",
                "phase1",
                "--phase2-root",
                "phase2",
            ],
            "run_phase3_analysis",
        ),
    ],
)
def test_cli_dispatches_study_commands(
    monkeypatch,
    tmp_path: Path,
    phase: str,
    arguments: list[str],
    target: str,
) -> None:
    captured = []

    def fake_analysis(*args, **kwargs):
        captured.append((args, kwargs))
        return [tmp_path / f"{phase}.csv"]

    monkeypatch.setattr(f"dynaphos.studies.{target}", fake_analysis)
    argv = ["study", *arguments[:1], str(tmp_path / phase), *arguments[1:]]
    assert main(argv) == 0
    assert captured
