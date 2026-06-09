from pathlib import Path

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
