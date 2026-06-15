from __future__ import annotations

from dynaphos.experiment import execution as runner
from dynaphos.experiment import runner as experiment_runner


class BrokenStderr:
    def write(self, value: str) -> int:
        raise OSError(22, "Invalid argument")

    def flush(self) -> None:
        raise OSError(22, "Invalid argument")


class RedirectedStderr:
    def __init__(self) -> None:
        self.writes = 0

    def isatty(self) -> bool:
        return False

    def write(self, value: str) -> int:
        self.writes += 1
        return len(value)

    def flush(self) -> None:
        pass


def test_progress_output_disables_itself_after_write_failure(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_PROGRESS_OUTPUT_ENABLED", True)
    monkeypatch.setattr(runner, "_PROGRESS_LINE_LEN", 0)
    monkeypatch.setattr(runner.sys, "stderr", BrokenStderr())

    runner.write_progress_line("frame 1")
    runner.write_progress_line("frame 2")

    assert runner._PROGRESS_OUTPUT_ENABLED is False
    assert runner._PROGRESS_LINE_LEN == 0


def test_progress_output_disables_itself_after_finish_failure(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_PROGRESS_OUTPUT_ENABLED", True)
    monkeypatch.setattr(runner, "_PROGRESS_LINE_LEN", 7)
    monkeypatch.setattr(runner.sys, "stderr", BrokenStderr())

    runner.finish_progress_line()

    assert runner._PROGRESS_OUTPUT_ENABLED is False
    assert runner._PROGRESS_LINE_LEN == 0


def test_progress_output_is_skipped_when_stderr_is_redirected(monkeypatch) -> None:
    stderr = RedirectedStderr()
    monkeypatch.setattr(runner, "_PROGRESS_OUTPUT_ENABLED", True)
    monkeypatch.setattr(runner, "_PROGRESS_LINE_LEN", 0)
    monkeypatch.setattr(runner.sys, "stderr", stderr)

    runner.write_progress_line("frame 1")

    assert stderr.writes == 0
    assert runner._PROGRESS_OUTPUT_ENABLED is True


def test_run_header_includes_input_and_protocol_details() -> None:
    lines = experiment_runner._run_header_lines(
        {
            "run_id": "phase1__001",
            "block": "amplitude_grid_preprocessing",
            "runtime_device": "cuda:0",
            "input": "C:/videos/SANPOgt25min.mp4",
            "coords_yaml": "C:/arrays/coords_800um.yaml",
            "output_directory": "C:/results/phase1__001",
            "preprocessing_method": "groundtruth",
            "amplitude_uA": 60.0,
            "appearance_threshold_uA": 30.0,
            "pulse_width_us": 170.0,
            "frequency_hz": 300.0,
            "internal_circuit_power_mw": 0.0,
            "track_electrical": True,
            "electrode_heat_enabled": True,
            "strategy": {"name": "direct", "import_path": None},
            "resolved_config": {
                "input": {"stage": "preprocessed"},
                "protocol": {"relative_stim_duration": 1.0},
            },
        }
    )

    output = "\n".join(lines)
    assert "Input | C:/videos/SANPOgt25min.mp4" in output
    assert "stage=preprocessed | preprocessing=groundtruth" in output
    assert "amplitude=60 uA" in output
    assert "threshold=30 uA" in output
    assert "pulse_width=170 us" in output
    assert "frequency=300 Hz" in output
    assert "electrodes=coords_800um" in output
    assert "strategy=direct" in output
