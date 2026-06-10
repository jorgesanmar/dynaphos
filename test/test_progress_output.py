from __future__ import annotations

from dynaphos.experiment import execution as runner


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
