from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dynaphos.safety.evaluation import SafetyReport


@dataclass(frozen=True)
class ExperimentResult:
    run_dir: Path
    manifest_path: Path
    metrics_path: Path
    report_json_path: Path
    report_html_path: Path
    summary_csv_path: Path
    report: SafetyReport
