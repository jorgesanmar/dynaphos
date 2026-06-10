from __future__ import annotations

import copy
import hashlib
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

import torch
import yaml
import cv2

from dynaphos.config import (
    ExperimentConfig,
    SweepConfig,
    load_experiment,
    load_sweep,
    load_yaml,
    set_config_value,
)
from dynaphos.experiment.results import ExperimentResult
from dynaphos.experiment.manifest import (
    SCHEMA_VERSION,
    complete_manifest,
    fail_manifest,
    start_manifest,
)
from dynaphos.experiment.manifest import load_manifest as load_run_manifest
from dynaphos.paths import package_file
from dynaphos.reporting import write_report_bundle
from dynaphos.experiment import execution
from dynaphos.strategies import load_strategy


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("dynaphos")
    except Exception:
        return "0+unknown"


def _git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_config(config: ExperimentConfig | str | Path) -> ExperimentConfig:
    if isinstance(config, ExperimentConfig):
        return copy.deepcopy(config)
    return load_experiment(config)


def _prepare_run_directory(config: ExperimentConfig) -> Path:
    run_dir = (config.output.root / config.output.run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not config.output.overwrite:
        raise FileExistsError(
            f"Run directory already contains outputs: {run_dir}. "
            "Set output.overwrite: true or choose another run_id."
        )
    if run_dir.exists() and config.output.overwrite:
        known_outputs = {
            "figures",
            "manifest.yaml",
            "metrics.npz",
            "report.html",
            "report.json",
            "summary.csv",
            "device_power_over_time.png",
            "thermal_response_over_time.png",
        }
        for path in run_dir.iterdir():
            if path.name not in known_outputs:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_electrode_path(config: ExperimentConfig) -> Path:
    if config.electrode_array.coordinates is not None:
        return config.electrode_array.coordinates.resolve()
    return package_file(str(config.electrode_array.builtin))


def _resolve_input_path(config: ExperimentConfig) -> Path:
    if config.input.path is not None:
        return config.input.path.resolve()
    return package_file(f"fixture_{config.input.builtin}")


def _video_input(path: Path, *, fps: float):
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        from contextlib import nullcontext

        return nullcontext(path)

    class _TemporaryVideo:
        def __init__(self, image_path: Path, frame_rate: float):
            self.image_path = image_path
            self.frame_rate = frame_rate
            self.tempdir = None

        def __enter__(self):
            self.tempdir = tempfile.TemporaryDirectory(prefix="dynaphos-input-")
            target = Path(self.tempdir.name) / "input.avi"
            frame = cv2.imread(str(self.image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Unable to read image input: {self.image_path}")
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(target),
                cv2.VideoWriter_fourcc(*"MJPG"),
                float(self.frame_rate),
                (width, height),
                True,
            )
            if not writer.isOpened():
                raise RuntimeError("Unable to create temporary video for image input.")
            writer.write(frame)
            writer.release()
            return target

        def __exit__(self, exc_type, exc, traceback):
            self.tempdir.cleanup()
            return False

    return _TemporaryVideo(path, fps)


def _build_params(config: ExperimentConfig, safety_path: Path) -> tuple[dict[str, Any], Path]:
    params_path = (
        package_file("params")
        if config.simulation.params is None
        else config.simulation.params.resolve()
    )
    params = load_yaml(params_path)
    params.setdefault("run", {})
    params["run"]["seed"] = int(config.simulation.seed)
    if config.simulation.resolution is not None:
        size = int(config.simulation.resolution)
        params["run"]["resolution"] = [size, size]
    params.setdefault("sampling", {})["stimulus_scale"] = (
        float(config.protocol.amplitude_uA) * 1e-6
    )
    params.setdefault("default_stim", {})
    params["default_stim"]["pw_default"] = float(config.protocol.pulse_width_us) * 1e-6
    params["default_stim"]["freq_default"] = float(config.protocol.frequency_hz)
    params["default_stim"]["relative_stim_duration"] = float(
        config.protocol.relative_stim_duration
    )
    params.setdefault("safety", {})["guidelines_path"] = str(safety_path)
    params.setdefault("bioheat", {})["device_constant_power_mw"] = float(
        config.protocol.internal_circuit_power_mW
    )
    params["bioheat"]["internal_circuit_power_mw"] = float(
        config.protocol.internal_circuit_power_mW
    )
    return params, params_path


def _manifest(
    config: ExperimentConfig,
    *,
    run_dir: Path,
    params_path: Path,
    safety_path: Path,
    coords_path: Path,
    input_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    source_root = (
        config.source_path.parent
        if config.source_path is not None
        else Path.cwd()
    )
    strategy_name = str(config.strategy.name or config.strategy.import_path or "custom")
    normalized_raster_mode = {
        "direct": "none",
        "pseudo_random": "random",
    }.get(strategy_name, strategy_name)
    metadata = dict(config.metadata)
    block = str(metadata.get("experiment_block", metadata.get("block", "")))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": config.output.run_id,
        "dynaphos_version": _version(),
        "git_revision": _git_revision(source_root),
        "runtime_device": str(device),
        "random_seed": int(config.simulation.seed),
        "input": str(input_path),
        "input_sha256": _sha256(input_path),
        "params": str(params_path),
        "safety_limits": str(safety_path),
        "coords_yaml": str(coords_path),
        "output_directory": str(run_dir),
        "video": str(input_path),
        "preprocessing_method": config.input.preprocessing_method,
        "source_input_label": config.input.preprocessing_method,
        "amplitude_uA": float(config.protocol.amplitude_uA),
        "appearance_threshold_uA": float(config.protocol.appearance_threshold_uA),
        "pulse_width_us": float(config.protocol.pulse_width_us),
        "frequency_hz": float(config.protocol.frequency_hz),
        "internal_circuit_power_mw": float(config.protocol.internal_circuit_power_mW),
        "raster_mode": strategy_name,
        "raster_mode_normalized": normalized_raster_mode,
        "raster_groups": int(config.strategy.options.get("groups", 1)),
        "electrode_heat_enabled": bool(config.safety.electrode_heat_enabled),
        "track_electrical": bool(config.safety.track_electrical),
        "block": block,
        "metadata": metadata,
        "strategy": {
            "name": config.strategy.name,
            "import_path": config.strategy.import_path,
            "options": dict(config.strategy.options),
        },
        "resolved_config": config.to_dict(),
    }


def _collect_figures(run_dir: Path) -> None:
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    for path in run_dir.glob("*.png"):
        target = figures_dir / path.name
        if target.exists():
            target.unlink()
        path.replace(target)


def run_experiment(config: ExperimentConfig | str | Path) -> ExperimentResult:
    resolved = _resolve_config(config)
    input_path = _resolve_input_path(resolved)
    if not input_path.exists():
        raise FileNotFoundError(f"Experiment input does not exist: {input_path}")
    run_dir = _prepare_run_directory(resolved)
    coords_path = _resolve_electrode_path(resolved)
    safety_path = (
        package_file("safety")
        if resolved.safety.limits is None
        else resolved.safety.limits.resolve()
    )
    params, params_path = _build_params(resolved, safety_path)
    strategy = load_strategy(resolved.strategy)
    device = execution.configure_runtime_device(
        params,
        force_cpu=bool(resolved.simulation.force_cpu),
    )
    manifest = _manifest(
        resolved,
        run_dir=run_dir,
        params_path=params_path,
        safety_path=safety_path,
        coords_path=coords_path,
        input_path=input_path,
        device=device,
    )
    manifest_path = run_dir / "manifest.yaml"
    start_manifest(manifest_path, manifest)

    try:
        # Raster selection belongs to the strategy, so the simulator's
        # internal raster mask stays disabled to avoid applying two masks.
        with _video_input(input_path, fps=float(params["run"]["fps"])) as simulation_input:
            execution.run_one_mode(
                params=params,
                coords_yaml=coords_path,
                video_path=simulation_input,
                mode_out_dirs={"with": run_dir},
                preview_out_dirs={"with": run_dir / "figures"},
                preprocessing_method=resolved.input.preprocessing_method,
                stim_scale=None,
                stimulus_scale_base=float(resolved.protocol.amplitude_uA) * 1e-6,
                raster_name="none",
                groups=1,
                max_frames=int(resolved.simulation.max_frames),
                internal_circuit_power_mw=float(resolved.protocol.internal_circuit_power_mW),
                preview_seconds=float(resolved.simulation.preview_seconds),
                save_every_n_frames=1,
                enable_cem43=bool(resolved.simulation.enable_cem43),
                device=device,
                phosphene_mode=resolved.simulation.phosphene_mode,
                cooldown_seconds=float(resolved.simulation.cooldown_seconds),
                cooldown_baseline_tolerance_C=float(
                    resolved.simulation.cooldown_baseline_tolerance_C
                ),
                appearance_threshold_uA=float(resolved.protocol.appearance_threshold_uA),
                track_electrical=bool(resolved.safety.track_electrical),
                electrode_heat_enabled=bool(resolved.safety.electrode_heat_enabled),
                strategy=strategy,
                input_stage=resolved.input.stage,
                preprocessing_options=resolved.input.preprocessing_options,
            )
    except Exception as exc:
        fail_manifest(
            manifest_path,
            manifest,
            error=exc,
            traceback_text=traceback.format_exc(),
        )
        raise

    metrics_path = run_dir / "metrics.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Experiment did not write canonical metrics: {metrics_path}")
    _collect_figures(run_dir)
    report = write_report_bundle(
        run_dir,
        metrics_path=metrics_path,
        safety_limits_path=safety_path,
        manifest=manifest,
        write_figures=bool(resolved.output.write_figures),
    )
    complete_manifest(
        manifest_path,
        manifest,
        report_status=report.status,
    )
    return ExperimentResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        report_json_path=run_dir / "report.json",
        report_html_path=run_dir / "report.html",
        summary_csv_path=run_dir / "summary.csv",
        report=report,
    )


def _variant_run_id(base_run_id: str, variant: dict[str, Any], index: int) -> str:
    labels = []
    for key, value in variant.items():
        leaf = key.lstrip("$").rsplit(".", 1)[-1]
        if isinstance(value, dict) and "label" in value:
            value = value["label"]
        text = str(value).replace(" ", "_").replace("/", "_").replace("\\", "_")
        labels.append(f"{leaf}-{text}")
    return f"{base_run_id}__{index:03d}__{'__'.join(labels)}"


def _completed_run(run_dir: Path) -> bool:
    manifest_path = run_dir / "manifest.yaml"
    metrics_path = run_dir / "metrics.npz"
    if not manifest_path.exists() or not metrics_path.exists():
        return False
    try:
        return load_run_manifest(manifest_path).get("status") == "completed"
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return False


def run_sweep(
    config: SweepConfig | str | Path,
    *,
    resume: bool = False,
) -> list[ExperimentResult]:
    sweep = config if isinstance(config, SweepConfig) else load_sweep(config)
    source = Path(sweep.experiment).resolve()
    base_values = load_yaml(source)
    results: list[ExperimentResult] = []
    errors: list[Exception] = []
    for index, variant in enumerate(sweep.variants(), start=1):
        values = copy.deepcopy(base_values)
        for dotted_path, value in variant.items():
            if isinstance(value, dict) and "patch" in value:
                patch = value["patch"]
                if not isinstance(patch, dict):
                    raise ValueError(f"Sweep patch for {dotted_path} must be a mapping.")
                for patch_path, patch_value in patch.items():
                    set_config_value(values, patch_path, patch_value)
                continue
            effective_value = (
                value["value"]
                if isinstance(value, dict) and "value" in value
                else value
            )
            set_config_value(values, dotted_path, effective_value)
        output = values.setdefault("output", {})
        base_run_id = str(output.get("run_id", source.stem))
        output["run_id"] = _variant_run_id(base_run_id, variant, index)
        if sweep.output_root is not None:
            output["root"] = str(sweep.output_root)
        experiment = ExperimentConfig.from_dict(values, path="experiment").resolve_paths(source)
        run_dir = (experiment.output.root / experiment.output.run_id).resolve()
        if resume and _completed_run(run_dir):
            print(f"Skipping completed run: {run_dir}")
            continue
        if resume and run_dir.exists():
            experiment.output.overwrite = True
        try:
            results.append(run_experiment(experiment))
        except Exception as exc:
            errors.append(exc)
            if sweep.stop_on_error:
                raise
    if errors and not results:
        raise RuntimeError(f"All {len(errors)} sweep runs failed.") from errors[0]
    return results
