"""End-to-end pipeline benchmark for a real video.

Drives the full DynaPhos pipeline (decode -> preprocess -> phosphene sim ->
electrical safety -> bioheat -> metrics serialization -> report) via the same
``run_experiment`` entry point the CLI uses, and times it.

Two modes:

  * default  - run the whole video (or ``--frames N``) once and report total
               wall-clock, end-to-end frames/s, peak dT, and safety status.
  * calibrate - run two short prefixes (N and 2N frames) and use the
               subtraction method to split fixed overhead (model build +
               report) from true per-frame cost, then project the full video
               and a 20-minute clip precisely.

Examples
--------
    # Quick projection without processing the whole clip (recommended first):
    python tools/benchmark_pipeline.py --video path/to/clip.mp4 --calibrate 120

    # Full run of the entire video on GPU at full resolution:
    python tools/benchmark_pipeline.py --video path/to/clip.mp4 --device cuda

    # Match a specific production setup:
    python tools/benchmark_pipeline.py --video clip.mp4 --electrodes coords_400um \
        --strategy checkerboard --groups 4 --resolution 256 --cem43 --figures
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script (python tools/benchmark_pipeline.py) by
# putting the repo root (which contains the dynaphos package) on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from dynaphos.config import ExperimentConfig
from dynaphos.experiment import run_experiment


def video_metadata(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    cap.release()
    return frames, fps


def build_config(args, run_id: str, max_frames: int) -> ExperimentConfig:
    strategy = (
        {"name": args.strategy, "options": {"groups": int(args.groups)}}
        if args.strategy != "direct"
        else {"name": "direct"}
    )
    preprocessing_options = {}
    if args.dog_sigma_low is not None:
        preprocessing_options["dog_sigma_low"] = float(args.dog_sigma_low)
    if args.dog_sigma_high is not None:
        preprocessing_options["dog_sigma_high"] = float(args.dog_sigma_high)
    if args.canny_low is not None:
        preprocessing_options["canny_low"] = float(args.canny_low)
    if args.canny_high is not None:
        preprocessing_options["canny_high"] = float(args.canny_high)
    values = {
        "input": {
            "path": str(Path(args.video).resolve()),
            "stage": args.stage,
            "preprocessing_method": args.preprocessing,
            "preprocessing_options": preprocessing_options,
        },
        "electrode_array": {"builtin": args.electrodes},
        "protocol": {
            "amplitude_uA": float(args.amplitude),
            "appearance_threshold_uA": float(args.appearance_threshold),
            "pulse_width_us": float(args.pulse_width_us),
            "frequency_hz": float(args.frequency_hz),
            "internal_circuit_power_mW": float(args.ic_power_mw),
        },
        "strategy": strategy,
        "simulation": {
            "force_cpu": args.device == "cpu",
            "resolution": int(args.resolution),
            "max_frames": int(max_frames),
            "preview_seconds": 0.0,
            "phosphene_mode": args.phosphene_mode,
            "cooldown_seconds": float(args.cooldown),
            "enable_cem43": bool(args.cem43),
        },
        "safety": {},
        "output": {
            "root": str(Path(args.out_root).resolve()),
            "run_id": run_id,
            "overwrite": True,
            "write_figures": bool(args.figures),
        },
    }
    return ExperimentConfig.from_dict(values, path="experiment")


def timed_run(cfg: ExperimentConfig):
    t0 = time.perf_counter()
    result = run_experiment(cfg)
    return time.perf_counter() - t0, result


def peak_dT(metrics_path: Path) -> float | None:
    try:
        with np.load(metrics_path, allow_pickle=True) as d:
            if "max_dT" in d.files and d["max_dT"].size:
                return float(np.nanmax(d["max_dT"]))
    except Exception:
        pass
    return None


def fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60.0)
    return f"{seconds:.1f}s ({int(m)}m{s:04.1f}s)" if seconds >= 60 else f"{seconds:.2f}s"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="Path to the input video.")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto",
                   help="auto/cuda use the GPU if available; cpu forces CPU.")
    p.add_argument("--frames", type=int, default=0, help="Limit processed frames (0 = whole video).")
    p.add_argument("--calibrate", type=int, default=0, metavar="N",
                   help="Run N and 2N frame prefixes to project precisely instead of a full run.")
    p.add_argument("--target-minutes", type=float, default=20.0,
                   help="Clip length to project the cost for (default 20).")
    # representative-pipeline knobs (faithful defaults; override to match a real run)
    p.add_argument("--electrodes", default="coords_800um")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--stage", choices=["original", "preprocessed"], default="original")
    p.add_argument("--preprocessing", default="dog")
    p.add_argument("--strategy", default="checkerboard")
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--phosphene-mode", choices=["safety_centers", "visual"], default="safety_centers")
    p.add_argument("--amplitude", type=float, default=60.0)
    p.add_argument("--appearance-threshold", type=float, default=30.0)
    p.add_argument("--pulse-width-us", type=float, default=170.0)
    p.add_argument("--frequency-hz", type=float, default=300.0)
    p.add_argument("--ic-power-mw", type=float, default=0.0)
    p.add_argument("--dog-sigma-low", type=float, default=None, help="DoG narrow kernel sigma.")
    p.add_argument("--dog-sigma-high", type=float, default=None, help="DoG wide kernel sigma.")
    p.add_argument("--canny-low", type=float, default=None, help="Canny low hysteresis threshold.")
    p.add_argument("--canny-high", type=float, default=None, help="Canny high hysteresis threshold.")
    p.add_argument("--cooldown", type=float, default=0.0, help="Post-stim cooldown seconds (0 = none).")
    p.add_argument("--prof", action="store_true",
                   help="Enable DYNAPHOS_PROF per-segment stage timing (sets the env var).")
    p.add_argument("--cem43", action="store_true", help="Enable CEM43 thermal-dose accumulation.")
    p.add_argument("--figures", action="store_true", help="Write report figures (adds one-time cost).")
    p.add_argument("--out-root", default="results", help="Output root directory.")
    args = p.parse_args()

    if args.prof:
        import os
        os.environ["DYNAPHOS_PROF"] = "1"

    video = Path(args.video).resolve()
    if not video.exists():
        raise SystemExit(f"Video not found: {video}")
    total_frames, src_fps = video_metadata(video)
    duration_s = total_frames / src_fps if src_fps > 0 else float("nan")

    print("=" * 70)
    print("DynaPhos full-pipeline benchmark")
    print("=" * 70)
    print(f"  video        : {video.name}")
    print(f"  frames/fps   : {total_frames:,} @ {src_fps:.2f} fps  (~{duration_s:.1f}s)")
    print(f"  device       : {args.device}")
    print(f"  electrodes   : {args.electrodes}   resolution={args.resolution}px")
    print(f"  strategy     : {args.strategy}(groups={args.groups})   phosphene={args.phosphene_mode}")
    print(f"  cem43={args.cem43}  figures={args.figures}  cooldown={args.cooldown}s")
    print("-" * 70)

    if args.calibrate > 0:
        n = int(args.calibrate)
        if total_frames and 2 * n > total_frames:
            raise SystemExit(f"--calibrate {n} needs >= {2*n} frames; video has {total_frames}.")
        print(f"Calibrating with {n} and {2*n} frame prefixes (subtraction method)...")
        warm = max(8, min(n, 16))
        print(f"  warm-up ({warm} frames, discarded: torch kernels, codec, file cache)...")
        timed_run(build_config(args, "bench_calib_warm", warm))
        t1, _ = timed_run(build_config(args, "bench_calib_n", n))
        print(f"  {n:5d} frames: {fmt_time(t1)}")
        t2, res2 = timed_run(build_config(args, "bench_calib_2n", 2 * n))
        print(f"  {2*n:5d} frames: {fmt_time(t2)}")
        marginal = (t2 - t1) / n
        fixed = t1 - marginal * n
        if marginal <= 0:
            print("-" * 70)
            print("  WARNING: non-positive marginal cost — measurement noise dominated "
                  "the\n           subtraction. Re-run with a larger --calibrate value "
                  "(e.g. 2-4x).")
            return
        print("-" * 70)
        print(f"  per-frame (marginal) : {marginal*1e3:.2f} ms/frame  ->  {1/marginal:.1f} frames/s")
        print(f"  fixed overhead       : {fixed:.2f}s  (model build + report)")
        dT = peak_dT(res2.metrics_path)
        if dT is not None:
            print(f"  peak dT (2N run)     : {dT:.3f} degC")

        def project(nframes: int) -> float:
            return fixed + marginal * nframes

        print("-" * 70)
        print("Projections:")
        if total_frames:
            print(f"  whole video ({total_frames:,} frames): {fmt_time(project(total_frames))}")
        for fps_label, fps in (("source", src_fps), ("15 (thermal dt)", 15.0), ("30", 30.0)):
            if fps and fps > 0:
                nf = int(args.target_minutes * 60 * fps)
                print(f"  {args.target_minutes:.0f} min @ {fps_label:>14} fps "
                      f"({nf:,} frames): {fmt_time(project(nf))}")
        return

    # Single full / limited run
    nframes = int(args.frames) if args.frames > 0 else 0
    label = f"{nframes} frames" if nframes else f"whole video ({total_frames:,} frames)"
    print(f"Running {label}...")
    wall, res = timed_run(build_config(args, "bench_full", nframes))
    processed = nframes if nframes else total_frames
    per = wall / max(processed, 1)
    dT = peak_dT(res.metrics_path)
    print("-" * 70)
    print(f"  total wall-clock     : {fmt_time(wall)}")
    print(f"  frames processed     : {processed:,}")
    print(f"  end-to-end per-frame : {per*1e3:.2f} ms/frame  ->  {1/per:.1f} frames/s")
    if dT is not None:
        print(f"  peak dT              : {dT:.3f} degC")
    print(f"  safety status        : {res.report.status_label}")
    print(f"  run dir              : {res.run_dir}")
    if nframes and total_frames and nframes < total_frames:
        print("-" * 70)
        print(f"  NOTE: processed {nframes}/{total_frames} frames; "
              f"naive full-video estimate ~ {fmt_time(per*total_frames)} "
              "(use --calibrate for a fixed/marginal split).")


if __name__ == "__main__":
    main()
