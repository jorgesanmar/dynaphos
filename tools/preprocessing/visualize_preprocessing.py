import argparse
import os
import sys
import time
from pathlib import Path
from typing import Set

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dynaphos.pipeline import (
    PREPROCESS_METHODS,
    is_image_file,
    make_panel,
    prepare_square_gray_frame,
    preprocess_gray_frame,
    resolve_media_inputs,
    resolve_repo_path,
)

VISUALIZATION_METHODS = tuple(method for method in PREPROCESS_METHODS if method != "sobel")


def sampled_indices(total_frames: int, n_samples: int) -> Set[int]:
    if total_frames <= 0 or n_samples <= 0:
        return set()
    n_samples = max(1, min(int(n_samples), int(total_frames)))
    return set(np.linspace(0, total_frames - 1, num=n_samples, dtype=int).tolist())


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def compute_preprocessing_frames(
    frame,
    *,
    render_resolution: int,
    methods,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float,
    canny_high: float,
    use_cuda: bool,
):
    original_gray = prepare_square_gray_frame(frame, int(render_resolution))
    processed_by_method = {}

    for method in methods:
        processed_by_method[method] = preprocess_gray_frame(
            original_gray,
            method,
            dog_sigma_low=dog_sigma_low,
            dog_sigma_high=dog_sigma_high,
            canny_low=canny_low,
            canny_high=canny_high,
            use_cuda=use_cuda,
        )

    return original_gray, processed_by_method


def build_preprocessing_panel(
    original_gray: np.ndarray,
    processed_by_method: dict[str, np.ndarray],
    *,
    methods,
):
    images = [original_gray]
    labels = ["original"]
    for method in methods:
        images.append(processed_by_method[method])
        labels.append(method)

    columns = min(3, len(images))
    return make_panel(images, labels, columns=columns)


def maybe_open_file(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            import subprocess

            subprocess.run(["open", str(path)], check=False)
        else:
            import subprocess

            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception as exc:
        print(f"Could not open preview automatically: {exc}")


def resolve_video_output_dir(input_path: Path, export_output_dir: Path) -> Path:
    video_dir = export_output_dir / input_path.stem
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def resolve_method_output_dir(input_path: Path, export_output_dir: Path, method: str) -> Path:
    method_dir = resolve_video_output_dir(input_path, export_output_dir) / str(method).strip().lower()
    method_dir.mkdir(parents=True, exist_ok=True)
    return method_dir


def resolve_panel_output_path(input_path: Path, export_output_dir: Path) -> Path:
    return resolve_video_output_dir(input_path, export_output_dir) / f"{input_path.stem}_preprocessing.mp4"


def resolve_panel_samples_dir(input_path: Path, export_output_dir: Path) -> Path:
    samples_dir = resolve_video_output_dir(input_path, export_output_dir) / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    return samples_dir


def resolve_export_output_path(input_path: Path, export_output_dir: Path, method: str) -> Path:
    _ = export_output_dir
    method_name = str(method).strip().lower()
    return input_path.with_name(f"{input_path.stem}_{method_name}.mp4")


def resolve_comparison_output_path(input_path: Path, export_output_dir: Path, method: str) -> Path:
    _ = export_output_dir
    method_name = str(method).strip().lower()
    return input_path.with_name(f"{input_path.stem}_{method_name}_comparison_with_original.mp4")


def resolve_sample_frames_dir(input_path: Path, export_output_dir: Path, method: str) -> Path:
    _ = export_output_dir
    method_name = str(method).strip().lower()
    frames_dir = input_path.with_name(f"{input_path.stem}_{method_name}_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    return frames_dir


def resolve_preview_paths(output_path: Path) -> tuple[Path, Path]:
    return (
        output_path.with_name(f"{output_path.stem}_preview{output_path.suffix}"),
        output_path.with_name(f"{output_path.stem}_preview_comparison{output_path.suffix}"),
    )


def prompt_continue_after_preview() -> bool:
    try:
        response = input("Preview ready. Continue with full preprocessing? [y/N]: ")
    except EOFError:
        return False
    return response.strip().lower() == "y"


def render_video_preview(
    video_path: Path,
    *,
    render_resolution: int,
    methods,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float,
    canny_high: float,
    use_cuda: bool,
    export_methods,
    export_output_dir: Path,
    preview_seconds: float,
    preview_start_seconds: float,
) -> list[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    preview_start_frame = max(0, int(round(float(preview_start_seconds) * fps)))
    preview_frame_count = max(0, int(round(float(preview_seconds) * fps)))
    if preview_frame_count <= 0:
        cap.release()
        return []

    if preview_start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, preview_start_frame)

    preview_writers = {}
    comparison_preview_writers = {}
    preview_comparison_paths = {}
    all_methods = list(dict.fromkeys([*methods, *export_methods]))
    written_frames = 0
    start_time = time.perf_counter()
    print(
        f"rendering {format_duration(float(preview_seconds))} preview for {video_path.name} "
        f"from {format_duration(float(preview_start_seconds))}"
    )
    try:
        while written_frames < preview_frame_count:
            ok, frame = cap.read()
            if not ok:
                break

            original_gray, processed_by_method = compute_preprocessing_frames(
                frame,
                render_resolution=render_resolution,
                methods=all_methods,
                dog_sigma_low=dog_sigma_low,
                dog_sigma_high=dog_sigma_high,
                canny_low=canny_low,
                canny_high=canny_high,
                use_cuda=use_cuda,
            )
            original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
            for method in export_methods:
                processed = processed_by_method[method]
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                comparison = np.hstack([original_bgr, processed_bgr])
                if method not in preview_writers:
                    output_path = resolve_export_output_path(video_path, export_output_dir, method)
                    preview_output_path, comparison_output_path = resolve_preview_paths(output_path)
                    preview_writers[method] = cv2.VideoWriter(
                        str(preview_output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (processed.shape[1], processed.shape[0]),
                        False,
                    )
                    comparison_preview_writers[method] = cv2.VideoWriter(
                        str(comparison_output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (comparison.shape[1], comparison.shape[0]),
                        True,
                    )
                    preview_comparison_paths[method] = comparison_output_path
                preview_writers[method].write(processed)
                comparison_preview_writers[method].write(comparison)

            written_frames += 1
    finally:
        cap.release()
        for preview_writer in preview_writers.values():
            preview_writer.release()
        for comparison_writer in comparison_preview_writers.values():
            comparison_writer.release()

    if written_frames == 0:
        raise RuntimeError(f"No preview frames were processed from: {video_path}")

    print(
        f"saved preview for {video_path.name}: {written_frames} frame(s) in "
        f"{format_duration(time.perf_counter() - start_time)}"
    )
    return [
        preview_comparison_paths[method]
        for method in export_methods
        if method in preview_comparison_paths
    ]


def process_image(
    image_path: Path,
    output_dir: Path,
    *,
    render_resolution: int,
    methods,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float,
    canny_high: float,
    use_cuda: bool,
) -> None:
    start_time = time.perf_counter()
    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Unable to load image: {image_path}")

    original_gray, processed_by_method = compute_preprocessing_frames(
        frame,
        render_resolution=render_resolution,
        methods=methods,
        dog_sigma_low=dog_sigma_low,
        dog_sigma_high=dog_sigma_high,
        canny_low=canny_low,
        canny_high=canny_high,
        use_cuda=use_cuda,
    )
    panel = build_preprocessing_panel(
        original_gray,
        processed_by_method,
        methods=methods,
    )
    out_path = output_dir / f"{image_path.stem}_preprocessing.png"
    cv2.imwrite(str(out_path), panel)
    print(f"saved preprocessing panel to: {out_path}")
    print(f"processed image in: {format_duration(time.perf_counter() - start_time)}")


def process_video(
    video_path: Path,
    *,
    render_resolution: int,
    methods,
    dog_sigma_low: float,
    dog_sigma_high: float,
    canny_low: float,
    canny_high: float,
    use_cuda: bool,
    max_frames: int,
    sample_frames: int,
    save_preprocessed_videos: bool,
    export_methods,
    export_output_dir: Path,
    preview_seconds: float,
    preview_start_seconds: float,
    preview_only: bool,
    open_preview: bool,
    confirm_preview: bool,
) -> None:
    if preview_only:
        preview_paths = render_video_preview(
            video_path,
            render_resolution=render_resolution,
            methods=methods,
            dog_sigma_low=dog_sigma_low,
            dog_sigma_high=dog_sigma_high,
            canny_low=canny_low,
            canny_high=canny_high,
            use_cuda=use_cuda,
            export_methods=export_methods,
            export_output_dir=export_output_dir,
            preview_seconds=preview_seconds,
            preview_start_seconds=preview_start_seconds,
        )
        if open_preview and preview_paths:
            maybe_open_file(preview_paths[0])
        return

    if confirm_preview and float(preview_seconds) > 0:
        preview_paths = render_video_preview(
            video_path,
            render_resolution=render_resolution,
            methods=methods,
            dog_sigma_low=dog_sigma_low,
            dog_sigma_high=dog_sigma_high,
            canny_low=canny_low,
            canny_high=canny_high,
            use_cuda=use_cuda,
            export_methods=export_methods,
            export_output_dir=export_output_dir,
            preview_seconds=preview_seconds,
            preview_start_seconds=preview_start_seconds,
        )
        if open_preview and preview_paths:
            maybe_open_file(preview_paths[0])
        if not prompt_continue_after_preview():
            print(f"skipped full preprocessing for {video_path.name}")
            return
        preview_seconds = 0.0

    process_start_time = time.perf_counter()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0 and total_frames > 0:
        total_frames = min(total_frames, max_frames)
    expected_total_frames = total_frames if total_frames > 0 else (max_frames if max_frames > 0 else None)
    sample_idxs = sampled_indices(total_frames, sample_frames)
    preview_start_frame = max(0, int(round(float(preview_start_seconds) * fps)))
    preview_frame_count = max(0, int(round(float(preview_seconds) * fps)))
    preview_end_frame = preview_start_frame + preview_frame_count

    panel_video_path = resolve_panel_output_path(video_path, export_output_dir)
    samples_dir = resolve_panel_samples_dir(video_path, export_output_dir)

    writer = None
    export_writers = {}
    comparison_writers = {}
    preview_writers = {}
    comparison_preview_writers = {}
    preview_comparison_paths = {}
    frame_idx = 0
    last_progress_report_time = process_start_time
    print(
        f"processing video: {video_path.name} | "
        f"target frames: {expected_total_frames if expected_total_frames is not None else 'unknown'}"
    )
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if max_frames > 0 and frame_idx >= max_frames:
                break

            all_methods = list(dict.fromkeys([*methods, *export_methods]))
            original_gray, processed_by_method = compute_preprocessing_frames(
                frame,
                render_resolution=render_resolution,
                methods=all_methods,
                dog_sigma_low=dog_sigma_low,
                dog_sigma_high=dog_sigma_high,
                canny_low=canny_low,
                canny_high=canny_high,
                use_cuda=use_cuda,
            )
            panel = build_preprocessing_panel(
                original_gray,
                processed_by_method,
                methods=methods,
            )

            if writer is None:
                height, width = panel.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(panel_video_path), fourcc, fps, (width, height), True)

            writer.write(panel)
            if frame_idx in sample_idxs:
                cv2.imwrite(str(samples_dir / f"frame_{frame_idx:06d}.png"), panel)

            original_bgr = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
            for method in export_methods:
                processed = processed_by_method[method]
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                comparison = np.hstack([original_bgr, processed_bgr])

                if frame_idx in sample_idxs:
                    sample_frames_dir = resolve_sample_frames_dir(video_path, export_output_dir, method)
                    cv2.imwrite(str(sample_frames_dir / f"frame_{frame_idx:06d}.png"), processed)

                if save_preprocessed_videos and not preview_only:
                    if method not in export_writers:
                        output_path = resolve_export_output_path(video_path, export_output_dir, method)
                        comparison_output_path = resolve_comparison_output_path(video_path, export_output_dir, method)
                        export_writers[method] = cv2.VideoWriter(
                            str(output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (processed.shape[1], processed.shape[0]),
                            False,
                        )
                        comparison_writers[method] = cv2.VideoWriter(
                            str(comparison_output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (comparison.shape[1], comparison.shape[0]),
                            True,
                        )
                    export_writers[method].write(processed)
                    comparison_writers[method].write(comparison)

            if preview_frame_count > 0 and preview_start_frame <= frame_idx < preview_end_frame:
                for method in export_methods:
                    processed = processed_by_method[method]
                    processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    comparison = np.hstack([original_bgr, processed_bgr])
                    if method not in preview_writers:
                        output_path = resolve_export_output_path(video_path, export_output_dir, method)
                        preview_output_path, comparison_output_path = resolve_preview_paths(output_path)
                        preview_writers[method] = cv2.VideoWriter(
                            str(preview_output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (processed.shape[1], processed.shape[0]),
                            False,
                        )
                        comparison_preview_writers[method] = cv2.VideoWriter(
                            str(comparison_output_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (comparison.shape[1], comparison.shape[0]),
                            True,
                        )
                        preview_comparison_paths[method] = comparison_output_path
                    preview_writers[method].write(processed)
                    comparison_preview_writers[method].write(comparison)

            frame_idx += 1
            now = time.perf_counter()
            should_report_progress = (
                frame_idx == 1
                or now - last_progress_report_time >= 1.0
                or (expected_total_frames is not None and frame_idx >= expected_total_frames)
            )
            if should_report_progress:
                elapsed = now - process_start_time
                frames_per_second = frame_idx / elapsed if elapsed > 0 else 0.0
                if expected_total_frames is not None:
                    progress_pct = 100.0 * frame_idx / max(expected_total_frames, 1)
                    remaining_frames = max(expected_total_frames - frame_idx, 0)
                    eta_seconds = remaining_frames / frames_per_second if frames_per_second > 0 else 0.0
                    print(
                        f"[{video_path.name}] {frame_idx}/{expected_total_frames} frames "
                        f"({progress_pct:5.1f}%) | {frames_per_second:5.1f} fps | "
                        f"elapsed {format_duration(elapsed)} | eta {format_duration(eta_seconds)}"
                    )
                else:
                    print(
                        f"[{video_path.name}] {frame_idx} frames | "
                        f"{frames_per_second:5.1f} fps | elapsed {format_duration(elapsed)}"
                    )
                last_progress_report_time = now
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        for export_writer in export_writers.values():
            export_writer.release()
        for comparison_writer in comparison_writers.values():
            comparison_writer.release()
        for preview_writer in preview_writers.values():
            preview_writer.release()
        for comparison_writer in comparison_preview_writers.values():
            comparison_writer.release()

    if frame_idx == 0:
        raise RuntimeError(f"No frames were processed from: {video_path}")

    total_elapsed = time.perf_counter() - process_start_time
    print(f"saved preprocessing video to: {panel_video_path}")
    if sample_idxs:
        print(f"saved sample panels to: {samples_dir}")
    for method in export_methods:
        output_path = resolve_export_output_path(video_path, export_output_dir, method)
        comparison_output_path = resolve_comparison_output_path(video_path, export_output_dir, method)
        sample_frames_dir = resolve_sample_frames_dir(video_path, export_output_dir, method)
        if sample_idxs:
            print(f"saved {method} sample frames to: {sample_frames_dir}")
        if method in export_writers:
            print(f"saved preprocessed video to: {output_path}")
            print(f"saved comparison video to: {comparison_output_path}")
        if method in preview_writers:
            preview_output_path, comparison_output_path = resolve_preview_paths(output_path)
            print(f"saved preview video to: {preview_output_path}")
            print(f"saved comparison preview to: {comparison_output_path}")

    if open_preview and preview_comparison_paths:
        first_method = export_methods[0]
        if first_method in preview_comparison_paths:
            maybe_open_file(preview_comparison_paths[first_method])

    print(
        f"finished {video_path.name}: processed {frame_idx} frame(s) in "
        f"{format_duration(total_elapsed)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize preprocessing methods on an image or video, and optionally export preprocessed videos with preview clips."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="videos",
        help="Input image, video, or directory containing media files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/preprocessing/visualize_preprocessing",
        help="Directory where visualization outputs will be saved.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["none", "dog", "canny"],
        choices=list(VISUALIZATION_METHODS),
        help="Preprocessing methods to visualize alongside the original frame.",
    )
    parser.add_argument("--render-resolution", type=int, default=256)
    parser.add_argument("--dog-sigma-low", type=float, default=2.0)
    parser.add_argument("--dog-sigma-high", type=float, default=6.0)
    parser.add_argument("--canny-low", type=float, default=30.0)
    parser.add_argument("--canny-high", type=float, default=80.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the full video.")
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=0,
        help="Number of per-method sample frames to save from video inputs.",
    )
    parser.set_defaults(save_preprocessed_videos=True)
    parser.add_argument(
        "--save-preprocessed-videos",
        dest="save_preprocessed_videos",
        action="store_true",
        help="Export standalone preprocessed videos for the selected export methods.",
    )
    parser.add_argument(
        "--skip-preprocessed-videos",
        dest="save_preprocessed_videos",
        action="store_false",
        help="Skip standalone preprocessed video and comparison exports.",
    )
    parser.add_argument(
        "--export-methods",
        nargs="+",
        default=None,
        choices=list(VISUALIZATION_METHODS),
        help="Methods to export as standalone preprocessed videos. Defaults to the same methods used in the visualization panel.",
    )
    parser.add_argument(
        "--preprocessed-output-dir",
        type=str,
        default="videos/preprocessed",
        help=(
            "Deprecated. Standalone preprocessed video outputs are saved next to each input "
            "video as <video_name>_<method>.mp4."
        ),
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=30.0,
        help="If > 0, save short per-method preview clips for the exported preprocessing methods.",
    )
    parser.add_argument(
        "--preview-start-seconds",
        type=float,
        default=0.0,
        help="Start time in seconds for the preview snippet.",
    )
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="Only write preview clips for exported methods and skip the full standalone preprocessed video exports.",
    )
    parser.add_argument(
        "--open-preview",
        action="store_true",
        default=True,
        help="Open the first comparison preview automatically after it is written.",
    )
    parser.add_argument(
        "--no-open-preview",
        dest="open_preview",
        action="store_false",
        help="Do not open the comparison preview automatically.",
    )
    parser.add_argument(
        "--confirm-preview",
        dest="confirm_preview",
        action="store_true",
        default=True,
        help="Ask for confirmation after the preview before processing the full video.",
    )
    parser.add_argument(
        "--skip-preview-confirm",
        dest="confirm_preview",
        action="store_false",
        help="Write previews without pausing for confirmation before full processing.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use OpenCV CUDA for DoG preprocessing when available.",
    )
    args = parser.parse_args()

    use_cuda = bool(
        args.use_cuda
        and hasattr(cv2, "cuda")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
    )
    if args.use_cuda and not use_cuda:
        print("CUDA was requested but is not available. Falling back to CPU.")

    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_output_dir = resolve_repo_path(args.preprocessed_output_dir)
    export_methods = args.export_methods or args.methods

    if args.preview_only and float(args.preview_seconds) <= 0:
        raise ValueError("--preview-only requires --preview-seconds > 0.")

    for media_path in resolve_media_inputs(args.input):
        if is_image_file(media_path):
            process_image(
                media_path,
                output_dir,
                render_resolution=int(args.render_resolution),
                methods=args.methods,
                dog_sigma_low=float(args.dog_sigma_low),
                dog_sigma_high=float(args.dog_sigma_high),
                canny_low=float(args.canny_low),
                canny_high=float(args.canny_high),
                use_cuda=use_cuda,
            )
            if args.save_preprocessed_videos or float(args.preview_seconds) > 0:
                print(f"standalone preprocessed video export is skipped for image input: {media_path.name}")
        else:
            process_video(
                media_path,
                render_resolution=int(args.render_resolution),
                methods=args.methods,
                dog_sigma_low=float(args.dog_sigma_low),
                dog_sigma_high=float(args.dog_sigma_high),
                canny_low=float(args.canny_low),
                canny_high=float(args.canny_high),
                use_cuda=use_cuda,
                max_frames=int(args.max_frames),
                sample_frames=int(args.sample_frames),
                save_preprocessed_videos=bool(args.save_preprocessed_videos),
                export_methods=export_methods,
                export_output_dir=export_output_dir,
                preview_seconds=float(args.preview_seconds),
                preview_start_seconds=float(args.preview_start_seconds),
                preview_only=bool(args.preview_only),
                open_preview=bool(args.open_preview),
                confirm_preview=bool(args.confirm_preview),
            )


if __name__ == "__main__":
    main()
