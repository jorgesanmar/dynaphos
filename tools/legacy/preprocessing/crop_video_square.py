from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg"}


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def is_video_file(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def center_crop_square(frame):
    height, width = frame.shape[:2]
    side = min(height, width)
    start_y = (height - side) // 2
    start_x = (width - side) // 2
    return frame[start_y:start_y + side, start_x:start_x + side]


def open_video_writer(path: Path, fps: float, size: tuple[int, int], is_color: bool = True) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, float(fps), tuple(int(v) for v in size), bool(is_color))


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def default_output_path(input_path: Path, output_dir: Path | None) -> Path:
    output_name = f"{input_path.stem}_square{input_path.suffix}"
    if output_dir is None:
        return input_path.with_name(output_name)
    return output_dir / output_name


def crop_video_to_square(
    input_path: Path,
    output_path: Path,
    *,
    output_size: int | None,
    max_frames: int,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0

    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_width <= 0 or source_height <= 0:
        cap.release()
        raise RuntimeError(f"Unable to read video dimensions: {input_path}")

    square_side = min(source_width, source_height)
    target_side = int(output_size) if output_size and output_size > 0 else square_side
    writer = open_video_writer(output_path, fps, (target_side, target_side), is_color=True)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open output video writer: {output_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = int(max_frames) if max_frames and max_frames > 0 else None
    if frame_limit is not None and total_frames > 0:
        total_frames = min(total_frames, frame_limit)

    start_time = time.perf_counter()
    frames_written = 0
    try:
        while True:
            if frame_limit is not None and frames_written >= frame_limit:
                break

            ok, frame = cap.read()
            if not ok:
                break

            square = center_crop_square(frame)
            if square.shape[0] != target_side or square.shape[1] != target_side:
                interpolation = cv2.INTER_AREA if square.shape[0] > target_side else cv2.INTER_LINEAR
                square = cv2.resize(square, (target_side, target_side), interpolation=interpolation)

            writer.write(square)
            frames_written += 1

            if frames_written % 100 == 0:
                progress = f"/{total_frames}" if total_frames > 0 else ""
                print(f"processed {frames_written}{progress} frames")
    finally:
        cap.release()
        writer.release()

    elapsed = format_duration(time.perf_counter() - start_time)
    print(f"saved square video to: {output_path}")
    print(f"frames written: {frames_written}")
    print(f"output size: {target_side}x{target_side}")
    print(f"processed video in: {elapsed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Center-crop a video to a square, optionally resizing the square output."
    )
    parser.add_argument("--input", help="Input video path. Relative paths are resolved from the repo root.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output video path. Defaults to '<input_stem>_square<input_suffix>' next to the input.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for the default output filename. Ignored when --output is provided.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=0,
        help="Optional square output size in pixels. Defaults to the shortest source side.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for quick previews. Defaults to the whole video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_repo_path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if not input_path.is_file() or not is_video_file(input_path):
        raise ValueError(f"Input must be a supported video file: {input_path}")

    output_dir = resolve_repo_path(args.output_dir) if args.output_dir else None
    output_path = resolve_repo_path(args.output) if args.output else default_output_path(input_path, output_dir)

    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path must be different from the input path.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop_video_to_square(
        input_path,
        output_path,
        output_size=args.size,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
