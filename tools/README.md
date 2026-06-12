# Research Tools

`tools/` contains unsupported, repository-only research experiments. Supported
workflows are available through the installed `dynaphos` command.

Bioheat3D experiments live under `tools/experiments/bioheat3d/`. They write
research outputs under `results/` and are intentionally excluded from pytest
collection.

## `benchmark_pipeline.py`

End-to-end timing of the full pipeline on a real video, driven through the same
`run_experiment` entry point as the CLI. Run from the repo root.

- `--video PATH` (required): input clip.
- `--calibrate N`: warm up, then time `N`- and `2N`-frame prefixes and use the
  subtraction method to split fixed overhead (model build + report) from true
  per-frame cost, then project the whole video and a `--target-minutes` clip.
  Preferred for a fast projection without processing the entire video.
- no `--calibrate`: run the whole video (or `--frames N`) once and report total
  wall-clock, end-to-end frames/s, peak dT, and safety status.
- `--device {cpu,cuda,auto}`, plus knobs to match a real run (`--electrodes`,
  `--resolution`, `--strategy`/`--groups`, `--phosphene-mode`, `--cem43`,
  `--figures`, protocol amplitudes, `--cooldown`). Defaults mirror a
  representative production run.

Examples:

    python tools/benchmark_pipeline.py --video clip.mp4 --calibrate 200
    python tools/benchmark_pipeline.py --video clip.mp4 --device cuda --figures

The `--calibrate` marginal cost is the number to trust for projecting a 20-min
video; the single-run total additionally includes one-time model build and
report generation.

<!-- Update this section if benchmark_pipeline.py CLI flags or output change. -->

