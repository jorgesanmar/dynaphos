# dynaphos

`dynaphos` is a differentiable and biologically grounded simulator of cortical prosthetic vision. The repository contains two closely related parts:

- The reusable `dynaphos` Python package for mapping cortical electrodes to phosphenes and rendering percepts.
- A research pipeline around that package for preprocessing videos, testing raster stimulation strategies, running electrical and thermal safety analyses, and generating figures and reports.

This README explains the repository as a pipeline so it is easier to see how the pieces fit together.

## Repository Pipeline At A Glance

The typical project flow is:

1. Start with an image or video in `videos/` or another local path.
2. Preprocess frames into stimulation-friendly grayscale representations such as `dog`, `canny`, or `sobel`.
3. Load an electrode grid from `config/coords_*um.yaml` and map those cortical coordinates into phosphene locations in visual-field space.
4. Sample the preprocessed frame at the phosphene receptive fields to produce per-electrode stimulation amplitudes.
5. Optionally raster the array so only one electrode group is active at a time.
6. Run the `GaussianSimulator` to update activation, trace, brightness, thresholds, impedance, and the rendered phosphene percept.
7. Derive electrical and thermal safety signals from the delivered stimulation and propagate device heating through the `Bioheat3D` model.
8. Save per-run outputs to `results/` and generate comparison plots and summaries across preprocessing methods, raster modes, and heat-model settings.

## What Lives Where

- `dynaphos/`: reusable simulation pipeline code, cortex mapping, image processing, phosphene rendering, safety tracking, and plotting utilities.
- `config/`: simulator defaults, safety thresholds, and electrode-grid coordinate YAML files.
- `tools/preprocessing/`: preprocessing visualizations and export of standalone preprocessed videos.
- `tools/electrode_grid/`: visualization of cortical electrode layouts and their phosphene mappings.
- `tools/phosphenes/`: rendering of phosphene outputs from images or videos, with or without rastering.
- `tools/safety/`: thin entry points for running and visualizing the safety experiment matrix.
- `tools/legacy/`: archived older scripts and one-off utilities.
- `examples/`: notebooks and lightweight demos for the package itself.
- `results/`: generated figures, reports, previews, and `.npz` metrics files.
- `videos/`: source media and exported preprocessed variants.

## Core Concepts

### 1. Input Media

The pipeline accepts images or videos. Video-based experiments usually begin with recordings in `videos/`, then branch into preprocessed exports under:

`videos/preprocessed/<video_name>/<method>/preprocessed.mp4`

### 2. Preprocessing

Preprocessing converts raw frames into stimulation masks. The repository currently uses:

- `none`: blurred grayscale input.
- `dog`: difference of Gaussians.
- `canny`: Canny edge detector.
- `sobel`: Sobel edge magnitude.

This stage normalizes frame geometry by converting to grayscale, center-cropping to square, resizing to the simulator resolution, and then applying the chosen preprocessing method.

### 3. Electrode Grid And Cortex Mapping

Electrode layouts are defined in YAML files such as:

- `config/coords_400um.yaml`
- `config/coords_800um.yaml`
- `config/coords_1200um.yaml`

These physical cortical coordinates are mapped into visual-field coordinates through the cortex model configured in `config/params.yaml`. That mapping determines where phosphenes appear and which receptive fields are sampled from the input image.

### 4. Stimulus Sampling

The simulator samples the preprocessed frame at phosphene locations or receptive fields and converts pixel intensity into stimulation amplitude using the scaling rules in `config/params.yaml`, especially:

- `run.resolution`
- `sampling.sampling_method`
- `sampling.RF_size`
- `sampling.stimulus_scale`

### 5. Raster Stimulation

Rastering partitions the electrode array into groups so only one group is active at a time. This repo supports:

- `none`
- `horizontal`
- `vertical`
- `checkerboard`
- `random`

Raster configuration matters both for percept generation and for downstream safety, because it changes which electrodes are simultaneously active and how charge and power are distributed over time.

### 6. Phosphene Simulation

`dynaphos.simulator.GaussianSimulator` is the core phosphene engine. For each frame it:

- updates activation and memory trace dynamics,
- applies thresholding and brightness saturation,
- optionally masks inactive raster groups,
- renders the phosphene percept as a sum of Gaussian activations.

### 7. Safety And Bioheat

The analysis pipeline tracks electrical metrics such as:

- charge per phase,
- charge density,
- current amplitude,
- Shannon K,
- charge accumulated over time,
- stimulated electrode counts,
- instantaneous and average power.

It also runs `dynaphos.safety.bioheat.Bioheat3D` to estimate temperature rise and hotspot area from activity-dependent device power. Safety thresholds come primarily from `config/safety.yaml`.

### 8. Reports And Comparisons

Per-run outputs are written as figures, summaries, preview videos, and raw `.npz` metric bundles. Comparison scripts aggregate those runs to answer questions like:

- How do `dog` and `canny` preprocessing differ?
- Does `checkerboard` rastering reduce peak temperature versus `none`?
- What changes when internal-circuit heat is included or excluded?

## Main Configuration Files

- `config/params.yaml`: default simulator, sampling, raster, impedance, temporal-dynamics, and bioheat settings.
- `config/safety.yaml`: electrical and thermal safety thresholds used by the safety pipeline.
- `config/safety_experiments.yaml`: example declarative safety matrix for comparing preprocessing, rastering, and stimulation parameters.
- `config/coords_*um.yaml`: physical electrode layouts.

If you want to change how the whole pipeline behaves, `config/params.yaml` is usually the first file to inspect.

## Typical Workflow

### 1. Install

```bash
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` lists the direct runtime, test, and notebook dependencies. The same runtime dependencies are declared in `pyproject.toml`, so package consumers can also install with `pip install -e .` and add development tools separately when needed.

### 2. Inspect Or Export Preprocessing Variants

```bash
python tools/preprocessing/visualize_preprocessing.py \
    --input videos \
    --methods none dog canny \
    --preview-seconds 5
```

This writes visualization panels to `results/preprocessing/visualize_preprocessing/` and exported stimulation videos to `videos/preprocessed/`.

### 3. Inspect The Electrode Grid And Phosphene Map

```bash
python tools/electrode_grid/visualize_electrode_grid.py --save-combined
```

This helps verify the physical cortical layout and the corresponding percept-space mapping before running larger experiments.

### 4. Render Phosphene Outputs

From original media:

```bash
python tools/phosphenes/render_video.py \
    --input videos/some_video.mp4 \
    --input-stage original \
    --preprocessing-method dog \
    --raster-mode checkerboard
```

From already preprocessed media:

```bash
python tools/phosphenes/render_video.py \
    --input videos/preprocessed/some_video/dog/preprocessed.mp4 \
    --input-stage preprocessed \
    --raster-mode checkerboard
```

For a still image, use `python tools/phosphenes/render_image.py --input path/to/image.png`.
By default these tools write to `results/phosphenes/` unless you override `--output-dir`.

### 5. Run Safety Simulations

The safety runner reads a declarative experiment matrix. Start from
`config/safety_experiments.yaml`, set the input video, electrode grid,
preprocessing methods, raster modes, and stimulation parameters you want to
compare, then list the planned runs without simulating:

```bash
python tools/safety/run_experiment_matrix.py --dry-run
```

Run the matrix and generate safety summaries/visuals:

```bash
python tools/safety/run_experiment_matrix.py
```

Use a custom matrix file when you want to keep a study-specific configuration outside the repo:

```bash
python tools/safety/run_experiment_matrix.py \
    --matrix-config path/to/my_safety_experiments.yaml \
    --blocks strategy_comparison
```

The experiment YAML is the source of truth for amplitude, electrode density,
preprocessing, internal-circuit power, rastering, and any other strategy axes
you want to sweep.

Each script writes manifests, summaries, previews when requested, and compact
`safety_metrics.npz` files under:

`results/safety/simulation_pipeline/<sweep_block>/<run_id>/`

The matrix runner defaults to metrics-only operation with `--preview-policy none`.
Electrical safety metrics are evaluated every frame when tracking is enabled,
and thermal bioheat metrics are summarized into reports and comparison plots.

For full 30-minute videos, the experiment matrix also defaults to treating the
last 300 seconds as an implant-off cooldown tail. During that tail it skips
video decoding, phosphene rendering, and electrical logging, sets device power
to zero, and continues only the bioheat cooldown.

CEM43 is disabled by default; pass `--enable-cem43` only if thermal dose metrics are needed.

### 6. Generate Safety Visuals And Comparisons

```bash
python tools/safety/visualize.py \
    --input-root results/safety/simulation_pipeline
```

This writes per-case dashboards and comparative plots under `results/safety/simulation_pipeline/visuals/`.

## Tests

Run the focused unit tests with:

```bash
pytest
```

The tests exercise the safety experiment expansion, visualization helpers, simulator thresholds, impedance/power helpers, and bioheat behavior.

## Package Usage

If you want to use the simulator directly in Python instead of the CLI tools, start from:

- `dynaphos/simulator.py`
- `dynaphos/pipeline.py`
- `dynaphos/safety/`
- `dynaphos/cortex_models.py`
- `dynaphos/image_processing.py`
- `examples/demo_simulator.ipynb`
- `examples/demo_webcam.py`

The package-level flow is:

1. load parameters,
2. load cortical electrode coordinates,
3. map them to phosphenes,
4. instantiate `GaussianSimulator`,
5. sample a stimulus image into electrode amplitudes,
6. call the simulator to render the percept.

## Citation

van der Grinten, M., van Steveninck, J. D. R., Lozano, A., Pijnacker, L., Rueckauer, B., Roelfsema, P., Marcel van Gerven, Richard van Wezel, Umut Guclu & Gucluturk, Y. (2024). Towards biologically plausible phosphene simulation for the differentiable optimization of visual cortical prostheses. eLife, 13, e85812. [https://doi.org/10.7554/eLife.85812](https://doi.org/10.7554/eLife.85812)

## Related Repositories

- End-to-end optimization experiments from the publication: [neuralcodinglab/viseon/tree/dynaphos-paper](https://github.com/neuralcodinglab/viseon/tree/dynaphos-paper)
- Experiment code built around this simulator: [neuralcodinglab/dynaphos-experiments](https://github.com/neuralcodinglab/dynaphos-experiments)

## Contact

Issues and questions are welcome.
