# dynaphos

`dynaphos` is a differentiable and biologically grounded simulator of cortical prosthetic vision. This repository provides:

- the reusable `dynaphos` Python package for mapping cortical electrodes to phosphenes and rendering percepts;
- command-line tools for preprocessing media, rendering a phosphene video or image, and running electrical and 2D thermal safety checks for one stimulation protocol.

This README describes the repository as a single-protocol simulation pipeline.

## Repository Pipeline At A Glance

The typical flow is:

1. Start with an image or video in `videos/` or another local path.
2. Optionally preprocess frames into grayscale stimulation masks such as `dog`, `canny`, or `sobel`.
3. Load an electrode grid from `config/coords_*um.yaml`.
4. Map each cortical electrode coordinate into a visual-field phosphene location.
5. Sample the preprocessed frame at the phosphene receptive fields to produce one stimulation amplitude per mapped electrode.
6. Optionally raster the array so only one electrode group is active at a time.
7. Run the `GaussianSimulator` to update activation, trace, brightness, thresholds, impedance, and the rendered phosphene percept.
8. Run the safety analysis for the same protocol, saving electrical metrics, 2D bioheat metrics, summaries, and per-run figures.

## What Lives Where

- `dynaphos/`: reusable simulation code, cortex mapping, image processing, phosphene rendering, safety tracking, bioheat, and plotting utilities.
- `config/`: simulator defaults, safety thresholds, and electrode-grid coordinate YAML files.
- `tools/preprocessing/`: preprocessing previews and exported preprocessed videos.
- `tools/electrode_grid/`: visualization of cortical electrode layouts and their phosphene mappings.
- `tools/phosphenes/`: phosphene rendering from images or videos.
- `tools/safety/`: entry points for safety simulations and visualizations.
- `examples/`: notebooks and lightweight package demos.
- `results/`: generated figures, reports, previews, and `.npz` metric files.
- `videos/`: source media and exported preprocessed variants.

## Core Concepts

### 1. Input Media

The pipeline accepts images or videos from any local path. The recommended video layout is to keep each source recording and its derived preprocessing variants in the same directory. For a source video:

`videos/<dataset_or_study>/<video_name>.mp4`

the preprocessing exporter writes standalone variants next to it:

- `videos/<dataset_or_study>/<video_name>_dog.mp4`
- `videos/<dataset_or_study>/<video_name>_canny.mp4`
- `videos/<dataset_or_study>/<video_name>_none.mp4`

It may also write nearby helper outputs such as `<video_name>_<method>_comparison_with_original.mp4` and `<video_name>_<method>_frames/`. This keeps a protocol's source and stimulation masks together, and the rendering and safety tools can infer the preprocessing method from the filename suffix.

### 2. Preprocessing

Preprocessing converts raw frames into stimulation masks. The repository currently supports:

- `none`: blurred grayscale input.
- `dog`: difference of Gaussians.
- `canny`: Canny edge detector.
- `sobel`: Sobel edge magnitude.

Frames are converted to grayscale, center-cropped to square, resized to the simulator resolution, and then passed through the selected preprocessing method.

### 3. Electrode Grid And Phosphene Mapping

Electrode layouts are stored as YAML files containing parallel `x` and `y` coordinate arrays. The row position in those arrays is the base electrode index: row `0` is electrode `0`, row `1` is electrode `1`, and so on.

The mapping step loads those cortical coordinates, applies the cortex model from `config/params.yaml`, and produces phosphene coordinates in visual-field space. Those coordinates determine where phosphenes appear in the rendered percept and where the input image is sampled. When full-field mapping is used, the base grid can be mirrored to represent both hemifields. In that case the saved metrics keep both:

- `electrode_ids`: the global electrode id used in safety arrays.
- `electrode_base_indices`: the original row in the coordinate YAML.
- `electrode_grid_ids` and `electrode_grid_names`: which mirrored grid or hemisphere the electrode belongs to.
- `electrode_xy_mm`: the cortical position used for impedance, power, charge maps, and bioheat placement.

This means a phosphene appearance can be traced back to the physical electrode row that generated it, and the same electrode identity is used for current, charge, impedance, power, and temperature tracking.

### 4. Stimulus Sampling

The simulator samples the preprocessed frame at phosphene locations or receptive fields and converts pixel intensity into stimulation amplitude using:

- `run.resolution`
- `sampling.sampling_method`
- `sampling.RF_size`
- `sampling.stimulus_scale`

With receptive-field sampling, each electrode samples a local region around its mapped phosphene location. With center sampling, it samples the center pixel for that mapped phosphene.

### 5. Raster Stimulation

Rastering partitions the electrode array into groups so only one group is active at a time. Supported modes include:

- `none`
- `horizontal`
- `vertical`
- `checkerboard`
- `random` / `pseudo-random`

Rastering affects both the percept and safety metrics because it changes which electrodes are active in each frame, how much charge is delivered per electrode, and how power is distributed over time.

### 6. Phosphene Simulation

`dynaphos.simulator.GaussianSimulator` is the phosphene engine. For each frame it:

- updates activation and memory trace dynamics,
- applies thresholding and brightness saturation,
- applies the raster mask when rastering is enabled,
- renders the phosphene percept as a sum of Gaussian activations.

### 7. 2D Bioheat Model

Safety simulations use `dynaphos.safety.bioheat.Bioheat2D`, a 2D Pennes-style temperature-rise model. The model tracks `dT`, the temperature increase above baseline, on a single tissue sheet:

`d(dT)/dt = alpha * Laplacian(dT) - beta * dT + Q / (rho * c)`

Two heat sources are inserted into this 2D model:

- Electrode load heat: the frame-level dissipated stimulation power for each electrode is placed at the nearest bioheat grid cell for that electrode. The power is converted to a volumetric source using `bioheat.electrode_source_thickness_mm`.
- Constant internal-circuit/device heat: `bioheat.internal_circuit_power_mw` / `bioheat.device_constant_power_mw` is spread over the convex hull of the electrode grid. This source uses `bioheat.ic_source_thickness_mm`.

The 2D model records mean and focal temperature rise, area above temperature thresholds, final temperature maps, and heatmap snapshots.

## Main Configuration Files

- `config/params.yaml`: simulator, sampling, raster, impedance, temporal-dynamics, and 2D bioheat settings.
- `config/safety.yaml`: electrical and thermal safety thresholds used by the safety pipeline.
- `config/coords_*um.yaml`: physical electrode layouts.

If you want to change how a run behaves, start with `config/params.yaml`.

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

This writes visualization panels to `results/preprocessing/visualize_preprocessing/`. Exported stimulation videos are saved next to each input video as `<video_name>_<method>.mp4`.

### 3. Inspect The Electrode Grid And Phosphene Map

```bash
python tools/electrode_grid/visualize_electrode_grid.py --save-combined
```

Use this before longer runs to verify the cortical electrode coordinates, electrode indexing, mirrored/full-field layout, and final percept-space phosphene map.

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
    --input videos/my_study/some_video_dog.mp4 \
    --input-stage preprocessed \
    --raster-mode checkerboard
```

For a still image, use:

```bash
python tools/phosphenes/render_image.py --input path/to/image.png
```

By default these tools write to `results/phosphenes/` unless you override `--output-dir`.

### 5. Run One Safety Simulation

The single-protocol safety runner is `dynaphos/safety/runner.py`. It runs one video, preprocessing method, stimulation configuration, raster mode, and internal-circuit heat setting at a time.

Example:

```bash
python -m dynaphos.safety.runner \
    --video videos/my_study/some_video_dog.mp4 \
    --preprocessing-methods dog \
    --raster-modes checkerboard \
    --ic-heat-modes with \
    --device_constant_power_mw 13 \
    --appearance_threshold_uA 30 \
    --preview_seconds 5
```

The runner writes one case directory containing:

- `run_manifest.yaml`: the protocol settings and paths used for the run.
- `summary.txt`: readable scalar summaries for configuration, electrical results, thermal results, and per-grid thermal results.
- `safety_metrics.npz`: raw arrays for analysis and visualization.
- `device_power_over_time.png`: active-electrode count and modeled power components over time.
- `thermal_response_over_time.png`: mean and maximum temperature rise over time.

`summary.txt` is the quickest text output to inspect. It records the run configuration, heatmap snapshot times, peak current, peak charge per phase, peak charge density, peak Shannon K, peak charge rate, final accumulated protocol charge, peak active-electrode count, peak/final temperature rise, hotspot areas above 1/2/3 degrees C, and per-grid final temperature values.

Preview videos are written under the configured visuals directory when `--preview_seconds` is greater than zero and `--phosphene-mode visual` is used.

By default, safety outputs are written under `results/safety/safety_analysis/<video>/<preprocessing>/<ic_heat_mode>/<raster_mode>/`. The current runner uses `config/coords_800um.yaml`; change that value in the runner or call `run_one_mode` directly if you need another electrode grid.

### 6. Generate Per-Protocol Safety Visuals

To build a compact dashboard for one completed case:

```bash
python -m dynaphos.safety.single_case \
    --input path/to/case_directory
```

This writes `single_case_overview.png`, a one-page figure with:

- delivered current over time,
- active/stimulated electrode counts,
- charge and charge-rate traces,
- mean and maximum temperature rise,
- hotspot area over time,
- a mid-simulation temperature map.

To generate the full set of per-protocol safety plots for completed cases under a root directory:

```bash
python tools/safety/visualize.py \
    --input-root results/safety/safety_analysis
```

For each case, the per-protocol visuals include:

- `single_case_overview.png`: compact dashboard and scalar footer.
- `current_amplitudes_over_time.png`: current distribution across active electrodes over time.
- `activated_electrodes_over_time.png`: active electrode count over time.
- `shannon_k_over_time.png`: Shannon K distribution over active electrodes over time.
- `charge_per_second_over_time.png`: per-electrode and summed charge rate over time.
- `charge_over_whole_protocol.png`: cumulative charge map, total accumulated charge, and per-electrode accumulated charge distribution.
- `temperature_heatmaps_<grid>.png`: all saved 2D temperature snapshots for each bioheat grid.
- `temperature_heatmaps_<grid>/`: one image per saved temperature snapshot.
- `mean_dT_over_time.png`: mean temperature rise over time.
- `max_focal_dT_over_time.png`: maximum focal temperature rise over time.

## Structure Of `safety_metrics.npz`

`safety_metrics.npz` is a NumPy archive. Most arrays are `float32`; ids and frame indices are integer arrays; labels are string arrays. Let:

- `T` = number of electrical/video samples.
- `Tt` = number of thermal samples.
- `N` = number of electrodes in the saved reference table.
- `S` = number of saved heatmap snapshots.
- `H, W` = bioheat grid height and width.

Core timing and run metadata:

- `time_s` `(T,)`: electrical/video sample times.
- `thermal_time_s` `(Tt,)`: thermal sample times.
- `device_power_time_s` `(Tt,)`: power sample times.
- `fps`, `save_every_n_frames`, `threshold_uA`, `relative_stim_duration`.
- `pulse_width_s` `(N,)`, `pulse_frequency_hz` `(N,)`.
- `raster_mode`, `raster_mode_normalized`, `raster_num_groups`, `raster_rate_hz`, `raster_group_step_rate_hz`, `raster_group_interval_s`, `raster_reshuffle_interval_s`.

Electrode identity and placement:

- `electrode_ids` `(N,)`: global saved electrode ids.
- `electrode_base_indices` `(N,)`: original row indices from the coordinate YAML.
- `electrode_xy_mm` `(N, 2)`: cortical x/y positions in millimeters.
- `electrode_impedance_ohm` `(N,)`: impedance used for power calculation.
- `electrode_grid_ids` `(N,)`, `electrode_grid_names` `(N,)`: grid/hemifield membership.
- `electrode_surface_area_cm2`: electrode surface area used for charge density.

Electrical metrics, present when electrical tracking is enabled:

- `amplitude_per_electrode_uA` `(T, N)`: delivered current amplitude.
- `charge_per_phase_per_electrode_nC` `(T, N)`: charge per phase.
- `charge_density_per_electrode_uc_cm2` `(T, N)`: charge density.
- `shannon_k_per_electrode` `(T, N)`: Shannon safety metric.
- `charge_per_second_per_electrode_nC_s` `(T, N)`: charge rate.
- `window_charge_per_electrode_nC` `(T, N)`: rolling-window accumulated charge.
- `window_charge_total_nC` `(T,)`: rolling-window accumulated charge summed over electrodes.
- `protocol_charge_per_electrode_nC` `(N,)`: final accumulated charge per electrode over the protocol.
- `power_per_electrode_W` `(T, N)`: stimulation load power used by bioheat.
- `active_electrode_count` `(T,)`: active electrodes per sample.
- `charge_window_s`: accumulation-window duration.
- `window_exceedance_time_start_s`, `window_exceedance_time_end_s`, `window_exceedance_scope`, `window_exceedance_electrode_id`, `window_exceedance_charge_nC`, `window_exceedance_limit_nC`: recorded rolling-window charge-limit exceedances.
- `raster_active_group` `(T,)`, `raster_group_assignment_frame_indices`, `raster_group_assignment_times_s`, `raster_group_assignments`: raster state and electrode group assignments.

Thermal and power metrics:

- `max_dT` `(Tt,)`: maximum focal temperature rise.
- `mean_dT` `(Tt,)`: mean temperature rise over the implant footprint.
- `area_gt1_mm2`, `area_gt2_mm2`, `area_gt3_mm2` `(Tt,)`: 2D area above 1, 2, and 3 degrees C.
- `dT_final` `(H, W)`: final 2D temperature-rise map for the hottest/reference grid.
- `internal_circuit_power_W`, `electrode_load_power_W` `(Tt,)`: internal-circuit heat and electrode-load/Joule-heat power used by the 2D bioheat model.
- `internal_circuit_power_total_mW`, `internal_circuit_footprint_pixels`, `internal_circuit_power_density_W_m3`.
- `extent_mm`, `voxel_size_mm`, `bioheat_model`.
- `final_peak_temperature_C`, `final_peak_dT_C`: final/reference-grid peak temperature and temperature-rise estimates.

Per-grid thermal maps use the grid name as a suffix:

- `thermal_grid_names`: available grid names.
- `dT_final_reference_grid`: grid used for unsuffixed `dT_final`.
- `dT_final_<grid>`.
- `final_peak_temperature_C_<grid>`, `final_peak_dT_C_<grid>`.
- `extent_mm_<grid>`, `voxel_size_mm_<grid>`.
- `internal_circuit_footprint_pixels_<grid>`, `internal_circuit_power_density_W_m3_<grid>`.
- `max_dT_<grid>`, `mean_dT_<grid>`: per-grid temperature traces.

Heatmap snapshots:

- `heatmap_frame_indices` `(S,)`, `heatmap_times_s` `(S,)`.
- `heatmap_target_fractions` `(S,)`, `heatmap_actual_fractions` `(S,)`.
- `heatmap_grid_names`.
- `dT_heatmaps_<grid>` `(S, H, W)`: saved 2D temperature-rise snapshots, one every 5% of total simulation time where possible.

Optional CEM43 thermal-dose metrics are only stored when `--enable-cem43` is used:

- `max_cem43`, `p99_cem43`, `cem43_final`, `cem43_extent_mm`.
- `cem43_final_<grid>`, `cem43_heatmaps_<grid>`.

## Tests

Run the focused unit tests with:

```bash
pytest
```

The tests exercise safety case expansion, visualization helpers, simulator thresholds, impedance/power helpers, and bioheat behavior.

## Package Usage

If you want to use the simulator directly in Python instead of the CLI tools, start from:

- `dynaphos/simulator.py`
- `dynaphos/pipeline.py`
- `dynaphos/safety/`
- `dynaphos/cortex_models.py`
- `dynaphos/image_processing.py`

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
- Original experiment code built around this simulator: [neuralcodinglab/dynaphos-experiments](https://github.com/neuralcodinglab/dynaphos-experiments)

## Contact

jorge.sanmartin24@gmail.com

Issues and questions are welcome.
