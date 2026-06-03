# Methodology: DynaPhos Simulation and Safety Tracking

## Overview

The safety analysis pipeline extends the DynaPhos phosphene simulator with
electrical and thermal accounting. The pipeline starts from a video of everyday
activity, converts each frame into a stimulation image, samples that image at
the simulated phosphene locations, updates the phosphene percept, and records
safety metrics for the delivered stimulation. The main safety contribution is a
tracker inside `GaussianSimulator` that computes per-frame charge per phase,
charge density, Shannon K, rolling accumulated charge, protocol accumulated
charge, and active-electrode statistics. A separate bioheat model then converts
the delivered electrical power into temperature rise maps.

At a high level, each video frame passes through the following steps:

1. The frame is read from the input video at the video FPS. The simulation time
   step is therefore `dt = 1 / FPS`.
2. The frame is converted to a grayscale stimulation image at the simulator
   resolution.
3. The cortical electrode coordinates are mapped into visual-field phosphene
   locations using the configured cortex model.
4. The stimulation image is sampled at each phosphene location, producing one
   current amplitude per simulated phosphene or electrode.
5. Optional rastering masks the array so only one electrode group is stimulated
   in the current frame.
6. `GaussianSimulator` updates the phosphene state, delivered charge, and
   impedance-based power.
7. Per-phosphene safety values are aggregated back to physical electrodes.
8. Electrical metrics and thermal metrics are written to `safety_metrics.npz`
   for later plotting and comparison.

## Safety Simulation Pipeline Design

Each protocol configuration is treated as an independent simulation case. A
case is defined by the preprocessed input video, electrode coordinate YAML,
stimulation amplitude scale, pulse frequency, pulse width, raster mode,
appearance threshold, and internal-circuit heat assumption.

For each case, the runner builds the simulator once, opens the already
preprocessed video, and then uses a single frame loop:

```text
preprocessed frame
  -> simulator-resolution grayscale stimulation image
  -> per-phosphene sampled raw amplitude
  -> appearance-threshold gate
  -> raster mask
  -> delivered amplitude vector
```

That delivered amplitude vector is the shared signal for three downstream
branches:

1. Phosphene preview branch. The percept renderer receives the delivered
   amplitudes and writes preview frames only for the first 60 seconds of the
   simulation. After that minute, the frame loop continues without rendering
   percept images.
2. Electrical safety branch. The safety tracker updates every video frame from
   delivered charge rate, pulse width, pulse frequency, and the frame duration.
   It records charge per phase, charge density, Shannon K, rolling accumulated
   charge, protocol charge, active-electrode count, and load-power diagnostics.
3. Thermal safety branch. The runner calculates stimulation-dependent device
   power from active delivered amplitudes, impedance, pulse frequency, pulse
   width, and biphasic duty cycle. Power is accumulated over a one-second
   window, and the 3D bioheat model is advanced once per second using the mean
   device power over that window. Temperature maps and hotspot metrics are
   carried forward between thermal updates so the electrical trace remains
   frame-resolved.

After the stimulation video ends, the runner can append a post-video thermal
cooldown phase. During cooldown it no longer decodes video frames, runs the
phosphene simulator, or logs electrical safety samples. Instead, it sets both
stimulation-dependent and constant device power to zero and advances only the
bioheat model until the peak temperature rise is back within the configured
baseline tolerance, or until the configured maximum cooldown duration is reached.

Independent protocol cases can also be run concurrently with the matrix
runner's worker setting. Within one case, the branches share the same sampled
amplitudes so phosphene previews, electrical metrics, and thermal metrics are
computed from the same delivered stimulation.

## Video Preprocessing and Electrode Mapping

For raw media, the visualizer path converts each original video frame to
grayscale, center-crops it to a square, resizes it to the simulator resolution,
applies Gaussian blur, and then applies the selected preprocessing method. The
available image preprocessing functions include no preprocessing, Canny edges,
Sobel edge magnitude, and Difference of Gaussians. Difference of Gaussians is
implemented by subtracting two Gaussian-blurred versions of the same frame and
normalizing the result to the 0-255 image range.

The safety runner usually consumes already-preprocessed videos. In that case,
each decoded video frame is converted to grayscale if necessary, resized to the
configured simulation resolution, and used directly as the stimulation image.
For binary-like inputs such as Canny edges or ground-truth/semantic masks, the
runner applies Otsu thresholding so compression artifacts do not turn a binary
mask into low-level gray stimulation.

Electrode layouts are loaded from YAML coordinate files. These cortical
coordinates are mapped into visual-field phosphene positions using the
configured DynaPhos cortex model, currently the dipole model in
`config/params.yaml`. The full-field mapping duplicates/mirrors the cortical
layout to represent both visual hemifields and keeps the original physical
electrode indices. This index bookkeeping is important because several
simulated phosphene entries can correspond to the same physical electrode after
mirroring or filtering; safety metrics are therefore summed back onto physical
electrodes before analysis.

For safety sweeps, the runner also disables habituation-like temporal changes by
setting the trace increase rate to zero and fixing the activation-threshold
standard deviation to zero. This makes the safety analysis deterministic and
keeps the measured electrical safety signals tied to the delivered stimulus
rather than to adaptation in the perceptual model.

## Amplitude Calculation Per Frame

Let `F_t(x, y)` be the preprocessed frame at time `t`, scaled to the interval
`[0, 1]`. The simulator converts this frame into a stimulation amplitude vector
by sampling it at the phosphene locations.

In the full visual simulator, the default sampling method is receptive-field
sampling. For phosphene `i`, the simulator finds the pixels lying inside that
phosphene's receptive-field mask and uses the maximum frame intensity in that
region:

```text
s_i(t) = max F_t(x, y), for pixels inside RF_i
```

The receptive-field size is defined in cortical millimeters and scaled by the
cortical magnification factor at that phosphene. If center sampling is used,
`s_i(t)` is simply the intensity at the phosphene center pixel. The memory-light
safety mode, `phosphene_mode="safety_centers"`, skips full phosphene distance
maps and therefore samples center pixels even when the global sampling method is
set to receptive fields.

When `rescale=True`, the sampled intensity is converted to current by
multiplying by `sampling.stimulus_scale`:

```text
I_raw_i(t) = s_i(t) * stimulus_scale
```

With the current default configuration, `stimulus_scale = 0.6e-4 A`, so a pixel
with intensity 1.0 corresponds to 60 uA. The safety CLI can also apply a
relative stimulation-scale factor, which multiplies this base scale.

The safety runner then applies a fixed firing threshold before safety and
thermal accounting. If the raw amplitude is not above the rheobase threshold,
the delivered amplitude is set to zero:

```text
I_i(t) = I_raw_i(t) if I_raw_i(t) > rheobase, otherwise 0
```

The default rheobase is `23.9e-6 A`. If raster stimulation is enabled, the
current raster group supplies a binary mask `m_i(t)`, so the final delivered
amplitude is:

```text
I_delivered_i(t) = I_i(t) * m_i(t)
```

Without rastering, all mask values are one. With rastering, electrodes are
assigned to coordinate-based groups such as horizontal, vertical, checkerboard,
or random groups. The safety runner sets the raster timing so one group advances
per video frame. With `N` raster groups and video frame rate `FPS`, the full
raster cycle rate is `FPS / N`.

The delivered current amplitude is stored for analysis as
`current_amplitude_per_electrode_uA` after converting amperes to microamperes
and aggregating simulated phosphene entries back to physical electrodes.

## Phosphene Simulation

The phosphene percept is generated by `GaussianSimulator`. For perceptual
dynamics, the simulator computes an effective stimulation term:

```text
C_eff_i(t) = max((I_delivered_i(t) - trace_i(t) - rheobase) * PW_i * f_i, 0)
```

where `PW_i` is pulse width in seconds and `f_i` is pulse frequency in hertz.
This term updates an activation state with a leaky integrator. A trace state can
model habituation by increasing with stimulation and decaying over time, although
this trace increase is disabled in the safety sweeps described above. Brightness
is obtained by applying a sigmoid saturation function to activation. A phosphene
is rendered only when activation is above its threshold.

The spatial appearance of each phosphene is a Gaussian centered at the
visual-field location of the corresponding electrode. The Gaussian width is
updated from the delivered current amplitude through the configured
current-to-size equation and then scaled by cortical magnification. The rendered
frame is the sum of all active Gaussian phosphene images, clipped to the display
range.

## Safety Tracker

The safety tracker is instantiated inside `GaussianSimulator`, so safety
accounting is updated at the same time as each simulation frame. It computes
metrics even when warning enforcement is disabled. Enforcement is controlled by
`safety.enable_charge_guard` and `safety.charge_warn_only`: the tracker can either
warn, raise an error, or silently compute metrics depending on configuration.
Safety limits are loaded from `config/safety.yaml` unless a different guidelines
path is provided through the parameters.

The tracker receives the delivered stimulation charge rate, not the perceptual
effective current. This is deliberate: safety should describe charge delivered
by the electrode pulse train, not the activation remaining after perceptual
thresholding or trace dynamics. The charge-rate input is:

```text
Qdot_i(t) = I_delivered_i(t) * PW_i * f_i
```

This has units of coulombs per second for one pulse phase.

## Charge Per Phase

Charge per phase is calculated by dividing the delivered charge rate by pulse
frequency and converting coulombs to nanocoulombs:

```text
Q_phase_i(t) = (Qdot_i(t) / f_i) * 1e9
             = I_delivered_i(t) * PW_i * 1e9
```

If the pulse frequency is zero, the charge per phase is set to zero. This metric
is one phase of the biphasic pulse; the factor of two used for total biphasic
charge is not applied to charge per phase.

Inside the simulator, the most recent per-phosphene charge-per-phase vector is
stored as `SafetyTracker.last_charge_per_phase_nC`. In the safety runner, this
vector is aggregated to physical electrodes as `q_phase_elec`. The output file
stores summary forms of this metric, including `charge_per_phase_mean_nC` and
`peak_charge_per_phase_nC_exact`. The full per-electrode charge-per-phase cloud
does not need to be stored separately because it can be reconstructed from the
stored current amplitude and pulse width:

```text
Q_phase_nC = current_amplitude_uA * pulse_width_s * 1e3
```

## Charge Density and Shannon K

Charge density is computed from charge per phase and electrode geometric surface
area:

```text
D_i(t) = Q_phase_i(t) / 1000 / A_elec
```

where `Q_phase_i` is in nanocoulombs, `Q_phase_i / 1000` converts to
microcoulombs, and `A_elec` is the electrode surface area in square
centimeters. The current configuration sets this area through
`safety.electrode_surface_area_cm2`.

Shannon K is then calculated as:

```text
K_i(t) = log10(Q_phase_uC_i(t)) + log10(D_i(t))
```

Only positive charge and positive charge density are valid for the logarithm.
Inactive electrodes are therefore assigned `-inf`. The tracker stores the latest
per-phosphene value in `SafetyTracker.last_shannon_k`; the safety runner
aggregates and saves `shannon_k_mean` and `peak_shannon_k_exact`. As with charge
per phase, full per-electrode Shannon K values can be reconstructed from the
stored amplitude traces, pulse widths, and electrode area.

## Charge Per Second and Total Charge

The analysis distinguishes between per-phase safety accounting inside
`SafetyTracker` and biphasic total delivered charge in the safety runner.

The safety tracker maintains a configurable rolling accumulated-charge window.
For each frame it adds:

```text
DeltaQ_tracker_i(t) = Qdot_i(t) * dt * relative_stim_duration * 1e9
```

The tracker stores this in nanocoulombs per electrode. It keeps an exact rolling
window by storing frame entries in a queue and trimming partial entries when the
window exceeds `safety.charge_window_s`. It also adds the same increment to
`protocol_charge_per_electrode_nC`, which accumulates from the start of the run.
The total rolling charge and total protocol charge are sums over electrodes.

For the main reported "charge per second" metric, the safety runner uses the
biphasic pulse convention. The charge delivered by electrode `i` during one
video frame is:

```text
DeltaQ_frame_i(t) =
    2 * I_i,uA(t) * PW_i,s * f_i,Hz * dt * relative_stim_duration * 1e3
```

The factor of 2 accounts for the two phases of a biphasic pulse. The factor
`1e3` converts `uA * s` from microcoulombs to nanocoulombs. A one-second rolling
queue stores these frame charges and trims partial frames at the one-second
boundary. The resulting vector is reported as per-electrode charge per second in
`nC/s`. The total charge per second is the sum across electrodes.

The output includes:

- `frame_charge_total_nC`: total biphasic charge delivered in each frame.
- `charge_per_second_total_nC_s`: total charge in the one-second rolling window.
- `charge_per_second_mean_per_electrode_nC_s`: mean active-electrode charge per
  second.
- `peak_charge_per_second_per_electrode_nC_s_exact`: exact peak over all frames.
- `protocol_charge_total_nC`: cumulative protocol charge tracked over the run.
- `final_protocol_charge_per_electrode_nC`: final cumulative protocol charge
  for each physical electrode, used for the spatial accumulated-charge heatmap.

For final total delivered charge, downstream plotting utilities prefer summing
`frame_charge_total_nC` when it is present, because that series uses the
biphasic delivered-charge convention.

## Thermal Tracking

Thermal tracking begins with impedance-aware power estimation. The simulator
assigns each electrode a real impedance using a Randles-style model:

```text
Z = R_tis + R_ct / (1 + j * omega * R_ct * C_dl) + sigma_w / sqrt(j * omega)
```

Only the real part is used for the electrical load-power diagnostic. A small
per-electrode random variation is applied to avoid treating all electrodes as
electrically identical. For each frame, instantaneous peak load power is:

```text
P_load_inst_i(t) = I_delivered_i(t)^2 * Re(Z_i)
```

The frame-average electrode load power then multiplies this peak power by the
biphasic duty cycle:

```text
P_load_frame_i(t) = P_load_inst_i(t) * 2 * PW_i * f_i * relative_stim_duration
```

The 2D bioheat model uses this frame-level electrode-load power as the
electrode-local Joule heat source. A separate constant internal-circuit heat
source can also be enabled; it is spread over the electrode-grid footprint. The
update interval can still be overridden in frames for performance or sensitivity
checks.

Temperature propagation is simulated with a coarse two-dimensional Pennes-style
bioheat model using temperature rise above the 37 deg C baseline:

```text
d(dT)/dt = alpha * Lap(dT) - beta * dT + Q / (rho * c)
```

The 2D thermal domain is a single tissue sheet. Electrode-load heat is inserted
at the nearest bioheat cell for each electrode, while internal-circuit heat is
spread uniformly over the convex hull of the electrode grid.

For every thermal sample, the runner records:

- maximum temperature rise, `max_dT`;
- mean temperature rise, `mean_dT`;
- projected tissue area above 1, 2, and 3 deg C;
- final temperature-rise heatmaps;
- final hotspot temperature, saved as `final_peak_temperature_C` and
  `final_peak_dT_C`;
- optional CEM43 thermal dose maps and summaries when CEM43 is enabled.

CEM43 is disabled by default in the common safety runs. When enabled, the runner
integrates thermal dose over time using the temperature-dependent CEM43 weighting
term and only accumulates dose where absolute tissue temperature is above
39 deg C.

## Experimental Configuration

The planned thesis simulations use approximately one hour of point-of-view video
of everyday navigation or activity. The main experimental factors are:

- stimulation amplitude, for example 40, 60, 80, 100, and 120 uA;
- electrode grid density, including the full 800 um grid and a checkerboard-like
  reduced-density grid;
- preprocessing method, including Difference of Gaussians, Canny edge
  extraction, and already-preprocessed semantic or ground-truth binary masks;
- rastering mode, including no rastering and grouped raster stimulation;
- internal-circuit heat assumptions, such as 0, 10, 50, or 100 mW.

For each run, the simulator stores the electrical and thermal time series in
`safety_metrics.npz`, alongside run metadata such as FPS, raster timing, pulse
width, pulse frequency, electrode positions, impedance values, and final thermal
maps. These outputs allow the safety contribution to be evaluated both as
instantaneous per-frame constraints, such as charge per phase and Shannon K, and
as cumulative constraints, such as one-second charge, full-protocol charge, and
temperature rise over the full video.
