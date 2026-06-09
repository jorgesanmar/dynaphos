# Experiment Configuration

Experiment YAML is strict: unknown fields, invalid ranges, and malformed nested
sections are rejected before simulation.

## Sections

### `input`

- `path`: image or video path.
- `builtin`: packaged fixture name, currently `cat`.
- `stage`: `original` or `preprocessed`.
- `preprocessing_method`: for example `dog`, `canny`, or `groundtruth`.
- `preprocessing_options`: method-specific values such as Canny thresholds.

Define exactly one of `path` and `builtin`.

### `electrode_array`

Define exactly one of:

```yaml
electrode_array:
  builtin: coords_800um
```

```yaml
electrode_array:
  coordinates: arrays/my_array.yaml
```

Built-ins are `coords_400um`, `coords_800um`, and `coords_1200um`.

### `protocol`

All units are explicit:

```yaml
protocol:
  amplitude_uA: 60
  appearance_threshold_uA: 30
  pulse_width_us: 170
  frequency_hz: 300
  relative_stim_duration: 1.0
  internal_circuit_power_mW: 0
```

### `strategy`

Use a built-in name or a custom Python import path:

```yaml
strategy:
  name: checkerboard
  options:
    groups: 4
```

```yaml
strategy:
  import_path: my_package.strategies:MyStrategy
  options:
    parameter: value
```

### `simulation`

Controls runtime, resolution, preview generation, cooldown, and optional CEM43
evaluation. `force_cpu: true` is useful for portable reproducibility.

### `safety`

Selects the safety-limit YAML and electrical or electrode-heat tracking.
Limits default to the packaged `safety.yaml`.

### `output`

`root` and `run_id` determine the result directory. Existing output is rejected
unless `overwrite: true`.

## Sweeps

A sweep points to one experiment and replaces dotted fields:

```yaml
experiment: base.yaml
matrix:
  protocol.amplitude_uA: [10, 60, 120]
  electrode_array.builtin: [coords_400um, coords_800um]
```

Use `{label, value}` entries for readable run IDs or `{label, patch}` entries
when one condition must update several fields. See `examples/studies/`.
