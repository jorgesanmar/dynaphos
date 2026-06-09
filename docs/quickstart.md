# Quickstart

## Run the bundled example

```bash
pip install -e .
dynaphos validate examples/experiments/minimal.yaml
dynaphos run examples/experiments/minimal.yaml
```

Open `results/minimal_example/report.html` after the run.

## Use your own stimulus

Copy the minimal YAML and replace:

```yaml
input:
  path: /path/to/stimulus.mp4
  stage: original
  preprocessing_method: dog
```

Use `stage: preprocessed` when the video already contains stimulation masks.
All relative paths are resolved relative to the experiment YAML, not the
current terminal directory.

## Try another strategy

```yaml
strategy:
  name: pseudo_random
  options:
    groups: 5
    reshuffle_interval_s: 5
```

List available built-ins with:

```bash
dynaphos strategies list
```
