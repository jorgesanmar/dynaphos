# Python API

## Load and run

```python
from dynaphos import load_experiment, run_experiment

config = load_experiment("experiment.yaml")
result = run_experiment(config)

print(result.run_dir)
print(result.report.status)
```

## Construct a config in Python

```python
from dynaphos import ExperimentConfig, InputConfig

config = ExperimentConfig(
    input=InputConfig(path="stimulus.mp4"),
)
```

For untrusted or external dictionaries, prefer
`ExperimentConfig.from_dict(...)`; it performs recursive conversion and strict
validation.

## Run sweeps

```python
from dynaphos import load_sweep, run_sweep

results = run_sweep(load_sweep("sweep.yaml"), resume=True)
```

## Read completed runs

```python
from dynaphos.experiment import discover_completed_runs, load_metrics

for run in discover_completed_runs("results/my_study"):
    metrics = load_metrics(run.metrics_path)
    print(run.run_dir, metrics["time_s"].shape)
```

## Stable extension imports

- `dynaphos.experiment`
- `dynaphos.strategies`
- `dynaphos.simulation`
- `dynaphos.media`
- `dynaphos.electrodes`
- `dynaphos.safety`
- `dynaphos.reporting`
- `dynaphos.studies`

The removed pre-0.2 root-module imports are listed in the
[migration guide](migration-0.2.md).
