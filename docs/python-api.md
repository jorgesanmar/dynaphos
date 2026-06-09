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

results = run_sweep(load_sweep("sweep.yaml"))
```

## Stable extension imports

- `dynaphos.strategies`
- `dynaphos.simulation`
- `dynaphos.safety.electrical`
- `dynaphos.safety.thermal`
- `dynaphos.reporting`
- `dynaphos.media`
