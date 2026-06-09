# Stimulation Strategies

A strategy receives the sampled baseline stimulation and returns the complete
per-electrode pulse command used by both the percept and safety calculations.

```python
class StimulationStrategy(Protocol):
    def initialize(self, context: StrategyContext) -> None: ...
    def step(self, frame: StrategyInput) -> StimulusCommand: ...
```

`StrategyInput` contains:

- simulation time and frame index;
- normalized stimulus image;
- baseline sampled amplitudes;
- stable electrode IDs and physical coordinates;
- a deterministically seeded NumPy generator.

`StimulusCommand` contains one array per electrode:

- `amplitude_A`;
- `pulse_width_s`;
- `frequency_hz`;
- `activation_mask`.

Every command is checked for shape, finite values, and non-negative physical
values before it reaches the simulator.

See [custom_strategy.py](../examples/experiments/custom_strategy.py) for a
small implementation.

Load it from YAML with:

```yaml
strategy:
  import_path: examples.experiments.custom_strategy:AlternatingHemifieldStrategy
  options:
    maximum_amplitude_uA: 60
```
