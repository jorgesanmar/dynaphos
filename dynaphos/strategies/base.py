from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class StrategyContext:
    electrode_ids: np.ndarray
    electrode_xy_mm: np.ndarray
    default_pulse_width_s: np.ndarray
    default_frequency_hz: np.ndarray
    fps: float
    seed: int

    @property
    def electrode_count(self) -> int:
        return int(self.electrode_ids.size)


@dataclass(frozen=True)
class StrategyInput:
    time_s: float
    frame_index: int
    stimulus_image: np.ndarray
    baseline_amplitude_A: np.ndarray
    electrode_ids: np.ndarray
    electrode_xy_mm: np.ndarray
    rng: np.random.Generator


@dataclass(frozen=True)
class StimulusCommand:
    amplitude_A: np.ndarray
    pulse_width_s: np.ndarray
    frequency_hz: np.ndarray
    activation_mask: np.ndarray

    @classmethod
    def from_baseline(
        cls,
        frame: StrategyInput,
        context: StrategyContext,
        *,
        activation_mask: np.ndarray | None = None,
    ) -> "StimulusCommand":
        mask = (
            np.ones(context.electrode_count, dtype=bool)
            if activation_mask is None
            else np.asarray(activation_mask, dtype=bool)
        )
        return cls(
            amplitude_A=np.asarray(frame.baseline_amplitude_A, dtype=np.float32),
            pulse_width_s=np.asarray(context.default_pulse_width_s, dtype=np.float32),
            frequency_hz=np.asarray(context.default_frequency_hz, dtype=np.float32),
            activation_mask=mask,
        )

    def validated(self, context: StrategyContext) -> "StimulusCommand":
        expected = (context.electrode_count,)
        arrays = {
            "amplitude_A": np.asarray(self.amplitude_A, dtype=np.float32),
            "pulse_width_s": np.asarray(self.pulse_width_s, dtype=np.float32),
            "frequency_hz": np.asarray(self.frequency_hz, dtype=np.float32),
            "activation_mask": np.asarray(self.activation_mask, dtype=bool),
        }
        for name, value in arrays.items():
            if value.shape != expected:
                raise ValueError(
                    f"Strategy command {name} has shape {value.shape}; expected {expected}."
                )
        for name in ("amplitude_A", "pulse_width_s", "frequency_hz"):
            value = arrays[name]
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Strategy command {name} contains non-finite values.")
            if np.any(value < 0.0):
                raise ValueError(f"Strategy command {name} contains negative values.")
        arrays["amplitude_A"] = arrays["amplitude_A"] * arrays["activation_mask"]
        return StimulusCommand(**arrays)


class StimulationStrategy(Protocol):
    def initialize(self, context: StrategyContext) -> None: ...

    def step(self, frame: StrategyInput) -> StimulusCommand: ...
