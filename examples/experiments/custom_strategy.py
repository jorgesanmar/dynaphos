from __future__ import annotations

import numpy as np

from dynaphos.strategies import StimulusCommand, StrategyContext, StrategyInput


class AlternatingHemifieldStrategy:
    """Alternate electrodes on the left and right side of the array."""

    def __init__(self, options: dict | None = None):
        self.options = options or {}
        self.context: StrategyContext | None = None

    def initialize(self, context: StrategyContext) -> None:
        self.context = context

    def step(self, frame: StrategyInput) -> StimulusCommand:
        if self.context is None:
            raise RuntimeError("Strategy has not been initialized.")
        left = frame.electrode_xy_mm[:, 0] < 0
        mask = left if frame.frame_index % 2 else ~left
        amplitude = np.minimum(
            frame.baseline_amplitude_A,
            float(self.options.get("maximum_amplitude_uA", 60.0)) * 1e-6,
        )
        return StimulusCommand(
            amplitude_A=amplitude,
            pulse_width_s=self.context.default_pulse_width_s,
            frequency_hz=self.context.default_frequency_hz,
            activation_mask=mask,
        )
