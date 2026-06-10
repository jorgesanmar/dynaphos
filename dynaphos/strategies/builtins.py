from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dynaphos.simulation.simulator import _coordinate_based_raster_groups
from dynaphos.strategies.base import StimulusCommand, StrategyContext, StrategyInput


@dataclass
class DirectStrategy:
    options: dict[str, Any] = field(default_factory=dict)
    context: StrategyContext | None = field(default=None, init=False)

    def initialize(self, context: StrategyContext) -> None:
        self.context = context

    def step(self, frame: StrategyInput) -> StimulusCommand:
        if self.context is None:
            raise RuntimeError("Strategy has not been initialized.")
        return StimulusCommand.from_baseline(frame, self.context)


@dataclass
class _GroupedStrategy(DirectStrategy):
    groups: np.ndarray | None = field(default=None, init=False)
    num_groups: int = field(default=4, init=False)

    def initialize(self, context: StrategyContext) -> None:
        super().initialize(context)
        self.num_groups = int(self.options.get("groups", 4))
        if self.num_groups <= 0:
            raise ValueError("Strategy option 'groups' must be greater than zero.")
        self.groups = self._build_groups(context)
        if self.groups.shape != (context.electrode_count,):
            raise ValueError("Built-in strategy produced an invalid group assignment.")

    def _build_groups(self, context: StrategyContext) -> np.ndarray:
        raise NotImplementedError

    def step(self, frame: StrategyInput) -> StimulusCommand:
        if self.context is None or self.groups is None:
            raise RuntimeError("Strategy has not been initialized.")
        active_group = (int(frame.frame_index) - 1) % self.num_groups
        mask = self.groups == active_group
        return StimulusCommand.from_baseline(frame, self.context, activation_mask=mask)


@dataclass
class CheckerboardStrategy(_GroupedStrategy):
    def _build_groups(self, context: StrategyContext) -> np.ndarray:
        points = np.asarray(context.electrode_xy_mm, dtype=np.float64)
        return (
            _coordinate_based_raster_groups(
                points[:, 0],
                points[:, 1],
                pattern="checkerboard",
                num_groups=self.num_groups,
                device="cpu",
            )
            .numpy()
            .astype(np.int64)
        )


@dataclass
class PseudoRandomStrategy(_GroupedStrategy):
    reshuffle_interval_s: float = field(default=5.0, init=False)
    _assignment_version: int = field(default=-1, init=False)

    def initialize(self, context: StrategyContext) -> None:
        self.reshuffle_interval_s = float(self.options.get("reshuffle_interval_s", 5.0))
        if self.reshuffle_interval_s <= 0.0:
            raise ValueError("Strategy option 'reshuffle_interval_s' must be greater than zero.")
        super().initialize(context)

    def _build_groups(self, context: StrategyContext) -> np.ndarray:
        rng = np.random.default_rng(context.seed + max(self._assignment_version, 0))
        points = np.asarray(context.electrode_xy_mm, dtype=np.float64)
        return (
            _coordinate_based_raster_groups(
                points[:, 0],
                points[:, 1],
                pattern="random",
                num_groups=self.num_groups,
                device="cpu",
                rng=rng,
            )
            .numpy()
            .astype(np.int64)
        )

    def step(self, frame: StrategyInput) -> StimulusCommand:
        if self.context is None:
            raise RuntimeError("Strategy has not been initialized.")
        version = int(frame.time_s // self.reshuffle_interval_s)
        if version != self._assignment_version:
            self._assignment_version = version
            self.groups = self._build_groups(self.context)
        return super().step(frame)
BUILTIN_STRATEGIES = {
    "direct": DirectStrategy,
    "none": DirectStrategy,
    "checkerboard": CheckerboardStrategy,
    "pseudo_random": PseudoRandomStrategy,
    "pseudo-random": PseudoRandomStrategy,
    "random": PseudoRandomStrategy,
}
