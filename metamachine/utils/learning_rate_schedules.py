"""Explicit opt-in schedules for SB3's remaining-progress interface."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LateLinearLearningRate:
    initial: float
    final: float
    decay_start_fraction: float = 0.5

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.initial, self.final, self.decay_start_fraction)):
            raise ValueError("Learning-rate schedule values must be finite")
        if not 0 < self.final <= self.initial:
            raise ValueError("Require 0 < final <= initial learning rate")
        if not 0 <= self.decay_start_fraction < 1:
            raise ValueError("decay_start_fraction must be in [0, 1)")

    def __call__(self, progress_remaining: float) -> float:
        completed = 1.0 - float(progress_remaining)
        fraction = min(1.0, max(0.0, (completed - self.decay_start_fraction) / (1.0 - self.decay_start_fraction)))
        return self.initial + fraction * (self.final - self.initial)
