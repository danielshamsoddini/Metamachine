"""Physics adapters. Optional engines are imported only when selected."""

from .base import PhysicsBackend
from .mujoco import MuJoCoBackend

__all__ = ["PhysicsBackend", "MuJoCoBackend"]
