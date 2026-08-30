"""Replay-buffer augmentation for reflection-symmetric ant locomotion tasks.

This module is intentionally opt-in.  It reflects complete transitions about
the commanded travel axis, including both observations and actions, while
leaving scalar rewards and termination flags unchanged.  Half of every vector
batch is reflected, with the assignment alternating each environment step so
that every worker contributes equally to the original and reflected datasets.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class AntCommandReflectionReplayBuffer(ReplayBuffer):
    """Reflection-augment flat, stacked actuator-aware ant observations.

    The expected per-frame layout is the approved 48-D actuator-aware layout::

        gravity(3), body_gyro(3), heading_cos(1), heading_sin(1),
        q(8), qd(8), filtered_target(8), tracking_error(8), last_action(8)

    ``reflection_axis_radians`` is expressed in the ant body XY convention.
    The approved forward task uses pi/2 (+Y), and the west-of-forward diagonal
    task uses 3*pi/4 (-X,+Y).
    """

    _JOINT_BLOCK_STARTS = (8, 16, 24, 32, 40)

    def __init__(
        self,
        *args: Any,
        reflection_axis_radians: float,
        frame_dim: int = 48,
        history_steps: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reflection_axis_radians = float(reflection_axis_radians)
        self.frame_dim = int(frame_dim)
        self.history_steps = int(history_steps)
        expected = self.frame_dim * self.history_steps
        if self.observation_space.shape != (expected,):
            raise ValueError(
                "AntCommandReflectionReplayBuffer requires a flat observation "
                f"of shape ({expected},), got {self.observation_space.shape}"
            )
        if self.action_space.shape != (8,):
            raise ValueError(
                "AntCommandReflectionReplayBuffer requires 8 actions, got "
                f"{self.action_space.shape}"
            )

        angle = self.reflection_axis_radians
        axis = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        self._xy_reflection = 2.0 * np.outer(axis, axis) - np.eye(2, dtype=np.float32)

        # Joint order: FR hip/ankle, FL, RL, RR.  A planar reflection reverses
        # the axial +Z hip coordinate, while the diagonal ankle axes map with
        # unchanged coordinate sign in this MJCF.
        if np.allclose(self._xy_reflection, [[-1.0, 0.0], [0.0, 1.0]], atol=1e-6):
            # Reflection about forward (+Y): FR<->FL, RL<->RR.
            self._joint_source = np.array([2, 3, 0, 1, 6, 7, 4, 5])
        elif np.allclose(self._xy_reflection, [[0.0, -1.0], [-1.0, 0.0]], atol=1e-6):
            # Reflection about west-of-forward (-X,+Y): FR<->RL; FL/RR fixed.
            self._joint_source = np.array([4, 5, 2, 3, 0, 1, 6, 7])
        else:
            raise ValueError(
                "Only the approved forward (pi/2) and west-of-forward diagonal "
                "(3*pi/4) reflection axes are supported"
            )
        self._joint_sign = np.array([-1.0, 1.0] * 4, dtype=np.float32)
        self._augmentation_step = 0

    def reflect_actions(self, actions: np.ndarray) -> np.ndarray:
        values = np.asarray(actions)
        return values[..., self._joint_source] * self._joint_sign

    def reflect_observations(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations)
        original_shape = values.shape
        frames = values.reshape(*original_shape[:-1], self.history_steps, self.frame_dim)
        reflected = frames.copy()

        # Projected gravity is a polar vector.
        reflected[..., 0:2] = frames[..., 0:2] @ self._xy_reflection.T
        # Angular velocity is an axial vector: det(M) M w, det(M) = -1.
        reflected[..., 3:5] = frames[..., 3:5] @ (-self._xy_reflection).T
        reflected[..., 5] = -frames[..., 5]
        # Reflection reverses signed command-heading error.
        reflected[..., 6] = frames[..., 6]
        reflected[..., 7] = -frames[..., 7]

        for start in self._JOINT_BLOCK_STARTS:
            block = frames[..., start : start + 8]
            reflected[..., start : start + 8] = (
                block[..., self._joint_source] * self._joint_sign
            )
        return reflected.reshape(original_shape)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        obs_aug = np.asarray(obs).copy()
        next_obs_aug = np.asarray(next_obs).copy()
        action_aug = np.asarray(action).copy()

        n_envs = obs_aug.shape[0]
        mask = (np.arange(n_envs) + self._augmentation_step) % 2 == 0
        if np.any(mask):
            obs_aug[mask] = self.reflect_observations(obs_aug[mask])
            next_obs_aug[mask] = self.reflect_observations(next_obs_aug[mask])
            action_aug[mask] = self.reflect_actions(action_aug[mask])
        self._augmentation_step += 1
        super().add(obs_aug, next_obs_aug, action_aug, reward, done, infos)

