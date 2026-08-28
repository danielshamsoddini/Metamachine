"""Physically structured observation noise for simulation environments."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _as_vector(value: Any, size: int, *, degrees: bool = False) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, size)
    if array.size != size:
        raise ValueError(f"Expected {size} noise values, got {array.size}")
    if np.any(array < 0.0):
        raise ValueError("Noise magnitudes must be non-negative")
    return np.deg2rad(array) if degrees else array


def _quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64)
    x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _rotation_vector_to_quat_xyzw(rotation_vector: np.ndarray) -> np.ndarray:
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64)
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotation_vector / angle
    half = 0.5 * angle
    return np.concatenate((axis * np.sin(half), [np.cos(half)]))


def _normalize_quat_xyzw(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


class SourceAwareObservationNoise:
    """Episode-persistent sensor biases plus bounded dynamic sensor errors.

    Orientation errors are composed in the torso/body sensor frame. The same
    perturbed torso quaternion is supplied as both ``quat`` and ``quats[0]`` so
    projected gravity and heading are derived from one coherent measurement.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        num_dof: int,
        dt: float,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.num_dof = int(num_dof)
        self.dt = float(dt)
        self.rng = rng
        if self.num_dof <= 0 or self.dt <= 0.0:
            raise ValueError("num_dof and dt must be positive")

        orientation = config.get("orientation", {})
        self.orientation_bias_limit = _as_vector(
            orientation.get("episode_bias_uniform_deg", [0.0, 0.0, 0.0]),
            3,
            degrees=True,
        )
        self.orientation_jitter_std = _as_vector(
            orientation.get("correlated_jitter_std_deg", [0.0, 0.0, 0.0]),
            3,
            degrees=True,
        )
        self.orientation_jitter_tau = float(
            orientation.get("correlation_time_seconds", 0.15)
        )
        if self.orientation_jitter_tau <= 0.0:
            raise ValueError("orientation correlation_time_seconds must be positive")
        self.orientation_walk_std = _as_vector(
            orientation.get(
                "random_walk_std_deg_per_sqrt_second", [0.0, 0.0, 0.0]
            ),
            3,
            degrees=True,
        )
        self.orientation_walk_limit = _as_vector(
            orientation.get("random_walk_limit_deg", [0.0, 0.0, 0.0]),
            3,
            degrees=True,
        )

        gyro = config.get("gyro", {})
        self.gyro_bias_limit = _as_vector(
            gyro.get("episode_bias_uniform_rad_per_sec", 0.0), 3
        )
        self.gyro_white_std = _as_vector(
            gyro.get("white_noise_std_rad_per_sec", 0.0), 3
        )

        dof_pos = config.get("dof_pos", {})
        self.dof_pos_bias_limit = _as_vector(
            dof_pos.get("episode_bias_uniform_rad", 0.0), self.num_dof
        )
        self.dof_pos_white_std = _as_vector(
            dof_pos.get("white_noise_std_rad", 0.0), self.num_dof
        )

        dof_vel = config.get("dof_vel", {})
        self.dof_vel_bias_limit = _as_vector(
            dof_vel.get("episode_bias_uniform_rad_per_sec", 0.0), self.num_dof
        )
        self.dof_vel_white_std = _as_vector(
            dof_vel.get("white_noise_std_rad_per_sec", 0.0), self.num_dof
        )

        all_magnitudes = np.concatenate(
            [
                self.orientation_bias_limit,
                self.orientation_jitter_std,
                self.orientation_walk_std,
                self.orientation_walk_limit,
                self.gyro_bias_limit,
                self.gyro_white_std,
                self.dof_pos_bias_limit,
                self.dof_pos_white_std,
                self.dof_vel_bias_limit,
                self.dof_vel_white_std,
            ]
        )
        self.has_nonzero_noise = bool(np.any(all_magnitudes > 0.0))
        self.reset()

    def reset(self) -> None:
        """Resample episode-persistent biases and clear dynamic noise state."""
        self.orientation_bias = self.rng.uniform(
            -self.orientation_bias_limit, self.orientation_bias_limit
        )
        self.gyro_bias = self.rng.uniform(-self.gyro_bias_limit, self.gyro_bias_limit)
        self.dof_pos_bias = self.rng.uniform(
            -self.dof_pos_bias_limit, self.dof_pos_bias_limit
        )
        self.dof_vel_bias = self.rng.uniform(
            -self.dof_vel_bias_limit, self.dof_vel_bias_limit
        )
        self.orientation_jitter = np.zeros(3, dtype=np.float64)
        self.orientation_walk = np.zeros(3, dtype=np.float64)

    def _clipped_normal(self, std: np.ndarray) -> np.ndarray:
        sample = self.rng.normal(0.0, std, size=std.shape)
        return np.clip(sample, -3.0 * std, 3.0 * std)

    def _sample_orientation_error(self) -> np.ndarray:
        alpha = float(np.exp(-self.dt / self.orientation_jitter_tau))
        innovation_std = self.orientation_jitter_std * np.sqrt(1.0 - alpha * alpha)
        self.orientation_jitter = (
            alpha * self.orientation_jitter + self._clipped_normal(innovation_std)
        )
        self.orientation_jitter = np.clip(
            self.orientation_jitter,
            -3.0 * self.orientation_jitter_std,
            3.0 * self.orientation_jitter_std,
        )

        walk_step_std = self.orientation_walk_std * np.sqrt(self.dt)
        self.orientation_walk += self._clipped_normal(walk_step_std)
        self.orientation_walk = np.clip(
            self.orientation_walk,
            -self.orientation_walk_limit,
            self.orientation_walk_limit,
        )
        return self.orientation_bias + self.orientation_jitter + self.orientation_walk

    def _perturb_quaternion(
        self, quaternion: np.ndarray, rotation_error: np.ndarray
    ) -> np.ndarray:
        error_quaternion = _rotation_vector_to_quat_xyzw(rotation_error)
        return _normalize_quat_xyzw(
            _quat_multiply_xyzw(np.asarray(quaternion), error_quaternion)
        )

    def apply(self, base_data: Mapping[str, Any]) -> dict[str, Any]:
        """Return policy-visible source fields with one new sensor sample."""
        if not self.has_nonzero_noise:
            return dict(base_data)

        noisy = dict(base_data)
        rotation_error = self._sample_orientation_error()
        torso_quaternion = None
        if "quat" in base_data:
            torso_quaternion = self._perturb_quaternion(
                np.asarray(base_data["quat"]), rotation_error
            )
            noisy["quat"] = torso_quaternion
        if "quats" in base_data:
            quaternions = np.array(base_data["quats"], copy=True)
            if quaternions.ndim == 1:
                quaternions = quaternions.reshape(1, -1)
            if len(quaternions):
                if torso_quaternion is None:
                    torso_quaternion = self._perturb_quaternion(
                        quaternions[0], rotation_error
                    )
                quaternions[0] = torso_quaternion
            noisy["quats"] = quaternions

        noisy_ang_vel = None
        if "ang_vel_body" in base_data:
            noisy_ang_vel = (
                np.asarray(base_data["ang_vel_body"], dtype=np.float64)
                + self.gyro_bias
                + self._clipped_normal(self.gyro_white_std)
            )
            noisy["ang_vel_body"] = noisy_ang_vel
        if "gyros" in base_data and noisy_ang_vel is not None:
            gyros = np.array(base_data["gyros"], copy=True)
            if gyros.ndim == 1:
                gyros = gyros.reshape(1, -1)
            if len(gyros):
                gyros[0] = noisy_ang_vel
            noisy["gyros"] = gyros

        if "dof_pos" in base_data:
            noisy["dof_pos"] = (
                np.asarray(base_data["dof_pos"], dtype=np.float64)
                + self.dof_pos_bias
                + self._clipped_normal(self.dof_pos_white_std)
            )
        if "dof_vel" in base_data:
            noisy["dof_vel"] = (
                np.asarray(base_data["dof_vel"], dtype=np.float64)
                + self.dof_vel_bias
                + self._clipped_normal(self.dof_vel_white_std)
            )
        return noisy
