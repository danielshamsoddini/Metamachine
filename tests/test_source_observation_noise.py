import numpy as np

from metamachine.environments.components.observation_noise import (
    SourceAwareObservationNoise,
)
from metamachine.utils.math_utils import quat_rotate_inverse
from metamachine.utils.math_utils import quat_apply


def _base_data() -> dict[str, np.ndarray]:
    return {
        "quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "quats": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (8, 1)),
        "ang_vel_body": np.array([0.1, -0.2, 0.3]),
        "gyros": np.zeros((8, 3)),
        "dof_pos": np.linspace(-0.4, 0.4, 8),
        "dof_vel": np.linspace(-1.0, 1.0, 8),
        "pos_world": np.array([1.0, 2.0, 3.0]),
    }


def _strong_config() -> dict:
    return {
        "mode": "source_aware",
        "orientation": {
            "episode_bias_uniform_deg": [2.5, 2.5, 5.0],
            "correlated_jitter_std_deg": [0.6, 0.6, 1.0],
            "correlation_time_seconds": 0.15,
            "random_walk_std_deg_per_sqrt_second": [0.05, 0.05, 0.20],
            "random_walk_limit_deg": [1.5, 1.5, 5.0],
        },
        "gyro": {
            "episode_bias_uniform_rad_per_sec": 0.05,
            "white_noise_std_rad_per_sec": 0.04,
        },
        "dof_pos": {
            "episode_bias_uniform_rad": 0.03,
            "white_noise_std_rad": 0.01,
        },
        "dof_vel": {
            "episode_bias_uniform_rad_per_sec": 0.10,
            "white_noise_std_rad_per_sec": 0.20,
        },
    }


def test_zero_strength_is_exact_and_does_not_mutate_sources():
    source = _base_data()
    originals = {key: value.copy() for key, value in source.items()}
    noise = SourceAwareObservationNoise(
        {"mode": "source_aware"},
        num_dof=8,
        dt=0.02,
        rng=np.random.default_rng(1),
    )
    result = noise.apply(source)

    for key in source:
        assert np.array_equal(result[key], originals[key])
        assert np.array_equal(source[key], originals[key])


def test_quaternion_is_normalized_and_shared_by_heading_and_gravity_sources():
    noise = SourceAwareObservationNoise(
        _strong_config(),
        num_dof=8,
        dt=0.02,
        rng=np.random.default_rng(2),
    )
    result = noise.apply(_base_data())

    assert np.isclose(np.linalg.norm(result["quat"]), 1.0, atol=1e-12)
    assert np.array_equal(result["quat"], result["quats"][0])
    projected_gravity = quat_rotate_inverse(
        result["quat"], np.array([0.0, 0.0, -1.0])
    )
    assert np.isclose(np.linalg.norm(projected_gravity), 1.0, atol=1e-12)


def test_episode_bias_persists_and_resamples():
    config = {
        "orientation": {"episode_bias_uniform_deg": [2.5, 2.5, 5.0]},
        "gyro": {"episode_bias_uniform_rad_per_sec": 0.05},
        "dof_pos": {"episode_bias_uniform_rad": 0.03},
        "dof_vel": {"episode_bias_uniform_rad_per_sec": 0.10},
    }
    noise = SourceAwareObservationNoise(
        config, num_dof=8, dt=0.02, rng=np.random.default_rng(3)
    )
    first = noise.apply(_base_data())
    second = noise.apply(_base_data())
    assert np.array_equal(first["quat"], second["quat"])
    assert np.array_equal(first["ang_vel_body"], second["ang_vel_body"])
    assert np.array_equal(first["dof_pos"], second["dof_pos"])
    assert np.array_equal(first["dof_vel"], second["dof_vel"])

    noise.reset()
    third = noise.apply(_base_data())
    assert not np.array_equal(first["quat"], third["quat"])
    assert not np.array_equal(first["dof_pos"], third["dof_pos"])


def test_seeded_sequences_are_reproducible_and_dynamic_errors_are_bounded():
    first = SourceAwareObservationNoise(
        _strong_config(),
        num_dof=8,
        dt=0.02,
        rng=np.random.default_rng(4),
    )
    second = SourceAwareObservationNoise(
        _strong_config(),
        num_dof=8,
        dt=0.02,
        rng=np.random.default_rng(4),
    )
    for _ in range(5000):
        first_result = first.apply(_base_data())
        second_result = second.apply(_base_data())
        for key in ("quat", "ang_vel_body", "dof_pos", "dof_vel"):
            assert np.array_equal(first_result[key], second_result[key])

    assert np.all(
        np.abs(first.orientation_jitter) <= 3.0 * first.orientation_jitter_std
    )
    assert np.all(
        np.abs(first.orientation_walk) <= first.orientation_walk_limit
    )
    assert np.all(np.abs(first.gyro_bias) <= first.gyro_bias_limit)
    assert np.all(np.abs(first.dof_pos_bias) <= first.dof_pos_bias_limit)
    assert np.all(np.abs(first.dof_vel_bias) <= first.dof_vel_bias_limit)


def test_world_yaw_rotation_preserves_minimal_observation_equivalence():
    def yaw_quaternion(angle: float) -> np.ndarray:
        return np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])

    def measured_heading(quaternion: np.ndarray) -> float:
        forward = quat_apply(quaternion, np.array([0.0, 1.0, 0.0]))
        return float(np.arctan2(forward[1], forward[0]))

    def wrap(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    first = SourceAwareObservationNoise(
        _strong_config(), num_dof=8, dt=0.02, rng=np.random.default_rng(5)
    )
    rotated = SourceAwareObservationNoise(
        _strong_config(), num_dof=8, dt=0.02, rng=np.random.default_rng(5)
    )
    world_rotation = 1.1

    initial = _base_data()
    initial["quat"] = yaw_quaternion(0.0)
    initial["quats"][0] = initial["quat"]
    rotated_initial = _base_data()
    rotated_initial["quat"] = yaw_quaternion(world_rotation)
    rotated_initial["quats"][0] = rotated_initial["quat"]
    noisy_initial = first.apply(initial)
    noisy_rotated_initial = rotated.apply(rotated_initial)

    current = _base_data()
    current["quat"] = yaw_quaternion(0.2)
    current["quats"][0] = current["quat"]
    rotated_current = _base_data()
    rotated_current["quat"] = yaw_quaternion(world_rotation + 0.2)
    rotated_current["quats"][0] = rotated_current["quat"]
    noisy_current = first.apply(current)
    noisy_rotated_current = rotated.apply(rotated_current)

    gravity = np.array([0.0, 0.0, -1.0])
    assert np.allclose(
        quat_rotate_inverse(noisy_current["quat"], gravity),
        quat_rotate_inverse(noisy_rotated_current["quat"], gravity),
        atol=1e-12,
    )
    heading_error = wrap(
        measured_heading(noisy_initial["quat"])
        - measured_heading(noisy_current["quat"])
    )
    rotated_heading_error = wrap(
        measured_heading(noisy_rotated_initial["quat"])
        - measured_heading(noisy_rotated_current["quat"])
    )
    assert np.isclose(heading_error, rotated_heading_error, atol=1e-12)
    for key in ("ang_vel_body", "dof_pos", "dof_vel"):
        assert np.array_equal(noisy_current[key], noisy_rotated_current[key])
