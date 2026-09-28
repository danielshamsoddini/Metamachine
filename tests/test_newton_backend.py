"""Semantic and physical comparisons, not contact-trajectory equivalence."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from metamachine.environments import make_env
from metamachine.environments.backends import MuJoCoBackend
from metamachine.environments.configs.config_registry import ConfigRegistry

ASSET = Path(__file__).parents[1] / "metamachine/assets/robots/rigid_body_probe.xml"


@pytest.fixture
def newton_class():
    pytest.importorskip(
        "newton", reason="install metamachine[newton] for physics comparisons"
    )
    from metamachine.environments.backends.newton import NewtonBackend

    return NewtonBackend


def make_backend(cls, xml=None):
    model = mujoco.MjModel.from_xml_string(xml or ASSET.read_text())
    backend = cls(model, mujoco.MjData(model))
    backend.reset(model.qpos0.copy(), np.zeros(model.nv))
    return backend


def test_mujoco_adapter_preserves_original_step():
    backend = make_backend(MuJoCoBackend)
    m = backend.model
    original = mujoco.MjData(m)
    mujoco.mj_forward(m, original)
    for ctrl in ([0.2], [-0.3], [1.4]):
        backend.step(ctrl, 10)
        original.ctrl[:] = ctrl
        mujoco.mj_step(m, original, nstep=10)
        mujoco.mj_rnePostConstraint(m, original)
        np.testing.assert_array_equal(backend.data.qpos, original.qpos)
        np.testing.assert_array_equal(backend.data.qvel, original.qvel)
        np.testing.assert_array_equal(
            backend.data.qfrc_actuator, original.qfrc_actuator
        )


@pytest.mark.parametrize("mass,length", [(2.0, 0.24), (3.1, 0.31)])
def test_compiled_morphology_and_joint_contract(newton_class, mass, length):
    xml = (
        ASSET.read_text()
        .replace('mass="2"', f'mass="{mass}"')
        .replace("0.24", str(length))
    )
    n = make_backend(newton_class, xml)
    m, nm = n.model, n.newton_model
    assert n.solver.__class__.__name__ == "SolverFeatherstone"
    assert (m.nq, m.nv, m.nu) == (8, 7, 1)
    for mid, nid in n.body_map.items():
        np.testing.assert_allclose(
            nm.body_mass.numpy()[nid], m.body_mass[mid], rtol=1e-6
        )
        np.testing.assert_allclose(
            nm.body_com.numpy()[nid], m.body_ipos[mid], atol=1e-7
        )
        rot = Rotation.from_quat(np.roll(m.body_iquat[mid], -1)).as_matrix()
        np.testing.assert_allclose(
            nm.body_inertia.numpy()[nid],
            rot @ np.diag(m.body_inertia[mid]) @ rot.T,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            n.state.body_q.numpy()[nid, :3], n.data.xpos[mid], atol=1e-7
        )
    mid = m.joint("hinge").id
    dof = n._vstarts[n.joint_map[mid]]
    np.testing.assert_allclose(nm.joint_armature.numpy()[dof], 0.01)
    np.testing.assert_allclose(nm.joint_damping.numpy()[dof], 0.03)
    np.testing.assert_allclose(
        [nm.joint_limit_lower.numpy()[dof], nm.joint_limit_upper.numpy()[dof]],
        [-1.2, 1.2],
    )


def test_rotated_free_joint_and_offset_com_round_trip(newton_class):
    n = make_backend(newton_class)
    q, v = n.model.qpos0.copy(), np.array([0.3, -0.2, 0.1, 0.8, -0.4, 0.6, 0.2])
    quat = Rotation.from_euler("xyz", [0.3, -0.5, 0.8]).as_quat()
    q[3:7], q[7] = np.roll(quat, 1), 0.2
    n.set_state(q, v)
    np.testing.assert_allclose(n.data.qpos, q, atol=2e-7)
    np.testing.assert_allclose(n.data.qvel, v, atol=2e-7)
    for mid, nid in n.body_map.items():
        np.testing.assert_allclose(
            n.state.body_q.numpy()[nid, :3], n.data.xpos[mid], atol=2e-7
        )
        # Independent world COM velocity from MuJoCo kinematics.
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            n.model, n.data, mujoco.mjtObj.mjOBJ_BODY, mid, velocity, 0
        )
        np.testing.assert_allclose(
            n.state.body_qd.numpy()[nid], np.r_[velocity[3:], velocity[:3]], atol=3e-7
        )


def test_zero_action_freefall(newton_class):
    engines = [make_backend(cls) for cls in (MuJoCoBackend, newton_class)]
    # 0.2 s in open air: compare analytic semi-implicit gravity integration.
    for engine in engines:
        engine.step(np.zeros(1), 200)
        dt, steps = engine.model.opt.timestep, 200
        expected_z = 2 - 9.81 * dt**2 * steps * (steps + 1) / 2
        np.testing.assert_allclose(engine.data.qpos[2], expected_z, atol=2e-5)
        np.testing.assert_allclose(engine.data.qvel[2], -9.81 * dt * steps, atol=1e-5)
        np.testing.assert_allclose(engine.data.qvel[[0, 1, 3, 4, 5, 6]], 0, atol=1e-5)
    np.testing.assert_allclose(engines[0].data.qpos, engines[1].data.qpos, atol=2e-5)


def test_fixed_control_sequence_in_air(newton_class):
    engines = [make_backend(cls) for cls in (MuJoCoBackend, newton_class)]
    for ctrl in [0.15, -0.25, 0.4, -0.1] * 5:
        for engine in engines:
            engine.step([ctrl], 10)
        # Small integration/damping differences are expected, even off contact.
        np.testing.assert_allclose(
            engines[0].data.qpos, engines[1].data.qpos, atol=2e-3
        )
        np.testing.assert_allclose(
            engines[0].data.qvel, engines[1].data.qvel, atol=1e-2
        )
        np.testing.assert_allclose(
            engines[0].data.qfrc_actuator, engines[1].data.qfrc_actuator, atol=1e-7
        )
    assert abs(engines[1].data.qpos[-1]) > 0.01


def test_actuator_clipping_gain_and_gear(newton_class):
    for cls in (MuJoCoBackend, newton_class):
        engine = make_backend(cls)
        engine.step([2.0], 1)
        np.testing.assert_allclose(engine.data.actuator_force, [0.8])
        np.testing.assert_allclose(engine.data.qfrc_actuator[-1], 1.6)


def test_newton_never_steps_mujoco(newton_class, monkeypatch):
    n = make_backend(newton_class)

    def forbidden(*args, **kwargs):
        raise AssertionError("MuJoCo dynamics called by Newton")

    monkeypatch.setattr(mujoco, "mj_step", forbidden)
    monkeypatch.setattr(mujoco, "mj_forward", forbidden)
    n.reset(n.model.qpos0, np.zeros(n.model.nv))
    n.step([0.2], 10)


def test_newton_reset_clears_episode_state(newton_class):
    n = make_backend(newton_class)
    q, v = n.model.qpos0.copy(), np.zeros(n.model.nv)
    n.step([0.3], 30)
    expected = n.data.qpos.copy(), n.data.qvel.copy()
    n.step([-0.8], 40)
    n.reset(q, v)
    assert n.data.time == 0
    np.testing.assert_array_equal(n.data.ctrl, 0)
    np.testing.assert_array_equal(n.control.joint_f.numpy(), 0)
    n.step([0.3], 30)
    np.testing.assert_array_equal(n.data.qpos, expected[0])
    np.testing.assert_array_equal(n.data.qvel, expected[1])


@pytest.fixture
def env_pair(newton_class, tmp_path):
    envs = []
    try:
        for mode in ("sim", "newton"):
            cfg = ConfigRegistry.create_from_name("rigid_body_probe")
            cfg.environment.mode = mode
            cfg.logging.log_dir = str(tmp_path / mode)
            envs.append(make_env(cfg))
        yield envs
    finally:
        for env in envs:
            env.close()


def test_task_actions_observations_rewards_and_reset(env_pair):
    first = [env.reset(seed=123)[0] for env in env_pair]
    np.testing.assert_allclose(first[0], first[1], atol=1e-7)
    assert first[0].shape == (27,)  # three frames: gravity, gyro, cos(q), qd, action
    for action in [0.2, -0.3, 0.5, -0.1] * 3:
        results = [env.step(np.array([action])) for env in env_pair]
        for env, (obs, reward, done, truncated, _info) in zip(
            env_pair, results, strict=True
        ):
            assert obs.shape == (27,) and np.isfinite(obs).all()
            np.testing.assert_allclose(env.filtered_target_flat, [action])
            frame = np.r_[
                env.state.derived.projected_gravity,
                env.state.get_observed_ang_vel_body(),
                np.cos(env.data.qpos[-1:]),
                env.data.qvel[-1:],
                [action],
            ]
            # MetaMachine history is oldest-first for this configuration.
            np.testing.assert_allclose(obs[-9:], frame, atol=1e-7)
            # The unchanged task reward uses its existing derived body velocity.
            forward = np.dot(
                env.state.accurate_vel_body, env.cfg.observation.projected_forward_vec
            )
            yaw = env.state.accurate_ang_vel_body[2]
            expected = 0.7 * np.exp(-((0.6 - forward) ** 2) / 0.15) + 0.3 * np.exp(
                -(yaw**2) / 0.15
            )
            np.testing.assert_allclose(reward, expected)
            assert not done and not truncated
        # MuJoCo's xquat/cvel are evaluated before its final 1 ms substep;
        # Newton publishes current kinematics. Allow 0.025 rad/s for gyro
        # timing, while checking joint state much more tightly below.
        np.testing.assert_allclose(results[0][0], results[1][0], atol=2.5e-2)
        np.testing.assert_allclose(
            env_pair[0].data.qpos, env_pair[1].data.qpos, atol=1e-4
        )
        np.testing.assert_allclose(
            env_pair[0].data.qvel, env_pair[1].data.qvel, atol=2e-3
        )
        np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-3)
    for env, expected in zip(env_pair, first, strict=True):
        np.testing.assert_allclose(env.reset(seed=123)[0], expected, atol=1e-7)
        assert env.step_count == 0 and env.termination_checker.current_step == 0
        assert env.reward_calculator.step_counter == 0


def test_task_height_termination_and_legacy_time_limit(env_pair):
    for env in env_pair:
        env.reset(seed=0)
        q = env.data.qpos.copy()
        q[2] = 0.2
        env.set_state(q, np.zeros(env.model.nv))
        assert env.step(np.zeros(1))[2:4] == (True, False)
        env.reset(seed=0)
        env.termination_checker.max_episode_steps = 2
        # Preserve the existing pre-increment time-limit check on both engines.
        assert [env.step(np.zeros(1))[3] for _ in range(3)] == [False, True, True]


def test_contact_smoke_checks_stability_not_matching_trajectories(newton_class):
    for cls in (MuJoCoBackend, newton_class):
        engine = make_backend(cls)
        engine.step([0.0], 1200)
        assert (
            np.isfinite(engine.data.qpos).all() and np.isfinite(engine.data.qvel).all()
        )
        assert 0.0 < engine.data.qpos[2] < 0.3
        assert np.linalg.norm(engine.data.qvel) < 3.0


@pytest.mark.parametrize(
    "edit,match",
    [
        (
            lambda x: x.replace('damping="0.03"', 'damping="0.03" frictionloss="0.1"'),
            "frictionloss",
        ),
        (
            lambda x: x.replace(
                'name="hinge" type="hinge"', 'name="hinge" type="hinge" ref="0.2"'
            ),
            "reference",
        ),
        (
            lambda x: x.replace(
                "</mujoco>", '<sensor><jointpos joint="hinge"/></sensor></mujoco>'
            ),
            "sensors",
        ),
    ],
)
def test_unsupported_model_features_fail(newton_class, edit, match):
    with pytest.raises(ValueError, match=match):
        make_backend(newton_class, edit(ASSET.read_text()))


def test_unsupported_task_features_fail_before_construction():
    cfg = ConfigRegistry.create_from_name("rigid_body_probe")
    cfg.task.termination_conditions.termination_strategy = "body_contact_floor"
    with pytest.raises(ValueError, match="contact-dependent"):
        make_env(cfg)


def test_repeated_config_lookup_retains_inheritance():
    for _ in range(2):
        cfg = ConfigRegistry.create_from_name("rigid_body_probe")
        assert len(cfg.observation.components) == 5


def test_density_based_morphology_is_compiled_once(newton_class):
    import xml.etree.ElementTree as ET

    root = ET.fromstring(ASSET.read_text())
    for body in root.findall(".//body"):
        body.remove(body.find("inertial"))
        body.find("geom").set("density", "730")
    n = make_backend(newton_class, ET.tostring(root, encoding="unicode"))
    assert not np.isclose(n.model.body_mass[1], 2.0)
    for mid, nid in n.body_map.items():
        np.testing.assert_allclose(
            n.newton_model.body_mass.numpy()[nid], n.model.body_mass[mid], rtol=1e-6
        )
        np.testing.assert_allclose(
            n.newton_model.body_com.numpy()[nid], n.model.body_ipos[mid], atol=1e-7
        )


def test_task_seeded_reset_and_action_clipping(env_pair):
    for env in env_pair:
        env.cfg.initialization.randomize_orientation = True
        env.cfg.randomization.init_joint_pos.enabled = True
        obs, _ = env.reset(seed=71)
        q = env.data.qpos.copy()
        env.step(np.array([2.0]))
        np.testing.assert_allclose(env.filtered_target_flat, [0.8])
        repeated, _ = env.reset(seed=71)
        np.testing.assert_allclose(env.data.qpos, q, atol=1e-7)
        np.testing.assert_allclose(repeated, obs, atol=1e-7)
        env.reset(seed=72)
        assert not np.allclose(env.data.qpos, q)
    np.testing.assert_allclose(env_pair[0].data.qpos, env_pair[1].data.qpos, atol=1e-7)


def test_rotated_initial_state_fixed_controls(newton_class):
    engines = [make_backend(cls) for cls in (MuJoCoBackend, newton_class)]
    q = engines[0].model.qpos0.copy()
    q[3:7] = np.roll(Rotation.from_euler("xyz", [0.3, -0.5, 0.8]).as_quat(), 1)
    q[7] = 0.2
    v = np.array([0.3, -0.2, 0.1, 0.8, -0.4, 0.6, 0.2])
    for engine in engines:
        engine.reset(q, v)
    for ctrl in [0.15, -0.25, 0.4, -0.1] * 3:
        for engine in engines:
            engine.step([ctrl], 10)
        np.testing.assert_allclose(
            engines[0].data.qpos, engines[1].data.qpos, atol=2e-3
        )
        np.testing.assert_allclose(
            engines[0].data.qvel, engines[1].data.qvel, atol=1e-2
        )
