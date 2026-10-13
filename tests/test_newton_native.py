"""Native/compat equivalence, GPU residency, batching and Tape smoke checks."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from metamachine.environments import make_env
from metamachine.environments.backends.newton import NewtonBackend
from metamachine.environments.backends.state import BackendCapabilityError
from metamachine.environments.configs.config_registry import ConfigRegistry

wp = pytest.importorskip("warp")
pytest.importorskip("newton")

ASSET = Path(__file__).parents[1] / "metamachine/assets/robots/rigid_body_probe.xml"


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    return request.param


def backend(mode, device, worlds=1, requires_grad=False):
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    return NewtonBackend(
        model, mode=mode, device=device, num_worlds=worlds, requires_grad=requires_grad
    )


def native_env(device, worlds=1):
    cfg = ConfigRegistry.create_from_name("rigid_body_probe")
    cfg.simulation.newton_mode = "native"
    cfg.simulation.newton_device = device
    cfg.environment.num_envs = worlds
    return make_env(cfg)


def assert_views_match(compat, native, atol=2e-6):
    np.testing.assert_allclose(
        native.state.joint_q.numpy(), compat.state.joint_q.numpy(), atol=atol
    )
    np.testing.assert_allclose(
        native.state.joint_qd.numpy(), compat.state.joint_qd.numpy(), atol=atol
    )
    body_q = native.state_view.body_transform.numpy()
    body_v = native.state_view.body_velocity.numpy()
    for mid, nid in compat.body_map.items():
        np.testing.assert_allclose(body_q[nid, :3], compat.data.xpos[mid], atol=atol)
        np.testing.assert_allclose(
            Rotation.from_quat(body_q[nid, 3:]).as_matrix(),
            compat.data.xmat[mid].reshape(3, 3),
            atol=atol,
        )
        com_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            compat.model, compat.data, mujoco.mjtObj.mjOBJ_BODY, mid, com_velocity, 0
        )
        np.testing.assert_allclose(
            body_v[nid], np.r_[com_velocity[3:], com_velocity[:3]], atol=atol
        )
    # Explicit root pose and free-joint velocity conversion at an offset COM.
    root_body = native.body_map[native.model.body("torso").id]
    rot = Rotation.from_quat(body_q[root_body, 3:]).as_matrix()
    origin_vel = body_v[root_body, :3] - np.cross(
        body_v[root_body, 3:], rot @ native.model.body_ipos[1]
    )
    np.testing.assert_allclose(origin_vel, compat.data.qvel[:3], atol=atol)
    np.testing.assert_allclose(
        rot.T @ body_v[root_body, 3:], compat.data.qvel[3:6], atol=atol
    )


def test_reset_and_1000_contact_enabled_physics_steps(device):
    compat, native = backend("compat", device), backend("native", device)
    q, v = compat.model.qpos0.copy(), np.array([0.1, -0.05, 0.0, 0.1, -0.2, 0.1, 0.05])
    q[3:7] = np.roll(Rotation.from_euler("xyz", [0.05, -0.1, 0.1]).as_quat(), 1)
    for b in (compat, native):
        b.reset(q, v)
    assert_views_match(compat, native)
    ctrl_device = wp.zeros(1, dtype=wp.float32, device=device)
    touched_ground = False
    for i in range(100):
        ctrl = np.array([0.03 * np.sin(i * 0.27)], dtype=np.float32)
        ctrl_device.assign(ctrl)
        compat.step(ctrl, 10)
        native.step(ctrl_device, 10)
        if i % 10 == 0 or i == 99:
            assert_views_match(compat, native, atol=3e-5)
            np.testing.assert_allclose(
                native.actuator_force.numpy(), compat.data.actuator_force, atol=1e-7
            )
            np.testing.assert_allclose(
                native.control.joint_f.numpy()[-1],
                compat.data.qfrc_actuator[-1],
                atol=1e-7,
            )
        touched_ground |= compat.data.qpos[2] < 0.15
    assert touched_ground
    assert np.isfinite(native.state.joint_q.numpy()).all()
    for b in (compat, native):
        b.reset(q, v)
    assert_views_match(compat, native)


def test_native_task_matches_existing_task_on_newton(device, tmp_path):
    cfg = ConfigRegistry.create_from_name("rigid_body_probe")
    cfg.simulation.newton_device = device
    cfg.logging.log_dir = str(tmp_path / "compat")
    compat = make_env(cfg)
    native = native_env(device)
    try:
        reset_obs, _ = compat.reset(seed=0)
        np.testing.assert_allclose(native.obs.numpy()[0], reset_obs, atol=1e-7)
        actions = wp.zeros((1, 1), dtype=wp.float32, device=device)
        for action in [0.2, -0.3, 1.7, -0.1] * 6:
            actions.assign(np.array([[action]], dtype=np.float32))
            obs, reward, done, truncated, _ = compat.step(
                np.array([action], dtype=np.float32)
            )
            native.step(actions)
            np.testing.assert_allclose(native.obs.numpy()[0], obs, atol=2e-5)
            np.testing.assert_allclose(native.reward.numpy()[0], reward, atol=2e-6)
            assert bool(native.terminated.numpy()[0]) == done
            assert bool(native.truncated.numpy()[0]) == truncated
        np.testing.assert_allclose(
            native.reset()[0].numpy()[0], compat.reset(seed=0)[0], atol=1e-7
        )
    finally:
        compat.close()
        native.close()


def test_batched_worlds_are_independent_and_selectively_reset(device):
    batch = native_env(device, 4)
    single = [native_env(device) for _ in range(4)]
    values = np.array([[0.2], [-0.15], [0.0], [0.4]], dtype=np.float32)
    actions = wp.array(values, dtype=wp.float32, device=device)
    one_actions = [
        wp.array(value.reshape(1, 1), dtype=wp.float32, device=device)
        for value in values
    ]
    for _ in range(70):  # Includes floor contact: no inter-world collisions.
        batch.step(actions)
        for env, act in zip(single, one_actions, strict=True):
            env.step(act)
    q = batch.state_view.joint_q.numpy().reshape(4, -1)
    obs = batch.obs.numpy()
    for i, env in enumerate(single):
        np.testing.assert_allclose(q[i], env.state_view.joint_q.numpy(), atol=3e-5)
        np.testing.assert_allclose(obs[i], env.obs.numpy()[0], atol=5e-5)
    batch.reset(mask=wp.array([0, 1, 0, 1], dtype=wp.int32, device=device))
    reset_q = batch.state_view.joint_q.numpy().reshape(4, -1)
    np.testing.assert_array_equal(reset_q[[0, 2]], q[[0, 2]])
    np.testing.assert_array_equal(batch.obs.numpy()[[0, 2]], obs[[0, 2]])
    initial = batch.physics.initial_q.numpy().reshape(4, -1)
    np.testing.assert_array_equal(reset_q[[1, 3]], initial[[1, 3]])
    np.testing.assert_array_equal(batch.steps.numpy(), [70, 0, 70, 0])


def test_native_step_reset_do_not_read_host_or_call_mujoco(device, monkeypatch):
    env = native_env(device, 4)
    actions = wp.zeros((4, 1), dtype=wp.float32, device=device)
    env.step(actions)  # Compile kernels and allocate Newton's lazy caches.

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Host read/synchronization or MuJoCo call in native rollout"
        )

    for name in (
        "mj_step",
        "mj_forward",
        "mj_kinematics",
        "mj_comPos",
        "mj_comVel",
        "mj_resetData",
    ):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(wp.array, "numpy", forbidden)
    for name in ("synchronize", "synchronize_device", "synchronize_stream"):
        monkeypatch.setattr(wp, name, forbidden)
    monkeypatch.setattr(np, "zeros", forbidden)
    for _ in range(5):
        env.step(actions)
    env.reset()
    with pytest.raises(BackendCapabilityError, match="no MjData"):
        _ = env.physics.data


def test_cuda_graph_replay_matches_eager():
    if not wp.is_cuda_available():
        pytest.skip("CUDA device unavailable")
    graph_env, eager = native_env("cuda:0", 8), native_env("cuda:0", 8)
    actions = wp.full((8, 1), 0.15, dtype=wp.float32, device="cuda:0")
    graph_env.step(actions)
    graph_env.reset()
    with wp.ScopedCapture(device="cuda:0") as capture:
        graph_env.step(actions)  # Probe has an even number of physics substeps.
    for _ in range(20):
        wp.capture_launch(capture.graph)
        eager.step(actions)
    np.testing.assert_allclose(
        graph_env.state_view.joint_q.numpy(),
        eager.state_view.joint_q.numpy(),
        atol=2e-6,
    )
    np.testing.assert_allclose(graph_env.obs.numpy(), eager.obs.numpy(), atol=2e-6)
    np.testing.assert_array_equal(graph_env.steps.numpy(), np.full(8, 20))


@wp.kernel
def final_joint_loss(q: wp.array[float], index: int, loss: wp.array[float]):
    loss[0] = q[index]


def test_functional_featherstone_tape_smoke(device):
    b = backend("native", device, requires_grad=True)
    buffers = [b.allocate_step() for _ in range(3)]
    assert len({id(buf.solver) for buf in buffers}) == len(buffers)
    controls = [
        wp.array([0.1], dtype=wp.float32, device=device, requires_grad=True)
        for _ in buffers
    ]
    loss = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    with wp.Tape() as tape:
        state = b.state
        for ctrl, out in zip(controls, buffers, strict=True):
            state = b.step_into(state, ctrl, out)
        wp.launch(
            final_joint_loss,
            dim=1,
            inputs=[state.joint_q, b.q_stride - 1],
            outputs=[loss],
            device=device,
        )
    tape.backward(loss)
    grads = [float(ctrl.grad.numpy()[0]) for ctrl in controls]
    assert np.isfinite(grads).all() and all(g > 0 for g in grads)
    # Independent finite difference on the first control, away from contacts.
    outputs = []
    for sign in (-1, 1):
        b.reset()
        controls[0].assign(np.array([0.1 + sign * 0.001], dtype=np.float32))
        state = b.state
        for ctrl, out in zip(controls, buffers, strict=True):
            state = b.step_into(state, ctrl, out)
        outputs.append(state.joint_q.numpy()[-1])
    np.testing.assert_allclose(
        grads[0], (outputs[1] - outputs[0]) / 0.002, rtol=0.02, atol=1e-7
    )
    with pytest.raises(BackendCapabilityError, match="step_into"):
        b.step(controls[0])


def test_native_capabilities_and_device_action_validation(device):
    env = native_env(device, 2)
    with pytest.raises(ValueError, match="Warp array"):
        env.step(np.zeros((2, 1)))
    actions = wp.array([[float("nan")], [0.1]], dtype=wp.float32, device=device)
    env.step(actions)
    np.testing.assert_array_equal(env.info["valid_actions"].numpy(), [0, 1])
    np.testing.assert_array_equal(env.terminated.numpy(), [1, 0])
    assert np.isfinite(env.state_view.joint_q.numpy()).all()
    cfg = ConfigRegistry.create_from_name("rigid_body_probe")
    cfg.simulation.newton_mode = "native"
    cfg.control.filter.enabled = True
    with pytest.raises(BackendCapabilityError, match="filter"):
        make_env(cfg)


def test_native_height_and_time_limit_boundaries(device, tmp_path):
    cfg = ConfigRegistry.create_from_name("rigid_body_probe")
    cfg.simulation.newton_device = device
    cfg.task.termination_conditions.height_threshold = 1.9995
    cfg.task.termination_conditions.max_episode_steps = 2
    cfg.logging.log_dir = str(tmp_path / "compat-boundaries")
    compat = make_env(cfg)
    cfg.simulation.newton_mode = "native"
    native = make_env(cfg)
    try:
        compat.reset(seed=0)
        actions = wp.zeros((1, 1), dtype=wp.float32, device=device)
        for expected in [(True, False), (True, True)]:
            reference = compat.step(np.zeros(1))
            native.step(actions)
            assert reference[2:4] == expected
            assert (
                bool(native.terminated.numpy()[0]),
                bool(native.truncated.numpy()[0]),
            ) == expected
        native.reset()
        np.testing.assert_array_equal(native.steps.numpy(), [0])
        np.testing.assert_array_equal(native.terminated.numpy(), [0])
        np.testing.assert_array_equal(native.truncated.numpy(), [0])
    finally:
        compat.close()
        native.close()


def test_multiple_motors_on_one_dof_and_fixed_gain(device):
    xml = ASSET.read_text().replace(
        "</actuator>",
        '<general name="second_motor" joint="hinge" gear="-3" gainprm="1.5"/>'
        "</actuator>",
    )
    model = mujoco.MjModel.from_xml_string(xml)
    native = NewtonBackend(model, mode="native", device=device)
    native.step(wp.array([2.0, 0.2], dtype=wp.float32, device=device))
    # First motor: clip ctrl, clip force, gear => 1.6 Nm.
    # Second motor: 0.2 * 1.5 * -3 => -0.9 Nm on the same DOF.
    np.testing.assert_allclose(native.actuator_force.numpy(), [0.8, 0.3], atol=1e-7)
    np.testing.assert_allclose(native.actuator_torque.numpy(), [1.6, -0.9], atol=1e-7)
    np.testing.assert_allclose(native.control.joint_f.numpy()[-1], 0.7, atol=2e-7)
