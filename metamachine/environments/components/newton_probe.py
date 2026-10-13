"""Warp implementation of the existing rigid_body_probe task contract.

Topology, gains and reward settings are resolved once from MetaMachine config.
These kernels consume the backend's device arrays, not a MuJoCo data mirror.
"""

import warp as wp


@wp.struct
class ProbeParameters:
    num_actions: int
    q_stride: int
    qd_stride: int
    body_stride: int
    torso: int
    history: int
    max_steps: int
    gravity: wp.vec3
    forward: wp.vec3
    height: float
    obs_clip: float
    linear_target: float
    linear_sigma: float
    linear_weight: float
    angular_target: float
    angular_sigma: float
    angular_weight: float


@wp.kernel
def probe_controls(
    actions: wp.array2d[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    qids: wp.array[int],
    vids: wp.array[int],
    action_scale: wp.array[float],
    action_limit: wp.array[float],
    default_pos: wp.array[float],
    kp: wp.array[float],
    kd: wp.array[float],
    spec: ProbeParameters,
    last_action: wp.array2d[float],
    ctrl: wp.array[float],
    valid: wp.array[int],
):
    world, motor = wp.tid()
    value = actions[world, motor]
    if not wp.isfinite(value):
        wp.atomic_min(valid, world, 0)
        value = 0.0
    # Existing ActionProcessor scales before clipping, then adds default pose.
    value = wp.clamp(
        value * action_scale[motor], -action_limit[motor], action_limit[motor]
    )
    last_action[world, motor] = value
    target = value + default_pos[motor]
    pos = joint_q[world * spec.q_stride + qids[motor]]
    vel = joint_qd[world * spec.qd_stride + vids[motor]]
    ctrl[world * spec.num_actions + motor] = (
        kp[motor] * (target - pos) - kd[motor] * vel
    )


@wp.kernel
def probe_observe(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    tree_weights: wp.array[float],
    qids: wp.array[int],
    vids: wp.array[int],
    spec: ProbeParameters,
    reset_phase: int,
    reset_mask: wp.array[int],
    valid_controls: wp.array[int],
    valid_actions: wp.array[int],
    last_action: wp.array2d[float],
    steps: wp.array[int],
    obs: wp.array2d[float],
    reward: wp.array[float],
    terminated: wp.array[int],
    truncated: wp.array[int],
    components: wp.array2d[float],
):
    world = wp.tid()
    if reset_phase != 0 and reset_mask[world] == 0:
        return
    frame = 6 + 3 * spec.num_actions
    base = (spec.history - 1) * frame
    body = world * spec.body_stride + spec.torso
    rotation = wp.transform_get_rotation(body_q[body])
    position = wp.transform_get_translation(body_q[body])
    omega = wp.spatial_bottom(body_qd[body])
    gyro = wp.quat_rotate_inv(rotation, omega)
    gravity = wp.quat_rotate_inv(rotation, spec.gravity)
    finite = int(1)  # noqa: UP018 - Warp dynamic loop accumulator
    for qi in range(spec.q_stride):
        if not wp.isfinite(joint_q[world * spec.q_stride + qi]):
            finite = 0
    for vi in range(spec.qd_stride):
        if not wp.isfinite(joint_qd[world * spec.qd_stride + vi]):
            finite = 0
    if reset_phase != 0:
        steps[world] = 0
        valid_actions[world] = 1
        for motor in range(spec.num_actions):
            last_action[world, motor] = 0.0
    else:
        for i in range(base):
            obs[world, i] = obs[world, i + frame]
    for axis in range(3):
        obs[world, base + axis] = wp.clamp(gravity[axis], -spec.obs_clip, spec.obs_clip)
        obs[world, base + 3 + axis] = wp.clamp(
            gyro[axis], -spec.obs_clip, spec.obs_clip
        )
    for motor in range(spec.num_actions):
        q = joint_q[world * spec.q_stride + qids[motor]]
        v = joint_qd[world * spec.qd_stride + vids[motor]]
        obs[world, base + 6 + motor] = wp.clamp(
            wp.cos(q), -spec.obs_clip, spec.obs_clip
        )
        obs[world, base + 6 + spec.num_actions + motor] = wp.clamp(
            v, -spec.obs_clip, spec.obs_clip
        )
        obs[world, base + 6 + 2 * spec.num_actions + motor] = wp.clamp(
            last_action[world, motor], -spec.obs_clip, spec.obs_clip
        )
    if reset_phase != 0:
        for i in range(base):
            obs[world, i] = obs[world, base + i % frame]
        reward[world] = 0.0
        terminated[world] = 0
        truncated[world] = 0
        components[world, 0] = 0.0
        components[world, 1] = 0.0
    else:
        # Legacy data.cvel is referenced at the root subtree COM, whereas
        # Newton body_qd is at each body's own COM. Preserve that reward frame.
        tree_com = wp.vec3(0.0)
        for b in range(spec.body_stride):
            bid = world * spec.body_stride + b
            tree_com += tree_weights[b] * wp.transform_point(body_q[bid], body_com[bid])
        torso_com = wp.transform_point(body_q[body], body_com[body])
        velocity = wp.spatial_top(body_qd[body]) + wp.cross(omega, tree_com - torso_com)
        forward_speed = wp.dot(wp.quat_rotate_inv(rotation, velocity), spec.forward)
        yaw_speed = wp.dot(-gravity, gyro)
        linear_error = spec.linear_target - forward_speed
        angular_error = spec.angular_target - yaw_speed
        linear = spec.linear_weight * wp.exp(
            -linear_error * linear_error / spec.linear_sigma
        )
        angular = spec.angular_weight * wp.exp(
            -angular_error * angular_error / spec.angular_sigma
        )
        components[world, 0] = linear
        components[world, 1] = angular
        reward[world] = linear + angular
        terminated[world] = int(
            position[2] < spec.height
            or finite == 0
            or valid_actions[world] == 0
            or valid_controls[world] == 0
        )
        truncated[world] = int(
            spec.max_steps > 0 and steps[world] >= spec.max_steps - 1
        )
        steps[world] += 1
        if finite == 0:
            reward[world] = 0.0
            components[world, 0] = 0.0
            components[world, 1] = 0.0
