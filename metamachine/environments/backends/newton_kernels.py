"""Device-side motor mapping and reset. Warp is an optional import."""

import warp as wp


@wp.kernel
def map_motor_controls(
    ctrl: wp.array[float],
    offsets: wp.array[int],
    motor_ids: wp.array[int],
    ctrl_low: wp.array[float],
    ctrl_high: wp.array[float],
    gain: wp.array[float],
    force_low: wp.array[float],
    force_high: wp.array[float],
    gear: wp.array[float],
    num_motors: int,
    num_dofs: int,
    joint_f: wp.array[float],
    actuator_force: wp.array[float],
    actuator_torque: wp.array[float],
    valid: wp.array[int],
):
    world, dof = wp.tid()
    total = float(0.0)  # noqa: UP018 - Warp dynamic loop accumulator
    for entry in range(offsets[dof], offsets[dof + 1]):
        motor = motor_ids[entry]
        index = world * num_motors + motor
        value = ctrl[index]
        if not wp.isfinite(value):
            wp.atomic_min(valid, world, 0)
            value = 0.0
        force = wp.clamp(value, ctrl_low[motor], ctrl_high[motor]) * gain[motor]
        force = wp.clamp(force, force_low[motor], force_high[motor])
        torque = force * gear[motor]
        actuator_force[index] = force
        actuator_torque[index] = torque
        total += torque
    # One writer per DOF: deterministic even for multiple motors on one joint.
    joint_f[world * num_dofs + dof] = total


@wp.kernel
def reset_coordinates(
    initial: wp.array[float],
    mask: wp.array[int],
    stride: int,
    out: wp.array[float],
):
    i = wp.tid()
    if mask[i // stride] != 0:
        out[i] = initial[i]


@wp.kernel
def reset_world_values(mask: wp.array[int], stride: int, out: wp.array[float]):
    i = wp.tid()
    if mask[i // stride] != 0:
        out[i] = 0.0


@wp.kernel
def reset_valid_controls(mask: wp.array[int], valid: wp.array[int]):
    world = wp.tid()
    if mask[world] != 0:
        valid[world] = 1
