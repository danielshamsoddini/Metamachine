"""Readable observation and reward primitives selected by brachiation.yaml.

All coefficients, enabling, clipping, preview count, and history live in YAML.
A genuinely new primitive is added here and declared in brachiation_spec.py.
"""
import jax.numpy as jp


def obs_projected_gravity(env, data, info, params):
    return data.xmat[env.torso].T @ jp.array([0., 0., -1.])


def obs_lin_vel_body(env, data, info, params):
    return data.qvel[env.root_v:env.root_v + 3] @ data.xmat[env.torso]


def obs_ang_vel_body(env, data, info, params):
    return data.qvel[env.root_v + 3:env.root_v + 6]


def obs_dof_pos(env, data, info, params):
    return data.qpos[env.qids]


def obs_dof_vel(env, data, info, params):
    return data.qvel[env.vids]


def obs_hand_positions(env, data, info, params):
    value = data.site_xpos[env.hands] - data.xpos[env.torso]
    if params["frame"] == "body":
        value = value @ data.xmat[env.torso]
    return value.ravel()


def obs_target_bar_positions(env, data, info, params):
    ids = jp.minimum(info["target"] + jp.arange(params["count"]), int(env.cfg.bars.count) - 1)
    value = data.mocap_pos[ids] - data.xpos[env.torso]
    if params["frame"] == "body":
        value = value @ data.xmat[env.torso]
    return value.ravel()


def obs_target_bar_axes(env, data, info, params):
    from .brachiation_courses import axis_from_quat
    ids = jp.minimum(info["target"] + jp.arange(params["count"]), int(env.cfg.bars.count) - 1)
    value = axis_from_quat(data.mocap_quat[ids])
    if params["frame"] == "body":
        value = value @ data.xmat[env.torso]
    return value.ravel()


def obs_last_action(env, data, info, params):
    return info["last_action"]


def obs_hook_contacts(env, data, info, params):
    return env._hook_contacts(data).astype(jp.float32)


def obs_cart_relative_position(env, data, info, params):
    return (data.xpos[env.cart_body] - data.xpos[env.torso]) @ data.xmat[env.torso]


def obs_cart_relative_velocity(env, data, info, params):
    return (data.qvel[env.cart_v:env.cart_v + 3] - data.qvel[env.root_v:env.root_v + 3]) @ data.xmat[env.torso]


def obs_cart_projected_gravity(env, data, info, params):
    return data.xmat[env.cart_body].T @ jp.array([0., 0., -1.])


def obs_torso_height(env, data, info, params):
    return data.xpos[env.torso, 2:3]


def obs_pd_target(env, data, info, params):
    return env.controller.targets(info["last_action"], xp=jp)


def obs_tracking_error(env, data, info, params):
    return obs_pd_target(env, data, info, params) - data.qpos[env.qids]


OBSERVATION_FUNCTIONS = {
    name: globals()["obs_" + name] for name in (
        "projected_gravity", "lin_vel_body", "ang_vel_body", "dof_pos", "dof_vel",
        "hand_positions", "target_bar_positions", "target_bar_axes", "last_action", "hook_contacts",
        "cart_relative_position", "cart_relative_velocity", "cart_projected_gravity",
        "torso_height", "pd_target", "tracking_error")
}


def reward_forward_progress(env, ctx, params):
    delta = ctx["after"].xpos[env.torso] - ctx["before"].xpos[env.torso]
    return jp.dot(delta, jp.asarray(params["direction"]))


def reward_target_distance(env, ctx, params):
    return (env._distance(ctx["before"], ctx["target"], params["metric"])
            - env._distance(ctx["after"], ctx["target"], params["metric"]))


def reward_bar_reached(env, ctx, params):
    return ctx["reached"].astype(jp.float32)


def reward_reach_improvement(env, ctx, params):
    return ctx["reach_improvement"]


def reward_action_l2(env, ctx, params):
    return jp.sum(ctx["action"] ** 2)


def reward_action_rate(env, ctx, params):
    return jp.sum((ctx["action"] - ctx["last_action"]) ** 2)


def reward_fall(env, ctx, params):
    return ctx["fall"].astype(jp.float32)


def reward_torque_l2(env, ctx, params):
    return ctx["mean_torque_l2"]


def reward_power_l1(env, ctx, params):
    return ctx["mean_power_l1"]


def reward_joint_velocity_l2(env, ctx, params):
    return jp.sum(ctx["after"].qvel[env.vids] ** 2)


REWARD_FUNCTIONS = {
    name: globals()["reward_" + name] for name in (
        "forward_progress", "target_distance", "bar_reached", "reach_improvement", "action_l2",
        "action_rate", "fall", "torque_l2", "power_l1", "joint_velocity_l2")
}
