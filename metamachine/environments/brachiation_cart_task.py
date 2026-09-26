"""Opt-in cart-assisted task rules; archived hanging tasks retain their recipe."""
import jax
import jax.numpy as jp
import numpy as np
from mujoco.mjx._src import support
from . import brachiation_courses as courses

GUARD_INFO_KEYS = ("transfer_counts", "last_transfer_hand", "no_hook_steps", "no_transfer_steps", "reach_best")


def setup(env):
    m = env._mj_model
    env.cart_body = m.body("cart").id
    env.cart_v = int(m.joint("cart_root").dofadr[0])
    root_body = m.body("torso").id
    def descendant(body, ancestor):
        while body > 0:
            if body == ancestor:
                return True
            body = int(m.body_parentid[body])
        return False
    env.robot_geom_mask = jp.array([descendant(int(b), root_body) for b in m.geom_bodyid])
    env.cart_geom_mask = jp.array([descendant(int(b), env.cart_body) for b in m.geom_bodyid])
    env.limb_geom_mask = env.robot_geom_mask & jp.array(m.geom_bodyid != root_body)
    env.hook_geom_mask = jp.array(np.isin(np.arange(m.ngeom), np.asarray(env.hook_geoms)))
    env.bar_geom_mask = jp.array(np.isin(np.arange(m.ngeom), np.asarray(env.bar_geoms)))
    env.floor_geom = m.geom("floor").id
    env.torso_geom = m.geom("torso_geom").id
    env.deck_top = float(m.geom("cart_deck").pos[2] + m.geom("cart_deck").size[2])
    env.torso_half = jp.array(m.geom("torso_geom").size)
    env.cradle_inner = env.torso_half[:2] + float(env.cfg.cart.cradle.clearance)


def contact_flags(env, geom, dist):
    """Any penetrating forbidden contact counts, including brief substep contacts."""
    a, b = geom[:, 0], geom[:, 1]
    active = (dist < 0) & (a >= 0) & (b >= 0)
    floor = ((a == env.floor_geom) & env.robot_geom_mask[b]) | ((b == env.floor_geom) & env.robot_geom_mask[a])
    limb_cart = (env.limb_geom_mask[a] & env.cart_geom_mask[b]) | (env.limb_geom_mask[b] & env.cart_geom_mask[a])
    nonhook = env.robot_geom_mask & ~env.hook_geom_mask
    bracing = (nonhook[a] & env.bar_geom_mask[b]) | (nonhook[b] & env.bar_geom_mask[a])
    seated = ((a == env.torso_geom) & env.cart_geom_mask[b]) | ((b == env.torso_geom) & env.cart_geom_mask[a])
    return jp.stack([jp.any(active & x) for x in (floor, limb_cart, bracing, seated)])


def outside_cart(env, data):
    """Cart-relative limits: allow clearance and small bounce, reject escape/tipping."""
    local = (data.xpos[env.torso] - data.xpos[env.cart_body]) @ data.xmat[env.cart_body]
    c = env.cfg.task.cart_task
    sideways = jp.any(jp.abs(local[:2]) > env.cradle_inner + float(c.containment_margin))
    low = local[2] < env.deck_top + 0.5 * env.torso_half[2]
    high = local[2] > env.deck_top + env.torso_half[2] + float(c.max_torso_lift)
    tipped = data.xmat[env.cart_body, 2, 2] < jp.cos(float(c.cart_tilt_limit))
    return sideways | low | high | tipped


def hook_loads(env, data):
    """Normal force per hand/bar in N; force threshold excludes unloaded touches."""
    contact = data._impl.contact
    normal = jp.zeros(contact.dist.shape)
    for dim in np.unique(contact.dim):
        force, indices = support.contact_force_dim(env._mjx_model, data, int(dim))
        normal = normal.at[indices].set(jp.maximum(force[:, 0], 0.))
    a, b = contact.geom[:, 0], contact.geom[:, 1]
    active = contact.dist < float(env.cfg.task.contact_distance)
    def hand_load(geoms):
        ha = jp.any(a[:, None] == geoms, axis=1)
        hb = jp.any(b[:, None] == geoms, axis=1)
        pairs = ((ha[:, None] & (b[:, None] == env.bar_geoms)) |
                 (hb[:, None] & (a[:, None] == env.bar_geoms)))
        return jp.sum(jp.where(pairs & active[:, None], normal[:, None], 0.), axis=0)
    return jax.vmap(hand_load)(env.hook_geoms)


def hand_distances(env, data, target):
    if courses.is_spatial(env.cfg):
        return courses.hand_distances(env, data, target)
    delta = data.site_xpos[env.hands] - data.mocap_pos[target]
    dy = jp.maximum(jp.abs(delta[:, 1]) - float(env.cfg.bars.half_length), 0.)
    return jp.sqrt(delta[:, 0] ** 2 + delta[:, 2] ** 2 + dy ** 2)


def reach_potential(env, data, target, last_hand):
    distances = hand_distances(env, data, target)
    eligible = (last_hand < 0) | (jp.arange(2) != last_hand)
    distance = jp.min(jp.where(eligible, distances, jp.inf))
    return jp.clip(1. - distance / float(env.cfg.task.cart_task.reach_radius), 0., 1.)


def initial_info(env, data):
    return dict(transfer_counts=jp.zeros(2, dtype=jp.int32), last_transfer_hand=jp.array(-1, dtype=jp.int32),
                no_hook_steps=jp.array(0, dtype=jp.int32), no_transfer_steps=jp.array(0, dtype=jp.int32),
                reach_best=reach_potential(env, data, jp.array(1), jp.array(-1)))


def advance(env, info, distances, loads, touching, seated, unsafe):
    """Same-hand loaded hold, other hand released, alternation, and failure gating."""
    c = env.cfg.task.cart_task
    any_hook = jp.any(touching)
    no_hook = jp.where(any_hook, 0, info["no_hook_steps"] + 1)
    elapsed = info["no_transfer_steps"] + 1
    failure = unsafe | (no_hook >= int(c.max_no_hook_steps))
    eligible = (info["last_transfer_hand"] < 0) | (jp.arange(2) != info["last_transfer_hand"])
    valid = ((loads[:, info["target"]] >= float(c.min_hook_force)) &
             (distances < float(env.cfg.task.target_tolerance)) & ~touching[::-1] & eligible & seated &
             ~failure & ~info["course_complete"])
    if courses.is_spatial(env.cfg):
        # An intersecting junction touching two loaded bars is not a new grasp.
        other_bar_loaded = jp.any((loads > 0) & (jp.arange(loads.shape[1]) != info["target"]), axis=1)
        valid &= ~other_bar_loaded
    counts = jp.where(valid, info["transfer_counts"] + 1, 0)
    reached = jp.any(counts >= int(c.transfer_steps))
    stalled = (elapsed >= int(c.max_no_transfer_steps)) & ~reached
    failure |= stalled
    reached &= ~failure
    hand = jp.where(reached, jp.argmax(counts).astype(jp.int32), info["last_transfer_hand"])
    return reached, failure, dict(transfer_counts=jp.where(reached | failure, 0, counts),
                                 last_transfer_hand=hand, no_hook_steps=no_hook,
                                 no_transfer_steps=jp.where(reached, 0, elapsed)), stalled
