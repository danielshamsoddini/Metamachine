"""Opt-in spatial rigid-bar courses; seeded sampling works under JIT and vmap."""
import jax
import jax.numpy as jp
import numpy as np


def is_spatial(cfg):
    return cfg.bars.get('layout', 'parallel') == 'spatial'


def validate_course(cfg):
    layout = cfg.bars.get('layout', 'parallel')
    if layout not in ('parallel', 'spatial'):
        raise ValueError('bars.layout must be parallel or spatial')
    if layout == 'parallel':
        if 'spatial' in cfg.bars:
            raise ValueError('bars.spatial requires bars.layout: spatial')
        return
    if cfg.task.start != 'hanging_in_cart' or cfg.task.remove_cart:
        raise ValueError('This spatial variant retains the cart-supported start')
    if cfg.bars.geometry != 'capsule':
        raise ValueError('Spatial courses require capsule bars')
    c = cfg.bars.get('spatial', {})
    fields = {'yaw_degrees', 'tilt_degrees', 'intersection_probability', 'intersection_margin',
              'start_clearance', 'candidates', 'preview_seed'}
    if set(c) != fields:
        raise ValueError('bars.spatial requires ' + ', '.join(sorted(fields)))
    for name in ('yaw_degrees', 'tilt_degrees'):
        values = np.asarray(c[name], dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or values[0] > values[1] or np.any(np.abs(values) >= 90):
            raise ValueError(name + ' must be ordered finite bounds strictly between -90 and 90 degrees')
        if not values[0] <= 0 <= values[1]:
            raise ValueError(name + ' must include zero for the safe transverse fallback')
    for name in ('intersection_probability', 'intersection_margin', 'start_clearance'):
        value = c[name]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
            raise ValueError(name + ' must be finite')
    if not 0 <= c.intersection_probability <= 1:
        raise ValueError('intersection_probability must lie in [0, 1]')
    if not 0 < c.intersection_margin < cfg.bars.half_length:
        raise ValueError('intersection_margin must lie inside the bar half-length')
    if not 0 < c.start_clearance < cfg.bars.spacing[0]:
        raise ValueError('start_clearance must be positive and smaller than the minimum spacing')
    if isinstance(c.candidates, bool) or not isinstance(c.candidates, int) or not 1 <= c.candidates <= 128:
        raise ValueError('candidates must be an integer from 1 to 128')
    if isinstance(c.preview_seed, bool) or not isinstance(c.preview_seed, int) or not 0 <= c.preview_seed < 2**32:
        raise ValueError('preview_seed must be a uint32 integer')


def axis_from_quat(quat):
    """Canonical bar direction: body +Y rotated by the WXYZ mocap quaternion.

    The source capsule geom uses -Y; a segment's sign does not affect geometry.
    """
    w, x, y, z = jp.moveaxis(quat, -1, 0)
    return jp.stack([2 * (x*y - w*z), 1 - 2*(x*x + z*z), 2*(y*z + w*x)], axis=-1)


def quat_from_axis(axis):
    """Shortest rotation from +Y; sampled directions always have positive Y."""
    w = jp.sqrt((1 + axis[..., 1]) / 2)
    return jp.stack([w, axis[..., 2] / (2*w), jp.zeros_like(w), -axis[..., 0] / (2*w)], axis=-1)


def segment_distances(points, center, axis, half_length):
    delta = points - center
    along = jp.clip(jp.sum(delta * axis, axis=-1), -half_length, half_length)
    return jp.linalg.norm(delta - along[..., None] * axis, axis=-1)


def hand_distances(env, data, target):
    return segment_distances(data.site_xpos[env.hands], data.mocap_pos[target],
                             axis_from_quat(data.mocap_quat[target]), float(env.cfg.bars.half_length))


def sample_spatial(key, cfg, start_height):
    """Return centers, unit quaternions, and flags for deliberate previous-bar links.

    Centers advance by sampled X spacings. A linked candidate intersects the
    previous finite segment exactly, while respecting the same Y/Z bounds.
    Rejection is bounded; a transverse bar is the collision-safe fallback.
    The entire initial robot column is kept clear of every later bar.
    """
    b, c = cfg.bars, cfg.bars.spatial
    n, k = int(b.count), int(c.candidates)
    keys = jax.random.split(key, 4)
    gaps = jax.random.uniform(keys[0], (n-1,), minval=b.spacing[0], maxval=b.spacing[1])
    shared_height = jax.random.uniform(keys[1], (), minval=b.height[0], maxval=b.height[1])
    random = jax.random.uniform(keys[2], (n-1, k, 5))
    want_link = jax.random.uniform(keys[3], (n-1,)) < float(c.intersection_probability)
    half = float(b.half_length)
    link_half = half - float(c.intersection_margin)
    base_height = start_height + shared_height
    initial_pos = jp.array([0., 0., start_height])
    initial_axis = jp.array([0., 1., 0.])

    def clear_start(centers, axes):
        norm2 = jp.sum(axes[:, :2] ** 2, axis=-1)
        t = jp.clip(-jp.sum(centers[:, :2] * axes[:, :2], axis=-1) / jp.maximum(norm2, 1e-8), -half, half)
        nearest = centers[:, :2] + t[:, None] * axes[:, :2]
        return jp.sum(nearest**2, axis=-1) >= float(c.start_clearance)**2

    def one(previous, inputs):
        previous_center, previous_axis = previous
        index, gap, r, requested_link = inputs
        yaw = jp.deg2rad(c.yaw_degrees[0] + r[:, 0] * (c.yaw_degrees[1] - c.yaw_degrees[0]))
        tilt = jp.deg2rad(c.tilt_degrees[0] + r[:, 1] * (c.tilt_degrees[1] - c.tilt_degrees[0]))
        axes = jp.stack([jp.sin(yaw)*jp.cos(tilt), jp.cos(yaw)*jp.cos(tilt), jp.sin(tilt)], axis=-1)
        x = previous_center[0] + gap
        y = b.lateral_offset[0] + r[:, 2] * (b.lateral_offset[1] - b.lateral_offset[0])
        z = base_height + b.height_offset[0] + r[:, 3] * (b.height_offset[1] - b.height_offset[0])
        free = jp.stack([jp.full(k, x), y, z], axis=-1)
        previous_t = (2*r[:, 4] - 1) * link_half
        joint = previous_center + previous_t[:, None] * previous_axis
        safe_x = jp.where(jp.abs(axes[:, 0]) > 1e-6, axes[:, 0], 1e-6)
        current_t = (joint[:, 0] - x) / safe_x
        linked = (joint - current_t[:, None] * axes).at[:, 0].set(x)
        link_valid = ((jp.abs(current_t) <= link_half) &
                      (linked[:, 1] >= b.lateral_offset[0]) & (linked[:, 1] <= b.lateral_offset[1]) &
                      (linked[:, 2] >= base_height+b.height_offset[0]) & (linked[:, 2] <= base_height+b.height_offset[1]) &
                      clear_start(linked, axes))
        free_valid = clear_start(free, axes)
        use_link = (index > 1) & requested_link & jp.any(link_valid)
        valid = jp.where(use_link, link_valid, free_valid)
        choice = jp.argmax(valid)
        center = jp.where(use_link, linked[choice], free[choice])
        axis = axes[choice]
        # Fixed topology and no unbounded retry loop. This fallback is safe by validation.
        fallback = ~jp.any(valid)
        center = jp.where(fallback, jp.array([x, .5*(b.lateral_offset[0]+b.lateral_offset[1]), base_height + .5*(b.height_offset[0]+b.height_offset[1])]), center)
        axis = jp.where(fallback, initial_axis, axis)
        return (center, axis), (center, axis, use_link & ~fallback)

    _, (centers, axes, links) = jax.lax.scan(one, (initial_pos, initial_axis),
        (jp.arange(1, n), gaps, random, want_link))
    centers = jp.concatenate([initial_pos[None], centers])
    axes = jp.concatenate([initial_axis[None], axes])
    return centers, quat_from_axis(axes), jp.concatenate([jp.array([False]), links])
