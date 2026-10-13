"""Device forces and contact reductions for Newton 1.6 Featherstone.

Contact normal forces use the same spring/damping law as Newton's
semi_implicit.kernels_contact.eval_body_contact (also used by Featherstone).
"""

import warp as wp


@wp.kernel
def tether_forces(
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    body_stride: int,
    a: int,
    b: int,
    anchor_a: wp.vec3,
    anchor_b: wp.vec3,
    slack: float,
    maximum: float,
    stiffness: float,
    damping: float,
    limit_ke: float,
    limit_kd: float,
    forces: wp.array[wp.spatial_vector],
    lengths: wp.array[float],
    tensions: wp.array[float],
):
    w = wp.tid()
    ia = w * body_stride + a
    ib = w * body_stride + b
    pa = wp.transform_point(poses[ia], anchor_a)
    pb = wp.transform_point(poses[ib], anchor_b)
    ra = pa - wp.transform_point(poses[ia], com[ia])
    rb = pb - wp.transform_point(poses[ib], com[ib])
    delta = pb - pa
    length = wp.length(delta)
    axis = delta / wp.max(length, 1.0e-8)
    va = wp.spatial_top(velocities[ia]) + wp.cross(
        wp.spatial_bottom(velocities[ia]), ra
    )
    vb = wp.spatial_top(velocities[ib]) + wp.cross(
        wp.spatial_bottom(velocities[ib]), rb
    )
    speed = wp.dot(vb - va, axis)
    # MuJoCo tendon damping acts even within its spring deadband.
    tension = stiffness * wp.max(length - slack, 0.0) + damping * speed
    # Unilateral compliant limit, calibrated from solreflimit and the compiled
    # reference effective mass. This is a penalty approximation, not MuJoCo's
    # coupled constraint solve. It never creates a compressive limit force.
    if length > maximum:
        tension += wp.max(limit_ke * (length - maximum) + limit_kd * speed, 0.0)
    force = tension * axis
    wp.atomic_add(forces, ia, wp.spatial_vector(force, wp.cross(ra, force)))
    wp.atomic_add(forces, ib, wp.spatial_vector(-force, wp.cross(rb, -force)))
    lengths[w] = length
    tensions[w] = tension


@wp.kernel
def contact_summary(
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    shape_body: wp.array[int],
    shape_world: wp.array[int],
    classes: wp.array[int],
    hands: wp.array[int],
    bars: wp.array[int],
    ke: wp.array[float],
    kd: wp.array[float],
    count: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    shape0: wp.array[int],
    shape1: wp.array[int],
    margin0: wp.array[float],
    margin1: wp.array[float],
    contact_ke: wp.array[float],
    contact_kd: wp.array[float],
    threshold: float,
    loads: wp.array3d[float],
    touching: wp.array2d[int],
    flags: wp.array2d[int],
):
    i = wp.tid()
    if i >= count[0]:
        return
    sa = shape0[i]
    sb = shape1[i]
    if sa < 0 or sb < 0 or sa == sb:
        return
    w = shape_world[sa]
    if w < 0:
        w = shape_world[sb]
    if w < 0:
        return
    ba = shape_body[sa]
    bb = shape_body[sb]
    n = -normals[i]
    pa = point0[i]
    pb = point1[i]
    va = wp.vec3(0.0)
    vb = wp.vec3(0.0)
    if ba >= 0:
        pa = wp.transform_point(poses[ba], pa) - margin0[i] * n
        ra = pa - wp.transform_point(poses[ba], com[ba])
        va = wp.spatial_top(velocities[ba]) + wp.cross(
            wp.spatial_bottom(velocities[ba]), ra
        )
    if bb >= 0:
        pb = wp.transform_point(poses[bb], pb) + margin1[i] * n
        rb = pb - wp.transform_point(poses[bb], com[bb])
        vb = wp.spatial_top(velocities[bb]) + wp.cross(
            wp.spatial_bottom(velocities[bb]), rb
        )
    d = wp.dot(n, pa - pb)
    ca = classes[sa]
    cb = classes[sb]
    if d < 0.0:
        if ((ca & 32) != 0 and (cb & 1) != 0) or ((cb & 32) != 0 and (ca & 1) != 0):
            wp.atomic_max(flags, w, 0, 1)
        if ((ca & 4) != 0 and (cb & 2) != 0) or ((cb & 4) != 0 and (ca & 2) != 0):
            wp.atomic_max(flags, w, 1, 1)
        if ((ca & 1) != 0 and (ca & 8) == 0 and (cb & 16) != 0) or (
            (cb & 1) != 0 and (cb & 8) == 0 and (ca & 16) != 0
        ):
            wp.atomic_max(flags, w, 2, 1)
        if ((ca & 64) != 0 and (cb & 2) != 0) or ((cb & 64) != 0 and (ca & 2) != 0):
            wp.atomic_max(flags, w, 3, 1)
    h = hands[sa]
    bar = bars[sb]
    if h < 0 or bar < 0:
        h = hands[sb]
        bar = bars[sa]
    if h >= 0 and bar >= 0 and d < threshold:
        wp.atomic_max(touching, w, h, 1)
        stiffness = 0.5 * (ke[sa] + ke[sb])
        damping = 0.5 * (kd[sa] + kd[sb])
        if contact_ke:
            if contact_ke[i] > 0.0:
                stiffness = contact_ke[i]
            if contact_kd[i] > 0.0:
                damping = contact_kd[i]
        normal = wp.max(
            -(d * stiffness + wp.min(wp.dot(n, va - vb), 0.0) * damping * wp.step(d)),
            0.0,
        )
        # Featherstone's nonadhesive contact law returns no force for d >= 0.
        if d >= 0.0:
            normal = 0.0
        wp.atomic_add(loads, w, h, bar, normal)
