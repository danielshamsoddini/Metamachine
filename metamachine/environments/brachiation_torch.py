"""Batched device task mathematics; no simulator calls or host tensor reads."""

import math

import torch


def rotate(q, v):
    """Rotate vectors by XYZW quaternions, broadcasting leading dimensions."""
    qv, v = torch.broadcast_tensors(q[..., :3], v)
    t = 2 * torch.linalg.cross(qv, v, dim=-1)
    return v + q[..., 3:] * t + torch.linalg.cross(qv, t, dim=-1)


def inverse_rotate(q, v):
    return rotate(torch.cat((-q[..., :3], q[..., 3:]), -1), v)


def distances(hands, centers, axes, half, target, metric="bar_axis"):
    rows = torch.arange(hands.shape[0], device=hands.device)
    delta = hands - centers[rows, target][:, None]
    if metric == "bar_axis":
        axis = axes[rows, target][:, None]
        along = (delta * axis).sum(-1).clamp(-half[target, None], half[target, None])
        delta = delta - along[..., None] * axis
    return torch.linalg.vector_norm(delta, dim=-1)


def potential(hands, centers, axes, half, target, last_hand, radius):
    d = distances(hands, centers, axes, half, target)
    eligible = (last_hand[:, None] < 0) | (
        torch.arange(2, device=hands.device) != last_hand[:, None]
    )
    return (1 - torch.where(eligible, d, torch.inf).amin(-1) / radius).clamp(0, 1)


def advance(cfg, info, distance, loads, touching, seated, unsafe, spatial=False):
    c = cfg.task.cart_task
    no_hook = torch.where(touching.any(-1), 0, info["no_hook_steps"] + 1)
    elapsed = info["no_transfer_steps"] + 1
    failure = unsafe | (no_hook >= int(c.max_no_hook_steps))
    eligible = (info["last_transfer_hand"][:, None] < 0) | (
        torch.arange(2, device=loads.device) != info["last_transfer_hand"][:, None]
    )
    target_load = loads.gather(
        2, info["target"][:, None, None].expand(-1, 2, 1)
    ).squeeze(-1)
    valid = (
        (target_load >= float(c.min_hook_force))
        & (distance < float(cfg.task.target_tolerance))
        & ~touching.flip(-1)
        & eligible
        & seated[:, None]
        & ~failure[:, None]
        & ~info["course_complete"][:, None]
    )
    if spatial:
        other = (
            torch.arange(loads.shape[-1], device=loads.device)
            != info["target"][:, None]
        )
        valid &= ~((loads > 0) & other[:, None]).any(-1)
    counts = torch.where(valid, info["transfer_counts"] + 1, 0)
    reached = (counts >= int(c.transfer_steps)).any(-1)
    stalled = (elapsed >= int(c.max_no_transfer_steps)) & ~reached
    failure |= stalled
    reached &= ~failure
    hand = torch.where(reached, counts.argmax(-1), info["last_transfer_hand"])
    return (
        reached,
        failure,
        {
            "transfer_counts": torch.where((reached | failure)[:, None], 0, counts),
            "last_transfer_hand": hand,
            "no_hook_steps": no_hook,
            "no_transfer_steps": torch.where(reached, 0, elapsed),
        },
        stalled,
    )


def reward_terms(cfg, spec, context):
    c = context
    raw = {
        "bar_reached": c["reached"].float(),
        "reach_improvement": c["reach_improvement"],
        "unfinished_step": (~c["terminal"]).float(),
        "action_l2": c["action"].square().sum(-1),
        "action_rate": (c["action"] - c["last_action"]).square().sum(-1),
        "fall": c["fall"].float(),
        "torque_l2": c["mean_torque_l2"],
        "power_l1": c["mean_power_l1"],
        "joint_velocity_l2": c["joint_velocity"].square().sum(-1),
    }
    if spec.potential_shaping:
        raw["reach_potential"] = (
            float(cfg.training.discounting)
            * torch.where(c["terminal"], 0.0, c["potential_after"])
            - c["potential_before"]
        )
    total = torch.zeros_like(c["mean_torque_l2"])
    metrics = {}
    for term in spec.rewards:
        kind = term["type"]
        if kind == "forward_progress":
            value = (c["displacement"] * c["directions"][term["name"]]).sum(-1)
        elif kind == "target_distance":
            value = c["distance_improvement"][term["params"]["metric"]]
        else:
            value = raw[kind]
        value = torch.where(c["finite"], value, 0.0)
        if spec.cart_task and term["weight"] > 0 and kind != "reach_potential":
            value = torch.where(c["fall"], 0.0, value)
        weighted = (
            float(term["weight"])
            * value
            * (float(cfg.simulation.control_dt) if term["scale_by_dt"] else 1.0)
        )
        metrics["reward_raw/" + term["name"]] = value
        metrics["reward_weighted/" + term["name"]] = weighted
        total = total + weighted
    result = total if spec.reward_clip is None else total.clamp(*spec.reward_clip)
    result = torch.where(c["finite"], result, float(cfg.task.nonfinite_reward))
    metrics["reward_adjustment"] = result - total
    return result, metrics


def sample_parallel(cfg, n, start_height, rand):
    b = cfg.bars
    count = int(b.count)

    def uniform(shape, bounds):
        return rand(shape) * (float(bounds[1]) - float(bounds[0])) + float(bounds[0])

    gaps = uniform((n, count - 1), b.spacing)
    x = torch.cat((torch.zeros_like(gaps[:, :1]), gaps.cumsum(-1)), -1)
    y = uniform((n, count), b.lateral_offset)
    z = uniform((n, 1), b.height) + uniform((n, count), b.height_offset)
    if cfg.task.start == "hanging_in_cart":
        z = z + start_height
        z[:, 0] = start_height
    return torch.stack((x, y, z), -1)


def sample_spatial(cfg, n, start_height, rand):
    b, c = cfg.bars, cfg.bars.spatial
    count, k = int(b.count), int(c.candidates)
    half, link_half = float(b.half_length), float(b.half_length - c.intersection_margin)
    gaps = rand((n, count - 1)) * float(b.spacing[1] - b.spacing[0]) + float(
        b.spacing[0]
    )
    base = (
        start_height
        + rand((n, 1)) * float(b.height[1] - b.height[0])
        + float(b.height[0])
    )
    r = rand((n, count - 1, k, 5))
    wants = rand((n, count - 1)) < float(c.intersection_probability)
    initial = torch.zeros((n, 3), device=r.device)
    initial[:, 2] = start_height
    initial_axis = torch.zeros_like(initial)
    initial_axis[:, 1] = 1
    centers, axes = [initial], [initial_axis]

    def clear(p, a):
        t = (
            -(p[..., :2] * a[..., :2]).sum(-1)
            / a[..., :2].square().sum(-1).clamp_min(1e-8)
        ).clamp(-half, half)
        return (p[..., :2] + t[..., None] * a[..., :2]).square().sum(-1) >= float(
            c.start_clearance
        ) ** 2

    rows = torch.arange(n, device=r.device)
    for i in range(1, count):
        ri = r[:, i - 1]
        yaw = (
            float(c.yaw_degrees[0])
            + ri[..., 0] * float(c.yaw_degrees[1] - c.yaw_degrees[0])
        ) * (math.pi / 180)
        tilt = (
            float(c.tilt_degrees[0])
            + ri[..., 1] * float(c.tilt_degrees[1] - c.tilt_degrees[0])
        ) * (math.pi / 180)
        a = torch.stack(
            (yaw.sin() * tilt.cos(), yaw.cos() * tilt.cos(), tilt.sin()), -1
        )
        x = centers[-1][:, 0] + gaps[:, i - 1]
        y = float(b.lateral_offset[0]) + ri[..., 2] * float(
            b.lateral_offset[1] - b.lateral_offset[0]
        )
        z = (
            base
            + float(b.height_offset[0])
            + ri[..., 3] * float(b.height_offset[1] - b.height_offset[0])
        )
        free = torch.stack((x[:, None].expand(-1, k), y, z), -1)
        joint = (
            centers[-1][:, None]
            + ((2 * ri[..., 4] - 1) * link_half)[..., None] * axes[-1][:, None]
        )
        safe = torch.where(a[..., 0].abs() > 1e-6, a[..., 0], 1e-6)
        t = (joint[..., 0] - x[:, None]) / safe
        linked = joint - t[..., None] * a
        linked[..., 0] = x[:, None]
        valid_link = (
            (t.abs() <= link_half)
            & (linked[..., 1] >= float(b.lateral_offset[0]))
            & (linked[..., 1] <= float(b.lateral_offset[1]))
            & (linked[..., 2] >= base + float(b.height_offset[0]))
            & (linked[..., 2] <= base + float(b.height_offset[1]))
            & clear(linked, a)
        )
        use = (i > 1) & wants[:, i - 1] & valid_link.any(-1)
        valid = torch.where(use[:, None], valid_link, clear(free, a))
        choice = valid.long().argmax(-1)
        p = torch.where(use[:, None], linked[rows, choice], free[rows, choice])
        axis = a[rows, choice]
        fallback = torch.stack(
            (
                x,
                torch.full_like(
                    x, 0.5 * float(b.lateral_offset[0] + b.lateral_offset[1])
                ),
                base[:, 0] + 0.5 * float(b.height_offset[0] + b.height_offset[1]),
            ),
            -1,
        )
        centers.append(torch.where(valid.any(-1)[:, None], p, fallback))
        axes.append(torch.where(valid.any(-1)[:, None], axis, initial_axis))
    return torch.stack(centers, 1), torch.stack(axes, 1)
