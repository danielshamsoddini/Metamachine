"""Shared upright gait measurements for rewards and offline acceptance checks."""
from collections import deque
import numpy as np


def capsule_floor_clearance(model, data, ids):
    """Lowest capsule surface above the horizontal floor (not center height)."""
    import mujoco
    ids = np.asarray(ids, dtype=int)
    if np.any(model.geom_type[ids] != int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
        raise ValueError('foot clearance expects capsule ankle geoms')
    axial_z = np.abs(data.geom_xmat[ids].reshape(-1, 3, 3)[:, 2, 2])
    extent = model.geom_size[ids, 0] + model.geom_size[ids, 1] * axial_z
    floor_z = data.geom_xpos[model.geom('floor').id, 2]
    return np.maximum(0., data.geom_xpos[ids, 2] - extent - floor_z)


class CommandLeanWindow:
    """Signed mean torso-up vector; optionally include forward lean.

    The default preserves the original lateral/backward-only measurement.

    Inputs are world-frame vectors. Yaw does not rotate the command frame.
    Clear the window when the command changes, so old commands cannot cancel
    new-command lean. Each update represents one completed control interval.
    """
    def __init__(self, dt, seconds=2.0, include_forward=False, reset_on_command_change=True):
        if not np.isfinite([dt, seconds]).all() or dt <= 0 or seconds <= 0:
            raise ValueError('positive finite window and dt required')
        self.size = int(np.ceil(seconds / dt - 1e-10))
        self.history = deque(maxlen=self.size)
        self.command = None
        self.include_forward = bool(include_forward)
        self.reset_on_command_change = bool(reset_on_command_change)
        if not self.reset_on_command_change and not self.include_forward:
            raise ValueError('continuous lean windows require total tilt (include_forward=True)')

    def reset(self):
        self.history.clear()
        self.command = None

    def update(self, up_world, command_xy):
        up = np.asarray(up_world, dtype=float)
        cmd = np.asarray(command_xy, dtype=float)
        if up.shape != (3,) or cmd.shape != (2,) or not np.isfinite(np.r_[up, cmd]).all():
            raise ValueError('finite up[3] and command[2] required')
        if np.linalg.norm(cmd) < 1e-8 or np.linalg.norm(up) < 1e-8:
            raise ValueError('nonzero vectors required')
        cmd = cmd / np.linalg.norm(cmd)
        if self.reset_on_command_change and self.command is not None and np.dot(cmd, self.command) < 1 - 1e-8:
            self.reset()
        self.command = cmd.copy()
        self.history.append(up / np.linalg.norm(up))
        if len(self.history) < self.size:
            return None
        mean = np.mean(self.history, axis=0)
        lateral = np.dot(mean[:2], [-cmd[1], cmd[0]])
        along = float(np.dot(mean[:2], cmd))
        backward = along if self.include_forward else min(along, 0.0)
        return float(np.degrees(np.arctan2(np.hypot(lateral, backward), mean[2])))


def validate_gains(kp, kd, floor=28.0):
    kp, kd = np.asarray(kp, dtype=float), np.asarray(kd, dtype=float)
    if kp.shape != (8,) or kd.shape != (8,) or not np.isfinite(np.r_[kp, kd]).all():
        raise ValueError('eight finite gains required for kp and kd')
    if np.any(kp < floor) or np.any(kp > 500) or np.any(kd < 0) or np.any(kd > 5):
        raise ValueError('gains outside Kp floor/CyberGear protocol range')


def gait_summary(time, position, up, command, qvel, torque, contacts, clearance):
    """Control-rate diagnostics, explicitly not substep impact/torque peaks."""
    t = np.asarray(time, dtype=float)
    if len(t) < 3 or np.any(np.diff(t) <= 0):
        raise ValueError('at least three increasing timestamps required')
    arrays = [np.asarray(x, dtype=float) for x in (position, up, command, qvel, torque, contacts, clearance)]
    if any(len(x) != len(t) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('aligned finite telemetry required')
    position, up, command, qvel, torque, contacts, clearance = arrays
    dt = float(np.median(np.diff(t)))
    window = CommandLeanWindow(dt)
    lean = [window.update(u, c) for u, c in zip(up[1:], command[1:])]
    valid = np.array([v for v in lean if v is not None])
    total_window = CommandLeanWindow(dt, include_forward=True)
    total_lean = [total_window.update(u, [1., 0.]) for u in up[1:]]
    total_valid = np.array([v for v in total_lean if v is not None])
    delta = np.diff(position[:, :2], axis=0)
    cmd = command[1:] / np.linalg.norm(command[1:], axis=1)[:, None]
    along = np.sum(delta * cmd, axis=1)
    lateral = np.sum(delta * np.c_[-cmd[:, 1], cmd[:, 0]], axis=1)
    acceleration = np.diff(qvel, axis=0) / np.diff(t)[:, None]
    duration = float(t[-1] - t[0])
    touchdowns = np.sum(np.diff(contacts.astype(int), axis=0) > 0, axis=0)
    # Count completed short stance/swing intervals; omit censored endpoints.
    short, intervals = 0, 0
    for foot in contacts.T:
        changes = np.flatnonzero(np.diff(foot.astype(int))) + 1
        lengths = np.diff(t[changes])
        short += int(np.sum(lengths < .12 - 1e-8))
        intervals += len(lengths)
    return dict(duration_s=duration, commanded_speed_mps=float(along.sum()/duration),
                cross_track_abs_integral_m=float(np.abs(lateral).sum()),
                cross_track_max_m=float(np.max(np.abs(np.cumsum(lateral)))),
                path_error_deg=float(np.degrees(np.arctan2(abs(lateral.sum()), along.sum()))),
                sustained_away_lean_max_deg=float(valid.max()) if len(valid) else None,
                sustained_away_lean_pass=bool(len(valid) and np.all(valid <= 10.0 + 1e-8)),
                sustained_total_tilt_max_deg=float(total_valid.max()) if len(total_valid) else None,
                sustained_total_tilt_pass=bool(len(total_valid) and np.all(total_valid <= 10.0 + 1e-8)),
                joint_velocity_rms_rad_s=float(np.sqrt(np.mean(qvel**2))),
                joint_acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration**2))),
                touchdown_hz_per_foot=(touchdowns/duration).tolist(),
                short_contact_interval_fraction=float(short/intervals) if intervals else None,
                clearance_p95_m=np.percentile(clearance, 95, axis=0).tolist(),
                torque_rms_nm=float(np.sqrt(np.mean(torque**2))),
                torque_above_rated_fraction=float(np.mean(abs(torque) > 4)),
                sample_peak_torque_nm=float(np.max(abs(torque))),
                sampling_note='control-rate samples; peak impacts and torque may be missed')
