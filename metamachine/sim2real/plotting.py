from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECTED_GRAVITY_RE = re.compile(r"frame\[(\d+)\]/projected_gravity\[(\d+)\]")
COMPONENT_NAMES = ["x", "y", "z"]
COMPONENT_COLORS = ["tab:red", "tab:green", "tab:blue"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _resolve_suite_path(summary_path: Path, summary: dict) -> Path:
    suite_path = Path(summary["suite_file"])
    if suite_path.exists():
        return suite_path
    candidate = (summary_path.parent / suite_path).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve suite file: {suite_path}")


def _extract_projected_gravity_indices(obs_labels: list[str]) -> dict[int, list[int]]:
    by_frame: dict[int, dict[int, int]] = {}
    for obs_idx, label in enumerate(obs_labels):
        match = PROJECTED_GRAVITY_RE.fullmatch(label)
        if match is None:
            continue
        frame_idx = int(match.group(1))
        component_idx = int(match.group(2))
        by_frame.setdefault(frame_idx, {})[component_idx] = obs_idx

    result: dict[int, list[int]] = {}
    for frame_idx, component_map in sorted(by_frame.items()):
        if all(component_idx in component_map for component_idx in range(3)):
            result[frame_idx] = [component_map[component_idx] for component_idx in range(3)]
    if not result:
        raise ValueError("No projected_gravity entries found in obs_labels")
    return result


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.array([0.0, 0.0, -1.0], dtype=np.float64)
    return vec / norm


def _rotation_align_vector(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = _normalize(np.asarray(src, dtype=np.float64))
    dst = _normalize(np.asarray(dst, dtype=np.float64))
    cross = np.cross(src, dst)
    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < 1e-8:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(src, axis))
        outer = np.outer(axis, axis)
        return 2.0 * outer - np.eye(3, dtype=np.float64)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / (cross_norm**2))


def _draw_frame(ax, rotation: np.ndarray, prefix: str, alpha: float, linewidth: float) -> None:
    basis = np.eye(3, dtype=np.float64)
    for axis_idx, color in enumerate(COMPONENT_COLORS):
        world_axis = rotation @ basis[:, axis_idx]
        ax.plot(
            [0.0, world_axis[0]],
            [0.0, world_axis[1]],
            [0.0, world_axis[2]],
            color=color,
            alpha=alpha,
            linewidth=linewidth,
        )
        ax.text(
            world_axis[0] * 1.1,
            world_axis[1] * 1.1,
            world_axis[2] * 1.1,
            f"{prefix}-{COMPONENT_NAMES[axis_idx]}",
            color=color,
            fontsize=8,
        )


def _set_equal_3d_axes(ax, limit: float = 1.1) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def _plot_module_case(
    *,
    module_id: int,
    module_idx: int,
    module_detail: dict,
    history_local_obs: np.ndarray,
    timestamps: np.ndarray,
    projected_gravity_indices: dict[int, list[int]],
    output_path: Path,
    case_name: str,
) -> float:
    actual = np.asarray(module_detail["actual_local_obs"], dtype=np.float64)
    expected = np.asarray(module_detail["expected_local_obs"], dtype=np.float64)
    relative_time = timestamps - timestamps[0]
    frame_indices = sorted(projected_gravity_indices.keys())
    frame0_indices = projected_gravity_indices[frame_indices[0]]

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])

    ax_time = fig.add_subplot(grid[0, :])
    frame0_history = history_local_obs[:, module_idx][:, frame0_indices]
    for component_idx, color in enumerate(COMPONENT_COLORS):
        ax_time.plot(
            relative_time,
            frame0_history[:, component_idx],
            color=color,
            linewidth=1.5,
            label=f"actual frame0 {COMPONENT_NAMES[component_idx]}",
        )
        ax_time.axhline(
            expected[frame0_indices[component_idx]],
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label=f"expected frame0 {COMPONENT_NAMES[component_idx]}",
        )
    ax_time.set_title(
        f"Module {module_id} projected_gravity history during {case_name}"
    )
    ax_time.set_xlabel("time since check start [s]")
    ax_time.set_ylabel("projected_gravity component")
    ax_time.grid(True, alpha=0.3)
    ax_time.legend(ncol=3, fontsize=8)

    ax_frames = fig.add_subplot(grid[1, 0])
    frame_positions = np.asarray(frame_indices, dtype=np.float64)
    for component_idx, color in enumerate(COMPONENT_COLORS):
        expected_values = [
            expected[projected_gravity_indices[frame_idx][component_idx]]
            for frame_idx in frame_indices
        ]
        actual_values = [
            actual[projected_gravity_indices[frame_idx][component_idx]]
            for frame_idx in frame_indices
        ]
        ax_frames.plot(
            frame_positions,
            expected_values,
            color=color,
            linestyle="--",
            marker="o",
            label=f"expected {COMPONENT_NAMES[component_idx]}",
        )
        ax_frames.plot(
            frame_positions,
            actual_values,
            color=color,
            linestyle="-",
            marker="x",
            label=f"actual {COMPONENT_NAMES[component_idx]}",
        )
    ax_frames.set_title("Final projected_gravity across history slots")
    ax_frames.set_xlabel("frame history index")
    ax_frames.set_ylabel("component value")
    ax_frames.grid(True, alpha=0.3)
    ax_frames.legend(fontsize=8, ncol=2)

    ax_frame = fig.add_subplot(grid[1, 1], projection="3d")
    gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    expected_vec = expected[frame0_indices]
    actual_vec = actual[frame0_indices]
    expected_rot = _rotation_align_vector(expected_vec, gravity_world)
    actual_rot = _rotation_align_vector(actual_vec, gravity_world)
    angle_deg = math.degrees(
        math.acos(
            float(
                np.clip(
                    np.dot(_normalize(expected_vec), _normalize(actual_vec)),
                    -1.0,
                    1.0,
                )
            )
        )
    )

    _draw_frame(ax_frame, expected_rot, "exp", alpha=0.9, linewidth=2.4)
    _draw_frame(ax_frame, actual_rot, "real", alpha=0.5, linewidth=1.6)
    ax_frame.plot(
        [0.0, gravity_world[0]],
        [0.0, gravity_world[1]],
        [0.0, gravity_world[2]],
        color="black",
        linewidth=2.0,
    )
    ax_frame.text(0.0, 0.0, -1.08, "gravity", color="black", fontsize=9)
    _set_equal_3d_axes(ax_frame)
    ax_frame.set_title(
        "Aligned gravity, showing implied frame offset\n"
        f"angle between projected_gravity vectors = {angle_deg:.1f} deg"
    )

    fig.suptitle(
        f"Sim-to-real projected gravity report: module {module_id}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return angle_deg


def _plot_joint_tracking_case(
    *,
    case_name: str,
    module_ids: list[int],
    timestamps: np.ndarray,
    sent_positions: np.ndarray,
    observed_dof_pos: np.ndarray,
    output_path: Path,
) -> None:
    relative_time = timestamps - timestamps[0]
    num_modules = len(module_ids)
    fig_height = max(3.0 * num_modules, 4.0)
    fig, axes = plt.subplots(
        num_modules,
        1,
        figsize=(14, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes[:, 0]

    for module_idx, module_id in enumerate(module_ids):
        ax = axes_flat[module_idx]
        target_pos = sent_positions[:, module_idx]
        actual_pos = observed_dof_pos[:, module_idx]
        tracking_err = actual_pos - target_pos

        ax.plot(
            relative_time,
            target_pos,
            color="tab:blue",
            linewidth=1.6,
            label="commanded target position",
        )
        ax.plot(
            relative_time,
            actual_pos,
            color="tab:orange",
            linewidth=1.3,
            label="measured joint position",
        )
        ax.fill_between(
            relative_time,
            target_pos,
            actual_pos,
            color="tab:red",
            alpha=0.12,
            label="tracking error",
        )
        ax.set_ylabel(f"M{module_id}\npos [rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        err_max = float(np.max(np.abs(tracking_err)))
        err_mean = float(np.mean(np.abs(tracking_err)))
        ax.set_title(
            f"Module {module_id}: max |err|={err_max:.3f}, mean |err|={err_mean:.3f}",
            fontsize=10,
        )

    axes_flat[-1].set_xlabel("time since check start [s]")
    fig.suptitle(
        f"Sim-to-real joint tracking report: {case_name}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_projected_gravity_plots(
    summary_json: Path,
    case_name: str | None = None,
) -> dict[str, object]:
    summary_path = summary_json.resolve()
    summary = _load_json(summary_path)
    suite = _load_json(_resolve_suite_path(summary_path, summary))
    projected_gravity_indices = _extract_projected_gravity_indices(
        list(suite["obs_labels"])
    )

    report_dir = summary_path.parent
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    timestamps_cache: dict[Path, np.ndarray] = {}
    history_cache: dict[Path, np.ndarray] = {}
    angle_summary: dict[str, dict[int, float]] = {}
    saved_files: list[str] = []

    for case in summary.get("cases", []):
        current_case_name = str(case["name"])
        if case_name is not None and current_case_name != case_name:
            continue

        history_path = Path(case["history_file"])
        if not history_path.is_absolute() and not history_path.exists():
            history_path = (summary_path.parent / history_path).resolve()
        if history_path not in history_cache:
            history = np.load(history_path)
            history_cache[history_path] = history["feedback_local_obs"]
            timestamps_cache[history_path] = history["timestamps"]

        feedback_local_obs = history_cache[history_path]
        timestamps = timestamps_cache[history_path]
        angle_summary[current_case_name] = {}

        for module_idx, module_detail in enumerate(case.get("modules", [])):
            module_id = int(module_detail["module_id"])
            output_path = figures_dir / f"{current_case_name}_module_{module_id}_projected_gravity.png"
            angle_deg = _plot_module_case(
                module_id=module_id,
                module_idx=module_idx,
                module_detail=module_detail,
                history_local_obs=feedback_local_obs,
                timestamps=timestamps,
                projected_gravity_indices=projected_gravity_indices,
                output_path=output_path,
                case_name=current_case_name,
            )
            angle_summary[current_case_name][module_id] = angle_deg
            saved_files.append(str(output_path))

    summary_path_out = figures_dir / "projected_gravity_angles.json"
    summary_path_out.write_text(json.dumps(angle_summary, indent=2) + "\n")
    saved_files.append(str(summary_path_out))
    return {
        "figures_dir": str(figures_dir),
        "saved_files": saved_files,
        "angle_summary": angle_summary,
        "angles_file": str(summary_path_out),
    }


def generate_joint_tracking_plots(
    summary_json: Path,
    case_name: str | None = None,
) -> dict[str, object]:
    summary_path = summary_json.resolve()
    summary = _load_json(summary_path)
    report_dir = summary_path.parent
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    tracking_summary: dict[str, dict[int, dict[str, float]]] = {}

    for case in summary.get("cases", []):
        current_case_name = str(case["name"])
        if case_name is not None and current_case_name != case_name:
            continue

        history_path = Path(case["history_file"])
        if not history_path.is_absolute() and not history_path.exists():
            history_path = (summary_path.parent / history_path).resolve()
        history = np.load(history_path)

        timestamps = np.asarray(history["timestamps"], dtype=np.float64)
        sent_positions = np.asarray(history["sent_positions"], dtype=np.float64)
        observed_dof_pos = np.asarray(history["observed_dof_pos"], dtype=np.float64)
        module_ids = [int(module["module_id"]) for module in case.get("modules", [])]

        output_path = figures_dir / f"{current_case_name}_joint_tracking.png"
        _plot_joint_tracking_case(
            case_name=current_case_name,
            module_ids=module_ids,
            timestamps=timestamps,
            sent_positions=sent_positions,
            observed_dof_pos=observed_dof_pos,
            output_path=output_path,
        )
        saved_files.append(str(output_path))

        tracking_summary[current_case_name] = {}
        for module_idx, module_id in enumerate(module_ids):
            err = observed_dof_pos[:, module_idx] - sent_positions[:, module_idx]
            tracking_summary[current_case_name][module_id] = {
                "max_abs_error": float(np.max(np.abs(err))),
                "mean_abs_error": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(np.square(err)))),
            }

    summary_path_out = figures_dir / "joint_tracking_summary.json"
    summary_path_out.write_text(json.dumps(tracking_summary, indent=2) + "\n")
    saved_files.append(str(summary_path_out))
    return {
        "figures_dir": str(figures_dir),
        "saved_files": saved_files,
        "tracking_summary": tracking_summary,
        "summary_file": str(summary_path_out),
    }


def generate_sim2real_report_plots(
    summary_json: Path,
    case_name: str | None = None,
) -> dict[str, object]:
    projected = generate_projected_gravity_plots(summary_json, case_name=case_name)
    tracking = generate_joint_tracking_plots(summary_json, case_name=case_name)
    saved_files = list(projected["saved_files"]) + [
        path for path in tracking["saved_files"] if path not in projected["saved_files"]
    ]
    return {
        "figures_dir": projected["figures_dir"],
        "saved_files": saved_files,
        "projected_gravity": projected,
        "joint_tracking": tracking,
    }
