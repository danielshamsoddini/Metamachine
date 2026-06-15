from __future__ import annotations

import json
import math
import os
import re
from collections import deque
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOCAL_OBS_COMPONENT_DIMS: dict[str, int] = {
    "projected_gravity": 3,
    "gyro": 3,
    "dof_pos": 1,
    "dof_vel": 1,
}
LOCAL_OBS_TIME_AXIS_CHOICES = ("step", "logical", "wall")
DOF_POS_LABEL_RE = re.compile(r"frame\[(\d+)\]/dof_pos\[0\]")


def build_local_observation_labels(
    *,
    components: Iterable[str],
    frame_history_steps: int,
    latent_history_steps: int,
    latent_dim: int,
) -> list[str]:
    labels: list[str] = []
    component_list = list(components)
    for frame_idx in range(int(frame_history_steps)):
        for name in component_list:
            dim = LOCAL_OBS_COMPONENT_DIMS.get(name)
            if dim is None:
                raise ValueError(f"Unsupported local observation component: {name}")
            if dim == 1:
                labels.append(f"frame[{frame_idx}]/{name}[0]")
            else:
                for component_idx in range(dim):
                    labels.append(f"frame[{frame_idx}]/{name}[{component_idx}]")
    for latent_idx in range(int(latent_history_steps)):
        for dim_idx in range(int(latent_dim)):
            labels.append(f"latent[{latent_idx}][{dim_idx}]")
    return labels


class FirmwareStyleLocalObservationBuilder:
    """Rebuild the firmware/local-policy observation with explicit history buffers."""

    def __init__(
        self,
        *,
        num_modules: int,
        local_obs_spec,
        command_context_dim: int,
    ) -> None:
        self.num_modules = int(num_modules)
        self.local_obs_spec = local_obs_spec
        self.command_context_dim = int(command_context_dim)
        self.frame_history_steps = int(local_obs_spec.frame_history_steps)
        self.latent_history_steps = int(local_obs_spec.latent_history_steps)
        self.frame_dim = int(local_obs_spec.frame_dim)
        self.obs_dim = int(
            local_obs_spec.per_module_total_obs_dim(self.command_context_dim)
        )
        self._frame_ring: deque[np.ndarray] = deque(maxlen=self.frame_history_steps)
        self._command_context_ring: deque[np.ndarray] = deque(
            maxlen=max(self.latent_history_steps, 1)
        )
        self.reset()

    def reset(self) -> None:
        self._frame_ring.clear()
        self._command_context_ring.clear()

    def append(self, state, command_context: np.ndarray | None = None) -> np.ndarray:
        current_frame = self.local_obs_spec.extract_all_local_frames(state).astype(np.float32)
        if command_context is None:
            current_context = np.zeros(self.command_context_dim, dtype=np.float32)
        else:
            current_context = np.asarray(command_context, dtype=np.float32).flatten()
            if current_context.size < self.command_context_dim:
                current_context = np.pad(
                    current_context,
                    (0, self.command_context_dim - current_context.size),
                    mode="constant",
                )
            elif current_context.size > self.command_context_dim:
                current_context = current_context[: self.command_context_dim]

        if not self._frame_ring:
            for _ in range(self.frame_history_steps):
                self._frame_ring.append(current_frame.copy())
        else:
            self._frame_ring.append(current_frame.copy())

        if self.latent_history_steps > 0:
            if not self._command_context_ring:
                for _ in range(self.latent_history_steps):
                    self._command_context_ring.append(current_context.copy())
            else:
                self._command_context_ring.append(current_context.copy())

        frame_history = np.asarray(self._frame_ring, dtype=np.float32).transpose(1, 0, 2)
        local_obs = frame_history.reshape(self.num_modules, -1)
        if self.latent_history_steps > 0:
            context_history = np.asarray(self._command_context_ring, dtype=np.float32)
            context_history = np.broadcast_to(
                context_history[None, :, :],
                (self.num_modules, context_history.shape[0], context_history.shape[1]),
            )
            local_obs = np.concatenate(
                [local_obs, context_history.reshape(self.num_modules, -1)],
                axis=-1,
            )
        return local_obs.astype(np.float32, copy=False)


def _stringify_mode(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        if value.size == 1:
            return str(value.reshape(-1)[0])
    return str(value)


def _load_record(record_npz: Path) -> dict[str, np.ndarray]:
    data = np.load(record_npz, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _load_record_payload(record_npz: Path) -> dict[str, object]:
    record_path = Path(record_npz).resolve()
    record = _load_record(record_path)
    mode, times, local_obs, module_ids = _extract_recorded_local_obs(record)
    labels = _infer_local_obs_labels(record, int(local_obs.shape[-1]))
    return {
        "record_path": record_path,
        "record": record,
        "mode": mode,
        "times": times,
        "local_obs": local_obs,
        "module_ids": module_ids,
        "labels": labels,
    }


def _extract_recorded_local_obs(record: dict[str, np.ndarray]) -> tuple[str, np.ndarray, np.ndarray, list[int]]:
    mode = _stringify_mode(record["mode"])
    module_ids = [int(v) for v in np.asarray(record["module_ids"], dtype=np.int32).tolist()]

    if "assembled_local_obs" in record and record["assembled_local_obs"].size > 0:
        timestamps = np.asarray(record["timestamps"], dtype=np.float64)
        local_obs = np.asarray(record["assembled_local_obs"], dtype=np.float32)
        return mode, timestamps, local_obs, module_ids

    if "policy_debug_local_obs" in record and record["policy_debug_local_obs"].size > 0:
        policy_times = np.asarray(record["policy_feedback_pc_time"], dtype=np.float64)
        policy_module_ids = np.asarray(record["policy_feedback_module_ids"], dtype=np.int32)
        policy_obs = np.asarray(record["policy_debug_local_obs"], dtype=np.float32)
        valid = (
            np.asarray(record["policy_debug_valid"], dtype=np.int32)
            if "policy_debug_valid" in record
            else np.ones(policy_obs.shape[0], dtype=np.int32)
        )
        sample_count = min(
            int(policy_times.shape[0]),
            int(policy_module_ids.shape[0]),
            int(policy_obs.shape[0]),
            int(valid.shape[0]),
        )
        if sample_count <= 0:
            raise ValueError("No policy_debug_local_obs samples found in recording")
        policy_times = policy_times[:sample_count]
        policy_module_ids = policy_module_ids[:sample_count]
        policy_obs = policy_obs[:sample_count]
        valid = valid[:sample_count]

        per_module_times: list[np.ndarray] = []
        per_module_obs: list[np.ndarray] = []
        min_len = None
        for module_id in module_ids:
            mask = (policy_module_ids == module_id) & (valid > 0)
            times = policy_times[mask]
            obs = policy_obs[mask]
            per_module_times.append(times)
            per_module_obs.append(obs)
            if min_len is None:
                min_len = int(obs.shape[0])
            else:
                min_len = min(min_len, int(obs.shape[0]))

        if min_len is None or min_len <= 0:
            raise ValueError("No valid policy_debug_local_obs samples found in recording")

        stacked_obs = np.stack([obs[:min_len] for obs in per_module_obs], axis=1)
        stacked_times = np.asarray(per_module_times[0][:min_len], dtype=np.float64)
        return mode, stacked_times, stacked_obs, module_ids

    raise ValueError("Recording does not contain assembled_local_obs or policy_debug_local_obs")


def _infer_local_obs_labels(record: dict[str, np.ndarray], obs_dim: int) -> list[str]:
    components = [str(v) for v in np.asarray(record["local_obs_components"]).tolist()]
    frame_history_steps = int(np.asarray(record["local_obs_frame_history_steps"]).item())
    latent_history_steps = int(np.asarray(record["local_obs_latent_history_steps"]).item())
    frame_dim = sum(LOCAL_OBS_COMPONENT_DIMS[name] for name in components)
    frame_obs_dim = frame_dim * frame_history_steps
    latent_dim = 0
    if latent_history_steps > 0 and obs_dim > frame_obs_dim:
        latent_dim = max(0, int((obs_dim - frame_obs_dim) / latent_history_steps))
    labels = build_local_observation_labels(
        components=components,
        frame_history_steps=frame_history_steps,
        latent_history_steps=latent_history_steps,
        latent_dim=latent_dim,
    )
    return labels[:obs_dim]


def _resolve_plot_axis(
    *,
    times: np.ndarray,
    num_samples: int,
    time_axis: str,
    local_dt: float | None,
) -> tuple[np.ndarray, str]:
    axis = str(time_axis).lower()
    if axis not in LOCAL_OBS_TIME_AXIS_CHOICES:
        raise ValueError(
            f"Unsupported time axis {time_axis!r}; expected one of {LOCAL_OBS_TIME_AXIS_CHOICES}"
        )

    if axis == "step":
        return np.arange(num_samples, dtype=np.float64), "local control step"

    if axis == "logical":
        if local_dt is None or float(local_dt) <= 0:
            raise ValueError("logical time axis requires local_dt > 0")
        return np.arange(num_samples, dtype=np.float64) * float(local_dt), "logical local time [s]"

    relative_time = np.asarray(times, dtype=np.float64) - float(times[0])
    return relative_time, "time since recording start [s]"


def _plot_local_obs_heatmap(
    *,
    x_axis: np.ndarray,
    x_label: str,
    local_obs: np.ndarray,
    labels: list[str],
    module_id: int,
    title_prefix: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    x0 = float(x_axis[0]) if x_axis.size > 0 else 0.0
    x1 = float(x_axis[-1]) if x_axis.size > 1 else x0 + 1e-6
    image = ax.imshow(
        local_obs.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[x0, x1, 0, local_obs.shape[1]],
        cmap="coolwarm",
    )
    tick_count = min(len(labels), 20)
    if tick_count > 0:
        tick_indices = np.linspace(0, len(labels) - 1, tick_count, dtype=int)
        ax.set_yticks(tick_indices + 0.5)
        ax.set_yticklabels([labels[idx] for idx in tick_indices], fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel("local observation index")
    ax.set_title(f"{title_prefix}: module {module_id} local observation heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_local_obs_trace_chunks(
    *,
    x_axis: np.ndarray,
    x_label: str,
    local_obs: np.ndarray,
    labels: list[str],
    module_id: int,
    title_prefix: str,
    output_dir: Path,
    desired_target_history: np.ndarray | None = None,
    chunk_size: int = 8,
) -> list[str]:
    saved: list[str] = []
    num_dims = int(local_obs.shape[1])
    num_chunks = max(1, math.ceil(num_dims / chunk_size))
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(num_dims, start + chunk_size)
        fig, axes = plt.subplots(end - start, 1, figsize=(14, max(2.4 * (end - start), 4.0)), sharex=True, squeeze=False)
        axes_flat = axes[:, 0]
        for axis_offset, obs_idx in enumerate(range(start, end)):
            ax = axes_flat[axis_offset]
            ax.plot(x_axis, local_obs[:, obs_idx], linewidth=1.2, color="tab:blue")
            match = DOF_POS_LABEL_RE.fullmatch(labels[obs_idx])
            if match is not None and desired_target_history is not None:
                history_idx = int(match.group(1))
                if history_idx < desired_target_history.shape[1]:
                    ax.plot(
                        x_axis,
                        desired_target_history[:, history_idx],
                        linewidth=1.1,
                        linestyle="--",
                        color="black",
                        label="desired",
                    )
            ax.set_ylabel(labels[obs_idx], fontsize=8)
            ax.grid(True, alpha=0.3)
            if axis_offset == 0 and match is not None and desired_target_history is not None:
                ax.legend(loc="upper right", fontsize=8)
        axes_flat[-1].set_xlabel(x_label)
        fig.suptitle(
            f"{title_prefix}: module {module_id} local observation traces [{start}:{end}]",
            fontsize=13,
        )
        fig.tight_layout()
        output_path = output_dir / f"module_{module_id}_local_obs_traces_{chunk_idx:02d}.png"
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        saved.append(str(output_path))
    return saved


def _extract_recorded_scalar_series_by_module(
    record: dict[str, np.ndarray],
    *,
    value_key: str,
    module_ids: list[int],
) -> np.ndarray | None:
    if value_key not in record or np.asarray(record[value_key]).size == 0:
        return None
    if "policy_feedback_module_ids" not in record:
        return None

    policy_module_ids = np.asarray(record["policy_feedback_module_ids"], dtype=np.int32)
    values = np.asarray(record[value_key], dtype=np.float32).reshape(-1)
    valid = (
        np.asarray(record["policy_debug_valid"], dtype=np.int32).reshape(-1)
        if "policy_debug_valid" in record and np.asarray(record["policy_debug_valid"]).size > 0
        else np.ones(values.shape[0], dtype=np.int32)
    )

    sample_count = min(
        int(policy_module_ids.shape[0]),
        int(values.shape[0]),
        int(valid.shape[0]),
    )
    if sample_count <= 0:
        return None
    policy_module_ids = policy_module_ids[:sample_count]
    values = values[:sample_count]
    valid = valid[:sample_count]

    per_module: list[np.ndarray] = []
    min_len = None
    for module_id in module_ids:
        mask = (policy_module_ids == int(module_id)) & (valid > 0)
        module_values = values[mask]
        if module_values.size == 0:
            return None
        per_module.append(module_values.astype(np.float32, copy=False))
        min_len = int(module_values.shape[0]) if min_len is None else min(
            min_len, int(module_values.shape[0])
        )

    if min_len is None or min_len <= 0:
        return None
    return np.stack([series[:min_len] for series in per_module], axis=1).astype(
        np.float32,
        copy=False,
    )


def _infer_local_debug_scenario_config(
    payload: dict[str, object],
) -> object | None:
    from hierarchical.local_debug_scenarios import (
        LOCAL_DEBUG_PROTOCOL_VERSION,
        LocalDebugScenarioConfig,
        SCENARIO_BY_ID,
    )

    labels = list(payload["labels"])
    local_obs = np.asarray(payload["local_obs"], dtype=np.float32)
    if local_obs.ndim != 3 or local_obs.shape[0] <= 0 or local_obs.shape[1] <= 0:
        return None

    label_to_index = {label: idx for idx, label in enumerate(labels)}
    required = [f"latent[0][{idx}]" for idx in range(6)]
    if any(label not in label_to_index for label in required):
        return None

    module0 = local_obs[0, 0]
    version = float(module0[label_to_index["latent[0][0]"]])
    if abs(version - float(LOCAL_DEBUG_PROTOCOL_VERSION)) > 1e-3:
        return None

    scenario_id = int(round(float(module0[label_to_index["latent[0][1]"]])))
    if scenario_id not in SCENARIO_BY_ID:
        return None

    amplitude = float(module0[label_to_index["latent[0][2]"]])
    frequency_hz = float(module0[label_to_index["latent[0][3]"]])
    phase_offset_rad = float(module0[label_to_index["latent[0][4]"]])
    bias = float(module0[label_to_index["latent[0][5]"]])
    return LocalDebugScenarioConfig(
        name=SCENARIO_BY_ID[scenario_id].name,
        amplitude=amplitude,
        frequency_hz=frequency_hz,
        phase_offset_deg=math.degrees(phase_offset_rad),
        bias=bias,
    )


def _reconstruct_desired_targets_from_local_debug_scenario(
    payload: dict[str, object],
    *,
    local_dt: float,
) -> np.ndarray | None:
    from hierarchical.local_debug_scenarios import evaluate_action_vector

    scenario_cfg = _infer_local_debug_scenario_config(payload)
    if scenario_cfg is None:
        return None

    local_obs = np.asarray(payload["local_obs"], dtype=np.float32)
    num_samples = int(local_obs.shape[0])
    num_modules = int(local_obs.shape[1])
    joint_offsets = np.asarray(payload["record"]["joint_offsets"], dtype=np.float32).reshape(-1)
    if joint_offsets.size < num_modules:
        joint_offsets = np.pad(
            joint_offsets,
            (0, num_modules - joint_offsets.size),
            mode="edge" if joint_offsets.size > 0 else "constant",
        )
    joint_offsets = joint_offsets[:num_modules]

    desired = np.zeros((num_samples, num_modules), dtype=np.float32)
    for step_idx in range(num_samples):
        raw_action = evaluate_action_vector(
            scenario_cfg,
            t_sec=float(step_idx) * float(local_dt),
            num_modules=num_modules,
        )
        desired[step_idx] = raw_action + joint_offsets
    return desired


def _extract_desired_target_series(
    payload: dict[str, object],
    *,
    local_dt: float,
) -> tuple[np.ndarray | None, str | None]:
    record = payload["record"]
    module_ids = [int(v) for v in payload["module_ids"]]

    motor_target = _extract_recorded_scalar_series_by_module(
        record,
        value_key="policy_debug_motor_target",
        module_ids=module_ids,
    )
    if motor_target is not None:
        return motor_target, "policy_debug_motor_target"

    reconstructed = _reconstruct_desired_targets_from_local_debug_scenario(
        payload,
        local_dt=local_dt,
    )
    if reconstructed is not None:
        return reconstructed, "local_debug_scenario_reconstructed"

    return None, None


def _build_desired_target_history(
    current_targets: np.ndarray,
    *,
    frame_history_steps: int,
) -> np.ndarray:
    current_targets = np.asarray(current_targets, dtype=np.float32)
    num_samples, num_modules = current_targets.shape
    history_ring: deque[np.ndarray] = deque(maxlen=int(frame_history_steps))
    stacked = np.zeros((num_samples, num_modules, int(frame_history_steps)), dtype=np.float32)

    for sample_idx in range(num_samples):
        target = current_targets[sample_idx]
        if not history_ring:
            for _ in range(int(frame_history_steps)):
                history_ring.append(target.copy())
        else:
            history_ring.append(target.copy())
        stacked[sample_idx] = np.asarray(history_ring, dtype=np.float32).transpose(1, 0)
    return stacked


def _comparison_output_dir(
    sim_record_path: Path,
    real_record_path: Path,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    sim_stem = sim_record_path.with_suffix("").name
    real_stem = real_record_path.with_suffix("").name
    return real_record_path.parent / f"{sim_stem}__vs__{real_stem}_figures"


def _plot_local_obs_comparison_trace_chunks(
    *,
    sim_x_axis: np.ndarray,
    sim_local_obs: np.ndarray,
    real_x_axis: np.ndarray,
    real_local_obs: np.ndarray,
    x_label: str,
    labels: list[str],
    slot_idx: int,
    sim_module_id: int,
    real_module_id: int,
    output_dir: Path,
    sim_desired_target_history: np.ndarray | None = None,
    real_desired_target_history: np.ndarray | None = None,
    chunk_size: int = 8,
) -> list[str]:
    saved: list[str] = []
    num_dims = int(min(sim_local_obs.shape[1], real_local_obs.shape[1], len(labels)))
    num_chunks = max(1, math.ceil(num_dims / chunk_size))

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(num_dims, start + chunk_size)
        fig, axes = plt.subplots(
            end - start,
            1,
            figsize=(15, max(2.6 * (end - start), 4.5)),
            sharex=True,
            squeeze=False,
        )
        axes_flat = axes[:, 0]
        for axis_offset, obs_idx in enumerate(range(start, end)):
            ax = axes_flat[axis_offset]
            ax.plot(
                sim_x_axis,
                sim_local_obs[:, obs_idx],
                linewidth=1.2,
                color="tab:blue",
                label="sim",
            )
            ax.plot(
                real_x_axis,
                real_local_obs[:, obs_idx],
                linewidth=1.2,
                color="tab:orange",
                label="real",
            )
            match = DOF_POS_LABEL_RE.fullmatch(labels[obs_idx])
            if match is not None:
                history_idx = int(match.group(1))
                if (
                    sim_desired_target_history is not None
                    and history_idx < sim_desired_target_history.shape[1]
                ):
                    ax.plot(
                        sim_x_axis,
                        sim_desired_target_history[:, history_idx],
                        linewidth=1.1,
                        linestyle="--",
                        color="black",
                        label="desired sim",
                    )
                if (
                    real_desired_target_history is not None
                    and history_idx < real_desired_target_history.shape[1]
                ):
                    ax.plot(
                        real_x_axis,
                        real_desired_target_history[:, history_idx],
                        linewidth=1.1,
                        linestyle=":",
                        color="dimgray",
                        label="desired real",
                    )
            ax.set_ylabel(labels[obs_idx], fontsize=8)
            ax.grid(True, alpha=0.3)
            if axis_offset == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes_flat[-1].set_xlabel(x_label)
        fig.suptitle(
            "local obs compare: "
            f"slot {slot_idx} (sim M{sim_module_id}, real M{real_module_id}) "
            f"[{start}:{end}]",
            fontsize=13,
        )
        fig.tight_layout()
        output_path = output_dir / (
            f"slot_{slot_idx}_sim_{sim_module_id}_real_{real_module_id}"
            f"_compare_{chunk_idx:02d}.png"
        )
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        saved.append(str(output_path))
    return saved


def generate_local_observation_record_plots(
    record_npz: Path,
    *,
    output_dir: Path | None = None,
    time_axis: str = "step",
    local_dt: float | None = None,
) -> dict[str, object]:
    payload = _load_record_payload(record_npz)
    record_path = payload["record_path"]
    mode = str(payload["mode"])
    times = np.asarray(payload["times"], dtype=np.float64)
    local_obs = np.asarray(payload["local_obs"], dtype=np.float32)
    module_ids = list(payload["module_ids"])
    labels = list(payload["labels"])
    frame_history_steps = int(
        np.asarray(payload["record"]["local_obs_frame_history_steps"]).item()
    )
    desired_targets, desired_source = _extract_desired_target_series(
        payload,
        local_dt=0.01 if local_dt is None else float(local_dt),
    )
    desired_target_history = (
        _build_desired_target_history(
            desired_targets,
            frame_history_steps=frame_history_steps,
        )
        if desired_targets is not None
        else None
    )
    x_axis, x_label = _resolve_plot_axis(
        times=times,
        num_samples=int(local_obs.shape[0]),
        time_axis=time_axis,
        local_dt=local_dt,
    )

    if output_dir is None:
        figures_dir = record_path.with_suffix("")
        figures_dir = figures_dir.parent / f"{figures_dir.name}_figures"
    else:
        figures_dir = Path(output_dir).resolve()
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    summary: dict[str, object] = {
        "record_file": str(record_path),
        "mode": mode,
        "num_modules": int(local_obs.shape[1]),
        "obs_dim": int(local_obs.shape[2]),
        "num_samples": int(local_obs.shape[0]),
        "module_ids": module_ids,
        "labels": labels,
        "desired_target_source": desired_source,
        "figures_dir": str(figures_dir),
    }

    title_prefix = f"{mode}"
    for module_idx, module_id in enumerate(module_ids):
        module_obs = np.asarray(local_obs[:, module_idx, :], dtype=np.float32)
        heatmap_path = figures_dir / f"module_{module_id}_local_obs_heatmap.png"
        _plot_local_obs_heatmap(
            x_axis=x_axis,
            x_label=x_label,
            local_obs=module_obs,
            labels=labels,
            module_id=module_id,
            title_prefix=title_prefix,
            output_path=heatmap_path,
        )
        saved_files.append(str(heatmap_path))
        saved_files.extend(
            _plot_local_obs_trace_chunks(
                x_axis=x_axis,
                x_label=x_label,
                local_obs=module_obs,
                labels=labels,
                module_id=module_id,
                title_prefix=title_prefix,
                output_dir=figures_dir,
                desired_target_history=(
                    desired_target_history[:, module_idx, :]
                    if desired_target_history is not None
                    else None
                ),
            )
        )

    summary_path = figures_dir / "local_obs_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    saved_files.append(str(summary_path))
    return {
        "figures_dir": str(figures_dir),
        "saved_files": saved_files,
        "summary_file": str(summary_path),
    }


def generate_local_observation_comparison_plots(
    sim_record_npz: Path,
    real_record_npz: Path,
    *,
    output_dir: Path | None = None,
    time_axis: str = "step",
    local_dt: float | None = None,
) -> dict[str, object]:
    sim_payload = _load_record_payload(sim_record_npz)
    real_payload = _load_record_payload(real_record_npz)

    sim_record_path = Path(sim_payload["record_path"])
    real_record_path = Path(real_payload["record_path"])
    sim_times = np.asarray(sim_payload["times"], dtype=np.float64)
    real_times = np.asarray(real_payload["times"], dtype=np.float64)
    sim_local_obs = np.asarray(sim_payload["local_obs"], dtype=np.float32)
    real_local_obs = np.asarray(real_payload["local_obs"], dtype=np.float32)
    sim_module_ids = [int(v) for v in sim_payload["module_ids"]]
    real_module_ids = [int(v) for v in real_payload["module_ids"]]
    sim_labels = list(sim_payload["labels"])
    real_labels = list(real_payload["labels"])
    sim_frame_history_steps = int(
        np.asarray(sim_payload["record"]["local_obs_frame_history_steps"]).item()
    )
    real_frame_history_steps = int(
        np.asarray(real_payload["record"]["local_obs_frame_history_steps"]).item()
    )

    num_slots = min(
        int(sim_local_obs.shape[1]),
        int(real_local_obs.shape[1]),
        len(sim_module_ids),
        len(real_module_ids),
    )
    if num_slots <= 0:
        raise ValueError("No overlapping module slots between sim and real recordings")

    clip_len = min(int(sim_local_obs.shape[0]), int(real_local_obs.shape[0]))
    if clip_len <= 0:
        raise ValueError("No overlapping samples between sim and real recordings")

    obs_dim = min(
        int(sim_local_obs.shape[2]),
        int(real_local_obs.shape[2]),
        len(sim_labels),
        len(real_labels),
    )
    if obs_dim <= 0:
        raise ValueError("No overlapping local observation dimensions to compare")

    labels = sim_labels[:obs_dim]
    if real_labels[:obs_dim] != labels:
        labels = [
            sim_labels[idx] if idx < len(sim_labels) else real_labels[idx]
            for idx in range(obs_dim)
        ]

    figures_dir = _comparison_output_dir(sim_record_path, real_record_path, output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sim_times = sim_times[:clip_len]
    real_times = real_times[:clip_len]
    sim_local_obs = sim_local_obs[:clip_len, :num_slots, :obs_dim]
    real_local_obs = real_local_obs[:clip_len, :num_slots, :obs_dim]
    sim_desired_targets, sim_desired_source = _extract_desired_target_series(
        sim_payload,
        local_dt=0.01 if local_dt is None else float(local_dt),
    )
    real_desired_targets, real_desired_source = _extract_desired_target_series(
        real_payload,
        local_dt=0.01 if local_dt is None else float(local_dt),
    )
    sim_desired_history = (
        _build_desired_target_history(
            sim_desired_targets[:clip_len, :num_slots],
            frame_history_steps=sim_frame_history_steps,
        )
        if sim_desired_targets is not None
        else None
    )
    real_desired_history = (
        _build_desired_target_history(
            real_desired_targets[:clip_len, :num_slots],
            frame_history_steps=real_frame_history_steps,
        )
        if real_desired_targets is not None
        else None
    )
    sim_x_axis, x_label = _resolve_plot_axis(
        times=sim_times,
        num_samples=clip_len,
        time_axis=time_axis,
        local_dt=local_dt,
    )
    real_x_axis, real_x_label = _resolve_plot_axis(
        times=real_times,
        num_samples=clip_len,
        time_axis=time_axis,
        local_dt=local_dt,
    )
    if real_x_label != x_label:
        x_label = real_x_label

    saved_files: list[str] = []
    summary: dict[str, object] = {
        "sim_record_file": str(sim_record_path),
        "real_record_file": str(real_record_path),
        "sim_mode": str(sim_payload["mode"]),
        "real_mode": str(real_payload["mode"]),
        "num_slots": num_slots,
        "clip_len": clip_len,
        "obs_dim": obs_dim,
        "labels": labels,
        "sim_module_ids": sim_module_ids[:num_slots],
        "real_module_ids": real_module_ids[:num_slots],
        "sim_desired_target_source": sim_desired_source,
        "real_desired_target_source": real_desired_source,
        "figures_dir": str(figures_dir),
    }

    for slot_idx in range(num_slots):
        saved_files.extend(
            _plot_local_obs_comparison_trace_chunks(
                sim_x_axis=sim_x_axis,
                sim_local_obs=sim_local_obs[:, slot_idx, :],
                real_x_axis=real_x_axis,
                real_local_obs=real_local_obs[:, slot_idx, :],
                x_label=x_label,
                labels=labels,
                slot_idx=slot_idx,
                sim_module_id=sim_module_ids[slot_idx],
                real_module_id=real_module_ids[slot_idx],
                output_dir=figures_dir,
                sim_desired_target_history=(
                    sim_desired_history[:, slot_idx, :]
                    if sim_desired_history is not None
                    else None
                ),
                real_desired_target_history=(
                    real_desired_history[:, slot_idx, :]
                    if real_desired_history is not None
                    else None
                ),
            )
        )

    summary_path = figures_dir / "local_obs_compare_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    saved_files.append(str(summary_path))
    return {
        "figures_dir": str(figures_dir),
        "saved_files": saved_files,
        "summary_file": str(summary_path),
    }
