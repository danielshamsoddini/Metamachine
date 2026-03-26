from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import numpy as np


class LiveJointTargetPlotter:
    """Interactive step-index plot of desired vs actual joint positions."""

    def __init__(
        self,
        *,
        num_joints: int,
        joint_names: Optional[list[str]] = None,
        history_length: int = 400,
        update_interval_sec: float = 0.05,
        figsize: tuple[int, int] = (12, 8),
        title: str = "Live Joint Tracking",
    ) -> None:
        self.num_joints = int(num_joints)
        self.joint_names = joint_names or [f"Joint {idx}" for idx in range(self.num_joints)]
        self.history_length = int(history_length)
        self.update_interval_sec = float(update_interval_sec)
        self.figsize = figsize
        self.title = title

        self._step_history: deque[int] = deque(maxlen=self.history_length)
        self._desired_history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._actual_history: deque[np.ndarray] = deque(maxlen=self.history_length)

        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._step_count = 0

        self._fig = None
        self._axes = None
        self._lines = None

    def start(self) -> None:
        if self._running:
            return

        self._step_history.clear()
        self._desired_history.clear()
        self._actual_history.clear()
        self._step_count = 0
        self._running = True
        self._thread = threading.Thread(target=self._plot_loop, daemon=True)
        self._thread.start()
        print(f"[LiveJointTargetPlotter] Started ({self.num_joints} joints)")

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        print("[LiveJointTargetPlotter] Stopped")

    def update(
        self,
        *,
        desired_joint_positions: np.ndarray,
        actual_joint_positions: np.ndarray,
    ) -> None:
        if not self._running:
            return

        desired = np.asarray(desired_joint_positions, dtype=np.float32).reshape(-1)
        actual = np.asarray(actual_joint_positions, dtype=np.float32).reshape(-1)
        if desired.size != self.num_joints or actual.size != self.num_joints:
            return

        with self._lock:
            self._step_history.append(self._step_count)
            self._desired_history.append(desired.copy())
            self._actual_history.append(actual.copy())
            self._step_count += 1

    def _plot_loop(self) -> None:
        import matplotlib

        backend = None
        for candidate in ("TkAgg", "Qt5Agg", "GTK3Agg", "WXAgg"):
            try:
                matplotlib.use(candidate)
                backend = candidate
                break
            except Exception:
                continue

        if backend is None:
            print(
                "[LiveJointTargetPlotter] Warning: no interactive matplotlib backend available; "
                "live plotting disabled."
            )
            self._running = False
            return

        import matplotlib.pyplot as plt

        plt.ion()
        fig_height = max(self.figsize[1], int(2.5 * self.num_joints))
        self._fig, axes = plt.subplots(
            self.num_joints,
            1,
            figsize=(self.figsize[0], fig_height),
            squeeze=False,
        )
        self._fig.suptitle(self.title, fontsize=14)
        self._axes = axes[:, 0]
        self._lines = []

        for joint_idx, ax in enumerate(self._axes):
            ax.set_title(self.joint_names[joint_idx], fontsize=10)
            ax.set_xlabel("local control step", fontsize=8)
            ax.set_ylabel("joint pos [rad]", fontsize=8)
            ax.grid(True, alpha=0.3)
            desired_line, = ax.plot([], [], "k--", linewidth=1.3, label="desired")
            actual_line, = ax.plot([], [], color="tab:red", linewidth=1.6, label="actual")
            self._lines.append((desired_line, actual_line))
            ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.show(block=False)

        while self._running:
            try:
                self._refresh_plot()
                plt.pause(self.update_interval_sec)
            except Exception as exc:
                print(f"[LiveJointTargetPlotter] Error: {exc}")
                break

        plt.ioff()
        if self._fig is not None:
            plt.close(self._fig)

    def _refresh_plot(self) -> None:
        with self._lock:
            if len(self._step_history) < 2:
                return
            steps = np.asarray(self._step_history, dtype=np.int32)
            desired = np.asarray(self._desired_history, dtype=np.float32)
            actual = np.asarray(self._actual_history, dtype=np.float32)

        for joint_idx, ax in enumerate(self._axes):
            desired_line, actual_line = self._lines[joint_idx]
            desired_line.set_data(steps, desired[:, joint_idx])
            actual_line.set_data(steps, actual[:, joint_idx])

            ymin = float(min(np.min(desired[:, joint_idx]), np.min(actual[:, joint_idx])))
            ymax = float(max(np.max(desired[:, joint_idx]), np.max(actual[:, joint_idx])))
            pad = max(0.05, 0.1 * max(1e-6, ymax - ymin))
            ax.set_xlim(int(steps[0]), int(steps[-1]) if len(steps) > 1 else int(steps[0]) + 1)
            ax.set_ylim(ymin - pad, ymax + pad)

        if self._fig is not None:
            self._fig.canvas.draw_idle()


class PolicyFeedbackJointStateBuffer:
    """Track latest per-module desired/actual joint values from policy feedback."""

    def __init__(self, *, module_ids: list[int] | tuple[int, ...] | np.ndarray) -> None:
        module_ids_arr = np.asarray(module_ids, dtype=np.int32).reshape(-1)
        self.module_ids = module_ids_arr.copy()
        self.num_joints = int(module_ids_arr.size)
        self._module_to_index = {
            int(module_id): idx for idx, module_id in enumerate(module_ids_arr.tolist())
        }
        self._desired = np.zeros(self.num_joints, dtype=np.float32)
        self._actual = np.zeros(self.num_joints, dtype=np.float32)
        self._valid = np.zeros(self.num_joints, dtype=bool)
        self._lock = threading.Lock()

    def update_from_feedback(self, module_id: int, msg) -> None:
        debug = getattr(msg, "policy_debug", None)
        if debug is None:
            return
        if int(getattr(debug, "valid", 0)) == 0:
            return
        joint_idx = self._module_to_index.get(int(module_id))
        if joint_idx is None:
            return

        desired = float(getattr(debug, "motor_target", 0.0))
        actual = float(getattr(debug, "dof_pos", 0.0))
        with self._lock:
            self._desired[joint_idx] = desired
            self._actual[joint_idx] = actual
            self._valid[joint_idx] = True

    def snapshot(self) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            if not np.all(self._valid):
                return None
            return self._desired.copy(), self._actual.copy()
