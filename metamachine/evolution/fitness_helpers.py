"""
Fitness Helper Functions

Low-level, robot-agnostic simulation helpers used by FitnessComponent
implementations.  This module contains *no* robot- or plugin-specific code.
All environment construction is delegated to a ``cfg_fn`` callable supplied
by the caller, keeping plugin details (e.g. lego configs) out of the public
codebase.

Copyright 2026 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
import shutil
import traceback
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _cleanup_tmpdir(tmpdir: Optional[str]) -> None:
    """Remove a temporary directory quietly."""
    if tmpdir is None:
        return
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


def _graph_num_balls(graph_dict: dict) -> int:
    """Count active (non-passive) ball joints in a serialised RobotGraph."""
    count = 0
    for comp in graph_dict.get("components", []):
        ctype = comp.get("component_type", "")
        is_passive = comp.get("params", {}).get("passive", False)
        if ctype == "ball" and not is_passive:
            count += 1
    return count


def _oscillation_action(
    osc_params: dict, t: float, num_actions: int
) -> np.ndarray:
    """Compute sinusoidal action vector at time *t*."""
    action = np.zeros(num_actions)
    for j in range(num_actions):
        p = osc_params.get(j, {"amplitude": 0.5, "frequency": 1.0, "phase": 0.0})
        action[j] = p["amplitude"] * math.sin(
            2 * math.pi * p["frequency"] * t + p["phase"]
        )
    return action


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def run_displacement_trial(
    genome: dict,
    cfg_fn: Callable[[dict], object],
    eval_steps: int = 500,
    dt: float = 0.05,
    early_stop: bool = True,
    verbose: bool = False,
) -> float:
    """
    Run an open-loop sinusoidal oscillation trial and return the XY displacement
    of the robot's centre of mass.

    All environment construction is handled by *cfg_fn*, which keeps any
    robot- or plugin-specific logic out of this module.

    Args:
        genome:     Genome dict with keys ``graph_dict`` and ``oscillation``.
        cfg_fn:     ``cfg_fn(graph_dict) -> cfg`` — caller-supplied function
                    that builds a MetaMachine OmegaConf config for the given
                    serialised graph.  The returned config object may carry a
                    ``_eval_tmpdir`` attribute pointing to a temporary directory
                    that will be cleaned up after the trial.
        eval_steps: Number of control steps to run.
        dt:         Control timestep (seconds).
        early_stop: If True, stop early when the episode terminates.
        verbose:    Print error details on failure.

    Returns:
        Euclidean XY displacement (metres); 0.0 on failure.
    """
    from metamachine.environments.env_sim import MetaMachine

    graph_dict = genome["graph_dict"]
    osc_params = genome["oscillation"]
    num_actions = _graph_num_balls(graph_dict)

    if num_actions == 0:
        return 0.0

    cfg = cfg_fn(graph_dict)
    tmpdir = getattr(cfg, "_eval_tmpdir", None)

    try:
        env = MetaMachine(cfg)
    except Exception as e:
        if verbose:
            print(f"    [fitness_helpers] Env creation failed: {e}")
        _cleanup_tmpdir(tmpdir)
        return 0.0

    displacement = 0.0
    try:
        obs, info = env.reset()
        start_pos = env.data.qpos[:2].copy()

        for step in range(eval_steps):
            t = step * dt
            action = _oscillation_action(osc_params, t, num_actions)
            obs, reward, done, truncated, info = env.step(action)
            if early_stop and (done or truncated):
                break

        end_pos = env.data.qpos[:2].copy()
        displacement = float(np.linalg.norm(end_pos - start_pos))
    except Exception as e:
        if verbose:
            print(f"    [fitness_helpers] Simulation failed: {e}")
            traceback.print_exc()
        displacement = 0.0
    finally:
        try:
            env.close()
        except Exception:
            pass
        _cleanup_tmpdir(tmpdir)

    return displacement


__all__ = [
    "run_displacement_trial",
]
