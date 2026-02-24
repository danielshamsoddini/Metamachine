"""
Fitness Helper Functions

Low-level, robot-agnostic simulation helpers used by FitnessComponent
implementations.  This module contains *no* robot- or plugin-specific code.
All environment construction is delegated to a ``cfg_fn`` callable supplied
by the caller, keeping plugin details (e.g. lego configs) out of the public
codebase.

Genome contract
---------------
``cfg_fn`` receives the **full genome dict** and must return a MetaMachine
OmegaConf config.  ``action_fn`` (optional) receives ``(genome, t, num_actions)``
and returns an ``ndarray`` of actions.  ``num_actions_fn`` (optional) receives
the genome and returns the number of actuated joints.

When these callbacks are ``None`` the system falls back to:
- ``num_actions`` read from ``cfg.control.num_actions`` after ``cfg_fn`` builds
  the config.
- Zero actions (let pose-optimised defaults drive the robot).

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
    """Count active (non-passive) ball joints in a serialised RobotGraph.

    This is a lego-specific helper kept for backward compatibility.  Prefer
    passing ``num_actions_fn`` to :func:`run_displacement_trial` instead.
    """
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
    """Compute sinusoidal action vector at time *t* (lego default)."""
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
    action_fn: Optional[Callable] = None,
    num_actions_fn: Optional[Callable[[dict], int]] = None,
) -> float:
    """
    Run a simulation trial and return the XY displacement of the robot's
    centre of mass.

    This function is **robot-agnostic**.  Robot-specific behaviour is injected
    through three optional callbacks:

    ``cfg_fn(genome) -> cfg``
        Builds a MetaMachine OmegaConf config from the full genome dict.
        **Required**.

    ``action_fn(genome, t, num_actions) -> ndarray``
        Returns the action vector at time *t*.  When ``None``, zero actions
        are sent (suitable for pose-optimised robots whose default joint
        positions already produce locomotion).

    ``num_actions_fn(genome) -> int``
        Returns the number of actuated joints.  When ``None``, the value is
        read from ``cfg.control.num_actions`` after the config is built.

    Args:
        genome:          Full genome dict.
        cfg_fn:          Config builder (required).
        eval_steps:      Number of control steps to run.
        dt:              Control timestep (seconds).
        early_stop:      Stop early when the episode terminates.
        verbose:         Print error details on failure.
        action_fn:       Optional action callback.
        num_actions_fn:  Optional action-count callback.

    Returns:
        Euclidean XY displacement (metres); 0.0 on failure.
    """
    from metamachine.environments.env_sim import MetaMachine

    cfg = cfg_fn(genome)
    tmpdir = getattr(cfg, "_eval_tmpdir", None)

    # Determine num_actions
    if num_actions_fn is not None:
        num_actions = num_actions_fn(genome)
    else:
        num_actions = getattr(getattr(cfg, "control", None), "num_actions", 0)

    if num_actions == 0:
        _cleanup_tmpdir(tmpdir)
        return 0.0

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
            if action_fn is not None:
                action = action_fn(genome, t, num_actions)
            else:
                action = np.zeros(num_actions)
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
    "_graph_num_balls",
    "_oscillation_action",
    "_cleanup_tmpdir",
]
