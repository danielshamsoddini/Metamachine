"""
Genome Constraint System for Evolutionary Robotics

Mirrors the FitnessComponent architecture: constraint checks are composable,
registerable, and injected into EvolutionEngine as a ``constraint_fn``.

Design
------
- ``GenomeConstraint`` (ABC): one hard or soft rule evaluated against a genome.
  Returns a ``ConstraintResult`` dataclass with pass/fail flag, a numeric
  *violation* score (0 = no violation, larger = worse), and a message.
- ``ConstraintChecker``: aggregates multiple constraints into a single
  ``check(genome) -> ConstraintReport`` call.  Reports whether the genome is
  feasible and provides per-constraint details for logging.
- ``ConstrainedEvolutionEngine``: thin subclass of ``EvolutionEngine`` that
  skips fitness evaluation for infeasible individuals (assigns fitness
  ``penalty`` instead), and logs constraint violation stats.

Hook-in pattern
---------------
The simplest integration: wrap the existing ``evaluate_fn`` with
``ConstraintChecker.guarded_evaluate()``.  This means no changes to
``EvolutionEngine`` are required for basic use.

For the advanced use-case (e.g. the joint-utility check), constraints that
require a simulation receive a ``cfg_fn`` callable exactly like
``DisplacementFitnessComponent`` does, keeping all robot-specific setup in
the plugin script.

Built-in constraints
--------------------
``MinBallsConstraint``          — genome must have at least N active ball joints.
``MinLegsConstraint``           — genome must have at least N leg chains.
``JointUtilityConstraint``      — every joint, when actuated alone, must produce
                                  measurable robot COM movement; joints that
                                  produce no movement are "useless" and the
                                  genome is rejected.

Example (programmatic)
----------------------
    from metamachine.evolution.constraint import (
        ConstraintChecker,
        MinBallsConstraint,
        JointUtilityConstraint,
    )
    from my_plugin.evolve import create_config_from_graph_dict

    checker = ConstraintChecker([
        MinBallsConstraint(min_balls=4),
        JointUtilityConstraint(cfg_fn=create_config_from_graph_dict,
                               movement_threshold=0.005),
    ])

    # Option A: wrap evaluate_fn (no engine changes needed)
    safe_evaluate = checker.guarded_evaluate(evaluate_fn, penalty=0.0)

    # Option B: check explicitly before evaluation
    report = checker.check(genome)
    if report.feasible:
        fitness = evaluate_fn(genome)

Copyright 2026 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# =============================================================================
# Result types
# =============================================================================

@dataclass
class ConstraintResult:
    """
    Outcome of a single GenomeConstraint evaluation.

    Attributes:
        satisfied:  True if the constraint is met (genome is feasible w.r.t. this rule).
        violation:  Non-negative numeric degree of violation.  0.0 means no
                    violation; larger values mean a worse violation.  This can
                    be used to rank infeasible genomes or as a penalty signal.
        message:    Human-readable explanation (empty string if satisfied).
        details:    Optional dict of extra diagnostic data (e.g. per-joint results).
    """
    satisfied: bool
    violation: float = 0.0
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstraintReport:
    """
    Aggregated result for all constraints checked against a single genome.

    Attributes:
        feasible:   True iff every constraint is satisfied.
        results:    Mapping from constraint name to its ConstraintResult.
        violations: Total summed violation score across all unsatisfied constraints.
        messages:   List of non-empty violation messages (for quick display).
    """
    feasible: bool
    results: dict[str, ConstraintResult] = field(default_factory=dict)
    violations: float = 0.0
    messages: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.feasible:
            return "ConstraintReport: FEASIBLE"
        return (
            f"ConstraintReport: INFEASIBLE "
            f"(total violation={self.violations:.3f})\n"
            + "\n".join(f"  - {m}" for m in self.messages)
        )


# =============================================================================
# Abstract base
# =============================================================================

class GenomeConstraint(ABC):
    """
    Base class for all genome constraints.

    Sub-classes must implement ``check(genome, cfg_fn) -> ConstraintResult``.

    Attributes:
        name:   Human-readable label used in reports and logs.
        hard:   If True (default), a violation makes the genome infeasible and
                fitness evaluation is skipped.  If False, the violation is
                recorded but the genome is still evaluated (soft constraint).
    """

    def __init__(self, name: str, hard: bool = True) -> None:
        self.name = name
        self.hard = hard

    @abstractmethod
    def check(self, genome: dict, cfg_fn: Optional[Callable] = None) -> ConstraintResult:
        """
        Evaluate the constraint against *genome*.

        Args:
            genome:  The genome dict (keys: ``graph_dict``, ``oscillation``,
                     ``budget_info``).
            cfg_fn:  Optional callable ``cfg_fn(graph_dict) -> cfg`` supplied
                     by the caller.  Required only for simulation-based
                     constraints such as ``JointUtilityConstraint``.

        Returns:
            A ``ConstraintResult`` with ``satisfied=True`` when the genome
            passes this rule.
        """


# =============================================================================
# Built-in constraints
# =============================================================================

class MinBallsConstraint(GenomeConstraint):
    """
    Reject genomes with fewer than ``min_balls`` active (non-passive) ball joints.

    This replaces the ad-hoc ``min_balls`` check scattered across the lego
    evolution script with a first-class constraint object.

    Parameters:
        min_balls (int): Minimum number of active ball joints required.

    Example::
        MinBallsConstraint(min_balls=4)
    """

    def __init__(self, min_balls: int, hard: bool = True) -> None:
        super().__init__(name=f"min_balls>={min_balls}", hard=hard)
        self.min_balls = min_balls

    def check(self, genome: dict, cfg_fn: Optional[Callable] = None) -> ConstraintResult:
        graph_dict = genome.get("graph_dict", {})
        count = 0
        for comp in graph_dict.get("components", []):
            if (comp.get("component_type") == "ball"
                    and not comp.get("params", {}).get("passive", False)):
                count += 1

        if count >= self.min_balls:
            return ConstraintResult(satisfied=True)
        violation = float(self.min_balls - count)
        return ConstraintResult(
            satisfied=False,
            violation=violation,
            message=(
                f"min_balls: has {count} active ball joint(s), "
                f"need >= {self.min_balls}"
            ),
            details={"active_balls": count, "required": self.min_balls},
        )


class MinLegsConstraint(GenomeConstraint):
    """
    Reject genomes whose ``budget_info`` declares fewer legs than required.

    This reads ``genome["budget_info"]["min_legs"]`` which is set by
    ``build_genome()`` in the lego evolution script.

    Parameters:
        min_legs (int): Minimum number of leg chains required.

    Example::
        MinLegsConstraint(min_legs=2)
    """

    def __init__(self, min_legs: int, hard: bool = True) -> None:
        super().__init__(name=f"min_legs>={min_legs}", hard=hard)
        self.min_legs = min_legs

    def check(self, genome: dict, cfg_fn: Optional[Callable] = None) -> ConstraintResult:
        declared = genome.get("budget_info", {}).get("min_legs", 0)
        if declared >= self.min_legs:
            return ConstraintResult(satisfied=True)
        violation = float(self.min_legs - declared)
        return ConstraintResult(
            satisfied=False,
            violation=violation,
            message=(
                f"min_legs: declared {declared} leg(s), need >= {self.min_legs}"
            ),
            details={"declared_min_legs": declared, "required": self.min_legs},
        )


class JointUtilityConstraint(GenomeConstraint):
    """
    Reject genomes that contain "useless" joints — joints whose actuation
    produces no observable movement of the robot's centre of mass.

    For each active ball joint *j*, a short simulation trial is run in which
    *only* joint *j* is driven with a sine wave while all other joints are
    held at zero.  If the resulting COM displacement is below
    ``movement_threshold`` metres, the joint is declared useless and the
    genome is rejected.

    This constraint requires a ``cfg_fn`` callable (same contract as
    ``DisplacementFitnessComponent``) to build the simulation environment.

    Parameters:
        cfg_fn (Callable):       ``cfg_fn(graph_dict) -> cfg``; builds the
                                 MetaMachine OmegaConf config.  Can be supplied
                                 here or at call-time via ``check(..., cfg_fn=...)``.
        movement_threshold (float):
                                 Minimum XY displacement (metres) required for
                                 a joint to be considered useful.
                                 Default: 0.005 m (5 mm).
        probe_steps (int):       Number of simulation steps per joint probe.
                                 Default: 100 (enough to complete ~1 cycle at 1 Hz).
        settle_steps (int):      Steps with all-zero actions run before each probe
                                 so the robot can drop from its spawn pose and reach
                                 a stable resting contact with the ground.
                                 The COM position is recorded *after* settling.
                                 Default: 50 (~2.5 s at dt=0.05).
        probe_dt (float):        Control timestep for the probe trial.  Default: 0.05 s.
        probe_amplitude (float): Sine amplitude used during the probe. Default: 0.8.
        probe_frequency (float): Sine frequency (Hz) during the probe. Default: 1.0.
        max_useless (int):       Maximum number of useless joints to tolerate before
                                 failing. Default: 0 (any useless joint fails).
        hard (bool):             Whether this is a hard constraint (default True).

    Example::
        JointUtilityConstraint(
            cfg_fn=create_config_from_graph_dict,
            movement_threshold=0.005,
            probe_steps=100,
            settle_steps=50,
        )
    """

    def __init__(
        self,
        cfg_fn: Optional[Callable] = None,
        movement_threshold: float = 0.005,
        probe_steps: int = 100,
        settle_steps: int = 50,
        probe_dt: float = 0.05,
        probe_amplitude: float = 0.8,
        probe_frequency: float = 1.0,
        max_useless: int = 0,
        hard: bool = True,
    ) -> None:
        super().__init__(name="joint_utility", hard=hard)
        self._cfg_fn = cfg_fn
        self.movement_threshold = movement_threshold
        self.probe_steps = probe_steps
        self.settle_steps = settle_steps
        self.probe_dt = probe_dt
        self.probe_amplitude = probe_amplitude
        self.probe_frequency = probe_frequency
        self.max_useless = max_useless

    def check(self, genome: dict, cfg_fn: Optional[Callable] = None) -> ConstraintResult:
        """
        Run a per-joint probe simulation and report useless joints.

        ``cfg_fn`` supplied here takes precedence over the one stored at
        construction time, so constraints can be reused with different robots.
        """
        effective_cfg_fn = cfg_fn or self._cfg_fn
        if effective_cfg_fn is None:
            raise ValueError(
                "JointUtilityConstraint requires a cfg_fn; pass it at "
                "construction time or as a keyword argument to check()."
            )

        useless = _probe_joint_utility(
            genome=genome,
            cfg_fn=effective_cfg_fn,
            movement_threshold=self.movement_threshold,
            probe_steps=self.probe_steps,
            settle_steps=self.settle_steps,
            probe_dt=self.probe_dt,
            probe_amplitude=self.probe_amplitude,
            probe_frequency=self.probe_frequency,
        )

        if len(useless) <= self.max_useless:
            return ConstraintResult(
                satisfied=True,
                details={"useless_joints": useless},
            )

        violation = float(len(useless) - self.max_useless)
        return ConstraintResult(
            satisfied=False,
            violation=violation,
            message=(
                f"joint_utility: {len(useless)} useless joint(s) found "
                f"(threshold={self.movement_threshold} m): joints {useless}"
            ),
            details={"useless_joints": useless, "max_allowed": self.max_useless},
        )


# =============================================================================
# Joint utility probe — internal implementation
# =============================================================================

def _probe_joint_utility(
    genome: dict,
    cfg_fn: Callable,
    movement_threshold: float,
    probe_steps: int,
    settle_steps: int,
    probe_dt: float,
    probe_amplitude: float,
    probe_frequency: float,
) -> list[int]:
    """
    For each active joint, run a short isolated probe simulation and return
    the list of joint indices that produce less than ``movement_threshold``
    metres of COM displacement.

    Each probe shares a single env creation to amortise startup cost when
    possible; if env reset is available it is used, otherwise a new env is
    created per joint.

    Returns:
        List of zero-based joint indices that are "useless".
    """
    from metamachine.evolution.fitness_helpers import _graph_num_balls, _cleanup_tmpdir

    try:
        from metamachine.environments.env_sim import MetaMachine
    except ImportError as e:
        raise ImportError(f"JointUtilityConstraint requires MetaMachine: {e}") from e

    graph_dict = genome["graph_dict"]
    num_joints = _graph_num_balls(graph_dict)

    if num_joints == 0:
        return []

    useless: list[int] = []

    for joint_idx in range(num_joints):
        displacement = _probe_single_joint(
            graph_dict=graph_dict,
            joint_idx=joint_idx,
            num_joints=num_joints,
            cfg_fn=cfg_fn,
            probe_steps=probe_steps,
            settle_steps=settle_steps,
            probe_dt=probe_dt,
            probe_amplitude=probe_amplitude,
            probe_frequency=probe_frequency,
        )
        if displacement < movement_threshold:
            useless.append(joint_idx)

    return useless


def _probe_single_joint(
    graph_dict: dict,
    joint_idx: int,
    num_joints: int,
    cfg_fn: Callable,
    probe_steps: int,
    settle_steps: int,
    probe_dt: float,
    probe_amplitude: float,
    probe_frequency: float,
) -> float:
    """
    Run a single-joint probe and return the COM XY displacement.

    Two phases:
      1. **Settle** (``settle_steps`` steps, all actions = 0): the robot drops
         from its spawn pose and reaches a stable resting contact with the
         ground.  COM position is recorded at the *end* of this phase.
      2. **Probe** (``probe_steps`` steps): only joint ``joint_idx`` is driven
         with a sine wave; all others remain at zero.  Displacement is measured
         from the post-settle position.

    Returns 0.0 on any failure.
    """
    from metamachine.environments.env_sim import MetaMachine
    from metamachine.evolution.fitness_helpers import _cleanup_tmpdir

    cfg = cfg_fn(graph_dict)
    tmpdir = getattr(cfg, "_eval_tmpdir", None)

    try:
        env = MetaMachine(cfg)
    except Exception:
        _cleanup_tmpdir(tmpdir)
        return 0.0

    displacement = 0.0
    try:
        obs, _ = env.reset()

        # --- Phase 1: settle with zero actions ---
        zero_action = np.zeros(num_joints)
        for _ in range(settle_steps):
            obs, _, done, truncated, _ = env.step(zero_action)
            if done or truncated:
                # Robot already fell over during settling — treat as useless
                return 0.0

        # Record COM position after the robot has settled
        start_pos = env.data.qpos[:2].copy()

        # --- Phase 2: probe the target joint ---
        action = np.zeros(num_joints)
        for step in range(probe_steps):
            t = step * probe_dt
            action[:] = 0.0
            action[joint_idx] = probe_amplitude * math.sin(
                2 * math.pi * probe_frequency * t
            )
            obs, _, done, truncated, _ = env.step(action)
            if done or truncated:
                break

        end_pos = env.data.qpos[:2].copy()
        displacement = float(np.linalg.norm(end_pos - start_pos))
    except Exception:
        displacement = 0.0
    finally:
        try:
            env.close()
        except Exception:
            pass
        _cleanup_tmpdir(tmpdir)

    return displacement


# =============================================================================
# ConstraintChecker
# =============================================================================

class ConstraintChecker:
    """
    Aggregates multiple ``GenomeConstraint`` instances and evaluates them all
    against a genome in one call.

    Only *hard* constraints (``constraint.hard == True``) affect feasibility.
    Soft constraints are recorded but do not mark the genome infeasible.

    Usage::
        checker = ConstraintChecker([
            MinBallsConstraint(min_balls=4),
            JointUtilityConstraint(cfg_fn=my_cfg_fn),
        ])

        report = checker.check(genome)
        if report.feasible:
            fitness = evaluate_fn(genome)

        # Or use the convenience wrapper:
        safe_evaluate = checker.guarded_evaluate(evaluate_fn, penalty=0.0)
        fitness = safe_evaluate(genome)

    Args:
        constraints:  List of ``GenomeConstraint`` instances.
        cfg_fn:       Optional default ``cfg_fn`` passed to every constraint
                      that does not have its own.
    """

    def __init__(
        self,
        constraints: list[GenomeConstraint],
        cfg_fn: Optional[Callable] = None,
    ) -> None:
        if not constraints:
            raise ValueError("ConstraintChecker requires at least one constraint.")
        self.constraints = constraints
        self._cfg_fn = cfg_fn

    def check(
        self,
        genome: dict,
        cfg_fn: Optional[Callable] = None,
    ) -> ConstraintReport:
        """
        Evaluate all constraints against *genome*.

        ``cfg_fn`` here takes precedence over the one supplied at construction,
        which in turn is used as fallback for any constraint that has no
        ``cfg_fn`` of its own.

        Args:
            genome:  The genome dict.
            cfg_fn:  Optional override for the config-builder callable.

        Returns:
            A ``ConstraintReport`` summarising all results.
        """
        effective_cfg_fn = cfg_fn or self._cfg_fn
        feasible = True
        total_violation = 0.0
        results: dict[str, ConstraintResult] = {}
        messages: list[str] = []

        for constraint in self.constraints:
            try:
                result = constraint.check(genome, cfg_fn=effective_cfg_fn)
            except Exception as e:
                # Treat a crashed constraint as a hard violation
                result = ConstraintResult(
                    satisfied=False,
                    violation=float("inf"),
                    message=f"{constraint.name}: ERROR — {e}",
                    details={"traceback": traceback.format_exc()},
                )

            results[constraint.name] = result

            if not result.satisfied:
                if constraint.hard:
                    feasible = False
                total_violation += result.violation
                if result.message:
                    messages.append(result.message)

        return ConstraintReport(
            feasible=feasible,
            results=results,
            violations=total_violation,
            messages=messages,
        )

    def guarded_evaluate(
        self,
        evaluate_fn: Callable[[dict], float],
        penalty: float = 0.0,
        cfg_fn: Optional[Callable] = None,
        verbose: bool = False,
    ) -> Callable[[dict], float]:
        """
        Wrap *evaluate_fn* so that infeasible genomes receive *penalty*
        instead of being simulated.

        This is the zero-change integration path: replace ``evaluate_fn``
        with ``checker.guarded_evaluate(evaluate_fn)`` and pass the result
        to ``EvolutionEngine``.

        Args:
            evaluate_fn: The original fitness evaluation callable.
            penalty:     Fitness value assigned to infeasible genomes (default 0.0).
            cfg_fn:      cfg_fn override forwarded to ``check()``.
            verbose:     Print constraint violations when they occur.

        Returns:
            A new callable with the same signature as *evaluate_fn*.
        """
        effective_cfg_fn = cfg_fn or self._cfg_fn

        def _guarded(genome: dict) -> float:
            report = self.check(genome, cfg_fn=effective_cfg_fn)
            if not report.feasible:
                if verbose:
                    print(f"    [constraint] INFEASIBLE — {'; '.join(report.messages)}")
                return penalty
            return evaluate_fn(genome)

        return _guarded

    def __str__(self) -> str:
        hard = [c.name for c in self.constraints if c.hard]
        soft = [c.name for c in self.constraints if not c.hard]
        lines = [f"ConstraintChecker ({len(self.constraints)} constraint(s)):"]
        if hard:
            lines.append(f"  Hard: {', '.join(hard)}")
        if soft:
            lines.append(f"  Soft: {', '.join(soft)}")
        return "\n".join(lines)


# =============================================================================
# Registry  (mirrors FITNESS_COMPONENT_REGISTRY)
# =============================================================================

CONSTRAINT_REGISTRY: dict[str, type[GenomeConstraint]] = {
    "min_balls": MinBallsConstraint,
    "min_legs": MinLegsConstraint,
    "joint_utility": JointUtilityConstraint,
}


def register_constraint(name: str, constraint_class: type) -> None:
    """
    Register a custom ``GenomeConstraint`` class under a string key.

    After registration, the constraint can be listed by
    ``list_available_constraints()``.

    Args:
        name:             Registry key.
        constraint_class: A subclass of ``GenomeConstraint``.

    Raises:
        ValueError: If *constraint_class* is not a GenomeConstraint subclass.
    """
    if not (isinstance(constraint_class, type)
            and issubclass(constraint_class, GenomeConstraint)):
        raise ValueError(
            f"constraint_class must be a subclass of GenomeConstraint, "
            f"got {constraint_class!r}"
        )
    CONSTRAINT_REGISTRY[name] = constraint_class


def list_available_constraints() -> list[str]:
    """Return names of all registered constraint types."""
    return list(CONSTRAINT_REGISTRY.keys())


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Types
    "ConstraintResult",
    "ConstraintReport",
    # Base class
    "GenomeConstraint",
    # Built-ins
    "MinBallsConstraint",
    "MinLegsConstraint",
    "JointUtilityConstraint",
    # Aggregator
    "ConstraintChecker",
    # Registry
    "CONSTRAINT_REGISTRY",
    "register_constraint",
    "list_available_constraints",
]
