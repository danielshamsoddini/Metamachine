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
``SelfCollisionConstraint``     — robot must have NO geometric self-intersections
                                  when all joints are at zero.  Uses analytic
                                  pairwise distance checks (not MuJoCo contacts).
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

class MinModulesConstraint(GenomeConstraint):
    """
    Reject genomes with fewer than ``min_modules`` modules.

    Works with modular-leg genomes where ``genome["morphology"]`` is a flat
    int list with groups of 4 per connection.

    Parameters:
        min_modules (int): Minimum number of modules (connections) required.

    Example::
        MinModulesConstraint(min_modules=3)
    """

    def __init__(self, min_modules: int, hard: bool = True) -> None:
        super().__init__(name=f"min_modules>={min_modules}", hard=hard)
        self.min_modules = min_modules

    def check(self, genome: dict, cfg_fn: Optional[Callable] = None) -> ConstraintResult:
        morphology = genome.get("morphology", [])
        actual = len(morphology) // 4
        if actual >= self.min_modules:
            return ConstraintResult(satisfied=True)
        violation = float(self.min_modules - actual)
        return ConstraintResult(
            satisfied=False,
            violation=violation,
            message=(
                f"min_modules: has {actual} module(s), "
                f"need >= {self.min_modules}"
            ),
            details={"actual_modules": actual, "required": self.min_modules},
        )


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


class SelfCollisionConstraint(GenomeConstraint):
    """
    Reject genomes whose robot has geometric self-intersection at zero pose.

    This is a **lightweight, geometry-only** check: the MuJoCo model is
    compiled from the morphology (via ``ModularLegs``), forward kinematics
    is run with all joints at zero, and pairwise geom distances are
    computed analytically.  No simulation stepping is needed.

    Why geometry-based instead of MuJoCo contacts?
    -----------------------------------------------
    MuJoCo's contact engine automatically filters contacts between
    parent–child bodies (``filterparent``) and **welded** bodies (those
    without joints are merged into the first ancestor that has a joint for
    collision purposes via ``body.weldid``).  In the modular-leg design many
    bodies share the same ``weldid`` (e.g.  the left half-shell plus several
    passive sticks are all welded to ``torso0``), so MuJoCo's broadphase
    will never generate contacts between them — even when they geometrically
    overlap.  A pure geometry distance check bypasses all of this.

    Body naming convention for modular legs
    ----------------------------------------
    Bodies are named ``l{N}`` / ``r{N}`` (left/right halves of module *N*)
    and ``passive{N}`` for connecting sticks.  Contacts between ``lN`` ↔
    ``rN`` (same module) are considered valid assembly contacts and do **not**
    count as self-collision.  Everything else is a real self-collision.

    Parameters:
        margin:        Extra distance tolerance (metres) before reporting
                       overlap.  Default 0.0 (strict).  A small positive
                       value (e.g. 0.005) can avoid flagging borderline
                       near-tangent geometries.
        hard:          Whether this is a hard constraint (default True).

    Example::
        SelfCollisionConstraint()
    """

    def __init__(
        self,
        margin: float = 0.0,
        hard: bool = True,
    ) -> None:
        super().__init__(name="no_self_collision", hard=hard)
        self.margin = margin

    def check(self, genome: dict, cfg_fn: Optional[Callable] = None) -> ConstraintResult:
        """
        Build the robot model from morphology, set all joints to zero, and
        check for geometric overlaps.
        """
        morphology = genome.get("morphology")
        if morphology is None:
            return ConstraintResult(
                satisfied=False,
                violation=float("inf"),
                message="no_self_collision: genome has no 'morphology' key",
            )

        try:
            n_collisions, details = _check_self_collision_at_zero(
                morphology=morphology,
                margin=self.margin,
            )
        except Exception as e:
            return ConstraintResult(
                satisfied=False,
                violation=float("inf"),
                message=f"no_self_collision: ERROR — {e}",
                details={"traceback": traceback.format_exc()},
            )

        if n_collisions == 0:
            return ConstraintResult(satisfied=True, details=details)

        return ConstraintResult(
            satisfied=False,
            violation=float(n_collisions),
            message=(
                f"no_self_collision: {n_collisions} self-intersection(s) at zero pose"
            ),
            details=details,
        )


class JointUtilityConstraint(GenomeConstraint):
    """
    Reject genomes that contain "useless" joints — joints whose actuation
    produces no observable movement of the robot's centre of mass.

    For each joint *j*, a short simulation trial is run in which *only*
    joint *j* is driven with a sine wave while all other joints are held at
    zero.  If the resulting COM displacement is below ``movement_threshold``
    metres, the joint is declared useless and the genome is rejected.

    This constraint is **robot-agnostic**.  Robot-specific behaviour is
    injected through callbacks:

    ``cfg_fn(genome) -> cfg``
        Builds a MetaMachine OmegaConf config from the full genome dict.
        **Required** (supplied at construction or at call-time).

    ``num_actions_fn(genome) -> int``  *(optional)*
        Returns the number of actuated joints.  When ``None``, the value
        is read from ``cfg.control.num_actions``.

    Parameters:
        cfg_fn (Callable):       ``cfg_fn(genome) -> cfg``.
        num_actions_fn (Callable, optional):
                                 ``num_actions_fn(genome) -> int``.
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

    Example (modular legs)::
        JointUtilityConstraint(
            cfg_fn=create_config_from_morphology,
            movement_threshold=0.005,
        )

    Example (lego legs)::
        JointUtilityConstraint(
            cfg_fn=lego_cfg_fn,
            num_actions_fn=lambda g: graph_num_balls(g["graph_dict"]),
            movement_threshold=0.005,
        )
    """

    def __init__(
        self,
        cfg_fn: Optional[Callable] = None,
        num_actions_fn: Optional[Callable] = None,
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
        self._num_actions_fn = num_actions_fn
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
            num_actions_fn=self._num_actions_fn,
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
# Self-collision check — internal implementation
# =============================================================================

def _check_self_collision_at_zero(
    morphology: list[int],
    margin: float = 0.0,
) -> tuple[int, dict]:
    """
    Build a MuJoCo model directly from a morphology sequence, set all joints
    to zero, compute forward kinematics, and check for geometric overlaps
    between geoms that belong to different kinematic groups.

    This uses a **pure geometry** approach rather than MuJoCo's contact
    engine, because the modular-leg robot has many welded bodies (bodies
    without joints are merged into the first ancestor with a joint via
    ``body.weldid``).  MuJoCo never generates contacts between geoms whose
    bodies share the same ``weldid``, even if they geometrically overlap.
    The ``filterparent`` mechanism further suppresses contacts between
    parent–child pairs.  As a result, MuJoCo's ``d.ncon`` is always 0 for
    these robots regardless of pose, despite clear geometric intersections.

    The geometry check works on the world-frame positions computed by
    ``mj_forward`` and uses conservative bounding-sphere overlap tests:

    - **Sphere** geoms: radius is ``size[0]``.
    - **Capsule** geoms: modeled as a line segment of half-length ``size[1]``
      with radius ``size[0]``.  The distance is computed as the closest point
      between the capsule's axis and the other geom's centre (or axis for
      capsule–capsule pairs).

    Excluded pairs
    ~~~~~~~~~~~~~~
    - Same-body geoms (same ``bodyid``).
    - ``lN`` ↔ ``rN`` pairs (left/right halves of the same module — these
      are co-located by design and are valid assembly contacts).
    - Geoms on the same body or on bodies sharing the same ``weldid``
      (these are rigidly attached and cannot move apart, so overlaps are
      part of the design, not collisions).

    Returns:
        ``(n_collisions, details_dict)`` — *n_collisions* is the total
        number of overlapping geom pairs found, and *details_dict* contains
        a list of colliding geom-name/body-name pairs for diagnostics.
    """
    import mujoco

    from metamachine.robot_factory.modular_legs.constants import MESH_DICT_DRAFT
    from metamachine.robot_factory.modular_legs.meta_designer import ModularLegs
    from metamachine.utils.visual_utils import get_joint_pos_addr

    # Build the model directly from morphology (no MetaMachine env needed).
    # Use MESH_DICT_DRAFT (primitives: SPHERE, CAPSULE) so we don't depend
    # on mesh files and get clean analytic collision geometry.
    ml = ModularLegs(morphology=list(morphology), mesh_dict=MESH_DICT_DRAFT)
    xml = ml.designer.builder.get_xml(fix_file_path=True)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    # Set robot in the air with all joints at zero
    qpos = np.zeros(m.nq)
    qpos[2] = 1.0  # height — above ground
    qpos[3] = 1.0  # quaternion w-component
    joint_addrs = get_joint_pos_addr(m)
    qpos[joint_addrs] = 0.0

    d.qpos[:] = qpos
    d.qvel[:] = np.zeros(m.nv)
    mujoco.mj_forward(m, d)

    # ── Collect geom info ──────────────────────────────────────────────
    # Skip the floor geom (body "world"), and internal non-collidable parts
    # (battery, pcb, motor — already excluded by MESH_DICT_DRAFT which maps
    # them to "NONE" so they don't appear).
    geom_type_sphere = 2
    geom_type_capsule = 3

    class _GeomInfo:
        __slots__ = ("idx", "gtype", "body_id", "body_name", "weld_id",
                      "pos", "mat", "radius", "half_len", "name")

    geoms: list[_GeomInfo] = []
    for gi in range(m.ngeom):
        g = m.geom(gi)
        bid = int(g.bodyid[0])
        bname = m.body(bid).name
        if bname == "world":
            continue
        gtype = int(g.type)
        if gtype not in (geom_type_sphere, geom_type_capsule):
            continue  # skip plane, mesh-only, etc.
        info = _GeomInfo()
        info.idx = gi
        info.gtype = gtype
        info.body_id = bid
        info.body_name = bname
        info.weld_id = int(m.body(bid).weldid[0])
        info.pos = d.geom_xpos[gi].copy()
        info.mat = d.geom_xmat[gi].reshape(3, 3).copy()
        info.radius = float(g.size[0])
        info.half_len = float(g.size[1]) if gtype == geom_type_capsule else 0.0
        info.name = g.name
        geoms.append(info)

    # ── Build set of adjacent weld-group pairs ─────────────────────────
    # Two weld groups are "adjacent" if they are connected by a single joint.
    # Geoms in adjacent weld groups naturally overlap at the joint boundary
    # (ball plugs into stick) — this is structural, not a collision.
    #
    # For each joint, the joint body and its parent body belong to two
    # (possibly different) weld groups.  Those weld groups are adjacent.
    adjacent_weld_pairs: set[tuple[int, int]] = set()
    for ji in range(m.njnt):
        jbid = int(m.jnt_bodyid[ji])
        pbid = int(m.body(jbid).parentid[0])
        w1 = int(m.body(jbid).weldid[0])
        w2 = int(m.body(pbid).weldid[0])
        if w1 != w2:
            pair = (min(w1, w2), max(w1, w2))
            adjacent_weld_pairs.add(pair)

    # ── Helper: extract module index from body name ────────────────────
    def _module_idx(bname: str) -> Optional[int]:
        """Return the module index from names like 'l3', 'r12'."""
        if bname and bname[0] in ("l", "r"):
            try:
                return int(bname[1:])
            except ValueError:
                return None
        return None

    # ── Helper: closest distance between two geom bounding shapes ──────
    def _geom_distance(a: _GeomInfo, b: _GeomInfo) -> float:
        """
        Compute the minimum distance between two geom bounding shapes.
        Returns distance between surfaces (negative = overlap).
        Supports sphere–sphere, sphere–capsule, and capsule–capsule.
        """
        if a.gtype == geom_type_sphere and b.gtype == geom_type_sphere:
            d_centers = np.linalg.norm(a.pos - b.pos)
            return d_centers - a.radius - b.radius

        if a.gtype == geom_type_capsule and b.gtype == geom_type_capsule:
            # Capsule–capsule: closest distance between two line segments
            # Each capsule axis = mat @ [0, 0, 1] * half_len
            axis_a = a.mat[:, 2] * a.half_len
            axis_b = b.mat[:, 2] * b.half_len
            d_seg = _segment_segment_dist(a.pos, axis_a, b.pos, axis_b)
            return d_seg - a.radius - b.radius

        # Sphere–capsule (ensure 'a' is the capsule)
        if a.gtype == geom_type_sphere:
            a, b = b, a
        # a is capsule, b is sphere
        axis_a = a.mat[:, 2] * a.half_len
        d_pt = _point_segment_dist(b.pos, a.pos, axis_a)
        return d_pt - a.radius - b.radius

    # ── Pairwise overlap check ─────────────────────────────────────────
    n_collisions = 0
    colliding_pairs: list[tuple[str, str]] = []

    for i in range(len(geoms)):
        for j in range(i + 1, len(geoms)):
            ga = geoms[i]
            gb = geoms[j]

            # Skip same body
            if ga.body_id == gb.body_id:
                continue

            # Skip same weld group (rigidly attached, overlap by design)
            if ga.weld_id == gb.weld_id:
                continue

            # Skip adjacent weld groups (connected by a single joint —
            # ball-stick overlaps at joint boundaries are structural)
            weld_pair = (min(ga.weld_id, gb.weld_id),
                         max(ga.weld_id, gb.weld_id))
            if weld_pair in adjacent_weld_pairs:
                continue

            # Skip lN <-> rN same-module pairs (assembly contacts)
            mi = _module_idx(ga.body_name)
            mj = _module_idx(gb.body_name)
            if mi is not None and mj is not None and mi == mj:
                # Same module's left/right halves — valid
                continue

            dist = _geom_distance(ga, gb)
            if dist < -margin:
                n_collisions += 1
                colliding_pairs.append((
                    f"{ga.name}({ga.body_name})",
                    f"{gb.name}({gb.body_name})",
                ))

    return n_collisions, {
        "n_collisions": n_collisions,
        "colliding_pairs": colliding_pairs,
    }


def _segment_segment_dist(
    p1: np.ndarray, d1: np.ndarray,
    p2: np.ndarray, d2: np.ndarray,
) -> float:
    """
    Minimum distance between two line segments.

    Segment 1: from ``p1 - d1`` to ``p1 + d1`` (centre + half-axis).
    Segment 2: from ``p2 - d2`` to ``p2 + d2``.

    Uses the standard closest-points-on-two-segments algorithm.
    """
    a1 = p1 - d1
    b1 = p1 + d1
    a2 = p2 - d2
    b2 = p2 + d2

    u = b1 - a1
    v = b2 - a2
    w = a1 - a2

    uu = np.dot(u, u)
    uv = np.dot(u, v)
    vv = np.dot(v, v)
    uw = np.dot(u, w)
    vw = np.dot(v, w)

    denom = uu * vv - uv * uv
    eps = 1e-12

    if denom < eps:
        # Parallel segments
        s = 0.0
        t = vw / vv if vv > eps else 0.0
    else:
        s = (uv * vw - vv * uw) / denom
        t = (uu * vw - uv * uw) / denom

    s = float(np.clip(s, 0.0, 1.0))
    t = float(np.clip(t, 0.0, 1.0))

    # Recompute to handle clamping
    t_new = (uv * s + vw) / vv if vv > eps else 0.0
    t_new = float(np.clip(t_new, 0.0, 1.0))
    s_new = (uv * t_new - uw) / uu if uu > eps else 0.0
    s_new = float(np.clip(s_new, 0.0, 1.0))

    closest1 = a1 + s_new * u
    closest2 = a2 + t_new * v
    return float(np.linalg.norm(closest1 - closest2))


def _point_segment_dist(
    pt: np.ndarray,
    seg_center: np.ndarray,
    seg_half_axis: np.ndarray,
) -> float:
    """
    Minimum distance from a point to a line segment.

    Segment: from ``seg_center - seg_half_axis`` to ``seg_center + seg_half_axis``.
    """
    a = seg_center - seg_half_axis
    b = seg_center + seg_half_axis
    ab = b - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-12:
        return float(np.linalg.norm(pt - a))
    t = float(np.clip(np.dot(pt - a, ab) / ab_sq, 0.0, 1.0))
    closest = a + t * ab
    return float(np.linalg.norm(pt - closest))


# =============================================================================
# Joint utility probe — internal implementation
# =============================================================================

def _probe_joint_utility(
    genome: dict,
    cfg_fn: Callable,
    num_actions_fn: Optional[Callable],
    movement_threshold: float,
    probe_steps: int,
    settle_steps: int,
    probe_dt: float,
    probe_amplitude: float,
    probe_frequency: float,
) -> list[int]:
    """
    For each joint, run a short isolated probe simulation and return the
    list of joint indices that produce less than ``movement_threshold``
    metres of COM displacement.

    The number of joints is determined by ``num_actions_fn(genome)`` when
    provided, otherwise by building a config and reading
    ``cfg.control.num_actions``.

    Returns:
        List of zero-based joint indices that are "useless".
    """
    from metamachine.evolution.fitness_helpers import _cleanup_tmpdir

    try:
        from metamachine.environments.env_sim import MetaMachine
    except ImportError as e:
        raise ImportError(f"JointUtilityConstraint requires MetaMachine: {e}") from e

    # Determine number of joints
    if num_actions_fn is not None:
        num_joints = num_actions_fn(genome)
    else:
        # Build a config to read num_actions
        cfg = cfg_fn(genome)
        num_joints = getattr(getattr(cfg, "control", None), "num_actions", 0)

    if num_joints == 0:
        return []

    useless: list[int] = []

    for joint_idx in range(num_joints):
        displacement = _probe_single_joint(
            genome=genome,
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
    genome: dict,
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

    cfg = cfg_fn(genome)
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
    "min_modules": MinModulesConstraint,
    "min_balls": MinBallsConstraint,
    "min_legs": MinLegsConstraint,
    "no_self_collision": SelfCollisionConstraint,
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
    "MinModulesConstraint",
    "MinBallsConstraint",
    "MinLegsConstraint",
    "SelfCollisionConstraint",
    "JointUtilityConstraint",
    # Aggregator
    "ConstraintChecker",
    # Registry
    "CONSTRAINT_REGISTRY",
    "register_constraint",
    "list_available_constraints",
]
