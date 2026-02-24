"""
Evolution Operators for Modular-Leg Robots

Genome construction, mutation, crossover, and population initialisation for
modular-leg morphologies.  These are reusable building blocks — evolution
scripts import and wire them into :class:`~metamachine.evolution.EvolutionEngine`.

Genome representation::

    {
        "morphology": [int, ...],   # flat list, groups of 4
        "num_modules": int,         # number of added modules (len/4)
        "oscillation": {            # sinusoidal controller per joint
            0: {"amplitude": float, "frequency": float, "phase": float},
            1: { ... },
            ...
        }
    }

Each group of 4 is ``[parent_module_id, parent_dock, child_dock, orientation]``.

Copyright 2026 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import copy
import math
import random
from typing import Optional

import numpy as np

from metamachine.robot_factory.modular_legs.meta_designer import ModularLegs
from metamachine.robot_factory.modular_legs.morphology import (
    DockPosition,
    ModuleConnection,
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _extend_morphology(
    robot_designer: ModularLegs,
    num_modules: int,
    rng: random.Random | None = None,
) -> list[int]:
    """Randomly attach *num_modules* to *robot_designer* and return the flat
    int sequence describing the new connections.

    Args:
        robot_designer: A :class:`ModularLegs` instance whose state already
                        reflects any prefix connections.
        num_modules:    How many modules to add.
        rng:            Optional :class:`random.Random` instance.  When
                        ``None`` the global ``random`` module is used.

    Returns:
        Flat int list ``[pid, pdock, cdock, rot, ...]``.
    """
    choice = rng.choice if rng is not None else random.choice

    pipe: list[int] = []
    for _ in range(num_modules):
        module_id = choice(robot_designer.get_available_module_ids())
        available_docks = robot_designer.get_available_docks(module_id)
        if not available_docks:
            break
        parent_dock = choice(available_docks)
        child_dock = choice(list(DockPosition)[:9])
        orientation = choice(
            robot_designer.get_available_rotation_ids(
                parent_dock.value, child_dock.value
            )
        )
        conn = ModuleConnection(
            parent_module_id=module_id,
            parent_dock=parent_dock,
            child_dock=child_dock,
            orientation=orientation,
        )
        robot_designer.add_module(conn)
        pipe.extend([module_id, parent_dock.value, child_dock.value, orientation])

    return pipe


def _replay_prefix(prefix: list[int]) -> ModularLegs:
    """Replay a flat morphology prefix to obtain the designer state."""
    robot_designer = ModularLegs()
    robot_designer.reset()
    for i in range(0, len(prefix), 4):
        conn = ModuleConnection.from_sequence(prefix[i : i + 4])
        robot_designer.add_module(conn)
    return robot_designer


# ---------------------------------------------------------------------------
# Oscillation helpers
# ---------------------------------------------------------------------------

def make_oscillation(num_joints: int, rng: random.Random | None = None) -> dict:
    """Create random sinusoidal oscillation parameters for *num_joints*.

    Each joint gets ``{amplitude, frequency, phase}`` drawn from sensible
    ranges for locomotion.

    Args:
        num_joints: Number of actuated joints.
        rng:        Optional :class:`random.Random` instance.

    Returns:
        ``{0: {amplitude, frequency, phase}, 1: {...}, ...}``
    """
    _uniform = rng.uniform if rng is not None else random.uniform
    osc: dict[int, dict] = {}
    for j in range(num_joints):
        osc[j] = {
            "amplitude": _uniform(0.3, 0.8),
            "frequency": _uniform(0.5, 2.0),
            "phase": _uniform(0.0, 2 * math.pi),
        }
    return osc


def oscillation_action(osc_params: dict, t: float, num_actions: int) -> np.ndarray:
    """Compute sinusoidal action vector at time *t*.

    Missing joints (index ≥ len(osc_params)) get a sensible default.

    Args:
        osc_params: ``{joint_idx: {amplitude, frequency, phase}, ...}``
        t:          Simulation time (seconds).
        num_actions: Size of the action vector.

    Returns:
        Action array of shape ``(num_actions,)``.
    """
    action = np.zeros(num_actions)
    for j in range(num_actions):
        p = osc_params.get(j, {"amplitude": 0.5, "frequency": 1.0, "phase": 0.0})
        action[j] = p["amplitude"] * math.sin(
            2 * math.pi * p["frequency"] * t + p["phase"]
        )
    return action


def _mutate_oscillation(osc: dict) -> dict:
    """Perturb oscillation parameters for one random joint."""
    if not osc:
        return osc
    joint = random.choice(list(osc.keys()))
    p = osc[joint]
    what = random.choice(["amplitude", "frequency", "phase"])
    if what == "amplitude":
        p["amplitude"] = max(0.1, min(1.0, p["amplitude"] + random.gauss(0, 0.15)))
    elif what == "frequency":
        p["frequency"] = max(0.2, min(4.0, p["frequency"] + random.gauss(0, 0.3)))
    elif what == "phase":
        p["phase"] = (p["phase"] + random.gauss(0, 0.5)) % (2 * math.pi)
    return osc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_morphology(
    num_modules: int = 4,
    seed: Optional[int] = None,
) -> list[int]:
    """Sample a random modular-leg morphology.

    Args:
        num_modules: Number of modules to add on top of the base module.
        seed:        Optional RNG seed (per-call; does **not** affect global
                     state).

    Returns:
        Flat int list ``[pid, pdock, cdock, rot, ...]`` of length
        ``4 * num_modules`` (may be shorter if some connections fail).
    """
    rng = random.Random(seed)
    robot_designer = ModularLegs()
    robot_designer.reset()
    return _extend_morphology(robot_designer, num_modules, rng)


def build_genome(
    num_modules: int = 4,
    num_joints_estimate: Optional[int] = None,
    seed: Optional[int] = None,
) -> dict:
    """Build a complete modular-leg genome (morphology + oscillation).

    Args:
        num_modules:        Number of modules to add.
        num_joints_estimate: Estimated number of actuated joints for the
                            initial oscillation dict.  When ``None``,
                            defaults to ``num_modules + 1`` (base module
                            contributes 1 joint).  The oscillation dict
                            gracefully handles mismatches at evaluation time.
        seed:               Optional RNG seed.

    Returns:
        ``{"morphology": [...], "num_modules": int, "oscillation": {...}}``
    """
    rng = random.Random(seed)
    morphology = sample_morphology(num_modules=num_modules, seed=seed)
    n_joints = num_joints_estimate if num_joints_estimate is not None else num_modules + 1
    osc = make_oscillation(n_joints, rng=rng)
    return {
        "morphology": morphology,
        "num_modules": num_modules,
        "oscillation": osc,
    }


def mutate_genome(genome: dict) -> dict:
    """Apply one random mutation to a modular-leg genome.

    Possible mutations (chosen at random):

    1. **Tweak morphology** (50 %): pick a random connection slot, keep the
       prefix up to that slot, and resample from that point onward.  The
       oscillation is preserved for joints that still exist and randomised
       for any new ones.
    2. **Regenerate morphology** (20 %): build a completely new random
       morphology with the same module count and fresh oscillation.
    3. **Perturb oscillation** (30 %): tweak amplitude / frequency / phase
       for one random joint (body is unchanged).
    """
    genome = copy.deepcopy(genome)
    morphology = genome["morphology"]
    num_modules = genome["num_modules"]

    roll = random.random()

    if roll < 0.30:
        # --- Mutation 3: perturb oscillation only ---
        genome["oscillation"] = _mutate_oscillation(genome.get("oscillation", {}))
    elif morphology and roll < 0.80:
        # --- Mutation 1: tweak morphology + patch oscillation ---
        num_connections = len(morphology) // 4
        idx = random.randint(0, num_connections - 1)
        prefix = morphology[: idx * 4]
        remaining = num_connections - idx

        robot_designer = _replay_prefix(prefix)
        new_tail = _extend_morphology(robot_designer, remaining)
        genome["morphology"] = prefix + new_tail

        # Patch oscillation: keep existing joints, randomise new ones
        n_joints = num_modules + 1
        old_osc = genome.get("oscillation", {})
        new_osc: dict = {}
        for j in range(n_joints):
            if j in old_osc:
                new_osc[j] = copy.deepcopy(old_osc[j])
            else:
                new_osc[j] = {
                    "amplitude": random.uniform(0.3, 0.8),
                    "frequency": random.uniform(0.5, 2.0),
                    "phase": random.uniform(0.0, 2 * math.pi),
                }
        genome["oscillation"] = new_osc
    else:
        # --- Mutation 2: full regeneration ---
        genome["morphology"] = sample_morphology(num_modules=num_modules)
        genome["oscillation"] = make_oscillation(num_modules + 1)

    return genome


def crossover_genomes(g1: dict, g2: dict) -> dict:
    """Crossover two modular-leg genomes.

    Strategy: take a random prefix of connection steps from one parent and
    re-sample the remaining slots (since downstream connections depend on
    the designer state, we cannot simply splice two morphologies).

    The oscillation is taken from one parent and per-joint mixed with the
    other, similar to the lego crossover.
    """
    target_modules = max(g1["num_modules"], g2["num_modules"])

    # Pick which parent donates the morphology prefix
    if random.random() < 0.5:
        morph_donor, osc_donor = g1, g2
    else:
        morph_donor, osc_donor = g2, g1

    donor_connections = len(morph_donor["morphology"]) // 4
    if donor_connections == 0:
        return build_genome(num_modules=target_modules)

    cut = random.randint(1, donor_connections)
    prefix = morph_donor["morphology"][: cut * 4]

    robot_designer = _replay_prefix(prefix)
    remaining = target_modules - cut
    tail = _extend_morphology(robot_designer, remaining)

    # Mix oscillation per-joint
    n_joints = target_modules + 1
    osc_a = morph_donor.get("oscillation", {})
    osc_b = osc_donor.get("oscillation", {})
    child_osc: dict = {}
    for j in range(n_joints):
        # Prefer morph donor's oscillation, randomly swap from the other
        if j in osc_b and random.random() < 0.5:
            child_osc[j] = copy.deepcopy(osc_b[j])
        elif j in osc_a:
            child_osc[j] = copy.deepcopy(osc_a[j])
        else:
            child_osc[j] = {
                "amplitude": random.uniform(0.3, 0.8),
                "frequency": random.uniform(0.5, 2.0),
                "phase": random.uniform(0.0, 2 * math.pi),
            }

    return {
        "morphology": prefix + tail,
        "num_modules": target_modules,
        "oscillation": child_osc,
    }


def make_init_population_fn(num_modules: int = 4):
    """Return a callable ``(n: int) -> list[dict]`` that creates *n* random
    modular-leg genomes, suitable for :class:`EvolutionEngine`'s
    ``init_population_fn`` parameter.
    """

    def _init(n: int) -> list[dict]:
        return [build_genome(num_modules=num_modules) for _ in range(n)]

    return _init
