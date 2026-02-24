"""
Fitness Components for Evolutionary Robotics

Component-based fitness system that mirrors the RewardComponent / RewardCalculator
architecture in metamachine.environments.components.reward.

Design
------
- FitnessComponent (ABC): one measurable objective, evaluated against a genome
  via ``calculate(genome, evaluator) -> float``.
- FitnessCalculator: aggregates weighted components into a single scalar fitness,
  similar to how RewardCalculator sums weighted RewardComponents.
- Built-in components ship for the most common evolutionary robotics objectives.
- Third-party components can be registered via ``register_fitness_component()``.

Genome contract
---------------
Components receive the full genome dict (see evolve_lego_robots.py for the schema)
and an optional ``evaluator`` object that provides shared resources such as a
pre-built simulation environment or caches.  For the built-in
``DisplacementFitnessComponent`` the evaluator is not required — it spins up its
own simulation internally — but more advanced components can share expensive
objects through it.

Example usage (YAML-driven)
---------------------------
    fitness_components:
      - name: displacement
        type: displacement
        weight: 1.0
        params:
          eval_steps: 500
      - name: smoothness
        type: oscillation_smoothness
        weight: 0.2

    from metamachine.evolution.fitness import FitnessCalculator
    calc = FitnessCalculator.from_list(cfg_list)
    fitness, info = calc.calculate(genome)

Example usage (programmatic)
-----------------------------
    from metamachine.evolution.fitness import (
        FitnessCalculator,
        DisplacementFitnessComponent,
        OscillationSmoothnessComponent,
    )
    calc = FitnessCalculator([
        DisplacementFitnessComponent("displacement", weight=1.0, eval_steps=500),
        OscillationSmoothnessComponent("smoothness", weight=0.2),
    ])
    fitness, info = calc.calculate(genome)

Copyright 2026 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


# =============================================================================
# Abstract base
# =============================================================================

class FitnessComponent(ABC):
    """
    Base class for all fitness components.

    Sub-classes must implement ``calculate(genome, evaluator)``.  The return
    value is an *un-weighted* scalar; weighting is applied by FitnessCalculator.

    Attributes:
        name:   Human-readable label used in logging and info dicts.
        weight: Multiplicative weight applied by FitnessCalculator.
        params: Extra keyword arguments supplied at construction time.
    """

    def __init__(self, name: str, weight: float = 1.0, **kwargs) -> None:
        self.name = name
        self.weight = weight
        self.params = kwargs

    @abstractmethod
    def calculate(self, genome: dict, evaluator: Any = None) -> float:
        """
        Compute the raw (un-weighted) fitness contribution for *genome*.

        Args:
            genome:    The genome dict (keys: ``graph_dict``, ``oscillation``,
                       ``budget_info``).
            evaluator: Optional shared resource object (e.g. a cached env
                       factory).  May be ``None`` for self-contained components.

        Returns:
            A scalar fitness value (higher is better).
        """

    def reset(self) -> None:
        """
        Reset any per-evaluation state.

        Called by FitnessCalculator before each ``calculate()`` invocation.
        The default implementation is a no-op; override when the component
        carries persistent state across calls.
        """


# =============================================================================
# Built-in components
# =============================================================================

class DisplacementFitnessComponent(FitnessComponent):
    """
    Measures how far the robot's centre of mass travels during a simulation
    trial.

    This is the primary locomotion fitness component.  It is **robot-agnostic**:
    all robot-specific behaviour is injected through three callbacks:

    ``cfg_fn(genome) -> cfg``
        Builds a MetaMachine OmegaConf config from the full genome dict.
        **Required**.

    ``action_fn(genome, t, num_actions) -> ndarray``  *(optional)*
        Returns the action vector at simulation time *t*.  Defaults to zero
        actions (suitable for pose-optimised robots).

    ``num_actions_fn(genome) -> int``  *(optional)*
        Returns the number of actuated joints.  Defaults to reading
        ``cfg.control.num_actions`` from the built config.

    Parameters (passed as keyword args or via ``params`` dict):
        cfg_fn:            ``cfg_fn(genome) -> cfg`` (required).
        action_fn:         Optional action callback.
        num_actions_fn:    Optional action-count callback.
        eval_steps (int):  Number of simulation steps (default: 500).
        dt (float):        Control timestep in seconds (default: 0.05).
        early_stop (bool): Stop the trial if the episode terminates early
                           (default: True).

    Example (modular legs — zero actions, pose-optimised):
        comp = DisplacementFitnessComponent(
            "displacement", weight=1.0,
            cfg_fn=create_config_from_morphology,
            eval_steps=500,
        )

    Example (lego — sinusoidal oscillation):
        comp = DisplacementFitnessComponent(
            "displacement", weight=1.0,
            cfg_fn=lego_cfg_fn,
            action_fn=lego_action_fn,
            num_actions_fn=lego_num_actions_fn,
            eval_steps=500,
        )
    """

    def calculate(self, genome: dict, evaluator: Any = None) -> float:
        from metamachine.evolution.fitness_helpers import run_displacement_trial

        cfg_fn = self.params.get("cfg_fn")
        if cfg_fn is None:
            raise ValueError(
                "DisplacementFitnessComponent requires a 'cfg_fn' parameter: "
                "a callable that builds a MetaMachine config from a genome dict."
            )
        eval_steps = self.params.get("eval_steps", 500)
        dt = self.params.get("dt", 0.05)
        early_stop = self.params.get("early_stop", True)
        action_fn = self.params.get("action_fn", None)
        num_actions_fn = self.params.get("num_actions_fn", None)
        return run_displacement_trial(
            genome, cfg_fn=cfg_fn,
            eval_steps=eval_steps, dt=dt, early_stop=early_stop,
            action_fn=action_fn, num_actions_fn=num_actions_fn,
        )


class OscillationSmoothnessComponent(FitnessComponent):
    """
    Penalises jerky oscillation patterns by measuring the mean absolute
    difference between consecutive oscillation parameter values across joints.

    A smooth, coordinated gait should have low variance across joints and
    gentle parameter transitions.  This is a *bonus* metric — it does not
    require a simulation — so it can be evaluated cheaply.

    Fitness contribution: ``exp(-variance_score)`` ∈ (0, 1].

    Parameters:
        variance_weight (float): Relative weight for frequency variance vs
                                  amplitude variance (default: 0.5).

    Example YAML:
        - name: smoothness
          type: oscillation_smoothness
          weight: 0.2
          params:
            variance_weight: 0.5
    """

    def calculate(self, genome: dict, evaluator: Any = None) -> float:
        osc = genome.get("oscillation", {})
        if not osc:
            return 0.0

        amplitudes = [p["amplitude"] for p in osc.values()]
        frequencies = [p["frequency"] for p in osc.values()]

        amp_var = float(np.var(amplitudes)) if len(amplitudes) > 1 else 0.0
        freq_var = float(np.var(frequencies)) if len(frequencies) > 1 else 0.0

        variance_weight = self.params.get("variance_weight", 0.5)
        score = variance_weight * freq_var + (1.0 - variance_weight) * amp_var
        return math.exp(-score)


class SymmetryFitnessComponent(FitnessComponent):
    """
    Rewards bilateral symmetry of oscillation parameters.

    Assumes that joints are indexed left-right in pairs: joint 0 ↔ 1,
    2 ↔ 3, etc.  Symmetry is measured as the mean absolute difference
    between paired amplitude and frequency values, transformed to [0, 1]
    via ``exp(-diff)``.

    If the number of joints is odd the last joint is ignored.

    Parameters:
        amp_weight (float): Relative weight for amplitude symmetry vs
                            frequency symmetry (default: 0.5).

    Example YAML:
        - name: symmetry
          type: symmetry
          weight: 0.1
    """

    def calculate(self, genome: dict, evaluator: Any = None) -> float:
        osc = genome.get("oscillation", {})
        n = len(osc)
        if n < 2:
            return 1.0  # trivially symmetric

        amp_weight = self.params.get("amp_weight", 0.5)
        diffs = []
        for i in range(0, n - 1, 2):
            p_left = osc.get(i, {})
            p_right = osc.get(i + 1, {})
            amp_diff = abs(p_left.get("amplitude", 0.5) - p_right.get("amplitude", 0.5))
            freq_diff = abs(p_left.get("frequency", 1.0) - p_right.get("frequency", 1.0))
            diffs.append(amp_weight * amp_diff + (1.0 - amp_weight) * freq_diff)

        mean_diff = float(np.mean(diffs))
        return math.exp(-mean_diff)


class CompositeFitnessComponent(FitnessComponent):
    """
    Wraps a list of sub-components and combines them as a weighted sum.

    This allows hierarchical composition: a ``CompositeFitnessComponent``
    can itself be used as a single component inside a ``FitnessCalculator``,
    or nested arbitrarily.

    The outer weight (``self.weight``) is applied by ``FitnessCalculator``
    on top of the internal weighted sum.

    Parameters:
        components (list[FitnessComponent]): Sub-components to aggregate.

    Example (programmatic):
        body_fitness = CompositeFitnessComponent(
            "body",
            weight=1.0,
            components=[
                DisplacementFitnessComponent("disp", weight=1.0, eval_steps=300),
                SymmetryFitnessComponent("sym", weight=0.1),
            ],
        )
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        components: Optional[list[FitnessComponent]] = None,
        **kwargs,
    ) -> None:
        super().__init__(name, weight, **kwargs)
        self.components: list[FitnessComponent] = components or []

    def calculate(self, genome: dict, evaluator: Any = None) -> float:
        total = 0.0
        for comp in self.components:
            comp.reset()
            total += comp.weight * comp.calculate(genome, evaluator)
        return total

    def reset(self) -> None:
        for comp in self.components:
            comp.reset()


# =============================================================================
# Registry
# =============================================================================

FITNESS_COMPONENT_REGISTRY: dict[str, type[FitnessComponent]] = {
    "displacement": DisplacementFitnessComponent,
    "oscillation_smoothness": OscillationSmoothnessComponent,
    "symmetry": SymmetryFitnessComponent,
}


def register_fitness_component(name: str, component_class: type) -> None:
    """
    Register a custom FitnessComponent class under a string key.

    After registration, the component can be instantiated from a config list
    via ``FitnessCalculator.from_list()``.

    Args:
        name:            Registry key (used in YAML ``type:`` field).
        component_class: A subclass of ``FitnessComponent``.

    Raises:
        ValueError: If *component_class* is not a FitnessComponent subclass.
    """
    if not (isinstance(component_class, type)
            and issubclass(component_class, FitnessComponent)):
        raise ValueError(
            f"component_class must be a subclass of FitnessComponent, "
            f"got {component_class!r}"
        )
    FITNESS_COMPONENT_REGISTRY[name] = component_class


def list_available_fitness_components() -> list[str]:
    """Return names of all registered fitness component types."""
    return list(FITNESS_COMPONENT_REGISTRY.keys())


# =============================================================================
# FitnessCalculator
# =============================================================================

class FitnessCalculator:
    """
    Aggregates multiple weighted FitnessComponents into a single scalar fitness.

    Mirrors the interface of ``RewardCalculator`` in
    ``metamachine.environments.components.reward``.

    Usage
    -----
        calc = FitnessCalculator([
            DisplacementFitnessComponent("displacement", weight=1.0, eval_steps=500),
            OscillationSmoothnessComponent("smoothness", weight=0.2),
        ])
        fitness, info = calc.calculate(genome)

    Args:
        components: List of FitnessComponent instances.
        evaluator:  Optional shared resource object passed to every component's
                    ``calculate()`` call.  Useful for sharing a simulation
                    environment or a cache across components evaluated on the
                    same genome.
    """

    def __init__(
        self,
        components: list[FitnessComponent],
        evaluator: Any = None,
    ) -> None:
        if not components:
            raise ValueError("FitnessCalculator requires at least one component.")
        self.components = components
        self.evaluator = evaluator

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_list(
        cls,
        component_configs: list[dict],
        evaluator: Any = None,
    ) -> "FitnessCalculator":
        """
        Build a FitnessCalculator from a list of config dicts.

        Each dict may contain:
            type   (str, required): Registry key.
            name   (str):           Display name (defaults to ``type``).
            weight (float):         Multiplicative weight (default 1.0).
            params (dict):          Forwarded as keyword arguments.

        Example config list::

            [
                {"type": "displacement", "weight": 1.0, "params": {"eval_steps": 500}},
                {"type": "oscillation_smoothness", "weight": 0.2},
            ]

        Args:
            component_configs: List of component specification dicts.
            evaluator:         Optional shared evaluator object.

        Returns:
            Initialised FitnessCalculator.

        Raises:
            ValueError: If a ``type`` is not in the registry or the list is empty.
        """
        if not component_configs:
            raise ValueError("No fitness component configs provided.")

        components: list[FitnessComponent] = []
        for cfg in component_configs:
            ctype = cfg["type"]
            if ctype not in FITNESS_COMPONENT_REGISTRY:
                available = ", ".join(FITNESS_COMPONENT_REGISTRY.keys())
                raise ValueError(
                    f"Unknown fitness component type: '{ctype}'. "
                    f"Available: {available}"
                )
            name = cfg.get("name", ctype)
            weight = float(cfg.get("weight", 1.0))
            params = dict(cfg.get("params", {}))
            component_class = FITNESS_COMPONENT_REGISTRY[ctype]
            components.append(component_class(name, weight, **params))

        return cls(components, evaluator=evaluator)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def calculate(
        self, genome: dict
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute the total weighted fitness for *genome*.

        All components are reset before evaluation so stateful components
        (e.g. those tracking history across multiple calls) start fresh.

        Args:
            genome: The genome dict to evaluate.

        Returns:
            (total_fitness, info_dict) where info_dict contains:
                ``component_values``  – raw (un-weighted) score per component.
                ``component_weights`` – weight of each component.
                ``weighted_values``   – weight × raw score per component.
                ``total_fitness``     – final scalar fitness.
                ``num_components``    – number of components.
        """
        component_values: dict[str, float] = {}
        weighted_values: dict[str, float] = {}
        total_fitness = 0.0

        for comp in self.components:
            comp.reset()
            raw = comp.calculate(genome, self.evaluator)
            weighted = comp.weight * raw
            component_values[comp.name] = raw
            weighted_values[comp.name] = weighted
            total_fitness += weighted

        info: dict[str, Any] = {
            "component_values": component_values,
            "component_weights": {c.name: c.weight for c in self.components},
            "weighted_values": weighted_values,
            "total_fitness": total_fitness,
            "num_components": len(self.components),
        }
        return total_fitness, info

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_component(self, name: str) -> Optional[FitnessComponent]:
        """Return the component with the given name, or None."""
        for comp in self.components:
            if comp.name == name:
                return comp
        return None

    @property
    def component_names(self) -> list[str]:
        """Names of all registered components."""
        return [c.name for c in self.components]

    def __str__(self) -> str:
        lines = [f"FitnessCalculator with {len(self.components)} component(s):"]
        for comp in self.components:
            lines.append(
                f"  - {comp.name}: {comp.__class__.__name__} "
                f"(weight: {comp.weight})"
            )
        return "\n".join(lines)


# =============================================================================
# Factory function
# =============================================================================

def create_fitness_calculator(
    component_configs: list[dict],
    evaluator: Any = None,
) -> FitnessCalculator:
    """
    Factory function — mirrors ``create_reward_calculator()`` in reward.py.

    Args:
        component_configs: List of component spec dicts (see ``FitnessCalculator.from_list``).
        evaluator:         Optional shared evaluator object.

    Returns:
        Initialised FitnessCalculator.
    """
    return FitnessCalculator.from_list(component_configs, evaluator=evaluator)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "FitnessComponent",
    "FitnessCalculator",
    "DisplacementFitnessComponent",
    "OscillationSmoothnessComponent",
    "SymmetryFitnessComponent",
    "CompositeFitnessComponent",
    "FITNESS_COMPONENT_REGISTRY",
    "register_fitness_component",
    "list_available_fitness_components",
    "create_fitness_calculator",
]
