"""
MetaMachine Evolution Package

Component-based evolutionary algorithm framework with:
- FitnessComponent: pluggable, composable fitness functions (mirrors reward.py style)
- FitnessCalculator: aggregates weighted FitnessComponents into a scalar fitness
- EvolutionEngine: generic (μ+λ) evolutionary strategy
- Individual / selection operators

Typical usage
-------------
    from metamachine.evolution import (
        EvolutionEngine,
        Individual,
        FitnessCalculator,
        DisplacementFitnessComponent,
        register_fitness_component,
    )

Copyright 2026 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from .engine import (
    EvolutionEngine,
    Individual,
    tournament_selection,
    truncation_selection,
    roulette_selection,
)
from .fitness import (
    FitnessComponent,
    FitnessCalculator,
    DisplacementFitnessComponent,
    OscillationSmoothnessComponent,
    SymmetryFitnessComponent,
    CompositeFitnessComponent,
    create_fitness_calculator,
    register_fitness_component,
    list_available_fitness_components,
    FITNESS_COMPONENT_REGISTRY,
)

__all__ = [
    # Engine
    "EvolutionEngine",
    "Individual",
    "tournament_selection",
    "truncation_selection",
    "roulette_selection",
    # Fitness
    "FitnessComponent",
    "FitnessCalculator",
    "DisplacementFitnessComponent",
    "OscillationSmoothnessComponent",
    "SymmetryFitnessComponent",
    "CompositeFitnessComponent",
    "create_fitness_calculator",
    "register_fitness_component",
    "list_available_fitness_components",
    "FITNESS_COMPONENT_REGISTRY",
]
