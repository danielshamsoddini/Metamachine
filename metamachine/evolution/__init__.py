"""
MetaMachine Evolution Package

Component-based evolutionary algorithm framework with:
- FitnessComponent: pluggable, composable fitness functions (mirrors reward.py style)
- FitnessCalculator: aggregates weighted FitnessComponents into a scalar fitness
- GenomeConstraint: composable hard/soft genome feasibility rules
- ConstraintChecker: aggregates constraints, wraps evaluate_fn for zero-code-change use
- EvolutionEngine: generic (μ+λ) evolutionary strategy
- Individual / selection operators
- Modular-leg genome operators (sample, mutate, crossover)

Typical usage
-------------
    from metamachine.evolution import (
        EvolutionEngine,
        Individual,
        FitnessCalculator,
        DisplacementFitnessComponent,
        register_fitness_component,
        ConstraintChecker,
        MinBallsConstraint,
        JointUtilityConstraint,
    )

    # Modular-leg operators
    from metamachine.evolution.modular_leg_operators import (
        build_genome, mutate_genome, crossover_genomes,
        make_init_population_fn, oscillation_action, make_oscillation,
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
from .constraint import (
    ConstraintResult,
    ConstraintReport,
    GenomeConstraint,
    MinModulesConstraint,
    MinBallsConstraint,
    MinLegsConstraint,
    SelfCollisionConstraint,
    JointUtilityConstraint,
    ConstraintChecker,
    CONSTRAINT_REGISTRY,
    register_constraint,
    list_available_constraints,
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
    # Constraint
    "ConstraintResult",
    "ConstraintReport",
    "GenomeConstraint",
    "MinModulesConstraint",
    "MinBallsConstraint",
    "MinLegsConstraint",
    "SelfCollisionConstraint",
    "JointUtilityConstraint",
    "ConstraintChecker",
    "CONSTRAINT_REGISTRY",
    "register_constraint",
    "list_available_constraints",
]
