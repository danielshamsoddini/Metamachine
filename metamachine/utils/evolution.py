"""
Evolutionary Algorithm Utilities

Generic evolutionary algorithm components that can be used with any robot type.
Provides:
- Individual: Wraps a genome (e.g., RobotGraph) with fitness tracking
- EvolutionEngine: Configurable (μ+λ) or (μ,λ) evolutionary strategy
- Selection operators: tournament, truncation, roulette
- Logging and checkpointing

This module is robot-agnostic. Robot-specific logic (mutation, crossover,
fitness evaluation) is injected via callables.

Usage:
    from metamachine.utils.evolution import EvolutionEngine, Individual

    engine = EvolutionEngine(
        population_size=20,
        offspring_size=20,
        mutate_fn=my_mutate,
        crossover_fn=my_crossover,
        evaluate_fn=my_evaluate,
    )
    best = engine.run(generations=50)

Copyright 2025 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import copy
import json
import os
import pickle
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TypeVar

import numpy as np


# =============================================================================
# Individual
# =============================================================================

@dataclass
class Individual:
    """
    An individual in the evolutionary population.

    Attributes:
        genome: The genome representation (e.g., RobotGraph dict, parameter vector).
        fitness: Evaluated fitness score (None if not yet evaluated).
        metadata: Arbitrary metadata (generation born, parent ids, etc.).
        id: Unique identifier.
    """
    genome: Any
    fitness: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    id: int = 0

    _id_counter: int = 0  # class-level counter

    def __post_init__(self):
        if self.id == 0:
            Individual._id_counter += 1
            self.id = Individual._id_counter

    def copy(self) -> "Individual":
        """Deep copy the individual (resets fitness)."""
        new = Individual(
            genome=copy.deepcopy(self.genome),
            fitness=None,
            metadata=copy.deepcopy(self.metadata),
        )
        return new

    @property
    def is_evaluated(self) -> bool:
        return self.fitness is not None

    def __repr__(self) -> str:
        fit_str = f"{self.fitness:.4f}" if self.fitness is not None else "N/A"
        return f"Individual(id={self.id}, fitness={fit_str})"


# =============================================================================
# Selection Operators
# =============================================================================

def tournament_selection(
    population: list[Individual],
    k: int = 3,
) -> Individual:
    """Select one individual via tournament selection."""
    contestants = random.sample(population, min(k, len(population)))
    return max(contestants, key=lambda ind: ind.fitness or float("-inf"))


def truncation_selection(
    population: list[Individual],
    n: int,
) -> list[Individual]:
    """Select top-n individuals by fitness."""
    ranked = sorted(
        population, key=lambda ind: ind.fitness or float("-inf"), reverse=True
    )
    return ranked[:n]


def roulette_selection(
    population: list[Individual],
) -> Individual:
    """Fitness-proportionate (roulette wheel) selection."""
    fitnesses = np.array([ind.fitness or 0.0 for ind in population])
    # Shift so minimum is 0
    shifted = fitnesses - fitnesses.min()
    total = shifted.sum()
    if total == 0:
        return random.choice(population)
    probs = shifted / total
    idx = np.random.choice(len(population), p=probs)
    return population[idx]


# =============================================================================
# Evolution Engine
# =============================================================================

class EvolutionEngine:
    """
    A generic (μ+λ) evolutionary strategy engine.

    The user provides:
    - mutate_fn(genome) -> genome : Mutation operator
    - crossover_fn(genome1, genome2) -> genome : Crossover operator (optional)
    - evaluate_fn(genome) -> float : Fitness evaluation
    - init_population_fn(n) -> list[genome] : Population initialiser

    Args:
        population_size: Number of parents (μ).
        offspring_size: Number of offspring per generation (λ).
        mutate_fn: Mutation function.
        evaluate_fn: Fitness evaluation function.
        crossover_fn: Optional crossover function.
        init_population_fn: Optional function to create initial genomes.
        selection: Selection method ("tournament", "truncation", "roulette").
        tournament_k: Tournament size (if selection="tournament").
        crossover_rate: Probability of crossover vs. mutation-only.
        elitism: Number of elites to carry over unchanged.
        seed: Random seed.
        log_dir: Directory for evolution logs.
    """

    def __init__(
        self,
        population_size: int = 20,
        offspring_size: int = 20,
        mutate_fn: Optional[Callable] = None,
        evaluate_fn: Optional[Callable] = None,
        crossover_fn: Optional[Callable] = None,
        init_population_fn: Optional[Callable] = None,
        selection: str = "tournament",
        tournament_k: int = 3,
        crossover_rate: float = 0.3,
        elitism: int = 2,
        seed: Optional[int] = None,
        log_dir: Optional[str] = None,
    ):
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutate_fn = mutate_fn
        self.evaluate_fn = evaluate_fn
        self.crossover_fn = crossover_fn
        self.init_population_fn = init_population_fn
        self.selection_method = selection
        self.tournament_k = tournament_k
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.seed = seed

        # Setup logging
        if log_dir is None:
            log_dir = os.path.join("logs", f"evolution_{int(time.time())}")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # State
        self.population: list[Individual] = []
        self.generation: int = 0
        self.history: list[dict[str, Any]] = []
        self.best_ever: Optional[Individual] = None

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # -----------------------------------------------------------------
    # Core loop
    # -----------------------------------------------------------------

    def run(
        self,
        generations: int = 50,
        initial_genomes: Optional[list[Any]] = None,
        verbose: bool = True,
        checkpoint_interval: int = 5,
        callback: Optional[Callable] = None,
    ) -> Individual:
        """
        Run the evolutionary loop.

        Args:
            generations: Number of generations.
            initial_genomes: Optional pre-built genomes for the initial population.
            verbose: Print progress.
            checkpoint_interval: Save checkpoint every N generations.
            callback: Called after each generation with (engine, gen_stats).

        Returns:
            Best individual found.
        """
        assert self.mutate_fn is not None, "mutate_fn is required"
        assert self.evaluate_fn is not None, "evaluate_fn is required"

        # --- Initialise population ---
        if initial_genomes is not None:
            self.population = [Individual(genome=g) for g in initial_genomes]
        elif self.init_population_fn is not None:
            genomes = self.init_population_fn(self.population_size)
            self.population = [Individual(genome=g) for g in genomes]
        else:
            raise ValueError(
                "Provide either initial_genomes or init_population_fn"
            )

        # Pad or trim to population_size
        while len(self.population) < self.population_size:
            parent = random.choice(self.population)
            child_genome = self.mutate_fn(copy.deepcopy(parent.genome))
            self.population.append(Individual(genome=child_genome))
        self.population = self.population[: self.population_size]

        # --- Evaluate initial population ---
        self._evaluate_population(self.population, verbose=verbose)
        self._update_best()

        if verbose:
            self._print_header()

        # Fire callback for generation 0 (initial population)
        if callback:
            init_stats = self._log_generation(0)
            callback(self, init_stats)
            # Remove the duplicate entry so the main loop doesn't double-count
            self.history.pop()

        # --- Main loop ---
        for gen in range(generations):
            self.generation = gen + 1
            t0 = time.time()

            # 1. Create offspring
            offspring = self._create_offspring()

            # 2. Evaluate offspring
            self._evaluate_population(offspring, verbose=verbose)

            # 3. Survivor selection (μ+λ with elitism)
            self.population = self._select_survivors(offspring)

            # 4. Update best
            self._update_best()

            # 5. Log
            gen_stats = self._log_generation(time.time() - t0)

            if verbose:
                self._print_generation(gen_stats)

            if callback:
                callback(self, gen_stats)

            # 6. Checkpoint
            if checkpoint_interval and (self.generation % checkpoint_interval == 0):
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint()
        self._save_history()

        if verbose:
            print(f"\n{'='*60}")
            print(f"Evolution complete! Best fitness: {self.best_ever.fitness:.4f}")
            print(f"Logs saved to: {self.log_dir}")

        return self.best_ever

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _evaluate_population(
        self, pop: list[Individual], verbose: bool = False
    ) -> None:
        """Evaluate fitness for all unevaluated individuals."""
        for i, ind in enumerate(pop):
            if ind.is_evaluated:
                continue
            try:
                ind.fitness = self.evaluate_fn(ind.genome)
            except Exception as e:
                print(f"  [WARN] Evaluation failed for individual {ind.id}: {e}")
                ind.fitness = 0.0
            if verbose:
                tag = "init" if self.generation == 0 else f"gen{self.generation}"
                print(
                    f"  [{tag}] Evaluated {i+1}/{len(pop)} "
                    f"(id={ind.id}) -> fitness={ind.fitness:.4f}"
                )

    def _select_parent(self) -> Individual:
        """Select one parent from the current population."""
        if self.selection_method == "tournament":
            return tournament_selection(self.population, self.tournament_k)
        elif self.selection_method == "roulette":
            return roulette_selection(self.population)
        else:
            # Default to tournament
            return tournament_selection(self.population, self.tournament_k)

    def _create_offspring(self) -> list[Individual]:
        """Create offspring via mutation and optional crossover."""
        offspring = []
        for _ in range(self.offspring_size):
            if (
                self.crossover_fn is not None
                and random.random() < self.crossover_rate
            ):
                p1 = self._select_parent()
                p2 = self._select_parent()
                child_genome = self.crossover_fn(
                    copy.deepcopy(p1.genome), copy.deepcopy(p2.genome)
                )
                child = Individual(
                    genome=child_genome,
                    metadata={
                        "parents": [p1.id, p2.id],
                        "born_gen": self.generation,
                        "operator": "crossover+mutation",
                    },
                )
            else:
                parent = self._select_parent()
                child_genome = self.mutate_fn(copy.deepcopy(parent.genome))
                child = Individual(
                    genome=child_genome,
                    metadata={
                        "parents": [parent.id],
                        "born_gen": self.generation,
                        "operator": "mutation",
                    },
                )
            offspring.append(child)
        return offspring

    def _select_survivors(self, offspring: list[Individual]) -> list[Individual]:
        """(μ+λ) selection with elitism."""
        # Elites from current population
        elites = truncation_selection(self.population, self.elitism)
        # Keep elites as-is (preserve fitness)
        elite_copies = []
        for e in elites:
            ec = Individual(
                genome=copy.deepcopy(e.genome),
                fitness=e.fitness,
                metadata={**e.metadata, "elite": True},
                id=e.id,
            )
            elite_copies.append(ec)

        # Pool = offspring + rest of current population
        pool = offspring + [
            ind for ind in self.population if ind.id not in {e.id for e in elites}
        ]

        # Select best from pool to fill remaining slots
        remaining_slots = self.population_size - len(elite_copies)
        selected = truncation_selection(pool, remaining_slots)

        return elite_copies + selected

    def _update_best(self) -> None:
        """Update the best-ever individual."""
        current_best = max(
            self.population, key=lambda ind: ind.fitness or float("-inf")
        )
        if self.best_ever is None or (
            current_best.fitness is not None
            and (self.best_ever.fitness is None or current_best.fitness > self.best_ever.fitness)
        ):
            self.best_ever = Individual(
                genome=copy.deepcopy(current_best.genome),
                fitness=current_best.fitness,
                metadata=copy.deepcopy(current_best.metadata),
                id=current_best.id,
            )

    def _log_generation(self, elapsed: float) -> dict[str, Any]:
        """Record generation statistics."""
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        stats = {
            "generation": self.generation,
            "best_fitness": max(fitnesses) if fitnesses else 0.0,
            "mean_fitness": float(np.mean(fitnesses)) if fitnesses else 0.0,
            "std_fitness": float(np.std(fitnesses)) if fitnesses else 0.0,
            "min_fitness": min(fitnesses) if fitnesses else 0.0,
            "best_ever_fitness": self.best_ever.fitness if self.best_ever else 0.0,
            "best_ever_id": self.best_ever.id if self.best_ever else None,
            "population_size": len(self.population),
            "elapsed_seconds": elapsed,
        }
        self.history.append(stats)
        return stats

    # -----------------------------------------------------------------
    # I/O
    # -----------------------------------------------------------------

    def save_checkpoint(self, path: Optional[str] = None) -> str:
        """Save a checkpoint of the current state."""
        if path is None:
            path = os.path.join(
                self.log_dir, f"checkpoint_gen{self.generation:04d}.pkl"
            )
        data = {
            "generation": self.generation,
            "population": self.population,
            "best_ever": self.best_ever,
            "history": self.history,
            "seed": self.seed,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load state from a checkpoint."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.generation = data["generation"]
        self.population = data["population"]
        self.best_ever = data["best_ever"]
        self.history = data["history"]
        self.seed = data.get("seed")

    def _save_history(self) -> None:
        """Save evolution history as JSON."""
        path = os.path.join(self.log_dir, "evolution_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

    # -----------------------------------------------------------------
    # Printing
    # -----------------------------------------------------------------

    def _print_header(self) -> None:
        print(f"\n{'='*70}")
        print(f"{'Gen':>4} | {'Best':>10} | {'Mean':>10} | {'Std':>8} | {'Best Ever':>10} | {'Time':>6}")
        print(f"{'-'*70}")
        # Print gen 0 stats
        fitnesses = [ind.fitness for ind in self.population if ind.fitness is not None]
        best = max(fitnesses) if fitnesses else 0.0
        mean = float(np.mean(fitnesses)) if fitnesses else 0.0
        std = float(np.std(fitnesses)) if fitnesses else 0.0
        best_ever = self.best_ever.fitness if self.best_ever else 0.0
        print(f"{'0':>4} | {best:>10.4f} | {mean:>10.4f} | {std:>8.4f} | {best_ever:>10.4f} | {'init':>6}")

    def _print_generation(self, stats: dict[str, Any]) -> None:
        print(
            f"{stats['generation']:>4} | "
            f"{stats['best_fitness']:>10.4f} | "
            f"{stats['mean_fitness']:>10.4f} | "
            f"{stats['std_fitness']:>8.4f} | "
            f"{stats['best_ever_fitness']:>10.4f} | "
            f"{stats['elapsed_seconds']:>5.1f}s"
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Individual",
    "EvolutionEngine",
    "tournament_selection",
    "truncation_selection",
    "roulette_selection",
]
