"""
Evolve Modular-Leg Robot Morphologies
======================================

Simple evolution script for modular-leg robots that leverages the
``metamachine.evolution`` package.

Pipeline per generation:
    mutate morphology & oscillation → constraint check → pose optimization → fitness

Pose optimization is integrated *inside* the MetaMachine environment when
the config has ``pose_optimization.enabled: true``.  So every time a new
environment is created for fitness evaluation, the MJX pose optimiser runs
automatically.  No separate optimisation call is needed here.

Genome representation
---------------------
A modular-leg genome is a dict with three keys:

    {
        "morphology": [int, ...],      # flat list of ints, groups of 4
        "num_modules": int,            # number of added modules (len/4)
        "oscillation": {               # sinusoidal controller per joint
            0: {"amplitude": float, "frequency": float, "phase": float},
            1: { ... },
            ...
        }
    }

Each group of 4 is ``[parent_module_id, parent_dock, child_dock, orientation]``.

Usage
-----
    # Quick smoke test (3 individuals, 2 generations)
    python examples/evolve_modular_legs.py --pop-size 3 --generations 2 \\
        --eval-steps 200

    # Typical run
    python examples/evolve_modular_legs.py --pop-size 20 --generations 30

    # With constraints
    python examples/evolve_modular_legs.py --pop-size 10 --generations 20 \\
        --min-modules 3 --joint-utility

Copyright 2026 Chen Yu <chenyu@u.northwestern.edu>
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import traceback
from typing import Any, Optional

from metamachine.robot_factory.modular_legs.config_builder import (
    create_config_from_morphology,
)
from metamachine.evolution.modular_leg_operators import (
    build_genome,
    mutate_genome,
    crossover_genomes,
    make_init_population_fn,
    oscillation_action,
)


# =============================================================================
# Genome → cfg adapter for the unified evolution system
# =============================================================================

# Default config name — set once from CLI, used by all adapters.
_CONFIG_NAME: str = "quadruped_pose_opt"
_POSE_OPT: bool = True


def modular_cfg_fn(genome: dict) -> object:
    """``cfg_fn(genome) -> cfg`` adapter for modular-leg genomes.

    Extracts ``genome["morphology"]`` and forwards to
    :func:`create_config_from_morphology`.  Used by
    ``DisplacementFitnessComponent`` and ``JointUtilityConstraint``.
    """
    return create_config_from_morphology(
        genome["morphology"],
        config_name=_CONFIG_NAME,
        pose_optimization=_POSE_OPT,
    )


def modular_action_fn(genome: dict, t: float, num_actions: int):
    """``action_fn(genome, t, num_actions) -> ndarray`` for modular-leg genomes.

    Reads the ``"oscillation"`` dict from the genome and computes a
    sinusoidal action vector at time *t*.
    """
    osc = genome.get("oscillation", {})
    return oscillation_action(osc, t, num_actions)


# =============================================================================
# Fitness evaluation — delegates to the unified DisplacementFitnessComponent
# =============================================================================

def evaluate_genome(
    genome: dict,
    eval_steps: int = 500,
    dt: float = 0.05,
    verbose: bool = False,
) -> float:
    """Evaluate a modular-leg genome via the unified evolution system.

    Uses :class:`~metamachine.evolution.fitness.DisplacementFitnessComponent`
    with :func:`modular_cfg_fn` and sinusoidal oscillation actions from the
    genome's ``"oscillation"`` parameters.

    Returns:
        Fitness = Euclidean XY displacement (metres).
    """
    from metamachine.evolution.fitness import DisplacementFitnessComponent

    comp = DisplacementFitnessComponent(
        "displacement",
        weight=1.0,
        cfg_fn=modular_cfg_fn,
        action_fn=modular_action_fn,
        eval_steps=eval_steps,
        dt=dt,
        early_stop=True,
    )
    return comp.calculate(genome)


# =============================================================================
# Video recording helpers
# =============================================================================

def _cleanup_tmpdir(tmpdir: Optional[str]) -> None:
    """Remove a temporary directory used during evaluation."""
    if tmpdir is None:
        return
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass


def record_best_video(
    genome: dict,
    video_path: str,
    eval_steps: int = 500,
    dt: float = 0.05,
    verbose: bool = True,
) -> Optional[str]:
    """Record an mp4 video of a modular-leg genome's behaviour.

    Args:
        genome:     Modular-leg genome dict.
        video_path: Full path for the output .mp4 file.
        eval_steps: Number of simulation steps to record.
        dt:         Control timestep.
        verbose:    Print progress info.

    Returns:
        The path to the saved video, or ``None`` if recording failed.
    """
    from metamachine.environments.env_sim import MetaMachine

    morphology = genome["morphology"]

    cfg = create_config_from_morphology(
        morphology,
        config_name=_CONFIG_NAME,
        pose_optimization=_POSE_OPT,
    )

    # Configure for mp4 recording
    cfg.simulation.render_mode = "mp4"
    cfg.simulation.render = True
    video_dir = os.path.dirname(video_path) or "."
    video_name = os.path.basename(video_path)
    os.makedirs(video_dir, exist_ok=True)
    cfg.simulation.video_path = video_dir
    cfg.simulation.video_name_pattern = video_name
    cfg.simulation.video_record_interval = 1

    try:
        env = MetaMachine(cfg)
    except Exception as e:
        if verbose:
            print(f"  [RECORD] Env creation failed: {e}")
        return None

    saved_path = None
    try:
        num_actions = cfg.control.num_actions
        osc_params = genome.get("oscillation", {})
        obs, info = env.reset()
        for step in range(eval_steps):
            t = step * dt
            action = oscillation_action(osc_params, t, num_actions)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        # Trigger video save
        env._post_done()
        expected = os.path.join(video_dir, video_name)
        if not expected.endswith(".mp4"):
            expected += ".mp4"
        if os.path.exists(expected):
            saved_path = expected
    except Exception as e:
        if verbose:
            print(f"  [RECORD] Simulation/recording failed: {e}")
            traceback.print_exc()
    finally:
        try:
            env.close()
        except Exception:
            pass

    if verbose and saved_path:
        print(f"  [RECORD] Video saved: {saved_path}")
    return saved_path


def visualize_best(genome: dict, eval_steps: int = 500, dt: float = 0.05):
    """Re-run the best genome with a viewer."""
    from metamachine.environments.env_sim import MetaMachine

    cfg = create_config_from_morphology(
        genome["morphology"],
        config_name=_CONFIG_NAME,
        pose_optimization=_POSE_OPT,
    )
    cfg.simulation.render_mode = "viewer"
    cfg.simulation.render = True

    num_actions = cfg.control.num_actions
    osc_params = genome.get("oscillation", {})
    env = MetaMachine(cfg)
    obs, info = env.reset()
    for step in range(eval_steps):
        t = step * dt
        action = oscillation_action(osc_params, t, num_actions)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()
    env.close()


def make_record_best_callback(
    log_dir: str,
    eval_steps: int = 500,
    verbose: bool = True,
):
    """Create a callback that records a video whenever a new best-ever is found.

    Returns:
        A callback function ``cb(engine, gen_stats)``.
    """
    _prev_best_fitness: list[Optional[float]] = [None]

    videos_dir = os.path.join(log_dir, "best_videos")
    os.makedirs(videos_dir, exist_ok=True)

    def _callback(engine, gen_stats):
        best = engine.best_ever
        if best is None:
            return

        # Only record when best-ever actually improved
        new_best = False
        if _prev_best_fitness[0] is None:
            new_best = True  # first generation
        elif best.fitness is not None and best.fitness > (_prev_best_fitness[0] or 0):
            new_best = True

        if not new_best:
            return

        _prev_best_fitness[0] = best.fitness

        gen = gen_stats["generation"]
        fit_str = f"{best.fitness:.4f}" if best.fitness is not None else "0"
        video_name = f"best_gen{gen:04d}_id{best.id}_fit{fit_str}.mp4"
        video_path = os.path.join(videos_dir, video_name)

        if verbose:
            print(f"  [RECORD] New best ever! Recording video → {video_name}")

        record_best_video(
            genome=best.genome,
            video_path=video_path,
            eval_steps=eval_steps,
            verbose=verbose,
        )

    return _callback


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve modular-leg robot morphologies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Quick smoke test
    python examples/evolve_modular_legs.py --pop-size 3 --generations 2 --eval-steps 200

    # Typical run
    python examples/evolve_modular_legs.py --pop-size 20 --generations 30

    # With constraints
    python examples/evolve_modular_legs.py --pop-size 10 --generations 20 \\
        --min-modules 3 --joint-utility
""",
    )

    # Population / generation
    parser.add_argument("--pop-size", type=int, default=10, help="Population size (μ)")
    parser.add_argument(
        "--offspring", type=int, default=None, help="Offspring count (λ); defaults to pop-size"
    )
    parser.add_argument("--generations", type=int, default=20, help="Number of generations")

    # Morphology
    parser.add_argument(
        "--num-modules", type=int, default=4, help="Number of modules per robot"
    )

    # Fitness evaluation
    parser.add_argument("--eval-steps", type=int, default=500, help="Sim steps per trial")
    parser.add_argument("--dt", type=float, default=0.05, help="Control timestep")
    parser.add_argument(
        "--config-name",
        type=str,
        default="quadruped_pose_opt",
        help="Config registry name (should have pose_optimization.enabled)",
    )
    parser.add_argument(
        "--no-pose-opt",
        action="store_true",
        help="Disable MJX pose optimisation (much faster, but robots start "
             "with default joint positions)",
    )

    # EA hyper-params
    parser.add_argument(
        "--selection",
        choices=["tournament", "truncation", "roulette"],
        default="tournament",
    )
    parser.add_argument("--tournament-k", type=int, default=3)
    parser.add_argument("--crossover-rate", type=float, default=0.3)
    parser.add_argument("--elitism", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)

    # Constraints
    parser.add_argument(
        "--min-modules", type=int, default=0, help="Minimum modules constraint"
    )
    parser.add_argument(
        "--joint-utility",
        action="store_true",
        help="Enable joint-utility constraint (reject robots with useless joints)",
    )
    parser.add_argument(
        "--joint-utility-threshold",
        type=float,
        default=0.005,
        help="Movement threshold for joint utility (metres)",
    )
    parser.add_argument(
        "--joint-utility-steps",
        type=int,
        default=100,
        help="Number of probe steps per joint",
    )

    # Logging
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--record-best",
        action="store_true",
        help="Record a video each time a new best-ever individual is found",
    )
    parser.add_argument(
        "--visualize-best",
        action="store_true",
        help="Open an interactive viewer for the best robot after evolution",
    )

    # Resume
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint .pkl to resume from"
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    offspring_size = args.offspring if args.offspring is not None else args.pop_size

    # Setup log dir
    if args.log_dir is None:
        log_dir = os.path.join(
            "logs",
            f"evo_modular_{time.strftime('%Y%m%d_%H%M%S')}",
        )
    else:
        log_dir = args.log_dir

    print("=" * 60)
    print("Modular-Leg Robot Morphology Evolution")
    print("=" * 60)
    print(f"  Population:  {args.pop_size}, Offspring: {offspring_size}")
    print(f"  Generations: {args.generations}")
    print(f"  Modules:     {args.num_modules} per robot")
    print(f"  Eval steps:  {args.eval_steps}")
    print(f"  Config:      {args.config_name}")
    print(f"  Pose opt:    {'OFF' if args.no_pose_opt else 'ON'}")
    print(f"  Log dir:     {log_dir}")
    print()

    # Import evolution engine + constraint system
    from metamachine.evolution import EvolutionEngine
    from metamachine.evolution.constraint import (
        ConstraintChecker, MinModulesConstraint, JointUtilityConstraint,
    )

    # Set the module-level config name so modular_cfg_fn uses the right config
    global _CONFIG_NAME, _POSE_OPT
    _CONFIG_NAME = args.config_name
    _POSE_OPT = not args.no_pose_opt

    # Build evaluation function
    eval_steps = args.eval_steps
    dt = args.dt
    verbose = not args.quiet

    def evaluate_fn(genome: dict) -> float:
        return evaluate_genome(
            genome,
            eval_steps=eval_steps,
            dt=dt,
            verbose=verbose,
        )

    # Build constraint checker
    constraints: list[Any] = []
    if args.min_modules > 0:
        constraints.append(MinModulesConstraint(min_modules=args.min_modules))
    if args.joint_utility:
        constraints.append(
            JointUtilityConstraint(
                cfg_fn=modular_cfg_fn,
                movement_threshold=args.joint_utility_threshold,
                probe_steps=args.joint_utility_steps,
            )
        )

    if constraints:
        checker = ConstraintChecker(constraints, cfg_fn=modular_cfg_fn)
        evaluate_fn = checker.guarded_evaluate(evaluate_fn, penalty=0.0, verbose=verbose)
        print(f"  Constraints: {[c.name for c in constraints]}")
        print()

    # Create evolution engine
    engine = EvolutionEngine(
        population_size=args.pop_size,
        offspring_size=offspring_size,
        mutate_fn=mutate_genome,
        crossover_fn=crossover_genomes,
        evaluate_fn=evaluate_fn,
        init_population_fn=make_init_population_fn(num_modules=args.num_modules),
        selection=args.selection,
        tournament_k=args.tournament_k,
        crossover_rate=args.crossover_rate,
        elitism=args.elitism,
        seed=args.seed,
        log_dir=log_dir,
    )

    # Resume if requested
    if args.resume:
        print(f"Resuming from: {args.resume}")
        engine.load_checkpoint(args.resume)

    # Build callback for recording best-ever videos
    record_callback = None
    if args.record_best:
        record_callback = make_record_best_callback(
            log_dir=log_dir,
            eval_steps=args.eval_steps,
            verbose=verbose,
        )

    # Run evolution
    best = engine.run(
        generations=args.generations,
        verbose=not args.quiet,
        checkpoint_interval=5,
        callback=record_callback,
    )

    print()
    print("=" * 60)
    print("Best individual:")
    print(f"  Fitness:    {best.fitness:.4f} m displacement")
    print(f"  Morphology: {best.genome['morphology']}")
    print(f"  Modules:    {len(best.genome['morphology']) // 4}")
    osc = best.genome.get("oscillation", {})
    if osc:
        print(f"  Oscillation ({len(osc)} joints):")
        for j, p in sorted(osc.items()):
            print(f"    Joint {j}: A={p['amplitude']:.3f}, "
                  f"f={p['frequency']:.3f} Hz, φ={p['phase']:.3f} rad")
    print("=" * 60)

    # Visualize best in interactive viewer
    if args.visualize_best:
        print("\nVisualizing best robot...")
        visualize_best(best.genome, eval_steps=args.eval_steps)


if __name__ == "__main__":
    main()
