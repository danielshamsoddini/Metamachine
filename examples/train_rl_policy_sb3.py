#!/usr/bin/env python3
"""
Train or play RL policies with Stable Baselines 3.

This is the general-purpose SB3 example script. It supports:
1. New training runs
2. Continue training / fine-tuning from checkpoints
3. Policy playback in simulation or on a real robot
4. Direct playback of exported `.pkl` policies
5. Open-loop sine playback without a trained policy
6. Real-robot idle / manual-position flows
7. Optional plugin loading for custom robot factories
8. Joint tracking plots and saved tracking/state logs

Usage:
    # Train with the default config
    python examples/train_rl_policy_sb3.py

    # Train with a custom config name or YAML path
    python examples/train_rl_policy_sb3.py --config modular_quadruped
    python examples/train_rl_policy_sb3.py --config ./my_robot.yaml

    # Continue training
    python examples/train_rl_policy_sb3.py --continue ./logs/my_experiment

    # Play a trained policy
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --real-robot
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --zero-action

    # Play an exported CrossQ .pkl policy
    python examples/train_rl_policy_sb3.py --policy-pkl ./policy_params.pkl --config real_one_module
    python examples/train_rl_policy_sb3.py --policy-pkl ./policy_params.pkl --config real_one_module --zero-action

    # Run open-loop sine playback
    python examples/train_rl_policy_sb3.py --openloop-sine
    python examples/train_rl_policy_sb3.py --openloop-sine --real-robot
    python examples/train_rl_policy_sb3.py --openloop-sine --zero-action
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


# Keep repo-root imports working when the script is launched directly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Global defaults
DEFAULT_CONFIG = "example_three_modules"
DEFAULT_SEED = 42
DEFAULT_TIMESTEPS = 1_000_000
DEFAULT_EXP_NAME = "Train 3 Modules"
DEFAULT_ALGORITHM = "CrossQ"
DEFAULT_CHECKPOINT_FREQ = 100_000
DEFAULT_PLUGIN_DIR = PROJECT_ROOT / "metamachine_plugins" / "private_plugins"
CAPYRL_CANDIDATE_DIRS = (
    PROJECT_ROOT.parent / "CapyRL",
    PROJECT_ROOT.parent / "capyrl_dev",
    PROJECT_ROOT.parent / "CapybaraRL",
)


def _default_plugin_dirs() -> list[str]:
    """Return default plugin directories that exist in this repo."""
    dirs = []
    if DEFAULT_PLUGIN_DIR.exists():
        dirs.append(str(DEFAULT_PLUGIN_DIR))
    return dirs


def _unique_paths(paths: Iterable[str]) -> list[str]:
    """Return unique paths while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        resolved = str(Path(path).expanduser())
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL policies with Stable Baselines 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with the default config
    python examples/train_rl_policy_sb3.py

    # Train with plugins for custom robot factories
    python examples/train_rl_policy_sb3.py --config ./robot.yaml --plugin-dir ./private_plugins

    # Continue training / fine-tune from checkpoint
    python examples/train_rl_policy_sb3.py --continue ./logs/my_experiment
    python examples/train_rl_policy_sb3.py --continue ./logs/my_experiment --config ./new_config.yaml

    # Play / visualize a trained policy
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --checkpoint 200000
    python examples/train_rl_policy_sb3.py --policy-pkl ./policy_params.pkl --config real_one_module
    python examples/train_rl_policy_sb3.py --policy-pkl ./policy_params.pkl --config real_one_module --real-robot
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --zero-action

    # Real robot playback with explicit module IDs
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --real-robot --module-ids 5 21 27
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --real-robot --viewer
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --real-robot --manual-position
    python examples/train_rl_policy_sb3.py --real-robot --viewer --zero-action

    # Open-loop sine playback without a policy
    python examples/train_rl_policy_sb3.py --openloop-sine
    python examples/train_rl_policy_sb3.py --openloop-sine --openloop-case zero_then_sine
    python examples/train_rl_policy_sb3.py --openloop-sine --openloop-amplitude 0.15 0.15 0.15
    python examples/train_rl_policy_sb3.py --openloop-sine --zero-action

    # Tracking / logging tools during playback
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --plot-tracking
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --save-tracking ./tracking_data.npz
    python examples/train_rl_policy_sb3.py --play ./logs/my_experiment --save-state ./state_data.pkl
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=DEFAULT_CONFIG,
        help=f"Config name or config YAML path (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--timesteps",
        "-t",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS})",
    )
    parser.add_argument(
        "--exp-name",
        "-n",
        type=str,
        default=DEFAULT_EXP_NAME,
        help=f"Experiment name (default: {DEFAULT_EXP_NAME})",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default=DEFAULT_ALGORITHM,
        help="RL algorithm: CrossQ, SAC, PPO, TD3, A2C, DDPG, DQN, TQC, TRPO, ARS, RecurrentPPO",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering during training",
    )
    parser.add_argument(
        "--plugin-dir",
        action="append",
        default=None,
        metavar="PATH",
        help="Plugin directory to load custom robot factories from. Can be passed multiple times.",
    )
    parser.add_argument(
        "--no-default-plugins",
        action="store_true",
        help="Do not auto-load the repo default plugin directory if it exists",
    )

    parser.add_argument(
        "--continue",
        dest="continue_from",
        type=str,
        default=None,
        metavar="LOG_DIR",
        help="Continue training from a log directory",
    )

    parser.add_argument(
        "--play",
        "-p",
        type=str,
        default=None,
        metavar="LOG_DIR",
        help="Play / visualize a trained policy from a log directory",
    )
    parser.add_argument(
        "--policy-pkl",
        type=str,
        default=None,
        metavar="PATH",
        help="Play a CrossQ policy exported as a .pkl file",
    )
    parser.add_argument(
        "--openloop-sine",
        action="store_true",
        help="Play the robot without a policy using an open-loop sine action sequence",
    )
    parser.add_argument(
        "--openloop-case",
        type=str,
        default=None,
        help="Named sim_to_real_check case from config to use for open-loop playback",
    )
    parser.add_argument(
        "--openloop-hold-sec",
        type=float,
        default=None,
        help="Optional initial zero-action hold before the sine segment",
    )
    parser.add_argument(
        "--openloop-duration-sec",
        type=float,
        default=None,
        help="Duration of the sine segment in seconds",
    )
    parser.add_argument(
        "--openloop-amplitude",
        type=float,
        nargs="+",
        default=None,
        help="Sine amplitude per joint (scalar or one value per joint)",
    )
    parser.add_argument(
        "--openloop-frequency-hz",
        type=float,
        default=None,
        help="Sine frequency in Hz",
    )
    parser.add_argument(
        "--openloop-offset",
        type=float,
        nargs="+",
        default=None,
        help="Sine offset per joint (scalar or one value per joint)",
    )
    parser.add_argument(
        "--openloop-phase",
        type=float,
        nargs="+",
        default=None,
        help="Sine phase per joint in radians (scalar or one value per joint)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="latest",
        help="Checkpoint to load: latest, final, best, or step number",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to play (0 = infinite)",
    )
    parser.add_argument(
        "--zero-action",
        action="store_true",
        help="Playback only: override all actions with zeros",
    )
    parser.add_argument(
        "--real-robot",
        action="store_true",
        help="Deploy to a real robot instead of simulation",
    )
    parser.add_argument(
        "--manual-position",
        action="store_true",
        help="Real robot only: set all kp/kd to 0 and run idle monitoring",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable MuJoCo viewer for real-robot playback or idle mode",
    )
    parser.add_argument(
        "--viewer-config",
        type=str,
        default=None,
        help="Optional config for the MuJoCo viewer model",
    )
    parser.add_argument(
        "--module-ids",
        type=int,
        nargs="+",
        default=None,
        help="Module IDs for real robot deployment (e.g. --module-ids 5 21 27)",
    )
    parser.add_argument(
        "--plot-tracking",
        action="store_true",
        help="Enable a real-time joint tracking plot",
    )
    parser.add_argument(
        "--save-tracking",
        type=str,
        default=None,
        metavar="PATH",
        help="Save joint tracking data to file for offline analysis",
    )
    parser.add_argument(
        "--save-state",
        type=str,
        default=None,
        metavar="PATH",
        help="Save full state data for offline analysis",
    )

    args = parser.parse_args()

    active_play_modes = sum(
        value is not None if not isinstance(value, bool) else value
        for value in (args.play, args.policy_pkl, args.openloop_sine)
    )
    if active_play_modes > 1:
        parser.error("--play, --policy-pkl, and --openloop-sine are mutually exclusive")
    if args.manual_position and not args.real_robot:
        parser.error("--manual-position requires --real-robot")

    return args


def _get_plugin_dirs(args) -> list[str]:
    """Resolve plugin directories from CLI arguments."""
    plugin_dirs: list[str] = []
    if not args.no_default_plugins:
        plugin_dirs.extend(_default_plugin_dirs())
    if args.plugin_dir:
        plugin_dirs.extend(args.plugin_dir)
    return _unique_paths(plugin_dirs)


def _resolve_num_episodes(args) -> int:
    """Choose a mode-aware default when --num-episodes is omitted."""
    if args.num_episodes is not None:
        return args.num_episodes

    if (
        (args.play or args.policy_pkl)
        and not (args.real_robot and (args.viewer or args.manual_position))
    ):
        return 5

    return 0


def _load_plugins(plugin_dirs: list[str], *, verbose: bool = True) -> None:
    """Load plugin directories if provided."""
    if not plugin_dirs:
        if verbose:
            print("[Plugins] No plugin directories requested")
        return

    from metamachine.robot_factory import list_factories, load_plugins_from

    for plugin_dir in plugin_dirs:
        path = Path(plugin_dir).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Plugin directory not found: {path}")
        load_results = load_plugins_from(str(path))
        if verbose:
            print(f"[Plugins] Loaded from {path}: {load_results}")

    if verbose:
        print(f"[Plugins] Available factories: {list_factories()}")


def _load_config(config_name_or_path: str):
    """Load a config by registry name or by file path."""
    from metamachine.environments.configs.config_registry import ConfigRegistry

    config_path = Path(config_name_or_path).expanduser()
    if config_path.exists():
        return ConfigRegistry.create_from_file(str(config_path))
    return ConfigRegistry.create_from_name(config_name_or_path)


def _create_play_env(
    config_name_or_path: str,
    *,
    real_robot: bool,
    module_ids: list[int] | None,
):
    """Create an environment directly from a config for playback or deployment."""
    from omegaconf import OmegaConf

    cfg = _load_config(config_name_or_path)

    if real_robot and module_ids:
        cfg_real = OmegaConf.create({"module_ids": module_ids})
        if "real" not in cfg or cfg.real is None:
            cfg.real = cfg_real
        else:
            cfg.real = OmegaConf.merge(cfg.real, cfg_real)
        print(f"Using module IDs: {module_ids}")

    if real_robot:
        from metamachine.environments.env_real import RealMetaMachine

        cfg.environment.mode = "real"
        env = RealMetaMachine(cfg)
    else:
        from metamachine.environments.env_sim import MetaMachine

        cfg.environment.mode = "sim"
        cfg.simulation.render_mode = "viewer"
        cfg.simulation.render = True
        cfg.simulation.video_record_interval = 0
        env = MetaMachine(cfg)

    return env, cfg


def _enable_manual_position_mode(env) -> None:
    """Zero all PD gains so the real robot can be moved by hand."""
    num_actions = len(getattr(env, "expected_module_ids", []))
    if num_actions <= 0 and hasattr(env, "action_space"):
        num_actions = int(env.action_space.shape[0])

    zeros = np.zeros(num_actions, dtype=np.float32)

    env.kps = zeros.copy()
    env.kds = zeros.copy()
    env.kp_default = 0.0
    env.kd_default = 0.0

    if hasattr(env, "cfg") and hasattr(env.cfg, "control"):
        env.cfg.control.kp = 0.0
        env.cfg.control.kd = 0.0

    print("\n[Mode] Manual positioning enabled")
    print("  All kp/kd gains set to 0.0")
    print("  Use 'e' to enable and move joints by hand")


def _run_real_robot_action_loop_with_tracking(
    env,
    action_fn,
    *,
    num_episodes: int,
    title: str,
    verbose: bool,
    enable_realtime_plot: bool,
    save_tracking_path: str | None,
    save_state_path: str | None,
    viewer: bool,
    viewer_config: str | None,
    plot_update_interval: float = 0.05,
    plot_history_length: int = 200,
):
    """Run a real-robot action loop with optional tracking and viewer support."""
    import time

    from metamachine.utils.realtime_plotter import (
        create_joint_logger_from_env,
        create_joint_plotter_from_env,
        create_state_logger_from_env,
    )
    from metamachine.utils.sb3_utils import _get_default_dof_pos, _get_joint_positions

    default_dof_pos = _get_default_dof_pos(env)
    plotter = None
    logger = None
    state_logger = None
    viewer_ctx = None
    viewer_handle = None
    viewer_updater = None

    if enable_realtime_plot:
        plotter = create_joint_plotter_from_env(env)
        plotter.update_interval = plot_update_interval
        plotter.history_length = plot_history_length
        plotter.start()

    if save_tracking_path:
        logger = create_joint_logger_from_env(env)

    if save_state_path:
        state_logger = create_state_logger_from_env(env)

    if viewer:
        import mujoco.viewer

        from metamachine.utils.viewer_utils import ViewerStateUpdater, create_viewer_model

        print("\nCreating MuJoCo viewer model...")
        model, data = create_viewer_model(env.cfg, viewer_config)
        viewer_updater = ViewerStateUpdater(model, data, verbose=False)
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
        viewer_handle = viewer_ctx.__enter__()
        print("\nViewer launched. Press Ctrl+C to stop.")

    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"  Episodes: {'infinite' if num_episodes == 0 else num_episodes}")
    print(f"  Viewer: {viewer}")
    print(f"  Real-time plot: {enable_realtime_plot}")
    if save_tracking_path:
        print(f"  Saving tracking to: {save_tracking_path}")
    if save_state_path:
        print(f"  Saving full state to: {save_state_path}")
    print(f"{'=' * 60}\n")

    episode_rewards = []
    episode_lengths = []
    episode_count = 0
    step_count = 0

    try:
        while num_episodes == 0 or episode_count < num_episodes:
            if viewer_handle is not None and not viewer_handle.is_running():
                break

            obs, info = env.reset()

            if logger:
                logger.reset()
            if state_logger:
                state_logger.reset(new_episode=True)

            episode_reward = 0.0
            episode_length = 0

            while True:
                if viewer_handle is not None and not viewer_handle.is_running():
                    return

                loop_start_time = time.time()
                action = np.asarray(action_fn(obs, step_count), dtype=np.float32)
                action = np.clip(action, env.action_space.low, env.action_space.high)

                obs, reward, terminated, truncated, info = env.step(action)

                joint_positions = _get_joint_positions(env)
                joint_commands = action + default_dof_pos

                if plotter:
                    plotter.update(
                        actions=action,
                        joint_positions=joint_positions,
                        joint_commands=joint_commands,
                    )

                if logger:
                    logger.log(
                        actions=action,
                        joint_positions=joint_positions,
                        joint_commands=joint_commands,
                    )

                if state_logger:
                    state_logger.log(
                        action=action,
                        obs=obs,
                        reward=reward,
                        info=info,
                        env=env,
                        joint_command=joint_commands,
                    )

                if viewer_handle is not None and viewer_updater is not None:
                    if hasattr(env, "observable_data"):
                        viewer_updater.update(env.observable_data)
                    with viewer_handle.lock():
                        viewer_handle.sync()

                    dt = getattr(getattr(env.cfg, "control", None), "dt", None)
                    if dt is not None:
                        elapsed = time.time() - loop_start_time
                        time.sleep(max(0.0, dt - elapsed))

                episode_reward += reward
                episode_length += 1
                step_count += 1

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1

            if verbose:
                print(
                    f"Episode {episode_count}: Reward = {episode_reward:.2f}, "
                    f"Length = {episode_length}"
                )

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        if plotter:
            plotter.stop()

        if logger and save_tracking_path:
            logger.save(save_tracking_path)
            summary_plot_path = save_tracking_path.rsplit(".", 1)[0] + "_summary.png"
            try:
                logger.plot_summary(save_path=summary_plot_path)
            except Exception as exc:
                print(f"[Warning] Could not generate summary plot: {exc}")

        if state_logger and save_state_path:
            state_logger.save(save_state_path)

        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)

        env.close()


def _resolve_openloop_case(args, cfg, num_actions: int):
    """Resolve an open-loop action sequence case from config or CLI overrides."""
    from metamachine.sim2real.local_observation_check import ActionSequenceCase
    from omegaconf import OmegaConf

    def _normalize_cli_vector(value):
        if value is None:
            return None
        if isinstance(value, list) and len(value) == 1:
            return value[0]
        return value

    def _cfg_cases():
        raw = cfg.get("sim_to_real_check", None)
        if raw is None:
            return []
        if hasattr(raw, "get"):
            return raw.get("cases", []) or []
        return []

    case_name = args.openloop_case or "zero_then_sine"
    has_overrides = any(
        value is not None
        for value in (
            args.openloop_hold_sec,
            args.openloop_duration_sec,
            args.openloop_amplitude,
            args.openloop_frequency_hz,
            args.openloop_offset,
            args.openloop_phase,
        )
    )

    if not has_overrides:
        for raw_case in _cfg_cases():
            if isinstance(raw_case, dict):
                case_dict = dict(raw_case)
            else:
                case_dict = OmegaConf.to_container(raw_case, resolve=True)
            if case_dict.get("name") == case_name:
                return ActionSequenceCase.from_config(case_dict, num_modules=num_actions)
        if args.openloop_case:
            raise ValueError(
                f"Open-loop case '{case_name}' not found in config sim_to_real_check.cases"
            )

    sequence = []
    hold_sec = args.openloop_hold_sec
    if hold_sec is None:
        hold_sec = 5.0 if not has_overrides else 0.0
    if hold_sec > 0:
        sequence.append(
            {
                "type": "hold",
                "duration_sec": hold_sec,
                "action": [0.0] * num_actions,
            }
        )

    duration_sec = (
        args.openloop_duration_sec if args.openloop_duration_sec is not None else 10.0
    )
    sequence.append(
        {
            "type": "sine",
            "duration_sec": duration_sec,
            "offset": (
                _normalize_cli_vector(args.openloop_offset)
                if args.openloop_offset is not None
                else 0.0
            ),
            "amplitude": (
                _normalize_cli_vector(args.openloop_amplitude)
                if args.openloop_amplitude is not None
                else 0.1
            ),
            "frequency_hz": (
                args.openloop_frequency_hz
                if args.openloop_frequency_hz is not None
                else 0.5
            ),
            "phase": (
                _normalize_cli_vector(args.openloop_phase)
                if args.openloop_phase is not None
                else 0.0
            ),
        }
    )

    return ActionSequenceCase.from_config(
        {
            "name": case_name,
            "description": "Open-loop sine playback",
            "sequence": sequence,
        },
        num_modules=num_actions,
    )


def _play_openloop_case_with_tracking(
    env,
    case,
    *,
    num_episodes: int,
    real_robot: bool,
    verbose: bool,
    enable_realtime_plot: bool,
    save_tracking_path: str | None,
    save_state_path: str | None,
    viewer: bool = False,
    viewer_config: str | None = None,
    plot_update_interval: float = 0.05,
    plot_history_length: int = 200,
):
    """Run an open-loop action sequence with optional tracking tools."""
    import time

    from metamachine.utils.realtime_plotter import (
        create_joint_logger_from_env,
        create_joint_plotter_from_env,
        create_state_logger_from_env,
    )
    from metamachine.utils.sb3_utils import _get_default_dof_pos, _get_joint_positions

    if real_robot and viewer:
        step_iter = None

        def action_fn(obs, step_count):
            nonlocal step_iter

            if step_iter is None:
                step_iter = iter(case.iter_steps(getattr(env, "dt", 0.05)))

            try:
                step = next(step_iter)
            except StopIteration:
                step_iter = iter(case.iter_steps(getattr(env, "dt", 0.05)))
                step = next(step_iter)

            return np.asarray(step.action, dtype=np.float32)

        _run_real_robot_action_loop_with_tracking(
            env,
            action_fn,
            num_episodes=num_episodes,
            title="Open-Loop Sine Play",
            verbose=verbose,
            enable_realtime_plot=enable_realtime_plot,
            save_tracking_path=save_tracking_path,
            save_state_path=save_state_path,
            viewer=True,
            viewer_config=viewer_config,
            plot_update_interval=plot_update_interval,
            plot_history_length=plot_history_length,
        )
        return

    default_dof_pos = _get_default_dof_pos(env)
    plotter = None
    logger = None
    state_logger = None

    if enable_realtime_plot:
        plotter = create_joint_plotter_from_env(env)
        plotter.update_interval = plot_update_interval
        plotter.history_length = plot_history_length
        plotter.start()

    if save_tracking_path:
        logger = create_joint_logger_from_env(env)

    if save_state_path:
        state_logger = create_state_logger_from_env(env)

    realtime_playback = not real_robot
    dt = getattr(env, "dt", 0.05)
    total_steps = case.total_steps(dt)
    total_duration_sec = case.total_duration_sec(dt)
    clip_warned = False

    print(f"\n{'=' * 60}")
    print("Open-Loop Sine Play")
    print(f"{'=' * 60}")
    print(f"  Case: {case.name}")
    if case.description:
        print(f"  Description: {case.description}")
    print(f"  Episodes: {'infinite' if num_episodes == 0 else num_episodes}")
    print(f"  Steps per episode: {total_steps}")
    print(f"  Duration per episode: {total_duration_sec:.2f}s")
    if num_episodes == 0:
        print("  Sequence looping: CONTINUOUS")
    print(f"  Real robot: {real_robot}")
    print(f"  Real-time plot: {enable_realtime_plot}")
    if save_tracking_path:
        print(f"  Saving tracking to: {save_tracking_path}")
    if save_state_path:
        print(f"  Saving full state to: {save_state_path}")
    if realtime_playback:
        print(f"  Real-time playback: ENABLED (dt={dt:.4f}s, {1 / dt:.1f}Hz)")
    print(f"{'=' * 60}\n")

    episode_rewards = []
    episode_lengths = []
    episode_count = 0

    try:
        while num_episodes == 0 or episode_count < num_episodes:
            obs, info = env.reset()

            if logger:
                logger.reset()
            if state_logger:
                state_logger.reset(new_episode=True)

            episode_reward = 0.0
            episode_length = 0
            terminated_early = False

            keep_running_episode = True
            while keep_running_episode:
                for step in case.iter_steps(dt):
                    if realtime_playback:
                        step_start_time = time.time()

                    raw_action = np.asarray(step.action, dtype=np.float32)
                    action = np.clip(raw_action, env.action_space.low, env.action_space.high)

                    if not clip_warned and not np.allclose(raw_action, action):
                        print(
                            "[Openloop] Warning: action exceeded action_space bounds and was clipped."
                        )
                        clip_warned = True

                    obs, reward, terminated, truncated, info = env.step(action)

                    joint_positions = _get_joint_positions(env)
                    joint_commands = action + default_dof_pos

                    if plotter:
                        plotter.update(
                            actions=action,
                            joint_positions=joint_positions,
                            joint_commands=joint_commands,
                        )

                    if logger:
                        logger.log(
                            actions=action,
                            joint_positions=joint_positions,
                            joint_commands=joint_commands,
                        )

                    if state_logger:
                        state_logger.log(
                            action=action,
                            obs=obs,
                            reward=reward,
                            info=info,
                            env=env,
                            joint_command=joint_commands,
                        )

                    episode_reward += reward
                    episode_length += 1

                    if terminated or truncated:
                        terminated_early = True
                        keep_running_episode = False
                        break

                    if realtime_playback:
                        elapsed = time.time() - step_start_time
                        sleep_time = max(0.0, dt - elapsed)
                        if sleep_time > 0:
                            time.sleep(sleep_time)

                if num_episodes != 0:
                    keep_running_episode = False

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1

            if verbose:
                suffix = " (terminated early)" if terminated_early else ""
                print(
                    f"Episode {episode_count}: Reward = {episode_reward:.2f}, "
                    f"Length = {episode_length}{suffix}"
                )

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        if plotter:
            plotter.stop()

        if logger and save_tracking_path:
            logger.save(save_tracking_path)
            summary_plot_path = save_tracking_path.rsplit(".", 1)[0] + "_summary.png"
            try:
                logger.plot_summary(save_path=summary_plot_path)
            except Exception as exc:
                print(f"[Warning] Could not generate summary plot: {exc}")

        if state_logger and save_state_path:
            state_logger.save(save_state_path)

        env.close()

    stats = {
        "num_episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "min_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "tracking_saved": save_tracking_path,
        "state_saved": save_state_path,
        "openloop_case": case.name,
    }

    if verbose and episode_rewards:
        print(f"\n{'=' * 60}")
        print(f"Summary ({len(episode_rewards)} episodes)")
        print(f"{'=' * 60}")
        print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"  Mean Episode Length: {stats['mean_length']:.1f}")
        if save_tracking_path:
            print(f"  Tracking data saved to: {save_tracking_path}")
        if save_state_path:
            print(f"  State data saved to: {save_state_path}")
        print(f"{'=' * 60}")

    return stats


def _load_policy_metadata(policy_path: str) -> dict:
    """Load raw metadata from an exported policy .pkl file."""
    import pickle

    with open(policy_path, "rb") as f:
        policy_data = pickle.load(f)

    if not isinstance(policy_data, dict):
        raise TypeError(f"Expected dict in policy file, got {type(policy_data)!r}")

    required = {"params", "batch_stats", "metadata"}
    missing = required - set(policy_data)
    if missing:
        raise KeyError(f"Missing keys in policy file: {sorted(missing)}")

    metadata = policy_data.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("Policy metadata must be a dict")
    return metadata


def _import_crossq():
    """Import CrossQ, falling back to known local repos if needed."""
    try:
        from capyrl import CrossQ
        return CrossQ
    except ImportError as first_exc:
        for repo_dir in CAPYRL_CANDIDATE_DIRS:
            if not repo_dir.exists():
                continue
            repo_dir_str = str(repo_dir)
            if repo_dir_str not in sys.path:
                sys.path.insert(0, repo_dir_str)
            try:
                from capyrl import CrossQ
                return CrossQ
            except ImportError:
                continue

        raise ImportError(
            "CrossQ .pkl loading requires the `capyrl` package. "
            "Tried the current environment and these local repos: "
            + ", ".join(str(path) for path in CAPYRL_CANDIDATE_DIRS)
        ) from first_exc


def _validate_pkl_policy_compatibility(env, metadata: dict, policy_path: str) -> None:
    """Fail early if the exported policy shape does not match the environment."""
    env_obs_shape = getattr(getattr(env, "observation_space", None), "shape", None)
    env_act_shape = getattr(getattr(env, "action_space", None), "shape", None)

    env_obs_dim = env_obs_shape[0] if env_obs_shape else None
    env_act_dim = env_act_shape[0] if env_act_shape else None
    policy_obs_dim = metadata.get("obs_dim")
    policy_act_dim = metadata.get("action_dim")

    errors = []
    if env_obs_dim is not None and policy_obs_dim is not None and env_obs_dim != policy_obs_dim:
        errors.append(
            f"observation dim mismatch: env={env_obs_dim}, policy={policy_obs_dim}"
        )
    if env_act_dim is not None and policy_act_dim is not None and env_act_dim != policy_act_dim:
        errors.append(
            f"action dim mismatch: env={env_act_dim}, policy={policy_act_dim}"
        )

    if errors:
        joined = "; ".join(errors)
        raise ValueError(
            f"Policy .pkl is incompatible with the selected config/environment ({joined}). "
            f"Policy: {policy_path}"
        )


def _load_pkl_policy_model(policy_path: str, env):
    """Load an exported CrossQ .pkl policy model."""
    CrossQ = _import_crossq()
    metadata = _load_policy_metadata(policy_path)
    _validate_pkl_policy_compatibility(env, metadata, policy_path)

    model = CrossQ.load_pkl(str(policy_path), env=env, device="cpu")
    return model, metadata


def _predict_action(model, obs, *, deterministic: bool = True) -> np.ndarray:
    """Normalize action prediction across SB3-style and capyrl-style models."""
    try:
        prediction = model.predict(obs, deterministic=deterministic)
    except TypeError as exc:
        if "deterministic" not in str(exc):
            raise
        prediction = model.predict(obs)

    action = prediction[0] if isinstance(prediction, tuple) else prediction
    return np.asarray(action, dtype=np.float32)


def _zero_action(env) -> np.ndarray:
    """Return a zero action matching the current environment action space."""
    return np.zeros(env.action_space.shape[0], dtype=np.float32)


def _play_loaded_model_with_tracking(
    env,
    model=None,
    *,
    title: str,
    num_episodes: int,
    real_robot: bool,
    deterministic: bool,
    verbose: bool,
    enable_realtime_plot: bool,
    save_tracking_path: str | None,
    save_state_path: str | None,
    action_fn=None,
    plot_update_interval: float = 0.05,
    plot_history_length: int = 200,
):
    """Play a loaded model or custom action function with optional tracking."""
    import time

    from metamachine.utils.realtime_plotter import (
        create_joint_logger_from_env,
        create_joint_plotter_from_env,
        create_state_logger_from_env,
    )
    from metamachine.utils.sb3_utils import _get_default_dof_pos, _get_joint_positions

    default_dof_pos = _get_default_dof_pos(env)
    plotter = None
    logger = None
    state_logger = None

    if enable_realtime_plot:
        plotter = create_joint_plotter_from_env(env)
        plotter.update_interval = plot_update_interval
        plotter.history_length = plot_history_length
        plotter.start()

    if save_tracking_path:
        logger = create_joint_logger_from_env(env)

    if save_state_path:
        state_logger = create_state_logger_from_env(env)

    realtime_playback = not real_robot
    dt = getattr(env, "dt", 0.05)

    print(f"\n{'=' * 60}")
    print(title)
    print(f"{'=' * 60}")
    print(f"  Episodes: {'infinite' if num_episodes == 0 else num_episodes}")
    print(f"  Deterministic: {deterministic}")
    print(f"  Real robot: {real_robot}")
    print(f"  Real-time plot: {enable_realtime_plot}")
    if save_tracking_path:
        print(f"  Saving tracking to: {save_tracking_path}")
    if save_state_path:
        print(f"  Saving full state to: {save_state_path}")
    if realtime_playback:
        print(f"  Real-time playback: ENABLED (dt={dt:.4f}s, {1 / dt:.1f}Hz)")
    print(f"{'=' * 60}\n")

    episode_rewards = []
    episode_lengths = []
    episode_count = 0

    try:
        while num_episodes == 0 or episode_count < num_episodes:
            obs, info = env.reset()

            if logger:
                logger.reset()
            if state_logger:
                state_logger.reset(new_episode=True)

            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                if realtime_playback:
                    step_start_time = time.time()

                if action_fn is not None:
                    action = np.asarray(action_fn(obs), dtype=np.float32)
                else:
                    if model is None:
                        raise ValueError("Either model or action_fn must be provided")
                    action = _predict_action(model, obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)

                joint_positions = _get_joint_positions(env)
                joint_commands = action + default_dof_pos

                if plotter:
                    plotter.update(
                        actions=action,
                        joint_positions=joint_positions,
                        joint_commands=joint_commands,
                    )

                if logger:
                    logger.log(
                        actions=action,
                        joint_positions=joint_positions,
                        joint_commands=joint_commands,
                    )

                if state_logger:
                    state_logger.log(
                        action=action,
                        obs=obs,
                        reward=reward,
                        info=info,
                        env=env,
                        joint_command=joint_commands,
                    )

                episode_reward += reward
                episode_length += 1
                done = terminated or truncated

                if realtime_playback:
                    elapsed = time.time() - step_start_time
                    sleep_time = max(0.0, dt - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_count += 1

            if verbose:
                print(
                    f"Episode {episode_count}: Reward = {episode_reward:.2f}, "
                    f"Length = {episode_length}"
                )

    except KeyboardInterrupt:
        print("\n[Interrupted]")

    finally:
        if plotter:
            plotter.stop()

        if logger and save_tracking_path:
            logger.save(save_tracking_path)
            summary_plot_path = save_tracking_path.rsplit(".", 1)[0] + "_summary.png"
            try:
                logger.plot_summary(save_path=summary_plot_path)
            except Exception as exc:
                print(f"[Warning] Could not generate summary plot: {exc}")

        if state_logger and save_state_path:
            state_logger.save(save_state_path)

        env.close()

    stats = {
        "num_episodes": len(episode_rewards),
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "std_reward": float(np.std(episode_rewards)) if episode_rewards else 0.0,
        "min_reward": float(np.min(episode_rewards)) if episode_rewards else 0.0,
        "max_reward": float(np.max(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "tracking_saved": save_tracking_path,
        "state_saved": save_state_path,
    }

    if verbose and episode_rewards:
        print(f"\n{'=' * 60}")
        print(f"Summary ({len(episode_rewards)} episodes)")
        print(f"{'=' * 60}")
        print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Min/Max Reward: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"  Mean Episode Length: {stats['mean_length']:.1f}")
        if save_tracking_path:
            print(f"  Tracking data saved to: {save_tracking_path}")
        if save_state_path:
            print(f"  State data saved to: {save_state_path}")
        print(f"{'=' * 60}")

    return stats


def main():
    args = parse_args()
    plugin_dirs = _get_plugin_dirs(args)
    num_episodes = _resolve_num_episodes(args)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if args.openloop_sine:
        print("=" * 60)
        print("Open-Loop Play Mode")
        print("=" * 60)

        _load_plugins(plugin_dirs, verbose=True)
        env, cfg = _create_play_env(
            args.config,
            real_robot=args.real_robot,
            module_ids=args.module_ids,
        )

        if args.manual_position:
            _enable_manual_position_mode(env)
            _run_real_robot_action_loop_with_tracking(
                env,
                lambda obs, step_count: _zero_action(env),
                num_episodes=num_episodes,
                title="Manual Position Mode",
                verbose=True,
                enable_realtime_plot=args.plot_tracking,
                save_tracking_path=args.save_tracking,
                save_state_path=args.save_state,
                viewer=args.viewer,
                viewer_config=args.viewer_config,
            )
            return

        if args.zero_action:
            if args.real_robot:
                _run_real_robot_action_loop_with_tracking(
                    env,
                    lambda obs, step_count: _zero_action(env),
                    num_episodes=num_episodes,
                    title="Zero Action Mode",
                    verbose=True,
                    enable_realtime_plot=args.plot_tracking,
                    save_tracking_path=args.save_tracking,
                    save_state_path=args.save_state,
                    viewer=args.viewer,
                    viewer_config=args.viewer_config,
                )
            else:
                _play_loaded_model_with_tracking(
                    env,
                    title="Zero Action Play",
                    num_episodes=num_episodes,
                    real_robot=False,
                    deterministic=True,
                    verbose=True,
                    enable_realtime_plot=args.plot_tracking,
                    save_tracking_path=args.save_tracking,
                    save_state_path=args.save_state,
                    action_fn=lambda obs: _zero_action(env),
                )
            return

        try:
            case = _resolve_openloop_case(args, cfg, env.action_space.shape[0])
        except ValueError as exc:
            env.close()
            raise SystemExit(str(exc)) from exc

        _play_openloop_case_with_tracking(
            env,
            case,
            num_episodes=num_episodes,
            real_robot=args.real_robot,
            verbose=True,
            enable_realtime_plot=args.plot_tracking,
            save_tracking_path=args.save_tracking,
            save_state_path=args.save_state,
            viewer=args.viewer,
            viewer_config=args.viewer_config,
        )
        return

    if args.play:
        print("=" * 60)
        print("Play Mode")
        print("=" * 60)

        _load_plugins(plugin_dirs, verbose=True)

        cfg_real = None
        if args.real_robot and args.module_ids:
            cfg_real = {"module_ids": args.module_ids}
            print(f"Using module IDs: {args.module_ids}")

        if args.real_robot and (args.viewer or args.manual_position or args.zero_action):
            from metamachine.utils.sb3_utils import load_from_checkpoint

            env, model, cfg = load_from_checkpoint(
                args.play,
                checkpoint=args.checkpoint,
                render_mode="none",
                real_robot=True,
                cfg_real=cfg_real,
            )

            if args.manual_position:
                _enable_manual_position_mode(env)

            if model is None and not (args.manual_position or args.zero_action):
                env.close()
                raise ValueError("No model checkpoint found to play")

            def action_fn(obs, step_count):
                if args.manual_position or args.zero_action:
                    return _zero_action(env)
                return _predict_action(model, obs, deterministic=True)

            _run_real_robot_action_loop_with_tracking(
                env,
                action_fn,
                num_episodes=num_episodes,
                title=(
                    "Manual Position Mode"
                    if args.manual_position
                    else "Zero Action Mode"
                    if args.zero_action
                    else "Play Mode"
                ),
                verbose=True,
                enable_realtime_plot=args.plot_tracking,
                save_tracking_path=args.save_tracking,
                save_state_path=args.save_state,
                viewer=args.viewer,
                viewer_config=args.viewer_config,
            )
        else:
            if args.zero_action:
                from metamachine.utils.sb3_utils import load_from_checkpoint

                env, model, cfg = load_from_checkpoint(
                    args.play,
                    checkpoint=args.checkpoint,
                    render_mode="viewer" if not args.real_robot else "none",
                    real_robot=args.real_robot,
                    cfg_real=cfg_real,
                )
                _play_loaded_model_with_tracking(
                    env,
                    title="Zero Action Play",
                    num_episodes=num_episodes,
                    real_robot=args.real_robot,
                    deterministic=True,
                    verbose=True,
                    enable_realtime_plot=args.plot_tracking,
                    save_tracking_path=args.save_tracking,
                    save_state_path=args.save_state,
                    action_fn=lambda obs: _zero_action(env),
                )
            else:
                from metamachine.utils.sb3_utils import play_checkpoint_with_tracking

                play_checkpoint_with_tracking(
                    log_dir=args.play,
                    checkpoint=args.checkpoint,
                    num_episodes=num_episodes,
                    render_mode="viewer" if not args.real_robot else "none",
                    real_robot=args.real_robot,
                    cfg_real=cfg_real,
                    deterministic=True,
                    verbose=True,
                    enable_realtime_plot=args.plot_tracking,
                    save_tracking_path=args.save_tracking,
                    save_state_path=args.save_state,
                )

        return

    if args.policy_pkl:
        print("=" * 60)
        print("PKL Policy Play Mode")
        print("=" * 60)

        _load_plugins(plugin_dirs, verbose=True)

        policy_path = Path(args.policy_pkl).expanduser()
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy .pkl not found: {policy_path}")
        if policy_path.suffix != ".pkl":
            raise ValueError(f"Expected a .pkl policy file, got: {policy_path}")

        env, cfg = _create_play_env(
            args.config,
            real_robot=args.real_robot,
            module_ids=args.module_ids,
        )
        model, metadata = _load_pkl_policy_model(str(policy_path), env)

        print(f"[Policy] Loaded .pkl policy from: {policy_path}")
        print(
            f"[Policy] Metadata: obs_dim={metadata.get('obs_dim')}, "
            f"action_dim={metadata.get('action_dim')}, "
            f"net_arch={metadata.get('net_arch')}"
        )

        if args.real_robot and (args.viewer or args.manual_position or args.zero_action):
            if args.manual_position:
                _enable_manual_position_mode(env)

            def action_fn(obs, step_count):
                if args.manual_position or args.zero_action:
                    return _zero_action(env)
                return _predict_action(model, obs, deterministic=True)

            _run_real_robot_action_loop_with_tracking(
                env,
                action_fn,
                num_episodes=num_episodes,
                title=(
                    "Manual Position Mode (.pkl)"
                    if args.manual_position
                    else "Zero Action Mode (.pkl)"
                    if args.zero_action
                    else "PKL Policy Play Mode"
                ),
                verbose=True,
                enable_realtime_plot=args.plot_tracking,
                save_tracking_path=args.save_tracking,
                save_state_path=args.save_state,
                viewer=args.viewer,
                viewer_config=args.viewer_config,
            )
        else:
            _play_loaded_model_with_tracking(
                env,
                num_episodes=num_episodes,
                real_robot=args.real_robot,
                deterministic=True,
                verbose=True,
                enable_realtime_plot=args.plot_tracking,
                save_tracking_path=args.save_tracking,
                save_state_path=args.save_state,
                title="Zero Action Play (.pkl)" if args.zero_action else "PKL Policy Play",
                model=model,
                action_fn=(lambda obs: _zero_action(env)) if args.zero_action else None,
            )

        return

    if args.real_robot and (args.viewer or args.manual_position or args.zero_action):
        print("=" * 60)
        print("Real Robot Zero-Action / Idle Mode")
        print("=" * 60)

        _load_plugins(plugin_dirs, verbose=True)
        env, cfg = _create_play_env(
            args.config,
            real_robot=True,
            module_ids=args.module_ids,
        )

        if args.manual_position:
            _enable_manual_position_mode(env)

        _run_real_robot_action_loop_with_tracking(
            env,
            lambda obs, step_count: _zero_action(env),
            num_episodes=num_episodes,
            title=(
                "Manual Position Mode"
                if args.manual_position
                else "Zero Action Mode"
                if args.zero_action
                else "Idle Viewer Mode"
            ),
            verbose=True,
            enable_realtime_plot=args.plot_tracking,
            save_tracking_path=args.save_tracking,
            save_state_path=args.save_state,
            viewer=args.viewer,
            viewer_config=args.viewer_config,
        )
        return

    if args.continue_from:
        print("=" * 60)
        print("Continue Training / Fine-tuning Mode")
        print("=" * 60)

        _load_plugins(plugin_dirs, verbose=True)

        from metamachine.utils.sb3_utils import continue_training

        new_config = None
        config_path = Path(args.config).expanduser()
        if config_path.exists():
            new_config = str(config_path)

        trainer = continue_training(
            log_dir=args.continue_from,
            new_config=new_config,
            checkpoint=args.checkpoint,
            total_timesteps=args.timesteps,
            exp_name=args.exp_name if args.exp_name != DEFAULT_EXP_NAME else None,
            show_config_diff=True,
            confirm_diff=True,
            checkpoint_freq=DEFAULT_CHECKPOINT_FREQ,
        )

        trainer.learn(total_timesteps=args.timesteps)
        trainer.save()

        print(f"\nFine-tuning complete! Logs saved to: {trainer.log_dir}")
        print(
            "To visualize: "
            f"python examples/train_rl_policy_sb3.py --play {trainer.log_dir}"
        )
        return

    print("=" * 60)
    print("Training Mode - SB3Trainer")
    print("=" * 60)

    _load_plugins(plugin_dirs, verbose=True)

    cfg = _load_config(args.config)
    if args.no_render:
        cfg.simulation.render_mode = "none"
        cfg.simulation.render = False

    print(f"[Config] Robot type: {cfg.morphology.get('robot_type', 'unknown')}")
    if hasattr(cfg, "control"):
        print(f"[Config] Num actions: {cfg.control.get('num_actions', 'unknown')}")

    from metamachine.environments.env_sim import MetaMachine
    from metamachine.utils.sb3_utils import SB3Trainer

    env = MetaMachine(cfg)

    print(f"[Environment] Action space: {env.action_space}")
    print(f"[Environment] Observation space: {env.observation_space}")
    print(f"[Environment] Log directory: {env._log_dir}")

    trainer = SB3Trainer(
        env,
        algorithm=args.algorithm,
        exp_name=args.exp_name,
        seed=args.seed,
        checkpoint_freq=DEFAULT_CHECKPOINT_FREQ,
    )
    trainer.learn(total_timesteps=args.timesteps)
    trainer.save()

    print(f"\nTraining complete! Logs saved to: {trainer.log_dir}")
    print(f"To visualize: python examples/train_rl_policy_sb3.py --play {trainer.log_dir}")


if __name__ == "__main__":
    main()
