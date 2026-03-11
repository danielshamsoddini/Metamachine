"""Utilities for MetaMachine."""

from .checkpoint_manager import (
    CheckpointManager,
    get_checkpoint,
    download_from_url,
    register_model,
    list_models,
    print_models,
    get_default_manager,
)

from .rollout_recorder import (
    RolloutRecorder,
    EpisodeData,
    StateSnapshot,
)

from .mujoco_utils import (
    find_parent_torso,
    get_all_weld_clusters,
    get_largest_weld_cluster_average_pos,
    get_weld_cluster_center_of_mass,
)

__all__ = [
    # Checkpoint manager
    "CheckpointManager",
    "get_checkpoint",
    "download_from_url",
    "register_model",
    "list_models",
    "print_models",
    "get_default_manager",
    # Rollout recorder
    "RolloutRecorder",
    "EpisodeData",
    "StateSnapshot",
    # MuJoCo utilities
    "find_parent_torso",
    "get_all_weld_clusters",
    "get_largest_weld_cluster_average_pos",
    "get_weld_cluster_center_of_mass",
    # SB3 utilities (optional, requires stable-baselines3)
    "SB3Trainer",
    "setup_sb3_training",
    "RewardComponentCallback",
    "ProgressBarCallback",
    "load_from_checkpoint",
    "play_checkpoint",
    "play_checkpoint_with_tracking",
    "continue_training",
    "compare_configs",
    # Policy runner utilities
    "PolicyRunner",
    "load_policies",
    "load_policy_standalone",
    "find_checkpoint_path",
    # Legacy aliases
    "MultiModelRunner",
    "load_multiple_models",
    "load_model_standalone",
    # Training callbacks (optional, requires stable-baselines3)
    "SB3TrainingProgressCallback",
    # Real-time plotting utilities
    "RealtimeJointPlotter",
    "JointTrackingLogger",
    "StateLogger",
    "create_joint_plotter_from_env",
    "create_joint_logger_from_env",
    "create_state_logger_from_env",
    # Rendering utilities for MuJoCo scene markers
    "add_marker_to_scene",
    "add_ground_disc_marker",
    "add_sphere_marker",
    "add_arrow_marker",
    "render_line",
    # Bearing estimation utilities
    "MultiPolicyBearingCollector",
    "BearingEstimatorV3",
    "BearingEstimatorRunner",
    "BearingAugmentedConfig",
    "BearingAugmentedPolicySwitchEnv",
    # Chase visualization utilities
    "add_chase_scene_markers",
    "add_bearing_metrics_overlay",
    "compute_bearing_from_pose",
    "compute_forward_direction",
    "draw_chase_frame_matplotlib",
    # MJX utilities (optional, requires jax and mujoco-mjx)
    "create_batched_env_fns",
    "create_single_env_fns",
    "warmup_jit",
    "run_batched_rollout",
    "run_single_rollout",
    "render_mjx_trajectory",
    "save_video",
    "get_mjx_data_as_mujoco",
    "print_mjx_info",
    "zero_policy",
    "random_policy",
]

# SB3 utilities (optional import - only available if stable-baselines3 is installed)
try:
    from .sb3_utils import (
        SB3Trainer,
        setup_sb3_training,
        RewardComponentCallback,
        ProgressBarCallback,
        load_from_checkpoint,
        play_checkpoint,
        play_checkpoint_with_tracking,
        continue_training,
        compare_configs,
    )
except ImportError:
    # SB3 not installed, provide placeholder
    SB3Trainer = None
    setup_sb3_training = None
    RewardComponentCallback = None
    ProgressBarCallback = None
    load_from_checkpoint = None
    play_checkpoint = None
    play_checkpoint_with_tracking = None
    continue_training = None
    compare_configs = None

# Policy runner utilities
try:
    from .policy_runner import (
        PolicyRunner,
        load_policies,
        load_policy_standalone,
        find_checkpoint_path,
        # Legacy aliases
        MultiModelRunner,
        load_multiple_models,
        load_model_standalone,
    )
except ImportError:
    PolicyRunner = None
    load_policies = None
    load_policy_standalone = None
    find_checkpoint_path = None
    MultiModelRunner = None
    load_multiple_models = None
    load_model_standalone = None

# Training callbacks (optional import - requires stable-baselines3)
try:
    from .training_callbacks import SB3TrainingProgressCallback
except ImportError:
    SB3TrainingProgressCallback = None

# Real-time plotting utilities
from .realtime_plotter import (
    RealtimeJointPlotter,
    JointTrackingLogger,
    StateLogger,
    create_joint_plotter_from_env,
    create_joint_logger_from_env,
    create_state_logger_from_env,
)

# Rendering utilities for MuJoCo scene markers
from .rendering import (
    add_marker_to_scene,
    add_ground_disc_marker,
    add_sphere_marker,
    add_arrow_marker,
    render_line,
)

# Bearing estimation utilities (optional)
try:
    from .bearing_estimation import (
        MultiPolicyBearingCollector,
        BearingEstimatorV3,
        BearingEstimatorRunner,
        BearingAugmentedConfig,
        BearingAugmentedPolicySwitchEnv,
    )
except ImportError:
    MultiPolicyBearingCollector = None
    BearingEstimatorV3 = None
    BearingEstimatorRunner = None
    BearingAugmentedConfig = None
    BearingAugmentedPolicySwitchEnv = None

# Chase visualization utilities
try:
    from .chase_visualization import (
        add_chase_scene_markers,
        add_bearing_metrics_overlay,
        compute_bearing_from_pose,
        compute_forward_direction,
        draw_chase_frame_matplotlib,
    )
except ImportError:
    add_chase_scene_markers = None
    add_bearing_metrics_overlay = None
    compute_bearing_from_pose = None
    compute_forward_direction = None
    draw_chase_frame_matplotlib = None

# MJX utilities (optional - requires jax and mujoco-mjx)
try:
    from .mjx_utils import (
        create_batched_env_fns,
        create_single_env_fns,
        warmup_jit,
        run_batched_rollout,
        run_single_rollout,
        render_mjx_trajectory,
        save_video,
        get_mjx_data_as_mujoco,
        print_mjx_info,
        zero_policy,
        random_policy,
    )
except ImportError:
    create_batched_env_fns = None
    create_single_env_fns = None
    warmup_jit = None
    run_batched_rollout = None
    run_single_rollout = None
    render_mjx_trajectory = None
    save_video = None
    get_mjx_data_as_mujoco = None
    print_mjx_info = None
    zero_policy = None
    random_policy = None
