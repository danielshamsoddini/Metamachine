#!/usr/bin/env python3
"""Convert a CrossQ .pkl policy into ESP32 deployment binaries.

This exporter targets the current `capy_esp32` firmware for the single-module
standalone policy path:
    obs_history -> fused MLP -> tanh -> rescale to action range -> + joint offset

It writes:
    capy_esp32/data/w1.bin ... b3.bin
    capy_esp32/include/deploy_config.h
    capy_esp32/include/local_obs_config.h

The generated headers own deploy-time settings such as:
    - policy loop frequency
    - hidden-layer sizes
    - action range
    - observation component transforms
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOCAL_FIELD_ENUM = {
    "projected_gravity": "ProjectedGravity",
    "ang_vel_body": "Gyro",
    "dof_pos": "DofPos",
    "dof_vel": "DofVel",
    "last_action": "LastAction",
}

LOCAL_FIELD_DIMS = {
    "projected_gravity": 3,
    "ang_vel_body": 3,
    "dof_pos": 1,
    "dof_vel": 1,
    "last_action": 1,
}

LOCAL_TRANSFORM_ENUM = {
    None: "None",
    "cos": "Cos",
    "sin": "Sin",
}


def _normalize_component_spec(comp_spec: Any) -> dict[str, Any]:
    if isinstance(comp_spec, str):
        return {"name": comp_spec}
    if isinstance(comp_spec, (list, tuple)):
        if len(comp_spec) >= 2:
            return {"name": comp_spec[0], "transform": comp_spec[1]}
        return {"name": comp_spec[0]}
    if hasattr(comp_spec, "items"):
        return dict(comp_spec)
    raise TypeError(f"Unsupported component spec: {comp_spec!r}")


def _load_config(config_ref: str):
    from metamachine.environments.configs.config_registry import ConfigRegistry

    config_path = Path(config_ref)
    if config_path.exists():
        return ConfigRegistry.create_from_file(str(config_path)), str(config_path)
    return ConfigRegistry.create_from_name(config_ref), config_ref


def _load_policy(policy_path: Path) -> dict[str, Any]:
    with policy_path.open("rb") as f:
        policy_data = pickle.load(f)
    if not isinstance(policy_data, dict):
        raise TypeError(f"Expected dict in {policy_path}, got {type(policy_data)!r}")
    required = {"params", "batch_stats", "metadata"}
    missing = required - set(policy_data)
    if missing:
        raise KeyError(f"Missing keys in policy file: {sorted(missing)}")
    return policy_data


def _array(tree: dict[str, Any], module: str, key: str) -> np.ndarray:
    return np.asarray(tree[module][key], dtype=np.float32)


def _fuse_bn_into_dense(
    dense_kernel: np.ndarray,
    dense_bias: np.ndarray,
    bn_scale: np.ndarray,
    bn_bias: np.ndarray,
    bn_mean: np.ndarray,
    bn_var: np.ndarray,
    *,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    scale = bn_scale / np.sqrt(bn_var + epsilon)
    fused_kernel = dense_kernel * scale[:, np.newaxis]
    fused_bias = dense_kernel.T @ (bn_bias - scale * bn_mean) + dense_bias
    return fused_kernel.astype(np.float32), fused_bias.astype(np.float32)


def _fuse_actor(policy_data: dict[str, Any]) -> tuple[np.ndarray, ...]:
    params = policy_data["params"]
    batch_stats = policy_data["batch_stats"]
    metadata = policy_data["metadata"]

    net_arch = list(metadata["net_arch"])
    if len(net_arch) != 2:
        raise ValueError(f"Expected 2 hidden layers, got {net_arch}")

    fused_layers: list[tuple[np.ndarray, np.ndarray]] = []
    for dense_idx in range(len(net_arch)):
        fused_layers.append(
            _fuse_bn_into_dense(
                _array(params, f"Dense_{dense_idx}", "kernel"),
                _array(params, f"Dense_{dense_idx}", "bias"),
                _array(params, f"BatchRenorm_{dense_idx}", "scale"),
                _array(params, f"BatchRenorm_{dense_idx}", "bias"),
                _array(batch_stats, f"BatchRenorm_{dense_idx}", "mean"),
                _array(batch_stats, f"BatchRenorm_{dense_idx}", "var"),
            )
        )

    mean_idx = len(net_arch)
    mean_kernel, mean_bias = _fuse_bn_into_dense(
        _array(params, f"Dense_{mean_idx}", "kernel"),
        _array(params, f"Dense_{mean_idx}", "bias"),
        _array(params, f"BatchRenorm_{mean_idx}", "scale"),
        _array(params, f"BatchRenorm_{mean_idx}", "bias"),
        _array(batch_stats, f"BatchRenorm_{mean_idx}", "mean"),
        _array(batch_stats, f"BatchRenorm_{mean_idx}", "var"),
    )
    fused_layers.append((mean_kernel, mean_bias))
    return (
        fused_layers[0][0],
        fused_layers[0][1],
        fused_layers[1][0],
        fused_layers[1][1],
        fused_layers[2][0],
        fused_layers[2][1],
    )


def _write_bin(path: Path, arr: np.ndarray) -> None:
    data = np.ascontiguousarray(arr, dtype=np.float32)
    data.tofile(str(path))
    print(f"  {path.name}: {data.shape} -> {data.nbytes} bytes")


def _fnv1a31_update(hash_value: int, blob: bytes) -> int:
    for byte in blob:
        hash_value ^= byte
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value


def _compute_policy_hash(*arrays: np.ndarray) -> int:
    hash_value = 2166136261
    for arr in arrays:
        hash_value = _fnv1a31_update(
            hash_value,
            np.ascontiguousarray(arr, dtype=np.float32).tobytes(),
        )
    return hash_value & 0x7FFFFFFF


def _obs_spec_from_cfg(cfg) -> dict[str, Any]:
    components = []
    for comp_spec in cfg.observation.components:
        spec = _normalize_component_spec(comp_spec)
        name = spec["name"]
        transform = spec.get("transform")
        if name not in LOCAL_FIELD_ENUM:
            raise ValueError(
                f"Unsupported observation component '{name}'. "
                f"Supported: {sorted(LOCAL_FIELD_ENUM)}"
            )
        if transform not in LOCAL_TRANSFORM_ENUM:
            raise ValueError(
                f"Unsupported transform '{transform}' for '{name}'. "
                f"Supported: {sorted(k for k in LOCAL_TRANSFORM_ENUM if k is not None)}"
            )
        components.append(
            {
                "name": name,
                "field_enum": LOCAL_FIELD_ENUM[name],
                "transform_enum": LOCAL_TRANSFORM_ENUM[transform],
                "dim": LOCAL_FIELD_DIMS[name],
            }
        )
    frame_history_steps = int(cfg.observation.include_history_steps)
    frame_dim = sum(item["dim"] for item in components)
    return {
        "components": components,
        "frame_history_steps": frame_history_steps,
        "frame_dim": frame_dim,
        "obs_dim": frame_dim * frame_history_steps,
    }


def _generate_local_obs_header(spec: dict[str, Any], header_path: Path, source_label: str) -> None:
    components = spec["components"]
    component_lines = ",\n    ".join(
        f"LocalObsField::{item['field_enum']}" for item in components
    )
    transform_lines = ",\n    ".join(
        f"LocalObsTransform::{item['transform_enum']}" for item in components
    )
    header = f"""\
#ifndef LOCAL_OBS_CONFIG_H
#define LOCAL_OBS_CONFIG_H

#include <cstddef>
#include <cstdint>

// Auto-generated by examples/convert_policy_pkl_to_bin.py
// Source config: {source_label}

enum class LocalObsField : uint8_t {{
    ProjectedGravity,
    Gyro,
    DofPos,
    DofVel,
    LastAction,
}};

enum class LocalObsTransform : uint8_t {{
    None,
    Cos,
    Sin,
}};

constexpr size_t LOCAL_LATENT_DIM = 0;
constexpr size_t LOCAL_FRAME_HISTORY_STEPS = {spec["frame_history_steps"]};
constexpr size_t LOCAL_LATENT_HISTORY_STEPS = 0;
constexpr size_t LOCAL_TRANSPORT_COMMAND_CONTEXT_DIM = 8;
constexpr size_t LOCAL_TRANSPORT_DEBUG_OBS_DIM = 40;
constexpr size_t LOCAL_FRAME_COMPONENT_COUNT = {len(components)};
constexpr LocalObsField LOCAL_FRAME_COMPONENTS[LOCAL_FRAME_COMPONENT_COUNT] = {{
    {component_lines}
}};
constexpr LocalObsTransform LOCAL_FRAME_TRANSFORMS[LOCAL_FRAME_COMPONENT_COUNT] = {{
    {transform_lines}
}};
constexpr size_t LOCAL_FRAME_DIM = {spec["frame_dim"]};
constexpr size_t LOCAL_OBS_DIM = {spec["obs_dim"]};
constexpr size_t LOCAL_DEBUG_COMMAND_CONTEXT_DIM = 0;
constexpr size_t LOCAL_DEBUG_OBS_DIM =
    (LOCAL_OBS_DIM < LOCAL_TRANSPORT_DEBUG_OBS_DIM)
        ? LOCAL_OBS_DIM
        : LOCAL_TRANSPORT_DEBUG_OBS_DIM;

#endif // LOCAL_OBS_CONFIG_H
"""
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header)
    print(f"  Generated: {header_path}")


def _generate_deploy_header(
    *,
    cfg,
    config_label: str,
    metadata: dict[str, Any],
    policy_hash: int,
    h1_dim: int,
    h2_dim: int,
    reference_mean: float,
    reference_action: float,
    header_path: Path,
    clip_mean: float,
    use_xbox_controller: bool,
) -> None:
    dt = float(cfg.control.dt)
    policy_loop_hz = max(1, int(round(1.0 / dt)))
    default_dof_pos = [float(x) for x in cfg.control.default_dof_pos]
    kp = float(cfg.control.kp)
    kd = float(cfg.control.kd)
    num_modules = int(cfg.control.num_actions)
    action_low = float(np.asarray(metadata["low"], dtype=np.float32).reshape(-1)[0])
    action_high = float(np.asarray(metadata["high"], dtype=np.float32).reshape(-1)[0])
    action_scale = float(cfg.control.action_scale)
    symmetric_limit = float(cfg.control.symmetric_limit)
    dof_pos_str = ", ".join(f"{value:.6f}f" for value in default_dof_pos)
    dof_pos_comment_lines = "\n".join(
        f"// Module {idx}: {value:.4f} rad"
        for idx, value in enumerate(default_dof_pos)
    )

    header = f"""\
#ifndef DEPLOY_CONFIG_H
#define DEPLOY_CONFIG_H

// Auto-generated by examples/convert_policy_pkl_to_bin.py
// Source config: {config_label}
//
// Local policy NN architecture:
//   input -> {h1_dim} -> {h2_dim} -> 1
//
// Inference pipeline on ESP32:
//   1. fused MLP outputs mean
//   2. mean is clipped to [-clip_mean, clip_mean]
//   3. tanh(mean) is unscaled to [action_low, action_high]
//   4. motor_target = nn_action + default_dof_pos[module_idx]

constexpr int DEPLOY_NUM_MODULES = {num_modules};
constexpr int DEPLOY_POLICY_LOOP_HZ = {policy_loop_hz};
constexpr int DEPLOY_LOCAL_H1_DIM = {h1_dim};
constexpr int DEPLOY_LOCAL_H2_DIM = {h2_dim};
constexpr bool DEPLOY_USE_XBOX_CONTROLLER = {"true" if use_xbox_controller else "false"};

// Per-module default joint position offset (radians)
{dof_pos_comment_lines}
constexpr float DEPLOY_DEFAULT_DOF_POS[{num_modules}] = {{{dof_pos_str}}};

constexpr float DEPLOY_KP = {kp:.6f}f;
constexpr float DEPLOY_KD = {kd:.6f}f;
constexpr float DEPLOY_ACTION_LOW = {action_low:.6f}f;
constexpr float DEPLOY_ACTION_HIGH = {action_high:.6f}f;
constexpr float DEPLOY_CLIP_MEAN = {clip_mean:.6f}f;
constexpr float DEPLOY_ACTION_SCALE = {action_scale:.6f}f;
constexpr float DEPLOY_SYMMETRIC_LIMIT = {symmetric_limit:.6f}f;
constexpr int DEPLOY_POLICY_HASH = {policy_hash};
constexpr float DEPLOY_SANITY_EXPECTED_MEAN = {reference_mean:.8f}f;
constexpr float DEPLOY_SANITY_EXPECTED_ACTION = {reference_action:.8f}f;

#endif // DEPLOY_CONFIG_H
"""
    header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header)
    print(f"  Generated: {header_path}")


def _esp32_forward(
    obs: np.ndarray,
    w0_t: np.ndarray,
    b0: np.ndarray,
    w1_t: np.ndarray,
    b1: np.ndarray,
    w2_t: np.ndarray,
    b2: np.ndarray,
    *,
    action_low: float,
    action_high: float,
    clip_mean: float,
) -> tuple[float, float]:
    h1 = np.maximum(w0_t @ obs + b0, 0.0)
    h2 = np.maximum(w1_t @ h1 + b1, 0.0)
    raw_mean = float((w2_t @ h2 + b2)[0])
    mean_clipped = float(np.clip(raw_mean, -clip_mean, clip_mean))
    action = action_low + (np.tanh(mean_clipped) + 1.0) * 0.5 * (action_high - action_low)
    return raw_mean, float(action)


def _run_sanity_check(
    policy_path: Path,
    *,
    w0_t: np.ndarray,
    b0: np.ndarray,
    w1_t: np.ndarray,
    b1: np.ndarray,
    w2_t: np.ndarray,
    b2: np.ndarray,
    obs_dim: int,
    action_low: float,
    action_high: float,
    clip_mean: float,
) -> tuple[float, float]:
    from capyrl import CrossQ

    model = CrossQ.load_pkl(str(policy_path), env=None, device="cpu")

    rng = np.random.RandomState(42)
    max_err = 0.0
    print("\n=== Sanity Check ===")
    for test_idx in range(10):
        obs = rng.randn(obs_dim).astype(np.float32)
        action_ref = float(model.predict(obs.reshape(1, -1))[0][0])
        mean_esp, action_esp = _esp32_forward(
            obs,
            w0_t,
            b0,
            w1_t,
            b1,
            w2_t,
            b2,
            action_low=action_low,
            action_high=action_high,
            clip_mean=clip_mean,
        )
        err = abs(action_ref - action_esp)
        max_err = max(max_err, err)
        if test_idx < 3 or err > 1e-5:
            print(
                f"  Test {test_idx}: action_ref={action_ref:.8f} "
                f"action_esp={action_esp:.8f} err={err:.2e} mean={mean_esp:.8f}"
            )
    print(f"  Max action error: {max_err:.2e}")
    if max_err > 1e-5:
        raise RuntimeError(f"Sanity check failed: max action error {max_err:.3e}")

    ref_obs = np.zeros(obs_dim, dtype=np.float32)
    ref_obs[0] = 0.5
    reference_mean, reference_action = _esp32_forward(
        ref_obs,
        w0_t,
        b0,
        w1_t,
        b1,
        w2_t,
        b2,
        action_low=action_low,
        action_high=action_high,
        clip_mean=clip_mean,
    )
    print("\n=== Reference Test Vector ===")
    print("  Input: obs[0]=0.5, rest zeros")
    print(f"  Raw mean output: {reference_mean:.8f}")
    print(f"  Final action:    {reference_action:.8f}")
    return reference_mean, reference_action


def _write_manifest(
    outdir: Path,
    *,
    policy_path: Path,
    config_label: str,
    policy_hash: int,
    obs_dim: int,
    action_low: float,
    action_high: float,
    control_source: str,
    reference_mean: float,
    reference_action: float,
) -> None:
    manifest = {
        "source_policy": str(policy_path),
        "source_config": config_label,
        "hash_algorithm": "fnv1a_32_mask31",
        "policy_hash": int(policy_hash),
        "policy_hash_hex": f"0x{policy_hash:08x}",
        "obs_dim": int(obs_dim),
        "action_low": float(action_low),
        "action_high": float(action_high),
        "control_source": str(control_source),
        "sanity_expected_mean": float(reference_mean),
        "sanity_expected_action": float(reference_action),
    }
    manifest_path = outdir / "policy_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  Generated: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path, help="Path to CrossQ .pkl policy")
    parser.add_argument(
        "--config",
        type=str,
        default="real_one_module",
        help="Config name or YAML path used to build the observation and control headers",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=PROJECT_ROOT / "capy_esp32" / "data",
        help="Directory for w1.bin ... b3.bin",
    )
    parser.add_argument(
        "--header-dir",
        type=Path,
        default=PROJECT_ROOT / "capy_esp32" / "include",
        help="Directory for deploy_config.h and local_obs_config.h",
    )
    parser.add_argument(
        "--clip-mean",
        type=float,
        default=2.0,
        help="Clamp applied to the actor mean before tanh",
    )
    parser.add_argument(
        "--control-source",
        choices=["pc", "xbox"],
        default="pc",
        help="Select whether the ESP32 expects PC keepalive packets or an onboard Xbox controller",
    )
    args = parser.parse_args()

    policy_path = args.policy.resolve()
    if not policy_path.exists():
        raise FileNotFoundError(policy_path)

    cfg, config_label = _load_config(args.config)
    obs_spec = _obs_spec_from_cfg(cfg)
    policy_data = _load_policy(policy_path)
    metadata = policy_data["metadata"]

    obs_dim = int(metadata["obs_dim"])
    action_dim = int(metadata["action_dim"])
    net_arch = list(metadata["net_arch"])
    if action_dim != 1:
        raise ValueError(f"Expected action_dim=1, got {action_dim}")
    if obs_dim != obs_spec["obs_dim"]:
        raise ValueError(
            f"Policy obs_dim={obs_dim} does not match config-derived obs_dim={obs_spec['obs_dim']}"
        )

    print("Policy config:")
    print(f"  policy:          {policy_path}")
    print(f"  config:          {config_label}")
    print(f"  obs_dim:         {obs_dim}")
    print(f"  action_dim:      {action_dim}")
    print(f"  net_arch:        {net_arch}")
    print(f"  use_batch_norm:  {metadata.get('use_batch_norm', False)}")
    print(f"  control_source:  {args.control_source}")

    dense0, bias0, dense1, bias1, mean_kernel, mean_bias = _fuse_actor(policy_data)
    w0_t = dense0.T.copy()
    w1_t = dense1.T.copy()
    w2_t = mean_kernel.T.copy()

    args.outdir.mkdir(parents=True, exist_ok=True)
    print("\nExporting weights:")
    _write_bin(args.outdir / "w1.bin", w0_t)
    _write_bin(args.outdir / "b1.bin", bias0)
    _write_bin(args.outdir / "w2.bin", w1_t)
    _write_bin(args.outdir / "b2.bin", bias1)
    _write_bin(args.outdir / "w3.bin", w2_t)
    _write_bin(args.outdir / "b3.bin", mean_bias)

    total_bytes = sum(
        (args.outdir / name).stat().st_size
        for name in ("w1.bin", "b1.bin", "w2.bin", "b2.bin", "w3.bin", "b3.bin")
    )
    print(f"\nTotal binary size: {total_bytes} bytes ({total_bytes / 1024:.1f} KB)")

    policy_hash = _compute_policy_hash(w0_t, bias0, w1_t, bias1, w2_t, mean_bias)
    print(f"Policy hash (FNV-1a, 31-bit): {policy_hash} (0x{policy_hash:08x})")

    action_low = float(np.asarray(metadata["low"], dtype=np.float32).reshape(-1)[0])
    action_high = float(np.asarray(metadata["high"], dtype=np.float32).reshape(-1)[0])
    reference_mean, reference_action = _run_sanity_check(
        policy_path,
        w0_t=w0_t,
        b0=bias0,
        w1_t=w1_t,
        b1=bias1,
        w2_t=w2_t,
        b2=mean_bias,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        clip_mean=args.clip_mean,
    )

    _generate_deploy_header(
        cfg=cfg,
        config_label=config_label,
        metadata=metadata,
        policy_hash=policy_hash,
        h1_dim=int(net_arch[0]),
        h2_dim=int(net_arch[1]),
        reference_mean=reference_mean,
        reference_action=reference_action,
        header_path=args.header_dir / "deploy_config.h",
        clip_mean=args.clip_mean,
        use_xbox_controller=(args.control_source == "xbox"),
    )
    _generate_local_obs_header(
        obs_spec,
        args.header_dir / "local_obs_config.h",
        config_label,
    )
    _write_manifest(
        args.outdir,
        policy_path=policy_path,
        config_label=config_label,
        policy_hash=policy_hash,
        obs_dim=obs_dim,
        action_low=action_low,
        action_high=action_high,
        control_source=args.control_source,
        reference_mean=reference_mean,
        reference_action=reference_action,
    )


if __name__ == "__main__":
    main()
