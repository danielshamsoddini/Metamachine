"""Sanity benchmark for synchronization-free native Newton execution.

Run from the repository root with PYTHONPATH=Metamachine. This measures physics
plus optional state observation, not policy inference, learning, or MJX speed.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np
import warp as wp

from metamachine.environments import make_env
from metamachine.environments.backends.newton import NewtonBackend
from metamachine.environments.configs.config_registry import ConfigRegistry


def measure(fn, device, iterations, warmup, worlds, steps_per_call=1):
    for _ in range(warmup):
        fn()
    wp.synchronize_device(device)
    start = perf_counter()
    for _ in range(iterations):
        fn()
    wp.synchronize_device(device)
    elapsed = perf_counter() - start
    batches = iterations * steps_per_call
    return {
        "worlds": worlds,
        "iterations": iterations,
        "warmup": warmup,
        "elapsed_seconds": elapsed,
        "milliseconds_per_control_step": 1000 * elapsed / batches,
        "environment_steps_per_second": batches * worlds / elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worlds", type=int, nargs="+", default=[1, 64, 1024])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 1 or min(args.worlds) <= 0:
        parser.error("iterations/worlds must be positive and warmup >= 1")
    device = wp.get_device(args.device)
    results = {
        "device": str(device),
        "gpu_name": device.name,
        "newton": importlib.metadata.version("newton"),
        "warp": wp.__version__,
        "mujoco": mujoco.__version__,
        "physics_dt": 0.001,
        "physics_steps_per_control": 10,
        "description": "Identical zero motor effort and floor geometry; warm compilation excluded. "
        "Native includes device observations/rewards; compat includes host mirror only. "
        "Batch throughput is not a per-robot speedup comparison.",
        "measurements": [],
    }
    for worlds in args.worlds:
        cfg = ConfigRegistry.create_from_name("rigid_body_probe")
        cfg.simulation.newton_mode = "native"
        cfg.simulation.newton_device = str(device)
        cfg.environment.num_envs = worlds
        native = make_env(cfg)
        ctrl = wp.zeros(worlds * native.action_size, dtype=wp.float32, device=device)

        def native_step(native=native, ctrl=ctrl):
            native.physics.step(ctrl, native.frame_skip)
            native._observe(0, native._no_worlds)

        if worlds == 1:
            compat = NewtonBackend(native.physics.model, mode="compat", device=device)
            host_ctrl = np.zeros(native.action_size, dtype=np.float32)

            def compat_step(
                compat=compat, host_ctrl=host_ctrl, frame_skip=native.frame_skip
            ):
                compat.step(host_ctrl, frame_skip)

            row = measure(compat_step, device, args.iterations, args.warmup, 1)
            row["contacts_after_timing"] = int(
                compat.contacts.rigid_contact_count.numpy()[0]
            )
            results["measurements"].append(dict(mode="compat", **row))
        row = measure(native_step, device, args.iterations, args.warmup, worlds)
        row["contacts_after_timing"] = int(
            native.physics.contacts.rigid_contact_count.numpy()[0]
        )
        results["measurements"].append(dict(mode="native_eager", **row))
        if device.is_cuda:
            native.reset()
            # Warm solver allocations and modules before capture. Use an even
            # total frame count so graph replays keep the same buffer pointers.
            native_step()
            native.reset()
            captured_steps = 1 if native.frame_skip % 2 == 0 else 2
            with wp.ScopedCapture(device=device) as capture:
                for _ in range(captured_steps):
                    native_step()
            row = measure(
                lambda: wp.capture_launch(capture.graph),
                device,
                args.iterations,
                args.warmup,
                worlds,
                captured_steps,
            )
            row["contacts_after_timing"] = int(
                native.physics.contacts.rigid_contact_count.numpy()[0]
            )
            results["measurements"].append(dict(mode="native_graph", **row))
        # An explicit post-measurement snapshot, never part of timed work.
        if not np.isfinite(native.state_view.joint_q.numpy()).all():
            raise RuntimeError("Nonfinite benchmark trajectory")
        native.close()
    output = json.dumps(results, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")


if __name__ == "__main__":
    main()
