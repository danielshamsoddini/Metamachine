# Newton: compatibility and native GPU execution

MetaMachine now has two execution modes behind `NewtonBackend`, both using
Newton 1.6.0 / Warp 1.17.0 **SolverFeatherstone**:

- `compat` (default): the existing single-environment Gymnasium task, with an
  explicit CPU/MuJoCo kinematic mirror after each control step.
- `native`: Warp state, actions, motor efforts, observations, rewards and
  termination flags stay on the selected device. Worlds are replicated into
  one Newton model with collision isolation. No MuJoCo calls, `.numpy()` reads
  or explicit host synchronization occur in native `step()` or device reset.

The original `sim`/`simulation`, `mjx` and real-robot routes remain available.
Native Newton is a Warp batched environment, not a JAX `vmap`/Brax environment.
The existing MJX trainer is not yet wired to it.

## Run the existing probe on GPU

From the repository root:

```bash
.venv-brachiation/bin/python -m pip install -e './Metamachine[newton]'
```

```python
import warp as wp
from metamachine.environments import make_env
from metamachine.environments.configs.config_registry import ConfigRegistry

cfg = ConfigRegistry.create_from_name("rigid_body_probe")
cfg.simulation.newton_mode = "native"
cfg.simulation.newton_device = "cuda:0"
cfg.environment.num_envs = 1024

env = make_env(cfg)
actions = wp.zeros((env.num_envs, env.action_size), dtype=wp.float32,
                   device=env.device)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(actions)
# Every returned numerical array is a Warp array on cuda:0.

# Device-side selective reset: int32 mask, one element per world.
mask = wp.ones(env.num_envs, dtype=wp.int32, device=env.device)
env.reset(mask=mask)

# Explicit diagnostic download, outside a rollout hot loop:
snapshot = env.snapshot()
env.close()
```

Use `newton_mode = "compat"`, `num_envs = 1` and NumPy actions for the original
Gymnasium path. Its device can also be CUDA; it still performs host mirroring.
Omitting `newton_mode` preserves that original behavior. `rigid_body_probe`
retains its previous CPU default, so set `newton_device` explicitly for GPU use.

Native outputs are borrowed reusable buffers. Copy/clone them on device if
retaining a rollout history. Persistent device action buffers avoid uploads;
Warp/PyTorch or DLPack interop belongs at the policy boundary, with explicit
stream/lifetime management. The native API rejects NumPy actions instead of
silently uploading them on every step. Device-detected nonfinite controls set
`valid_controls=0` (nonfinite policy actions set `valid_actions=0`), disable the
invalid input, and terminate the affected task world without a host read.

## Architecture and changed files

| File | Responsibility |
| --- | --- |
| `environments/backends/newton.py` | Shared solver, state buffers, modes, replicated worlds, device reset, `step_into` |
| `environments/backends/newton_model.py` | Startup-only validation, MJCF translation, compiled inertia and coordinate mapping |
| `environments/backends/newton_compat.py` | Optional host download and MuJoCo kinematic mirror |
| `environments/backends/newton_kernels.py` | Device actuator mapping and masked reset |
| `environments/backends/state.py` | Small `DeviceStateView` contract and `BackendCapabilityError` |
| `environments/components/newton_probe.py` | Device implementation of the existing probe's control, observation and task terms |
| `environments/env_newton_native.py` | Native batched task lifecycle and configuration capability checks |
| `environments/__init__.py` | `make_env` routing through `simulation.newton_mode` |
| `tests/test_newton_native.py` | CPU/CUDA equivalence, contact, batching, residency, graph and gradient tests |
| `scripts/benchmark_newton.py` | Synchronization-aware compat/native/graph timing utility |
| `benchmarks/newton_native_b300.json` | Measured GPU benchmark artifact |

`NEWTON_BACKEND.md` and the README describe the interface. The existing
`test_newton_backend.py`, morphology asset, task configuration, MuJoCo/MJX
implementations and legacy task components are retained.

The native task prepares its morphology, gains and initial pose using the
existing `MetaMachine` construction path **once**, without creating a temporary
log run. MuJoCo still compiles the MJCF and computes the initial configuration
at startup. The shared translation imports that compiled morphology and copies
resolved masses, COM offsets and rotated inertia tensors explicitly. Native
rollout then uses Newton exclusively; it has no `MjData` mirror. Reading
`native.physics.data` raises a capability error. `set_state(qpos, qvel)` is an
explicit host upload for diagnostics/setup, separate from device-only reset.

No damping, stiffness, mass, inertia, friction, contact parameters or timestep
were changed for this refactor. `angular_damping=0.0` and zero implicit Newton
servo gains remain intentional. Replicated worlds share local coordinates;
world IDs isolate their contacts without translating the physical scene.
Model parameters are fixed after construction in both modes.

## Controls and state access

The actuator topology is prepared once as device CSR arrays (DOF offsets and
motor IDs). A Warp kernel applies control clipping, fixed gain, force clipping,
and gear multiplication, then sums all motors attached to a DOF. Each DOF has
one writer, avoiding atomic torque-sum ordering differences. The same kernel
is used in compat and native mode. Only compat uploads NumPy controls.

At task level, policy actions are scaled, clipped, offset by the default pose,
and converted by the existing PD formula into actuator controls on device.
The probe holds this control for the configured ten physics substeps, matching
the existing no-latency controller. Implicit servos remain disabled.

`physics.state_view` exposes borrowed arrays directly:

- `joint_q`, `joint_qd`: Newton generalized coordinates and velocities, with
  explicit per-world `q_stride`/`qd_stride` and static actuator index mappings.
- `body_transform`: XYZ position plus XYZW quaternion.
- `body_velocity`: world COM linear velocity followed by world angular velocity.
- `body_com`: static local COM offsets, shared between state buffers.
- `body_stride`, `num_worlds`: explicit indexing for repeated topology.

The task consumes those arrays; there is no CPU-backed imitation of `MjData`.
MuJoCo's free-joint velocity uses the body origin and body-local angular axes,
so explicit host uploads translate that convention. The compatibility mirror
converts back. The native reward also preserves the existing torso `cvel`
reference point at the **root subtree COM**, rather than accidentally using
Newton's per-body COM velocity.

The supported native task retains the probe's three-frame, oldest-first
observation history, gravity/gyro/cosine-position/velocity/action ordering,
linear/angular velocity tracking rewards, torso height termination, and
existing episode time-limit boundary. Native reset is deterministic and
supports device masks. It does not automatically reset terminated worlds.
The config guard rejects unimplemented filters, latency, randomization,
command-linked rewards, other observation layouts and termination policies.
Sites, sensor/contact-force task features, rendering and cart/hook brachiation
remain unsupported; they never fall back to MuJoCo in the native hot loop.

## CUDA graphs and throughput

Warm up once, then capture stable inference buffers. For the probe, every task
step has ten physics substeps, so ping-pong physics buffers return to the same
addresses after each captured step:

```python
env.step(actions)  # compile kernels and initialize lazy solver buffers
env.reset()
with wp.ScopedCapture(device=env.device) as capture:
    env.step(actions)
for _ in range(100):
    wp.capture_launch(capture.graph)
```

If changing to an odd number of substeps, capture **two task steps** per graph
so the buffers return to their original parity. Keep actions and output arrays
alive at the same addresses and put policy output into the captured action
buffer on the appropriate stream. Reset modifies buffers in place and does not
invalidate a captured graph. CUDA graph replay is tested against eager output.

Run the benchmark:

```bash
PYTHONPATH=Metamachine MUJOCO_GL=disable \
  .venv-brachiation/bin/python Metamachine/scripts/benchmark_newton.py \
  --device cuda:0 --worlds 1 64 1024 --warmup 100 --iterations 50 \
  --output /tmp/newton-benchmark.json
```

Measured on NVIDIA B300 SXM6 AC, 2026-09-28, with 1 ms physics and ten substeps
per control step. Compilation and 100 warmup control steps are excluded.
`wp.synchronize_device` brackets each timed run, not each native step. Warmup
runs through landing; the final contact counts are 6, 384 and 6,144 for the
three world counts. Full results are in
[`benchmarks/newton_native_b300.json`](benchmarks/newton_native_b300.json).

| Path | Worlds | ms per control batch | Environment steps/s |
| --- | ---: | ---: | ---: |
| Compat on CUDA | 1 | 8.03 | 125 |
| Native eager | 1 | 7.55 | 132 |
| Native graph | 1 | 1.38 | 724 |
| Native graph | 64 | 1.74 | 36,864 |
| Native eager | 1,024 | 7.46 | 137,197 |
| Native graph | 1,024 | 3.25 | 315,322 |

This is a small zero-effort rigid-body/contact benchmark, not a locomotion or
training benchmark. Native timing includes observation/reward calculation;
compat timing includes physics and mirroring only. It excludes policy
inference, learning, PD target calculation and auto-reset scheduling. Larger
batches measure aggregate throughput, not a per-robot speedup. The limited
single-world eager gain shows that Python launch overhead remains significant;
graph capture and batching are necessary for MJX-like GPU utilization.

## Differentiability and remaining work

Inference `step()` deliberately reuses two states and task history buffers.
For gradients, construct `NewtonBackend(mode="native", requires_grad=True)`
and preallocate distinct `allocate_step()` outputs for each retained timestep:

```python
buffers = [backend.allocate_step() for _ in range(T)]
# controls[t] are persistent device float32 arrays with requires_grad=True.
with wp.Tape() as tape:
    state = backend.state
    for t in range(T):
        state = backend.step_into(state, controls[t], buffers[t])
    # launch a loss kernel over state into a scalar requires_grad loss array
# tape.backward(loss)
```

Each differentiable timestep owns its state, control, contacts, collision
scratch **and solver matrix caches**. Newton 1.6 keeps J/M/H/L on the solver;
sharing those caches across retained timesteps could overwrite values required
by backward. Allocate outside Tape/capture, retain inputs until backward, and
warm lazy solver allocations before graph capture. `step()` rejects gradient
mode so it cannot silently reuse differentiable states.

A three-step contact-free control gradient is checked against finite
differences on CPU and CUDA. This is not full differentiable-task support:
observation history, episode counters, termination and masked reset currently
use inference-oriented mutation. A future functional task state is needed for
end-to-end task/policy gradients. Contact topology changes and clipping are
nonsmooth; contact gradients and long-horizon stability have not been verified.
Startup MJCF/NumPy construction is also outside the graph, so differentiating
morphology compilation is not implemented. Per-timestep solver/contact storage
has a memory cost; a long-horizon pipeline needs checkpointing or a validated
solver adjoint strategy.

The probe already runs as 1,024 worlds on one GPU. Remaining training work is
policy/trainer interop (the MJX/Brax loop cannot call this as a JAX function),
device randomization, auto-reset scheduling, rollout-buffer ownership and
migration of additional task terms. Large articulated robots and dense contact
scenes need separate collision-capacity, solver-memory and throughput checks.
No trained gait, cart task or brachiation policy has been ported in this change.

## Validation

```bash
PYTHONPATH=Metamachine MUJOCO_GL=disable \
  .venv-brachiation/bin/python -m pytest \
  Metamachine/tests/test_newton_backend.py Metamachine/tests/test_newton_native.py
```

Validation on Python 3.12.4 / MuJoCo 3.14.0 / Newton 1.6.0 / Warp 1.17.0:

- **37 Newton tests passed:** all 20 original comparisons retained, plus 17
  native CPU/CUDA checks. They cover reset, joint/root/body state and velocity
  conventions, torque application, 1,000 driven physics steps through landing,
  shared task outputs, independent batches and selective reset, no-host-read
  guards, CUDA graph replay and finite-difference gradient checks.
- **53 brachiation regression tests passed.**
- Existing MetaMachine suite: **75 passed, 2 pre-existing reward-test failures**
  from duplicate `tracking_sigma` arguments in the unchanged test helper.
- Ruff checks on the added/refactored Python files passed.

The contact regression uses the actuated probe, not a learned locomotion gait.
MuJoCo-vs-Newton trajectory tolerances remain relaxed where solver/timing
semantics differ; compat-vs-native Newton comparisons are much tighter. The
residency test patches `.numpy()`, NumPy allocation, Warp synchronization and
MuJoCo runtime functions to fail during native stepping/reset. Optional Newton
and CUDA tests skip when the corresponding dependency/device is unavailable.

API references: [Featherstone](https://newton-physics.github.io/newton/latest/api/_generated/newton.solvers.SolverFeatherstone.html),
[model replication](https://newton-physics.github.io/newton/latest/api/_generated/newton.ModelBuilder.html).

## Native brachiation extension

`metamachine.environments.env_brachiation_newton.BrachiationNewton` adds a batched
Torch device task for the existing cart-assisted brachiation schema. The native
factory selects it for `task.schema_version` plus `task.cart_task`; provide
`simulation.source_xml` resolved relative to the configuration directory. The
repository-level `train_brachiation_newton.py` launcher handles path resolution.

The extension supports mesh wheels, resettable fixed bars, compound shoulder
hinges, and a two-site safety tether behind `backends/newton_brachiation.py`.
General Newton imports retain their original validation restrictions. The tether
upper limit is an explicit compliant penalty approximation. Four integration
substeps preserve the original PD interval while stabilizing explicit contacts.
CUDA graph step/reset, device observations/rewards, and GPU PPO are implemented;
MuJoCo/MJX code paths are retained. See the parent repository's
`BRACHIATION_NEWTON.md` for the full contract and run commands.
