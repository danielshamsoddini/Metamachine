# Experimental Newton backend

Newton is a third, optional physics engine selected with
`environment.mode: newton`. It uses **SolverFeatherstone**, not Newton's
MuJoCo solver. Existing `sim` / `simulation`, `mjx`, and real-robot routes
remain available. This first milestone is a single, headless Gymnasium
environment; it does not port the MJX/Brax brachiation training pipeline.

Install from the repository root:

```bash
.venv-brachiation/bin/python -m pip install -e './Metamachine[newton]'
```

The optional extra pins Newton 1.6.0 and Warp 1.17.0, whose floating-joint
velocity convention is covered by tests. Normal MetaMachine installation and
MuJoCo/MJX use do not require either dependency. CPU is the validated device;
`simulation.newton_device` is passed to Newton, but GPU/batched execution has
not been validated here.

Run the minimal probe (no training):

```python
import numpy as np
from metamachine.environments import make_env
from metamachine.environments.configs.config_registry import ConfigRegistry

cfg = ConfigRegistry.create_from_name("rigid_body_probe")
# Set cfg.environment.mode = "sim" to run this same task with MuJoCo.
env = make_env(cfg)
try:
    obs, info = env.reset(seed=0)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.array([0.2]))
        if terminated or truncated:
            break
finally:
    env.close()
```

The probe is a 2 kg floating box with a 0.5 kg hinged arm, nonzero COM offsets,
joint damping and armature, a gear ratio of 2, and explicit control/force
limits. It starts at 2 m with a 1 ms physics timestep and 10 ms control period.
It inherits MetaMachine's existing position-target processing, PD controller,
three-frame observations (gravity, gyro, cosine of joint position, joint
velocity, previous action), linear/angular velocity tracking rewards, torso
height termination, and episode time limit. It logs under `./runs`.

## Implementation boundary

`environments/backends/base.py` defines the stateful physics protocol. The
MuJoCo adapter preserves the previous `mj_step` / `mj_rnePostConstraint`
sequence. The Newton adapter owns model translation, state buffers, joint
mapping, collision generation, motor forces, stepping and reset. The existing
MJX functional implementation remains separate and unchanged.

`NewtonMetaMachine` reuses the existing MetaMachine task lifecycle and
morphology asset/factory pipeline. The compiled MJCF is imported into Newton;
compiled masses, COM offsets and rotated inertia tensors are copied explicitly
to avoid independent density/default inference. Body and joint names define
the mapping, so action ordering remains the compiled actuator ordering.
Changing supported morphology dimensions/masses before construction follows
the same path on both engines. Runtime model randomization/reloads are not
supported yet.

MuJoCo remains a dependency for morphology compilation and the legacy task
state view. Newton alone integrates dynamics. The adapter calls only
`mj_kinematics`, `mj_comPos`, and `mj_comVel` to populate that view; it does not
call `mj_step` or `mj_forward`. This bridge avoids duplicating task math while
the task-state API is still coupled to MuJoCo metadata. It is not a standalone
MuJoCo-free implementation.

Free-joint quaternions convert WXYZ to XYZW. MuJoCo's world body-origin linear
velocity and local angular velocity convert to Newton's world COM linear and
world angular velocity, including the angular cross-product COM offset.
Motor controls apply control clipping, fixed gain, force clipping, then gear;
multiple motors on a joint add their efforts. Newton's extra global angular
damping and implicit servo gains are disabled.

## Limits and expected differences

The initial model support is primitive rigid geometry with named bodies and
free, hinge or slide joints (one joint per body); the environment requires a
free root first. Fixed links retain their inertia. Joint damping, armature and
limits are imported. Contacts and joint limits use Newton's penalty solver;
MuJoCo solver parameters and trajectories are not equivalent. Timestep and
stiffness must be checked when extending the probe to other morphologies.

Unsupported models fail explicitly: meshes/heightfields, ball or compound
joints, nonzero joint references, springs/frictionloss, joint-level force
limits, equality/loop constraints, tendons, mocap, flexible bodies, plugins,
fluid forces, gravity compensation, sensors and non-motor actuators. The task
guard limits observations/rewards to the validated state-based families and
rejects contact-dependent termination, rendering, model/force randomization,
and simulation-based reset settling. Legacy contact/acceleration fields are
not populated with a second simulator's results. In particular, the existing
cart/hook brachiation task is **not** supported on Newton in this milestone.

Newton reports current derived kinematics. MuJoCo's default stepping leaves
some derived kinematics at the start of the last physics substep. Accordingly,
the tests allow a small gyro observation discrepancy at the 1 ms timestep
while checking generalized positions and velocities more tightly. Existing
MuJoCo observation timing is unchanged.

## Validation

```bash
PYTHONPATH=Metamachine MUJOCO_GL=disable \
  .venv-brachiation/bin/python -m pytest Metamachine/tests/test_newton_backend.py
```

Tests exercise compiled morphology, COM/rotation conventions, actuator units
and clipping, analytic zero-action freefall, a fixed control sequence in air,
shared task observations/reward/termination, repeatable resets, unsupported
features, and exact preservation of the MuJoCo integration sequence. Newton
tests skip when the optional dependency is absent. Contact tests check finite,
bounded behavior after landing; they deliberately do not assert identical
contact trajectories.

Newton API references: [Featherstone solver](https://newton-physics.github.io/newton/latest/api/_generated/newton.solvers.SolverFeatherstone.html),
[MJCF model import](https://newton-physics.github.io/newton/latest/api/_generated/newton.ModelBuilder.html).

Validation run (2026-09-28, Python 3.12.4, MuJoCo 3.14.0, Newton 1.6.0,
Warp 1.17.0, CPU): 20/20 backend tests and 53/53 brachiation regression tests
passed. The existing MetaMachine suite passed 75/77 tests. The remaining two
reward tests fail before constructing their component because their helper
passes `tracking_sigma` twice; both failures reproduce from the committed test
file. Reward implementation and those tests were not changed. Ruff checks on
the added Python files passed.
