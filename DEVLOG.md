# Development Log

## 2026-02-18: Fixed double `default_dof_pos` offset bug in transformer evaluation

### Problem

The transformer policy worked well on v1 but appeared to completely fail on v2 — the v2 robot barely moved (0.336m in 1000 steps) despite the v2 expert walking fast.

### Root Cause

The expert data was recorded with `action_as_actuator_command=True` in `batch_record_rollout.py`, meaning actions stored in the `.pkl` files are **absolute joint position targets** (i.e., `raw_policy_action + default_dof_pos`).

However, during evaluation in `evaluate_transformer.py`, the transformer's predicted action (already an absolute position) was passed directly to `env.step()`. Internally, `env.step()` applies `action + default_dof_pos` again (see `action.py` line 472), **doubling the offset**.

For v1 this was invisible because `default_dof_pos = [0, 0, 0, 0, 0]` — adding zero changes nothing. For v2 with `default_dof_pos = [-2.78, 1.30, -3.09, -2.57, 1.06]`, the joints were sent to completely wrong positions (e.g., `-2.78 + -2.78 = -5.56` instead of `-2.78`).

### Fix

In `evaluate_transformer.py`, subtract `default_dof_pos` from the transformer's output before passing to `env.step()`:

```python
default_dof_pos = np.array(cfg.control.default_dof_pos)
# In the rollout loop:
env_action = action - default_dof_pos
obs, reward, done, truncated, info = env.step(env_action)
```

### Results

| Robot | Before Fix | After Fix |
|-------|-----------|-----------|
| v1    | 383.65 reward, 2.125m | (unchanged, offset is zero) |
| v2    | 210.29 reward, 0.336m | 207.79 reward, **7.322m** |

### Lesson

When expert data is recorded with `action_as_actuator_command=True`, the actions already include `default_dof_pos`. The evaluation loop must account for the environment's internal `action + default_dof_pos` by subtracting it beforehand. This bug is silent for any robot where `default_dof_pos` is zero.
