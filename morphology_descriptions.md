# Robot Morphology Variations - Experiment Summary

This document describes the different robot morphology configurations created for batch RL training. All variations use the `modular_legs` robot type with different leg attachment points on the central torso (Module 0).

## Dock Mapping Reference (Module 0)
*   **0 - 6**: Upper Stick (0 is the furthest end, 6 is closest to the ball)
*   **7 - 8**: Upper Ball
*   **9 - 10**: Lower Ball
*   **11 - 17**: Lower Stick (11 is closest to the ball, 17 is the furthest end)

---

## Configuration Details

### 1. `new_five_modules_v2.yaml` (Asymmetrical/Standard)
*   **Docks**: `[1, 2, 8, 13]`
*   **Description**: A mixed placement using one upper stick point, one upper ball point, and one lower stick point. This creates an asymmetrical leg distribution.

### 2. `new_five_modules_v3.yaml` (Staggered Mid-Stick) ✅ MOVABLE
*   **Docks**: `[4, 5, 11, 16]`
*   **Description**: Uses points further down the sticks. It staggered the legs along the length of the torso modules to test longitudinal stability.

### 3. `new_five_modules_v4.yaml` (Centralized/Compact) ⚠️ LOW MOVABILITY
*   **Docks**: `[7, 8, 9, 10]`
*   **Description**: All four legs are attached directly to the central balls. This creates a very compact "spider-like" footprint with a high center of gravity relative to the leg spread.
*   **Pose Optimization**: `stablefast`

### 3.1 `new_five_modules_v4_1.yaml` (Centralized/Compact - Hybrid Pose) 🧪 A/B TEST ⚠️ LOW MOVABILITY
*   **Docks**: `[7, 8, 9, 10]` (same as v4)
*   **Description**: Identical morphology to v4, but uses **hybrid pose optimization** (bigbase → stablefast).
*   **Pose Optimization**: `hybrid` (Stage 1: find stable base, Stage 2: optimize for movement)
*   **Purpose**: Test if two-stage optimization helps compact morphologies achieve better performance.

### 4. `new_five_modules_v5.yaml` (Extreme Wide) ⚠️ LOW MOVABILITY
*   **Docks**: `[0, 6, 11, 17]`
*   **Description**: Legs are attached to the extreme ends of both the upper and lower sticks. This creates the widest possible base, maximizing static stability but potentially making turning more difficult due to the large moment of inertia.

### 5. `new_five_modules_v6.yaml` (Balanced Mid-Range) ⚠️ LOW MOVABILITY
*   **Docks**: `[3, 4, 14, 15]`
*   **Description**: A "Standard Quadruped" setup. Legs are placed in the middle sections of the sticks. This is intended to be a balanced configuration between the compact v4 and the extreme v5.

### 6. `new_five_modules_v7.yaml` (Mirror of v2) ⚠️ LOW MOVABILITY
*   **Docks**: `[16, 15, 9, 4]`
*   **Description**: Mirror of v2 across the central axis. Asymmetrical but stable.

### 7. `new_five_modules_v8.yaml` (Balanced Staggered) ⚠️ LOW MOVABILITY
*   **Docks**: `[1, 5, 12, 16]`
*   **Description**: Uses points near the ends and mid-sections of both sticks.
*   **Note**: Observed to have limited locomotion capability in testing.

### 8. `new_five_modules_v9.yaml` (Spiral/Diagonal) ⚠️ LOW MOVABILITY
*   **Docks**: `[2, 6, 11, 15]`
*   **Description**: Spiral-like placement around the torso for potential agility.
*   **Note**: Observed to have limited locomotion capability in testing.

### 9. `new_five_modules_v10.yaml` (Wide Asymmetrical) ✅ MOVABLE
*   **Docks**: `[0, 6, 9, 13]`
*   **Description**: Combines wide points on one stick with ball and inner points on the other.

### 10. `new_five_modules_v11.yaml` (Narrow Asymmetrical) ✅ MOVABLE
*   **Docks**: `[3, 4, 8, 11]`
*   **Description**: Clustered points around the center for a narrow footprint.

### 11. `new_five_modules_v12.yaml` (Mixed Symmetry)
*   **Docks**: `[1, 6, 11, 16]`
*   **Description**: Symmetrical across the sticks but using different points on each.

### 12. `new_five_modules_v13.yaml` (v2 Variant - Near Asymmetric)
*   **Docks**: `[1, 3, 9, 14]`
*   **Description**: Similar to v2 but with tighter clustering. Variation on the successful asymmetric pattern.

### 13. `new_five_modules_v14.yaml` (Mid-Spread Diagonal)
*   **Docks**: `[2, 5, 12, 15]`
*   **Description**: Diagonal spread across mid-sections of both sticks.

### 14. `new_five_modules_v15.yaml` (Balanced Asymmetric Wide)
*   **Docks**: `[2, 4, 10, 13]`
*   **Description**: Asymmetric with moderate width, inspired by v2's success.

---

## Batch Training
All these models are trained using the `train_variations.sh` script for 1,000,000 timesteps each.

### Reward Structure (ALL VERSIONS)

**Anti-Hacking Reward** - Applied to v2-v15 (v13-v15 proved successful, applied to all):

| Component | Type | Weight | Purpose |
|-----------|------|--------|---------|
| `forward_velocity` | linear_velocity_tracking | **1.0** | Primary reward for forward movement (0.6 m/s target) |
| `go_straight` | **conditional_go_straight** | **0.5** | Only rewards if speed > 0.1 m/s, ensures straight-line movement |
| `orientation` | orientation_reward | **0.05** | Reduced weight to discourage "just stand upright" hacking |
| `smooth_motion` | action_rate | **-0.01** | Penalizes jerky movements |
| `move_penalty` | stillness_penalty | **0.5** | Strong penalty for not moving (threshold: 0.1 m/s) |

**Key Anti-Hacking Features:**
- `conditional_go_straight`: Only gives directional reward when robot is actually moving
- Strong `move_penalty`: Makes standing still very costly
- Reduced `orientation` weight: Prevents getting high scores just by standing upright

### A/B Test Strategy

#### **Pose Optimization A/B Test (v4 vs v4.1)**
Testing the effect of hybrid pose optimization on compact morphologies:
*   **v4**: `stablefast` (standard)
*   **v4.1**: `hybrid` (bigbase → stablefast, two-stage)
*   **Same morphology**: `[7, 8, 9, 10]` - compact spider configuration
*   **Same reward**: Anti-hacking reward structure
*   **Goal**: Determine if hybrid optimization helps challenging morphologies

#### **Morphology Notes**
*   **v2 - v12**: Using `stablefast` optimization
*   **v4.1**: Using `hybrid` optimization (A/B test)
*   **v3, v10, v11**: Marked as ✅ MOVABLE based on training results
*   **v4, v5, v6, v7, v8, v9**: Marked as ⚠️ LOW MOVABILITY (range v3-v11)
*   **v13-v15**: New asymmetric morphologies based on successful v2 pattern

