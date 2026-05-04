# V7 Yaw Failure Analysis: Angular Wraparound in Flow Matching

## What Happened

Version 7 added a yaw (facing direction) dimension to the person trajectory, changing the representation from `(px, py, pz)` to `(px, py, pz, yaw)` where yaw is in radians. The model was trained on 660k samples (E.T. + AMASS + HumanML3D) with person_dim=4. Training loss converged normally (0.241), but generation produced completely flat outputs for all motion types. Mode collapse was total.

Version 6 (person_dim=3, no yaw, 326k samples) had been working well for orbit, static, dolly-in, and track. Adding yaw and more data should have been an improvement. Instead the model collapsed.

## Root Cause: Angular Wraparound

Yaw is an angle measured in radians, with range [-pi, pi]. At the boundaries, -pi and +pi represent the same physical direction (facing backward). But as numbers, they are 2*pi apart (roughly 6.28).

Concrete example from the training data:

| Sample | Yaw at frame 0 | Yaw at frame 47 | Physical meaning |
|--------|----------------|-----------------|------------------|
| Sample 2 | -2.98 rad | +3.10 rad | Person is facing backward, turning slightly |
| Numeric distance | | 6.08 | Huge jump in number space |
| Angular distance | | 0.16 rad (9 degrees) | Tiny turn in real life |

Flow Matching interpolates between noise and data along a straight line in the data space:

```
x_t = (1 - t) * noise + t * data
```

This interpolation assumes the data space is continuous and that nearby numbers represent nearby states. For position dimensions (px, py, pz) this is true. For yaw it is not. When data has yaw = -2.98 and noise has some random value, the interpolation passes through all intermediate values, creating a training signal that says "yaw should be near 0" even though the real direction is near ±pi.

Across the full training set, samples with yaw near +pi and samples with yaw near -pi provide contradictory gradients: one pushes the prediction toward +3.14, the other toward -3.14. The MSE-optimal compromise is to predict 0 for everyone. This is exactly what mode collapse looks like.

## Why It Did Not Affect Position Dimensions

Position dimensions (px, py, pz) have no wraparound. A position of 3.0 meters and -3.0 meters are genuinely far apart. Linear interpolation between them passes through 0.0 meters, which is a valid intermediate position. The flow matching framework is mathematically correct for these dimensions.

## The Fix: Sin/Cos Representation

Replace the single yaw angle with its sine and cosine components:

```
Before: (px, py, pz, yaw)          -- 4 dims, discontinuous at ±pi
After:  (px, py, pz, sin(yaw), cos(yaw))  -- 5 dims, continuous everywhere
```

Properties of the sin/cos representation:

| Yaw (radians) | sin(yaw) | cos(yaw) |
|---------------|----------|----------|
| 0 (forward) | 0.00 | 1.00 |
| pi/2 (right) | 1.00 | 0.00 |
| pi (backward) | 0.00 | -1.00 |
| -pi (backward) | 0.00 | -1.00 |
| -2.98 | -0.16 | -0.99 |
| +3.10 | +0.04 | -1.00 |

The last two rows show the key improvement. In raw radians, -2.98 and +3.10 differ by 6.08. In sin/cos space, (-0.16, -0.99) and (0.04, -1.00) differ by only 0.20. Linear interpolation between them stays near (-1.0 cos, ~0 sin), which correctly represents the "backward-facing" direction throughout the interpolation path.

The sin/cos representation has no discontinuities anywhere. It is the standard approach in computer graphics, robotics, and recent motion generation literature (MDM, MotionDiffuse, etc.) for encoding angular quantities in neural network inputs and outputs.

To recover the original angle at inference time: `yaw = atan2(sin_yaw, cos_yaw)`.

## Changes Made

1. Preprocessing (E.T., AMASS, HumanML3D): store `sin(yaw), cos(yaw)` instead of raw yaw
2. person_dim: 4 -> 5
3. Total joint vector dimension: 48*(5+6) = 528 (previously 48*(4+6) = 480)
4. Smoothing: instead of angular unwrap/rewrap, smooth sin and cos independently then renormalize to unit circle
5. Visualization: recover yaw via `atan2(sin, cos)` for display
6. Fallback for no-orientation data: sin=0, cos=1 (representing yaw=0, facing forward)

## Lesson Learned

When adding a new data dimension to a generative model, check whether the dimension's topology matches the model's assumptions. Flow matching (and diffusion models generally) assume a Euclidean data space where linear interpolation is meaningful. Angular quantities live on a circle, not on a line. The mismatch is invisible during training (loss converges normally) but produces mode collapse at generation because the model learns to predict the circular mean, which is the compromise between contradictory wraparound gradients.
