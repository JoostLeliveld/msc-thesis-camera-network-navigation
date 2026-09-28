# Belief-space planner

This document describes the single planner used by the thesis campaign.

## Belief prediction

For candidate input `u_j`, the unicycle mean and covariance propagate as:

```text
x_j^- = f(x_(j-1)^+, u_j)
P_j^- = F_j P_(j-1)^+ F_j^T + Q_j
```

The process covariance `Q_j` is the encoder-derived input-dependent model in
`docs/PROCESS_NOISE.md`.

For covariance model `m`, five sigma points `chi_(j,s)` represent the
predicted planar belief. Total expected information is:

```text
I_j^m = sum_active_cameras sum_sigma_points
        w_s [R_i^m(chi_(j,s))]^-1
```

The expected position update changes covariance but not mean:

```text
Pbar_j^(+,m) = [(P_j^-)^-1 + S^T I_j^m S]^-1
```

Removing a camera removes its information from every candidate route.

## Objective

The planner minimizes the arrival-gated discounted mean:

```text
J(U) = sum_j eta_j gamma^(j-1)
       [J_risk,j + J_ambiguity,j + J_obstacle,j]
       / sum_j eta_j gamma^(j-1)
```

The arrival gate smoothly suppresses predicted steps after the mean enters the
0.10 m arrival region.

### Goal risk

Risk is the Gaussian KL divergence between the predicted position belief and a
goal prior. The goal standard deviation decreases from 5.0 m to 0.10 m using
smoothstep progress with exponent 0.9.

Before evaluating the KL term, the position covariance is adjusted by:

```text
Ptilde_j = Pxy_j
         + max(s_goal,j^2 - 0.5 trace(Pxy_j), 0) I
```

This treats the goal covariance as an upper uncertainty tolerance instead of
penalizing a belief solely for being more concentrated.

### Ambiguity

The effective observation covariance is:

```text
R_eff,j = (I_j^m + epsilon_amb I)^-1
```

Ambiguity is the non-negative log-determinant ratio relative to
`R_ref = (1.5 mm)^2 I`.

### Obstacle penalty

At every planar belief sigma point, the oriented 0.80 by 0.55 m body is tested
against obstacle geometry using the predicted mean heading. If `h_(j,s)` is
clearance and `b = 0.05 m` is the warning band:

```text
d_(j,s) = max(b - h_(j,s), 0) / b
phi(d)  = 50 [d^2 + 100 max(d - 1, 0)^2]
J_obstacle,j = sum_s w_s phi(d_(j,s))
```

The term is zero outside the warning band and grows rapidly after overlap.
Exact swept-footprint checks remain the hard route-validity criterion.

## Route solve

- Five tasks are defined in `pipeline/tasks.yaml`.
- Every model/network arm receives the same route initializations.
- The global optimizer uses 200 iterations and one control per block.
- Routes are solved once before execution and bound into per-seed campaign
  configurations.
- `pipeline/replay_routes.py` verifies every bound route through the same
  `ff_fb` follower used during the campaign.

## Locked parameters

| Parameter | Value |
| --- | ---: |
| integration step | 0.25 s |
| horizon | 20 |
| discount factor | 0.995 |
| maximum linear speed | 1.0 m/s |
| risk multiplier | 1.0 |
| ambiguity multiplier | 1.0 |
| control multiplier | 0.0 |
| ambiguity regularizer | 1.0 m^-2 |
| goal prior start/final sigma | 5.0 / 0.10 m |
| arrival radius | 0.10 m |
| obstacle warning band | 0.05 m |
| obstacle near/contact weights | 50 / 100 |
| route optimizer iterations | 200 |
| execution tracker | `ff_fb` |

The route-planning template is `pipeline/route_planning_template.yaml`. The
execution template is `pipeline/execution_template.yaml`.

## Verification

Planner integrity is covered by tests under `tests/planning/`, including:

- NumPy/CasADi objective agreement;
- camera-network information handling;
- encoder-derived process noise;
- obstacle and swept-footprint geometry;
- route installation and follower safety; and
- estimator correction and timing invariants.
