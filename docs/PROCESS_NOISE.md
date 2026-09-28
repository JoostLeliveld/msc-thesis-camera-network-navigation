# Encoder-derived process noise

The thesis uses the encoder model in `planning.core.encoder_noise_model`.

Q is not fitted and not tuned. It is the white-noise equivalent of the noise the
simulator injects into the encoder, placed in the continuous-time unicycle
derivation below (the IWAI appendix derivation, unchanged).

## The simulated encoder

`sim/encoder_noise_node.py`, input_source `ground_truth`: the encoder starts from
the TRUE body velocity (from /ground_truth_tf) and adds only the declared noise:

    v_enc = v_sys (1 + s_v) + n_v,   w_enc = w_sys (1 + s_w) + n_w

- s_v, s_w: AR(1) slip, coefficient 0.80, stationary std 0.125 / 0.075;
- n_v, n_w: white per sample (50 Hz), std 0.004 m/s / 0.050 rad/s;
- (v_sys, w_sys): systematic wheel errors, `systematic_wheel_velocities`:
  w_sys = rho w - (e/b) v, v_sys = v - (e b / 4) w, with the UMBmark TRC LabMate
  values e = 0.00121 (D_R/D_L) and rho = 337.2/340 (wheelbase), b = 0.44 m.

Ground truth arrives at about 200 Hz; the encoder decimates it to its 50 Hz period
(`encoder_period_s`), because the noise above is declared per 50 Hz sample.
Smoke test 2026-09-27 (headless sim, 5.9 m driven, 155 deg turned): 45 Hz output,
end error 3.5 % of distance and 0.68 deg heading for one seed.

Why ground truth and not the DiffDrive odometry: the DiffDrive plugin integrates
the wheel joints, so Gazebo's simulated wheel-floor slip already corrupts it,
by an amount no parameter describes (measured 2026-09-27: 0.52 % of distance
and 0.34 deg heading, median, 10 s windows). Starting from true motion makes the
declared noise the only odometry error, so Q can be the model itself.

## White-noise equivalents (encoder period D = 0.02 s)

    white per-sample std s         -> PSD s^2 D
    AR(1) slip on rate r           -> PSD s^2 D (1 + alpha)/(1 - alpha) r^2
    systematic bias rate           -> PSD bias^2 T, T = 12 s (the random walk with the
                                      same variance at the longest camera-free stretch)

giving input-dependent PSDs

    sigma_v^2 = a_v + b_v v^2 + c_v w^2,   sigma_w^2 = a_w + b_w w^2 + c_w v^2

(`ENCODER_PSD`). Heading noise now follows turning and distance, not time.
The one declared assumption is T; the slip correlation time (0.02/(1-0.8) = 0.1 s)
is shorter than a filter step, so the white approximation holds.

## Where Q comes from

Continuous unicycle with zero-mean Gaussian disturbances on the actuation
signals. The motion Jacobian is nilpotent (F^2 = 0), so the matrix exponential
truncates exactly and the discrete process noise integrates in closed form:

    Q_d = Q_c dt + 0.5 (F Q_c + Q_c F^T) dt^2 + (1/3) F Q_c F^T dt^3
    Q_c = L(theta) diag(sigma_v^2, sigma_omega^2) L(theta)^T

Implemented identically in NumPy (`dynamics.py`) and CasADi
(`casadi_efe.py: unicycle_process_noise_ca`). Q therefore scales with **speed**
(v^2 in the cross-track terms), **heading** (L(theta) rotates it) and **dt**
(first, second and third order).

## Check (tests/planning/test_encoder_process_noise.py)

The test reproduces the encoder noise on known true motion (600 seeds, straight and
turning drives) and propagates Q with the planner's own closed form. The 95 % region
contains the odometry error in 0.965-0.995 of cases at 1, 3 and 10 s (position and
heading). NumPy and CasADi Q agree to machine precision; the model constants are
tested against the simulator's.

## The lock

- Defaults set in `src/experiments/experiments/core/visibility_launch_common.py`
  and `src/planning/planning/nodes/unicycle_planner_node.py`.

Changing the encoder constants or the 12 s bias horizon is a method change and
requires a new campaign.
