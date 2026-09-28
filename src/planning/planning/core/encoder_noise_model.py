"""Process-noise model SET from the simulated encoder (nothing fitted).

The encoder reports true body velocity corrupted by (see sim/encoder_noise_node.py and the
appendix noise table):
    v_enc = v_sys (1 + s_v) + n_v,   w_enc = w_sys (1 + s_w) + n_w,
    s AR(1) with coefficient alpha and stationary std sigma_s, n white per sample,
    (v_sys, w_sys) the systematic wheel errors of systematic_wheel_velocities.

White-noise equivalents on the continuous-time unicycle of the IWAI appendix
(Q_d = int e^{F t} L Q_c L^T e^{F^T t} dt), at the encoder period D:
    white per-sample std sigma            -> PSD sigma^2 D
    AR(1) slip, multiplying the rate r    -> PSD sigma_s^2 D (1 + alpha)/(1 - alpha) r^2
    systematic bias rate b                -> PSD b^2 T, the random walk with the same
                                             variance at the horizon T (longest camera gap)
So the PSDs depend on the input:
    sigma_v^2 = a_v + b_v v^2 + c_v w^2,   sigma_w^2 = a_w + b_w w^2 + c_w v^2.
"""
from __future__ import annotations

# Declared simulator noise (must equal experiments/core/visibility_launch_common.py).
ENCODER_PERIOD_S = 0.02                     # 50 Hz odometry
SLIP_ALPHA = 0.80
LINEAR_SLIP_STD = 0.125
ANGULAR_SLIP_STD = 0.075
LINEAR_ADDITIVE_STD = 0.004                 # m/s per sample
ANGULAR_ADDITIVE_STD = 0.050                # rad/s per sample
WHEEL_DIAMETER_RATIO_ERROR = 0.00121        # UMBmark TRC LabMate
WHEELBASE_RATIO = 337.2 / 340.0             # UMBmark TRC LabMate
WHEEL_SEPARATION_M = 0.44
BIAS_HORIZON_S = 12.0                       # longest camera-free stretch in the tasks

_AR = (1.0 + SLIP_ALPHA) / (1.0 - SLIP_ALPHA)
ENCODER_PSD = dict(
    a_v=LINEAR_ADDITIVE_STD ** 2 * ENCODER_PERIOD_S,
    b_v=LINEAR_SLIP_STD ** 2 * ENCODER_PERIOD_S * _AR,
    c_v=(0.25 * WHEEL_DIAMETER_RATIO_ERROR * WHEEL_SEPARATION_M) ** 2 * BIAS_HORIZON_S,
    a_w=ANGULAR_ADDITIVE_STD ** 2 * ENCODER_PERIOD_S,
    b_w=(ANGULAR_SLIP_STD ** 2 * ENCODER_PERIOD_S * _AR
         + (1.0 - WHEELBASE_RATIO) ** 2 * BIAS_HORIZON_S),
    c_w=(WHEEL_DIAMETER_RATIO_ERROR / WHEEL_SEPARATION_M) ** 2 * BIAS_HORIZON_S,
)


def encoder_psd(v, w, psd=ENCODER_PSD):
    """(sigma_v^2, sigma_w^2) at the operating point. Plain arithmetic: NumPy or CasADi."""
    return (psd['a_v'] + psd['b_v'] * v ** 2 + psd['c_v'] * w ** 2,
            psd['a_w'] + psd['b_w'] * w ** 2 + psd['c_w'] * v ** 2)
