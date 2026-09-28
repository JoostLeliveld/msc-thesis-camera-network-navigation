"""(1) Realised odometry drift in the simulation. (2) The METHOD.md Q check at two q_theta.

Data: the 36 superseded runs (not evaluation, same odometry/noise code as the campaign).
Windows start every 0.5 s. At the window start the odometry pose is aligned to ground truth
(position and heading); the odometry increment over the window is carried into the map frame
by that alignment. Error at the window end = aligned odometry minus ground truth.

(1) drift: position error per metre travelled (% of distance), in windows of >= 2 m, and
    heading error per metre and per 90 degrees turned.
(2) check (METHOD.md): from the true pose with zero covariance, predict on odometry alone with
    the planner's own dynamics and Q; fraction of windows whose true error lies inside the 95%
    ellipse (position, chi2_2 5.991) and 95% interval (heading) at 1, 3 and 10 s.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src/planning'))
from planning.core.dynamics import unicycle_jacobian, unicycle_process_noise, unicycle_step  # noqa: E402

LOGS = REPO / 'logs/thesis'
SETS = [LOGS / 'superseded_taskC_blind_goal_20260925',
        LOGS / 'revisions/bc_dropout_swap/superseded_blindgoal_20260926']
QXY, BASE_DT = 0.02, 0.25


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def predict_cov(ts, v, w, theta0, qth):
    """Covariance from zero along the odometry inputs, planner dynamics and Q."""
    m = np.array([0.0, 0.0, theta0]); P = np.zeros((3, 3))
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        u = np.array([v[i], w[i]])
        F = unicycle_jacobian(m, u, dt)
        Q = unicycle_process_noise(QXY, qth, dt, theta=float(m[2]), v=float(u[0]), base_dt=BASE_DT)
        m = np.asarray(unicycle_step(m, u, dt), dtype=float)
        P = F @ P @ F.T + Q
    return P


drift, check = [], []
for base in SETS:
    for exp in sorted(p.parent for p in base.rglob('experiment.csv')):
        e = pd.read_csv(exp / 'experiment.csv', usecols=['odom_noisy_stamp', 'odom_noisy_x', 'odom_noisy_y',
                        'odom_noisy_yaw', 'odom_noisy_v', 'odom_noisy_w']).dropna()
        e = e.drop_duplicates('odom_noisy_stamp').sort_values('odom_noisy_stamp')
        g = pd.read_csv(exp / 'ground_truth_pose.csv').sort_values('stamp_s')
        t = e.odom_noisy_stamp.to_numpy()
        ox, oy, oyaw = e.odom_noisy_x.to_numpy(), e.odom_noisy_y.to_numpy(), np.unwrap(e.odom_noisy_yaw.to_numpy())
        gx, gy = np.interp(t, g.stamp_s, g.x), np.interp(t, g.stamp_s, g.y)
        gyaw = np.interp(t, g.stamp_s, np.unwrap(g.yaw))
        dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(gx), np.diff(gy)))])
        turn = np.concatenate([[0], np.cumsum(np.abs(np.diff(gyaw)))])
        for i0 in range(0, len(t), 5):
            rot = gyaw[i0] - oyaw[i0]
            c, s = np.cos(rot), np.sin(rot)
            for horizon in (1.0, 3.0, 5.0, 10.0):
                i1 = np.searchsorted(t, t[i0] + horizon)
                if i1 >= len(t):
                    continue
                dx, dy = ox[i1] - ox[i0], oy[i1] - oy[i0]
                ex = gx[i0] + c * dx - s * dy - gx[i1]
                ey = gy[i0] + s * dx + c * dy - gy[i1]
                eh = wrap(oyaw[i1] + rot - gyaw[i1])
                D, T = dist[i1] - dist[i0], turn[i1] - turn[i0]
                drift.append((horizon, D, np.degrees(T), np.hypot(ex, ey), np.degrees(eh)))
                if horizon in (1.0, 3.0, 10.0):
                    sl = slice(i0, i1 + 1)
                    for qth in (0.08, 0.0358):
                        P = predict_cov(t[sl], e.odom_noisy_v.to_numpy()[sl], e.odom_noisy_w.to_numpy()[sl], gyaw[i0], qth)
                        d = np.array([ex, ey])
                        nees = float(d @ np.linalg.solve(P[:2, :2] + 1e-12 * np.eye(2), d))
                        check.append((horizon, qth, nees <= 5.991, abs(eh) <= 1.96 * np.sqrt(P[2, 2])))
    print('.', end='', flush=True)
print()
dr = pd.DataFrame(drift, columns=['h', 'dist_m', 'turn_deg', 'pos_err_m', 'head_err_deg'])
w = dr[(dr.h == 10.0) & (dr.dist_m >= 2.0)]
print(f'(1) DRIFT, 10 s windows with >= 2 m travelled (n={len(w)}):')
pct = 100 * w.pos_err_m / w.dist_m
print(f'    position error per distance: median {pct.median():.2f} %, p95 {pct.quantile(.95):.2f} %')
print(f'    heading error per metre:     median {(w.head_err_deg.abs() / w.dist_m).median():.3f} deg/m, '
      f'p95 {(w.head_err_deg.abs() / w.dist_m).quantile(.95):.3f} deg/m')
tw = w[w.turn_deg >= 45]
print(f'    heading error per 90 deg turned (windows turning >= 45 deg, n={len(tw)}): '
      f'median {(90 * tw.head_err_deg.abs() / tw.turn_deg).median():.2f} deg, p95 {(90 * tw.head_err_deg.abs() / tw.turn_deg).quantile(.95):.2f} deg')
st = w[w.turn_deg < 10]
print(f'    heading error on straight windows (< 10 deg turned, n={len(st)}): median {st.head_err_deg.abs().median():.2f} deg, '
      f'p95 {st.head_err_deg.abs().quantile(.95):.2f} deg')
ck = pd.DataFrame(check, columns=['h', 'q_theta', 'pos_in95', 'head_in95'])
print('\n(2) METHOD.md CHECK: fraction of windows inside 95% (odometry only, from true pose, zero covariance)')
print(ck.groupby(['q_theta', 'h'])[['pos_in95', 'head_in95']].mean().round(3).to_string())
